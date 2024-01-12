import warnings
warnings.filterwarnings("ignore")

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
import wandb
import time

import numpy as np
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    

parser = argparse.ArgumentParser(description='Finetune bert classifier for sentiment classification')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--num_train_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--train_subsample_size', type=int, default=1000, help='selected a subsample from the training set for fine tuning')
parser.add_argument('--model_name', type=str, default="distilbert-base-uncased", help='Transformer model to be fine-tuned')

args = parser.parse_args()

# login to wandb and Initialize W&B run
datetime = time.strftime("%Y%m%d-%H%M%S")
wandb.login()
wandb.init(
      # Set the project where this run will be logged
      project="sutd-mlops-project", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_session3_run_{datetime}", 
      # Track hyperparameters and run metadata
      config={
          "learning_rate": 2e-5,
          "weight_decay": 0.01,
          "num_train_epochs": 2,
          "train_subsample_size": 1000,
          "architecture": "distilbert",
          "dataset_name": "rotten_tomatoes",
          "model_name": "distilbert-base-uncased"
      })
config = wandb.config

# load dataset and sub-sample a small dataset for fine tuning
dataset = load_dataset(config.dataset_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenized_datasets = dataset.map(
                            lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), 
                            batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(config.train_subsample_size))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(100))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

# train the model
num_labels = len(np.unique(dataset['train']['label']))
model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=num_labels)

metric = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir=".",
    report_to="wandb",
    evaluation_strategy="epoch",
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    num_train_epochs=config.num_train_epochs,
    logging_steps=20)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Test the model on whole test set - for fair comparison with the classification performance achieved by SGD in previous sessions
def predict(tokenized_test_data, trainer):
    output_array = trainer.predict(tokenized_test_data)[0]
    pred_prob = np.exp(output_array)/np.sum(np.exp(output_array), axis = 1)[..., None]
    pred = np.argmax(pred_prob, axis = 1)
    return pred_prob, pred 

pred_prob, pred  = predict(tokenized_datasets["test"], trainer)
accuracy = np.sum(pred == dataset["test"]['label'])/len(dataset["test"]['label'])
print(f"Accuracy: {accuracy}")
wandb.sklearn.plot_precision_recall(dataset["test"]['label'], pred_prob, ["negative", "positive"])


wandb.finish()


