import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import wandb
import time

import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Experimenting different hyperparameters to Train a linear model with SGD')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--penalty', type=str, default='l2', help='penalty')
parser.add_argument('--loss', type=str, default='log_loss', help='loss function')

args = parser.parse_args()

# login to wandb and Initialize W&B run
datetime = time.strftime("%Y%m%d-%H%M%S")
wandb.login()
wandb.init(
      # Set the project where this run will be logged
      project="sutd-mlops-project", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_session2_run_{datetime}", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": args.learning_rate,
      "loss": args.loss,
      "penalty": args.penalty,
      "architecture": "SGDClassifier",
      "dataset_name": "rotten_tomatoes",
      })
config = wandb.config

# load dataset and perform vectorize transform on the text inputs
dataset = load_dataset(config.dataset_name)
count_vect = CountVectorizer()
train_features = count_vect.fit_transform(dataset['train']['text'])
test_features = count_vect.transform(dataset['test']['text'])

# train a linear model with SGD
model = SGDClassifier(
            loss = config.loss, 
            penalty = config.penalty,
            learning_rate = 'constant', 
            eta0 = config.learning_rate
        ).fit(train_features, dataset['train']['label'])

test_predicted = model.predict(test_features)
test_proba = model.predict_proba(test_features)
accuracy = metrics.accuracy_score(dataset['test']['label'], test_predicted)
print(classification_report(dataset['test']['label'], test_predicted))
print("accuracy: ", accuracy)

wandb.log({"accuracy": accuracy})
wandb.sklearn.plot_precision_recall(
    dataset['test']['label'], 
    test_proba, 
    ["negative", "positive"])

wandb.finish()


