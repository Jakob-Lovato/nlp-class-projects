### CS 691 - Natural Language Processing Spring 2023
### Jakob Lovato

import os
import pandas as pd
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

def load_data(fname):
  # The data has six columns, but we only care about
  # column 0 (which contains the labels)
  # and column 5 (which contains the tweets)
  
  # Use encoding = 'latin-1' to avoid unicode error, and use
  # header = None to add column names
  data = pd.read_csv(fname, encoding = 'latin-1', header = None)
  data = data.drop(data.columns[[1, 2, 3, 4]], axis = 1)
  
  labels = list(data[data.columns[0]])
  tweets = list(data[data.columns[1]])
  
  # Replace 4s with 1s in labels
  labels = [1 if x == 4 else x for x in labels]
  
  return tweets, labels


def clean_data(documents):
  # Remove any tagged accounts (i.e. @ladygaga), any links, and punctuation except for apostrophes to keep contractions
  # Then make everything lowercase
  cleaned_tweets = [' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(http.+com)", " ", x.lower()).split()) for x in documents]
  
  return cleaned_tweets


def train_doc2vec(cleaned_documents):
  # Tokenize the data
  tagged_data = [TaggedDocument(words = word_tokenize(d), tags = [str(i)]) for i, d in enumerate(cleaned_documents)]
  
  # Train doc2vec
  model = Doc2Vec(vector_size = 150,
                  #alpha = alpha, 
                  #min_alpha = 0.00025,
                  min_count = 100,
                  dm = 1, #use dm = 1 to preserve order
                  max_vocab_size = 150000,
                  epochs = 15) 
    
  model.build_vocab(tagged_data)
  model.train(tagged_data, total_examples = model.corpus_count, epochs = model.epochs)
 
  model.save("d2v.model")
  return model


def tokenize_dataset(cleaned_documents, d2v_model):
  # We need to tokenize the data again first in order to infer the vectors
  tokenized_tweets = [word_tokenize(tweet) for tweet in cleaned_documents]
  
  # Now vectorize the tweets using the Doc2Vec model
  vectorized_tweets = [d2v_model.infer_vector(tweet) for tweet in tokenized_tweets]
  
  return vectorized_tweets


def train(X_train, y_train):
  # First model: logistic regression
  lr = LogisticRegression().fit(X_train, y_train)
  
  # Second model: random forest
  rf = RandomForestClassifier().fit(X_train, y_train)
  
  return {"Logistic Regression": lr, "Random Forest": rf}


def test(trained_models_dict, X_test, y_test):
  # Logistic regression
  lr = trained_models_dict["Logistic Regression"]
  lr_preds = lr.predict(X_test)
  lr_accuracy = accuracy_score(y_test, lr_preds)
  lr_balanced_accuracy = balanced_accuracy_score(y_test, lr_preds)
  lr_f1 = f1_score(y_test, lr_preds)
  
  # Random forest
  rf = trained_models_dict["Random Forest"]
  rf_preds = rf.predict(X_test)
  rf_accuracy = accuracy_score(y_test, rf_preds)
  rf_balanced_accuracy = balanced_accuracy_score(y_test, rf_preds)
  rf_f1 = f1_score(y_test, rf_preds)
  
  metrics = {"Logistic Regression Accuracy": lr_accuracy,
  "Logistic Regression Balanced Accuracy": lr_balanced_accuracy,
  "Logistic Regression F1 Score": lr_f1,
  "Random Forest Accuracy": rf_accuracy,
  "Random Forest Balanced Accuracy": rf_balanced_accuracy,
  "Random Forest F1 Score": rf_f1}
  
  return metrics
  



################ TEST SCRIPT
# from sklearn.model_selection import train_test_split
# 
# # example: "./twitter_sentiment_data.csv"
# #fname = input("Please input the path to the dataset: ")
# fname = "/Users/jakoblovato/Desktop/CS 691/Project 3/Project3/twitter_sentiment_data.csv"
# 
# documents, labels = load_data(fname)
# 
# cleaned_documents = clean_data(documents)
# 
# d2v_model = train_doc2vec(cleaned_documents)
# 
# vectorized_cleaned_documents = tokenize_dataset(cleaned_documents, d2v_model)
# 
# X_train, X_test, y_train, y_test = train_test_split(vectorized_cleaned_documents, labels, test_size=0.33, random_state=1, stratify=labels)
# 
# trained_models_dict = train(X_train, y_train)
# 
# model_performance_metric_dict = test(trained_models_dict, X_test, y_test)
# 
# for dict_key in model_performance_metric_dict.keys():
#     print(f"{dict_key}: {model_performance_metric_dict[dict_key]}")
