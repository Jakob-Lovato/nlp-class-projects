### CS 691 - Natural Language Processing Spring 2023
### Jakob Lovato
### Logistic Regression for spam detection

import numpy as np
import pandas as pd

def train_logistic_regression(train_data):
  # In my load_spam_data function, I store bags of words per email 
  # (only containing words in the email)
  # This needs to be converted to bags of words with 0's for ALL words
  # in training data
  
  rows = []
  
  for doc in train_data["spam"]:
    rows.append(train_data["spam"][doc])
  
  for doc in train_data["ham"]:
    rows.append(train_data["ham"][doc])
    
  x = pd.DataFrame.from_dict(rows, orient = "columns")
  x = x.fillna(0)
  x.reset_index(inplace = True, drop = True)
  
  # Create vector of response y (class - spam is 0, ham is 1)
  y1 = np.repeat(0, len(train_data["spam"]))
  y2 = np.repeat(1, len(train_data["ham"]))
  y = np.append(y1, y2)
  y = pd.DataFrame(y)
  
  # Shuffle x and y
  randomize = np.arange(len(y))
  np.random.shuffle(randomize)
  x = x.iloc[randomize]
  y = y.iloc[randomize]
  
  x.reset_index(inplace = True, drop = True)
  y.reset_index(inplace = True, drop = True)

  ### Now we train the logistic regression model using gradient descent
  # Initialize random weights and bias
  w = pd.DataFrame(np.random.uniform(-3, 3, size = (1, x.shape[1])), columns = x.columns.tolist())
  b = np.random.uniform(-3, 3)
  
  # Set learning rate 
  eta = 0.02
  
  # Gradient descent algorithm
  # Run for 50 epochs
  for epoch in range(50):
    all_loss = []
    for i in range(x.shape[0]):
      y_true = y.iloc[i].item()
      # Make prediction using sigmoid function
      # For some reason, np.array(w) makes a 2d array, so .ravel() fixes this
      y_hat = 1 / (1 + np.exp(-(np.dot(np.array(x.iloc[i,:]), np.array(w).ravel()) + b)))
      
      # Calculate loss function so we can stop loop after loss is sufficiently small
      # Nevermind - this gave crazy values, I think due to the sparse nature of the bag of words
      # Instead I just calculated the error y_hat - y_true
      all_loss.append(y_hat - y_true)
      
      #I'm using the derivatives of crossentropy loss posted on Piazza
      dLdw = (y_hat - y_true) * x.iloc[i,:]
      dLdb = y_hat - y_true

      #update weights and bias
      w = w.sub(eta * dLdw)
      b = b - (eta * dLdb)
      
    #print(np.mean(all_loss))
    
  
  return (w, b)


def test_logistic_regression(test_data, model):
  # Get weights and bias from model object
  w = model[0]
  b = model[1]
  
  # In my load_spam_data function, I store bags of words per email 
  # (only containing words in the email)
  # This needs to be converted to bags of words with 0's for ALL words
  # in test data
  # However now I need to only use words that appeared in the training data
  
  rows = []
  
  for doc in test_data["spam"]:
    rows.append(test_data["spam"][doc])
  
  for doc in test_data["ham"]:
    rows.append(test_data["ham"][doc])
    
  x = pd.DataFrame.from_dict(rows, orient = "columns")
  x = x.fillna(0)
  
  # Now get only the columns present in the weights matrix, then add empty columns to fill anything
  # in the training data that wasn't in the test data
  col_names = w.columns.tolist()
  intersect_colnames = list(set(col_names).intersection(set(x.columns.tolist())))

  x = x.loc[:,intersect_colnames]
  x = x.reindex(columns = col_names)
  x = x.fillna(0)
  
  # Create vector of response y (class - spam is 0, ham is 1)
  y1 = np.repeat(0, len(test_data["spam"]))
  y2 = np.repeat(1, len(test_data["ham"]))
  y = np.append(y1, y2)
  y = pd.DataFrame(y)
  
  # Go through every sample and get a prediction
  preds = []
  
  for i in range(x.shape[0]):
    y_hat = 1 / (1 + np.exp(-(np.dot(np.array(x.iloc[i,:]), np.array(w).ravel()) + b)))
    if y_hat > 0.5:
      preds.append(1)
    else:
      preds.append(0)
  
  # Get accuracy
  true_y = y.values.flatten()
  true_y = np.array(true_y)
  preds = np.array(preds)
  accuracy = np.mean(true_y == preds)
  
  return accuracy
  
