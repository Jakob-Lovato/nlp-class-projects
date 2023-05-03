### CS 691 - Natural Language Processing Spring 2023
### Jakob Lovato
### Helper functions for project 2
import os
import random

def load_pos_data(fname):
  # I store the words and POS tags as two vectors of strings
  # Then combine them into a 2d array to return one object
  words = []
  tags = []
  
  fin = open(fname, 'r')
  
  for line in fin:
    # Check if line is empty. If so, skip the line
    if len(line.strip()) == 0:
      continue
    else:
      word = line.split(' ')[0]
      pos = line.split(' ')[1]
      words.append(word)
      tags.append(pos)
        
  fin.close()
  
  data = [words, tags]
  
  return data


def load_spam_data(dirname):
  # I store the spam/ham data as a bag of words in a dictionary containing counts
  # I randomly shuffle the order of the files to read in so that the training/test
  # data is not skewed by certain words that may be used more frequently during
  # certain time periods (i.e., around 9/11, words in the ham files such as "america"
  # and "terrorism" are used frequently. We don't want these words to appear more
  # frequently in the train or test data)
  spam_data_train = {}
  ham_data_train = {}
  spam_data_test = {}
  ham_data_test = {}
  spam_files = os.listdir(os.path.join(dirname, "spam"))
  random.shuffle(spam_files)
  ham_files = os.listdir(os.path.join(dirname, "ham"))
  random.shuffle(ham_files)
  
  # Create one BIG bag of words for spam and ham, to see total counts of all words
  # This was, we can later remove words that occur very fequently or very rarely
  spam_big_bag = {}
  ham_big_bag = {}
  path = os.path.join(dirname, "spam")
  for file in spam_files:
    filepath = os.path.join(path, file)
    fin = open(filepath, 'r', errors = 'ignore')
    for line in fin:
      for word in line.split(' '):
        if word not in spam_big_bag:
          spam_big_bag[word] = 1
        else:
          spam_big_bag[word] += 1
  fin.close()
  
  path = os.path.join(dirname, "ham")
  for file in ham_files:
    filepath = os.path.join(path, file)
    fin = open(filepath, 'r', errors = 'ignore')
    for line in fin:
      for word in line.split(' '):
        if word not in ham_big_bag:
          ham_big_bag[word] = 1
        else:
          ham_big_bag[word] += 1
  
  # First create dictionary bag of words for spam files
  doc_num = 0
  eighty_percent_thresh = round(len(spam_files) * 0.8)
  path = os.path.join(dirname, "spam") 
  for file in spam_files:
    filepath = os.path.join(path, file)
    fin = open(filepath, 'r', errors = 'ignore')
    # If we've hit the 80% threshold, start adding to test data
    if doc_num > eighty_percent_thresh:
      spam_data_test[doc_num] = {}
      for line in fin:
        for word in line.split(' '):
          # If the word occurs more than 1000 times or less then 5 times in the total data, skip it
          #if 5 < spam_big_bag[word] < 1000:
          if word not in spam_data_test[doc_num]:
            spam_data_test[doc_num][word] = 1
          else:
            spam_data_test[doc_num][word] += 1
    # If we haven't hit 80% threshold, add to train data
    else:
      spam_data_train[doc_num] = {}
      for line in fin:
        for word in line.split(' '):
          # If the word occurs more than 1000 times or less then 5 times in the total data, skip it
          if 5 < spam_big_bag[word] < 1000:
            if word not in spam_data_train[doc_num]:
              spam_data_train[doc_num][word] = 1
            else:
              spam_data_train[doc_num][word] += 1
            
    doc_num += 1
    
    fin.close()
      
  # Now create dictionary bag of words for ham files
  doc_num = 0
  eighty_percent_thresh = round(len(ham_files) * 0.8)
  path = os.path.join(dirname, "ham") 
  for file in ham_files:
    filepath = os.path.join(path, file)
    fin = open(filepath, 'r', errors = 'ignore')
    # If we've hit the 80% threshold, start adding to test data
    if doc_num > eighty_percent_thresh:
      ham_data_test[doc_num] = {}
      for line in fin:
        for word in line.split(' '):
          # If the word occurs more than 1000 times or less then 5 times in the total data, skip it
          #if 5 < ham_big_bag[word] < 1000:
          if word not in ham_data_test[doc_num]:
            ham_data_test[doc_num][word] = 1
          else:
            ham_data_test[doc_num][word] += 1
    # If we haven't hit 80% threshold, add to train data
    else:
      ham_data_train[doc_num] = {}
      for line in fin:
        for word in line.split(' '):
          # If the word occurs more than 1000 times or less then 5 times in the total data, skip it
          if 5 < ham_big_bag[word] < 1000:
            if word not in ham_data_train[doc_num]:
              ham_data_train[doc_num][word] = 1
            else:
              ham_data_train[doc_num][word] += 1
            
    doc_num += 1
    
    fin.close()
      
  train_data = {"spam": spam_data_train, "ham": ham_data_train}
  test_data = {"spam": spam_data_test, "ham": ham_data_test}
  
  return train_data, test_data
