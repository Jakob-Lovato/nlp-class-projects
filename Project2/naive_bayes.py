### CS 691 - Natural Language Processing Spring 2023
### Jakob Lovato
### Naive Bayes for spam detection

def train_naive_bayes(train_data):
  # Store the model as a dictionary holding class probabilities
  # as well as conditional word probabilities for each class
  model = {}
  
  # Get size of vocabulary as well as number of words (not unique!) in spam and ham sets
  spam_set = set()
  spam_count = 0
  for doc in train_data["spam"]:
    spam_set = set.union(spam_set, set(train_data["spam"][doc]))
    spam_count += sum(train_data["spam"][doc].values())
    
  num_spam_words = len(spam_set)
  
  ham_set = set()
  ham_count = 0
  for doc in train_data["ham"]:
    ham_set = set.union(ham_set, set(train_data["ham"][doc]))
    ham_count += sum(train_data["ham"][doc].values())
    
  num_ham_words = len(ham_set)

  vocab = set.union(spam_set, ham_set)
  v = len(vocab)

  # Get prior probabilities of belonging to spam or ham
  num_docs = len(train_data["spam"]) + len(train_data["ham"])
  p_spam = len(train_data["spam"]) / num_docs
  p_ham = len(train_data["ham"]) / num_docs
  
  model["priors"] = {}
  model["priors"]["p_spam"] = p_spam
  model["priors"]["p_ham"] = p_ham
  
  # Calculate conditional probabilities for each word
  spam_word_counts = {}
  for word in spam_set:
    count = 0
    for doc in train_data["spam"]:
      if word not in train_data["spam"][doc]:
        continue
      else:
        count += train_data["spam"][doc][word]
        
    spam_word_counts[word] = count
    
  # Perform laplace smoothing  
  spam_probs = {key: ((val + 1) / (num_spam_words + v)) for key, val in spam_word_counts.items()}
  
  #### TESTING SOMETHING: no laplace smoothing
  #spam_probs = {key: val / num_spam_words for key, val in spam_word_counts.items()}
  
  
  ham_word_counts = {}
  for word in ham_set:
    count = 0
    for doc in train_data["ham"]:
      if word not in train_data["ham"][doc]:
        continue
      else:
        count += train_data["ham"][doc][word]
        
    ham_word_counts[word] = count
    
  # Perform laplace smoothing  
  ham_probs = {key: ((val + 1) / (num_ham_words + v)) for key, val in ham_word_counts.items()}
  
  #### TESTING SOMETHING: no laplace smoothing
  #ham_probs = {key: val / num_ham_words for key, val in ham_word_counts.items()}
  
  # Add the probs to the model and then return it
  model["spam_probs"] = spam_probs
  model["ham_probs"] = ham_probs
  
  return model
  


def test_naive_bayes(test_data, model):
  # Store counts of correct vs incorrect classifications
  correct = 0
  incorrect = 0
  
  # Go through every document in test data and classify it
  # First spam
  for doc in test_data["spam"]:
    prob_spam = model["priors"]["p_spam"]
    for word in list(test_data["spam"][doc]):
      if word in model["spam_probs"] and word in model["ham_probs"]:
        prob_spam *= model["spam_probs"][word]
      
    prob_ham = model["priors"]["p_ham"]
    for word in list(test_data["spam"][doc]):
      if word in model["ham_probs"] and word in model["spam_probs"]:
        prob_ham *= model["ham_probs"][word]
      
    if prob_spam > prob_ham:
      correct += 1
    else:
      incorrect += 1
  
  # Now ham
  for doc in test_data["ham"]:
    prob_spam = model["priors"]["p_spam"]
    for word in list(test_data["ham"][doc]):
      if word in model["spam_probs"] and word in model["ham_probs"]:
        prob_spam *= model["spam_probs"][word]
      
    prob_ham = model["priors"]["p_ham"]
    for word in list(test_data["ham"][doc]):
      if word in model["ham_probs"] and word in model["spam_probs"]:
        prob_ham *= model["ham_probs"][word]
      
    if prob_spam < prob_ham:
      correct += 1
    else:
      incorrect += 1
      
  accuracy = correct / (correct + incorrect)
  
  return accuracy

  
