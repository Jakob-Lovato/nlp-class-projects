### CS 691 - Natural Language Processing Spring 2023
### Jakob Lovato
### Hidden Markov Model for POS Tagging
import pandas as pd
import numpy as np

def train_hmm(train_data):
  # Store emission and transition matrices as pandas dataframes
  
  # train_data[0] contains the words and train_data[1] contains the POS tags
  words = train_data[0]
  tags = train_data[1]
  
  set_of_tags = set(tags)
  num_tags = len(set_of_tags)
  
  set_of_words = set(words)
  num_words = len(set_of_words)
  
  # First create transition matrix
  # Create dataframe of all POS tags to store counts; initialize counts as zero
  trans = pd.DataFrame(np.zeros((num_tags, num_tags)), columns = set_of_tags, index = set_of_tags)
  
  # Populate transition matrix with the counts of each POS tag going to each other POS tag
  for i in range(len(tags) - 1):
    trans.loc[tags[i], tags[i + 1]] += 1
    
  # Now turn these counts into probabilities by dividing each row by its row sum
  trans_row_sums = trans.sum(axis = 1)
  trans = trans.div(trans_row_sums, axis = 0)
  
  # Now create the emission matrix
  # Create dataframe of all counts of how many times each word is tagged as each POS
  # Rows are POS tags, columns are words
  emission = pd.DataFrame(np.zeros((num_tags, num_words)), columns = set_of_words, index = set_of_tags)
  
  # Populate emission matrix with the counts of times each word is tagged with each POS
  for i in range(len(tags)):
    emission.loc[tags[i], words[i]] += 1
    
  # Now turn these counts into probabilities by dividing each row by its row sum
  emission_row_sums = emission.sum(axis = 1)
  emission = emission.div(emission_row_sums, axis = 0)

  return trans, emission


def test_hmm(test_data, hmm_transition, hmm_emission):
  # test_data[0] contains the words and test_data[1] contains the POS tags
  words = test_data[0]
  tags = test_data[1]
  
  # Start by splitting the input data into individual sequences to tag
  # Store each sequence of words in a list of lists
  # Also store the corresponding true tags so we can get accuracy at the end
  # Split at each period
  all_word_seq = []
  all_tag_seq = []
  
  curr_count = 0
  last_count = 0
  for word in words:
    if word == '.':
      curr_count += 1
      all_word_seq.append(words[last_count:curr_count])
      all_tag_seq.append(tags[last_count:curr_count])
      last_count = curr_count
    else:
      curr_count += 1
    
  ### Now implement the Viterbi algorithm and tag each sequence in all_word_seq
  ### Viterbi code referenced from Wikipedia pseudocode for Viterbi algorithm
  # Store all the predicted sequence of tags in all_pred_tag_seq
  all_pred_tag_seq = []
  
  # Get number of hidden states (POS tags)
  num_states = hmm_transition.shape[0]
  
  # Initialize uniform prior probabilities for first step
  pi = 1 / num_states
  
  # Run through each sequence and tag it
  for seq in all_word_seq:
    seq_len = len(seq)
    
    #Initialize empty trellis and pointers matrix to backtrace
    trellis = np.zeros([num_states, seq_len])
    pointers = np.zeros([num_states, seq_len]).astype("int")

    for i in range(num_states):
      # Take into account that first word may not have been in training data; give it prob 1 so there is uniform prob on start of trellis
      if seq[0] in hmm_emission:
        trellis[i, 0] = pi * hmm_emission.loc[hmm_emission.index[i], seq[0]]
      else:
        trellis[i, 0] = pi * 1

    for i in range(1, seq_len):
      for j in range(num_states):
        # Take into account that word may not have been in training data; give it emission prob 1 so that only the POS transition matters
        if seq[i] in hmm_emission:
          k = np.argmax([trellis[k, i - 1] * hmm_transition.loc[hmm_transition.index[k], hmm_transition.columns[j]] * hmm_emission.loc[hmm_emission.index[j], seq[i]] for k in range(num_states)])
          trellis[j, i] = trellis[k, i - 1] * hmm_transition.loc[hmm_transition.index[k], hmm_transition.columns[j]] * hmm_emission.loc[hmm_emission.index[j], seq[i]]
          pointers[j, i] = k
        else:
          k = np.argmax([trellis[k, i - 1] * hmm_transition.loc[hmm_transition.index[k], hmm_transition.columns[j]] * 1 for k in range(num_states)])
          trellis[j, i] = trellis[k, i - 1] * hmm_transition.loc[hmm_transition.index[k], hmm_transition.columns[j]] * 1
          pointers[j, i] = k

    best_seq = []
    k = np.argmax([trellis[k, seq_len - 1] for k in range(num_states)])
    
    for i in range(seq_len - 1, -1, -1):
      best_seq.insert(0, hmm_emission.index[k])
      k = pointers[k, i]
      
    all_pred_tag_seq.append(best_seq)
    
  # Get accuracy per sequence and overall accuracy
  all_acc = []
  for pred, truth in zip(all_pred_tag_seq, all_tag_seq):
    accuracy = np.mean(np.array(pred) == np.array(truth))
    all_acc.append(accuracy)
  
  mean_acc = np.mean(all_acc)
  
  return mean_acc, all_acc
    

