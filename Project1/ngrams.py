### CS 691 NLP Spring 2023 Project 1
### Jakob Lovato

import random

def train_ngram(train_data, n):
  #get list where each word is a list element
  token_list = [word for line in train_data for word in line.split()]
  n_tokens = len(token_list)
  
  #get list of unique tokens, aka types
  types_list = list(set(token_list))
  n_types = len(types_list)
  
  #get counts for each word
  word_counts = {word: 0 for word in types_list}
  for word in token_list:
    word_counts[word] += 1
  
  #get ngrams
  ngrams = []
  for i in range(len(token_list) - n + 1):
    gram = token_list[i]
    for j in range(1, n):
      gram = gram + " " + token_list[i + j]
    ngrams.append(gram)
    
  #this takes into account 'restarting' finding ngrams at each new lines
  if n != 1:
    for gram in ngrams[:]:
      if gram.startswith("</s>") or gram.endswith("<s>"): 
        ngrams.remove(gram)
  
  #get ngram probabilities
  if n == 1:
    ngram_probs = {gram: 0 for gram in ngrams}
    for gram in ngrams:
      c_gram = ngrams.count(gram)
      prob = c_gram / n_tokens
      ngram_probs[gram] = prob
      
  if n == 2:
    ngram_probs = {gram: 0 for gram in ngrams}
    for gram in ngrams:
      c_gram = ngrams.count(gram)
      c_prior = token_list.count(gram.split()[0])
      prob = c_gram / c_prior
      ngram_probs[gram] = prob
      
  if n == 3:
    single_sentence = " ".join(token_list)
    ngram_probs = {gram: 0 for gram in ngrams}
    for gram in ngrams:
      c_gram = ngrams.count(gram)
      c_prior_1 = token_list.count(gram.split()[0])
      c_prior_2 = single_sentence.count(" ".join(gram.split()[:-1]))
      prob = c_gram / c_prior_2
      ngram_probs[gram] = prob
  
  return(ngram_probs)
  
  
def generate_language(ngram_model, max_words):
  n = len(list(ngram_model.keys())[0].split())
  #list of words to choose from
  temp = list(ngram_model.keys())
  words = list(set([word for gram in temp for word in gram.split()]))
  types_list = list(ngram_model.keys())
  
  #initialize utterance
  utterance = ["<s>"]
  
  #generate sentence
  while len(utterance) < max_words:
    #if unigram model, all choices are independent
    if n == 1:
      next_word = "".join(random.choices(list(ngram_model.keys()), weights = ngram_model.values()))
      utterance.append(next_word)
      if next_word == "</s>":
        utterance = " ".join(utterance)
        return(utterance)
    else:
      if len(utterance) == 1:
        #get the possible words after "<s>"
        options = [gram.split()[1] for gram in list(ngram_model.keys()) if gram.startswith("<s>")]
        #randomly select word from list of options after "<s>"
        #since words can be repeated in 'options', and ramdom.choice() picks a word with
        #uniform probability, the probability distributrion of the next word is taken care of
        next_word = random.choice(options)
        utterance.append(next_word)
    #if bigram model
    if n == 2:
      #subset of options for next word based on prior word
      subset = {gram:ngram_model[gram] for gram in ngram_model if gram.startswith(utterance[len(utterance) - 1])}
      next_gram = "".join(random.choices(list(subset.keys()), weights = subset.values()))
      next_word = next_gram.split()[1]
      utterance.append(next_word)
      if next_word == "</s>":
        utterance = " ".join(utterance)
        return(utterance)
    #if trigram model
    if n == 3:
      #subset of options for next word based on prior word
      prior_two = utterance[(len(utterance) - 2):len(utterance)]
      prior_two = " ".join(prior_two)
      subset = {gram:ngram_model[gram] for gram in ngram_model if gram.startswith(prior_two)}
      next_gram = "".join(random.choices(list(subset.keys()), weights = subset.values()))
      next_word = next_gram.split()[2]
      utterance.append(next_word)
      if next_word == "</s>":
        utterance = " ".join(utterance)
        return(utterance)
      
  #convert result from list of strings to a single string
  utterance = " ".join(utterance)
  return(utterance)


def calculate_probability(utterance, ngram_model):
  n = len(list(ngram_model.keys())[0].split())
  
  #split utterance into ngrams
  token_list = utterance.split()
  ngrams = []
  for i in range(len(token_list) - n + 1):
    gram = token_list[i]
    for j in range(1, n):
      gram = gram + " " + token_list[i + j]
    ngrams.append(gram)
    
  #calculate probability
  prob = 1
  for gram in ngrams:
    if gram in ngram_model:
      prob *= ngram_model[gram]
    else:
      return(0)
    
  return(prob)
    
    
