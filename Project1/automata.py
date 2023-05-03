### CS 691 NLP Spring 2023 Project 1
### Jakob Lovato

import random

def create_automata(option):
  if option == 1:
    dfa = {"q0":{"b":"q1"},
           "q1":{"a":"q2"},
           "q2":{"a":"q3"},
           "q3":{"a":"q3"},
           "accepting":["q3"]}
           
  elif option == 2:
    dfa = {"q0":{"abba":"q1", "baab":"q2"},
           "q1":{"abba":"q1"},
           "q2":{"baab":"q2"},
           "accepting":["q1", "q2"]}
           
  elif option == 3:
    dfa = {"q0":{"zero":"q2",
               "one":"q2",
               "two":"q2",
               "three":"q2",
               "four":"q2",
               "five":"q2",
               "six":"q2",
               "seven":"q2",
               "eight":"q2",
               "nine":"q2",
               "ten":"q2",
               "eleven":"q2",
               "twelve":"q2",
               "thirteen":"q2",
               "fourteen":"q2",
               "fifteen":"q2",
               "sixteen":"q2",
               "seventeen":"q2",
               "eighteen":"q2",
               "nineteen":"q2",
               "twenty":"q1",
               "thirty":"q1",
               "fourty":"q1",
               "fifty":"q1",
               "sixty":"q1",
               "seventy":"q1",
               "eighty":"q1",
               "ninety":"q1"},
           "q1":{" one":"q2",
               " two":"q2",
               " three":"q2",
               " four":"q2",
               " five":"q2",
               " six":"q2",
               " seven":"q2",
               " eight":"q2",
               " nine":"q2"},
           "q2":[],
           "accepting":["q1", "q2"]}
                
  return dfa


def generate_language(automata):
  utterance = ""
  current_state = "q0"
  
  while 1:
    if current_state in automata["accepting"] and len(automata[current_state]) == 0:
      break
    elif current_state in automata["accepting"] and len(automata[current_state]) != 0:
      decision = random.randint(0, 1)
      if decision == 0:
        break
    choice = random.choice(list(automata[current_state].keys()))
    utterance = utterance + choice
    next_state = automata[current_state][choice]
    current_state = next_state
    
  return utterance
  

def recognize_language(automata, utterance):
  auto1 = create_automata(1)
  auto2 = create_automata(2)
  auto3 = create_automata(3)
  
  #if we are using the baa+ automata, split every character
  if automata == auto1:
    n = 1
    string = [utterance[i:i+n] for i in range(0, len(utterance), n)]
    
  #if we are using the (abba)+ or (baab)+ automata, split every 4 characters
  if automata == auto2:
    n = 4
    string = [utterance[i:i+n] for i in range(0, len(utterance), n)]
    
  #if we are using the zero-ninety nine automata, split at spaces
  if automata == auto3:
    string = utterance.split()
    string = [" " + string[i] for i in range(0, len(string))]
    string[0] = string[0].lstrip()
  
  current_state = "q0"
  
  for item in string:
    if item in automata[current_state]:
      next_state = automata[current_state][item]
    else:
      return 0
    current_state = next_state
  
  if current_state in automata["accepting"]:
    return 1
  else:
    return 0
