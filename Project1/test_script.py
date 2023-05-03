import random
import automata
import edit_distance
import ngrams

def load_data(fname):
  data = []
  fin = open(fname,'r')
  for line in fin:
    line = line.strip()
    data.append(line)
  fin.close()
  return data

#test automata
print("Automata Option 1")
automaton = automata.create_automata(1)
utterance = automata.generate_language(automaton)
print("Generated Utterance:", utterance)
accept = automata.recognize_language(automaton, 'baabaa')
print("baabaa", accept)
accept = automata.recognize_language(automaton, 'baabbaa')
print("baabbaa", accept)
print("-----------------------------------------------")

print("Automata Option 2")
automaton = automata.create_automata(2)
utterance = automata.generate_language(automaton)
print("Generated Utterance:", utterance)
accept = automata.recognize_language(automaton, 'abbaabbaabba')
print("abbaabbaabba", accept)
accept = automata.recognize_language(automaton, 'baaab')
print("baaab", accept)
print("-----------------------------------------------")


print("Automata Option 3")
automaton = automata.create_automata(3)
utterance = automata.generate_language(automaton)
print("Generated Utterance:", utterance)
accept = automata.recognize_language(automaton, 'twenty five')
print("twenty five", accept)
accept = automata.recognize_language(automaton, 'twelve one')
print("twelve one", accept)
print("===============================================\n")

#test edit distance
source = "intention"
target = "execution"
dist = edit_distance.calc_min_edit_dist(source, target)
print("Edit Distance:", dist)
s, t, o = edit_distance.align(source, target)
print("Alignment:")
print(s)
print(t)
print(o)
print("===============================================\n")

#test ngrams
train_data = load_data('data1.txt')
model = ngrams.train_ngram(train_data, 1)
utterance = ngrams.generate_language(model, 10)
print("Unigram Model")
print("Generated Utterance:", utterance)
prob = ngrams.calculate_probability("<s> Sam I am </s>", model)
print("Probability:", prob)
print("-----------------------------------------------")
model = ngrams.train_ngram(train_data, 2)
utterance = ngrams.generate_language(model, 10)
print("Bigram Model")
print("Generated Utterance:", utterance)
prob = ngrams.calculate_probability("<s> Sam I am </s>", model)
print("Probability:", prob)
print("-----------------------------------------------")
model = ngrams.train_ngram(train_data, 3)
utterance = ngrams.generate_language(model, 10)
print("Trigram Model")
print("Generated Utterance:", utterance)
prob = ngrams.calculate_probability("<s> Sam I am </s>", model)
print("Probability:", prob)
print("===============================================\n")





