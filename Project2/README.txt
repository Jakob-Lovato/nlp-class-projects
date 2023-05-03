CS 691 - Natural Language Processing
Spring 2023
Project 2
Jakob Lovato

NOTE: The code takes A LONG TIME to run (about 45 minutes for me)

### For helpers.py:
For this portion of the project I used the packages 'os' and 'random'

# For load_pos_data(fname):
This function reads in words and accompanying POS tags from a file, and returns a list of two lists: the first contains a continuous list of all words (and symbols, characters, etc... whatever the input file considered a "word"). The second list contains all of the POS tags for the accompanying words in the first list.

# For load_spam_data(dirname):
I store the spam and ham data in a bag of words as a dictionary containing counts of how many times each word appears in a spam or a ham email per email. (I originally just had a bag of words for all spam and all ham emails, but realized later that this wouldn't work when going to test, so I added even deeper sub dictionaries to keep track of counts per email. I think this vastly overcomplicated things and made it a bit of a headache later on...) The data is also split into training and testing dictionaries. I do this by reading into the training set for the first 80% of the data, then checking if we've passed the 80% threshold and start reading into the testing set. 
I use the random package to first shuffle the order of all of the emails in the spam and ham folders respectively, so that when it is split into training and testing data, certain words used more frequently in the first 80% of the emails or latter 20% of the emails won't skew the data. For example, the set of emails have dates in the late 90s and early 2000s. So around the time that 9/11 happened, words such as "terrorism", "america", etc. happen much more frequently. So we want to make sure these are interspersed between training and testing data, as opposed to appearing much more frequently in one or the other. 
I also pre-processed the training data by removing words that occurred less than 5 times or greater than 1000 times.
For me, the hardest part of this portion was navigating through a directory with multiple subdirectories, so my code might be a bit janky for that part!

### For hmm.py
For this portion of the project I used the packages 'pandas' and 'numpy'

# For train_hmm(train_data)
I store the transition and emission matrices as pandas dataframes, so that I can label the rows and columns with the string names of the POS tags and the words in the data (to be able to access them later). For the transition matrix, I just iterate through the entirety of the list of tags from load_pos_data(), and sum up how many times each POS tag goes to every other tag. These counts are stored in the dataframe. I then divide each row by its rowsum in order to convert the counts to probabilities. The same procedure was done for the emission matrix, just that the counts represent the number of times each word is tagged as each POS.

# For test_hmm(test_data, hmm_transition, hmm_emission)
I first start by pre-processing the data, and splitting up the two individual lists of words and POS tags into sub lists, splitting every time we see a period. Each of these sequences (sentences) is then stored as a list of lists. The Viterbi algorithm is then implemented to go through each of those sequences and tag the POS. Since there are no given initial probabilities, I used a uniform distribution for the probabilities of the POS tag of the starting word. This is probably not the best way to do things, since I would assume sentences are more likely to start on certain POS than others, but it was done for ease of use and simplicity. For the Viterbi algorithm itself I referenced pseudocode on the Wikipedia article for the Viterbi algorithm. I modified it quite a bit obviously to work with the structure of my data, but the 'guts' follow the logic from Wikipedia. After each sequence is tagged, I get the per-sequence accuracy and overall mean accuracy. The first two sequences in the test data I got almost 90% accuracy.

### For naive_bayes.py
# For train_naive_bayes(train_data):
I think this is where storing my data as nested dictionaries came back to bite me in the ass a bit, as it made counting the words much more complicated than it needed to be. But once again, I stored my model as a dictionary that contained a subdictionary for the prior probabilities, as well as the laplace smoothed probability of each word belonging to each class.

# For test_naive_bayes(test_data, model):
I just iterated through each test email and used my overly complicated dictionary nesting to locate the probabilities of each word belonging to each class. Then I counted how many correct and incorrect classifications were made. At first, when calculating the probabilities of each document belonging to each class, I was accidentally penalizing correct predictions by multiplying the correct-class probability with the probability of each word if that word was in the correct-class training data, BUT NOT multiplying any probability when calculating the incorrect-class probability. So before this fix I was getting ~46% accuracy but after the fix I got almost 77% accuracy. Yay!

### For logistic_regression.py
# For train_logisitic_regression(train_data):
Here, yet again, the way I stored my data came back to bite me a bit. Since I stored each bag of words as a dictionary per email, this didn't include 0's for all words in the vocab but not in each specific email. So I first had to transform my data again. I stored the data and weights in pandas dataframes, which I feel was a mistake. I find pandas to be incredibly janky, especially compared to base dataframes in R, which I am more used to. Anyway... I get an overflow error when calculating the sigmoid function, I think due to the very sparse nature of the bag of words. However even though there is an overflow warning, the code still runs. I also decided to run the code for 50 epochs, both because it took very long to run, and also because I noticed the error rate plateaued around that point.

# For test_logistic_regression(test_data, model):
I once again had to transform my data, and then remove columns for words that weren't in the training data, and then add empty rows in places where words that occurred in the training data, but not test data, just in order to have the dimensions align with the weights. I found my model was extremely overfit, getting about 99% for the training accuracy but only ~57% for the test accuracy. I tried modifying the hyperparameters in the training function, but always found this overfitting problem. I have heard that logistic regression is prone to overfitting in high dimensions, and this data is clearly very high dimensional, so I'm not sure if this is something I can overcome...