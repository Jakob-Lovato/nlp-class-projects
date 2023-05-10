CS 691 Spring 2023 - Natural Language Processing
Project 3
Jakob Lovato


def load_data(fname):
I used the read_csv() function from Pandas to read in the data into a dataframe. I used the argument. 'encoding = 'latin-1'' because I was at first having unicode errors, but this fixed that. I also used the argument 'header = None' to specify that the data did not have column names, so this added fillers integers for the column names so that I could easily remove the unnecessary columns (we only wanted to keep column 0 which contained the labels, and column 5 which contained the tweets).
I also swapped the labels from 4's to 1's in this function instead of in the clean_data() function, since the test script only asks for the cleaned documents to be returned from clean_data(), but not the cleaned labels.


def clean_data(documents):
Again, I did not clean the labels in this function; this was done in the prior function. I cleaned all the tweets by removing any tagged accounts in the tweets (such as @ladygaga) and removing any hyperlinks. I felt that tagged accounts and links would not contribute to the sentiment of the tweet, and this would also help reduce the vocabulary size greatly. I also removed punctuation as to just focus on the words in each tweet, since that is what contributes most to being able to discern sentiment (note: I kept in apostrophes because otherwise the regex I used would change "don't" to "don t"). Finally, I made all of the text lowercase.


def train_doc2vec(cleaned_documents):
I messed around with the hyperparameters of doc2vec a bit, but couldn't spend too much time fine tuning it just due to the length it took to train the model. So I mostly looked through the api documentation and used the provided Medium article for guidance. I set the vector size at 150. I feel that perhaps I should have set it higher, since there is a very large vocabulary, and the documentation used a vector size of 50 for a much smaller corpus. However I think this would have increased training time even more, which I wanted to avoid. I set min_count to 100 to remove words that occur fewer than 100 times. I initially used a value of 3 for this argument, but found an improvement in accuracy when I increased it a bunch (with 1.6 million tweets, words occurring less than 100 times likely don't contribute much to any meaning). The dm argument i set to 1, which uses a 'distributed memory' training algorithm, which maintains the order of the tweets, as opposed to dm=0 which just uses a bag of words, which I figured would create a less accurate model. I set max_vocab_size to 150,000 which was sort of just a guess. I figured this would include the most common words that would help to determine sentiment, including emotional adjectives. I think using this meant I also didn't need to use the min_count argument however. Finally, I set the epoch count to 15, as the documentation recommended 10-20 epochs for corpuses of 10's of thousands to millions of documents.


def tokenize_data(cleaned_documents, d2v_model):
NOTE: The project guidelines call this tokenize_data, but in the test script it calls tokenize_dataset. So in my code, I called it tokenize_dataset so that it would work with the test script. 


def train(X_train, y_train):
I chose to use logistic regression and a random forest, as I wanted to try a linear and non-linear classifier to see if there was any major differences. I used the default parameters for both, since trying to fine tune these hyperparameters in addition to the doc2vec ones would have been very time consuming. I mainly wanted to see how well the doc2vec model was fitted as opposed to the classification models.


def test(trained_models_dict, X_test, y_test):
Logistic regression slightly outperformed the random forest in all metrics, which surprised me a bit. I thought a random forest might perform better here as it can be more flexible, but since the vector size is 150, this might have thrown it off a bit.
The results I got are:
Logistic Regression Accuracy: 0.7455359848484848Logistic Regression Balanced Accuracy: 0.7455359848484848Logistic Regression F1 Score: 0.7556696775225996Random Forest Accuracy: 0.7385473484848485Random Forest Balanced Accuracy: 0.7385473484848485Random Forest F1 Score: 0.7410850096122285



