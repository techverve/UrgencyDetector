import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from nltk import PorterStemmer

import re
import nltk
import numpy as np
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from termcolor import colored
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import pandas as pd
from termcolor import colored
from sklearn.model_selection import train_test_split

import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
from termcolor import colored
import pickle

# Read dataset


# Function to expand tweet
def expand_tweet(tweet):
	expanded_tweet = []
	for word in tweet:
		if re.search("n't", word):
			expanded_tweet.append(word.split("n't")[0])
			expanded_tweet.append("not")
		else:
			expanded_tweet.append(word)
	return expanded_tweet

# Function to process tweets
def clean_tweet(data, wordNetLemmatizer, porterStemmer):
	data['Clean_Content'] = data['Content']
	print(colored("Removing user handles starting with @", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].str.replace("@[\w]*","")
	print(colored("Removing numbers and special characters", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].str.replace("[^a-zA-Z' ]","")
	print(colored("Removing urls", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")
	print(colored("Removing single characters", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].replace(re.compile(r"(^| ).( |$)"), " ")
	print(colored("Tokenizing", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].str.split()
	print(colored("Removing stopwords", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].apply(lambda tweet: [word for word in tweet if word not in STOPWORDS])
	print(colored("Expanding not words", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].apply(lambda tweet: expand_tweet(tweet))
	print(colored("Lemmatizing the words", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].apply(lambda tweet: [wordNetLemmatizer.lemmatize(word) for word in tweet])
	print(colored("Stemming the words", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].apply(lambda tweet: [porterStemmer.stem(word) for word in tweet])
	print(colored("Combining words back to tweets", "yellow"))
	data['Clean_Content'] = data['Clean_Content'].apply(lambda tweet: ' '.join(tweet))
	return data

# Define processing methods
if __name__ == "__main__":


        dataset = pd.read_csv('./urgency.csv')

# Train test split
        print(colored("Splitting train and test dataset into 80:20", "yellow"))
        X_train, X_test, y_train, y_test = train_test_split(dataset['Content'], dataset['Target'], test_size = 0.20, random_state = 100)
        train_dataset = pd.DataFrame({
            'Content': X_train,
            'Target': y_train
            })
        print(colored("Train data distribution:", "yellow"))
        print(train_dataset['Target'].value_counts())
        test_dataset = pd.DataFrame({
            'Content': X_test,
            'Target': y_test
            })
        print(colored("Test data distribution:", "yellow"))
        print(test_dataset['Target'].value_counts())
        print(colored("Split complete", "yellow"))

        # Save train data
        print(colored("Saving train data", "yellow"))

        train_dataset.to_csv('train.csv', index = False)
        print(colored("Train data saved to train.csv", "green"))

        # Save test data
        print(colored("Saving test data", "yellow"))
        test_dataset.to_csv('test.csv', index = False)
        print(colored("Test data saved to test.csv", "green"))

        # Import datasets
        print("Loading data")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')

        # Setting stopwords
        STOPWORDS = set(stopwords.words('english'))
        STOPWORDS.remove("not")
  
        wordNetLemmatizer = WordNetLemmatizer()
        porterStemmer = PorterStemmer()

        # Pre-processing the tweets
        print(colored("Processing train data", "green"))
        train_data = clean_tweet(train_data, wordNetLemmatizer, porterStemmer)
        train_data.to_csv('clean_train.csv', index = False)
        print(colored("Train data processed and saved to clean_train.csv", "green"))
        print(colored("Processing test data", "green"))
        test_data = clean_tweet(test_data, wordNetLemmatizer, porterStemmer)
        test_data.to_csv('clean_test.csv', index = False)
        print(colored("Test data processed and saved to clean_test.csv", "green"))
        print(colored("Loading train and test data", "yellow"))
        train_data = pd.read_csv('clean_train.csv')
        test_data = pd.read_csv('clean_test.csv')
        print(colored("Data loaded", "yellow"))

        # Tf-IDF
        print(colored("Applying TF-IDF transformation", "yellow"))
        tfidfVectorizer = TfidfVectorizer(min_df = 5, max_df = 79)
        tfidfVectorizer.fit(train_data['Clean_Content'].apply(lambda x: np.str_(x)))
        train_tweet_vector = tfidfVectorizer.transform(train_data['Clean_Content'].apply(lambda x: np.str_(x)))
        test_tweet_vector = tfidfVectorizer.transform(test_data['Clean_Content'].apply(lambda x: np.str_(x)))

        # Training
        print(colored("Training Random Forest Classifier", "yellow"))
        randomForestClassifier = RandomForestClassifier()
        randomForestClassifier.fit(train_tweet_vector, train_data['Target'])

        # Prediction
        print(colored("Predicting on train data", "yellow"))
        prediction = randomForestClassifier.predict(train_tweet_vector)
        print(colored("Training accuracy: {}%".format(accuracy_score(train_data['Target'], prediction)*100), "green"))

        print(colored("Predicting on test data", "yellow"))
        prediction = randomForestClassifier.predict(test_tweet_vector)
        print(colored("Testing accuracy: {}%".format(accuracy_score(test_data['Target'], prediction)*100), "green"))
        

#
# Create your model here (same as above)
#

# Save to file in the current working directory
        pkl_filename = "pickle_model.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(randomForestClassifier, file)
        pickle.dump(TfidfVectorizer, open('tfid.pkl', 'wb'))
        # print(colored("Loading train and test data", "yellow"))
        # train_data = pd.read_csv('clean_train.csv')
        # test_data = pd.read_csv('clean_test.csv')
        # print(colored("Data loaded", "yellow"))

        # # Tokenization
        # print(colored("Tokenizing and padding data", "yellow"))
        # tokenizer = Tokenizer(num_words = 2000, split = ' ')
        # tokenizer.fit_on_texts(train_data['Clean_Content'].astype(str).values)
        # train_tweets = tokenizer.texts_to_sequences(train_data['Clean_Content'].astype(str).values)
        # max_len = max([len(i) for i in train_tweets])
        # train_tweets = pad_sequences(train_tweets, maxlen = max_len)
        # test_tweets = tokenizer.texts_to_sequences(test_data['Clean_Content'].astype(str).values)
        # test_tweets = pad_sequences(test_tweets, maxlen = max_len)
        # print(colored("Tokenizing and padding complete", "yellow"))

        # # Building the model
        # print(colored("Creating the LSTM model", "yellow"))
        # model = Sequential()
        # model.add(Embedding(2000, 128, input_length = train_tweets.shape[1]))
        # model.add(SpatialDropout1D(0.4))
        # model.add(LSTM(256, dropout = 0.2))
        # model.add(Dense(3, activation = 'softmax'))
        # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        # model.summary()