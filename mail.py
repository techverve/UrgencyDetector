from imap_tools import MailBox, AND
from tkinter import *
import tkinter as tk
from tkinter import messagebox
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from termcolor import colored
import pickle
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
def inference(loaded_model, vector):
        imap_gmail = "imap.gmail.com"
        my_email = "hemankith@gmail.com"
        my_pass = "pes2201900094"
        # get list of email subjects from INBOX folder
        # with MailBox(imap_gmail).login(my_email, my_pass) as mailbox:
        #     subjects = [msg.subject for msg in mailbox.fetch()]
        #     print(subjects)

        
        # get list of email subjects from INBOX folder - equivalent verbose version
        mailbox = MailBox(imap_gmail)
        #mailbox.login(my_email, my_pass, initial_folder="All Mail")  # or mailbox.folder.set instead 3d arg
        mailbox.login(my_email, my_pass)
        messages = [msg.text for msg in mailbox.fetch(criteria=AND(seen=False,from_="pes.edu"),
                                mark_seen=False,
                                bulk=True)]
        data = {'Content':[messages[0]]}
        # tfidfVectorizer = TfidfVectorizer(min_df=5, max_df=79)
        df_new = pd.DataFrame(data)
        test_tweet_vector = vector.fit_transform(df_new['Content'].apply(lambda x: np.str_(x)))
        pred = loaded_model.predict(test_tweet_vector)
        root = tk.Tk()
        root.withdraw()
        if pred == 0 or pred == 1:
            messagebox.showwarning("Important", "Urgent Please check email")
        else:
            messagebox.showinfo("Not Important", "No urgent mail")


if __name__ == '__main__':
    # vector = pickle.load(open('tfid.pkl', 'rb'))
    # loaded_model = pickle.load(open('./pickle_model.pkl', 'rb'))
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
    tfidfVectorizer = TfidfVectorizer(min_df = 1, max_df = 80, max_features = 100)
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
    inference(loaded_model = randomForestClassifier, vector = tfidfVectorizer)