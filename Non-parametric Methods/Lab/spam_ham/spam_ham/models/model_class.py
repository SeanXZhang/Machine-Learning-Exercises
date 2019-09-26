import os.path
import time
from typing import List

import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import spam_ham.config as config
from spam_ham.util import preprocess, utils

"""
If you work with a big dataset or a big model, which requires a long time to train,
you will want to be able to save your trained model and use it to predict new data later.

In Python, we can use the pickle package to save/load any binary files.

Useful resources: http://scikit-learn.org/stable/modules/model_persistence.html
"""

en_stopwords = set(stopwords.words('english'))  # the set of stopwords

BIAS = 1.0
REMOVE_STOPWORDS = True
MIN_DF = 2


class Model:
    def __init__(self, clf_type='NB'):
        self.remove_stopwords = REMOVE_STOPWORDS

        if clf_type == 'NB':
            self.clf = MultinomialNB(alpha=BIAS, fit_prior=True)
        elif clf_type == 'LR':
            self.clf = LogisticRegression(class_weight='balanced')
        elif clf_type == 'DT':
            self.clf = DecisionTreeClassifier(class_weight='balanced')
        else:
            raise ValueError(f'Unsupported model type {clf_type}')

        self.vectorizer = TfidfVectorizer(
            min_df=MIN_DF)  # the TF-IDF vectorizer used to vectorize texts

        # The model folder to save the model and vectorizer
        model_subdir = os.path.join(config.MODELS_SUBDIR, clf_type)
        # If the folder does not exists, create it first
        if not os.path.exists(model_subdir):
            os.makedirs(model_subdir)
        self.model_path = os.path.join(model_subdir, 'model')
        self.vectorizer_path = os.path.join(model_subdir, 'vectorizer')

    def load_model(self):
        """
        To load a previously trained model, we need both the vectorizer and the model
        :return: None
        """
        print("Loading vectorizer and model...")
        self.vectorizer = utils.load_binary(self.vectorizer_path)
        self.clf = utils.load_binary(self.model_path)
        print("Finished loading vectorizer and model.\n")

    def process_train_data(self, msg_train: List[str]):
        # for each email, we apply preprocess()
        words_train = [preprocess.preprocess(message, remove_stopwords=self.remove_stopwords) for
                       message in msg_train]

        # use sklearn's tf-idf vectorizer to vectorizer `words_train`
        # tf-idf vectorizer needs to work with list of strings,
        # so we need to combine the words into a string
        text_train = [' '.join(words) for words in words_train]

        # fit a TF-IDF vectorizer on `text_train`
        X_train = self.vectorizer.fit_transform(text_train)

        return X_train

    def process_test_data(self, msg_test: List[str]):
        # use the same preprocessing on the test email message
        words_test = [preprocess.preprocess(message, remove_stopwords=self.remove_stopwords) for
                      message in msg_test]

        # use the fitted vectorizer to **transform** the test words
        text_test = [' '.join(words) for words in words_test]
        X_test = self.vectorizer.transform(text_test)  # DO NOT call `fit_transform`!!

        return X_test

    def train(self, msg_train: List[str], y_train: List[str]):
        prev_time = time.time()
        print("Start transforming training data...")
        X_train = self.process_train_data(msg_train)
        print(
            f'Finished transforming training data in {round(time.time() - prev_time, 2)} seconds\n')

        # Fit Naive Bayes classifier
        prev_time = time.time()
        print("Start fitting Naive Bayes classifier...")
        self.clf.fit(X_train, y_train)
        print(f'Finished training in {round(time.time() - prev_time, 2)} seconds\n')

        """
        Important: save both the vectorizer and the classifier 
        """
        print(f'Saving the fit vectorizer to {self.vectorizer_path}...')
        utils.save_binary(self.vectorizer, self.vectorizer_path)
        print(f'Finished saving the vectorizer.\n')

        print(f'Saving the fit model to {self.model_path}...')
        utils.save_binary(self.clf, self.model_path)
        print(f'Finished saving the model.\n')

    def predict(self, msg_test: List[str]) -> np.array:
        X_test = self.process_test_data(msg_test)

        return self.clf.predict(X_test)

    def evaluate(self, msg_train: List[str], msg_test: List[str], y_train: List[str],
                 y_test: List[str]):
        # train the model
        self.train(msg_train, y_train)
        
        # use the trained model to generate predictions
        predictions = self.predict(msg_test)

        # evaluate the model by comparing the predictions with y_test
        utils.evaluate_prediction(predictions=predictions, y_test=y_test)

    def full_train(self, training_data_path: str):
        msg_train, msg_test, y_train, y_test = utils.prepare_data(file_path=training_data_path)

        self.evaluate(msg_train, msg_test, y_train, y_test)


def train():
    training_data_path = os.path.join(config.DATA_SUBDIR, 'spam.csv')
    model = Model()

    model.full_train(training_data_path=training_data_path)


def predict():
    # Load the previously trained model to predict new message
    model = Model()
    model.load_model()

    while True:
        message = input('Input a message: ')

        if not message:
            break

        # Note: predict() takes a list of message as input and a list (numpy array) as output
        predictions = model.predict([message])
        print(f'Spam or ham: {predictions[0]}')


def compare_models():
    training_data_path = os.path.join(config.DATA_SUBDIR, 'spam.csv')
    msg_train, msg_test, y_train, y_test = utils.prepare_data(file_path=training_data_path)

    for clf_type in ['NB', 'LR', 'DT']:
        model = Model(clf_type=clf_type)

        print(f'Training classifier type {clf_type}')

        model.evaluate(msg_train=msg_train, msg_test=msg_test, y_train=y_train, y_test=y_test)

        print('=' * 50)


if __name__ == '__main__':
    train()
    # predict()
    # compare_models()
