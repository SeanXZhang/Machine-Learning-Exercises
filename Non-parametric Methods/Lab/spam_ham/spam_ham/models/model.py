import os.path
import time
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import spam_ham.config as config
from spam_ham.util import preprocess, utils


"""
For a small piece of code, it is ok to just write a few functions and pass along options
by function parameters.
"""


"""
Global variables: hyper-parameters controlling your feature extraction
"""
BIAS = 1.0
REMOVE_STOPWORDS = True
MIN_DF = 2


def feature_extraction(msg_train: List[str], msg_test: List[str]) -> (np.array, np.array):
    """
    Process the training and testing raw messages into matrices
    :param msg_train: the list of messages for training
    :param msg_test: the list of messages for testing
    :return: X_train, X_test
    """
    # for each email, we apply preprocess() to performance preprocessing.
    words_train = [preprocess.preprocess(message, remove_stopwords=REMOVE_STOPWORDS) for
                   message in msg_train]

    # use sklearn's tf-idf vectorizer to vectorizer `words_train`
    # tf-idf vectorizer needs to work with list of strings,
    # so we need to combine the words into a string
    text_train = [' '.join(words) for words in words_train]

    # fit a TF-IDF vectorizer on `text_train`
    vectorizer = TfidfVectorizer(min_df=MIN_DF)
    X_train = vectorizer.fit_transform(text_train)

    # TODO: use the same preprocessing on the test email message
    words_test = [preprocess.preprocess(message, remove_stopwords=REMOVE_STOPWORDS) for
                   message in msg_test]
    text_test = [' '.join(words) for words in words_test]
    # TODO: use the fitted vectorizer on the test words to get X_test
    X_test = vectorizer.transform(text_test)

    # save the vectorizer to disk
    utils.save_binary(vectorizer, os.path.join(config.MODELS_SUBDIR, 'vectorizer'))

    return X_train, X_test


def full_train(training_data_path: str):
    msg_train, msg_test, y_train, y_test = utils.prepare_data(file_path=training_data_path)

    # Preprocess training and test messages
    prev_time = time.time()
    print("Start transforming data...")
    X_train, X_test = feature_extraction(msg_train, msg_test)
    print(f'Finished transforming data in {round(time.time() - prev_time, 2)} seconds')

    # Fit Naive Bayes classifier
    prev_time = time.time()
    print("Start fitting Naive Bayes classifier...")

    # TODO: construct a MultinomialNB classifier.
    #  Look at sklearn's documentation to see what are the possible parameters for constructing this classifier
    #  Fit your classifier on the training data
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(f'Finished training in {round(time.time() - prev_time, 2)} seconds')

    # save the trained model to disk
    utils.save_binary(nb, os.path.join(config.MODELS_SUBDIR, 'nb'))

    # TODO: Generate predictions on the test data
    predictions = nb.predict(X_test)
    # evaluate your predictions against y_test
    utils.evaluate_prediction(predictions=predictions, y_test=y_test)


def predict_message(message: str, vectorizer: TfidfVectorizer, nb: MultinomialNB) -> str:
    """
    This function generates the prediction for a new message.
    You will need to perform the preprocessing and feature extraction on the message,
    and use the model to generate the prediction
    :param message:
    :param vectorizer: the TfIdfVectorizer fitted on the training data
    ":param nb: the trained MultinomialNB model
    :return: spam/ham prediction
    """

    # TODO: Step 1, use preprocess() function to perform preprocessing on the input message.
    message_test = preprocess(message=message, remove_stopwords=True)

    # Step 2, use vectorizer to transform the message.
    # note that the message here is a single message, and the vectorizer.transform() function
    # is expecting a list of strings, because it is supposed to handle a batch of messages.
    # Therefore, the input to the transform() function is a list with a single element "message"
    X_test = vectorizer.transform([message_test])

    # TODO: Step 3: use nb to generate prediction on X_test
    preds = nb.predict(X_test)

    # The output of our model would also be a list, therefore we need to return the first element
    return preds[0]


def prediction():
    # Load the previously saved vectorizer and model
    vectorizer = utils.load_binary(os.path.join(config.MODELS_SUBDIR, 'vectorizer'))
    nb = utils.load_binary(os.path.join(config.MODELS_SUBDIR, 'nb'))

    while True:
        message = input('Input a message: ')

        if not message:
            break

        # Note: predict() takes a list of message as input and a list (numpy array) as output
        predictions = predict_message(message, vectorizer, nb)
        print(f'Spam or ham: {predictions[0]}')


if __name__ == '__main__':
    full_train(training_data_path=os.path.join(config.DATA_SUBDIR, 'spam.csv'))
