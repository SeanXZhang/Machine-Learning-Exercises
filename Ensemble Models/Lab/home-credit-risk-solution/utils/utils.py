import pickle  # for saving/loading binary files (serializing/deserializing)
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer


def prepare_data(file_path, sample_size, target_col='TARGET'):

    start = time.time()
    print('Reading training data...')
    df = pd.read_csv(file_path)
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))

    # the original dataset is quite large. You can randomly sample a good amount of rows from it for this task.
    if sample_size is not None:
        df = df.sample(sample_size)

    y = df[target_col].values
    X = df.drop([target_col], axis=1)
    data = [dict(row) for _, row in X.iterrows()]

    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y)

    return data_train, data_test, y_train, y_test


def process_train_test_data(data_train, data_test, vectorizer=DictVectorizer(), imputer=Imputer(), scaler=MaxAbsScaler()):

    data_train = vectorizer.fit_transform(data_train)
    data_test = vectorizer.transform(data_test)

    # fill in nan values
    data_train = imputer.fit_transform(data_train)
    data_test = imputer.transform(data_test)

    # scaling data by columns so different features have roughly the same magnitude
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)

    print(f'Shape of X_train is: {data_train.shape}')
    print(f'Shape of X_test is: {data_test.shape}')

    return data_train, data_test


def evaluate_prediction(predictions, y_test):
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f'accuracy: {round(accuracy, 3)}')


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        # print(results)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
