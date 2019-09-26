import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List
import pickle


def prepare_data(file_path) -> (List[str], List[str], List[str], List[str]):
    df = pd.read_csv(file_path)

    y = df['label'].values  # the labels
    X = df['message'].values  # the messages

    # TODO: split the messages and labels into 80% training and 20% tests
    msg_train, msg_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return msg_train, msg_test, y_train, y_test


def evaluate_prediction(predictions: 'np.array', y_test: 'np.array'):
    print(classification_report(predictions, y_test))
    accuracy = accuracy_score(predictions, y_test)
    print(f'accuracy: {round(accuracy, 3)}')


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))
