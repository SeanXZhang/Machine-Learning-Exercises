import pickle  # for saving/loading binary files (serializing/deserializing)
import time
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def prepare_data(file_path: str, sample_size: int=None) -> (List[str], List[str], List[str], List[str]):
    target_col = 'TARGET'

    start = time.time()
    print('Reading training data...')
    df = pd.read_csv(file_path)
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))
    if sample_size is not None:
        df = df.sample(sample_size)

    y = []  # the labels
    data = []  # the features
    features = list([x for x in df.columns if x != target_col])

    for row in tqdm(df.to_dict('records')):
        y.append(row[target_col])
        data.append({k: row[k] for k in features})

    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y)

    return data_train, data_test, y_train, y_test


def evaluate_prediction(predictions: 'np.array', y_test: 'np.array'):
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
