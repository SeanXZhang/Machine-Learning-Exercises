import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def load_image_data():
    digits = datasets.load_digits()
    print(f'Loaded {len(digits.images)} images and {len(digits.target)} labels.')

    return digits.images, digits.target


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def evaluate_prediction(predictions: 'np.array', y_test: 'np.array'):
    print(classification_report(predictions, y_test))
    accuracy = accuracy_score(predictions, y_test)
    print(f'accuracy: {round(accuracy, 3)}')
    return accuracy