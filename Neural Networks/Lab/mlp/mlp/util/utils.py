import pickle  # for saving/loading binary files (serializing/deserializing)

import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


def prepare_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    return X_train, X_test, y_train, y_test


def evaluate_prediction(predictions: 'np.array', y_test: 'np.array'):
    print(classification_report(predictions, y_test))
    accuracy = accuracy_score(predictions, y_test)
    print(f'accuracy: {round(accuracy, 3)}')
    return accuracy


def error_analysis(images_test: 'np.array', predictions: 'np.array', y_test: 'np.array'):
    errors = []
    for (i, image_test) in enumerate(images_test):
        if predictions[i] != y_test[i]:
            errors.append((image_test, predictions[i], y_test[i]))

    print(f'Total errors: {len(errors)}')
    # Investigate the first 8 errors we made by showing the images with the label vs. prediction
    for (i, (image_test, pred, label)) in enumerate(errors[:8]):
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(image_test, cmap=plt.cm.gray_r, interpolation='gaussian')
        plt.title(f'{pred} (truth: {label})')

    plt.show()


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))


