import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


def prepare_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

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


def report_parameter_tuning(cv_results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(cv_results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                cv_results['mean_test_score'][candidate],
                cv_results['std_test_score'][candidate]))
            print("Parameters: {0}".format(cv_results['params'][candidate]))
            print("")
