import pickle  # for saving/loading binary files (serializing/deserializing)

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from svm_tuning.util import logger


def load_image_data():
    digits = datasets.load_digits()
    logger.log_info(f'Loaded {len(digits.images)} images and {len(digits.target)} labels.')

    return digits.images, digits.target


def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test


def prepare_cross_validation_data(X, y, n_folds: int = 5):

    # We will perform train/val split n_folds times using sklearn's StratifiedKFold
    # StratifiedKFold will make sure the class distribution in train/val will be roughly the same
    # across all folds
    kfold = StratifiedKFold(n_splits=n_folds)

    fold_data = []
    for train_idx, test_idx in kfold.split(X, y):
        # train_idx is a list of index referring to the data to be used in training
        # test_idx is a list of index referring to the data to be used in testing/validation

        X_train = X[train_idx]  # you can do this because X is a numpy matrix
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_data.append((X_train, X_test, y_train, y_test))

    return fold_data


def evaluate_prediction(predictions, y_test):
    accuracy = accuracy_score(predictions, y_test)
    logger.log_info(f'Accuracy: {round(accuracy, 3)}')

    return accuracy


def error_analysis(images_test, predictions, y_test):
    errors = []
    for (i, image_test) in enumerate(images_test):
        if predictions[i] != y_test[i]:
            errors.append((image_test, predictions[i], y_test[i]))

    logger.log_info(f'Total errors: {len(errors)}')
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
