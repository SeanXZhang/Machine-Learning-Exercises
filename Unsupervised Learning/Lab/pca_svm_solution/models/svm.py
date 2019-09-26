import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from util import preprocess, utils

"""
From last `svm.py`, we take one step further to evaluate our SVM classifier on 5-fold cross validation
"""


def cross_validation(perform_pca: bool = False):
    # Load images data
    images, labels = utils.load_image_data()

    # cross validation
    fold_data = utils.prepare_cross_validation_data(X=images, y=labels)

    accuracies = []
    for (i, (images_train, images_test, y_train, y_test)) in enumerate(fold_data):
        print(f'Training and evaluating fold {i + 1}...')

        X_train, X_test = preprocess.process_data(images_train, images_test)

        if perform_pca:
            pca = PCA(n_components=50)
            scaler = StandardScaler()

            prev_time = time.time()
            print(f'Start performing PCA transformation on train and test data...')
            X_train = preprocess.transform_data(X_train, scaler=scaler, pca=pca, for_training=True)
            print(
                f'Finished performing PCA transformation in {round(time.time() - prev_time, 2)} seconds')

            X_test = preprocess.transform_data(X_test, scaler, pca, for_training=False)

        print("Start fitting SVM classifier...")

        # Fit a linear SVM classifier with all hyperparameters set to their default values
        prev_time = time.time()

        print("Start fitting SVM classifier...")
        clf = SVC(kernel='linear')

        clf.fit(X_train, y_train)
        print(f'Finished training in {round(time.time() - prev_time, 2)} seconds')

        predictions = clf.predict(X_test)

        acc = utils.evaluate_prediction(predictions=predictions, y_test=y_test)
        accuracies.append(acc)

        print('=' * 50)

    print(f'mean accuracy: %.3f, min: %.3f, max: %.3f, std: %.3f' % (np.mean(accuracies),
                                                                     np.min(accuracies),
                                                                     np.max(accuracies),
                                                                     np.std(accuracies)))


if __name__ == '__main__':
    cross_validation(perform_pca=True)
