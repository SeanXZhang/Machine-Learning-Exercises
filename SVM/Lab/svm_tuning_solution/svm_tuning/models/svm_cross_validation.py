import time
import numpy as np
from sklearn.svm import SVC

from svm_tuning.util import preprocess, utils, logger

"""
From `svm.py`, we take one step further to evaluate our SVM classifier on 5-fold cross validation
"""


def cross_validation():
    # Load images data
    images, labels = utils.load_image_data()

    # Separate
    fold_data = utils.prepare_cross_validation_data(X=images, y=labels)

    logger.log_info('>>>>>>>>>>>>>>>>Run svm_cross_validation.py')
    accuracies = []

    for (i, (images_train, images_test, y_train, y_test)) in enumerate(fold_data):
        logger.log_info(f'Training and evaluating fold {i+1}...')

        X_train, X_test = preprocess.process_data(images_train, images_test)

        # Fit a linear SVM classifier with all hyperparameters set to their default values
        prev_time = time.time()

        logger.log_info("Start fitting SVM classifier...")
        clf = SVC(kernel='linear')

        clf.fit(X_train, y_train)
        logger.log_info(f'Finished training in {round(time.time() - prev_time, 2)} seconds')

        predictions = clf.predict(X_test)

        acc = utils.evaluate_prediction(predictions=predictions, y_test=y_test)
        accuracies.append(acc)

    logger.log_info(f'mean accuracy: %.3f, min: %.3f, max: %.3f, std: %.3f' % (np.mean(accuracies),
                                                                               np.min(accuracies),
                                                                               np.max(accuracies),
                                                                               np.std(accuracies)))

    logger.log_info('>>>>>>>>>>>>>>>>End of svm_cross_validation.py')


if __name__ == '__main__':
    cross_validation()
