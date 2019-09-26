import os.path
import time
import numpy as np
import numpy.random
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

import svm_tuning.config as config
from svm_tuning.util import preprocess, utils, logger

"""
References: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
"""

logger.log_info('>>>>>>>>>>>>>>>>Run svm_parameter_tuning.py')

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_grid_search_cv(clf):
    param_grid = {"C": [0.1, 1, 5, 10, 100],
                  "kernel": ['linear', 'rbf', 'poly'],
                  "max_iter": [100, 1000, -1]
                  }

    # run grid search
    return GridSearchCV(clf, param_grid=param_grid)


def get_randomized_search_cv(clf):
    # TODO: specify parameters and distributions to sample from and return RandomizedSearchCV
    pass


def parameter_tuning(grid_search: bool = True):
    # We need to define the parameter grid in which we will exhaustively search for the best combination
    # In order to do so, we need to understand what the available hyperparameters SVC() has.
    clf = SVC()
    images, labels = utils.load_image_data()
    images = preprocess.preprocess(images)

    start = time.time()
    if grid_search:
        search_cv = get_grid_search_cv(clf)
    else:
        # TODO: implement the `get_randomized_search_cv()` functions above for this part
        pass

    search_cv.fit(images, labels)

    logger.log_info("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                    % (time.time() - start, len(search_cv.cv_results_['params'])))
    report(search_cv.cv_results_)

    # By default, sklearn will refit the model with the best parameters on the entire dataset
    # This refitted model is accessible by calling `grid_search.best_estimator_`
    best_clf = search_cv.best_estimator_
    best_score = search_cv.best_score_

    logger.log_info('Best clf (%.3f validation score):' % best_score)
    logger.log_info(best_clf)

    # let's save this best classifier for later use
    model_path = os.path.join(config.MODELS_SUBDIR, f'svm_grid_search={grid_search}.clf')
    logger.log_info(f'Saving fitted model to {model_path}')
    utils.save_binary(best_clf, model_path)


def predict_images(grid_search: bool = True):
    images, labels = utils.load_image_data()
    images_train, images_test, y_train, y_test = utils.prepare_data(X=images, y=labels)

    _, X_test = preprocess.process_data(images_train, images_test)

    # assume the model is already trained and saved using the parameter_tuning() step above
    model_path = os.path.join(config.MODELS_SUBDIR, f'svm_grid_search={grid_search}.clf')
    clf = utils.load_binary(model_path)

    # tp see how our best model generalizes to unseen data
    # let's corrupt the test images by some noise such that they are not exactly the same as the ones
    # the model has seen already
    noise_intensity = 2
    noise = numpy.random.normal(0, 1, X_test.shape) * noise_intensity
    X_test = np.clip(X_test + noise, 0, 255)
    predictions = clf.predict(X_test)

    utils.evaluate_prediction(predictions, y_test)


if __name__ == '__main__':
    grid_search = False

    parameter_tuning(grid_search=grid_search)
    predict_images(grid_search=grid_search)
    logger.log_info('>>>>>>>>>>>>>>>>End of svm_parameter_tuning.py')
