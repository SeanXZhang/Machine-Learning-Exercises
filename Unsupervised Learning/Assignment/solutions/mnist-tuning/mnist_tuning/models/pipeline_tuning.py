import time

from sklearn.svm import SVC
import numpy as np
from mnist_tuning.utils import preprocess, utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def load_data():
    # Load images data
    images, labels = utils.load_image_data()

    # Separate data into train and validation
    images_train, images_test, y_train, y_test = utils.prepare_data(X=images, y=labels)

    # Preprocess training and test images
    X_train, X_test = preprocess.process_data(images_train, images_test)
    return X_train, X_test, y_train, y_test


def get_pipeline():
    scaler = StandardScaler()
    pca = PCA()
    clf = SVC(kernel='linear')
    pipelines = [
        ('scaler', scaler),
        ('pca', pca),
        ('clf', clf)
    ]

    return Pipeline(pipelines)


def get_param_grid():
    param_grid = {
        'pca__n_components': [5, 20, 30, 40, 50, 60],
        'clf__C': [0.01, 0.1, 1, 2, 5, 10],
    }
    return param_grid


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


def full_train():
    prev_time = time.time()
    print("Start loading and preprocessing data...")

    X_train, X_test, y_train, y_test = load_data()
    print(f'\nFinished transforming data in {round(time.time() - prev_time, 2)} seconds')

    pipeline = get_pipeline()

    prev_time = time.time()
    print("\nStart parameter tuning...")
    param_grid = get_param_grid()
    search = GridSearchCV(pipeline, param_grid, iid=False, cv=5,
                          return_train_score=False, refit=True)

    # this is needed because otherwise sklearn will complain the scaler is not fitted after parameter tuning, even with `refit=True`
    pipeline.fit(X_train, y_train)
    search.fit(X_train, y_train)
    report(search.cv_results_)

    print(f'\nFinished tuning in {round(time.time() - prev_time, 2)} seconds')

    predictions = pipeline.predict(X_test)

    utils.evaluate_prediction(predictions=predictions, y_test=y_test)


# the entry point of this script if run standalone
if __name__ == '__main__':
    full_train()
