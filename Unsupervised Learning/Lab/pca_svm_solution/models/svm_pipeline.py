import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from util import preprocess, utils

"""
In this script, we take advantage of sklearn's Pipeline class to perform train/test transformation"""


def get_pipeline(perform_pca=False):
    scaler = StandardScaler()
    pca = PCA(n_components=20)
    clf = SVC(kernel='linear')
    pipelines = []

    if perform_pca:
        # If PCA is enabled, we need to perform standardization first
        pipelines.extend([
            ('scaler', scaler),
            ('pca', pca)])

    pipelines.append(('clf', clf))
    # print('steps:')
    # for step in pipelines:
    #     print(step)

    pl = Pipeline(pipelines)

    # You can set the parameters using the names issued
    # For instance, fit using n_components=30 for PCA
    pl.set_params(pca__n_components=30)
    # equivalent to constructing PCA with n_components=30
    return pl


def cross_validation(perform_pca=False):
    # Load images data
    images, labels = utils.load_image_data()

    # cross validation
    fold_data = utils.prepare_cross_validation_data(X=images, y=labels)

    accuracies = []
    for (i, (images_train, images_test, y_train, y_test)) in enumerate(fold_data):
        print(f'Training and evaluating fold {i + 1}...')

        X_train, X_test = preprocess.process_data(images_train, images_test)

        # call the data pipeline here
        pl = get_pipeline(perform_pca=perform_pca)

        pl.fit(X_train, y_train)

        predictions = pl.predict(X_test)
        acc = utils.evaluate_prediction(predictions=predictions, y_test=y_test)
        accuracies.append(acc)

        print('=' * 50)

    print(f'mean accuracy: %.3f, min: %.3f, max: %.3f, std: %.3f' % (np.mean(accuracies),
                                                                     np.min(accuracies),
                                                                     np.max(accuracies),
                                                                     np.std(accuracies)))


if __name__ == '__main__':
    cross_validation(perform_pca=True)
