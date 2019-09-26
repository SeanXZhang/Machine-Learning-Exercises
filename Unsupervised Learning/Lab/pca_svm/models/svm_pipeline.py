import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from util import preprocess, utils

"""
In this script, we take advantage of sklearn's Pipeline class to perform train/test transformation"""


# TODO: complete this function
def get_pipeline(perform_pca=False):
    """
    Build a data pipeline with transformers (e.g. StandScaler) and estimators (e.g. SVM).
    PCA will be an option enabled by the variable: perform_pca

    :return: sklearn pipeline
    """
    scaler = StandardScaler()
    pca = PCA(n_components=20)
    clf = SVC(kernel='linear')

    pass


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
