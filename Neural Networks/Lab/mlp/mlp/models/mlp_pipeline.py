from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.pipeline import Pipeline
from util import preprocess, utils

import warnings
warnings.filterwarnings("ignore")


def get_pipeline():
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 512),
        batch_size=100,
        max_iter=3,
        solver='adam',
        early_stopping=True
    )

    return Pipeline([
        ('clf', clf)
    ])


def full_train():
    # Load images data
    X_train, X_test, y_train, y_test = utils.prepare_data()
    X_train, X_test = preprocess.process_data(X_train, X_test)
    y_train, y_test = preprocess.process_label(y_train, y_test)

    pl = get_pipeline()

    pl.fit(X_train, y_train)

    predictions = pl.predict(X_test)
    utils.evaluate_prediction(predictions=predictions, y_test=y_test)


if __name__ == '__main__':
    full_train()
