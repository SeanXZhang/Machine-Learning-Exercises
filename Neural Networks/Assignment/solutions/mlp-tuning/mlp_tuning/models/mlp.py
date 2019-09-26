from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

from mlp_tuning.utils import preprocess, utils

"""
  Keras has a wrapper to implement sklearn's classifier/regressor APIs, i.e., .fit(), .predict() and so on.
  
  More info: https://keras.io/scikit-learn-api/
  Example: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
  Example tutorial: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""


def get_model(input_shape, num_classes: int, num_units: int, num_layers: int, activation: str):
    model = Sequential()
    model.add(Dense(num_units, activation=activation, input_shape=(input_shape,)))
    model.add(Dropout(0.2))
    for _ in range(num_layers):
        model.add(Dense(num_units, activation=activation))
        model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model


def sample_training_data(X_train, y_train, sample_size: int = None):
    """
    To quickly test your code, you can use a small subset of training data to get your pipeline working.
    Once your code is working, you can switch back to using the entire training set to get the best performance.
    :param X_train:
    :param y_train:
    :param sample_size:
    :return:
    """
    if sample_size is not None:
        print(f'Samping {sample_size} data from the training set...')
        sample_size = min(sample_size, X_train.shape[0])
        return X_train[:sample_size], y_train[:sample_size]

    return X_train, y_train


def cross_validation(training_sample_size: int = None):
    # Load images data
    X_train, X_test, y_train, y_test = utils.prepare_data()

    X_train, y_train = sample_training_data(X_train, y_train, training_sample_size)

    X_train, X_test = preprocess.process_data(X_train, X_test)

    model = KerasClassifier(build_fn=get_model, input_shape=X_train.shape[1], num_classes=10,
                            verbose=0)

    param_dist = {
        'num_units': [10, 20, 30, 40],
        'num_layers': [1, 2],
        'epochs': [50, 100, 150],
        'activation': ['relu', 'sigmoid']
    }

    # to quickly test if your code is working, set `n_iter` to be a small number, e.g., 1
    additional_training_params = {
        'callbacks': [EarlyStopping(monitor='val_loss', patience=1)],
        'batch_size': 32,
        'validation_split': 0.2
    }
    search_cv = RandomizedSearchCV(model, param_dist, n_iter=5, refit=True)
    search_cv.fit(X_train, y_train, **additional_training_params
                  )
    # search_cv.best_estimator_
    utils.report_parameter_tuning(search_cv.cv_results_, n_top=5)

    model = search_cv.best_estimator_
    predictions = model.predict(X_test)

    utils.evaluate_prediction(predictions=predictions, y_test=y_test)


if __name__ == '__main__':
    """
    To quickly test your code, you can use a small subset of training data to get your pipeline working.
    Once your code is working, you can switch back to using the entire training set (set `sample_size=None`)
    to get the best performance.
    
    Note: when using the entire dataset for hyper-parameter tuning, it can take a very long time.
    """
    sample_size = 100
    cross_validation(training_sample_size=sample_size)
