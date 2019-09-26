import warnings

from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils import plot_model

from util import utils

warnings.filterwarnings("ignore")

"""
Setup Keras with Tensorflow backend:
Install Tensorflow: https://www.tensorflow.org/install/

"""


def get_model(num_classes):
    # Initialize a linear stack of layers: a model without any branching.
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def mlp():
    # Load images data
    X_train, X_test, y_train, y_test = utils.prepare_data()

    model = get_model(num_classes=10)

    # Get model architecture summary
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.fit(X_train, y_train, batch_size=100, epochs=3)

    model.evaluate(X_test, y_test)


if __name__ == '__main__':
    mlp()
