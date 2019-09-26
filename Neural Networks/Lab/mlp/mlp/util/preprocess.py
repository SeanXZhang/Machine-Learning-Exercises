import keras.utils
import numpy as np


def preprocess(images: 'np.array') -> 'np.array':
    """
    Input images is a (n_samples, 8, 8) matrix.
    To apply a classifier on this data, we need to flatten the image, i.e.,
    turn the data in a (samples, n_dim) matrix, where n_dim = 8*8

    :param images: a 3D matrix with shape (n_samples, 8, 8)
    :return: a flattened image matrix of shape (n_samples, 64)
    """

    # raw images, as pixels, are already in matrix format.
    # So a simple reshaping operation (to reshape images from 3D matrix to 2D)
    # is sufficient for this simple dataset.

    print(f'Shape before preprocessing: {images.shape}')
    n_samples = images.shape[0]
    data = images.reshape((n_samples, -1))
    data = data.astype('float32')
    #
    # # convert the pixel intensity ranging from 0 and 255 to between 0 and 1
    data /= 255

    print(f'Shape after preprocessing: {data.shape}')

    return data


def process_label(labels_train: np.array, labels_test: np.array) -> (np.array, np.array):
    # convert class vectors to binary class matrices (one-hot encoding for labels)
    y_train = keras.utils.to_categorical(labels_train, 10)
    y_test = keras.utils.to_categorical(labels_test, 10)

    return y_train, y_test


def process_data(images_train: 'np.array', images_test: 'np.array') -> (np.array, np.array):
    """
    Process the training and testing raw images into flattened matrices
    :param images_train: the raw images for training
    :param images_test: the raw images for testing
    :return: X_train, X_test
    """

    print(f'Transforming training images...')
    images_train = preprocess(images=images_train)

    print(f'\nTransforming test images...')
    images_test = preprocess(images=images_test)

    return images_train, images_test
