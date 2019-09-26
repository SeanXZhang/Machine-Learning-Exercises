import numpy as np
from svm_tuning.util import logger

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

    logger.log_info(f'Shape before preprocessing: {images.shape}')
    n_samples = images.shape[0]
    data = images.reshape((n_samples, -1))

    logger.log_info(f'Shape after preprocessing: {data.shape}')

    return data


def process_data(images_train: 'np.array', images_test: 'np.array') -> (np.array, np.array):
    """
    Process the training and testing raw images into flattened matrices
    :param images_train: the raw images for training
    :param images_test: the raw images for testing
    :return: X_train, X_test
    """

    logger.log_info(f'Transforming training images...')
    images_train = preprocess(images=images_train)

    logger.log_info(f'\nTransforming test images...')
    images_test = preprocess(images=images_test)

    return images_train, images_test


