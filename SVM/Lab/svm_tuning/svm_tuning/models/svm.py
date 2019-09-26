import time
from sklearn.svm import SVC
from svm_tuning.util import preprocess, utils, logger


def full_train():
    # Load images data
    logger.log_info('>>>>>>>>>>>>>>>>Run svm.py')
    images, labels = utils.load_image_data()

    # Separate
    images_train, images_test, y_train, y_test = utils.prepare_data(X=images, y=labels)

    # Preprocess training and test messages
    prev_time = time.time()
    logger.log_info("Start transforming data...")

    X_train, X_test = preprocess.process_data(images_train, images_test)
    logger.log_info(f'Finished transforming data in {round(time.time() - prev_time, 2)} seconds')

    # Fit a linear SVM classifier with all hyperparameters set to their default values
    prev_time = time.time()
    logger.log_info("Start fitting SVM classifier...")
    clf = SVC(kernel='linear')

    clf.fit(X_train, y_train)
    logger.log_info(f'Finished training in {round(time.time() - prev_time, 2)} seconds')
    predictions = clf.predict(X_test)

    utils.evaluate_prediction(predictions=predictions, y_test=y_test)

    utils.error_analysis(images_test, predictions, y_test)
    logger.log_info('>>>>>>>>>>>>>>>>End of svm.py')


if __name__ == '__main__':
    full_train()
