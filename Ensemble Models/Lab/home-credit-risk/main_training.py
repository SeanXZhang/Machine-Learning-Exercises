import warnings

from models.bagging_trees import *
from utils.utils import *

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data_train, data_test, y_train, y_test = prepare_data(os.path.join(DATA_SUBDIR, 'application_train.csv'),
                                                          sample_size=10000
                                                          )

    # TODO: Can you process train and test data separately by 1) fitting the NAN in features, 2) convert categoricals to one-hot-encoding 3)rescale numerical features
    data_train, data_test = process_train_test_data(data_train, data_test)

    # after some potentially expensive preprocessing, you can also save the processed data to files so next time you can
    # just load these files and skip those expensive preprocessing
    save_binary((data_train, data_test, y_train, y_test), os.path.join(RESULTS_DATA_SUB, 'data.h5'))

    print(f'Training Bagging classifier using decision trees')
    bagging_decision_trees(data_train, y_train, data_test, y_test)

    # print(f'Parameter tuning by random grid')
    # TODO: implement random grid search here, using Bagginer Classifer with Decision Trees as base
    # param_tuning(data_train, y_train, data_test, y_test)
