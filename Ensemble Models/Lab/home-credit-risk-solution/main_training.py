import warnings

from models.bagging_trees import *
from utils.utils import *

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data_train, data_test, y_train, y_test = prepare_data(os.path.join(DATA_SUBDIR, 'application_train.csv'),
                                                          sample_size=10000
                                                          )

    # after some potentially expensive preprocessing, you can also save the processed data to files so next time you can
    # just load these files and skip those expensive preprocessing
    data_train, data_test = process_train_test_data(data_train, data_test)

    save_binary((data_train, data_test, y_train, y_test), os.path.join(RESULTS_DATA_SUB, 'data.h5'))

    # print(f'Training Bagging classifier using decision trees')
    # bagging_decision_trees(data_train, y_train, data_test, y_test)

    print(f'Parameter tuning by random grid')
    param_tuning(data_train, y_train, data_test, y_test)
