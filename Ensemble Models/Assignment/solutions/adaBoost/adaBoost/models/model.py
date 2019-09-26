import os.path
import time

from scipy.stats import randint as sp_randint
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier

import adaBoost.config as config
from adaBoost.config import MODELS_SUBDIR
from adaBoost.utils.utils import save_binary, report, prepare_data, evaluate_prediction


class Model:
    def __init__(self, ensemble: bool = True):
        self.vectorizer = DictVectorizer()
        self.imputer = Imputer()
        self.scaler = MaxAbsScaler()

        self.ensemble = ensemble
        self.clf = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(class_weight='balanced'))if ensemble else DecisionTreeClassifier(
            class_weight='balanced')

        # these can be simplified by using Pipeline
        model_version = 'ensemble' if ensemble else 'single'
        self.model_path = os.path.join(MODELS_SUBDIR, f'{model_version}.clf')
        self.vectorizer_path = os.path.join(MODELS_SUBDIR, f'vectorizer_{model_version}.clf')
        self.imputer_path = os.path.join(MODELS_SUBDIR, f'imputer_{model_version}.clf')
        self.scaler_path = os.path.join(MODELS_SUBDIR, f'scaler_{model_version}.clf')

    def process_train_data(self, data_train):
        X_train = self.vectorizer.fit_transform(data_train)

        # fill in nan values
        X_train = self.imputer.fit_transform(X_train)

        # scaling data by columns so different features have roughly the same magnitude
        X_train = self.scaler.fit_transform(X_train)

        print(X_train.shape)

        return X_train

    def process_test_data(self, data_test):
        X_test = self.vectorizer.transform(data_test)

        # fill in nan values
        X_test = self.imputer.transform(X_test)

        # scaling data by columns so different features have roughly the same magnitude
        X_test = self.scaler.transform(X_test)

        return X_test

    def param_tuning(self, data_train, y_train):
        X_train = self.process_train_data(data_train)

        if self.ensemble:
            search_params = {
                'n_estimators': sp_randint(10, 100),
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 1]
            }

            n_iter_search = 20
            cv = RandomizedSearchCV(self.clf, param_distributions=search_params, n_iter=n_iter_search,
                                    scoring={'score': 'f1'}, n_jobs=-1,
                                    refit='score')  # accuracy is not a good metric for imbalanced dataset

            start = time.time()
            cv.fit(X_train, y_train)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time.time() - start), n_iter_search))
            print(cv.best_params_)
            # report(cv.cv_results_)

            self.clf = cv.best_estimator_
        else:
            self.clf.fit(X_train, y_train)

        # by the end of randomized search, the model will be refitted on the entire data set
        print('saving model to file...')
        save_binary(self.clf, self.model_path)

        print('saving vectorizer to file...')
        save_binary(self.vectorizer, self.vectorizer_path)

        print('saving imputer to file...')
        save_binary(self.imputer, self.imputer_path)

        print('saving scaler to file...')
        save_binary(self.scaler, self.scaler_path)

        X_test = self.process_test_data(data_test)
        preds = self.clf.predict(X_test)
        evaluate_prediction(preds, y_test)


if __name__ == '__main__':
    data_train, data_test, y_train, y_test = prepare_data(os.path.join(config.DATA_SUBDIR, 'application_train.csv'),
                                                          sample_size=10000
                                                          )

    # after some potentially expensive preprocessing, you can also save the processed data to files so next time you can
    # just load these files and skip those expensive preprocessing
    # save_binary((data_train, data_test, y_train, y_test), os.path.join(MODELS_SUBDIR, 'data.dat'))

    print(f'Training AdaBoost classifier')
    model = Model(ensemble=True)
    model.param_tuning(data_train, y_train)

    print(f'\nTraining decision tree classifier')
    model = Model(ensemble=False)
    model.param_tuning(data_train, y_train)
