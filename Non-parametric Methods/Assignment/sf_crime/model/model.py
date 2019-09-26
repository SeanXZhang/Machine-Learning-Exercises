import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import pickle

from utils import preprocess
import warnings
warnings.filterwarnings('ignore')


def transformation(x_train, x_test):

    imputer = Imputer(strategy='median')
    tfidf_vectorizer = TfidfVectorizer()
    dict_vectorizer = DictVectorizer()
    scaler = StandardScaler()

    x_train, tfidf_vectorizer, \
    dict_vectorizer, imputer, scaler = preprocess.feature_extraction(x_train, tfidf_vectorizer, dict_vectorizer,
                                                                     imputer, scaler, test_set=False)
    x_test = preprocess.feature_extraction(x_test, tfidf_vectorizer, dict_vectorizer,
                                           imputer, scaler, test_set=True)

    return x_train, x_test


def cross_validation(x, y, clf, cv=3):

    train_score_lst = []
    test_score_lst = []
    kfcv = KFold(n_splits=cv)
    for train_index, test_index in kfcv.split(x, y):

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train, x_test = transformation(x_train, x_test)
        train_score, test_score = modeling(x_train, x_test, y_train, y_test, clf)

        train_score_lst.append(train_score)
        test_score_lst.append(test_score)

    return np.array(train_score_lst), np.array(test_score_lst)


def modeling(x_train, x_test, y_train, y_test, clf):

    clf.fit(x_train, y_train)

    y_pred_train = clf.predict_proba(x_train)
    y_pred = clf.predict_proba(x_test)

    train_score = log_loss(y_train, y_pred_train, labels=y_train)
    test_score = log_loss(y_test, y_pred, labels=y_train)

    return train_score, test_score


def full_train(x_train, x_test, y_train, y_test, clf, save_model=True):

    clf.fit(x_train, y_train)
    y_pred_train = clf.predict_proba(x_train)
    y_pred = clf.predict_proba(x_test)

    train_score = log_loss(y_train, y_pred_train, labels=y_train)
    test_score = log_loss(y_test, y_pred, labels=y_train)

    if save_model:
        dir = '../classifier/sf_crime_model'
        pickle.dump(clf, open(dir, 'wb'))
        print(f'Model saved to: {dir}')

    return train_score, test_score, clf


def predict_one(x, y, clf):

    random_index = np.random.choice(x.shape[0], 1)
    sample = x.tocsr()[random_index]
    label = y[random_index]
    label_inverse = encoder.inverse_transform(label)

    pred = clf.predict(sample)
    pred_inverse = encoder.inverse_transform(pred)

    print(f'Predicting from index: {random_index}:')
    print(f'y_true: {label}, inverse: {label_inverse}')
    print(f'y_pred: {pred}, inverse: {pred_inverse}')


if __name__ == '__main__':

    x_train, x_test, y_train, y_test, encoder = preprocess.prepare_data('../data/train_2015.csv')
    train_score_vec, test_score_vec = cross_validation(x_train, y_train, cv=5, clf=RandomForestClassifier())

    x_train, x_test = transformation(x_train, x_test)
    train_score, test_score, clf = full_train(x_train, x_test, y_train, y_test,
                                              clf=RandomForestClassifier(), save_model=True)

    print('- ' * 30)
    print(f'Logloss - Train: \n\tmean: {train_score_vec.mean()}\n\t{train_score_vec}')
    print('- ' * 30)
    print(f'Logloss - Validation: \n\tmean: {test_score_vec.mean()}\n\t{test_score_vec}')
    print('- ' * 30)
    print(f'Logloss - Full Train: {train_score}')
    print(f'Logloss - Full Test: {test_score}')
    print('- ' * 30)
    print('Use "predict_one" method to test on one random sample.')


