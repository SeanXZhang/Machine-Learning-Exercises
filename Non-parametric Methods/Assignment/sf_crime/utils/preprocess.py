import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from scipy.sparse import hstack
from utils import utils


def prepare_data(path):

    df = pd.read_csv(path, parse_dates=['Dates'])
    encoder = LabelEncoder()

    y = encoder.fit_transform(df['Category'])
    x = preprocess(df.drop('Category', axis=1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test, encoder


def preprocess(df):

    df['year'] = df['Dates'].map(lambda x: x.year).astype(str)
    df['month'] = df['Dates'].map(lambda x: x.month).astype(str)
    df['date'] = df['Dates'].map(lambda x: x.day).astype(str)
    df['hour'] = df['Dates'].map(lambda x: x.hour).astype(str)

    df['XY_outlier'] = ((df['X'] > -122.38) & (df['Y'] > 37.80)).astype(int)
    df['X'] = df.apply(lambda row: np.nan if row['XY_outlier'] == 1 else row['X'], axis=1)
    df['Y'] = df.apply(lambda row: np.nan if row['XY_outlier'] == 1 else row['Y'], axis=1)

    df_text = df[['Descript']]
    df_notext = df.drop(['Dates', 'Descript', 'Address'], axis=1)
    df = pd.merge(df_text, pd.get_dummies(df_notext, drop_first=True), left_index=True, right_index=True)

    return df


def feature_extraction(df, tfidf_vectorizer, dict_vectorizer, imputer, scaler, test_set=False):

    df_text = df['Descript'].map(utils.remove_stopwords)
    df_notext = df.drop('Descript', axis=1)

    if not test_set:
        s1 = tfidf_vectorizer.fit_transform(df_text)
        s2 = imputer.fit_transform(df_notext)
        s3 = scaler.fit_transform(s2)
        s4 = dict_vectorizer.fit_transform([dict(enumerate(row)) for row in s3])
        array = hstack([s1, s4])

        return array, tfidf_vectorizer, dict_vectorizer, imputer, scaler
    else:
        s1 = tfidf_vectorizer.transform(df_text)
        s2 = imputer.transform(df_notext)
        s3 = scaler.fit_transform(s2)
        s4 = dict_vectorizer.transform([dict(enumerate(row)) for row in s3])
        array = hstack([s1, s4])

        return array




