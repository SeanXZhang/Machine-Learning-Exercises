import pandas as pd
import numpy as np
import math
import operator
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
from sklearn.preprocessing import StandardScaler


def load_dataset(data_path):
    df = pd.read_csv(data_path)
    return df


def split_data(df):
    x = df.drop(['Name'], axis=1)
    y = df['Name']
    feature_names = x.columns
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test


def scaling(x, train=True):
    scaler = StandardScaler()
    if train:
        x_rescaled = scaler.fit_transform(x)
    else:
        x_rescaled = scaler.transform(x)
    return x_rescaled





