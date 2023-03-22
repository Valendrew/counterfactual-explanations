import math
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import gdown


def read_csv(filename: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(filename, **kwargs)
    assert isinstance(df, pd.DataFrame)
    return df


def download_helper(id: str, name: str, quiet: bool=False):
    gdown.download(id=id, output=name, quiet=quiet)
        

def split_train_test(data, test_size: float, rng):
    if isinstance(data, pd.DataFrame):
        return train_test_split(data, test_size=test_size, random_state=rng)
    elif isinstance(data, list):
        return train_test_split(*data, test_size=test_size, random_state=rng)


def count_frequency_labels(series: pd.Series) -> pd.DataFrame:
    count_series = series.value_counts()
    count_sum = count_series.sum()
    frequency_series = count_series.apply(lambda x: f"{(x / count_sum * 100):.2f}%")

    df = pd.DataFrame({"Frequency": frequency_series, "Count": count_series})
    return df


def encode_normalize_df(df, std_scaler=None, label_bin=None):
    X_raw, y_raw = df.iloc[:, 1:], df.iloc[:, 0]

    # standardization with mean 0 and unit variance
    if std_scaler is None:
        std_scaler = StandardScaler()
        X = std_scaler.fit_transform(X_raw)
    else:
        X = std_scaler.transform(X_raw)

    # encoding of the label
    if label_bin is None:
        label_bin = LabelBinarizer(neg_label=0, pos_label=1)
        y = label_bin.fit_transform(y_raw)
    else:
        y = label_bin.transform(y_raw)

    return X, y, std_scaler, label_bin
