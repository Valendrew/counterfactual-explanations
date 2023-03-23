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


<<<<<<< HEAD
def plot_dataframe(data, labels=None, vmin=-9, vmax=0.15, figsize=None, s=4):
    plt.figure(figsize=figsize)
    plt.imshow(data.T.iloc[:, :], aspect="auto", cmap="RdBu", vmin=vmin, vmax=vmax)
    plt.yticks(range(0, data.shape[1]))
    if labels is not None:
        # nonzero = data.index[labels != 0]
        ncol = len(data.columns)
        lvl = -0.05 * ncol
        # plt.scatter(nonzero, lvl*np.ones(len(nonzero)),
        #         s=s, color='tab:orange')
        plt.scatter(
            labels.index,
            np.ones(len(labels)) * lvl,
            s=s,
            color=plt.get_cmap("tab10")(np.mod(labels, 10)),
        )
    plt.tight_layout()


def encode_normalize_df(df, std_scaler=None, label_bin=None, neg_label=0, pos_label=1):
=======
def encode_normalize_df(df, std_scaler=None, label_bin=None):
>>>>>>> 409be0da238cc32f2f56698562f57b83b9c799d2
    X_raw, y_raw = df.iloc[:, 1:], df.iloc[:, 0]

    # standardization with mean 0 and unit variance
    if std_scaler is None:
        std_scaler = StandardScaler()
        X = std_scaler.fit_transform(X_raw)
    else:
        X = std_scaler.transform(X_raw)

    # encoding of the label
    if label_bin is None:
        label_bin = LabelBinarizer(neg_label=neg_label, pos_label=pos_label)
        y = label_bin.fit_transform(y_raw)
    else:
        y = label_bin.transform(y_raw)

    return X, y, std_scaler, label_bin
<<<<<<< HEAD


def plot_correlated_features(feat_1, feat_2, n_cols=3, sx=4, sy=5):
    # Plot w.r.t feat_1
    n_rows = math.ceil(feat_1.nunique() / n_cols)
    fig, axs = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(sx * n_cols, sy * n_rows)
    )
    faxs = axs.ravel()

    for i, v in enumerate(feat_1.drop_duplicates().sort_values()):
        idx = feat_1[feat_1 == v].index
        feat_2_sub = feat_2[idx].sort_values()
        faxs[i].plot(range(len(idx)), feat_2_sub)

        faxs[i].set_title(f"{feat_1.name}=={v}")
        faxs[i].title.set_fontsize(9)

        y_min, y_max, y_step = feat_2_sub.min(), feat_2_sub.max() + 1, 6
        faxs[i].set_yticks(
            np.arange(y_min, y_max, step=math.ceil((y_max - y_min) / y_step))
        )
        b_min, b_max, step = 0, len(idx), 6
        faxs[i].set_xticks(
            np.arange(b_min, b_max, step=math.ceil((b_max - b_min) / step))
        )

    for i in range(1, 1 + (n_rows * n_cols - feat_1.nunique())):
        faxs[-i].set_visible(False)
    fig.show()


param_dist = {
    "objective": "binary:logistic",
    "eval_metric": "error",
    "max_depth": 6,
    "eta": 0.3,
    "min_child_weight": 1,
    "gamma": 0,
}
=======
>>>>>>> 409be0da238cc32f2f56698562f57b83b9c799d2
