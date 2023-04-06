import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import gdown

# import kaggle only if doesn't have errors
try:
    import kaggle
except OSError:
    print("kaggle.json not found, you cannot use kaggle module.")


class DownloadHelper:
    def __init__(self, url: str, name: str, mode: str, quiet: bool = False):
        self.url = url
        self.name = name

        # path variables
        self.root_path = os.path.join(os.getcwd(), "data", "raw")
        self.file_path = os.path.join(self.root_path, self.name)

        # download mode
        self.mode = mode
        self.quiet = quiet

    def download(self):
        # check if file already exists
        if os.path.exists(self.file_path):
            print(f"File {self.name} already exists. Skip download.")
            return

        # download with gdown
        if self.mode == "gdown":
            gdown.download(id=self.url, output=self.file_path, quiet=self.quiet)
        # download with kaggle
        elif self.mode == "kaggle":
            self.__download_kaggle()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __download_kaggle(self):
        # "name" file already exists
        if os.path.exists(self.file_path):
            print(f"File {self.name} already exists. Skip download.")
            return

        zip_path = os.path.join(self.root_path, f"{self.url.split('/', 1)[1]}.zip")
        # "zip_path" file already exists
        if os.path.exists(zip_path):
            print(f"File {zip_path} already exists.")
        else:
            print("Download file from kaggle.")
            os.system(f"kaggle datasets download -d {self.url} -p {self.root_path}")

        print("Unzip file and move it to the correct location.")
        try:
            shutil.unpack_archive(zip_path, extract_dir=self.root_path)
            # remove zip file
            os.remove(zip_path)
        except ValueError as e:
            print(f"File {zip_path} invalid")

    def read_csv(self, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(self.file_path, **kwargs)
        assert isinstance(df, pd.DataFrame)
        return df


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


def encode_normalize_df(
    df, target: str, std_scaler=None, label_bin=None, neg_label=0, pos_label=1
):
    X_raw, y_raw = df.iloc[:, df.columns != target], df[target]

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


def compute_outlier(df, col_name, threshold=1.5):
    if not isinstance(df, pd.DataFrame):
        series = df
    else:
        series = df[col_name]
    # compute the IQR
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # compute the lower and upper bound
    lower_bound = max(q1 - (threshold * iqr), series.min())
    upper_bound = min(q3 + (threshold * iqr), series.max())
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

    # compute the outliers
    outliers = df.loc[(series < lower_bound) | (series > upper_bound)]
    print(f"Number of outliers: {len(outliers)}")

    return outliers
