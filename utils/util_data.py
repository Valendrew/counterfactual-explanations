import os
import shutil
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler, FunctionTransformer, RobustScaler

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


def get_bins(
    df: pd.DataFrame, label: str, log=True, cap=False, cap_value=None, verbose=False
) -> pd.Series:
    vvprint = lambda x: print(x, end="\n\n") if verbose else lambda *a, **k: None
    if cap:
        assert cap_value is not None

    label_series = (
        df[label].clip(lower=cap_value["low"], upper=cap_value["up"]).copy()
        if cap
        else df[label].copy()
    )
    label_series = np.log(label_series) if log else label_series

    q1 = label_series.quantile(0.25)
    q3 = label_series.quantile(0.75)
    vvprint(f"q1: {np.exp(q1) if log else q1:.2f}, q3: {np.exp(q3) if log else q3:.2f}")
    # Take inter-quartile range
    iqr = q3 - q1
    # lower whiskers as 1.5 smaller than iqr
    lower_bound = max(label_series.min(), q1 - iqr * 1.25)
    # upper whiskers as 1.5 greater than iqr
    upper_bound = min(label_series.max(), q3 + iqr * 1.25)

    vvprint(
        f"lower_bound: {np.exp(lower_bound) if log else lower_bound:.2f}, upper_bound: {np.exp(upper_bound) if log else upper_bound:.2f}"
    )

    # dataframe with outliers removed
    label_iqr = label_series[
        label_series.between(lower_bound, upper_bound, inclusive="both")
    ]
    # label of the bins
    target_labels = ["low", "low-medium", "medium", "high"]
    # _, bins = pd.cut(label_iqr, bins=len(target_labels), retbins=True, labels=target_labels)
    _, bins = pd.qcut(
        label_iqr, q=4, retbins=True, labels=target_labels, duplicates="raise"
    )

    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        vvprint(
            f"Range {target_labels[i]}: {np.exp(l) if log else l:.2f} - {np.exp(u) if log else u:.2f}"
        )

    bins[0] = label_series.min()
    bins[-1] = label_series.max()
    lab_cat = pd.Series(index=label_series.index, dtype="object")

    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        vvprint(
            f"Range {target_labels[i]}: {np.exp(l) if log else l:.2f} - {np.exp(u) if log else u:.2f}"
        )
        idx = label_series.between(l, u, inclusive="right")
        lab_cat.loc[idx] = i

    return lab_cat


class LabelEncoder(FunctionTransformer):
    def __init__(self, bins="doane", **kwargs):
        self.bins = bins
        super().__init__(
            self.compute_encoding,
            inverse_func=self.compute_decoding,
            **kwargs,
        )

    def compute_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute label encoding for a single column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with a single column to be encoded.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded column.
        """
        assert X.shape[1] == 1
        assert isinstance(X, pd.DataFrame)

        _, bin_edges = np.histogram(X, bins=self.bins)
        # save bin edges for inverse transform
        self.bin_edges_ = np.array(bin_edges).reshape(1, -1)
        # add small value to last bin edge to include all values
        bin_edges[-1] = bin_edges[-1] + 1e-6

        # convert to labels and then to DataFrame to preserve index
        bin_converter = np.digitize(X, bin_edges, right=False) - 1
        df_transform = pd.DataFrame(
            bin_converter, columns=["misc_price"], index=X.index
        )
        return df_transform

    def get_bin_edges(self, label: float) -> pd.Series:
        """Get min and max bin edges for a given label.

        Parameters
        ----------
        label : float
            Label for which to get bin edges.

        Returns
        -------
        pd.Series
            Series with bin edges.
        """
        label = int(label)
        return pd.Series(
            self.bin_edges_[0][label : label + 2],
            index=["misc_price_min", "misc_price_max"],
        )

    def compute_decoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute label decoding for a single column.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with a single column to be decoded.

        Returns
        -------
        pd.DataFrame
            DataFrame with decoded column.
        """
        assert X.shape[1] == 1, "X must be a DataFrame with one column"
        assert (
            X.columns[0] == "misc_price"
        ), "X must be a DataFrame with one column named 'misc_price'"
        return X.misc_price.apply(self.get_bin_edges)

    def get_feature_names_out(self, input_features: npt.ArrayLike = None) -> np.ndarray:
        return np.array(["misc_price"])


class DisplayEncoder(FunctionTransformer):
    def __init__(self, resolutions: pd.DataFrame, **kwargs):
        self.resolutions = resolutions
        assert self.resolutions.shape[1] == 2

        super().__init__(self.compute_encoding, **kwargs)

    def compute_distance(self, x: pd.Series):
        min_idx = np.argmin(np.linalg.norm(x - self.resolutions, ord=2, axis=1))
        return self.resolutions.index[min_idx]

    def compute_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        series = X.apply(self.compute_distance, axis=1)
        return pd.DataFrame(series, columns=["display_resolution"], index=X.index)

    def get_feature_names_out(self, input_features: npt.ArrayLike = None) -> np.ndarray:
        return np.array(["display_resolution"])


class ClipEncoder(FunctionTransformer):
    def __init__(self, lower: float, upper: float, clip: bool, **kwargs):
        self.lower = lower
        self.upper = upper
        self.clip = clip
        super().__init__(self.compute_encoding, **kwargs)

    def compute_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
        if self.clip:
            return X.clip(lower=self.lower, upper=self.upper)
        else:
            return X[(X >= self.lower) & (X <= self.upper)]

    def get_feature_names_out(self, input_features: npt.ArrayLike = None) -> np.ndarray:
        return np.array(["clip"])


def transform_cont_feature(name_feat: str, X: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Inverse transform of a continuous feature.

    Parameters
    ----------
    name_feat : str
        name of the feature to transform
    X : pd.DataFrame
        dataframe with the feature to transform
    pipeline : Pipeline
        pipeline with the transformation

    Returns
    -------
    pd.DataFrame
        dataframe with the transformed feature and the original feature

    Raises
    ------
    ValueError
        The feature to transform is not found in the dataframe
    ValueError
        The feature to transform is not found in the pipeline
    """    
    if name_feat not in X.columns:
        raise ValueError(f"Feature {name_feat} not found in dataframe")

    col_inv_name = f"{name_feat}_inv"
    feature_vals = np.sort(X[name_feat].unique())
    feat_names = pipeline.feature_names_in_
    # Get the index of the feature in the pipeline
    idx = (
        sum(i + 1 if feat == name_feat else 0 for i, feat in enumerate(feat_names)) - 1
    )
    if idx < 0:
        raise ValueError(f"Feature {name_feat} not found in pipeline")

    # Create a matrix with the feature values only for the feature to transform
    feat_trans = np.zeros((pipeline.feature_names_in_.shape[0], feature_vals.shape[0]))
    feat_trans[idx, :] = feature_vals

    df_feat_trans = pd.DataFrame(feat_trans.T, columns=pipeline.feature_names_in_)
    inverse_feat_trans = pipeline.inverse_transform(df_feat_trans)
    df_inverse_feat_trans = pd.DataFrame(
        inverse_feat_trans, columns=pipeline.feature_names_in_
    ).rename(columns={name_feat: col_inv_name})
    return pd.concat(
        [df_feat_trans[name_feat], df_inverse_feat_trans[col_inv_name]], axis=1
    )


def try_scaler(
    scaler: str, name_feat: str, X: pd.DataFrame, log=True, round=False, **kwargs
) -> Tuple[pd.DataFrame, Pipeline]:
    if name_feat not in X.columns:
        raise ValueError(f"Feature {name_feat} not found in dataframe")

    feature_vals = X[[name_feat]]
    # feature_vals = feature_vals.values.reshape(-1, 1)

    pipeline_scalers = list()
    # Add the round scaler if needed
    if round:
        round_scaler = FunctionTransformer(np.ceil, validate=False)
        pipeline_scalers.append(("round", round_scaler))
    
    # Add the log scaler if needed
    if log:
        log_scaler = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False, check_inverse=False)
        pipeline_scalers.append(("log", log_scaler))

    if scaler == "robust":
        num_scaler = RobustScaler(**kwargs)
    elif scaler == "standard":
        num_scaler = StandardScaler(**kwargs)
    else:
        raise ValueError(f"Scaler {scaler} not found")
    
    pipeline_scalers.append(("scaler", num_scaler))


    pipeline = Pipeline(pipeline_scalers)
    transform_vals = pipeline.fit_transform(feature_vals)

    return pd.DataFrame(transform_vals, columns=[name_feat]), pipeline

def compare_scalers(name_feat: str, df: pd.DataFrame, original_df: pd.DataFrame, original_pipeline: Pipeline, **kwargs):
    # Try Robust Scaler
    robust_1_dict = {
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": (25.0, 75.0),
        "unit_variance": True
    }
    df_ret_robust, num_pipeline = try_scaler("robust", name_feat, df, **robust_1_dict, **kwargs)
    unique_robust = transform_cont_feature(name_feat, df_ret_robust, num_pipeline)

    # Try Standard Scaler
    standard_1_dict = {
        "with_mean": True,
        "with_std": True,
    }
    df_ret_standard, num_pipeline = try_scaler("standard", name_feat, df, **standard_1_dict, **kwargs)
    unique_standard = transform_cont_feature(name_feat, df_ret_standard, num_pipeline)

    # Convert original data
    unique_original = transform_cont_feature(name_feat, original_df, original_pipeline)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    faxs = axs.ravel()

    df_scalers_dict = {
        "Robust Scaler": df_ret_robust,
        "Standard Scaler": df_ret_standard,
    }

    for i, (name_scaler, df_scaler) in enumerate(df_scalers_dict.items()):
        df_scaler[name_feat].hist(bins=50, ax=faxs[i])
        faxs[i].set_title(name_scaler)
    fig.tight_layout()

    return unique_standard.merge(unique_robust, how="outer", left_index=True, right_index=True, suffixes=["_standard", "_robust"]).\
        merge(unique_original, how="outer", left_index=True, right_index=True)