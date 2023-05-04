import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


def cast_round_type(df, type_dict):
    '''
    It casts the types of the features in df to the types passed
    in the dictionary.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe to work on.
    type_dict: dict
        The dictionary that contains for each feature a dictionary with the
        mapping from a type to the new one.

    Returns:
    --------
    pd.DataFrame
        The dataframe with the new types.
    '''
    for feat, val in type_dict.items():
        if val == float:
            df[feat] = df[feat].astype(val).round(2)
        elif val == int:
            df[feat] = df[feat].astype(float).round(0).astype(int)
        else:
            df[feat] = df[feat].astype(val)
            
    return df


def join_merge_columns(df_cf, df_original):
    '''
    It merges the two passed dataframes using the inner join.

    Parameters:
    -----------
    df_cf: pd.DataFrame
        The dataframe that contains the counterfactuals.
    df_original:
        The dataframe with the original samples.

    Returns:
    --------
    pd.DataFrame:
        The dataframe obtained from the join between the two given
        in input.
    '''
    columns = df_original.columns
    lsuffix, rsuffix = "_cf", "_original"

    inner_df = df_cf.join(df_original, how="inner", lsuffix=lsuffix, rsuffix=rsuffix)
    for c in columns:
        left_c = c + lsuffix
        right_c = c + rsuffix
        if inner_df[left_c].dtype != inner_df[right_c].dtype:
            raise ValueError(f"Column '{c}' has different dtypes: {inner_df[left_c].dtype} and {inner_df[right_c].dtype}")
        equal_values = inner_df[left_c] == (inner_df[right_c])
        assert isinstance(equal_values, pd.Series)

        if equal_values.any():
            inner_df.loc[equal_values, left_c] = np.nan 
    
    return inner_df


def count_changed_features(series, num_features):
    '''
    It returns the number of features that has been changed in the
    counterfactual.

    Parameters:
    -----------
    series: pd.Series
        The series that contains the value to consider.
    num_features: int
        The number of features.

    Returns:
    --------
    int
        The number of changed features.
    '''
    return num_features - series.isna().sum()


def plot_changed_features_count(df):
    '''
    It plots four different graphs that represent the number of changed features
    in the computed counterfactuals, divided by range of price change.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe that contains the data to plot.
    '''
    assert isinstance(df, pd.DataFrame)
    grouped_df = df.groupby(['misc_price_min_original', 'misc_price_min_cf'], axis=0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    faxs = axs.ravel()

    for i, (name, group) in enumerate(grouped_df):
        original_price = (group["misc_price_min_original"].unique().item(), group["misc_price_max_original"].unique().item())
        assert len(original_price) == 2
        cf_price = (group["misc_price_min_cf"].unique().item(), group["misc_price_max_cf"].unique().item())
        assert len(cf_price) == 2

        num_changed = group.apply(lambda x: count_changed_features(x, 17), axis=1)
        num_changed.value_counts().sort_index().plot.bar(ax=faxs[i], title=f"Original price range: {original_price} - CF price range: {cf_price}")

    fig.suptitle("Number of changed features per price range")
    fig.tight_layout()
    

def count_type_features(series, columns: pd.Index):
    na_columns = series[series.isna()].index
    assert isinstance(na_columns, pd.Index)
    na_columns = na_columns.str.removesuffix("_cf")
    return columns.difference(na_columns).to_numpy()


def plot_changed_features(df, feature_columns):
    '''
    It plots different graphs to show which features have been changed the most
    during counterfactual generation.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe that contains the data to plot.
    feature_columns: pd.Index
        The columns that the function will use to count the features.
    '''
    assert isinstance(df, pd.DataFrame)
    grouped_df = df.groupby(['misc_price_min_original', 'misc_price_min_cf'], axis=0)
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    faxs = axs.ravel()

    for i, (name, group) in enumerate(grouped_df):
        original_price = (group["misc_price_min_original"].unique().item(), group["misc_price_max_original"].unique().item())
        assert len(original_price) == 2
        cf_price = (group["misc_price_min_cf"].unique().item(), group["misc_price_max_cf"].unique().item())
        assert len(cf_price) == 2

        features_changed = Counter(np.concatenate(group.apply(count_type_features, columns=feature_columns, axis=1).values))
        features_changed = pd.Series(features_changed).sort_values()
        features_changed.plot.barh(title=f"Original price range: {original_price} - CF price range: {cf_price}", ax=faxs[i])

    fig.suptitle("Number of changed features per price range")
    fig.tight_layout()


def show_sample(df, idx):
    '''
    It returns the counterfactual sample highlighting the changed features.
    '''
    sample = df.loc[idx]
    not_changed_cols = []
    assert isinstance(sample, pd.Series)
    for c in sample.index[sample.isna()]:
        prefix_col = c.rsplit("_", 1)[0]
        with_value_col = prefix_col + "_original"
        sample = sample.rename(index={with_value_col: prefix_col})

        not_changed_cols.append(prefix_col)

    df_sample = sample[sample.notna()].sort_index().to_frame()
    return df_sample.style.apply(lambda x: ["background: blue; opacity: 0.50; color: white;" if v not in not_changed_cols else "" for v in x.index], axis=0)


def convert_string_feat(df, feat_map):
    for feat, val in feat_map.items():
        df[feat] = df[feat].replace(val)
    return df


def compute_distance(x, y):
    return np.linalg.norm(x - y, ord=2)