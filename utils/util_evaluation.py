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
    counterfactual, subtracting to the number of features, the ones
    that are Nan (not changed) in the series.

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


def count_type_features(series, columns: pd.Index):
    na_columns = series[series.isna()].index
    assert isinstance(na_columns, pd.Index)
    na_columns = na_columns.str.removesuffix("_cf")
    return columns.difference(na_columns).to_numpy()


def convert_string_feat(df, feat_map):
    for feat, val in feat_map.items():
        df[feat] = df[feat].replace(val)
    return df


def compute_distance(x, y, ord=2):
    return np.linalg.norm(x - y, ord=ord)


def subplots_changed_features(df, feature_columns, plot_mode, plot_title, figsize=(12, 8)):
    '''
    It plots four different graphs that represent the number of changed features
    in the computed counterfactuals, split by range of price change.

    Parameters:
    -----------
    Check the documentation of 'plot_cfs_stats'.
    '''
    assert isinstance(df, pd.DataFrame)
    feat_prices_min = ['misc_price_min_original', 'misc_price_min_cf']
    feat_prices_max = ['misc_price_max_original', 'misc_price_max_cf']

    grouped_df = df.groupby(feat_prices_min, axis=0)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    faxs = axs.ravel()

    for i, (_, group) in enumerate(grouped_df):
        original_price = (group[feat_prices_min[0]].unique().item(), group[feat_prices_max[0]].unique().item())
        assert len(original_price) == 2
        cf_price = (group[feat_prices_min[1]].unique().item(), group[feat_prices_max[1]].unique().item())
        assert len(cf_price) == 2

        if plot_mode == "changed_feat":
            num_changed = group.apply(lambda x: count_changed_features(x, 17), axis=1)
            num_changed.value_counts().sort_index().plot.bar(ax=faxs[i], rot=0, title=f"Original price range: {original_price} - CF price range: {cf_price}")
        elif plot_mode == "feat_count":
            features_changed = Counter(np.concatenate(group.apply(count_type_features, columns=feature_columns, axis=1).values))
            features_changed = pd.Series(features_changed).sort_values()
            features_changed.plot.barh(title=f"Original price range: {original_price} - CF price range: {cf_price}", ax=faxs[i])
        else:
            raise Exception("The selected plot mode is not supported.")

    fig.suptitle(plot_title)
    fig.tight_layout()
    

def plot_changed_features(df, feature_columns, plot_mode, plot_title, figsize=(12, 8)):
    '''
    It plots a general chart that represent the number of changed features in the 
    computed counterfactuals or the number of changes per each feature.

    Parameters:
    -----------
    Check the documentation of 'plot_cfs_stats'.
    '''
    assert isinstance(df, pd.DataFrame)

    if plot_mode == "changed_feat":
        num_changed = df.apply(lambda x: count_changed_features(x, len(feature_columns)), axis=1)
        num_changed.value_counts().sort_index().plot.bar(rot=0, title=plot_title)
        plt.tight_layout()
    elif plot_mode == "feat_count":
        features_changed = Counter(np.concatenate(df.apply(count_type_features, columns=feature_columns, axis=1).values))
        features_changed = pd.Series(features_changed).sort_index() #if you want to normalize divide by / len(merge_df)
        features_changed.plot.bar(title=plot_title, rot=60, figsize=figsize)
        plt.tight_layout()
    else:
        raise Exception("The selected plot mode is not supported.")

    plt.tight_layout()


def plot_cfs_stats(df, feature_columns, plot_mode, plot_title, split_ranges, figsize=(12, 8)):
    '''
    It plots a single chart or some subplots that represent the number of changed features
    in the computed counterfactuals or the number of changes per each feature.

    Parameters:
    -----------
    df: pd.DataFrame
        The dataframe that contains the data to plot.
    feature_columns: pd.Index
        The columns that the function will consider to work on the features.
    plot_mode: str
        The plot can show the number of changed features for each counterfactual, passing
        the "changed_feat" parameter, or the number of times that each feature has been 
        changed passing "feat_count" parameter.
    plot_title: str
        The general title to use for the plot. The subtitles for the different subplots
        represent the ranges of the counterfactuals.
    split_ranges: bool
        If True then the function will show 4 different subplots, one for each counterfactual
        range, otherwise it will show a general plot.
    figsize: tuple
        The size to use for the figure plotted by the function.
    '''
    if split_ranges == True:
        subplots_changed_features(df, feature_columns, plot_mode, plot_title, figsize)
    else:
        plot_changed_features(df, feature_columns, plot_mode, plot_title, figsize)


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