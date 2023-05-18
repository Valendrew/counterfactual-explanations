import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from utils import util_plot


def relative_feature_changes(sample: pd.Series, ignore_cols: list, subfix: list, sep="_"):
    def get_columsn_with_subfix(columns: pd.Index, remove_subfix):
        if remove_subfix:
            columns = columns.str.rsplit(sep, n=1).str[0]
        return [f"{col}{sep}{sub}" for sub in subfix for col in columns]

    if len(subfix) < 2:
        raise ValueError("The subfix list must contain at least two elements")
    
    ignore_cols = get_columsn_with_subfix(ignore_cols, False)
    # check columns are in sample
    assert all([col in sample.index for col in ignore_cols])

    # feature columns (have subfixes)
    features_cols = sample.index.drop(ignore_cols)
    features_cols = features_cols.str.rsplit(sep, n=1).str[0].unique()

    difference_values = {}
    for c in features_cols:
        # first element of subfix is the relative name
        if sample[f"{c}{sep}{subfix[1]}"] == np.nan:
            difference_values[c] = np.nan
        else:
            relative_dff = (sample[f"{c}{sep}{subfix[1]}"] - sample[f"{c}{sep}{subfix[0]}"]) / sample[f"{c}{sep}{subfix[0]}"]
            difference_values[c] = relative_dff
    
    return pd.Series(difference_values)


def get_failed_index(df: pd.DataFrame, corr_ind: pd.Index, m_idx: pd.Index, labels: list):
    low_df = df[(df["misc_price_min"] == labels[0])].index
    high_df = df[(df["misc_price_min"] == labels[2])].index
    failed_low = m_idx.difference(low_df)
    failed_high = m_idx.difference(high_df)
    failed_other = corr_ind.difference(df.index)
    return failed_low.union(failed_high).union(failed_other)

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
            df[feat] = df[feat].astype(val).round(4)
        elif val == int:
            df[feat] = df[feat].astype(float).round(0).astype(int)
        else:
            df[feat] = df[feat].astype(val)
            
    return df

def join_merge_columns(df_cf: pd.DataFrame, df_original: pd.DataFrame, label: list[str]):
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
    if columns.isin(label).any() == False:
        raise Exception(f"The label {label} is not present in the original dataframe.")
    
    lsuffix, rsuffix = "_cf", "_original"

    inner_df = df_cf.join(df_original, how="inner", lsuffix=lsuffix, rsuffix=rsuffix)
    for c in columns:
        left_c = c + lsuffix
        right_c = c + rsuffix
        if inner_df[left_c].dtype != inner_df[right_c].dtype:
            raise TypeError(f"Column '{c}' has different dtypes: {inner_df[left_c].dtype} and {inner_df[right_c].dtype}")
        
        if c not in label:
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


def convert_string_feat(df: pd.DataFrame, feat_map: dict):
    for feat, val in feat_map.items():
        df[feat] = df[feat].replace(val)
    return df


def compute_distance(x, y, ord=2):
    return np.linalg.norm(x - y, ord=ord)


def subplots_changed_features(df: pd.DataFrame, feature_columns: pd.Index, plot_mode: str, plot_title: str, ignore_cols=None, **kwargs):
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
    n_rows, n_cols = 2, 2

    if grouped_df.ngroups > n_rows * n_cols:
        raise Exception(f"The number of groups is greater than the number of subplots: {grouped_df.ngroups} > {n_rows * n_cols}")
    
    fig, axs = plt.subplots(n_rows, n_cols, **kwargs)
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
        elif plot_mode == "relative_change":
            vals_changed_dice_corr = group.apply(relative_feature_changes, axis=1, ignore_cols=ignore_cols, subfix=["original", "cf"])
            mean_changed_dice = vals_changed_dice_corr.mean(axis=0) * 100
            util_plot.plot_relative_changes(mean_changed_dice, faxs[i], f"Original price: {original_price} - CF price: {cf_price}", remove_right=i % 2)
        else:
            raise Exception("The selected plot mode is not supported.")

    for i in range(1, n_rows * n_cols - grouped_df.ngroups + 1):
        faxs[-i].axis("off")
    fig.suptitle(plot_title)
    fig.tight_layout()
    

def plot_changed_features(df: pd.DataFrame, feature_columns: pd.Index, plot_mode: str, plot_title: str, ignore_cols, **kwargs):
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
        num_changed.value_counts().sort_index().plot.bar(rot=0, title=plot_title, **kwargs)
    elif plot_mode == "feat_count":
        features_changed = Counter(np.concatenate(df.apply(count_type_features, columns=feature_columns, axis=1).values))
        features_changed = pd.Series(features_changed).sort_values(ascending=False) #if you want to normalize divide by / len(merge_df)
        features_changed.plot.bar(title=plot_title, rot=60, **kwargs)
    elif plot_mode == "relative_change":
        vals_changed_dice_corr = df.apply(relative_feature_changes, axis=1, ignore_cols=ignore_cols, subfix=["original", "cf"])
        mean_changed_dice = vals_changed_dice_corr.mean(axis=0) * 100
        util_plot.plot_relative_changes(mean_changed_dice, plt.gca(), plot_title, remove_right=True)
    else:
        raise Exception("The selected plot mode is not supported.")

    plt.tight_layout()


def plot_cfs_stats(df: pd.DataFrame, feature_columns: pd.Index, plot_mode: str, plot_title: str, split_ranges: bool, **kwargs):
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
    **kwargs: tuple
        The arguments to give as input to a matplotlib plot function.
    '''
    if split_ranges == True:
        subplots_changed_features(df, feature_columns, plot_mode, plot_title, **kwargs)
    else:
        plot_changed_features(df, feature_columns, plot_mode, plot_title, **kwargs)


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

def compute_most_similar_value(cf_sample: pd.Series, label: str, search_df: pd.DataFrame, same_label=False):
    '''
    A function that computes the most similar sample to the one with
    index idx present in cfs.
    '''
    X = cf_sample.drop(label)
    y = cf_sample[label]

    search_X = search_df.drop(label, axis=1)
    search_y = search_df[label]

    if same_label == True:
        search_index = search_y[search_y == y].index.difference([cf_sample.name])
    else:
        search_index = search_y.index.difference([cf_sample.name])

    if not isinstance(search_index, pd.Index):
        raise TypeError

    diff_df = (search_X.loc[search_index] - X).abs().mean(axis=1)
    # diff_df = search_X.loc[search_index].apply(lambda x: np.mean(np.abs(x - X)), axis=1)
    if not isinstance(diff_df, pd.Series):
        raise TypeError(f"norm_df is a {type(diff_df)} instead of a pd.Series")
    
    most_sim_idx = diff_df.idxmin()
    
    if diff_df.loc[most_sim_idx] > 0.1:
        print(f"WARNING: the most similar sample for the cf {cf_sample.name} has a difference greater than the threshold. The difference is {diff_df.loc[most_sim_idx]:.2f} for sample {most_sim_idx}")
        return
            
    # return search_df.loc[most_sim_idx]
    return most_sim_idx