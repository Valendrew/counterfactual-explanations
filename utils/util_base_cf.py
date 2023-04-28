# python imports
from abc import abstractmethod

# 3rd party imports
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def inverse_pipeline(cols_pipeline, df):
    """
    It compute the inverse of the transformations applied by the pipeline
    on the features.

    Parameters:
    -----------
    cols_pipeline:
        The pipeline used to transform the data.
    df: pd.DataFrame
        The dataframe that contains the data to be transformed.

    Returns:
    --------
    results: pd.DataFrame
        The dataframe with the converted values obtained applying the
        inverse_transform function of the pipeline.
    """
    inverted_df = pd.DataFrame()
    for name, pl, in_cols in cols_pipeline.transformers_:
        if name == "remainder":
            continue
        pl: Pipeline = pl
        in_cols: list[str] = in_cols
        # assert isinstance(t[1], Pipeline)

        out_cols = pl.get_feature_names_out(in_cols)
        # print(f"{name}: {out_cols}", end="\n\n")
        results = pl.inverse_transform(df[out_cols])
        # if the result is a numpy array, convert it to a dataframe
        if isinstance(results, np.ndarray):
            results = pd.DataFrame(results, columns=out_cols, index=df.index)

        inverted_df = pd.concat([inverted_df, pd.DataFrame(results)], axis=1)

    return inverted_df


class BaseCounterfactual:
    """
    It's the basic class for counterfactual that contains only the
    generic methods useful for both OMLT and Dice.
    """

    def __init__(self, model, X: pd.DataFrame, y: pd.Series, feature_props: dict):
        self.model = model
        self.X = X
        self.y = y
        
        columns = X.columns
        self.feature_props = dict(sorted(feature_props.items(), key=lambda x: columns.get_loc(x[0])))
        self.start_samples = None
        self.CFs = None

    def get_property_values(self):
        pass

    def destandardize_cfs_orig(self, pipeline):
        """
        It works on the last generated counterfactuals and the relative
        starting samples, inverting the transform process applied to the data.

        Parameters:
        -----------
        pipeline:
            The pipeline used to preprocess the dataset.

        Returns:
        --------
        list
            It returns a list of pairs sample - counterfactuals with
            unstandardized values, in practice are both pd.DataFrame.
        """
        assert (
            self.CFs is not None or self.start_samples is not None
        ), "The cfs or the samples are None"
        # If called by OMLT it gets a numpy array
        if isinstance(self.start_samples, np.ndarray):
            try:
                features = self.X.columns.tolist() + [self.y.name]
                samples = pd.DataFrame(
                    self.start_samples.reshape(1, -1), columns=features
                )
            except Exception as e:
                print(
                    "It tries to read the X and y value from the class but it's not present."
                )
                raise e
        else:
            samples = self.start_samples

        denom_samples = inverse_pipeline(pipeline, samples)
        denom_cfs = [inverse_pipeline(pipeline, cf) for cf in self.CFs]

        pairs = []
        for i in range(denom_samples.shape[0]):
            pairs.append((denom_samples.iloc[[i]], denom_cfs[i]))

        return pairs

    @staticmethod
    def compare_sample_cf(pairs):
        """
        It returns a dataframe that has the features as index, a column for
        the original sample and a column for each generated counterfactual.

        Parameters:
        -----------
        pairs: list
            The list of pairs returned by the 'destandardize_cfs_orig'
            function.

        Returns:
        --------
        list
            A list of dataframe which have in each column the values of
            the original sample and the counterfactuals.
        """
        comp_dfs = []
        for i in range(len(pairs)):
            sample = pairs[i][0].transpose().round(3)
            cfs = pairs[i][1].transpose().round(3)

            # Rename the dataframes correctly
            sample.columns = ["Original sample"]
            cfs.columns = [f"Counterfactual_{k}" for k in range(cfs.shape[1])]

            comp_df = pd.concat([sample, cfs], axis=1)
            comp_dfs.append(comp_df)
        return comp_dfs
    
    @abstractmethod
    def generate_counterfactuals(self, X, y, **kwargs):
        """
        It generates the counterfactuals for the given sample.

        Parameters:
        -----------
        X: pd.DataFrame
            The sample to be explained.
        y: int
            The target class.
        **kwargs:
            The parameters needed to generate the counterfactuals.

        Returns:
        --------
        list
            A list of pd.DataFrame which contains the counterfactuals.
        """
        pass