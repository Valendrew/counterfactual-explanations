from typing import Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# user imports
from util_dice import DiceCounterfactual
from util_omlt import OmltCounterfactual

################################
# Functions for generic usage
################################


def get_counterfactual_class(initial_class: int, num_classes: int, lower=True):
    """
    It returns the counterfactual class given the initial class, the number
    of classes and if the counterfactual needs to be lower or higher. The
    function considers only counterfactuals that differs by 1 from the original
    class.
    """
    assert isinstance(initial_class, int)
    if initial_class >= num_classes or initial_class < 0:
        print("ERROR: the initial class has not a valid value.")
        return None

    idx_check = 0 if lower else num_classes - 1
    counterfactual_op = -1 if lower else 1
    if initial_class == idx_check:
        print(
            "WARNING: the desired value was out of range, hence the opposite operation has been performed."
        )
        return initial_class - counterfactual_op
    return initial_class + counterfactual_op


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

    def __init__(self, model, continuous_feat):
        self.model = model
        self.continuous_feat = continuous_feat
        self.start_samples = None
        self.CFs = None

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

    def __color_df_diff(self, row, color):
        """
        It returns for a single row the color needed to highlight the difference
        between the counterfactual and the original sample.

        Parameters:
        -----------
        row: pd.Series
            The series to consider for the operations.
        color: str
            The name of the color to use for the border of the cells.

        Returns:
        --------
        list[str]
            It returns the list of strings that will be used to color the dataframe.
        """
        cell_border = f"border: 1px solid {color}"
        res = []
        for i in range(1, len(row)):
            # If string
            if isinstance(row[0], str):
                if row[0] != row[i]:
                    res.append(cell_border)
                else:
                    res.append("")
            # If numerical value
            else:
                if row[0] - row[i] != 0:
                    res.append(cell_border)
                else:
                    res.append("")

        if any(res):
            res.insert(0, cell_border)
        else:
            res.append("")
        return res

    def compare_sample_cf(self, pairs, highlight_diff=True, color="red"):
        """
        It returns a dataframe that has the features as index, a column for
        the original sample and a column for each generated counterfactual.

        Parameters:
        -----------
        pairs: list
            The list of pairs returned by the 'destandardize_cfs_orig'
            function.
        highlight_diff: bool
            If True, the border of the changed features will be colored.
        color: str
            The color to use for highlight the differences beteween columns.

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
            if highlight_diff:
                comp_df = comp_df.style.apply(
                    self.__color_df_diff, color=color, axis=1
                ).format(precision=3)
            comp_dfs.append(comp_df)
        return comp_dfs


def generate_counterfactual_from_sample(
    model,
    cf_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample: pd.DataFrame,
    sample_label: int,
    cont_feat=None,
    type_cf="lower",
    backend="PYT",
    target_dice="misc_price",
    dice_method="random",
    pipeline=None,
    **kwargs_cf,
) -> Union[pd.DataFrame, list[pd.DataFrame]]:
    """
    It initializes the Omlt or Dice counterfactual class with the given parameters
    and it runs the models to generate a counterfactual for the passed sample.

    Parameters:
    -----------
    model:
        The model to use for the predictions during the counterfactual. For the
        omlt class only pytorch neural networks are allowed.
    cf_class: str
        A string between 'omlt' and 'dice' to decide which counterfactual class
        to use for the generation.
    X_train: pd.DataFrame
        The X data used for the training of the passed model.
    y_train: pd.Series
        The labels for the data used during the training of the model.
    sample: pd.DataFrame
        The dataframe that contains the values of a sample for which we want to generate
        the counterfactual.
    sample_label: int
        The class of the passed sample.
    cont_feat: list[str]
        A list of continuous features, mandatory only for the Omlt class.
    type_cf: str
        The new counterfactual class, if it needs to be lowered, increased or kept the same.
    backend: str
        The backend to use to initialize the Dice model.
    target_dice: str
        The target feature of the dataset.
    dice_method: str
        A method to use for the generation between 'random' and 'genetic'.
    pipeline:
        The pipeline to denormalize the counterfactuals and the given sample at the
        end of the process.
    **kwargs: dict
        The dictionary with all the parameters to pass when the models generate the
        counterfactual.

    Returns:
    --------
    A list of dataframes that contains the comparison between the input samples and
    the generated counterfactuals.
    """
    # Initialize a counterfactual model
    if cf_type == "omlt":
        assert (
            cont_feat is not None
        ), "You need to pass a list of continuous features for the Omlt class."
        cf_model = OmltCounterfactual(
            X_train, y_train, model, cont_feat, continuous_bounds=(-2, 2)
        )
        # Add the label as last value of the array
        sample = np.append(sample.values[0], sample_label)

    elif cf_type == "dice":
        df_dice = pd.concat([X_train, y_train], axis=1)
        # We need to pass all the features as numerical
        cont_feat = list(X_train.columns)

        cf_model = DiceCounterfactual(
            model, backend, df_dice, cont_feat, target=target_dice
        )
        cf_model.create_explanation_instance(method=dice_method)
        sample.loc[:, target_dice] = sample_label

    else:
        raise Exception("Counterfactual class not recognized.")

    # Get the counterfactual class
    if type_cf in ["lower", "increase"]:
        lower_cf = True if type_cf == "lower" else False
        cf_type = get_counterfactual_class(sample_label, 3, lower_cf)
    elif type_cf == "same":
        cf_type = sample_label
    else:
        raise Exception("Counterfactual class not recognized.")

    # Generate the counterfactuals
    cf = cf_model.generate_counterfactuals(sample, cf_type, **kwargs_cf)

    # Denormalize the counterfactuals
    if pipeline is not None:
        pairs = cf_model.destandardize_cfs_orig(pipeline=pipeline)
    else:
        print(
            "WARNING: the pipeline is not passed, therefore only the found counterfactual will be returned."
        )
        return cf

    compare_dfs = cf_model.compare_sample_cf(pairs)
    return compare_dfs
