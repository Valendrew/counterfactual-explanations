from typing import Union

import numpy as np
import pandas as pd

# user imports
from utils.util_dice import DiceCounterfactual
from utils.util_omlt import OmltCounterfactual

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


def generate_counterfactual_from_sample(
    model,
    cf_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample: pd.DataFrame,
    sample_label: int,
    feature_props: dict,
    type_cf="lower",
    backend="PYT",
    target_column="misc_price",
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
    feature_props: dict
        A dictionary that contains the properties of the features of the dataset.
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
        cf_model = OmltCounterfactual(X_train, y_train, model, feature_props)
    elif cf_type == "dice":
        df_dice = pd.concat([X_train, y_train], axis=1)
        # We need to pass all the features as numerical
        cont_feat = list(X_train.columns)

        cf_model = DiceCounterfactual(
            model, backend, df_dice, cont_feat, target=target_column
        )
        cf_model.create_explanation_instance(method=dice_method)
    else:
        raise Exception("Counterfactual class not recognized.")

    sample.loc[:, target_column] = sample_label

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
