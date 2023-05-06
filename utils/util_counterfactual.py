from typing import Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# user imports
from utils.util_dice import DiceCounterfactual
from utils.util_omlt import OmltCounterfactual

from dice_ml.model import UserConfigValidationException

################################
# Functions for generic usage
################################

def create_feature_props(df: pd.DataFrame, cont_feat: list, cat_feat: list, weights: np.ndarray) -> dict:
    def compute_discrete(values: np.ndarray, bounds):
        values = np.unique(values)
        if isinstance(bounds, tuple):
            return list(values[(values >= bounds[0]) & (values <= bounds[1])])
        else:
            raise ValueError("Bounds must be a tuple.")

    feature_props = {}
    for idx, feat in cont_feat:
        if df[feat].unique().shape[0] < 30:
            feature_props[feat] = {
                "weight": weights[idx],
                "type": "continuous",
                "bounds": (-2, 2),
                "discrete": compute_discrete(df[feat], (-2, 2))
            }
        else:
            feature_props[feat] = {
                "weight": weights[idx],
                "type": "continuous",
                "bounds": (-2, 2),
            }
    for idx, feat in cat_feat:
        feature_props[feat] = {
            "weight": weights[idx],
            "type": "categorical",
            "bounds": (int(df[feat].min()), int(df[feat].max()))
        }
    return feature_props


def get_counterfactual_class(initial_class: int, num_classes: int, lower: bool, verbose: bool = True):
    """
    It returns the counterfactual class given the initial class, the number
    of classes and if the counterfactual needs to be lower or higher. The
    function considers only counterfactuals that differs by 1 from the original
    class.
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    # Check all parameters are of the correct type
    assert type(initial_class) in [np.int64, np.int32, int], "The initial class must be an integer."
    assert type(num_classes) in [np.int64, np.int32, int], "The number of classes must be an integer."
    assert isinstance(lower, bool), "The lower parameter must be a boolean."

    if initial_class >= num_classes or initial_class < 0:
        raise ValueError("ERROR: the initial class has not a valid value.")

    idx_check = 0 if lower else num_classes - 1
    counterfactual_op = -1 if lower else 1
    if initial_class == idx_check:
        vprint(
            "WARNING: the desired value was out of range, hence the opposite operation has been performed."
        )
        return initial_class - counterfactual_op
    return initial_class + counterfactual_op

def generate_counterfactual(sample: pd.Series, sample_label: int, target_column: str, type_cf: str, cf_model, pipeline, **kwargs_cf):
    """Generates a counterfactual for the given sample.

    Parameters
    ----------
    sample : pd.Series
        The sample for which we want to generate the counterfactual.
    sample_label : int
        The label of the sample.
    target_column : str
        The target column of the label in the dataset.
    type_cf : str
        The type of counterfactual to generate. It can be 'lower', 'increase' or 'same'.
    cf_model : _type_
        The counterfactual model to use.
    pipeline : _type_
        The pipeline to use to destandardize the counterfactual.

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        The type of counterfactual is not recognized.
    UserConfigValidationException
        Dice cannot find a counterfactual.
    """    
    # Check all parameters are of the correct type
    assert isinstance(sample, pd.DataFrame), "The sample must be a pandas dataframe."
    assert type(sample_label) in [np.int64, np.int32, int], "The sample label must be an integer or a pandas series."
    assert isinstance(target_column, str), "The target column must be a string."
    assert isinstance(type_cf, str), "The type of counterfactual must be a string."
    assert isinstance(pipeline, ColumnTransformer) or pipeline is None, "The pipeline must be a ColumnTransformer or None."
    assert isinstance(kwargs_cf, dict), "The kwargs must be a dictionary."

    sample.loc[:, target_column] = sample_label

    # Get the counterfactual class
    if type_cf in ["lower", "increase"]:
        lower_cf = True if type_cf == "lower" else False
        type_cf_value = get_counterfactual_class(sample_label, 3, lower_cf, False)
    elif type_cf == "same":
        type_cf_value = sample_label
    else:
        raise ValueError(f"Counterfactual type '{type_cf}' not recognized as valid. Please use 'lower', 'increase' or 'same'.")
    
    # Generate the counterfactuals
    try:
        cf = cf_model.generate_counterfactuals(sample, int(type_cf_value), **kwargs_cf)
    except ValueError as e:
        raise ValueError(e)
    except UserConfigValidationException as e:
        raise UserConfigValidationException(e) 

    # Denormalize the counterfactuals
    if pipeline is not None:
        try:
            pairs = cf_model.destandardize_cfs_orig(pipeline=pipeline)
        except ValueError as e:
            raise ValueError(e)
    else:
        print(
            "WARNING: the pipeline is not passed, therefore only the found counterfactual will be returned."
        )
        return cf

    compare_dfs = cf_model.compare_sample_cf(pairs)

    return compare_dfs

def generate_counterfactual_from_sample(
    model,
    cf_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample: pd.DataFrame,
    sample_label: int,
    feature_props: dict,
    type_cf,
    target_column,
    backend=None,
    dice_method=None,
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
        assert backend is not None
        assert dice_method is not None

        cf_model = DiceCounterfactual(
            model, backend, X_train, y_train, feature_props=feature_props,
            target=target_column
        )
        cf_model.create_explanation_instance(method=dice_method)
    else:
        raise ValueError("Counterfactual class not recognized.")

    return generate_counterfactual(
        sample,
        sample_label,
        target_column,
        type_cf,
        cf_model,
        pipeline,
        **kwargs_cf,
    )

def generate_counterfactuals_from_sample_list(
    model,
    class_cf: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_list: pd.DataFrame,
    label_list: pd.Series,
    feature_props: dict,
    type_cf: str,
    target_column: str,
    save_filename = None,
    backend=None,
    dice_method=None,
    pipeline=None,
    **kwargs_cf,
) -> pd.DataFrame:
    """Generate counterfactuals for a list of samples.

    Parameters
    ----------
    model :
        The model to use for the predictions during the counterfactual.
    class_cf : str
        The class to use for the generation between 'omlt' and 'dice'.
    X_train : pd.DataFrame
        The features data used for the training of the passed model.
    y_train : pd.Series
        The labels for the data used during the training of the model.
    sample_list : pd.DataFrame
        The dataframe that contains the values of a sample for which we want to generate the counterfactual.
    label_list : pd.Series
        The classes of the passed samples. It must be of the same length of the sample_list.
    feature_props : dict
        A dictionary that contains the properties of the features to use for the generation.
    type_cf : str
        The new counterfactual class, if it needs to be lowered, increased or kept the same.
    target_column : str
        The target feature name of the dataset.
    save_filename : str, optional
        The name of the file where to save the generated counterfactuals.
    backend : str, optional
        The backend to use to initialize the Dice model, by default None
    dice_method : str, optional
        A method to use for the generation in Dice between 'random' and 'genetic', by default None
    pipeline : ColumnTransformer, optional
        The pipeline to denormalize the counterfactuals and the given sample at the end of the process, by default None

    Returns
    -------
    pd.DataFrame
        A dataframe that contains the generated counterfactuals.        

    Raises
    ------
    ValueError
        The sample_list and label_list parameters must have the same length.
    ValueError
        The class_cf parameter must be either 'omlt' or 'dice'.
    """    
    # Assert all the parameters are of the correct type
    assert isinstance(class_cf, str), "The class_cf parameter must be a string."
    assert isinstance(X_train, pd.DataFrame), "The X_train parameter must be a dataframe."
    assert isinstance(y_train, pd.Series) and y_train.dtype in [np.int64, np.int32, int], "The y_train parameter must be a series of integers."
    assert isinstance(sample_list, pd.DataFrame), "The sample_list parameter must be a dataframe."
    assert isinstance(label_list, pd.Series) and label_list.dtype in [np.int64, np.int32, int], "The label_list parameter must be a series of integers."
    assert isinstance(feature_props, dict), "The feature_props parameter must be a dictionary."
    assert isinstance(type_cf, str), "The type_cf parameter must be a string."
    assert isinstance(target_column, str), "The target_column parameter must be a string."
    assert isinstance(save_filename, str) or save_filename is None, "The save_filename parameter must be a string or None"
    assert isinstance(backend, str) or backend is None, "The backend parameter must be a string or None."
    assert isinstance(dice_method, str) or dice_method is None, "The dice_method parameter must be a string or None."
    assert isinstance(pipeline, ColumnTransformer) or pipeline is None, "The pipeline parameter must be a ColumnTransformer or None."
    assert isinstance(kwargs_cf, dict), "The kwargs_cf parameter must be a dictionary."

    vprint = print if kwargs_cf.get('verbose', True) else lambda *args, **kwargs: None
    if len(sample_list) != len(label_list):
        raise ValueError("The sample list and the label list must have the same length.")
    
    # Initialize a counterfactual model
    if class_cf == "omlt":
        cf_model = OmltCounterfactual(X_train, y_train, model, feature_props)
    elif class_cf == "dice":
        assert backend is not None
        assert dice_method is not None

        cf_model = DiceCounterfactual(
            model, backend, X_train, y_train, feature_props=feature_props, target=target_column
        )
        cf_model.create_explanation_instance(method=dice_method)
    else:
        raise ValueError(f"Counterfactual class '{class_cf}' not recognized. Please use 'omlt' or 'dice'.")
    
    cfs_generated = pd.DataFrame()
    # Iterate over the samples and generate the counterfactuals
    for idx, i in enumerate(sample_list.index):
        vprint(f"[{idx}] Generating counterfactual for sample {i}.")
        
        if i not in label_list.index:
            print(f"The index '{i}' of the sample list is not present in the label list.")
            continue
        
        try:
            cfs = generate_counterfactual(
                sample_list.loc[[i]],
                label_list.loc[i],
                target_column,
                type_cf,
                cf_model,
                pipeline,
                **kwargs_cf,
            )
        except ValueError as e:
            print(e)
            continue
        except UserConfigValidationException as e:
            print(e)
            continue

        if cfs is not None and len(cfs) > 0:
            sample_generated: pd.Series = cfs[0]["Counterfactual_0"].rename(i)
            assert isinstance(sample_generated, pd.Series), "The generated counterfactual must be a series."

            # Concat the generated counterfactual with the previous ones
            cfs_generated = pd.concat([cfs_generated, sample_generated.to_frame().T], axis=0)

    if save_filename is not None:
        cfs_generated.to_csv(save_filename, index=True)
    return cfs_generated
