import math

import numpy as np
import torch
import pandas as pd

# pyomo for optimization
import pyomo.environ as pyo

# omlt for interfacing our neural network with pyomo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io.onnx import (
    write_onnx_model_with_bounds,
    load_onnx_neural_network_with_bounds,
)
import tempfile
import onnx

import dice_ml


##############################################
# Functions to compute the OMLT objectives
##############################################

def handmade_softmax(input, n_class, real_class):
    """
    It returns the probability of the desired class after having computed the
    softmax for the input array.
    """
    exps = [pyo.exp(input[i]) for i in range(n_class)]
    probs = []
    for exp in exps:
        res = exp / sum(exps)
        probs.append(res)
    return probs[real_class]


def features_constraints(pyo_model, feat_info):
    """
    Set the bounds and the domain for each features given a dictionary that
    contains the bounds as a tuple, the domain as a pyomo domain and the position
    of the feature in the columns.
    """
    for feat, info in feat_info.items():
        bounds = info["bounds"]
        domain = info["domain"]
        idx = info["index"]

        pyo_model.nn.inputs[idx].domain = domain
        pyo_model.nn.inputs[idx].bounds = bounds


def compute_obj_1(pyo_model, cf_class, num_classes=3, min_logit=2):
    """
    It creates the objective function to minimize the distance between the
    predicted and the desired class.
    
    Parameters:
    -----------
    pyo_model:
        The pyomo model where the variables and constraints will be added.
    cf_class: int
        The class that the counterfactual should have after the generation.
    num_classes: int
        The number of classes of the task.
    min_logit: float
        An accepted value for the logit value of the predicted class.
    
    Returns:
    --------
        It returns the pyomo variable that contains the value to optimize.
    """
    # prob_y = lambda x: handmade_softmax(x, num_classes, cf_class)

    # something
    pyo_model.obj1_q_relu = pyo.Var(within=pyo.Binary, initialize=0)

    # constraints
    pyo_model.obj1_z_lower_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_lower_bound_zhat_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_zhat_relu = pyo.Constraint()

    l, u = (-15, 15)

    # set dummy parameters here to avoid warning message from Pyomo
    pyo_model.obj1_big_m_lb_relu = pyo.Param(default=-l, mutable=False)
    pyo_model.obj1_big_m_ub_relu = pyo.Param(default=u, mutable=False)

    # define difference of the output
    pyo_model.obj1_diff_prob = pyo.Var(within=pyo.Reals, bounds=(l, u), initialize=0)

    # define variable for max(0, output)
    pyo_model.obj1_max_val = pyo.Var(
        within=pyo.NonNegativeReals, bounds=(0, u), initialize=0
    )

    # constrains the difference of the probabilities (logits)
    pyo_model.obj1_diff_prob_constr = pyo.Constraint(
        expr=pyo_model.obj1_diff_prob == min_logit - pyo_model.nn.outputs[cf_class]
    )

    pyo_model.obj1_z_lower_bound_relu = pyo_model.obj1_max_val >= 0
    pyo_model.obj1_z_lower_bound_zhat_relu = (
        pyo_model.obj1_max_val >= pyo_model.obj1_diff_prob
    )
    pyo_model.obj1_z_upper_bound_relu = (
        pyo_model.obj1_max_val <= pyo_model.obj1_big_m_ub_relu * pyo_model.obj1_q_relu
    )
    pyo_model.obj1_z_upper_bound_zhat_relu = (
        pyo_model.obj1_max_val
        <= pyo_model.obj1_diff_prob
        - pyo_model.obj1_big_m_lb_relu * (1.0 - pyo_model.obj1_q_relu)
    )

    return pyo_model.obj1_max_val


def compute_obj_1_marginal_softmax(
    pyo_model, cf_class: int, num_classes: int, min_probability: float
):
    """
    It creates the objective function to minimize the distance between the
    predicted and the desired class, using the marginal softmax.
    
    Parameters:
    -----------
    pyo_model:
        The pyomo model to consider.
    cf_class: int
        The class that the counterfactual should have after the generation.
    num_classes: int
        The number of classes of the task.
    min_probability: float
        An accepted value for the probability of the predicted class.
    
    Returns:
    --------
        It returns the pyomo variable that contains the value to optimize.
    """
    # prob_y = lambda x: handmade_softmax(x, num_classes, cf_class)

    # something
    pyo_model.obj1_q_relu = pyo.Var(within=pyo.Binary, initialize=0)

    # constraints
    pyo_model.obj1_z_lower_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_lower_bound_zhat_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_zhat_relu = pyo.Constraint()

    l, u = (0, 10)

    # set dummy parameters here to avoid warning message from Pyomo
    pyo_model.obj1_big_m_lb_relu = pyo.Param(default=-l, mutable=False)
    pyo_model.obj1_big_m_ub_relu = pyo.Param(default=u, mutable=False)

    # define difference of the output
    pyo_model.obj1_diff_prob = pyo.Var(within=pyo.Reals, bounds=(l, u), initialize=0)
    # define variable for max(0, output)
    pyo_model.obj1_max_val = pyo.Var(
        within=pyo.NonNegativeReals, bounds=(0, u), initialize=0
    )
    # define variable for marginal softmax
    pyo_model.obj1_marginal_softmax = pyo.Var(bounds=(l, u), initialize=0)

    # Constraints the marginal softmax
    def softmax_constr_rule(m):
        return (
            m.obj1_marginal_softmax
            == pyo.log(sum([pyo.exp(m.nn.outputs[i]) for i in range(num_classes)]))
            - m.nn.outputs[cf_class]
        )

    pyo_model.obj1_marginal_softmax_constr = pyo.Constraint(
        rule=softmax_constr_rule(pyo_model)
    )
    # constrains the difference of the probabilities
    pyo_model.obj1_diff_prob_constr = pyo.Constraint(
        expr=pyo_model.obj1_diff_prob
        == pyo_model.obj1_marginal_softmax - min_probability
    )

    pyo_model.obj1_z_lower_bound_relu = pyo_model.obj1_max_val >= 0
    pyo_model.obj1_z_lower_bound_zhat_relu = (
        pyo_model.obj1_max_val >= pyo_model.obj1_diff_prob
    )
    pyo_model.obj1_z_upper_bound_relu = (
        pyo_model.obj1_max_val <= pyo_model.obj1_big_m_ub_relu * pyo_model.obj1_q_relu
    )
    pyo_model.obj1_z_upper_bound_zhat_relu = (
        pyo_model.obj1_max_val
        <= pyo_model.obj1_diff_prob
        - pyo_model.obj1_big_m_lb_relu * (1.0 - pyo_model.obj1_q_relu)
    )

    return pyo_model.obj1_max_val


def gower_distance(
    x, idx_cat, idx_cont, cat_weights, cont_weights, pyo_model, bounds
):
    """
    It computes an adapted version of the Gower distance. In this case the
    function will also compute the distance between the categorical features
    in the same way the original formula computes the distance between 
    numerical ones.
    
    Parameters:
    -----------
    x: np.ndarray
        The array of features of the original sample.
    idx_cat: list[int]
        The indexes of the categorical features in the DataFrame columns.
    idx_cont: list[int]
        The indexes of the continuous features in the DataFrame columns.
    cat_weights: list[float]
        The list of weights to consider for the features.
    cont_weights: list[float]
        The list of weights to consider for the continuous features.
    pyo_model:
        The pyomo model where the variables and constraints will be added.
    bounds: tuple(pd.Series, pd.Series)
        The series of minimum and maximum values for each feature.
    
    Returns
    -------
    It returns the pyomo variable that contains the sum to minimize. 
    """
    # If the weights are not specified, we set them to 1
    if len(cat_weights) == 0:
        cat_weights = [1] * len(idx_cat)
    if len(cont_weights) == 0:
        cont_weights = [1] * len(idx_cont)

    for type_feat, type_weight in zip([idx_cat, idx_cont], [cat_weights, cont_weights]):
        if len(type_weight) != len(type_feat):
            raise ValueError(
                "The number of weights is not equal to the number of features."
            )

    cont_bounds, cont_dist = 0, 0
    cat_bounds, cat_dist = 0, 0
    # Compute the sum of the bounds for the continuous features
    for i, idx in enumerate(idx_cont):
        range_i = (bounds[1][idx] - bounds[0][idx]) ** 2
        cont_bounds += range_i
        cont_dist += (
            (1 / range_i) * ((x[idx] - pyo_model.nn.inputs[idx]) ** 2) * cont_weights[i]
        )

    # Compute the sum of the bounds for the categorical features
    for i, idx in enumerate(idx_cat):
        range_i = (bounds[1][idx] - bounds[0][idx]) ** 2
        cat_bounds += range_i
        cat_dist += (
            (1 / range_i) * ((x[idx] - pyo_model.nn.inputs[idx]) ** 2) * cat_weights[i]
        )
    # Continuous feature variables and constraints
    pyo_model.obj2_cont_sum = pyo.Var(
        domain=pyo.Reals, bounds=(0, cont_bounds), initialize=0
    )
    pyo_model.obj2_cont_sum_constr = pyo.Constraint(
        expr=pyo_model.obj2_cont_sum == cont_dist
    )
    # Categorical feature variables and constraints
    pyo_model.obj2_cat_sum = pyo.Var(
        domain=pyo.Reals, bounds=(0, cat_bounds), initialize=0
    )
    pyo_model.obj2_cat_sum_constr = pyo.Constraint(
        expr=pyo_model.obj2_cat_sum == cat_dist
    )

    return (pyo_model.obj2_cat_sum + pyo_model.obj2_cont_sum) / len(x)


def compute_obj_3(pyo_model, bounds, sample):
    """
    It creates the third objective function, that limits the number of features
    changed during counterfactual.

    Parameters:
    -----------
    pyo_model
        The model in which the variables and the constraints will be added.
    bounds: tuple[pd.Series, pd.Series]
        The minimum and maximum values for each feature.
    sample: np.ndarray
        The original sample for which the counterfactual is created.
    
    Returns:
    --------
    It returns the pyomo variable that represents the number of changed variables.
    """
    n_feat = len(sample)
    # Set of indexes for the features
    feat_set = pyo.Set(initialize=range(0, n_feat))
    # Variables to handle the values 
    pyo_model.b_o3 = pyo.Var(feat_set, domain=pyo.Binary)
    pyo_model.sum_o3 = pyo.Var(domain=pyo.NonNegativeIntegers, bounds=(0, n_feat), initialize=0)
    pyo_model.diff_o3 = pyo.Var(feat_set, domain=pyo.Reals)
    # Constraints for the if then else
    pyo_model.constr_diff_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_less_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_great_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_sum_o3 = pyo.Constraint()

    for i in range(n_feat):
        range_i = (bounds[1][i] - bounds[0][i]) ** 2
        threshold = 1e-3
        pyo_model.constr_diff_o3[i] = pyo_model.diff_o3[i] == (sample[i] - pyo_model.nn.inputs[i]) ** 2
    
        pyo_model.constr_less_o3[i] = pyo_model.diff_o3[i] >= pyo_model.b_o3[i] - 1 + threshold
        pyo_model.constr_great_o3[i] = (
            pyo_model.diff_o3[i] <= (pyo_model.b_o3[i] * range_i) 
        )
    pyo_model.constr_sum_o3 = pyo_model.sum_o3 == sum(
        [pyo_model.b_o3[i] for i in range(n_feat)]
    )
    return pyo_model.sum_o3


def limit_counterfactual(pyo_model, sample, features, pyo_info):
    """
    It sets some constraints to avoid the change of some features during
    counterfactual generation.
    
    Parameters:
    -----------
    pyo_model:
        The pyomo model to consider.
    sample: np.ndarray
        The sample for which a counterfactual is generated.
    features: list[str]
        The features that the model must not change.
    pyo_info: dict
        The dictionary that contains information about the features.

    """
    if features is None or len(features) < 1:
        return None

    # Remove the batch dimension if present
    sample = sample.squeeze()

    feat_set = pyo.Set(initialize=range(0, len(sample)))
    pyo_model.lim_constr = pyo.Constraint(feat_set)

    for feat in features:
        idx = pyo_info[feat]["index"]
        # Set the counterfactual feature equals to the sample feature
        pyo_model.lim_constr[idx] = pyo_model.nn.inputs[idx] == sample[idx]


################################
# Functions for generic usage
################################

def get_counterfactual_class(initial_class, num_classes, lower=True):
    """
    It returns the counterfactual class given the initial class, the number
    of classes and if the counterfactual needs to be lower or higher. The
    function considers only counterfactuals that differs by 1 from the original
    class.
    """
    if initial_class >= num_classes or initial_class < 0:
        print("ERROR: the initial class has not a valid value.")
        return None
    initial_class = round(initial_class)
    idx_check = 0 if lower else num_classes - 1
    counterfactual_op = -1 if lower else 1
    if initial_class == idx_check:
        print(
            "WARNING: the desired value was out of range, hence the opposite operation has been performed."
        )
        return initial_class - counterfactual_op
    return initial_class + counterfactual_op


def create_feature_pyomo_info(
    X: pd.DataFrame,
    continuous_feat: list,
    categorical_feat: list,
    continuous_bounds=None,
    categorical_bounds=None,
):
    """
    It creates a dictionary that contains, for each feature in X:
    the domain, the bounds and the column index of the feature in the dataframe.

    Parameters:
    -----------
    X: pd.DataFrame
        Dataframe containing the features.
    continuous_feat: list
        List of the continuous features.
    categorical_feat: list
        List of the categorical features.
    continuous_bounds: str or tuple, optional
        Bounds to use for the continuous features. Defaults to None.
    categorical_bounds: str or tuple, optional
        Bounds to use for the categorical features. Defaults to None.

    Returns:
    --------
    dict: 
        Dictionary containing the information about the features.
    """
    features_info = {}

    # Check the bounds passed as parameters are valid
    continuous_bounds = check_bounds(
        X, continuous_bounds, continuous_feat, "continuous"
    )
    categorical_bounds = check_bounds(
        X, categorical_bounds, categorical_feat, "categorical"
    )

    for i, col in enumerate(X.columns.tolist()):
        # Categorical features are encoded as integers
        if col in categorical_feat:
            features_info[col] = {}
            features_info[col]["domain"] = pyo.Integers
            # Set the 2 extremes as bounds for the feature
            features_info[col]["bounds"] = tuple(categorical_bounds.loc[col].to_list())
            features_info[col]["index"] = i
        # Continuous features are encoded as reals
        elif col in continuous_feat:
            features_info[col] = {}
            features_info[col]["domain"] = pyo.Reals
            features_info[col]["bounds"] = tuple(continuous_bounds.loc[col].to_list())
            features_info[col]["index"] = i
        else:
            raise (Exception, f"Feature {col} not present in the dataset.")

    return features_info


def check_bounds(X: pd.DataFrame, bounds, feat: list, name: str) -> pd.DataFrame:
    """
    It checks if the bounds passed as parameters are valid. The supported
    bounds are "min-max" and a tuple of length 2. The min-max bounds are
    computed using the min and max values of the features in X, while the
    tuple of length 2 is used to specify the bounds manually. If the bounds
    are not in the supported ones, an exception is raised.

    Parameters:
    -----------
    X: pd.DataFrame
        Dataframe containing the features.
    bounds: str or tuple
        Bounds to use for the features.
    feat: list
        List of the features.
    name: str
        Name of the category the features belong to.

    Returns:
    --------
    pd.DataFrame: 
        Dataframe containing the bounds for the features.
    """
    if bounds == "min-max":
        bound_min = X[feat].min().values.reshape(-1, 1)
        bound_max = X[feat].max().values.reshape(-1, 1)
    elif len(feat) > 0 and bounds is None:
        raise (Exception, f"Bounds {name} must be specified.")
    elif isinstance(bounds, tuple) and len(bounds) != 2:
        raise (Exception, f"Bounds {name} must be a tuple of length 2.")
    else:
        # bounds = np.array(bounds).reshape(1, -1)
        bound_min = np.repeat(bounds[0], len(feat)).reshape(-1, 1)
        bound_max = np.repeat(bounds[1], len(feat)).reshape(-1, 1)

        if np.any(bound_min < X[feat].min().values):
            print(f"WARNING: {name} lower bound is out of range for some features")
        if np.any(bound_max > X[feat].max().values):
            print(f"WARNING: {name} upper bound is out of range for some features")

    bounds = np.concatenate((bound_min, bound_max), axis=1)
    bounds = pd.DataFrame(bounds, index=feat)
    return bounds


def get_bin_edges(label: float, cols_pipeline):
    label = int(label)
    return pd.Series(cols_pipeline.transformers_[0][1][2].bin_edges_.item()[label: label+2], index=["price_min", "price_max"])


def inverse_pipeline(cols_pipeline, df):
    '''
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
    '''
    results = pd.DataFrame()

    for name, p, cols in cols_pipeline.transformers_:
        if name == "pipeline-1":
            bin_edges = df.misc_price.apply(get_bin_edges, cols_pipeline=cols_pipeline)
            exp_edges = p[1].inverse_transform(bin_edges)
            results = pd.concat([results, exp_edges], axis=1)

        elif name == "pipeline-2":
            quantile_inv = p[0].inverse_transform(df[cols])
            quantile_df = pd.DataFrame(quantile_inv, columns=cols, index=df.index)
            results = pd.concat([results, quantile_df], axis=1)

        elif name == "pipeline-3":
            results = pd.concat([results, df[cols]], axis=1)

        elif name == "pipeline-4":
            ordinal_inv = p[0].inverse_transform(df[cols])
            ordinal_df = pd.DataFrame(ordinal_inv, columns=cols, index=df.index)
            results = pd.concat([results, ordinal_df], axis=1)

        elif name == "pipeline-5":
            cols = p[1].feature_names_in_
            ordinal_inv = p[1].inverse_transform(df[cols])
            ordinal_df = pd.DataFrame(ordinal_inv, columns=cols, index=df.index)
            results = pd.concat([results, ordinal_df], axis=1)

    return results


class BaseCounterfactual:
    '''
    It's the basic class for counterfactual that contains only the
    generic methods useful for both OMLT and Dice.
    '''
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
        assert self.CFs is not None or self.start_samples is not None, "The cfs or the samples are None"
        # If called by OMLT it gets a numpy array
        if isinstance(self.start_samples, np.ndarray):
            try:
                features = self.X.columns.tolist() + [self.y.name]
                samples = pd.DataFrame(self.start_samples.reshape(1, -1), columns=features)
            except Exception as e:
                print("It tries to read the X and y value from the class but it's not present.")
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
                comp_df = comp_df.style.apply(self.__color_df_diff, color=color, axis=1) \
                                       .format(precision=3)
            comp_dfs.append(comp_df)
        return comp_dfs


class OmltCounterfactual(BaseCounterfactual):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nn_model,
        continuous_feat=[],
        continuous_bounds=(-1, 1),
        categorical_bounds="min-max",
    ):
        """
        It is the class used to generate counterfactuals with OMLT.

        Parameters:
        -----------
        X: pd.DataFrame
            The dataframe that contains the X data.
        y: pd.Series
            The series that contains the correct class for the data. The number of
            classes is inferred from this series.
        nn_model:
            The pytorch neural network that the class will use for the counterfactual 
            generation.
        continuous_feat: list, optional 
            List of the continuous features. Defaults to [].
        continuous_bounds: str or tuple, optional
            Bounds to use for the continuous features. Defaults to (-1, 1).
        categorical_bounds: str or tuple, optional
            Bounds to use for the categorical features. Defaults to "min-max".
        """
        super().__init__(nn_model, continuous_feat)
        self.X = X
        self.y = y
        self.num_classes = self.y.nunique()
        self.categorical_feat = X.columns.drop(continuous_feat).to_list()
        self.SUPPORTED_OBJECTIVES = 3
        self.AVAILABLE_SOLVERS = {"mip": "cplex", "nlp": "ipopt"}

        # Create the dictionary that contains the information about the features
        self.feat_info = create_feature_pyomo_info(
            self.X,
            self.continuous_feat,
            self.categorical_feat,
            continuous_bounds=continuous_bounds,
            categorical_bounds=categorical_bounds,
        )

        # Check if the solvers are available in the system
        self.__check_available_solvers()
        # Create the network formulation
        self.__create_network_formulation(-1, 1)


    def __check_available_solvers(self):
        """
        It checks if the solvers that are used are available.

        Raises:
        -------
        Exception: If the solver is not available.
        """
        for solver in self.AVAILABLE_SOLVERS.values():
            if not pyo.SolverFactory(solver).available():
                raise Exception("The solver {} is not available.".format(solver))


    def __create_network_formulation(self, lb: float, ub: float):
        """
        It computes the formulation of the network first converting the model
        to an onnx model and then using pyomo.

        Parameters:
        -----------
        lb: float
            Lower bound for the input features.
        ub: float
            Upper bound for the input features.
        """
        # Create a dummy sample to export the model
        num_features = self.X.shape[1]
        dummy_sample = torch.zeros(size=(1, num_features), dtype=torch.float)
        # Set bound arrays by repeating the bounds for each feature
        lb = np.repeat(lb, num_features)
        ub = np.repeat(ub, num_features)

        # Set the bounds for each feature
        input_bounds = {}
        for i in range(num_features):
            input_bounds[i] = (float(lb[i]), float(ub[i]))

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            # Export neural network to ONNX
            torch.onnx.export(
                self.model,
                dummy_sample,
                f,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            # Write ONNX model and its bounds using OMLT
            write_onnx_model_with_bounds(f.name, None, input_bounds)
            # Load the network definition from the ONNX model
            network_definition = load_onnx_neural_network_with_bounds(f.name)

        self.formulation = FullSpaceNNFormulation(network_definition)


    def __build_model(self):
        """
        It actually builds the formulation of the network to use the model to
        solve an optimization problem.
        """
        # Create a new pyomo model
        self.pyo_model = pyo.ConcreteModel()

        # Create an OMLT block for the nn and build the formulation
        self.pyo_model.nn = OmltBlock()
        self.pyo_model.nn.build_formulation(self.formulation)


    def __compute_objectives(
        self,
        sample: np.ndarray,
        cf_class: int,
        min_probability: float,
        objective_weights=[1, 1, 1],
        cat_weights=[],
        cont_weights=[],
        fixed_features=[],
    ):
        """
        It computes the objective functions to optimize to generate the counterfactuals.
        For the parameters explanation check "generate_counterfactuals" function.

        Parameters:
        -----------
        sample: np.ndarray
            Sample to generate the counterfactual.
        cf_class: int
            Class of the counterfactual.
        min_probability: float
            Minimum probability of the softmax output.
        objective_weights: list, optional
            Weights of the objectives. Defaults to [1, 1, 1].
        cat_weights: list, optional
            The weights for the categorical features in the Gower distance.
        cont_weights: list, optional
            The weights for the continuous features in the Gower distance.
        fixed_features: list, optional
            List of features which are fixed. Defaults to [].
        """
        assert (
            len(objective_weights) == self.SUPPORTED_OBJECTIVES
        ), "The number of objectives is not correct."

        # Set the domain and bounds for each feature of the counterfactual
        features_constraints(self.pyo_model, self.feat_info)

        # OBJECTIVE 1 - generate the counterfactual with the correct class
        if objective_weights[0] == 0:
            obj_1 = 0
            raise Exception("Objective 1 must be computed.")
        else:
            obj_1 = compute_obj_1_marginal_softmax(
                self.pyo_model, cf_class, self.num_classes, min_probability
            )
            # obj_1 = compute_obj_1(self.pyo_model, cf_class, self.num_classes, min_probability)

        # OBJECTIVE 2 - generate counterfactual with limited distances from original features
        cont_df = self.X.loc[:, self.continuous_feat]
        cat_df = self.X.loc[:, self.categorical_feat]

        # We need the index of the features to differentiate in the Gower distance
        idx_cont = [self.X.columns.get_loc(col) for col in cont_df.columns]
        idx_cat = [self.X.columns.get_loc(col) for col in cat_df.columns]

        bounds = self.X.min().values, self.X.max().values
        if objective_weights[1] == 0:
            print(f"Objective 2 is set to 0, so the Gower distance will be 0")
            gower_dist = 1
        else:
            gower_dist = gower_distance(
                sample,
                idx_cat,
                idx_cont,
                cat_weights,
                cont_weights,
                self.pyo_model,
                bounds,
            )

        # OBJECTIVE 3 -  change the minimum number of features
        if objective_weights[2] == 0:
            obj_3 = 1
            print(
                f"Objective 3 is set to 0, so the number of changed features is not minimized."
            )
        else:
            obj_3 = compute_obj_3(self.pyo_model, bounds, sample)

        # Don't change some features
        limit_counterfactual(self.pyo_model, sample, fixed_features, self.feat_info)

        final_obj = (
            objective_weights[0] * obj_1
            + objective_weights[1] * gower_dist
            + objective_weights[2] * obj_3
        )

        # Set the objective function
        self.pyo_model.obj = pyo.Objective(expr=final_obj)


    def generate_counterfactuals(
        self,
        sample: np.ndarray,
        cf_class: int,
        min_probability: float,
        obj_weights=[1, 1, 1],
        cont_weights=[],
        cat_weights=[],
        fixed_features=[],
        solver_options={},
        verbose=True,
    ) -> pd.DataFrame:
        """
        It generates the counterfactual for a given sample, considering the passed
        parameters.

        Parameters:
        -----------
        sample: np.ndarray
            Sample to generate the counterfactual, with the label as last column.
        cf_class: int
            Class of the counterfactual the sample should be classified as.
        min_probability: float
            Minimum probability of the softmax output.
        obj_weights: list, optional
            Weights of the objectives. Defaults to [1, 1, 1].
        cont_weights: list, optional
            The weights to consider for the continuous features in the Gower distance.
        cat_weights: list, optional
            The weights to consider for the categorical features in the Gower distance.
        fixed_features: list, optional
            List of features which are fixed. Defaults to [].
        solver_options: dict, optional
            Options for the solver. Defaults to {}.
        verbose: bool, optional
            Verbose mode for the solver. Defaults to True.

        Returns:
        --------
        pd.DataFrame: 
            A dataframe with the counterfactual sample.
        """
        # Reset the pyomo model
        self.__build_model()
        # Set the objective function
        self.__compute_objectives(
            sample[:-1],
            cf_class,
            min_probability,
            obj_weights,
            cont_weights,
            cat_weights,
            fixed_features,
        )

        # Set the solver to mindtpy
        solver_factory = pyo.SolverFactory("mindtpy")

        # Set the solver options. These options are used for all solvers except mindtpy
        # for key, value in solver_options.items():
        #     # solver_factory.options[key] = value
        #     pass

        # The mindtpy solver has different options for the mip and nlp solver
        pyo_solution = solver_factory.solve(
            self.pyo_model,
            tee=verbose,
            time_limit=solver_options["timelimit"],
            mip_solver=self.AVAILABLE_SOLVERS["mip"],
            nlp_solver=self.AVAILABLE_SOLVERS["nlp"],
        )
        self.start_samples = sample
        # Convert the pyomo solution to a dataframe
        counterfactual_sample = list(self.pyo_model.nn.inputs.get_values().values())
        counterfactual_df = pd.DataFrame(
            np.array(counterfactual_sample, ndmin=2), columns=self.X.columns
        )
        # Find the predicted label for the counterfactual
        logit_dict = self.pyo_model.nn.outputs.get_values()
        out_label = max(logit_dict, key=logit_dict.get)
        counterfactual_df['misc_price'] = out_label

        self.CFs = [counterfactual_df]
        return counterfactual_df
    

class DiceCounterfactual(BaseCounterfactual):
    """
    It's a class that allows you to create Dice counterfactuals, taking as input
    a model, the dataframe with the data, the continuous feature and the target feature.
    """

    def __init__(
        self, model, backend: str, data: pd.DataFrame, cont_feat: list[str], target: str
    ):
        """
        Parameters:
        -----------
        model:
            It's the model to use for counterfactuals (torch, sklearn or tensorflow).
        backend: str
            A string between 'sklearn', 'TF1', 'TF2' and 'PYT'.
        data: pd.DataFrame
            The data used from Dice for statistics.
        cont_feat: list[str]
            The list of names of continuous features.
        target: str
            The name of the target feature.
        """
        dice_mod = dice_ml.Model(model=model, backend=backend)
        super().__init__(dice_mod, cont_feat)

        self.data = dice_ml.Data(
            dataframe=data, continuous_features=cont_feat, outcome_name=target
        )
        self.target = target
        self.backend = backend
        self.explanation = None


    def create_explanation_instance(self, method: str = "genetic"):
        """
        It generates the Dice explanation instance using the model and the
        data passed during the initialization.

        Parameters:
        -----------
        method: str
            The method that will be used during the counterfactual generation.
            A string between 'random', 'kdtree', 'genetic' or 'gradient' for differentiable
            models.
        """
        self.explanation = dice_ml.Dice(self.data, self.model, method=method)


    def generate_counterfactuals(
        self, sample: pd.DataFrame, new_class: int, target: str, n_cf: int=1, 
        proximity_weight: float=0.4, sparsity_weight: float=0.7, stopping_threshold: float=0.5,
        feature_weights="inverse_mad", features_to_vary='all'
    ):
        """
        It generates the counterfactuals using an explanation instance.

        Parameters:
        -----------
        sample: pd.DataFrame
            The dataframe that contains the samples for which the model
            will generate the counterfactuals.
        new_class: int
            The new label the counterfactuals will be predicted with.
        target: str
            The name of the target feature, that will be removed from the
            sample before generating the counterfactuals.
        n_cf: int
            The number of counterfactuals to generate for each sample.
        proximity_weight: float
            The weight to consider for the proximity of the counterfactual
            to the original sample, used by 'genetic' generation.
        sparsity_weight: float
            The weight to change less features when the counterfactual is
            created, used by 'genetic' generation.
        stopping_threshold: float
            The minimum threshold to for counterfactual target probability.
        features_weights: str or dict
            The dictionary with the name of the features as key and the weight
            as value. By default is the 'inverse_mad'.
        features_to_vary: str or list[str]
            The string 'all' to consider all the features or a list of features
            present in the dataframe columns.
        
        Returns:
        --------
        list: pd.DataFrame
            It returns a list with a DataFrame that contains the counterfactual
            values for each sample, if n_cf > 1 the dataframe will contain n_cf rows.
        """
        assert isinstance(
            sample, pd.DataFrame
        ), "The samples need to be in a dataframe."
        if self.explanation is None:
            print(
                "WARNING: you didn't create an explanation instance, therefore a default one will be created in order to proceed.\n"
            )
            self.create_explanation_instance()

        # Save the passed samples
        self.start_samples = sample.copy()
        raw_CFs = self.explanation.generate_counterfactuals(
            sample.drop(target, axis=1), total_CFs=n_cf,
            desired_class=new_class, proximity_weight=proximity_weight,
            sparsity_weight=sparsity_weight, stopping_threshold=stopping_threshold,
            feature_weights=feature_weights, features_to_vary=features_to_vary
        )
        self.CFs = [cf.final_cfs_df.astype(float) for cf in raw_CFs.cf_examples_list]

        return self.CFs
    


def generate_counterfactual_from_sample(
        model, cf_class, X_train, y_train, sample, sample_label, 
        cont_feat=None, lower_cf=True, backend="PYT", target_dice="misc_price",
        dice_method='random', pipeline=None, **kwargs_cf
):
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
    lower_cf: bool
        If the counterfactual class needs to be 1 value lower than 'sample_label' if
        possible.
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
    if cf_class == "omlt":
        assert cont_feat is not None, "You need to pass a list of continuous features for the Omlt class."
        cf_model = OmltCounterfactual(X_train, y_train, model, cont_feat)
        # Add the label as last value of the array
        sample = np.append(sample.values[0], sample_label)

    elif cf_class == "dice":
        df_dice = pd.concat([X_train, y_train], axis=1)
        # We need to pass all the features as numerical
        cont_feat = list(X_train.columns)

        cf_model = DiceCounterfactual(model, backend, df_dice, cont_feat, target=target_dice)
        cf_model.create_explanation_instance(method=dice_method)
        sample.loc[:, target_dice] = sample_label
        
    else:
        raise Exception("Counterfactual class not recognized.")
    
    cf_class = get_counterfactual_class(sample_label, 3, lower_cf)
    cf = cf_model.generate_counterfactuals(
        sample, cf_class, **kwargs_cf
    )

    if pipeline is not None:
        pairs = cf_model.destandardize_cfs_orig(pipeline=pipeline)
    else:
        print("WARNING: the pipeline is not passed, therefore only the found counterfactual will be returned.")
        return cf
    
    compare_dfs = cf_model.compare_sample_cf(pairs)
    return compare_dfs
