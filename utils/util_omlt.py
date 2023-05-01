# python imports
import time

# 3d party imports
import numpy as np
import pandas as pd
import torch

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

# user imports
from utils.util_base_cf import BaseCounterfactual
from utils import util_models

##############################################
# Functions to compute the OMLT objectives
##############################################

class SolverException(Exception):
    pass


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


def features_constraints(pyo_model, feature_props: dict):
    """
    Set the bounds and the domain for each features given a dictionary that
    contains the bounds as a tuple, the domain as a pyomo domain and the position
    of the feature in the columns.

    Parameters:
    -----------
    pyo_model:
        The pyomo model to consider.
    feature_props: dict
        A dictionary that contains the bounds as a tuple and the domain of the
        feature as either 'continuous' or 'categorical'.

    Raises:
    -------
    ValueError:
        If the type of the feature is not valid.
    """
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(feature_props, dict), "feature_props must be a dictionary"

    # Iterate over the features and set the bounds and the domain
    for idx, info in enumerate(feature_props.values()):
        if info["type"] == "categorical":
            # Categorical features are encoded as non-negative integers
            pyo_model.nn.inputs[idx].domain = pyo.NonNegativeIntegers
        elif info["type"] == "continuous":
            # Continuous features are encoded as real numbers
            pyo_model.nn.inputs[idx].domain = pyo.Reals
        else:
            raise ValueError("The type of the feature is not valid.")

        pyo_model.nn.inputs[idx].bounds = info["bounds"]


def features_discrete_values(pyo_model, feature_props: dict):
    """Generate the constraints for the discrete features.

    Parameters
    ----------
    pyo_model :
        The pyomo model to consider.
    feature_props : dict
        A dictionary that contains the properties of the features. Only
        the discrete features are considered.
    """
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(feature_props, dict), "feature_props must be a dictionary"

    # Get the discrete features as a list of tuples (index, values)
    discrete_features = [
        (i, info["discrete"])
        for i, info in enumerate(feature_props.values())
        if info.get("discrete") is not None
    ]
    # Sum the number of discrete values among all the features
    sum_features = sum((len(f[1]) for f in discrete_features))

    # Compute the start indexes for each feature
    start_indexes = [0]
    for idx, (_, feat) in enumerate(discrete_features):
        if idx == len(discrete_features) - 1:
            continue
        start_indexes.append(start_indexes[-1] + len(feat))

    # Set the discrete set
    pyo_model.n_discrete_set = pyo.RangeSet(0, sum_features - 1)
    pyo_model.discrete_set = pyo.RangeSet(0, len(discrete_features) - 1)

    # Set the discrete variables
    pyo_model.discrete_q = pyo.Var(pyo_model.n_discrete_set, domain=pyo.Binary)

    # Set the constraints
    def discrete_sum_rule(model, i):
        return (
            sum(
                model.discrete_q[j + start_indexes[i]]
                for j in range(len(discrete_features[i][1]))
            )
            == 1
        )

    pyo_model.discrete_constr_sum = pyo.Constraint(
        pyo_model.discrete_set, rule=discrete_sum_rule
    )

    def discrete_input_rule(model, i):
        return model.nn.inputs[discrete_features[i][0]] == sum(
            discrete_features[i][1][j] * model.discrete_q[j + start_indexes[i]]
            for j in range(len(discrete_features[i][1]))
        )

    pyo_model.discrete_constr_input = pyo.Constraint(
        pyo_model.discrete_set, rule=discrete_input_rule
    )


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
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert type(cf_class) in [int, np.int32, np.int64], "cf_class must be an integer"
    assert isinstance(num_classes, int), "num_classes must be an integer"
    assert isinstance(min_probability, float), "min_probability must be a float"

    # something
    pyo_model.obj1_q_relu = pyo.Var(within=pyo.Binary, initialize=0)

    # constraints
    pyo_model.obj1_z_lower_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_lower_bound_zhat_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_relu = pyo.Constraint()
    pyo_model.obj1_z_upper_bound_zhat_relu = pyo.Constraint()

    l, u = (-min_probability - 1e-2, 1)

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
    pyo_model.obj1_marginal_softmax = pyo.Var(
        bounds=(0, u), initialize=0, within=pyo.NonNegativeReals
    )

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


def initialize_sample_distance(pyo_model, sample: np.ndarray, feat_ranges: pd.DataFrame, feat_weights: list[float]):
    """_summary_

    Parameters
    ----------
    pyo_model : _type_
        _description_
    x : np.ndarray
        _description_
    feat_ranges : pd.DataFrame
        _description_
    feat_weights : list[float]
        _description_
    """
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(sample, np.ndarray), "x must be a numpy array"
    assert isinstance(feat_ranges, pd.DataFrame), "feat_ranges must be a pandas dataframe"
    assert isinstance(feat_weights, list), "feat_weights must be a list"
    assert sum(feat_weights) <= len(feat_weights), "feat_weights must sum to the length of the features"

     # Set of the features
    pyo_model.n_sample_set = pyo.RangeSet(0, len(sample) - 1)

    # Define the variable for the distance
    pyo_model.sample_distances = pyo.Var(
        pyo_model.n_sample_set, initialize=0, bounds=(0, 1), domain=pyo.NonNegativeReals
    )
    
    # Constraint for the distance
    def dist_constr_rule(m, i):
        return m.sample_distances[i] == (
            (1 / (feat_ranges["max"].iloc[i] - feat_ranges["min"].iloc[i]) ** 2)
            * (sample[i] - m.nn.inputs[i]) ** 2
        ) * feat_weights[i]
    
    pyo_model.constr_distances = pyo.Constraint(
        pyo_model.n_sample_set, rule=dist_constr_rule
    )

def gower_distance(
    pyo_model, sample: np.ndarray
):
    """
    It computes an adapted version of the Gower distance. In this case the
    function will also compute the distance between the categorical features
    in the same way the original formula computes the distance between
    numerical ones.

    Parameters:
    -----------
    pyo_model:
        The pyomo model where the variables and constraints will be added.
    x: np.ndarray
        The array of features of the original sample.
    feat_ranges: pd.DataFrame
        The dataframe containing the ranges of the features.
    feat_weights: list[float]
        The list of weights for the features.

    Returns
    -------
    It returns the pyomo variable that contains the sum to minimize.

    Raises
    ------
    ValueError
        If the type of the features is not valid.
    """
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(sample, np.ndarray), "x must be a numpy array"

    # Set of the features
    # pyo_model.obj2_set = pyo.RangeSet(0, len(sample) - 1)

    # # Define the variable for the distance
    # pyo_model.obj2_dist = pyo.Var(
    #     pyo_model.obj2_set, within=pyo.NonNegativeReals, initialize=0, bounds=(0, 1)
    # )
    
    # # Constraint for the distance
    # def dist_constr_rule(m, i):
    #     return m.obj2_dist[i] == (
    #         (1 / (feat_ranges["max"].iloc[i] - feat_ranges["min"].iloc[i]) ** 2)
    #         * (sample[i] - m.nn.inputs[i]) ** 2
    #     ) * feat_weights[i]
    
    # pyo_model.obj2_dist_constr = pyo.Constraint(
    #     pyo_model.obj2_set, rule=dist_constr_rule
    # )

    # # Feature variables and constraints
    pyo_model.obj2_sum = pyo.Var(
        domain=pyo.NonNegativeReals, bounds=(0, 1), initialize=0
    )

    num_features = len(sample)
    def sum_constr_rule(m):
        return m.obj2_sum == sum(m.sample_distances[i] for i in pyo_model.n_sample_set) / num_features
    
    pyo_model.obj2_sum_constr = pyo.Constraint(rule=sum_constr_rule)
    return pyo_model.obj2_sum


def compute_obj_3(pyo_model, sample: np.ndarray, feat_ranges: pd.DataFrame, feat_props: list[float]):
    """
    It creates the third objective function, that limits the number of features
    changed during counterfactual.

    Parameters:
    -----------
    pyo_model
        The model in which the variables and the constraints will be added.
    sample: np.ndarray
        The original sample for which the counterfactual is created.
    feat_ranges: pd.DataFrame
        The dataframe with the minimum and maximum values for each feature.

    Returns:
    --------
    It returns the pyomo variable that represents the number of changed variables.
    """
    # TODO TO FINISH
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(sample, np.ndarray), "sample must be a numpy array"
    assert isinstance(
        feat_ranges, pd.DataFrame
    ), "feat_ranges must be a pandas dataframe"
    assert isinstance(feat_props, list), "feat_props must be a list"

    categorical_features = [(idx, feat_prop) for idx, feat_prop in enumerate(feat_props) if feat_prop == "categorical"]
    continuous_features = [(idx, feat_prop) for idx, feat_prop in enumerate(feat_props) if feat_prop == "continuous"]
    assert len(categorical_features) + len(continuous_features) == len(feat_props), "feat_props must contain only 'categorical' or 'continuous' values"

    # Set of indexes for the features
    pyo_model.obj3_cat_set = pyo.RangeSet(0, len(categorical_features) - 1)
    pyo_model.obj3_cont_set = pyo.RangeSet(0, len(continuous_features) - 1)
    pyo_model.obj3_feat_set = pyo.RangeSet(0, len(sample) - 1)

    def sample_cat_param_rule(m, i):
        idx = categorical_features[i][0]
        return sample[idx]
    
    def sample_cont_param_rule(m, i):
        idx = continuous_features[i][0]
        return sample[idx]
    
    def range_cat_param_rule(m, i):
        idx = categorical_features[i][0]
        return (feat_ranges["max"].iloc[idx] - feat_ranges["min"].iloc[idx]) ** 2
    
    def range_cont_param_rule(m, i):
        idx = continuous_features[i][0]
        return (feat_ranges["max"].iloc[idx] - feat_ranges["min"].iloc[idx]) ** 2

    # Categorical features    
    pyo_model.obj3_sample_cat = pyo.Param(
        pyo_model.obj3_cat_set, initialize=sample_cat_param_rule,
        mutable=False, within=pyo.NonNegativeIntegers)

    pyo_model.obj3_range_cat = pyo.Param(
        pyo_model.obj3_cat_set, initialize=range_cat_param_rule, mutable=False, within=pyo.NonNegativeIntegers)

    # Continuous features
    pyo_model.obj3_sample_cont = pyo.Param(
        pyo_model.obj3_cont_set, initialize=sample_cont_param_rule,
        mutable=False, within=pyo.Reals)
    
    pyo_model.obj3_range_cont = pyo.Param(
        pyo_model.obj3_cont_set, initialize=range_cont_param_rule, mutable=False, within=pyo.NonNegativeReals)
    
    # Lower bound parameter
    pyo_model.obj3_lower_bound = pyo.Param(initialize=1e-5, mutable=False)

    # pyo_model.diff_o3_cat = pyo.Var(pyo_model.obj3_cat_set, domain=pyo.NonNegativeIntegers, initialize=0)
    # pyo_model.diff_o3_cont = pyo.Var(pyo_model.obj3_cont_set, domain=pyo.NonNegativeReals, initialize=0)

    # Variables to handle the values
    pyo_model.b_o3 = pyo.Var(pyo_model.obj3_feat_set, domain=pyo.Binary, initialize=0)
    # Constraints for the if then else
    # pyo_model.constr_diff_o3 = pyo.Constraint(pyo_model.obj3_feat_set)
    pyo_model.constr_less_o3 = pyo.Constraint(pyo_model.obj3_feat_set)
    pyo_model.constr_great_o3 = pyo.Constraint(pyo_model.obj3_feat_set)

    # for i in pyo_model.obj3_cat_set:
    #     idx = categorical_features[i][0]
    #     pyo_model.diff_o3_cat[i].bounds = (0, pyo_model.obj3_range_cat[i])

    #     pyo_model.constr_diff_o3[idx] = (
    #         pyo_model.diff_o3_cat[i] == (pyo_model.obj3_sample_cat[i] - pyo_model.nn.inputs[idx]) ** 2
    #     )

    #     pyo_model.constr_less_o3[idx] = (
    #         pyo_model.diff_o3_cat[i] >= pyo_model.b_o3[idx] - 1 + pyo_model.obj3_lower_bound
    #     )
    #     pyo_model.constr_great_o3[idx] = pyo_model.diff_o3_cat[i] <= (
    #         pyo_model.b_o3[idx] * pyo_model.obj3_range_cat[i]
    #     )

    for i in pyo_model.obj3_cont_set:
        idx = continuous_features[i][0]
        # pyo_model.diff_o3_cont[i].bounds = (0, pyo_model.obj3_range_cont[i])

        # pyo_model.constr_diff_o3[idx] = (
        #     pyo_model.sample_distances[i] == (pyo_model.obj3_sample_cont[i] - pyo_model.nn.inputs[idx]) ** 2
        # )

        pyo_model.constr_less_o3[idx] = (
            pyo_model.sample_distances[idx] >= pyo_model.b_o3[idx] - 1 + pyo_model.obj3_lower_bound
        )
        pyo_model.constr_great_o3[idx] = pyo_model.sample_distances[idx] <= (
            pyo_model.b_o3[idx] * pyo_model.obj3_range_cont[i]
        )

    pyo_model.sum_o3 = pyo.Var(
        domain=pyo.Reals, bounds=(0, 1), initialize=0
    )

    def sum_constr_rule(m):
        return m.sum_o3 == sum(m.b_o3[i] for i in pyo_model.obj3_feat_set) / len(sample)

    pyo_model.constr_sum_o3 = pyo.Constraint(rule=sum_constr_rule)
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
    # TODO TO FINISH
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(sample, np.ndarray), "sample must be a numpy array"

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


class OmltCounterfactual(BaseCounterfactual):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nn_model,
        feature_props: dict,
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
        # Assert all the parameters are of the correct type
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe"
        assert isinstance(y, pd.Series), "y must be a pandas series"
        assert isinstance(
            feature_props, dict
        ), "feature_props must be a dictionary with the features properties"

        super().__init__(nn_model, X, y, feature_props)

        self.num_classes = self.y.nunique()
        self.SUPPORTED_OBJECTIVES = 3
        self.AVAILABLE_SOLVERS = {"mip": "cplex", "nlp": "ipopt"}

        # Check if the solvers are available in the system
        self.__check_available_solvers()
        # Create the network formulation
        self.__create_network_formulation(-1.0, 1.0)

    def __check_available_solvers(self):
        """
        It checks if the solvers that are used are available.

        Raises:
        -------
        ValueError: If the solver is not available.
        """
        for solver in self.AVAILABLE_SOLVERS.values():
            if not pyo.SolverFactory(solver).available():
                raise ValueError("The solver {} is not available.".format(solver))

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
        # Assert all the parameters are of the correct type
        assert isinstance(lb, float), "The lower bound must be a float."
        assert isinstance(ub, float), "The upper bound must be a float."

        # Create a dummy sample to export the model
        num_features = self.X.shape[1]
        dummy_sample = torch.zeros(size=(1, num_features), dtype=torch.float)
        # Set bound arrays by repeating the bounds for each feature
        lb_values = np.repeat(lb, num_features)
        ub_values = np.repeat(ub, num_features)

        # Set the bounds for each feature
        input_bounds = {}
        for i in range(num_features):
            input_bounds[i] = (float(lb_values[i]), float(ub_values[i]))

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
        objective_weights: list[float],
        verbose: bool
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
        fixed_features: list
            List of features which are fixed.
        verbose: bool
            If True, it prints the optimization process.
        """
        # Assert all the parameters are of the correct type
        assert isinstance(sample, np.ndarray), "The sample must be a numpy array."
        assert type(cf_class) in [
            np.int64,
            np.int32,
            int,
        ], "The cf_class must be an integer."
        assert isinstance(min_probability, float) and (
            min_probability >= 0
        ), "The minimum probability must be a float greater or equal to 0."
        assert isinstance(
            objective_weights, list
        ), "The objective weights must be a list."
        assert isinstance(verbose, bool), "The verbose must be a boolean."

        assert (
            len(objective_weights) == self.SUPPORTED_OBJECTIVES
        ), "The number of objectives is not correct."

        # Verbose print
        vprint = print if verbose else lambda *a, **k: None

        # Define the ranges of the features in a dataframe
        feat_ranges = pd.concat([self.X.min(), self.X.max()], axis=1).rename(
            columns={0: "min", 1: "max"}
        )

        initialize_sample_distance(self.pyo_model, sample, feat_ranges, self.get_property_values("weight", 1))

        # Set the domain and bounds for each feature of the counterfactual
        features_constraints(self.pyo_model, self.feature_props)

        # Set the discrete values for some numerical features
        features_discrete_values(self.pyo_model, self.feature_props)

        # OBJECTIVE 1 - generate the counterfactual with the correct class
        if objective_weights[0] == 0:
            raise ValueError("Objective 1 must be computed.")
        else:
            obj_1 = compute_obj_1_marginal_softmax(
                self.pyo_model, cf_class, self.num_classes, min_probability
            )

        

        # OBJECTIVE 2 - generate counterfactual with limited distances from original features
        if objective_weights[1] == 0:
            vprint(f"Objective 2 is set to 0, so the Gower distance will be 0")
            gower_dist = 1
        else:
            gower_dist = gower_distance(self.pyo_model, sample,)

        # OBJECTIVE 3 -  change the minimum number of features
        if objective_weights[2] == 0:
            obj_3 = 1
            vprint(
                f"Objective 3 is set to 0, so the number of changed features is not minimized."
            )
        else:
            obj_3 = compute_obj_3(self.pyo_model, sample, feat_ranges, self.get_property_values("type", None))

        # Don't change some features
        # limit_counterfactual(self.pyo_model, sample, fixed_features, self.feature_props)

        final_obj = (
            objective_weights[0] * obj_1
            + objective_weights[1] * gower_dist
            + objective_weights[2] * obj_3
        )

        # Set the objective function
        self.pyo_model.obj = pyo.Objective(expr=final_obj, sense=pyo.minimize)

    def generate_counterfactuals(
        self,
        df_sample: pd.DataFrame,
        cf_class: int,
        min_probability: float,
        obj_weights: list[float],
        solver: str,
        solver_options: dict,
        verbose: bool,
    ) -> pd.DataFrame:
        """
        It generates the counterfactual for a given sample, considering the passed
        parameters.

        Parameters:
        -----------
        sample: pd.DataFrame
            Sample to generate the counterfactual, with the label as last column.
        cf_class: int
            Class of the counterfactual the sample should be classified as.
        min_probability: float
            Minimum probability of the softmax output.
        obj_weights: list, optional
            Weights of the objectives.
        solver: str
            The solver to use for computing the solution, one between 'mindtpy' and 'multistart'.
            In the case of 'mindtpy', cplex and ipopt will be used, in the other case only ipopt.
        solver_options: dict, optional
            Options for the solver.
        verbose: bool, optional
            Verbose mode for the solver.

        Returns:
        --------
        pd.DataFrame:
            A dataframe with the counterfactual sample.

        Raises:
        -------
        ValueError:
            If the solver is not supported. Only 'mindtpy' and 'multistart' are supported.
        ValueError:
            If the prediction of the sample is not the same as the cf_class.
        ValueError:
            If the solver does not find a solution.
        ValueError:
            If the sample is equal to a sample of zeros.
        """
        # Assert all the parameters are of the correct type
        assert isinstance(df_sample, pd.DataFrame), "The sample must be a dataframe."
        assert type(cf_class) in [
            np.int64,
            np.int32,
            int,
        ], "The cf_class must be an integer."
        assert isinstance(min_probability, float) and (
            min_probability >= 0
        ), "The minimum probability must be a float greater or equal to 0."
        assert isinstance(obj_weights, list), "The obj_weights must be a list."
        assert sum(obj_weights) <= self.SUPPORTED_OBJECTIVES, f"The sum of the obj_weights must be less than {self.SUPPORTED_OBJECTIVES}."
        assert isinstance(solver, str), "The solver must be a string."
        assert isinstance(
            solver_options, dict
        ), "The solver_options must be a dictionary."
        assert isinstance(verbose, bool), "The verbose must be a boolean."
        # Check the input sample has only one row
        assert df_sample.shape[0] == 1, "Only one sample at a time is supported."

        # Verbose print function
        vprint = print if verbose else lambda *args, **kwargs: None

        sample: np.ndarray = df_sample.values[0]

        # Reset the pyomo model
        self.__build_model()
        # Set the objective function
        self.__compute_objectives(
            sample[:-1],
            cf_class,
            min_probability,
            obj_weights,
            verbose
        )

        # Set the solver to mindtpy
        solver_factory = pyo.SolverFactory(solver)

        # Set the solver options and solve the problem
        start_time = time.time()
        if solver == "mindtpy":
            # TODO replace false with verbose
            try:
                pyo_solution = solver_factory.solve(
                    self.pyo_model,
                    tee=True,
                    time_limit=solver_options["timelimit"],
                    mip_solver=self.AVAILABLE_SOLVERS["mip"],
                    nlp_solver=self.AVAILABLE_SOLVERS["nlp"],
                )
            except ValueError as e:
                raise ValueError(e)

        elif solver == "multistart":
            vprint("\nStarting the search for a counterfactual ...")
            pyo_solution = solver_factory.solve(
                self.pyo_model,
                solver_args={"timelimit": solver_options["timelimit"]},
                suppress_unbounded_warning=True,
                strategy=solver_options["strategy"],
            )
            vprint("The counterfactual has been generated!\n")
        else:
            raise ValueError(
                "The selected solver is not available, choose between 'mindtpy' and 'multistart'."
            )
        end_time = time.time()
        vprint(f"Time elapsed: {end_time - start_time}")
        vprint(f"Solver status: {pyo_solution.solver.status}")

        # Save the pyomo model to a file
        self.pyo_model.pprint(open("./pyomo_model.log", "w"))

        self.start_samples = sample
        # Convert the pyomo solution to a dataframe
        counterfactual_sample = list(self.pyo_model.nn.inputs.get_values().values())

        # Check if some features have changed only by a small amount
        for i, feat in enumerate(self.X.columns):
            if abs(counterfactual_sample[i] - sample[i]) < 1e-3:
                counterfactual_sample[i] = sample[i]

        dummy_sample = np.zeros(len(self.X.columns))
        if np.linalg.norm(np.array(counterfactual_sample) - dummy_sample, ord=1) < 1e-3:
            raise ValueError("The counterfactual is the same as a sample of zeros.")

        counterfactual_df = pd.DataFrame(
            np.array(counterfactual_sample, ndmin=2),
            columns=self.X.columns,
            index=df_sample.index,
        )

        # Inference the counterfactual to check if it is valid
        counterfactual_series = pd.Series(counterfactual_sample, index=self.X.columns)
        y_cf_pred = util_models.evaluate_sample(
            self.model,
            counterfactual_series,
            cf_class,
            verbose=verbose,
        )
        vprint(f"Counterfactual predicted class: {y_cf_pred}")
        if y_cf_pred != cf_class:
            raise ValueError(
                f"Counterfactual predicted class is {y_cf_pred} instead of {cf_class}."
            )
        counterfactual_df["misc_price"] = cf_class

        self.CFs = [counterfactual_df]
        return counterfactual_df
