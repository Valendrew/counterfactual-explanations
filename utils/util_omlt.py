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
            domain = pyo.NonNegativeIntegers

        elif info["type"] == "continuous":
            # Continuous features are encoded as real numbers
            domain = pyo.Reals
        else:
            raise ValueError("The type of the feature is not valid.")
        
        # Set the domain
        pyo_model.nn.inputs[idx].domain = domain
        pyo_model.nn.scaled_inputs[idx].domain = domain
        # Set the bounds
        # pyo_model.nn.inputs[idx].bounds = info["bounds"]
        # pyo_model.nn.scaled_inputs[idx].bounds = info["bounds"]


def features_discrete_values(pyo_model, feature_discrete: list[list]):
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
    assert isinstance(feature_discrete, list), "feature_discrete must be a list"

    num_discrete_values = [0 if f is None else len(f) for f in feature_discrete]
    index_not_none = [i for i, f in enumerate(feature_discrete) if f is not None]

    # Sets
    pyo_model.discrete_features = pyo.Set(initialize=index_not_none)
    pyo_model.discrete_max_values = pyo.RangeSet(0, max(num_discrete_values) - 1)
    pyo_model.discrete_features_values = pyo.Set(initialize=pyo_model.discrete_features * pyo_model.discrete_max_values)
    
    def discrete_feature_values_init(model, i):
        for j in range(num_discrete_values[i]):
            yield j
                    
    pyo_model.discrete_feature_values = pyo.Set(
        pyo_model.discrete_features, initialize=discrete_feature_values_init
    )

    # Set the discrete variables
    pyo_model.discrete_q = pyo.Var(
        pyo_model.discrete_features_values, domain=pyo.Binary, initialize=0
    )

    # Set the constraints
    def discrete_sum_rule(m, i):
        return (
            sum(m.discrete_q[i, j] for j in m.discrete_feature_values[i]) == 1
        )

    pyo_model.discrete_constr_q_sum = pyo.Constraint(
        pyo_model.discrete_features, rule=discrete_sum_rule
    )

    def discrete_input_rule(m, i):
        return m.nn.inputs[i] == sum(
            feature_discrete[i][j] * m.discrete_q[i, j]
            for j in m.discrete_feature_values[i])

    pyo_model.discrete_constr_input = pyo.Constraint(
        pyo_model.discrete_features, rule=discrete_input_rule
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

    l, u = (-min_probability - 1e-2, 3)

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
            == pyo.log(sum([pyo.exp(m.nn.scaled_outputs[i]) for i in range(num_classes)]))
            - m.nn.scaled_outputs[cf_class]
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


def initialize_sample_distance(
    pyo_model, sample: np.ndarray, feat_ranges: pd.DataFrame, feat_weights: list[float], feat_type: list[str]
):
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
    assert isinstance(
        feat_ranges, pd.DataFrame
    ), "feat_ranges must be a pandas dataframe"
    assert isinstance(feat_weights, list), "feat_weights must be a list"
    assert sum(feat_weights) <= len(
        feat_weights
    ), "feat_weights must sum to the length of the features"
    assert isinstance(feat_type, list), "feat_type must be a list"

    # Set of the features
    pyo_model.features_set = pyo.RangeSet(0, len(sample) - 1)

    def feature_ranges_init(m, i):
        return feat_ranges["max"].iloc[i] - feat_ranges["min"].iloc[i]
    
    pyo_model.feature_ranges = pyo.Param(
        pyo_model.features_set, initialize=feature_ranges_init, mutable=False
    )

    # Absolute value of the difference between the sample and the input between the range of the feature
    pyo_model.absolute_binary = pyo.Var(pyo_model.features_set, within=pyo.Binary, initialize=0)

    # i - s <= r * b
    def absolute_constr_1_rule(m, i):
        return m.nn.scaled_inputs[i] - sample[i] <= m.feature_ranges[i] * m.absolute_binary[i]

    pyo_model.constr_absolute_1 = pyo.Constraint(pyo_model.features_set, rule=absolute_constr_1_rule)

    # s - i <= r * (1 - b)
    def absolute_constr_2_rule(m, i):
        return sample[i] - m.nn.scaled_inputs[i] <= m.feature_ranges[i] * (1 - m.absolute_binary[i])
    
    pyo_model.constr_absolute_2 = pyo.Constraint(pyo_model.features_set, rule=absolute_constr_2_rule)

    # Absolute distance between the sample and the input
    def absolute_distance_within(m, i):
        if feat_type[i] == "categorical":
            return pyo.NonNegativeIntegers
        else:
            return pyo.NonNegativeReals

    pyo_model.absolute_distance = pyo.Var(pyo_model.features_set, within=absolute_distance_within, initialize=0)

    # i - s <= d
    def absolute_distance_constr_1_rule(m, i):
        return m.nn.scaled_inputs[i] - sample[i] <= m.absolute_distance[i]
    
    pyo_model.constr_absolute_distance_1 = pyo.Constraint(pyo_model.features_set, rule=absolute_distance_constr_1_rule)

    # s - i <= d
    def absolute_distance_constr_2_rule(m, i):
        return sample[i] - m.nn.scaled_inputs[i] <= m.absolute_distance[i]
    
    pyo_model.constr_absolute_distance_2 = pyo.Constraint(pyo_model.features_set, rule=absolute_distance_constr_2_rule)

    # Absolute distance between the double of the ranges
    # d <= i - s + 2 * r * (1 - b)
    def absolute_bounds_constr_1_rule(m, i):
        return m.absolute_distance[i] <= m.nn.scaled_inputs[i] - sample[i] + 2 * m.feature_ranges[i] * (1 - m.absolute_binary[i])
    
    pyo_model.constr_absolute_bounds_1 = pyo.Constraint(pyo_model.features_set, rule=absolute_bounds_constr_1_rule)

    # d <= s - i + 2 * r * b
    def absolute_bounds_constr_2_rule(m, i):
        return m.absolute_distance[i] <= sample[i] - m.nn.scaled_inputs[i] + 2 * m.feature_ranges[i] * m.absolute_binary[i]
    
    pyo_model.constr_absolute_bounds_2 = pyo.Constraint(pyo_model.features_set, rule=absolute_bounds_constr_2_rule)

    # SAMPLE DISTANCE
    # Define the variable for the distance
    pyo_model.sample_distances = pyo.Var(
        pyo_model.features_set, initialize=0, bounds=(0, 1), domain=pyo.NonNegativeReals
    )

    # Constraint for the distance
    def dist_constr_rule(m, i):
        return (
            m.sample_distances[i]
            == (
                (1 / m.feature_ranges[i])
                * m.absolute_distance[i]
            )
            * feat_weights[i]
        )

    pyo_model.constr_distances = pyo.Constraint(
        pyo_model.features_set, rule=dist_constr_rule
    )


def gower_distance(pyo_model, sample: np.ndarray):
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

    # # Feature variables and constraints
    pyo_model.gower_distance = pyo.Var(
        domain=pyo.NonNegativeReals, bounds=(0, 1), initialize=0
    )

    num_features = len(sample)

    def sum_constr_rule(m):
        return (
            m.gower_distance
            == sum(m.sample_distances[i] for i in pyo_model.features_set) / num_features
        )

    pyo_model.obj2_sum_constr = pyo.Constraint(rule=sum_constr_rule)
    return pyo_model.gower_distance


def compute_obj_3(
    pyo_model, sample: np.ndarray, feat_ranges: pd.DataFrame, feat_props: list[float]
):
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
    # Assert all the parameters are of the correct type
    assert isinstance(pyo_model, pyo.ConcreteModel), "pyo_model must be a pyomo model"
    assert isinstance(sample, np.ndarray), "sample must be a numpy array"
    assert isinstance(
        feat_ranges, pd.DataFrame
    ), "feat_ranges must be a pandas dataframe"
    assert isinstance(feat_props, list), "feat_props must be a list"

    # Lower bound parameter
    pyo_model.obj3_lower_bound = pyo.Param(initialize=1e-5, mutable=False)

    # Variables to handle the values
    pyo_model.b_o3 = pyo.Var(pyo_model.features_set, domain=pyo.Binary, initialize=0)
    # Constraints for the if then else
    # pyo_model.constr_diff_o3 = pyo.Constraint(pyo_model.obj3_feat_set)
    pyo_model.constr_less_o3 = pyo.Constraint(pyo_model.features_set)
    pyo_model.constr_great_o3 = pyo.Constraint(pyo_model.features_set)

    for i in pyo_model.features_set:
        pyo_model.constr_less_o3[i] = (
            pyo_model.sample_distances[i]
            >= pyo_model.b_o3[i] - 1 + pyo_model.obj3_lower_bound
        )
        pyo_model.constr_great_o3[i] = pyo_model.sample_distances[i] <= (
            pyo_model.b_o3[i]
        )

    pyo_model.sum_o3 = pyo.Var(domain=pyo.Reals, bounds=(0, 1), initialize=0)

    def sum_constr_rule(m):
        return m.sum_o3 == sum(m.b_o3[i] for i in pyo_model.features_set) / len(sample)

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
        pyo_model.lim_constr[idx] = pyo_model.nn.scaled_inputs[idx] == sample[idx]


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
        self.__create_network_formulation(-3.0, 3.0)

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
        # lb_values = np.repeat(lb, num_features)
        # ub_values = np.repeat(ub, num_features)
        bounds = self.get_property_values("bounds", None)
        assert all(b is not None for b in bounds), "All the features must have bounds"
        assert len(bounds) == num_features, "The number of bounds must be equal to the number of features"

        # Set the bounds for each feature
        input_bounds = {}
        for i in range(num_features):
            # input_bounds[i] = (float(lb_values[i]), float(ub_values[i]))
            input_bounds[i] = (float(bounds[i][0]), float(bounds[i][1]))

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
        verbose: bool,
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

        # Set the domain and bounds for each feature of the counterfactual
        # TODO evaluate whether to remove
        features_constraints(self.pyo_model, self.feature_props)

        # Set the discrete values for some numerical features
        features_discrete_values(
            self.pyo_model, self.get_property_values("discrete", None)
        )

        # OBJECTIVE 1 - generate the counterfactual with the correct class
        if objective_weights[0] == 0:
            raise ValueError("Objective 1 must be computed.")
        else:
            obj_1 = compute_obj_1_marginal_softmax(
                self.pyo_model, cf_class, self.num_classes, min_probability
            )

        # OBJECTIVE 2 and 3
        # Define the ranges of the features in a dataframe
        feat_ranges = pd.concat([self.X.min(), self.X.max()], axis=1).rename(
            columns={0: "min", 1: "max"}
        )
        if any([(sample < feat_ranges["min"]).any(), (sample > feat_ranges["max"]).any()]):
            raise ValueError("The sample is out of the bounds of the dataset.")
        # bounds_props = self.get_property_values("bounds", None)
        # for i in range(len(bounds_props)):
            # feat_ranges["min"].iloc[i] = max(bounds_props[i][0], feat_ranges["min"].iloc[i])
            # feat_ranges["max"].iloc[i] = min(bounds_props[i][1], feat_ranges["max"].iloc[i])


        if objective_weights[1] > 0 or objective_weights[2] > 0:
            initialize_sample_distance(
                self.pyo_model, sample, feat_ranges, 
                self.get_property_values("weight", 1), self.get_property_values("type", None)
            )

        # OBJECTIVE 2 - generate counterfactual with limited distances from original features
        if objective_weights[1] == 0:
            vprint(f"Objective 2 is set to 0, so the Gower distance will be 0")
            gower_dist = 1
        else:
            gower_dist = gower_distance(
                self.pyo_model,
                sample,
            )

        # OBJECTIVE 3 -  change the minimum number of features
        if objective_weights[2] == 0:
            obj_3 = 1
            vprint(
                f"Objective 3 is set to 0, so the number of changed features is not minimized."
            )
        else:
            obj_3 = compute_obj_3(
                self.pyo_model,
                sample,
                feat_ranges,
                self.get_property_values("type", None),
            )

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
        assert (
            sum(obj_weights) <= self.SUPPORTED_OBJECTIVES
        ), f"The sum of the obj_weights must be less than {self.SUPPORTED_OBJECTIVES}."
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
            sample[:-1], cf_class, min_probability, obj_weights, verbose
        )

        # Set the solver to mindtpy
        solver_factory = pyo.SolverFactory(solver)

        # Set the solver options and solve the problem
        start_time = time.time()
        if solver == "mindtpy":
            try:
                pyo_solution = solver_factory.solve(
                    self.pyo_model,
                    tee=verbose,
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
        counterfactual_sample = list(self.pyo_model.nn.scaled_inputs.get_values().values())

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
