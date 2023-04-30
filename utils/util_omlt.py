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
    """
    for idx, info in enumerate(feature_props.values()):
        if info["type"] == "categorical":
            pyo_model.nn.inputs[idx].domain = pyo.NonNegativeIntegers
        elif info["type"] == "continuous":
            pyo_model.nn.inputs[idx].domain = pyo.Reals
        else:
            raise ValueError("The type of the feature is not valid.")

        pyo_model.nn.inputs[idx].bounds = info["bounds"]


def features_discrete_values(pyo_model, feature_props: dict):
    discrete_features = [(i, info["discrete"]) for i, info in enumerate(feature_props.values()) if info.get("discrete") is not None]
    sum_features = sum((len(f[1]) for f in discrete_features))

    start_indexes = [0]
    for idx, (_, feat) in enumerate(discrete_features):
        if idx == len(discrete_features) - 1:
            continue
        start_indexes.append(start_indexes[-1] + len(feat))

    print(f"Length: {len(discrete_features)}")
    print(f"Sum of lengths: {sum_features}")
    print(f"Start indexes: {start_indexes}")

    pyo_model.n_discrete_set = pyo.RangeSet(0, sum_features - 1)
    pyo_model.discrete_set = pyo.RangeSet(0, len(discrete_features) - 1)

    pyo_model.discrete_q = pyo.Var(pyo_model.n_discrete_set, domain=pyo.Binary)


    def discrete_sum_rule(model, i):
        return sum(model.discrete_q[j + start_indexes[i]] for j in range(len(discrete_features[i][1]))) == 1

    pyo_model.discrete_constr_sum = pyo.Constraint(pyo_model.discrete_set, rule=discrete_sum_rule)

    def discrete_input_rule(model, i):
        return model.nn.inputs[discrete_features[i][0]] == sum(
            discrete_features[i][1][j] * model.discrete_q[j + start_indexes[i]]
            for j in range(len(discrete_features[i][1]))
        )

    pyo_model.discrete_constr_input = pyo.Constraint(pyo_model.discrete_set, rule=discrete_input_rule)


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
    pyo_model, x: np.ndarray, feat_ranges: pd.DataFrame, feat_props: dict
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
    bounds: tuple(pd.Series, pd.Series)
        The series of minimum and maximum values for each feature.

    Returns
    -------
    It returns the pyomo variable that contains the sum to minimize.
    """
    ub_ranges = 0  # Upper bound of the range
    sum_distances = 0  # Sum of the distances
    for i, props in enumerate(feat_props.values()):
        range_i = (feat_ranges["max"].iloc[i] - feat_ranges["min"].iloc[i]) ** 2
        weights_i = props.get("weight", 1)

        ub_ranges += range_i
        # TODO replace euclidean distance with absolute distance
        if props["type"] == "categorical":
            # TODO should we use the same distance for categorical features?
            sum_distances += (
                (1 / range_i) * ((x[i] - pyo_model.nn.inputs[i]) ** 2) * weights_i
            )
        elif props["type"] == "continuous":
            sum_distances += (
                (1 / range_i) * ((x[i] - pyo_model.nn.inputs[i]) ** 2) * weights_i
            )
        else:
            raise ValueError("The type of feature is not valid.")

    # Feature variables and constraints
    pyo_model.obj2_sum = pyo.Var(domain=pyo.Reals, bounds=(0, ub_ranges), initialize=0)
    pyo_model.obj2_sum_constr = pyo.Constraint(expr=pyo_model.obj2_sum == sum_distances)
    return pyo_model.obj2_sum / len(x)


def compute_obj_3(pyo_model, sample: np.ndarray, feat_ranges: pd.DataFrame):
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
    num_feat = len(sample)
    # Set of indexes for the features
    feat_set = pyo.Set(initialize=range(0, num_feat))
    # Variables to handle the values
    pyo_model.b_o3 = pyo.Var(feat_set, domain=pyo.Binary, initialize=0)
    pyo_model.sum_o3 = pyo.Var(
        domain=pyo.NonNegativeIntegers, bounds=(0, 1), initialize=0
    )
    pyo_model.diff_o3 = pyo.Var(feat_set, domain=pyo.NonNegativeReals, initialize=0)
    # Constraints for the if then else
    pyo_model.constr_diff_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_less_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_great_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_sum_o3 = pyo.Constraint()

    for i in range(num_feat):
        range_i = (feat_ranges["max"].iloc[i] - feat_ranges["min"].iloc[i]) ** 2
        threshold = 1e-3

        pyo_model.constr_diff_o3[i] = (
            pyo_model.diff_o3[i] == (sample[i] - pyo_model.nn.inputs[i]) ** 2
        )

        pyo_model.constr_less_o3[i] = (
            pyo_model.diff_o3[i] >= pyo_model.b_o3[i] - 1 + threshold
        )
        pyo_model.constr_great_o3[i] = pyo_model.diff_o3[i] <= (
            pyo_model.b_o3[i] * range_i
        )
    pyo_model.constr_sum_o3 = pyo_model.sum_o3 == sum(
        [pyo_model.b_o3[i] for i in range(num_feat)]
    )
    return pyo_model.sum_o3 / num_feat


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
        super().__init__(nn_model, X, y, feature_props)

        self.num_classes = self.y.nunique()
        self.SUPPORTED_OBJECTIVES = 3
        self.AVAILABLE_SOLVERS = {"mip": "cplex", "nlp": "ipopt"}

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
        objective_weights=[0, 0, 0],
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
        features_constraints(self.pyo_model, self.feature_props)

        features_discrete_values(self.pyo_model, self.feature_props)

        # OBJECTIVE 1 - generate the counterfactual with the correct class
        if objective_weights[0] == 0:
            raise Exception("Objective 1 must be computed.")
        else:
            obj_1 = compute_obj_1_marginal_softmax(
                self.pyo_model, cf_class, self.num_classes, min_probability
            )

        # OBJECTIVE 2 - generate counterfactual with limited distances from original features
        feat_ranges = pd.concat([self.X.min(), self.X.max()], axis=1).rename(
            columns={0: "min", 1: "max"}
        )

        if objective_weights[1] == 0:
            print(f"Objective 2 is set to 0, so the Gower distance will be 0")
            gower_dist = 1
        else:
            gower_dist = gower_distance(
                self.pyo_model, sample, feat_ranges, self.feature_props
            )

        # OBJECTIVE 3 -  change the minimum number of features
        if objective_weights[2] == 0:
            obj_3 = 1
            print(
                f"Objective 3 is set to 0, so the number of changed features is not minimized."
            )
        else:
            obj_3 = compute_obj_3(self.pyo_model, sample, feat_ranges)

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
        obj_weights=[1, 1, 1],
        solver="mindtpy",
        solver_options={},
        verbose=True,
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
            Weights of the objectives. Defaults to [1, 1, 1].
        solver: str
            The solver to use for computing the solution, one between 'mindtpy' and 'multistart'.
            In the case of 'mindtpy', cplex and ipopt will be used, in the other case only ipopt.
        solver_options: dict, optional
            Options for the solver. Defaults to {}.
        verbose: bool, optional
            Verbose mode for the solver. Defaults to True.

        Returns:
        --------
        pd.DataFrame:
            A dataframe with the counterfactual sample.
        
        Raises:
        -------
        AssertionError:
            If the sample is not a dataframe or if it contains more than one sample.

        AssertionError:
            If the solver is not supported.
s
        ValueError:
            If the prediction of the sample is not the same as the cf_class.        
        """
        # Check the input
        assert isinstance(df_sample, pd.DataFrame), "The sample must be a dataframe."
        assert df_sample.shape[0] == 1, "Only one sample at a time is supported."
        sample: np.ndarray = df_sample.values[0]

        # Reset the pyomo model
        self.__build_model()
        # Set the objective function
        self.__compute_objectives(
            sample[:-1],
            cf_class,
            min_probability,
            obj_weights,
        )

        assert solver in [
            "mindtpy",
            "multistart",
        ], "The selected solver is not available, choose between 'mindtpy' and 'multistart'."
        # Set the solver to mindtpy
        solver_factory = pyo.SolverFactory(solver)

        vprint = print if verbose else lambda *args, **kwargs: None

        if solver == "mindtpy":
            pyo_solution = solver_factory.solve(
                self.pyo_model,
                tee=verbose,
                time_limit=solver_options["timelimit"],
                mip_solver=self.AVAILABLE_SOLVERS["mip"],
                nlp_solver=self.AVAILABLE_SOLVERS["nlp"],
            )
        else:
            vprint("\nStarting the search for a counterfactual ...")
            pyo_solution = solver_factory.solve(
                self.pyo_model,
                solver_args={"timelimit": solver_options["timelimit"]},
                suppress_unbounded_warning=True,
                strategy=solver_options["strategy"],
            )
            vprint("The counterfactual has been generated!\n")

        # Check the value of the objective function
        self.pyo_model.pprint(open("./pyomo_model.log", "w"))

        self.start_samples = sample
        # Convert the pyomo solution to a dataframe
        counterfactual_sample = list(self.pyo_model.nn.inputs.get_values().values())
        
        vprint(f"Counterfactual sample: {counterfactual_sample}")
        # Check if some features have changed only by a small amount
        for i, feat in enumerate(self.X.columns):
            if abs(counterfactual_sample[i] - sample[i]) < 1e-3:
                counterfactual_sample[i] = sample[i]

        counterfactual_df = pd.DataFrame(
            np.array(counterfactual_sample, ndmin=2),
            columns=self.X.columns,
            index=df_sample.index,
        )
        # Find the predicted label for the counterfactual
        logit_dict = self.pyo_model.nn.outputs.get_values()
        out_label = max(logit_dict, key=logit_dict.get)
        counterfactual_df["misc_price"] = out_label

        # Inference the counterfactual
        y_cf_pred = util_models.evaluate_sample(
            self.model, pd.Series(counterfactual_sample, index=self.X.columns), cf_class, verbose=verbose
        )
        vprint(f"Counterfactual predicted class: {y_cf_pred}")
        if y_cf_pred != cf_class:
            raise ValueError(
                f"Counterfactual predicted class is {y_cf_pred} instead of {cf_class}."
            )

        self.CFs = [counterfactual_df]
        return counterfactual_df
