import numpy as np
import torch
import pandas as pd

#pyomo for optimization
import pyomo.environ as pyo

#omlt for interfacing our neural network with pyomo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import tempfile
import onnx

import dice_ml


##############################################
# Functions to compute the OMLT objectives
##############################################

def my_softmax(input, n_class, real_class):
    '''
        It returns the probability of the desired class after having computed the
        softmax for the input array.
    '''
    exps = [pyo.exp(input[i]) for i in range(n_class)]
    probs = []
    for exp in exps:
        res = exp/sum(exps)
        probs.append(res)
    return probs[real_class]


def features_constraints(pyo_model, feat_info):
    '''
        Set the bounds and the domain for each features given a dictionary that
        contains the bounds as a tuple, the domain as a pyomo domain and the position
        of the feature in the columns.
    '''
    for feat, info in feat_info.items():
        bounds = info["bounds"]
        domain = info["domain"]
        idx = info["index"]

        pyo_model.nn.inputs[idx].domain = domain
        pyo_model.nn.inputs[idx].bounds = bounds


def compute_obj_1(pyo_model, cf_class, range_prob=0.51):
    ''' 
        It creates the objective function to minimize the distance between the
        predicted and the desired class.

        Parameters:
            - pyo_model:
                The pyomo model to consider.
            - cf_class: int
                The class that the counterfactual should have after the generation.
            - range_prob: float
                An accepted value for the probability of the predicted class.

        Returns:
            It returns the pyomo variable that contains the value to optimize.
    '''
    range_prob = range_prob
    prob_y = lambda x: my_softmax(x, 4, cf_class)

    # something
    pyo_model.q_relu = pyo.Var(within=pyo.Binary)
    # constraints
    pyo_model._z_lower_bound_relu = pyo.Constraint()
    pyo_model._z_lower_bound_zhat_relu = pyo.Constraint()
    pyo_model._z_upper_bound_relu = pyo.Constraint()
    pyo_model._z_upper_bound_zhat_relu = pyo.Constraint()

    # set dummy parameters here to avoid warning message from Pyomo
    pyo_model._big_m_lb_relu = pyo.Param(default=-1e6, mutable=True)
    pyo_model._big_m_ub_relu = pyo.Param(default=1e6, mutable=True)

    # define difference of the output
    lb, ub = (-1, 1)
    pyo_model.diff_prob = pyo.Var(within=pyo.Reals, bounds=(lb, ub), initialize=0)
    pyo_model.diff_prob = range_prob - prob_y(pyo_model.nn.outputs)

    # define variable for max(0, output)
    pyo_model.max_val = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, ub))
    pyo_model._big_m_lb_relu = lb
    pyo_model._big_m_ub_relu = ub

    pyo_model._z_lower_bound_relu = pyo_model.max_val >= 0
    pyo_model._z_lower_bound_zhat_relu = pyo_model.max_val >= pyo_model.diff_prob
    pyo_model._z_upper_bound_relu= pyo_model.max_val <= pyo_model._big_m_ub_relu * pyo_model.q_relu
    pyo_model._z_upper_bound_zhat_relu = pyo_model.max_val <= pyo_model.diff_prob - pyo_model._big_m_lb_relu * (1.0 - pyo_model.q_relu)

    return pyo_model.max_val


def create_cat_constraints_obj_2(pyo_model, bounds, idx_cat, sample):
    '''
        It creates the sum value for the categorical features of the second 
        objective function.

        Parameters:
            - pyo_model
                The model in which the variables and the constraints will be added.
            - bounds: tuple(int)
                The lower and upper value for the constraints.
            - idx_cat: list[int]
                The indexes of the categorical features that we need to compare.
            - sample: np.ndarray
                The values for the categorical features of the original sample 
                for which the counterfactual is generated.
    '''
    L, U = bounds
    # Set of indexes for the features
    feat_set = pyo.Set(initialize=range(0, len(idx_cat)))

    pyo_model.b_o2 = pyo.Var(feat_set, domain=pyo.Binary)
    pyo_model.diff_o2 = pyo.Var(feat_set, domain=pyo.Integers)
    pyo_model.constr_less_o2 = pyo.Constraint(feat_set)
    pyo_model.constr_great_o2 = pyo.Constraint(feat_set)

    cat_dist = 0
    for i, idx in enumerate(idx_cat):
        pyo_model.diff_o2[i] = (sample[i] - pyo_model.nn.inputs[idx])**2

        pyo_model.constr_less_o2[i] = pyo_model.diff_o2[i] >= (pyo_model.b_o2[i]*(-L+1))+L
        # Add a +1 at the end because pyomo needs <= and not <
        pyo_model.constr_great_o2[i] = pyo_model.diff_o2[i] <= (pyo_model.b_o2[i]*(U-1) + 1)+1
        cat_dist += pyo_model.b_o2[i]

    return cat_dist


def gower_distance(x, cat, num, ranges, pyo_model, feat_info):
    '''
        It computes the Gower distance.

        Parameters: 
            - x: np.ndarray
                The array of features of the original sample.
            - cat: list[int]
                The indexes of the categorical features.
            - num: list[int]
                The indexes of the continuous features.
            - ranges: np.ndarray
                The list of ranges for the continuous features.
            - feat_info: dict
                It contains the information about the features to set the bounds
                and domain for each one.
    '''
    features_constraints(pyo_model, feat_info)

    num_dist = 0
    for i, idx in enumerate(num):
        num_dist += (1/ranges[i])*((x[idx]-pyo_model.nn.inputs[idx])**2)
    
    cat_dist = create_cat_constraints_obj_2(pyo_model, (0, 20), cat, x[cat])

    return (cat_dist+num_dist)/len(x)


def compute_obj_3(pyo_model, bounds, n_feat, sample):
    '''
        It creates the third objective function, that limits the number of features
        changed during counterfactual.

        Parameters:
            - pyo_model
                The model in which the variables and the constraints will be added.
            - bounds: tuple[int]
                The bounds to use for the constraints.
            - n_feat: int
                The number of features of the sample.
            - sample: np.ndarray
                The original sample for which the counterfactual is created.
    '''
    L, U = bounds
    # Set of indexes for the features
    feat_set = pyo.Set(initialize=range(0, n_feat))

    pyo_model.b_o3 = pyo.Var(feat_set, domain=pyo.Binary)
    pyo_model.diff_o3 = pyo.Var(feat_set, domain=pyo.Reals)
    pyo_model.constr_less_o3 = pyo.Constraint(feat_set)
    pyo_model.constr_great_o3 = pyo.Constraint(feat_set)

    changed = 0
    for i in range(n_feat):
        pyo_model.diff_o3[i] = (sample[i] - pyo_model.nn.inputs[i])**2

        pyo_model.constr_less_o3[i] = pyo_model.diff_o3[i] >= (pyo_model.b_o3[i]*(-L+1))+L
        # Add a +1 at the end because pyomo needs <= and not <
        pyo_model.constr_great_o3[i] = pyo_model.diff_o3[i] <= (pyo_model.b_o3[i]*(U-1) + 1)+1
        changed += pyo_model.b_o3[i]

    return changed


def limit_counterfactual(pyo_model, sample, features, pyo_info):
    '''
        It sets some constraints to avoid the change of some features during
        counterfactual generation.

        Parameters:
            - pyo_model: 
                The pyomo model to consider.
            - sample: np.ndarray
                The sample for which a counterfactual is generated.
            - features: list[str]
                The features that the model must not change.
            - pyo_info: dict
                The dictionary that contains information about the features.

    '''
    if features is None or len(features) < 1:
        return None

    # Remove the batch dimension if present
    sample = sample.squeeze()

    feat_set = pyo.Set(initialize=range(0, len(sample)))
    pyo_model.lim_constr = pyo.Constraint(feat_set)

    for feat in features:
        idx = pyo_info[feat]['index']
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
        print("WARNING: the desired value was out of range, hence the opposite operation has been performed.")
        return initial_class - counterfactual_op
    return initial_class + counterfactual_op
 

def create_feature_pyomo_info(X_test, num_cols):
    '''
        It creates a dictionary that contains, for each feature in X_test, the
        domain, the bounds and the position of the feature (column index).

        Parameters: 
            - X_test: pd.DataFrame
                The data to consider to compute the information.
            - num_cols: list[str]
                The list of numerical features.

        Returns:
            - dict
                The dictionary with the information for each feature.
    '''
    feat_info = {}
    binary_var = lambda x: 1 if ("has_" in x or "is_" in x) else 0
    
    categorical_features = X_test.columns[~X_test.columns.isin(num_cols)]
    # Categorical features
    for col in categorical_features:
        feat_info[col] = {}
        if not binary_var(col):
            feat_info[col]["domain"] = pyo.Integers
            # Set the 2 extremes as bounds for the feature
            bounds = tuple(np.sort(X_test[col].unique())[[0, -1]])
            feat_info[col]["bounds"] = bounds
        else:
            feat_info[col]["domain"] = pyo.Binary
            feat_info[col]["bounds"] = (0, 1)
        feat_info[col]["index"] = X_test.columns.tolist().index(col)

    return feat_info



class OmltCounterfactual:

    def __init__(self, X_test, y_test, nn_model, num_feat):
        '''
            It is the class used to generate counterfactuals with OMLT.

            Parameters:
                - X_test: pd.DataFrame
                    The dataframe that contains the X data.
                - y_test: pd.Series 
                    The series that contains the correct class for the data.
                - nn_model: 
                    The pytorch neural network that the class will use for the
                    counterfactual generation.
                - num_feat: list[str]
                    The list of continuous features.
        '''
        self.X_test = X_test
        self.y_test = y_test
        self.nn_model = nn_model
        self.num_feat = num_feat
        self.feat_info = create_feature_pyomo_info(self.X_test, self.num_feat)

        self.__create_network_formulation(-1, 1)



    def __create_network_formulation(self, lb, ub):
        '''
            It computes the formulation of the network first converting the model
            to an onnx model and then using pyomo.

            Parameters: 
                - lb: int
                    The value to use as lower bound for the features.
                - ub: int
                    The value to use as upper bound for the features.
        '''
        tmp_sample = self.X_test.values[0].reshape((1, -1))
        dummy_sample = torch.tensor(tmp_sample, dtype=torch.float)

        lb = np.repeat(lb, dummy_sample.size(1))
        ub = np.repeat(ub, dummy_sample.size(1))

        input_bounds = {}
        for i in range(self.X_test.shape[1]):
            input_bounds[i] = (float(lb[i]), float(ub[i]))

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            # Export neural network to ONNX
            torch.onnx.export(
                self.nn_model,
                dummy_sample,
                f,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            # Write ONNX model and its bounds using OMLT
            write_onnx_model_with_bounds(f.name, None, input_bounds)
            # Load the network definition from the ONNX model
            network_definition = load_onnx_neural_network_with_bounds(f.name)

        self.formulation = FullSpaceNNFormulation(network_definition)

    
    def __build_model(self):
        '''
            It actually builds the formulation of the network to use the model
            to solve an optimization problem.
        '''
        # Create a pyomo model
        self.pyo_model = pyo.ConcreteModel()

        # Create an OMLT block for the nn and build the formulation
        self.pyo_model.nn = OmltBlock()
        self.pyo_model.nn.build_formulation(self.formulation)


    def __compute_objectives(self, orig_sample, cf_class, range_prob, bounds_o3=(0, 20),
                             obj_weights=[1,1,1], not_vary=[]):
        '''
            It computes the objective functions to optimize to generate the
            counterfactuals.
            For the parameters explanation check "generate_counterfactuals" function.
        '''
        
        # OBJECTIVE 1
        obj_1 = compute_obj_1(self.pyo_model, cf_class, range_prob)

        # OBJECTIVE 2
        # Dataframes with continuous and categorical features
        num_df = self.X_test.loc[:, self.X_test.columns.isin(self.num_feat)]
        cat_df = self.X_test.loc[:, ~self.X_test.columns.isin(self.num_feat)]

        # We need the index of the features to differentiate in the gower distance
        idx_cont = [self.X_test.columns.get_loc(col) for col in num_df.columns]
        idx_cat = [self.X_test.columns.get_loc(col) for col in cat_df.columns]

        cont_ranges = (num_df.max() - num_df.min()).values
        gower_dist = gower_distance(orig_sample, idx_cat, idx_cont, cont_ranges, 
                                    self.pyo_model, self.feat_info)

        # OBJECTIVE 3
        obj_3 = compute_obj_3(self.pyo_model, (0, 20), 
                                        len(orig_sample), orig_sample)
        
        # Don't change some features
        limit_counterfactual(self.pyo_model, orig_sample, not_vary, self.feat_info)

        if len(obj_weights) != 3:
            print("WARNING: you don't provide the correct number of weights for the objectives, the default value will be used.")
            obj_weights = [1, 1, 1]

        final_obj = obj_weights[0]*obj_1 + obj_weights[1]*gower_dist + obj_weights[2]*obj_3
        self.pyo_model.obj = pyo.Objective(expr=final_obj)

    
    def generate_counterfactuals(self, orig_sample, new_class, solver_path, range_prob,
                                 bounds_o3=(0, 20), obj_weights=[1, 1, 1], not_vary=[],
                                 verbose=True):
        '''
            It generates the counterfactual for the passed sample and it returns 
            it.
            
            Parameters:
                - orig_sample: np.ndarray
                    The sample for which the model has to produce the counterfactuals.
                - new_class: int
                    The class that the generated counterfactual has to belong to.
                - solver_path: str
                    The path to the cplex solver executable.
                - range_prob: int
                    The probability used in the first objective function.
                - bounds_o3: tuple
                    The bounds used in the third objective function.
                - obj_weights: list[float]
                    The list of weights to use for each objective function, by 
                    default all the objectives have the same weight.
                - not_vary: list[str]
                    The list of features that cannot be changed by the optimizer.
                - verbose: bool
                    If true the optimization process will be printed in the console.

            Returns:
                It returns the counterfactual as a pandas dataframe.
        '''
        # Reset the pyomo model
        self.__build_model()
        # Set the objective function
        self.__compute_objectives(orig_sample, new_class, range_prob, bounds_o3,
                                  obj_weights, not_vary)

        pyo_solution = pyo.SolverFactory('cplex', executable=solver_path).solve(self.pyo_model, tee=verbose)
        
        # Convert the dictionary to a list of values
        cf = list(self.pyo_model.nn.inputs.get_values().values())

        return pd.DataFrame(np.array(cf, ndmin=2), columns=self.X_test.columns)



class DiceCounterfactual:
    """
    It's a class that allows you to create Dice counterfactuals, taking as input
    a model, the dataframe with the data, the continuous feature and the target.
    """

    def __init__(
        self, model, backend: str, data: pd.DataFrame, cont_feat: list[str], target: str
    ):
        """
        Parameters:
            - model:
                It's the model to use for counterfactuals.
            - backend: str
                A string between 'sklearn', 'TF1', 'TF2' and 'PYT'.
            - data: pd.DataFrame
                The data used from Dice for statistics.
            - cont_feat: list[str]
                The list of names of continuous features.
            - target: str
                The name of the target feature.
        """
        self.model = dice_ml.Model(model=model, backend=backend)
        self.data = dice_ml.Data(
            dataframe=data, continuous_features=cont_feat, outcome_name=target
        )
        self.cont_feat = cont_feat
        self.target = target
        self.backend = backend
        self.explanation = None
        self.CFs = None

    def create_explanation_instance(self, method: str = "genetic"):
        """
        It generates the Dice explanation instance using the model and the
        data passed during the initialization.

        Parameters:
            - method: str
                The method that will be used during the counterfactual generation.
                A string between 'random', 'kdtree', 'genetic'.
        """
        self.explanation = dice_ml.Dice(self.data, self.model, method=method)

    def generate_counterfactuals(
        self, sample: pd.DataFrame, new_class: int, target: str, n_cf: int = 1, **kwargs
    ):
        """
        It generates the counterfactuals using an explanation instance.

        Parameters:
            - sample: pd.DataFrame
                The dataframe that contains the samples for which the model
                will generate the counterfactuals.
            - new_class: int
                The new label the counterfactuals will be predicted with.
            - target: str
                The name of the target feature, that will be removed from the
                sample before generating the counterfactuals.
            - n_cf: int
                The number of counterfactuals to generate for each sample.
            - **kwargs:
                Additional parameters to pass to the Dice method for generating
                counterfactuals.

        Returns:
            - list[pd.DataFrame]
                It returns a list with a DataFrame that contains the counterfactual
                values for each sample, if n_cf > 1 the dataframe will contain
                n_cf rows.
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
            sample.drop(target, axis=1),
            total_CFs=n_cf,
            desired_class=new_class,
            **kwargs,
        )
        self.CFs = [cf.final_cfs_df.astype(float) for cf in raw_CFs.cf_examples_list]

        return self.CFs

    def destandardize_cfs_orig(self, scaler_num):
        """
        It works on the last generated counterfactuals and the relative
        starting samples, inverting the transform process applied to standardize
        the data.

        Parameters:
            - scaler_num:
                The standard scaler that normalized the continuous features.

        Returns:
            - list
                It returns a list of pairs sample - counterfactuals with
                unstandardized values, in practice are both pd.DataFrame.
        """
        assert self.CFs is not None, "The CFs have not been created yet."

        std_feat = scaler_num.feature_names_in_
        inv_standardize = lambda df: scaler_num.inverse_transform(df)
        # Destandardize the samples for which we create the cfs
        self.start_samples[std_feat] = inv_standardize(self.start_samples[std_feat])

        # Apply to the different dataframes of the counterfactuals
        for cf in self.CFs:
            cf[std_feat] = inv_standardize(cf[std_feat])

        pairs = []
        for i in range(self.start_samples.shape[0]):
            pairs.append((self.start_samples.iloc[[i]], self.CFs[i]))

        return pairs


    def __color_df_diff(self, row, color):
        # left_cell = f"border: 1px solid {color}; border-right: 0px"
        # right_cell = f"border: 1px solid {color}; border-left: 0px"
        # return [left_cell, right_cell] if x[0] - x[1] != 0 else ["", ""]
        cell_border = f"border: 1px solid {color}"
        res = []
        for i in range(1, len(row)):
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
            - pairs: list
                The list of pairs returned by the 'destandardize_cfs_orig'
                function.
            - highlight_diff: bool
                If True, the border of the changed features will be colored.
            - color: str
                The color to use for highlight the differences beteween columns.

        Returns:
            - list
                A list of dataframe which have in each column the values of
                the original sample and the counterfactuals.
        """
        comp_dfs = []
        for i in range(len(pairs)):
            sample, cfs = pairs[i][0].transpose().round(3), pairs[i][
                1
            ].transpose().round(3)
            # Rename the dataframes correctly
            sample.columns = ["Original sample"]
            cfs.columns = [f"Counterfactual_{k}" for k in range(cfs.shape[1])]

            comp_df = pd.concat([sample, cfs], axis=1)
            comp_df = comp_df.style.apply(
                self.__color_df_diff, color=color, axis=1
            ).format(precision=3)
            comp_dfs.append(comp_df)
        return comp_dfs




