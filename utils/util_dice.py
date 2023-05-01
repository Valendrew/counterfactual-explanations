# 3rd party imports
import pandas as pd
import dice_ml

# user imports
from utils.util_base_cf import BaseCounterfactual


class DiceCounterfactual(BaseCounterfactual):
    """
    It's a class that allows you to create Dice counterfactuals, taking as input
    a model, the dataframe with the data, the continuous feature and the target feature.
    """

    def __init__(
        self, model, backend: str, X: pd.DataFrame, y: pd.DataFrame, feature_props: dict, target: str
    ):
        """
        Parameters:
        -----------
        model:
            It's the model to use for counterfactuals (torch, sklearn or tensorflow).
        backend: str
            A string between 'sklearn', 'TF1', 'TF2' and 'PYT'.
        X: pd.DataFrame
            The dataframe with the data.
        y: pd.DataFrame
            The dataframe with the target.
        feature_props: dict
            The dictionary with the feature properties.
        target: str
            The name of the target feature.
        """
        dice_mod = dice_ml.Model(model=model, backend=backend)
        super().__init__(dice_mod, X, y, feature_props)

        self.data = dice_ml.Data(
            dataframe=pd.concat([X, y], axis=1), continuous_features=X.columns.to_list(), outcome_name=target
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
        self.dice_method = method

    def generate_counterfactuals(
        self,
        sample: pd.DataFrame,
        new_class: int,
        target: str,
        n_cf: int = 1,
        proximity_weight: float = 0.4,
        sparsity_weight: float = 0.7,
        stopping_threshold: float = 0.5,
        features_to_vary="all",
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

        feature_weights = {feat: info['weight'] for feat, info in self.feature_props.items()}
        # features_to_vary = [feat for feat, info in self.feature_props.items() if info['fixed']]
        # If all the weights are equal
        if len(set(feature_weights.values())) == 1:
            feature_weights="inverse_mad"

        try:
            if self.dice_method == "genetic":
                    raw_CFs = self.explanation.generate_counterfactuals(
                        sample.drop(target, axis=1),
                        total_CFs=n_cf,
                        desired_class=new_class,
                        proximity_weight=proximity_weight,
                        sparsity_weight=sparsity_weight,
                        stopping_threshold=stopping_threshold,
                        feature_weights=feature_weights,
                        features_to_vary=features_to_vary,
                    )
            else:
                # Random method doesn't support some parameters
                raw_CFs = self.explanation.generate_counterfactuals(
                    sample.drop(target, axis=1),
                    total_CFs=n_cf,
                    desired_class=new_class,
                    stopping_threshold=stopping_threshold,
                    features_to_vary=features_to_vary,
                )
        except dice_ml.model.UserConfigValidationException as e:
            print(f"Counterfactual not found for the sample {sample.index}.")
            raise(e)

        self.CFs = [cf.final_cfs_df.astype(float) for cf in raw_CFs.cf_examples_list]

        return self.CFs
