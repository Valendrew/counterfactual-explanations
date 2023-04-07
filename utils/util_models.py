import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import pandas as pd
import dice_ml

from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from typing import Callable
import math
import warnings


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        if isinstance(X_data, np.ndarray):
            X_data = torch.as_tensor(X_data, dtype=torch.float32)
        if isinstance(y_data, np.ndarray):
            y_data = torch.as_tensor(y_data, dtype=torch.long)

        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        if isinstance(X_data, np.ndarray):
            X_data = torch.as_tensor(X_data, dtype=torch.float32)
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class NNClassification(nn.Module):
    def __init__(self, hidden_dims, num_feat: int, num_class: int):
        super(NNClassification, self).__init__()
        self.n_hidden_dims = len(hidden_dims)

        self.layer_1 = nn.Linear(num_feat, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(self.n_hidden_dims - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.layer_n = nn.Linear(hidden_dims[-1], num_class)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))

        for i in range(self.n_hidden_dims - 1):
            x = self.relu(self.hidden_layers[i](x))

        x = self.layer_n(x)

        return x

    def reset_weights(self):
        for layer in self.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class LightGBM:
    """
    This class allows to build a lightgbm model using the sklearn api.
    """

    def __init__(self, parameters: dict, train_data: tuple, val_data: tuple):
        """
        Parameters:
            - parameters: dict
                It's the dictionary of parameters to pass to the LighGBM model.
            - train_data: tuple
                The training data to use for the model, a tuple with X_train
                in the first position and y_train in the second one.
            - val_data: tuple
                The validation dataset to use for the training, a tuple with
                X_val in the first position and y_val in the second one.
        """
        self.parameters = parameters
        self.fitted = False

        self.model = lgb.LGBMClassifier(**parameters)
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data

    def train_model(self, verbose: int = -100):
        """
        It trains the model.

        Parameters:
            - verbose: int
                It's the value to make the output of the training verbose or not,
                by default it disables the output.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                verbose=verbose,
            )

            self.fitted = True
            self.train_eval = self.model.evals_result_

    def predict(self, data):
        """
        It predicts the class given some data.

        Parameters:
            - data: pd.DataFrame
                A dataframe in which there are the samples you want
                to predict the class.

        Returns:
            - np.ndarray
                It returns a numpy array with the predicted class for
                each sample.
        """
        assert self.fitted, "You need to train the model beforehand."
        predictions = self.model.predict(data)

        return predictions

    def compute_score(self, score_fn: Callable, data, y_true: list):
        """
        It compute a score between the predicted and the original label.

        Parameters:
            - score_fn: Callable
                A score function that will be used to compute a score.
            - data: pd.DataFrame
                The dataframe that contains the data we want to test.
            - y_true: list
                The array with the true labels for the data.

        Returns:
            - float
                The score computed by the passed function
        """
        y_pred = self.predict(data)
        score = score_fn(y_true, y_pred)

        return score

    def plot_metrics(self, metric: str = "all", **kwargs):
        """
        It plots the metrics computed during the training.

        Parameters:
            - metric: str
                A string between 'all' and the name of a metric used during
                the training. If 'all' selected then the plot for all the
                metrics will be printed.
            - **kwargs
                The parameters to pass to the lgb.plot_metric function.
        """
        assert (
            self.fitted
        ), "You need to train the model with an evaluation set beforehand."

        metrics = self.parameters["metric"]
        if metric == "all":
            for met in metrics:
                lgb.plot_metric(
                    self.train_eval,
                    metric=met,
                    title=f"{met} during training",
                    **kwargs,
                )
        else:
            assert (
                metric in metrics
            ), "The specified metric was not used during training."
            lgb.plot_metric(
                self.train_eval,
                metric=metric,
                title=f"{metric} during training",
                **kwargs,
            )

    def plot_info(self, kind, **kwargs):
        """
        It plots the chart of the tree diagram or the importance
        histogram.

        Parameters:
            - kind: str
                The type of chart to draw, one between 'importance' and 'tree'.
            - **kwargs
                The additional parameters to pass to the lgb.plot_{importance|tree}
                function.
        """
        assert self.fitted, "You need to train the model beforehand."
        if kind == "importance":
            lgb.plot_importance(self.model, **kwargs)
        elif kind == "tree":
            lgb.plot_tree(self.model, **kwargs)
        else:
            print("ERROR: the selected kind of chart is not available.")


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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum().item()

    return acc


def train_loop(data_loader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # forward pass
        y_pred = model(X_batch)
        # compute the loss
        loss = loss_fn(y_pred, y_batch)

        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += multi_acc(y_pred, y_batch)
        # epoch_acc += (y_pred.argmax(1) == y_batch).type(torch.float).sum().item()

    return epoch_loss / len(data_loader), epoch_acc * 100 / len(data_loader.dataset)


def test_loop(data_loader, model, loss_fn, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # inference
            y_pred = model(X_batch)
            # loss
            epoch_loss += loss_fn(y_pred, y_batch).item()
            # accuracy
            epoch_acc += multi_acc(y_pred, y_batch)

    return epoch_loss / len(data_loader), epoch_acc * 100 / len(data_loader.dataset)


def train_model(train_loader, val_loader, model, device, LR, EPOCHS, print_every=2):
    model.reset_weights()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for e in range(1, EPOCHS + 1):
        train_loss, train_acc = train_loop(
            train_loader, model, loss_fn, optimizer, device
        )
        val_loss, val_acc = test_loop(val_loader, model, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if e % print_every == 0:
            print(
                f"Epoch {e+0:03}: | Loss: {train_loss:.5f} | Acc: {train_acc:.3f} | Val loss: {val_loss:.5f} | Acc: {val_acc:.3f}"
            )
    return {"train": train_losses, "val": val_losses}, {
        "train": train_accuracies,
        "val": val_accuracies,
    }


def test_model(data_loader, model, device):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch.to(device)
            # inference
            y_pred = model(X_batch)
            y_prob = torch.log_softmax(y_pred, dim=1)
            _, y_pred_tags = torch.max(y_prob, dim=1)

            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [y.item() for ys in y_pred_list for y in ys]
    return y_pred_list


def kfold_train_model(
    X, y, n_splits, seed, batch_size, model, device, lr, epochs, print_every
):
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_losses, n_accuracies = [], []

    for i, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        train_data = TrainData(X[train_idx], y[train_idx])
        val_data = TrainData(X[val_idx], y[val_idx])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        losses, accuracies = train_model(
            train_loader, val_loader, model, device, lr, epochs, print_every
        )
        n_losses.append(losses)
        n_accuracies.append(accuracies)

        print(f"kfold on group {i+1} accuracy: {accuracies['val'][-1]: .2f}\n")

    return n_losses, n_accuracies


class FixedLinearRegressor(LinearRegression):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.coef_ = np.array([value])
        self.intercept_ = 0
        self.trainable = False


def compare_models(X, y, models, model_names, rng):
    # split the data into training and test sets
    if X.shape[0] > 30:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rng
        )
    else:
        X_test, y_test = X, y

    y_preds = []
    for model, name in zip(models, model_names):
        # fit the model
        if model.__dict__.get("trainable", True):
            try:
                model.fit(X_train, y_train)
            except NameError:
                raise NameError(
                    "X_train and y_train are not defined. Try increasing the number of rows in the dataset."
                )
        # prediction
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)

        # compute the errors
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{name} coefficients: {model.coef_}")
        print(f"{name} RMSE: {math.sqrt(mse):.2f}")
        print(f"{name} r2-score: {r2:.2f}")
        print(f"{name} MAE: {mae:.2f}", end="\n\n")

    return X_test, y_test, y_preds