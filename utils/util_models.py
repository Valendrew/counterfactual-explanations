import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import pandas as pd

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
