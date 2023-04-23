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


def multi_acc(y_pred, y_test=None):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    if y_test is None:
        return y_pred_tags

    return (y_pred_tags == y_test).sum().float()


def binary_accuracy(y_pred, y_test=None):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    if y_test is None:
        return y_pred_tag

    return (y_pred_tag == y_test).sum().float()


def compute_inverse_class_frequency(y, device):
    class_labels, class_frequency = np.unique(y, return_counts=True)
    inverse_class_frequency = 1 - (class_frequency / class_frequency.sum())
    return torch.tensor(inverse_class_frequency, dtype=torch.float).to(device)


class TrainTestNetwork:
    def __init__(self, model, metric_fn, device, seed):
        self.model = model
        self.metric_fn = metric_fn
        self.device = device
        self.state_rng = np.random.RandomState(seed)
        self.torch_rng = torch.manual_seed(seed)

    def __train_loop(self, train_loader, loss_fn, optimizer):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            # forward pass
            y_pred = self.model(X_batch)
            # compute the loss
            loss = loss_fn(y_pred, y_batch)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += self.metric_fn(y_pred, y_batch)

        loss_val = epoch_loss / len(train_loader)
        accuracy_val = epoch_acc * 100 / len(train_loader.dataset)
        return loss_val, accuracy_val

    def __test_loop(self, test_loader, loss_fn):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                # inference
                y_pred = self.model(X_batch)
                # loss
                epoch_loss += loss_fn(y_pred, y_batch).item()
                # accuracy
                epoch_acc += self.metric_fn(y_pred, y_batch)

        loss_val = epoch_loss / len(test_loader)
        accuracy_val = epoch_acc * 100 / len(test_loader.dataset)
        return loss_val, accuracy_val

    def train_model(
        self,
        train_data,
        val_data,
        epochs,
        batch_size=64,
        lr=0.001,
        print_every=2,
        reset_weights=True,
        ce_weights=None,
        reduce_lr=False,
        max_accuracy=0,
        name_model="best_model.pt",
    ):
        # create the data loaders for the training and validation sets
        train_data = TrainData(train_data[0], train_data[1])
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=self.torch_rng
        )
        val_data = TrainData(val_data[0], val_data[1])
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False, generator=self.torch_rng
        )
        # reset the weights of the model
        if reset_weights:
            self.model.reset_weights()

        # set the loss function and the optimizer
        loss_fn = torch.nn.CrossEntropyLoss(weight=ce_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, threshold=1e-2, factor=0.5, verbose=True, mode="max"
        )

        # list to store the losses and accuracies
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        # train the model for the specified number of epochs
        for e in range(1, epochs + 1):
            train_loss, train_acc = self.__train_loop(train_loader, loss_fn, optimizer)
            val_loss, val_acc = self.__test_loop(val_loader, loss_fn)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if val_acc > max_accuracy:
                max_accuracy = val_acc
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "epoch": e,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    name_model,
                )
                print(f"Model saved with accuracy: {val_acc:.3f}")

            if reduce_lr:
                scheduler.step(val_acc)

            if e % print_every == 0:
                print(
                    f"Epoch {e+0:03}: | Loss: {train_loss:.5f} | Acc: {train_acc:.3f} | Val loss: {val_loss:.5f} | Acc: {val_acc:.3f}"
                )

        # return the losses and accuracies
        return {"train": train_losses, "val": val_losses}, {
            "train": train_accuracies,
            "val": val_accuracies,
        }

    def test_model(self, X_test, batch_size=64):
        test_data = TestData(X_test)
        test_loader = DataLoader(
            dataset=test_data, batch_size=batch_size, generator=self.torch_rng
        )
        y_pred_list = []

        self.model.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)

                # inference
                y_pred = self.model(X_batch)
                y_pred_tags = self.metric_fn(y_pred)

                y_pred_list.append(y_pred_tags.cpu().numpy())

        y_pred_list = [y for ys in y_pred_list for y in ys]
        return y_pred_list

    def kfold_train_model(self, X, y, n_splits, **kwargs):
        skfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.state_rng
        )
        n_losses, n_accuracies = [], []

        for i, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
            train_data = (X[train_idx], y[train_idx])
            val_data = (X[val_idx], y[val_idx])

            losses, accuracies = self.train_model(train_data, val_data, **kwargs)

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


def evaluate_sample(model, sample_idx, y_idx, device=torch.device("cpu"), verbose=True):
    '''
    It evaluates a given sample, running the model on it and computing
    the logits, the softmax, the predicted class and the marginal softmax.
    I returns a warning if sample_idx predicted label is different from y_idx.
    '''
    vprint = print if verbose else lambda *args, **kwargs: None
    model.eval()
    with torch.no_grad():
        model = model.to(device)
        if isinstance(sample_idx, pd.DataFrame): 
            sample_idx = torch.tensor(sample_idx.values, dtype=torch.float).view(1, -1).to(device)
        else:
            sample_idx = torch.tensor(sample_idx, dtype=torch.float).view(1, -1).to(device)

        # inference and print logits
        y_logit = model(sample_idx)
        vprint(f"Logits: {y_logit.squeeze()}")
        
        # compute softmax
        y_prob = torch.softmax(y_logit, dim=1)
        vprint(f"Softmax: {y_prob.squeeze()}")

        # print softmax of predicted class
        y_pred = torch.argmax(y_prob, dim=1)
        vprint(f"Predicted class {y_pred.item()} with probability: {y_prob.squeeze()[y_pred].item():.3f} and logit: {y_logit.squeeze()[y_pred].item():.3f}")
        
        # marginlal softmax
        marginal_softmax = torch.log(torch.sum(torch.exp(y_logit))) - y_logit.squeeze()[y_pred].item()
        vprint(f"Marginal softmax: {marginal_softmax:.3f}\n")

        if y_pred.item() != y_idx:
            vprint("WARNING: the predicted label for the sample is different from the groundtruth.")
        else:
            vprint("The predicted class for the sample is equal to the groundtruth.")
        
    return y_pred.item()