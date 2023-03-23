import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        if isinstance(X_data, np.ndarray):
            X_data = torch.as_tensor(X_data, dtype=torch.float32)
        if isinstance(y_data, np.ndarray):
            y_data = torch.as_tensor(y_data, dtype=torch.float32)

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


class BinaryClassification(nn.Module):
    def __init__(self, hidden_dims, num_feat: int):
        super(BinaryClassification, self).__init__()
        self.n_hidden_dims = len(hidden_dims)

        self.layer_1 = nn.Linear(num_feat, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(self.n_hidden_dims - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.layer_n = nn.Linear(hidden_dims[-1], 1)

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


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    acc = (y_pred_tag == y_test).type(torch.float).sum().item()
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
        epoch_acc += binary_acc(y_pred, y_batch)
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
            epoch_acc += binary_acc(y_pred, y_batch)

    return epoch_loss / len(data_loader), epoch_acc * 100 / len(data_loader.dataset)


def train_model(train_loader, val_loader, model, device, LR, EPOCHS):
    model.reset_weights()

    loss_fn = torch.nn.BCEWithLogitsLoss()
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

        if e % 2 == 0:
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
            y_prob = torch.sigmoid(y_pred)
            y_tag = torch.round(y_prob)

            y_pred_list.append(y_tag.cpu().numpy())

    y_pred_list = [y for ys in y_pred_list for y in ys]
    return y_pred_list


def kfold_train_model(X, y, n_splits, seed, batch_size, model, device, lr, epochs):
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_losses, n_accuracies = [], []

    for i, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        train_data = TrainData(X[train_idx], y[train_idx])
        val_data = TrainData(X[val_idx], y[val_idx])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        losses, accuracies = train_model(
            train_loader, val_loader, model, device, lr, epochs
        )
        n_losses.append(losses)
        n_accuracies.append(accuracies)

        print(f"kfold on group {i+1} accuracy: {accuracies['val'][-1]: .2f}\n")

    return n_losses, n_accuracies