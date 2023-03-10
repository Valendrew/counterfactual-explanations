import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
    def __init__(self, num_feat: int):
        super(BinaryClassification, self).__init__()
        # Number of input features is 9.
        self.layer_1 = nn.Linear(num_feat, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)

        return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    acc = (y_pred_tag == y_test).type(torch.float).sum().item()
    return acc


def train_loop(data_loader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    epoch_acc = 0

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


def train_model(model, train_loader, val_loader, device, LR, EPOCHS):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_acc = train_loop(
            train_loader, model, loss_fn, optimizer, device
        )
        val_loss, val_acc = test_loop(val_loader, model, loss_fn, device)

        if e % 5 == 0:
            print(
                f"Epoch {e+0:03}: | Loss: {train_loss:.5f} | Acc: {train_acc:.3f} | Val loss: {val_loss:.5f} | Acc: {val_acc:.3f}"
            )


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
