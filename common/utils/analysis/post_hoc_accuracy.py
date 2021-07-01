from torch.utils.data import TensorDataset
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from ..meter import AverageMeter
from ..metric import accuracy


def calculate_multidomain_acc(model: nn.Sequential, feature: torch.Tensor, label: torch.Tensor,
              device, progress=True, training_epochs=20):
    """
    
    """
    dataset = TensorDataset(feature, label)
    length = len(dataset)
    train_size = int(0.8 * length)
    val_size = length - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, drop_last=True)

    optimizer = SGD(model.parameters(), lr=0.01)
    best_acc = 0.
    for epoch in range(training_epochs):
        model.train()
        for (x, label) in train_loader:
            x = x.to(device)
            label = label.to(device)
            model.zero_grad()
            y = model(x)
            loss = F.cross_entropy(y, label)
            loss.backward()
            optimizer.step()

        y_true = []
        y_preds = []
        model.eval()
        acc_meter = AverageMeter("accuracy", ":4.2f")
        with torch.no_grad():
            for (x, label) in val_loader:
                x = x.to(device)
                label = label.to(device)
                y_true += label.tolist()
                y = model(x)
                _, y_pred = y.topk(1)
                y_preds.append(y_pred)
                acc = accuracy(y, label)[0]
                acc = max(best_acc, acc)
                acc_meter.update(acc, x.shape[0])
        #error = 1 - meter.avg / 100
        if progress:
            print("epoch {} accuracy: {}".format(epoch, acc_meter.avg)) #, error))
    y_preds = torch.squeeze(torch.cat(y_preds, dim=0)).tolist()
    return acc_meter.avg, y_true, y_preds

