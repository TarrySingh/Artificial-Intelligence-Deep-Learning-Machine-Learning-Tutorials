import numpy as np
import torch
import torch.nn.functional as F
from torch import nn,optim,tensor
from torch.utils.data import TensorDataset, DataLoader


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl: loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses,nums = zip(*[loss_batch(model, loss_func, xb, yb)
                                for xb,yb in valid_dl])
        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(epoch, val_loss)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func=func

    def forward(self, x): return self.func(x)


class WrappedDataLoader():
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self): return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches: yield(self.func(*b))

