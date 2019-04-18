import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

import numpy as np
from one_cycle import OneCycle, update_lr, update_mom

# Functions for training
def get_dataloader(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    print('EPOCH', '\t', 'Val Loss', '\t', 'Accuracy')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Train
        model.train()
        mean_loss = 0.0
        for xb, yb in train_dl:
            loss, batch_size = loss_batch(model, loss_func, xb, yb, opt)
            mean_loss += loss/batch_size
        train_losses.append(mean_loss)
        
        # Validate
        model.eval()
        mean_loss = 0.0
        predictions = []
        with torch.no_grad():
            for xb, yb in valid_dl: 
                loss, batch_size = loss_batch(model, loss_func, xb, yb)
                mean_loss += loss/batch_size
                predictions.append(torch.argmax(model(xb), dim=1))
                
        val_losses.append(mean_loss)
        predictions = torch.cat(predictions, dim=0).numpy()

        accuracy = np.mean((predictions == valid_dl.dataset.tensors[1].numpy()))

        print(f'{epoch}: \t', f'{mean_loss:.05f}', '\t', f'{accuracy:.05f}')
        
    return model, train_losses, val_losses

def fit_one_cycle(epochs, model, loss_func, opt, train_dl, valid_dl, one_cycle_scheduler):
    print('EPOCH', '\t', 'Val Loss', '\t', 'Accuracy')
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
            lr, mom = one_cycle_scheduler.calc()
            update_lr(opt, lr)
            update_mom(opt, mom)

        model.eval()
        with torch.no_grad():
            loss = [loss_func(model(xb), yb) for xb, yb in valid_dl]
            loss = torch.stack(loss, dim=0).numpy()
            predictions = [torch.argmax(model(xb), dim=1) for xb, yb in valid_dl]
#             set_trace()
            predictions = torch.cat(predictions, dim=0).numpy()
#         set_trace()
        val_loss = np.mean(loss)
        accuracy = np.mean((predictions == valid_dl.dataset.tensors[1].numpy()))

        print(f'{epoch}: \t', f'{val_loss:.05f}', '\t', f'{accuracy:.05f}')
        
    return model