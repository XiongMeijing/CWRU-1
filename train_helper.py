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
    '''
        Get dataloaders of the training and validation set.

        Parameter:
            train_ds: Dataset
                Training set
            valid_ds: Dataset
                Validation set
            bs: Int
                Batch size
        
        Return:
            (train_dl, valid_dl): Tuple of DataLoader
                Dataloaders of training and validation set.
    '''
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_func, xb, yb, opt=None):
    '''
        Parameter:
            model: Module
                Your neural network model
            loss_func: Loss
                Loss function, e.g. CrossEntropyLoss()
            xb: Tensor
                One batch of input x
            yb: Tensor
                One batch of true label y
            opt: Optimizer
                Optimizer, e.g. SGD()
        
        Return:
            loss.item(): Python number
                Loss of the current batch
            len(xb): Int
                Number of examples of the current batch
    '''
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, one_cycle=None):
    '''
        Train the NN model and return the model at the final step.
        Lists of the training and validation losses at each epochs are also 
        returned.

        Parameter:
            epochs: int
                Number of epochs to run.
            model: Module
                Your neural network model
            loss_func: Loss
                Loss function, e.g. CrossEntropyLoss()
            opt: Optimizer
                Optimizer, e.g. SGD()
            train_dl: DataLoader
                Dataloader of the training set.
            valid_dl: DataLoader
                Dataloader of the validation set.
            one_cycle: OneCycle
                See one_cycle.py. Object to calculate and update the learning 
                rates and momentums at the end of each training iteration (not 
                epoch) based on the one cycle policy.

        Return:
            model: Module
                Model at the last training step
            train_losses: List
                List of the training loss at each epochs.
            val_losses: List
                List of the validation loss at each epochs.
    '''
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
            
            if one_cycle:
                lr, mom = one_cycle.calc()
                update_lr(opt, lr)
                update_mom(opt, mom)
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
