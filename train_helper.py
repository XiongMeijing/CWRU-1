import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
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
            pred: ndarray
                Predictions (class with highest probability) of the minibatch 
                input xb
    '''
    out = model(xb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, one_cycle=None, train_metric=False):
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
            train_metric: Bool
                Default is False. If False, the train loss and accuracy will be
                set to 0.
                If True, the loss and accuracy of the train set will also be 
                computed.

        Return:
            model: Module
                Trained model.
            metrics: DataFrame
                DataFrame which contains the train and validation loss at each
                epoch.
    '''
    print(
        'EPOCH', '\t', 
        'Train Loss', '\t',
        'Val Loss', '\t', 
        'Train Acc', '\t',
        'Val Acc', '\t')
    # Initialize dic to store metrics for each epoch.
    metrics_dic = {}
    metrics_dic['train_loss'] = []
    metrics_dic['train_accuracy'] = []
    metrics_dic['val_loss'] = []
    metrics_dic['val_accuracy'] = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_examples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, batch_size, pred = loss_batch(model, loss_func, xb, yb, opt)
            if train_metric == False:
                train_loss += loss
                num_examples += batch_size
            
            if one_cycle:
                lr, mom = one_cycle.calc()
                update_lr(opt, lr)
                update_mom(opt, mom)

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, _ = validate(model, valid_dl, loss_func)
            if train_metric:
                train_loss, train_accuracy, _ = validate(model, train_dl, loss_func)
            else:
                train_loss = train_loss / num_examples

        metrics_dic['val_loss'].append(val_loss)
        metrics_dic['val_accuracy'].append(val_accuracy)
        metrics_dic['train_loss'].append(train_loss)
        metrics_dic['train_accuracy'].append(train_accuracy)
        
        print(
            f'{epoch} \t', 
            f'{train_loss:.05f}', '\t',
            f'{val_loss:.05f}', '\t', 
            f'{train_accuracy:.05f}', '\t'
            f'{val_accuracy:.05f}', '\t')
        
    metrics = pd.DataFrame.from_dict(metrics_dic)

    return model, metrics

def validate(model, dl, loss_func):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for xb, yb in dl: 
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_size, pred = loss_batch(model, loss_func, xb, yb)
        total_loss += loss*batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(yb.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)