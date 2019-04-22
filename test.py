## Imports
# Data science libraries
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Pytorch
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss

# Others
from IPython.core.debugger import set_trace
from pathlib import Path

from helper import *
from train_helper import *
import nn_model
from one_cycle import OneCycle, update_lr, update_mom

working_dir = Path()
save_model_path = working_dir / 'Model'
normal_path = working_dir / 'Data' / 'Normal'
DE_path = working_dir / 'Data' / '12k_DE'

if __name__ == "__main__":
    df_all = get_df_all(normal_path, DE_path, segment_length=500, normalize=True)
    features = df_all.columns[2:]
    target = 'label'
    subsample_size = 1000#len(df_all)
    subsample_idx = np.random.permutation(len(df_all))[:subsample_size]
    X_train, X_valid, y_train, y_valid = train_test_split(df_all[features].iloc[subsample_idx], 
                                                        df_all[target].iloc[subsample_idx], 
                                                        test_size=0.20, 
                                                        random_state=42, 
                                                        shuffle=True
                                                        )

    lr = 0.01
    bs = 64
    wd = 1e-5
    epochs = 3
    loss_func = CrossEntropyLoss()
    onecycle = OneCycle(int(len(X_train) * epochs / bs), lr, prcnt=10, div=25, momentum_vals=(0.95, 0.8))


    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_valid = torch.tensor(y_valid.values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    train_dl, valid_dl = get_dataloader(train_ds, valid_ds, bs)

    model = nn_model.CNN_1D_2L(len(features))
    opt = optim.Adam(model.parameters(), lr=lr/10, betas=(0.9, 0.999), weight_decay=wd)
    model = fit(epochs, model, loss_func, opt, train_dl, valid_dl, one_cycle=onecycle, train_metric=True)