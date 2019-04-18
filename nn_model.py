
import torch
from torch import nn
from torch.nn import functional as F
# 2 Layers CNN
class CNN_1D_2L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(self.n_in*128 //4, 4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.n_in*128//4)
        return self.linear1(x)


# 3 Layers CNN
class CNN_1D_3L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
                
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, (5,), stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(self.n_in*128 //8, 4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.n_in*128//8)
        return self.linear1(x)
        