import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.head = nn.Linear(192, 10)

    def forward(self, x, feature=False):
        x = self.encoder(x)
        if feature:
            return self.head(x), x
        else:
            return self.head(x)