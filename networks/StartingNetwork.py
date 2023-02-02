import torch
import torch.nn as nn
class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 800x600x3 images.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(800 * 600 * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
