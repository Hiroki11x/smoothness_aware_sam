# define mlp for MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_Mnist(nn.Module):
    def __init__(self, num_classes):
        super(MLP_Mnist, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out