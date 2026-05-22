import torch


class Algorithm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
