import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from hessian import HessianCalculator  # Assuming HessianCalculator is saved in a separate module

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Dummy dataset
def create_dummy_data(batch_size, sample_size):
    # Create dummy inputs and labels for a binary classification problem
    inputs = torch.randn(sample_size, 3, 32, 32)  # Example shape for an image input
    labels = torch.randint(0, 2, (sample_size,))
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    # Model and config setup
    model = SimpleModel().cuda()
    config = {
        'pyhessian_batch_size_per_gpu': 10,
        'pyhessian_sample_size': 100,
        'pyhessian_top_n': 5,
        'calc_gradnorm': False
    }

    # DataLoader
    dataloader = create_dummy_data(10, 100)

    # HessianCalculator instantiation
    hessian_calculator = HessianCalculator(model, config)

    # Calculate Hessian metrics for debugging
    trace_h, eigenvalues = hessian_calculator.calc_hessian_from_pyhessian(
        model, dataloader, config['pyhessian_batch_size_per_gpu'], 
        config['pyhessian_sample_size'], config['pyhessian_top_n']
    )
    print("Trace of the Hessian:", trace_h)
    print("Top eigenvalues of the Hessian:", eigenvalues)

if __name__ == "__main__":
    main()
