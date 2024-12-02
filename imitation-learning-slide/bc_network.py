"""
Behavior Cloning Neural Network
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

class BCnetworkPanda(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(input_dim, 256)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(256, 128)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(128, output_dim)),
                    ("tanh", nn.Tanh()),  # Squish between [-1, 1]
                ]
            )
        )

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCnetworkPanda(input_dim=24, output_dim=3).to(device)

    summary(model, (1, 24))

    # Check that model behaves as expected
    input = torch.linspace(-10, 10, 24).to(device)
    print(input.shape)
    output = model.forward(input)

    print(f"Input size {input.shape} -> Output size {output.shape}")
