"""
Behavior Cloning Neural Network
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

class BCnetwork(nn.Module):
    """
    Behavior Cloning model architecture for Mujoco Pusher
    """
    
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

        self.scalar = 2.0

    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (Tensor): Input to the model

        Returns:
            output (Tensor): Output of the model
        """
        output = self.network(x)
        output = output * self.scalar  # Action space values need to be [-2, 2]
        return output


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCnetwork(input_dim=23, output_dim=7).to(device)

    summary(model, (1, 23))

    # Check that model behaves as expected
    input = torch.linspace(-10, 10, 23).to(device)
    print(input.shape)
    output = model.forward(input)

    print(f"Input size {input.shape} -> Output size {output.shape}")
