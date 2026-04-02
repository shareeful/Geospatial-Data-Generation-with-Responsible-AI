import torch
import torch.nn as nn
from typing import List


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims: List[int], leaky_relu_alpha: float = 0.2,
                 dropout_rate: float = 0.3):
        super().__init__()
        self.n_classes = n_classes

        full_input_dim = input_dim + n_classes
        layers = []
        prev_dim = full_input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(leaky_relu_alpha))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(labels.size(0), self.n_classes, device=x.device)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
        combined = torch.cat([x, one_hot], dim=1)
        return self.model(combined)
