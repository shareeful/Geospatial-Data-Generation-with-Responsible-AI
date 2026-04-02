import torch
import torch.nn as nn
from typing import List


class Generator(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int, output_dim: int,
                 hidden_dims: List[int], leaky_relu_alpha: float = 0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        input_dim = latent_dim + n_classes
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(leaky_relu_alpha))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(labels.size(0), self.n_classes, device=z.device)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
        x = torch.cat([z, one_hot], dim=1)
        return self.model(x)
