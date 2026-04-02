import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from .generator import Generator
from .discriminator import Discriminator


class CGAN:
    def __init__(self, latent_dim: int, n_classes: int, feature_dim: int,
                 generator_hidden: list, discriminator_hidden: list,
                 leaky_relu_alpha: float = 0.2, dropout_rate: float = 0.3,
                 device: str = "cpu"):
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.device = torch.device(device)

        self.generator = Generator(
            latent_dim=latent_dim,
            n_classes=n_classes,
            output_dim=feature_dim,
            hidden_dims=generator_hidden,
            leaky_relu_alpha=leaky_relu_alpha,
        ).to(self.device)

        self.discriminator = Discriminator(
            input_dim=feature_dim,
            n_classes=n_classes,
            hidden_dims=discriminator_hidden,
            leaky_relu_alpha=leaky_relu_alpha,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def generate(self, n_samples: int, class_label: int) -> np.ndarray:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            labels = torch.full((n_samples,), class_label, dtype=torch.long, device=self.device)
            synthetic = self.generator(z, labels)
        return synthetic.cpu().numpy()

    def generate_balanced(self, n_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
        all_samples = []
        all_labels = []

        for cls in range(self.n_classes):
            samples = self.generate(n_per_class, cls)
            all_samples.append(samples)
            all_labels.append(np.full(n_per_class, cls))

        X = np.vstack(all_samples)
        y = np.concatenate(all_labels)
        shuffle_idx = np.random.permutation(len(X))
        return X[shuffle_idx], y[shuffle_idx]

    def save(self, path: str):
        torch.save({
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "config": {
                "latent_dim": self.latent_dim,
                "n_classes": self.n_classes,
                "feature_dim": self.feature_dim,
            },
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])
