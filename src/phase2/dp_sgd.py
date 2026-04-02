import torch
import numpy as np
from typing import Tuple

from ..utils.logging_utils import setup_logger

logger = setup_logger("dp_sgd")


class DPSGDEngine:
    def __init__(self, max_grad_norm: float = 1.0, noise_multiplier: float = 1.1,
                 target_epsilon: float = 5.0, target_delta: float = 1e-5,
                 sample_rate: float = 0.01):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.sample_rate = sample_rate
        self.steps = 0
        self.rdp_orders = np.arange(2, 128, 0.5)
        self.rdp_budget = np.zeros_like(self.rdp_orders)

    def clip_gradients(self, model: torch.nn.Module) -> float:
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return total_norm

    def add_noise(self, model: torch.nn.Module):
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.grad.device,
                )
                param.grad.data.add_(noise)

    def step(self, model: torch.nn.Module) -> float:
        grad_norm = self.clip_gradients(model)
        self.add_noise(model)
        self.steps += 1
        self._accumulate_rdp()
        return grad_norm

    def _compute_rdp_single_step(self, order: float) -> float:
        q = self.sample_rate
        sigma = self.noise_multiplier

        if q == 0:
            return 0.0

        if order == 1:
            return q * q / (2 * sigma * sigma)

        if sigma == 0:
            return float("inf")

        log_term = (order * q * q) / (2 * sigma * sigma)
        return min(log_term, order * q * q / (2 * sigma * sigma) + np.log(1 + q * (np.exp((order - 1) / (sigma * sigma)) - 1)) / (order - 1))

    def _accumulate_rdp(self):
        for idx, order in enumerate(self.rdp_orders):
            self.rdp_budget[idx] += self._compute_rdp_single_step(order)

    def get_epsilon(self) -> float:
        eps_candidates = self.rdp_budget - np.log(self.target_delta) / (self.rdp_orders - 1)
        return float(np.min(eps_candidates))

    def check_budget(self) -> Tuple[bool, float]:
        current_eps = self.get_epsilon()
        exceeded = current_eps > self.target_epsilon
        return not exceeded, current_eps

    def get_privacy_guarantee(self) -> Tuple[float, float]:
        return self.get_epsilon(), self.target_delta

    def reset(self):
        self.steps = 0
        self.rdp_budget = np.zeros_like(self.rdp_orders)
