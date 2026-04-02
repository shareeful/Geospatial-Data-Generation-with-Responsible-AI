import numpy as np
from typing import Dict, List, Tuple, Callable

from ..utils.logging_utils import setup_logger

logger = setup_logger("feedback")


class FeedbackLoop:
    def __init__(self, config: Dict):
        fb_cfg = config.get("feedback", {})
        self.max_iterations = fb_cfg.get("max_iterations", 10)
        self.convergence_threshold = fb_cfg.get("convergence_threshold", 0.005)
        self.eta_min = fb_cfg.get("cosine_eta_min", 0.001)
        self.eta_max = fb_cfg.get("cosine_eta_max", 0.1)
        self.history: List[Dict] = []

    def _cosine_eta(self, k: int) -> float:
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * k / self.max_iterations)
        )

    def _compute_delta(self, current: Dict[str, float], previous: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        deltas = {}
        for key in current:
            if key in previous:
                deltas[key] = abs(current[key] - previous[key])
        linf = max(deltas.values()) if deltas else float("inf")
        return linf, deltas

    def run(self, train_fn: Callable, evaluate_fn: Callable,
            initial_params: Dict) -> Dict:

        logger.info("=== Feedback Loop Started ===")

        params = initial_params.copy()
        previous_metrics = None

        for k in range(self.max_iterations):
            eta = self._cosine_eta(k)

            train_result = train_fn(params)
            eval_metrics = evaluate_fn(train_result)

            iteration_record = {
                "iteration": k + 1,
                "eta": eta,
                "metrics": eval_metrics.copy(),
            }

            if previous_metrics is not None:
                delta_linf, delta_per_metric = self._compute_delta(eval_metrics, previous_metrics)
                iteration_record["delta"] = delta_linf
                iteration_record["delta_per_metric"] = delta_per_metric

                logger.info(
                    f"Iteration {k + 1}: Δ={delta_linf:.6f}, η={eta:.4f}, "
                    f"metrics={eval_metrics}"
                )

                if delta_linf < self.convergence_threshold:
                    iteration_record["converged"] = True
                    self.history.append(iteration_record)
                    logger.info(f"Converged at iteration {k + 1} (Δ={delta_linf:.6f} < {self.convergence_threshold})")
                    break
            else:
                iteration_record["delta"] = float("inf")
                logger.info(f"Iteration {k + 1} (first pass): metrics={eval_metrics}")

            self.history.append(iteration_record)
            previous_metrics = eval_metrics.copy()

            feedback_signal = {}
            for key, value in eval_metrics.items():
                if key == "seod":
                    target = 0.05
                    feedback_signal[key] = eta * (value - target)
                elif key == "macro_f1":
                    target = 0.90
                    feedback_signal[key] = -eta * (target - value)
                elif key == "zpr":
                    target = 0.90
                    feedback_signal[key] = -eta * (target - value)

            for key, adj in feedback_signal.items():
                if key in params:
                    params[key] = params[key] - adj

        return {
            "history": self.history,
            "final_metrics": self.history[-1]["metrics"] if self.history else {},
            "converged": self.history[-1].get("converged", False) if self.history else False,
            "n_iterations": len(self.history),
        }
