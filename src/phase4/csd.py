import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize

from ..utils.logging_utils import setup_logger

logger = setup_logger("csd")


class CSDExplainer:
    def __init__(self, max_iter: int = 100, step_sizes: list = None):
        self.max_iter = max_iter
        self.step_sizes = step_sizes or [0.01, 0.05, 0.1, 0.25]

    def compute_csd(self, X: np.ndarray, predictions: np.ndarray,
                     classifier, target_class_offset: int = -1) -> np.ndarray:
        n_samples = X.shape[0]
        csd_values = np.full(n_samples, np.nan)
        unique_classes = np.unique(predictions)

        for idx in range(n_samples):
            current_class = predictions[idx]
            target_class = current_class + target_class_offset

            if target_class not in unique_classes:
                target_class = current_class - target_class_offset
            if target_class not in unique_classes:
                continue

            x_orig = X[idx].copy()
            best_norm = float("inf")

            for step in self.step_sizes:
                for feat in range(X.shape[1]):
                    for direction in [-1.0, 1.0]:
                        x_mod = x_orig.copy()
                        for mult in range(1, self.max_iter):
                            x_mod[feat] = x_orig[feat] + direction * step * mult
                            pred = classifier.predict(x_mod.reshape(1, -1))[0]
                            if pred == target_class:
                                delta_norm = np.linalg.norm(x_mod - x_orig)
                                best_norm = min(best_norm, delta_norm)
                                break

            if best_norm < float("inf"):
                csd_values[idx] = best_norm

        valid = ~np.isnan(csd_values)
        logger.info(
            f"CSD computed: {np.sum(valid)}/{n_samples} valid, "
            f"mean={np.nanmean(csd_values):.4f}, median={np.nanmedian(csd_values):.4f}"
        )
        return csd_values

    def compute_csd_optimized(self, X: np.ndarray, predictions: np.ndarray,
                               classifier, target_class_offset: int = -1) -> np.ndarray:
        n_samples = X.shape[0]
        csd_values = np.full(n_samples, np.nan)
        unique_classes = np.unique(predictions)

        for idx in range(n_samples):
            current_class = predictions[idx]
            target_class = current_class + target_class_offset

            if target_class not in unique_classes:
                target_class = current_class - target_class_offset
            if target_class not in unique_classes:
                continue

            x_orig = X[idx].copy()

            def objective(delta):
                x_new = x_orig + delta
                pred = classifier.predict(x_new.reshape(1, -1))[0]
                if pred == target_class:
                    return np.linalg.norm(delta)
                return np.linalg.norm(delta) + 1000.0

            best_norm = float("inf")
            for _ in range(5):
                x0 = np.random.normal(0, 0.1, X.shape[1])
                result = minimize(objective, x0, method="Nelder-Mead",
                                  options={"maxiter": 200, "xatol": 1e-4})
                if result.fun < best_norm and result.fun < 999:
                    best_norm = result.fun

            if best_norm < float("inf"):
                csd_values[idx] = best_norm

        return csd_values

    def identify_suds_locations(self, csd_values: np.ndarray, coordinates: np.ndarray,
                                 n_recommendations: int = 7,
                                 percentile_threshold: float = 10.0) -> np.ndarray:
        valid_mask = ~np.isnan(csd_values)
        valid_csd = csd_values[valid_mask]
        valid_coords = coordinates[valid_mask]

        threshold = np.percentile(valid_csd, percentile_threshold)
        sensitive_mask = valid_csd <= threshold

        sensitive_coords = valid_coords[sensitive_mask]
        sensitive_csd = valid_csd[sensitive_mask]

        if len(sensitive_coords) <= n_recommendations:
            return sensitive_coords

        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(sensitive_coords, method="ward")
        clusters = fcluster(Z, t=n_recommendations, criterion="maxclust")

        centroids = []
        for c in range(1, n_recommendations + 1):
            c_mask = clusters == c
            if np.any(c_mask):
                centroid = np.mean(sensitive_coords[c_mask], axis=0)
                centroids.append(centroid)

        return np.array(centroids)
