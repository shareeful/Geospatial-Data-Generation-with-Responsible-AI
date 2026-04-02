import numpy as np
from typing import Dict, Tuple
from collections import Counter

from ..metrics.spatial_metrics import SpatialMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("fairness")


class FairnessAssurance:
    def __init__(self, target_balance: float = 0.25, zpr_threshold: float = 0.10):
        self.target_balance = target_balance
        self.zpr_threshold = zpr_threshold

    def compute_zone_coverage(self, zone_assignments: np.ndarray,
                               valid_mask: np.ndarray) -> np.ndarray:
        unique_zones = np.unique(zone_assignments)
        coverage = np.zeros(len(unique_zones))

        for idx, zone in enumerate(unique_zones):
            zone_mask = zone_assignments == zone
            coverage[idx] = np.mean(valid_mask[zone_mask])

        return coverage

    def geo_stratified_sampling(self, X: np.ndarray, y: np.ndarray,
                                 zone_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_zones = np.unique(zone_assignments)
        zone_sizes = np.array([np.sum(zone_assignments == z) for z in unique_zones])
        target_per_zone = int(np.mean(zone_sizes))

        resampled_X = []
        resampled_y = []
        resampled_zones = []

        for zone in unique_zones:
            zone_mask = zone_assignments == zone
            X_zone = X[zone_mask]
            y_zone = y[zone_mask]
            n_zone = len(X_zone)

            if n_zone == 0:
                continue

            if n_zone < target_per_zone:
                indices = np.random.choice(n_zone, target_per_zone, replace=True)
            elif n_zone > target_per_zone:
                indices = np.random.choice(n_zone, target_per_zone, replace=False)
            else:
                indices = np.arange(n_zone)

            resampled_X.append(X_zone[indices])
            resampled_y.append(y_zone[indices])
            resampled_zones.append(np.full(len(indices), zone))

        X_out = np.vstack(resampled_X)
        y_out = np.concatenate(resampled_y)
        zones_out = np.concatenate(resampled_zones)

        logger.info(
            f"Geo-stratified sampling: {len(X)} -> {len(X_out)} samples across {len(unique_zones)} zones"
        )
        return X_out, y_out, zones_out

    def minority_class_oversampling(self, X: np.ndarray, y: np.ndarray,
                                     zone_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        class_counts = Counter(y)
        n_classes = len(class_counts)
        target_count = len(y) // n_classes

        augmented_X = [X]
        augmented_y = [y]
        augmented_zones = [zone_assignments]

        for cls, count in class_counts.items():
            if count < target_count:
                deficit = target_count - count
                cls_mask = y == cls
                cls_indices = np.where(cls_mask)[0]

                if len(cls_indices) == 0:
                    continue

                oversample_indices = np.random.choice(cls_indices, deficit, replace=True)
                noise = np.random.normal(0, 0.01, (deficit, X.shape[1]))

                augmented_X.append(X[oversample_indices] + noise)
                augmented_y.append(y[oversample_indices])
                augmented_zones.append(zone_assignments[oversample_indices])

        X_out = np.vstack(augmented_X)
        y_out = np.concatenate(augmented_y)
        zones_out = np.concatenate(augmented_zones)

        new_counts = Counter(y_out)
        logger.info(f"Minority oversampling: {dict(class_counts)} -> {dict(new_counts)}")
        return X_out, y_out, zones_out

    def assess_fairness(self, X: np.ndarray, y: np.ndarray,
                         zone_assignments: np.ndarray,
                         valid_mask: np.ndarray = None) -> Dict[str, float]:
        if valid_mask is None:
            valid_mask = np.ones(len(X), dtype=bool)

        coverage = self.compute_zone_coverage(zone_assignments, valid_mask)
        ggi = SpatialMetrics.geographic_gini_index(coverage)

        class_dist = Counter(y)
        total = sum(class_dist.values())
        class_proportions = {k: v / total for k, v in class_dist.items()}

        return {
            "ggi": ggi,
            "zone_coverage": coverage,
            "class_distribution": class_proportions,
            "n_samples": len(X),
            "n_zones": len(np.unique(zone_assignments)),
        }

    def correct_biases(self, X: np.ndarray, y: np.ndarray,
                       zone_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_geo, y_geo, zones_geo = self.geo_stratified_sampling(X, y, zone_assignments)
        X_balanced, y_balanced, zones_balanced = self.minority_class_oversampling(X_geo, y_geo, zones_geo)

        shuffle_idx = np.random.permutation(len(X_balanced))
        return X_balanced[shuffle_idx], y_balanced[shuffle_idx], zones_balanced[shuffle_idx]
