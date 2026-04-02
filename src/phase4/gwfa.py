import numpy as np
from typing import Dict, Optional

from ..metrics.spatial_metrics import SpatialMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("gwfa")


class GWFAExplainer:
    def __init__(self, adaptive: bool = True, default_bandwidth: float = 500.0):
        self.adaptive = adaptive
        self.default_bandwidth = default_bandwidth

    def compute_attributions(self, X: np.ndarray, y: np.ndarray,
                              coordinates: np.ndarray,
                              target_locations: Optional[np.ndarray] = None,
                              feature_names: list = None) -> Dict:

        if target_locations is None:
            n_grid = min(50, int(np.sqrt(len(X))))
            x_range = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), n_grid)
            y_range = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), n_grid)
            xx, yy = np.meshgrid(x_range, y_range)
            target_locations = np.column_stack([xx.ravel(), yy.ravel()])

        attributions = SpatialMetrics.geographically_weighted_feature_attribution(
            X=X,
            y=y.astype(float),
            coordinates=coordinates,
            target_locations=target_locations,
            bandwidth=self.default_bandwidth,
            adaptive=self.adaptive,
        )

        result = {
            "attributions": attributions,
            "target_locations": target_locations,
        }

        if feature_names is not None:
            n_grid = int(np.sqrt(len(target_locations)))
            attribution_maps = {}
            for f_idx, f_name in enumerate(feature_names[:X.shape[1]]):
                if n_grid * n_grid == len(target_locations):
                    attribution_maps[f_name] = attributions[:, f_idx].reshape(n_grid, n_grid)
                else:
                    attribution_maps[f_name] = attributions[:, f_idx]
            result["maps"] = attribution_maps

        logger.info(f"GWFA computed for {len(target_locations)} locations, {X.shape[1]} features")
        return result

    def spatial_attribution_consistency(self, attributions: np.ndarray,
                                         coordinates: np.ndarray,
                                         k_neighbours: int = 5) -> float:
        from scipy.spatial.distance import cdist

        n = len(attributions)
        if n < k_neighbours + 1:
            return 1.0

        dists = cdist(coordinates, coordinates)
        consistencies = []

        for i in range(n):
            nearest = np.argsort(dists[i])[1:k_neighbours + 1]
            neighbour_attrs = attributions[nearest]
            similarities = []
            for j in range(len(nearest)):
                cos_sim = np.dot(attributions[i], neighbour_attrs[j]) / (
                    np.linalg.norm(attributions[i]) * np.linalg.norm(neighbour_attrs[j]) + 1e-12
                )
                similarities.append(cos_sim)
            consistencies.append(np.mean(similarities))

        return float(np.mean(consistencies))
