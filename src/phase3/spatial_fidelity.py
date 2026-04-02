import numpy as np
from typing import Dict

from ..metrics.spatial_metrics import SpatialMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("spatial_fidelity")


class SpatialFidelityEvaluator:
    def __init__(self):
        self.sm = SpatialMetrics()

    def evaluate(self, X_real: np.ndarray, X_synthetic: np.ndarray,
                 coords_real: np.ndarray, coords_synthetic: np.ndarray) -> Dict:

        sfid = self.sm.compute_sfid(X_real, X_synthetic)

        real_values = X_real[:, 0] if X_real.shape[1] > 0 else np.zeros(len(X_real))
        syn_values = X_synthetic[:, 0] if X_synthetic.shape[1] > 0 else np.zeros(len(X_synthetic))

        var_fidelity = self.sm.variogram_fidelity(
            coords_real, real_values, coords_synthetic, syn_values
        )

        morans_real = self.sm.morans_i(real_values, coords_real)
        morans_syn = self.sm.morans_i(syn_values, coords_synthetic)

        morans_preservation = 1.0 - abs(morans_real - morans_syn) / (abs(morans_real) + 1e-8)
        morans_preservation = max(0.0, min(1.0, morans_preservation))

        logger.info(
            f"Spatial fidelity: sFID={sfid:.4f}, "
            f"Variogram={var_fidelity:.4f}, "
            f"Moran's I preservation={morans_preservation:.4f}"
        )

        return {
            "sfid": sfid,
            "variogram_fidelity": var_fidelity,
            "morans_i_real": morans_real,
            "morans_i_synthetic": morans_syn,
            "morans_i_preservation": morans_preservation,
        }
