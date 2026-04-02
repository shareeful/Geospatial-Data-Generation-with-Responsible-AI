import numpy as np
from scipy.stats import ks_2samp
from typing import Dict

from ..metrics.spatial_metrics import SpatialMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("attribution_divergence")


class AttributionDivergenceAnalyzer:
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def compute(self, attributions_real: np.ndarray,
                attributions_synthetic: np.ndarray) -> Dict:

        ad_values = SpatialMetrics.attribution_divergence(attributions_real, attributions_synthetic)

        mean_ad = float(np.mean(ad_values))
        pct_below_threshold = float(np.mean(ad_values < self.threshold)) * 100

        logger.info(
            f"Attribution Divergence: mean={mean_ad:.4f}, "
            f"{pct_below_threshold:.1f}% below threshold {self.threshold}"
        )

        return {
            "ad_values": ad_values,
            "mean_ad": mean_ad,
            "median_ad": float(np.median(ad_values)),
            "pct_below_threshold": pct_below_threshold,
            "threshold": self.threshold,
        }

    def compare_csd_distributions(self, csd_real: np.ndarray,
                                    csd_synthetic: np.ndarray) -> Dict:

        valid_real = csd_real[~np.isnan(csd_real)]
        valid_syn = csd_synthetic[~np.isnan(csd_synthetic)]

        if len(valid_real) == 0 or len(valid_syn) == 0:
            return {"ks_statistic": 1.0, "ks_pvalue": 0.0, "equivalent": False}

        ks_stat, ks_pvalue = ks_2samp(valid_real, valid_syn)

        logger.info(f"CSD KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "equivalent": ks_pvalue > 0.05,
        }
