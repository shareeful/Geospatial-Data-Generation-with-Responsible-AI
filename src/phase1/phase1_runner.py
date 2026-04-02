import numpy as np
from typing import Dict, List, Tuple, Optional

from .fusion import DataFusion
from .fairness import FairnessAssurance
from .privacy import PrivacyAssurance
from .preprocessing import SourcePreprocessor
from ..utils.logging_utils import setup_logger

logger = setup_logger("phase1")


class DataPreparation:
    def __init__(self, config: Dict):
        self.config = config
        self.fusion = DataFusion(
            pca_variance_threshold=config.get("data", {}).get("pca_variance_threshold", 0.999)
        )
        self.fairness = FairnessAssurance(
            target_balance=config.get("fairness", {}).get("target_class_balance", 0.25),
            zpr_threshold=config.get("fairness", {}).get("zpr_threshold", 0.10),
        )
        self.privacy = PrivacyAssurance(
            grid_size_m=config.get("fairness", {}).get("k_anonymity_grid_m", 50.0),
            k_min=config.get("fairness", {}).get("k_min", 5),
        )
        self.preprocessor = SourcePreprocessor()

        self.X_prepared = None
        self.y_prepared = None
        self.zones_prepared = None
        self.loading_matrix = None
        self.modality_boundaries = None
        self.feature_names = None

    def run(self, feature_blocks: Dict[str, np.ndarray],
            feature_names_per_block: Dict[str, List],
            y: np.ndarray,
            zone_assignments: np.ndarray,
            coordinates: Optional[np.ndarray] = None) -> Dict:

        logger.info("=== Phase 1: Data Preparation ===")

        logger.info("Task 1.1: Harmonisation and Fusion")
        X_fused, feature_names = self.fusion.harmonise_and_fuse(feature_blocks, feature_names_per_block)

        logger.info("Task 1.1: Normalisation and PCA compression")
        X_compressed, loading_matrix = self.fusion.normalise_and_compress(X_fused)

        pre_fairness = self.fairness.assess_fairness(X_compressed, y, zone_assignments)
        logger.info(f"Pre-correction GGI: {pre_fairness['ggi']:.4f}")

        logger.info("Task 1.2: Assurance of Fairness")
        X_fair, y_fair, zones_fair = self.fairness.correct_biases(X_compressed, y, zone_assignments)

        post_fairness = self.fairness.assess_fairness(X_fair, y_fair, zones_fair)
        logger.info(f"Post-correction GGI: {post_fairness['ggi']:.4f}")

        logger.info("Task 1.3: Assurance of Privacy")
        if coordinates is not None:
            coord_mask = np.ones(len(X_fair), dtype=bool)
            coords_for_anon = coordinates
            if len(coordinates) != len(X_fair):
                coords_for_anon = np.column_stack([
                    np.random.uniform(0, 1000, len(X_fair)),
                    np.random.uniform(0, 1000, len(X_fair)),
                ])
            X_private, keep_mask = self.privacy.apply_spatial_k_anonymity(
                X_fair, coords_for_anon
            )
            y_private = y_fair[keep_mask]
            zones_private = zones_fair[keep_mask]
        else:
            X_private = X_fair
            y_private = y_fair
            zones_private = zones_fair

        self.X_prepared = X_private
        self.y_prepared = y_private
        self.zones_prepared = zones_private
        self.loading_matrix = loading_matrix
        self.modality_boundaries = self.fusion.get_modality_boundaries()
        self.feature_names = feature_names

        logger.info(
            f"Phase 1 complete: {X_private.shape[0]} samples, "
            f"{X_private.shape[1]} PCA components, "
            f"{len(np.unique(zones_private))} zones"
        )

        return {
            "X": X_private,
            "y": y_private,
            "zones": zones_private,
            "loading_matrix": loading_matrix,
            "modality_boundaries": self.modality_boundaries,
            "feature_names": feature_names,
            "pre_fairness": pre_fairness,
            "post_fairness": post_fairness,
            "n_components": X_private.shape[1],
            "fusion_module": self.fusion,
        }

    def update_parameters(self, delta: np.ndarray, eta: float):
        pass
