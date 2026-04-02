import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import hashlib

from ..utils.logging_utils import setup_logger

logger = setup_logger("fusion")


class DataFusion:
    def __init__(self, pca_variance_threshold: float = 0.999):
        self.pca_variance_threshold = pca_variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.n_components_retained = 0
        self.loading_matrix = None
        self.feature_names: List[str] = []
        self.provenance_registry: Dict[str, Dict] = {}

    def register_modality(self, modality_id: str, timestamp: str, source_hash: str,
                          n_features: int, source_type: str):
        self.provenance_registry[modality_id] = {
            "modality_id": modality_id,
            "timestamp": timestamp,
            "source_hash": source_hash,
            "n_features": n_features,
            "source_type": source_type,
        }

    def quality_filter(self, data: np.ndarray, source_type: str,
                       tau_satellite: float = 0.50, tau_static: float = 0.70) -> Tuple[np.ndarray, bool]:
        valid_ratio = np.mean(~np.isnan(data) & np.isfinite(data))
        threshold = tau_satellite if source_type == "satellite" else tau_static
        passed = valid_ratio >= threshold
        logger.info(f"Quality filter: valid_ratio={valid_ratio:.3f}, threshold={threshold:.2f}, passed={passed}")
        return data, passed

    def temporal_median_composite(self, multi_temporal: np.ndarray) -> np.ndarray:
        return np.nanmedian(multi_temporal, axis=0)

    def terrain_conditioned_interpolation(self, dem: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter
        nan_mask = np.isnan(dem)
        if not np.any(nan_mask):
            return dem

        smoothed = uniform_filter(np.nan_to_num(dem, nan=np.nanmean(dem)), size=3)
        result = dem.copy()
        result[nan_mask] = smoothed[nan_mask]
        return result

    def harmonise_and_fuse(self, feature_blocks: Dict[str, np.ndarray],
                            feature_names_per_block: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
        all_features = []
        all_names = []
        modality_boundaries = []

        current_idx = 0
        for mod_id in sorted(feature_blocks.keys()):
            block = feature_blocks[mod_id]
            names = feature_names_per_block[mod_id]

            if block.ndim == 1:
                block = block.reshape(-1, 1)

            all_features.append(block)
            all_names.extend(names)
            modality_boundaries.append((current_idx, current_idx + block.shape[1]))
            current_idx += block.shape[1]

        X = np.hstack(all_features)
        nan_mask = np.any(np.isnan(X), axis=1)
        X[nan_mask] = np.nanmedian(X, axis=0)

        self.feature_names = all_names
        self.modality_boundaries = modality_boundaries

        logger.info(f"Fused {len(feature_blocks)} modalities into matrix of shape {X.shape}")
        return X, all_names

    def normalise_and_compress(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.fit_transform(X)

        self.pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
        X_pca_full = self.pca.fit_transform(X_scaled)

        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        self.n_components_retained = int(np.searchsorted(cumvar, self.pca_variance_threshold) + 1)
        self.n_components_retained = min(self.n_components_retained, X_pca_full.shape[1])

        X_compressed = X_pca_full[:, :self.n_components_retained]
        self.loading_matrix = self.pca.components_[:self.n_components_retained]

        logger.info(
            f"PCA: {X.shape[1]} features -> {self.n_components_retained} components "
            f"({cumvar[self.n_components_retained - 1]:.4f} variance retained)"
        )
        return X_compressed, self.loading_matrix

    def get_modality_boundaries(self) -> List[Tuple[int, int]]:
        return self.modality_boundaries

    def build_provenance_hash(self, sample_idx: int, epoch: int) -> str:
        loading_bytes = self.loading_matrix.tobytes()
        combined = f"{sample_idx}_{epoch}_{hashlib.sha256(loading_bytes).hexdigest()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        X_full_pca = np.zeros((X_pca.shape[0], self.pca.n_components_))
        X_full_pca[:, :self.n_components_retained] = X_pca
        X_scaled = self.pca.inverse_transform(X_full_pca)
        return self.scaler.inverse_transform(X_scaled)
