import numpy as np
from typing import Tuple

from ..utils.logging_utils import setup_logger

logger = setup_logger("privacy")


class PrivacyAssurance:
    def __init__(self, grid_size_m: float = 50.0, k_min: int = 5):
        self.grid_size_m = grid_size_m
        self.k_min = k_min

    def apply_spatial_k_anonymity(self, X: np.ndarray, coordinates: np.ndarray,
                                   coordinate_columns: list = None) -> Tuple[np.ndarray, np.ndarray]:
        grid_x = np.floor(coordinates[:, 0] / self.grid_size_m).astype(int)
        grid_y = np.floor(coordinates[:, 1] / self.grid_size_m).astype(int)
        cell_ids = grid_x * 100000 + grid_y

        unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
        valid_cells = set(unique_cells[cell_counts >= self.k_min])

        keep_mask = np.array([cid in valid_cells for cid in cell_ids])

        anonymised_coords = np.column_stack([
            (grid_x + 0.5) * self.grid_size_m,
            (grid_y + 0.5) * self.grid_size_m,
        ])

        n_removed = np.sum(~keep_mask)
        logger.info(
            f"Spatial k-anonymity (k={self.k_min}, grid={self.grid_size_m}m): "
            f"removed {n_removed} samples from underpopulated cells, "
            f"retained {np.sum(keep_mask)}/{len(X)}"
        )

        X_out = X[keep_mask].copy()

        if coordinate_columns is not None:
            for col_idx in coordinate_columns:
                if col_idx < X_out.shape[1]:
                    X_out[:, col_idx] = anonymised_coords[keep_mask, 0] if col_idx == coordinate_columns[0] else anonymised_coords[keep_mask, 1]

        return X_out, keep_mask

    def verify_k_anonymity(self, coordinates: np.ndarray) -> Tuple[bool, int]:
        grid_x = np.floor(coordinates[:, 0] / self.grid_size_m).astype(int)
        grid_y = np.floor(coordinates[:, 1] / self.grid_size_m).astype(int)
        cell_ids = grid_x * 100000 + grid_y

        unique_cells, cell_counts = np.unique(cell_ids, return_counts=True)
        min_count = int(np.min(cell_counts)) if len(cell_counts) > 0 else 0

        passed = min_count >= self.k_min
        return passed, min_count
