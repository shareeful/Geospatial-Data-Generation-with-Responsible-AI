import numpy as np
from scipy.ndimage import uniform_filter, median_filter, binary_erosion, binary_dilation
from typing import Dict, Tuple

from ..utils.logging_utils import setup_logger

logger = setup_logger("preprocessing")


class SourcePreprocessor:

    @staticmethod
    def temporal_median_compositing(multi_temporal: np.ndarray) -> np.ndarray:
        return np.nanmedian(multi_temporal, axis=0)

    @staticmethod
    def terrain_conditioned_interpolation(dem: np.ndarray) -> np.ndarray:
        nan_mask = np.isnan(dem)
        if not np.any(nan_mask):
            return dem

        filled = dem.copy()
        mean_val = np.nanmean(dem)
        filled[nan_mask] = mean_val

        for _ in range(3):
            smoothed = uniform_filter(filled, size=5)
            filled[nan_mask] = smoothed[nan_mask]

        return filled

    @staticmethod
    def majority_resampling(categorical: np.ndarray, factor: int = 2) -> np.ndarray:
        from scipy.stats import mode
        h, w = categorical.shape
        new_h, new_w = h // factor, w // factor
        result = np.zeros((new_h, new_w), dtype=categorical.dtype)

        for i in range(new_h):
            for j in range(new_w):
                block = categorical[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
                vals, counts = np.unique(block, return_counts=True)
                result[i, j] = vals[np.argmax(counts)]

        return result

    @staticmethod
    def morphological_filtering(binary_mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        struct = np.ones((3, 3))
        cleaned = binary_erosion(binary_mask, structure=struct, iterations=iterations)
        cleaned = binary_dilation(cleaned, structure=struct, iterations=iterations)
        return cleaned.astype(binary_mask.dtype)

    @staticmethod
    def compute_proximity_surface(points: np.ndarray, grid_shape: Tuple[int, int],
                                   bounds: Tuple[float, float, float, float]) -> np.ndarray:
        rows, cols = grid_shape
        min_x, min_y, max_x, max_y = bounds

        y_coords = np.linspace(min_y, max_y, rows)
        x_coords = np.linspace(min_x, max_x, cols)
        xx, yy = np.meshgrid(x_coords, y_coords)

        proximity = np.full(grid_shape, np.inf)

        for point in points:
            dist = np.sqrt((xx - point[0]) ** 2 + (yy - point[1]) ** 2)
            proximity = np.minimum(proximity, dist)

        return proximity

    @staticmethod
    def compute_density_surface(points: np.ndarray, grid_shape: Tuple[int, int],
                                 bounds: Tuple[float, float, float, float],
                                 bandwidth: float = 100.0) -> np.ndarray:
        rows, cols = grid_shape
        min_x, min_y, max_x, max_y = bounds

        y_coords = np.linspace(min_y, max_y, rows)
        x_coords = np.linspace(min_x, max_x, cols)
        xx, yy = np.meshgrid(x_coords, y_coords)

        density = np.zeros(grid_shape)

        for point in points:
            dist_sq = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
            density += np.exp(-dist_sq / (2 * bandwidth ** 2))

        return density

    @staticmethod
    def compute_spectral_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        indices = {}

        if "nir" in bands and "red" in bands:
            nir = bands["nir"].astype(np.float64)
            red = bands["red"].astype(np.float64)
            denominator = nir + red
            indices["ndvi"] = np.where(denominator != 0, (nir - red) / denominator, 0.0)

        if "green" in bands and "nir" in bands:
            green = bands["green"].astype(np.float64)
            nir = bands["nir"].astype(np.float64)
            denominator = green + nir
            indices["ndwi"] = np.where(denominator != 0, (green - nir) / denominator, 0.0)

        if "swir" in bands and "nir" in bands:
            swir = bands["swir"].astype(np.float64)
            nir = bands["nir"].astype(np.float64)
            denominator = swir + nir
            indices["ndbi"] = np.where(denominator != 0, (swir - nir) / denominator, 0.0)

        return indices

    @staticmethod
    def compute_terrain_derivatives(dem: np.ndarray, resolution: float = 10.0) -> Dict[str, np.ndarray]:
        dy, dx = np.gradient(dem, resolution)
        slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        aspect = np.arctan2(-dx, dy)

        flow_acc = np.zeros_like(dem)
        padded = np.pad(dem, 1, mode="edge")
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                center = padded[i, j]
                neighborhood = padded[i - 1:i + 2, j - 1:j + 2]
                flow_acc[i - 1, j - 1] = np.sum(neighborhood > center)

        return {
            "slope": slope,
            "aspect": aspect,
            "flow_accumulation": flow_acc,
        }

    def preprocess_all(self, raw_data: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        processed = {}

        for source_id, source_info in raw_data.items():
            data = source_info["data"]
            source_type = source_info["type"]
            problem = source_info.get("problem", None)

            if problem == "cloud_gaps":
                data = self.temporal_median_compositing(data)
            elif problem == "resampling_artefacts":
                data = self.terrain_conditioned_interpolation(data)
            elif problem == "boundary_noise":
                data = self.morphological_filtering(data)
            elif problem == "mixed_pixels" and data.ndim == 2:
                data = self.majority_resampling(data, factor=2)

            processed[source_id] = data
            logger.info(f"Preprocessed {source_id}: type={source_type}, shape={data.shape}")

        return processed
