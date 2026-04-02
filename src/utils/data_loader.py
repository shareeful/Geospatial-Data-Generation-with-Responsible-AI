import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import hashlib

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    rasterio = None

try:
    from pyproj import CRS
except ImportError:
    CRS = None

from ..utils.logging_utils import setup_logger

logger = setup_logger("data_loader")


class ModalityRecord:
    def __init__(self, modality_id: str, data: np.ndarray, timestamp: str, source_type: str, crs: str):
        self.modality_id = modality_id
        self.data = data
        self.timestamp = timestamp
        self.source_type = source_type
        self.crs = crs
        self.hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]


class DataLoader:
    def __init__(self, target_crs: str = "EPSG:32630", resolution_m: float = 10.0):
        self.target_crs = CRS.from_string(target_crs) if CRS is not None else None
        self.target_crs_str = target_crs
        self.resolution_m = resolution_m
        self.modality_registry: Dict[str, ModalityRecord] = {}

    def ingest_raster(self, filepath: str, modality_id: str, timestamp: str) -> ModalityRecord:
        if rasterio is None:
            raise ImportError("rasterio is required for raster ingestion. Install with: pip install rasterio")
        logger.info(f"Ingesting raster modality: {modality_id} from {filepath}")
        with rasterio.open(filepath) as src:
            data = src.read()
            source_crs = str(src.crs)

        record = ModalityRecord(
            modality_id=modality_id,
            data=data,
            timestamp=timestamp,
            source_type="raster",
            crs=source_crs,
        )
        self.modality_registry[modality_id] = record
        return record

    def ingest_vector(self, filepath: str, modality_id: str, timestamp: str) -> ModalityRecord:
        if gpd is None:
            raise ImportError("geopandas is required for vector ingestion. Install with: pip install geopandas")
        logger.info(f"Ingesting vector modality: {modality_id} from {filepath}")
        gdf = gpd.read_file(filepath)
        source_crs = str(gdf.crs)
        data = np.array(gdf.geometry.apply(lambda g: np.array(g.centroid.coords[0])).tolist())

        record = ModalityRecord(
            modality_id=modality_id,
            data=data,
            timestamp=timestamp,
            source_type="vector",
            crs=source_crs,
        )
        self.modality_registry[modality_id] = record
        return record

    def ingest_array(self, data: np.ndarray, modality_id: str, timestamp: str,
                     source_type: str, crs: str) -> ModalityRecord:
        record = ModalityRecord(
            modality_id=modality_id,
            data=data,
            timestamp=timestamp,
            source_type=source_type,
            crs=crs,
        )
        self.modality_registry[modality_id] = record
        return record

    def reproject_raster(self, data: np.ndarray, src_crs: str, src_transform,
                         src_shape: Tuple[int, int]) -> Tuple[np.ndarray, object]:
        if rasterio is None:
            raise ImportError("rasterio is required for reprojection. Install with: pip install rasterio")
        dst_crs = self.target_crs_str
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src_shape[1], src_shape[0],
            *rasterio.transform.array_bounds(src_shape[0], src_shape[1], src_transform),
            resolution=self.resolution_m,
        )

        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        n_bands = data.shape[0]
        dst_data = np.zeros((n_bands, height, width), dtype=data.dtype)

        for band_idx in range(n_bands):
            reproject(
                source=data[band_idx],
                destination=dst_data[band_idx],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        return dst_data, transform

    def load_tabular(self, filepath: str, zone_column: str = "ward") -> pd.DataFrame:
        path = Path(filepath)
        if path.suffix == ".csv":
            return pd.read_csv(filepath)
        elif path.suffix in (".shp", ".gpkg", ".geojson"):
            if gpd is None:
                raise ImportError("geopandas required for spatial file formats")
            return gpd.read_file(filepath)
        elif path.suffix == ".parquet":
            return pd.read_parquet(filepath)
        raise ValueError(f"Unsupported file format: {path.suffix}")

    def build_feature_matrix(self, zone_assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        feature_columns = []
        feature_blocks = []

        for mod_id, record in sorted(self.modality_registry.items()):
            if record.data.ndim == 1:
                feature_blocks.append(record.data.reshape(-1, 1))
                feature_columns.append(mod_id)
            elif record.data.ndim == 2:
                feature_blocks.append(record.data)
                for col_idx in range(record.data.shape[1]):
                    feature_columns.append(f"{mod_id}_{col_idx}")
            else:
                flat = record.data.reshape(record.data.shape[0], -1).T
                feature_blocks.append(flat)
                for col_idx in range(flat.shape[1]):
                    feature_columns.append(f"{mod_id}_b{col_idx}")

        X = np.hstack(feature_blocks)
        return X, zone_assignments, feature_columns

    def get_provenance_chain(self) -> Dict[str, Dict]:
        chain = {}
        for mod_id, record in self.modality_registry.items():
            chain[mod_id] = {
                "modality_id": record.modality_id,
                "timestamp": record.timestamp,
                "source_type": record.source_type,
                "original_crs": record.crs,
                "data_hash": record.hash,
                "shape": record.data.shape,
            }
        return chain
