# RAI-GeoAI

Implementation of *"Multi-Modal Geospatial Data Fusion and Generation with Responsible AI Practice"* (Hassan et al., Transactions in GIS, 2026).

Funded by UKRI CHIST-ERA grant EP/Z003490/1 (AI4MultiGIS).

## Structure

```
├── main.py          # Entry point — runs full pipeline (Experiments 1–5)
├── pipeline.py      # Four phases + ablation + feedback loop
├── models.py        # cGAN (Generator/Discriminator), DP-SGD, training loop
├── metrics.py       # All spatial metrics (GGI, sEOD, ZPR, PCS, MCT, GWFA, CSD, AD)
├── visualize.py     # Paper figures
├── config.yaml      # Hyperparameters
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Outputs go to `results/` (JSON metrics + accountability docs) and `figures/` (plots).

## Pipeline

**Phase 1** — Fuses modalities, PCA compression (99.9% variance), geo-stratified sampling, minority oversampling, spatial k-anonymity (50m grid, k≥5).

**Phase 2** — Trains conditional GAN with DP-SGD (ε=5, δ=10⁻⁵) and sEOD fairness penalty in discriminator loss. Generates balanced synthetic dataset with provenance hashes.

**Phase 3** — TSTR evaluation (5-seed RF), membership inference attack, spatial fidelity (sFID, variogram, Moran's I).

**Phase 4** — GWFA attribution maps, Counterfactual Spatial Distance, Attribution Divergence (KL), accountability documents (Model Card, Synthetic Datasheet, Provenance Certificate).

## Using your own data

Replace `load_chelmsford()` in `main.py`. The pipeline expects a dict:

```python
{
    "features": {
        "sentinel2": np.ndarray,   # (n, bands)
        "lidar": np.ndarray,       # (n, 4)
        "flood_vec": np.ndarray,   # (n, 3)
        ...
    },
    "y": np.ndarray,               # integer class labels
    "zones": np.ndarray,           # ward/zone IDs
    "coords": np.ndarray,          # (n, 2) easting/northing
    "classes": ["High", "Moderate", "Low", "Negligible"],
}
```

## Key parameters (config.yaml)

| Parameter | Value | Paper reference |
|---|---|---|
| PCA variance | 0.999 | §4.1 Task 1.1 |
| k-anonymity | k≥5, 50m | §4.1 Task 1.3 |
| DP-SGD ε | 5.0 | §4.2 Task 2.3 |
| sEOD penalty λ | 0.1 | §4.2 Task 2.4, Eq. 8 |
| Feedback Δ | 0.005 | §4.3 Task 3.5, Eq. 13 |

## Citation

```bibtex
@article{hassan2026multimodal,
  title={Multi-Modal Geospatial Data Fusion and Generation with Responsible AI Practice},
  author={Hassan, Muhammad and Sardar, Bilal and Islam, Shareeful and Imani, Maryam and Hakiri, Akram},
  journal={Transactions in GIS},
  year={2026}
}
```

## Code

https://github.com/shareeful/Geospatial-Data-Generation-with-Responsible-AI
