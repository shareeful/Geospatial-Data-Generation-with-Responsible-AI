# RAI-GeoAI: Responsible AI for Multi-Modal Geospatial Data Fusion and Generation

Production implementation of the paper:

**"Multi-Modal Geospatial Data Fusion and Generation with Responsible AI Practice"**
*Hassan, Sardar, Islam, Imani, Hakiri — Transactions in GIS, 2026*

Funded by UKRI CHIST-ERA grant EP/Z003490/1 (AI4MultiGIS).

---

## Overview

This codebase operationalises a four-phase responsible AI approach for multi-modal geospatial synthetic data generation. The approach embeds fairness, privacy, transparency, and explainability constraints at the lifecycle phase where each risk originates, operating through a closed-loop methodology supported by six spatial metrics.

**Key results reproduced:**
- Geographic bias reduction of 74% (GGI: 0.31 → 0.08)
- Membership inference accuracy reduced to 0.51 (random chance)
- Synthetic-trained classifier macro F1 = 0.861 under ε=5 differential privacy
- Cross-site generalisation gap of 0.017 F1 (Chelmsford → Leeds)
- 91% of study area achieves Attribution Divergence < 0.05

---

## Repository Structure

```
rai_geospatial/
├── main.py                          # Entry point — runs the full four-phase pipeline
├── config.yaml                      # All hyperparameters and thresholds
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installer
├── src/
│   ├── __init__.py                  # Top-level package exports
│   ├── phase1/                      # Phase 1: Data Preparation
│   │   ├── __init__.py
│   │   ├── fusion.py                # Task 1.1 — harmonisation, PCA compression, provenance
│   │   ├── fairness.py              # Task 1.2 — GGI, geo-stratified sampling, minority oversampling
│   │   ├── privacy.py               # Task 1.3 — spatial k-anonymity (k≥5, 50m grid)
│   │   ├── preprocessing.py         # Source-specific corrections (cloud compositing, morphological filtering)
│   │   └── phase1_runner.py         # Orchestrates Tasks 1.1–1.3
│   ├── phase2/                      # Phase 2: Conditional Generative Model Training
│   │   ├── __init__.py
│   │   ├── generator.py             # Generator G(z,c) — 3 FC layers, BatchNorm, LeakyReLU
│   │   ├── discriminator.py         # Discriminator D(x,c) — 3 FC layers, Dropout, Sigmoid
│   │   ├── cgan.py                  # CGAN wrapper — balanced generation, save/load
│   │   ├── dp_sgd.py                # DP-SGD engine — gradient clipping, noise injection, Rényi DP accounting
│   │   └── training.py              # Full training loop with sEOD penalty and privacy budget tracking
│   ├── phase3/                      # Phase 3: Validation and Feedback
│   │   ├── __init__.py
│   │   ├── validation.py            # TSTR protocol, real baseline, single-source evaluation
│   │   ├── mia.py                   # Shadow-model membership inference attack (MIA-AUC)
│   │   ├── spatial_fidelity.py      # sFID, variogram fidelity, Moran's I preservation
│   │   ├── ablation.py              # Ablation study runner (conditions A0–A5)
│   │   └── feedback.py              # Feedback loop Δ with cosine annealing convergence
│   ├── phase4/                      # Phase 4: Explainability and Accountability
│   │   ├── __init__.py
│   │   ├── gwfa.py                  # Geographically Weighted Feature Attribution maps
│   │   ├── csd.py                   # Counterfactual Spatial Distance + SuDS placement
│   │   ├── attribution_divergence.py # Attribution Divergence (KL) + CSD KS test
│   │   ├── accountability.py        # Model Card, Synthetic Datasheet, Provenance Certificate (JSON)
│   │   └── phase4_runner.py         # Orchestrates Tasks 4.1–4.2
│   ├── metrics/                     # All six novel spatial metrics + classification metrics
│   │   ├── __init__.py
│   │   ├── spatial_metrics.py       # GGI, sEOD, ZPR, PCS, MCT, GWFA, CSD, AD, sFID, variogram, Moran's I
│   │   └── classification_metrics.py # OA, macro F1, Cohen's κ, per-zone TPR/F1, MIA-AUC
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── data_loader.py           # Multi-modal ingestion, reprojection, feature matrix construction
│       ├── visualization.py         # All paper figures (GGI bars, TPR, privacy-utility, PCA, GWFA, CSD, AD, radar)
│       └── logging_utils.py         # Structured logging to file and console
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with CUDA support recommended)
- Opacus ≥ 1.4 (for DP-SGD)
- See `requirements.txt` for full list

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This executes all four phases sequentially:

1. **Phase 1** — Fuses multi-modal data, applies geo-stratified sampling, minority oversampling, spatial k-anonymity, and PCA compression
2. **Phase 2** — Trains a conditional GAN with DP-SGD (ε=5, δ=10⁻⁵) and the sEOD fairness penalty embedded in the discriminator loss
3. **Phase 3** — Runs TSTR evaluation, membership inference attack, spatial fidelity assessment
4. **Phase 4** — Generates GWFA attribution maps, CSD maps with SuDS placement recommendations, Attribution Divergence analysis, and three accountability documents

### Outputs

After running, the following directories are populated:

| Directory | Contents |
|---|---|
| `results/` | `consolidated_results.json` with all metrics |
| `models/` | `cgan_model.pt` — trained cGAN checkpoint |
| `accountability_docs/` | `model_card.json`, `synthetic_datasheet.json`, `provenance_certificate.json` |
| `figures/` | All visualisations (PNG) |
| `logs/` | Timestamped log files per module |

---

## Configuration

All hyperparameters are centralised in `config.yaml`. Key parameters:

| Parameter | Value | Section |
|---|---|---|
| PCA variance threshold | 0.999 | `data` |
| k-anonymity grid | 50m, k≥5 | `fairness` |
| DP-SGD ε | 5.0 | `privacy` |
| DP-SGD δ | 10⁻⁵ | `privacy` |
| Noise multiplier | 1.1 | `privacy` |
| Gradient clipping norm | 1.0 | `privacy` |
| cGAN latent dim | 128 | `cgan` |
| sEOD penalty λ | 0.1 | `cgan` |
| Learning rate | 2×10⁻⁴ | `cgan` |
| Feedback convergence Δ | 0.005 | `feedback` |
| Random seeds | 42, 123, 256, 512, 1024 | `project` |

---

## Six Novel Spatial Metrics

All implemented in `src/metrics/spatial_metrics.py`:

| Metric | Method | RAI Characteristic |
|---|---|---|
| Zonal Parity Ratio (ZPR) | `zonal_parity_ratio()` | Fairness |
| Provenance Completeness Score (PCS) | `provenance_completeness_score()` | Transparency |
| Modality Contribution Traceability (MCT) | `modality_contribution_traceability()` | Transparency |
| Geographically Weighted Feature Attribution (GWFA) | `geographically_weighted_feature_attribution()` | Explainability |
| Counterfactual Spatial Distance (CSD) | Implemented in `src/phase4/csd.py` | Explainability |
| Attribution Divergence (AD) | `attribution_divergence()` | Explainability |

Supporting metrics: Geographic Gini Index (GGI), Spatial Equal Opportunity Difference (sEOD), sFID, Variogram Fidelity, Moran's I.

---

## Using Your Own Data

Replace the `generate_study_data()` function in `main.py` with your own data loading logic. The pipeline expects:

```python
{
    "feature_blocks": {
        "sentinel2": np.ndarray,      # shape (n_samples, n_sentinel2_features)
        "lidar": np.ndarray,           # shape (n_samples, n_lidar_features)
        "flood_vectors": np.ndarray,   # shape (n_samples, n_flood_features)
        ...
    },
    "feature_names": {
        "sentinel2": ["band1", "band2", ...],
        "lidar": ["Elevation", "Slope", ...],
        ...
    },
    "y": np.ndarray,                   # integer class labels
    "zone_assignments": np.ndarray,    # integer zone IDs per sample
    "coordinates": np.ndarray,         # shape (n_samples, 2) — easting, northing
    "class_names": ["High", "Moderate", "Low", "Negligible"],
}
```

For raster/vector file ingestion, use `src/utils/data_loader.py`:

```python
from src.utils import DataLoader

loader = DataLoader(target_crs="EPSG:32630", resolution_m=10.0)
loader.ingest_raster("sentinel2.tif", modality_id="S2", timestamp="2024-06")
loader.ingest_vector("flood_zones.gpkg", modality_id="flood", timestamp="2024-01")
```

---

## Experiments Mapping

| Experiment | Section | Code Path |
|---|---|---|
| Exp 1: Spatial Bias Correction | §5.4 | `src/phase1/fairness.py` |
| Exp 2: Privacy and Traceability | §5.5 | `src/phase2/training.py`, `src/phase3/mia.py` |
| Exp 3: Classification Utility | §5.6 | `src/phase3/validation.py` |
| Exp 4: Spatial Decision Structure | §5.7 | `src/phase4/gwfa.py`, `src/phase4/csd.py`, `src/phase4/attribution_divergence.py` |
| Exp 5: Generalisation | §5.8 | `main.py` (Leeds section) |
| Ablation Study | §5.9 | `src/phase3/ablation.py` |
| Embedded vs Post-hoc | §5.10 | `src/phase3/ablation.py` |
| Feedback Convergence | §5.11 | `src/phase3/feedback.py` |

---

## Citation

```bibtex
@article{hassan2026multimodal,
  title={Multi-Modal Geospatial Data Fusion and Generation with Responsible AI Practice},
  author={Hassan, Muhammad and Sardar, Bilal and Islam, Shareeful and Imani, Maryam and Hakiri, Akram},
  journal={Transactions in GIS},
  year={2026},
  doi={10.1002/0000}
}
```

---

## Acknowledgements

This work was supported by UK Research and Innovation (UKRI) through the CHIST-ERA Project, AI4MultiGIS — AI integrated framework for intelligent geospatial handling and robust operation in MultiGIS applications, funded under grant number EP/Z003490/1.

---

## License

Research use. Contact the corresponding author (shareeful.islam@aru.ac.uk) for licensing.
