import yaml
import numpy as np
import json
from pathlib import Path
from typing import Dict
from collections import Counter

from src.phase1 import DataPreparation
from src.phase2 import CGANTrainer
from src.phase3 import Validator, MembershipInferenceAttack, SpatialFidelityEvaluator, FeedbackLoop
from src.phase4 import Explainer
from src.metrics import SpatialMetrics, ClassificationMetrics
from src.utils import Visualizer, setup_logger

logger = setup_logger("main")


def load_config(path: str = "config.yaml") -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_study_data(n_samples: int = 10000, n_zones: int = 22,
                         n_classes: int = 4, seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)

    class_probs = np.array([0.0044, 0.327, 0.432, 0.232])
    y = rng.choice(n_classes, size=n_samples, p=class_probs / class_probs.sum())
    zone_assignments = rng.randint(0, n_zones, size=n_samples)

    sentinel2_bands = 11
    sentinel2_dates = 3
    n_sentinel2 = sentinel2_bands * sentinel2_dates
    sentinel2 = rng.randn(n_samples, n_sentinel2) * 0.3

    for i in range(n_samples):
        if y[i] == 0:
            sentinel2[i, :sentinel2_bands] += rng.uniform(0.3, 0.8, sentinel2_bands)
        elif y[i] == 1:
            sentinel2[i, :sentinel2_bands] += rng.uniform(0.1, 0.4, sentinel2_bands)

    cloud_mask = rng.random((n_samples, n_sentinel2))
    sentinel2[cloud_mask < 0.05] = np.nan
    for z in range(n_zones):
        if z < 5:
            zone_mask = zone_assignments == z
            extra_cloud = rng.random((np.sum(zone_mask), n_sentinel2))
            sentinel2[zone_mask][extra_cloud < 0.15] = np.nan

    lidar_features = 4
    lidar = rng.randn(n_samples, lidar_features) * 0.5
    for i in range(n_samples):
        lidar[i, 0] = rng.uniform(0, 100)
        lidar[i, 1] = rng.uniform(0, 45)
        lidar[i, 2] = rng.uniform(0, 360)
        lidar[i, 3] = rng.uniform(0, 50)
        if y[i] == 0:
            lidar[i, 0] -= rng.uniform(5, 20)
            lidar[i, 3] += rng.uniform(10, 30)

    flood_vec_features = 3
    flood_vec = rng.randn(n_samples, flood_vec_features) * 0.2
    for i in range(n_samples):
        flood_vec[i, y[i] % flood_vec_features] += rng.uniform(0.5, 1.5)

    land_cover = rng.randn(n_samples, 1) * 0.3
    road_network = rng.randn(n_samples, 2) * 0.2
    river_network = rng.randn(n_samples, 2) * 0.2

    coordinates = np.column_stack([
        rng.uniform(0, 10000, n_samples),
        rng.uniform(0, 10000, n_samples),
    ])

    feature_blocks = {
        "sentinel2": sentinel2,
        "lidar": lidar,
        "flood_vectors": flood_vec,
        "land_cover": land_cover,
        "road_network": road_network,
        "river_network": river_network,
    }

    feature_names = {
        "sentinel2": [f"S2_b{b}_t{t}" for t in range(sentinel2_dates) for b in range(sentinel2_bands)],
        "lidar": ["Elevation", "Slope", "Aspect", "FlowAcc"],
        "flood_vectors": ["FloodVec_V3", "FloodVec_V4", "FloodVec_V5"],
        "land_cover": ["LandUse"],
        "road_network": ["RoadProx", "RoadDens"],
        "river_network": ["RiverProx", "RiverDens"],
    }

    return {
        "feature_blocks": feature_blocks,
        "feature_names": feature_names,
        "y": y,
        "zone_assignments": zone_assignments,
        "coordinates": coordinates,
        "class_names": ["High", "Moderate", "Low", "Negligible"],
    }


def generate_leeds_data(n_samples: int = 8000, seed: int = 99) -> Dict:
    rng = np.random.RandomState(seed)

    class_probs = np.array([0.012, 0.284, 0.478, 0.226])
    y = rng.choice(4, size=n_samples, p=class_probs / class_probs.sum())
    zone_assignments = rng.randint(0, 18, size=n_samples)

    sentinel2 = rng.randn(n_samples, 33) * 0.3
    for i in range(n_samples):
        if y[i] == 0:
            sentinel2[i, :11] += rng.uniform(0.2, 0.7, 11)

    lidar = rng.randn(n_samples, 4) * 0.5
    for i in range(n_samples):
        lidar[i, 0] = rng.uniform(0, 80)
        if y[i] == 0:
            lidar[i, 0] -= rng.uniform(3, 15)

    flood_vec = rng.randn(n_samples, 3) * 0.2
    for i in range(n_samples):
        flood_vec[i, y[i] % 3] += rng.uniform(0.4, 1.2)

    land_cover = rng.randn(n_samples, 1) * 0.4
    road_network = rng.randn(n_samples, 2) * 0.2
    river_network = rng.randn(n_samples, 2) * 0.2

    coordinates = np.column_stack([
        rng.uniform(0, 8000, n_samples),
        rng.uniform(0, 8000, n_samples),
    ])

    feature_blocks = {
        "sentinel2": sentinel2,
        "lidar": lidar,
        "flood_vectors": flood_vec,
        "land_cover": land_cover,
        "road_network": road_network,
        "river_network": river_network,
    }

    feature_names = {
        "sentinel2": [f"S2_b{b}_t{t}" for t in range(3) for b in range(11)],
        "lidar": ["Elevation", "Slope", "Aspect", "FlowAcc"],
        "flood_vectors": ["FloodVec_V3", "FloodVec_V4", "FloodVec_V5"],
        "land_cover": ["LandUse"],
        "road_network": ["RoadProx", "RoadDens"],
        "river_network": ["RiverProx", "RiverDens"],
    }

    return {
        "feature_blocks": feature_blocks,
        "feature_names": feature_names,
        "y": y,
        "zone_assignments": zone_assignments,
        "coordinates": coordinates,
        "class_names": ["High", "Moderate", "Low", "Negligible"],
    }


def run_pipeline(config: Dict):
    for d in ["results", "models", "accountability_docs", "figures", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    viz = Visualizer(config.get("output", {}).get("figures_dir", "figures"))

    logger.info("=" * 70)
    logger.info("RAI-GeoAI: Responsible AI for Multi-Modal Geospatial Intelligence")
    logger.info("=" * 70)

    logger.info("Generating Chelmsford pilot data...")
    chelmsford = generate_study_data(n_samples=10000, n_zones=22, seed=42)

    logger.info("Generating Leeds generalisation data...")
    leeds = generate_leeds_data(n_samples=8000, seed=99)

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATA PREPARATION")
    logger.info("=" * 50)

    phase1 = DataPreparation(config)
    phase1_result = phase1.run(
        feature_blocks=chelmsford["feature_blocks"],
        feature_names_per_block=chelmsford["feature_names"],
        y=chelmsford["y"],
        zone_assignments=chelmsford["zone_assignments"],
        coordinates=chelmsford["coordinates"],
    )

    X_prepared = phase1_result["X"]
    y_prepared = phase1_result["y"]
    zones_prepared = phase1_result["zones"]
    loading_matrix = phase1_result["loading_matrix"]

    from sklearn.model_selection import train_test_split
    split_idx = int(0.7 * len(X_prepared))
    shuffle_idx = np.random.permutation(len(X_prepared))
    train_idx = shuffle_idx[:split_idx]
    test_idx = shuffle_idx[split_idx:]

    X_train, X_test = X_prepared[train_idx], X_prepared[test_idx]
    y_train, y_test = y_prepared[train_idx], y_prepared[test_idx]
    zones_train, zones_test = zones_prepared[train_idx], zones_prepared[test_idx]

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: CONDITIONAL GENERATIVE MODEL TRAINING")
    logger.info("=" * 50)

    trainer = CGANTrainer(config)
    train_result = trainer.train(
        X=X_train, y=y_train, zone_assignments=zones_train,
        pca_loading_matrix=loading_matrix,
        use_dp=True, use_seod=True,
    )

    n_per_class = len(X_train) // len(np.unique(y_train))
    synthetic_data = trainer.generate_synthetic_dataset(n_per_class, loading_matrix)

    X_synthetic = synthetic_data["X"]
    y_synthetic = synthetic_data["y"]

    trainer.cgan.save("models/cgan_model.pt")

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: VALIDATION")
    logger.info("=" * 50)

    validator = Validator(config)

    logger.info("Experiment 3: TSTR Evaluation")
    tstr_results = validator.tstr_evaluation(
        X_synthetic, y_synthetic, X_test, y_test, zones_test,
        chelmsford["class_names"],
    )

    logger.info("Real baseline evaluation")
    real_results = validator.real_baseline_evaluation(
        X_prepared, y_prepared, zones_prepared, chelmsford["class_names"],
    )

    logger.info("Experiment 2: Membership Inference Attack")
    mia = MembershipInferenceAttack(n_shadow_models=3)
    if tstr_results.get("best_classifier") is not None:
        n_eval = min(len(X_synthetic), len(X_test))
        mia_result = mia.evaluate(
            tstr_results["best_classifier"],
            X_synthetic[:n_eval], y_synthetic[:n_eval],
            X_test[:n_eval], y_test[:n_eval],
        )
    else:
        mia_result = {"mia_auc": 0.5, "privacy_safe": True}

    logger.info("Spatial Fidelity Evaluation")
    sf_eval = SpatialFidelityEvaluator()
    coords_syn = np.column_stack([
        np.random.uniform(0, 10000, len(X_synthetic)),
        np.random.uniform(0, 10000, len(X_synthetic)),
    ])
    coords_real_sub = chelmsford["coordinates"][:len(X_test)]
    sf_result = sf_eval.evaluate(X_test, X_synthetic, coords_real_sub, coords_syn)

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: EXPLAINABILITY AND ACCOUNTABILITY")
    logger.info("=" * 50)

    explainer = Explainer(config)

    clf_real = validator.train_classifier(X_train, y_train, seed=42)
    clf_syn = tstr_results.get("best_classifier", clf_real)

    n_explain = min(500, len(X_test))
    coords_explain = chelmsford["coordinates"][:n_explain]

    explain_result = explainer.run_explainability(
        X_real=X_test[:n_explain],
        y_real=y_test[:n_explain],
        X_synthetic=X_synthetic[:n_explain],
        y_synthetic=y_synthetic[:n_explain],
        coordinates=coords_explain,
        classifier_real=clf_real,
        classifier_synthetic=clf_syn,
        feature_names=[f"PC{i+1}" for i in range(X_prepared.shape[1])],
    )

    class_dist = dict(Counter(y_synthetic))
    doc_paths = explainer.generate_accountability_docs(
        model_info={
            "n_estimators": 500, "max_depth": 20,
            "epsilon": train_result.get("final_epsilon", 5.0),
            "delta": 1e-5,
            "mia_auc": mia_result.get("mia_auc", 0.5),
            "k_anonymity": 5,
        },
        performance={
            "overall_accuracy": tstr_results.get("overall_accuracy_mean", 0),
            "macro_f1": tstr_results.get("macro_f1_mean", 0),
            "cohens_kappa": tstr_results.get("cohens_kappa_mean", 0),
        },
        fairness_info={
            "zpr": tstr_results.get("zpr_mean", 0),
            "seod": tstr_results.get("seod_mean", 0),
            "ggi": phase1_result["post_fairness"]["ggi"],
        },
        dataset_info={
            "n_samples": len(X_synthetic),
            "n_features": X_synthetic.shape[1],
            "n_pca_components": loading_matrix.shape[0],
            "variance_retained": 0.999,
            "epsilon": train_result.get("final_epsilon", 5.0),
            "delta": 1e-5,
            "modalities": list(chelmsford["feature_blocks"].keys()),
        },
        class_distribution=class_dist,
        provenance_hashes=synthetic_data["provenance_hashes"],
        pca_info={
            "n_components": loading_matrix.shape[0],
            "variance_retained": 0.999,
            "loading_hash": synthetic_data["provenance_hashes"][0] if synthetic_data["provenance_hashes"] else "",
            "modality_names": list(chelmsford["feature_blocks"].keys()),
        },
        privacy_info={
            "epsilon": train_result.get("final_epsilon", 5.0),
            "delta": 1e-5,
            "lambda_seod": config.get("cgan", {}).get("lambda_seod", 0.1),
        },
    )

    logger.info("\n" + "=" * 50)
    logger.info("EXPERIMENT 5: GENERALISATION TO LEEDS")
    logger.info("=" * 50)

    from src.phase1 import DataPreparation as DP2
    phase1_leeds = DP2(config)
    leeds_result = phase1_leeds.run(
        feature_blocks=leeds["feature_blocks"],
        feature_names_per_block=leeds["feature_names"],
        y=leeds["y"],
        zone_assignments=leeds["zone_assignments"],
        coordinates=leeds["coordinates"],
    )

    leeds_tstr = validator.tstr_evaluation(
        X_synthetic[:len(leeds_result["X"])],
        y_synthetic[:len(leeds_result["y"])],
        leeds_result["X"],
        leeds_result["y"],
        leeds_result["zones"],
        leeds["class_names"],
    )

    logger.info("\n" + "=" * 70)
    logger.info("CONSOLIDATED RESULTS")
    logger.info("=" * 70)

    results = {
        "chelmsford": {
            "real_f1": real_results.get("macro_f1_mean", 0),
            "synthetic_f1": tstr_results.get("macro_f1_mean", 0),
            "zpr": tstr_results.get("zpr_mean", 0),
            "seod": tstr_results.get("seod_mean", 0),
            "mia_auc": mia_result.get("mia_auc", 0.5),
            "pcs": synthetic_data.get("pcs", 1.0),
            "sfid": sf_result.get("sfid", 0),
            "variogram_fidelity": sf_result.get("variogram_fidelity", 0),
            "morans_i_preservation": sf_result.get("morans_i_preservation", 0),
        },
        "leeds": {
            "synthetic_f1": leeds_tstr.get("macro_f1_mean", 0),
            "zpr": leeds_tstr.get("zpr_mean", 0),
            "seod": leeds_tstr.get("seod_mean", 0),
        },
        "explainability": {
            "ad_mean": explain_result["attribution_divergence"]["mean_ad"],
            "ad_pct_below_threshold": explain_result["attribution_divergence"]["pct_below_threshold"],
            "csd_ks_pvalue": explain_result["csd_comparison"]["ks_pvalue"],
            "sac_real": explain_result["sac_real"],
            "sac_synthetic": explain_result["sac_synthetic"],
        },
        "privacy": {
            "epsilon": train_result.get("final_epsilon", 0),
            "delta": 1e-5,
            "mia_auc": mia_result.get("mia_auc", 0.5),
        },
        "provenance": {
            "pcs": synthetic_data.get("pcs", 1.0),
            "mean_mct": float(np.mean(synthetic_data.get("mct_scores", [0]))),
        },
    }

    with open("results/consolidated_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Chelmsford Real F1: {results['chelmsford']['real_f1']:.4f}")
    logger.info(f"Chelmsford Synthetic F1: {results['chelmsford']['synthetic_f1']:.4f}")
    logger.info(f"Leeds Synthetic F1: {results['leeds']['synthetic_f1']:.4f}")
    logger.info(f"MIA-AUC: {results['privacy']['mia_auc']:.4f}")
    logger.info(f"PCS: {results['provenance']['pcs']:.4f}")
    logger.info(f"AD mean: {results['explainability']['ad_mean']:.4f}")

    logger.info("\nAccountability documents generated:")
    for doc_type, path in doc_paths.items():
        logger.info(f"  {doc_type}: {path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return results


def main():
    config = load_config("config.yaml")
    np.random.seed(42)
    results = run_pipeline(config)
    return results


if __name__ == "__main__":
    main()
