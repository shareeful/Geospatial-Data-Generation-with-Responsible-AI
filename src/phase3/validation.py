import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple

from ..metrics.classification_metrics import ClassificationMetrics
from ..metrics.spatial_metrics import SpatialMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("validation")


class Validator:
    def __init__(self, config: Dict):
        val_cfg = config.get("validation", {})
        self.n_splits = val_cfg.get("tstr_splits", 5)
        self.n_estimators = val_cfg.get("classifier_n_estimators", 500)
        self.max_depth = val_cfg.get("classifier_max_depth", 20)
        self.seeds = config.get("project", {}).get("seed_list", [42, 123, 256, 512, 1024])

    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                          seed: int = 42) -> RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        return clf

    def tstr_evaluation(self, X_synthetic: np.ndarray, y_synthetic: np.ndarray,
                         X_real_test: np.ndarray, y_real_test: np.ndarray,
                         zone_test: np.ndarray,
                         class_names: List[str]) -> Dict:
        results_per_seed = []

        for seed in self.seeds:
            clf = self.train_classifier(X_synthetic, y_synthetic, seed=seed)
            y_pred = clf.predict(X_real_test)

            metrics = ClassificationMetrics.full_evaluation(
                y_real_test, y_pred, zone_test, class_names
            )
            metrics["seed"] = seed
            metrics["classifier"] = clf
            results_per_seed.append(metrics)

        agg = self._aggregate_results(results_per_seed)
        logger.info(
            f"TSTR results: OA={agg['overall_accuracy_mean']:.4f}±{agg['overall_accuracy_std']:.4f}, "
            f"F1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}, "
            f"κ={agg['cohens_kappa_mean']:.4f}±{agg['cohens_kappa_std']:.4f}"
        )
        return agg

    def real_baseline_evaluation(self, X_real: np.ndarray, y_real: np.ndarray,
                                  zone_assignments: np.ndarray,
                                  class_names: List[str]) -> Dict:
        results_per_seed = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_real, y_real)):
            seed = self.seeds[fold_idx % len(self.seeds)]
            clf = self.train_classifier(X_real[train_idx], y_real[train_idx], seed=seed)
            y_pred = clf.predict(X_real[test_idx])

            metrics = ClassificationMetrics.full_evaluation(
                y_real[test_idx], y_pred, zone_assignments[test_idx], class_names
            )
            metrics["seed"] = seed
            metrics["classifier"] = clf
            results_per_seed.append(metrics)

        return self._aggregate_results(results_per_seed)

    def single_source_evaluation(self, X_source: np.ndarray, y: np.ndarray,
                                  zone_assignments: np.ndarray,
                                  class_names: List[str],
                                  source_name: str) -> Dict:
        results = self.real_baseline_evaluation(X_source, y, zone_assignments, class_names)
        results["source_name"] = source_name
        logger.info(f"Single-source ({source_name}): F1={results['macro_f1_mean']:.4f}")
        return results

    def _aggregate_results(self, results_list: List[Dict]) -> Dict:
        metric_keys = ["overall_accuracy", "macro_f1", "cohens_kappa", "zpr", "seod"]
        agg = {}

        for key in metric_keys:
            values = [r[key] for r in results_list if key in r]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))

        agg["all_results"] = results_list
        if results_list and "classifier" in results_list[0]:
            agg["best_classifier"] = results_list[0]["classifier"]

        return agg

    def compare_regimes(self, regime_results: Dict[str, Dict]) -> Dict:
        comparison = {}
        for name, results in regime_results.items():
            comparison[name] = {
                "OA": results.get("overall_accuracy_mean", 0),
                "F1": results.get("macro_f1_mean", 0),
                "kappa": results.get("cohens_kappa_mean", 0),
            }
        return comparison
