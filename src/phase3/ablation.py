import numpy as np
from typing import Dict, List
from copy import deepcopy

from .validation import Validator
from .mia import MembershipInferenceAttack
from ..utils.logging_utils import setup_logger

logger = setup_logger("ablation")


class AblationRunner:
    def __init__(self, config: Dict):
        self.config = config

    def run_condition(self, condition_id: str, X: np.ndarray, y: np.ndarray,
                      zone_assignments: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      zone_test: np.ndarray,
                      pca_loading_matrix: np.ndarray,
                      class_names: List[str]) -> Dict:

        modified_config = deepcopy(self.config)

        use_dp = True
        use_seod = True

        if condition_id == "A1":
            modified_config["cgan"]["lambda_seod"] = 0.0
            use_seod = False
        elif condition_id == "A2":
            use_dp = False
        elif condition_id == "A3":
            pass
        elif condition_id == "A4":
            pass

        logger.info(f"Running ablation condition {condition_id}")

        from ..phase2.training import CGANTrainer
        trainer = CGANTrainer(modified_config)
        train_result = trainer.train(
            X, y, zone_assignments, pca_loading_matrix,
            use_dp=use_dp, use_seod=use_seod,
        )

        n_per_class = len(X) // len(np.unique(y))
        syn_data = trainer.generate_synthetic_dataset(n_per_class, pca_loading_matrix)

        validator = Validator(modified_config)
        tstr_results = validator.tstr_evaluation(
            syn_data["X"], syn_data["y"], X_test, y_test, zone_test, class_names
        )

        mia = MembershipInferenceAttack()
        mia_result = {"mia_auc": 0.5}
        if tstr_results.get("best_classifier") is not None:
            n_eval = min(len(syn_data["X"]), len(X_test))
            mia_result = mia.evaluate(
                tstr_results["best_classifier"],
                syn_data["X"][:n_eval], syn_data["y"][:n_eval],
                X_test[:n_eval], y_test[:n_eval],
            )

        return {
            "condition_id": condition_id,
            "macro_f1": tstr_results.get("macro_f1_mean", 0),
            "seod": tstr_results.get("seod_mean", 0),
            "zpr": tstr_results.get("zpr_mean", 0),
            "mia_auc": mia_result.get("mia_auc", 0.5),
            "pcs": syn_data.get("pcs", 1.0),
        }

    def run_all(self, X: np.ndarray, y: np.ndarray,
                zone_assignments: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                zone_test: np.ndarray,
                pca_loading_matrix: np.ndarray,
                class_names: List[str]) -> Dict[str, Dict]:

        conditions = ["A0", "A1", "A2", "A3", "A4"]
        results = {}

        for cond in conditions:
            results[cond] = self.run_condition(
                cond, X, y, zone_assignments,
                X_test, y_test, zone_test,
                pca_loading_matrix, class_names,
            )

        return results

    def run_posthoc_comparison(self, X_syn_embedded: np.ndarray, y_syn_embedded: np.ndarray,
                                X_syn_posthoc: np.ndarray, y_syn_posthoc: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                zone_test: np.ndarray,
                                class_names: List[str]) -> Dict:

        validator = Validator(self.config)

        embedded_results = validator.tstr_evaluation(
            X_syn_embedded, y_syn_embedded, X_test, y_test, zone_test, class_names
        )

        posthoc_results = validator.tstr_evaluation(
            X_syn_posthoc, y_syn_posthoc, X_test, y_test, zone_test, class_names
        )

        return {
            "embedded": {
                "macro_f1": embedded_results.get("macro_f1_mean", 0),
                "seod": embedded_results.get("seod_mean", 0),
                "zpr": embedded_results.get("zpr_mean", 0),
            },
            "posthoc": {
                "macro_f1": posthoc_results.get("macro_f1_mean", 0),
                "seod": posthoc_results.get("seod_mean", 0),
                "zpr": posthoc_results.get("zpr_mean", 0),
            },
        }
