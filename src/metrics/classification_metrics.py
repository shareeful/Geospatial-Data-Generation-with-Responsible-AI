import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from typing import Dict, List, Tuple


class ClassificationMetrics:

    @staticmethod
    def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    @staticmethod
    def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(cohen_kappa_score(y_true, y_pred))

    @staticmethod
    def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str]) -> Dict:
        report = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
        return report

    @staticmethod
    def per_zone_tpr(y_true: np.ndarray, y_pred: np.ndarray,
                     zone_assignments: np.ndarray,
                     target_class: int) -> np.ndarray:
        unique_zones = np.unique(zone_assignments)
        tpr_per_zone = np.zeros(len(unique_zones))

        for idx, zone in enumerate(unique_zones):
            mask = zone_assignments == zone
            yt = y_true[mask]
            yp = y_pred[mask]

            positives = yt == target_class
            if np.sum(positives) == 0:
                tpr_per_zone[idx] = 0.0
            else:
                tpr_per_zone[idx] = np.sum((yp == target_class) & positives) / np.sum(positives)

        return tpr_per_zone

    @staticmethod
    def per_zone_f1(y_true: np.ndarray, y_pred: np.ndarray,
                    zone_assignments: np.ndarray) -> np.ndarray:
        unique_zones = np.unique(zone_assignments)
        f1_per_zone = np.zeros(len(unique_zones))

        for idx, zone in enumerate(unique_zones):
            mask = zone_assignments == zone
            if np.sum(mask) == 0:
                f1_per_zone[idx] = 0.0
                continue
            f1_per_zone[idx] = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)

        return f1_per_zone

    @staticmethod
    def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                        zone_assignments: np.ndarray,
                        class_names: List[str],
                        target_class: int = 0) -> Dict:
        oa = ClassificationMetrics.overall_accuracy(y_true, y_pred)
        f1 = ClassificationMetrics.macro_f1(y_true, y_pred)
        kappa = ClassificationMetrics.cohens_kappa(y_true, y_pred)
        zone_f1 = ClassificationMetrics.per_zone_f1(y_true, y_pred, zone_assignments)
        zone_tpr = ClassificationMetrics.per_zone_tpr(y_true, y_pred, zone_assignments, target_class)

        from ..metrics.spatial_metrics import SpatialMetrics
        zpr = SpatialMetrics.zonal_parity_ratio(zone_f1)
        seod = SpatialMetrics.spatial_equal_opportunity_difference(zone_tpr)

        return {
            "overall_accuracy": oa,
            "macro_f1": f1,
            "cohens_kappa": kappa,
            "zpr": zpr,
            "seod": seod,
            "zone_f1_scores": zone_f1,
            "zone_tpr_scores": zone_tpr,
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    @staticmethod
    def mia_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
        labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
        scores = np.concatenate([member_scores, nonmember_scores])
        return float(roc_auc_score(labels, scores))
