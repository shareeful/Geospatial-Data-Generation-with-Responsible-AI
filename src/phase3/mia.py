import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from ..metrics.classification_metrics import ClassificationMetrics
from ..utils.logging_utils import setup_logger

logger = setup_logger("mia")


class MembershipInferenceAttack:
    def __init__(self, n_shadow_models: int = 5):
        self.n_shadow_models = n_shadow_models

    def _train_shadow_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             seed: int) -> RandomForestClassifier:
        shadow = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        shadow.fit(X_train, y_train)
        return shadow

    def _get_confidence_scores(self, model: RandomForestClassifier,
                                X: np.ndarray, y: np.ndarray) -> np.ndarray:
        probas = model.predict_proba(X)
        scores = np.zeros(len(X))
        classes = model.classes_

        for i in range(len(X)):
            class_idx = np.where(classes == y[i])[0]
            if len(class_idx) > 0:
                scores[i] = probas[i, class_idx[0]]
            else:
                scores[i] = np.max(probas[i])

        return scores

    def evaluate(self, target_model: RandomForestClassifier,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray) -> Dict:

        member_scores = self._get_confidence_scores(target_model, X_train, y_train)
        nonmember_scores = self._get_confidence_scores(target_model, X_test, y_test)

        n_eval = min(len(member_scores), len(nonmember_scores))
        member_scores = member_scores[:n_eval]
        nonmember_scores = nonmember_scores[:n_eval]

        mia_auc = ClassificationMetrics.mia_auc(member_scores, nonmember_scores)

        attack_features = np.concatenate([
            member_scores.reshape(-1, 1),
            nonmember_scores.reshape(-1, 1),
        ])
        attack_labels = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(nonmember_scores)),
        ])

        attack_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        idx = np.random.permutation(len(attack_labels))
        split = int(0.7 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]

        attack_clf.fit(attack_features[train_idx], attack_labels[train_idx])
        attack_preds = attack_clf.predict(attack_features[test_idx])
        attack_accuracy = np.mean(attack_preds == attack_labels[test_idx])

        logger.info(f"MIA-AUC: {mia_auc:.4f}, Attack accuracy: {attack_accuracy:.4f}")

        return {
            "mia_auc": mia_auc,
            "attack_accuracy": attack_accuracy,
            "member_confidence_mean": float(np.mean(member_scores)),
            "nonmember_confidence_mean": float(np.mean(nonmember_scores)),
            "privacy_safe": mia_auc < 0.55,
        }

    def full_shadow_evaluation(self, X: np.ndarray, y: np.ndarray,
                                seeds: list) -> Dict:
        all_auc = []

        for seed in seeds[:self.n_shadow_models]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )

            shadow = self._train_shadow_model(X_train, y_train, seed)
            result = self.evaluate(shadow, X_train, y_train, X_test, y_test)
            all_auc.append(result["mia_auc"])

        return {
            "mia_auc_mean": float(np.mean(all_auc)),
            "mia_auc_std": float(np.std(all_auc)),
            "all_auc": all_auc,
            "privacy_safe": np.mean(all_auc) < 0.55,
        }
