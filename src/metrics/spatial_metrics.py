import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional


class SpatialMetrics:

    @staticmethod
    def geographic_gini_index(zone_coverages: np.ndarray) -> float:
        n = len(zone_coverages)
        if n == 0:
            return 0.0
        c_bar = np.mean(zone_coverages)
        if c_bar == 0:
            return 0.0
        abs_diffs = np.abs(zone_coverages[:, None] - zone_coverages[None, :])
        ggi = np.sum(abs_diffs) / (2 * n * n * c_bar)
        return float(ggi)

    @staticmethod
    def spatial_equal_opportunity_difference(tpr_per_zone: np.ndarray) -> float:
        if len(tpr_per_zone) < 2:
            return 0.0
        return float(np.max(tpr_per_zone) - np.min(tpr_per_zone))

    @staticmethod
    def zonal_parity_ratio(zone_f1_scores: np.ndarray, threshold: float = 0.10) -> float:
        if len(zone_f1_scores) == 0:
            return 0.0
        global_mean = np.mean(zone_f1_scores)
        within_threshold = np.abs(zone_f1_scores - global_mean) <= threshold
        return float(np.mean(within_threshold))

    @staticmethod
    def provenance_completeness_score(provenance_flags: np.ndarray) -> float:
        if len(provenance_flags) == 0:
            return 0.0
        return float(np.mean(provenance_flags))

    @staticmethod
    def modality_contribution_traceability(pca_loading_matrix: np.ndarray,
                                            modality_boundaries: List[Tuple[int, int]]) -> np.ndarray:
        n_components = pca_loading_matrix.shape[0]
        mct_scores = np.zeros(n_components)

        for pc_idx in range(n_components):
            loadings = np.abs(pca_loading_matrix[pc_idx])
            modality_contributions = []

            for start, end in modality_boundaries:
                modality_contributions.append(np.sum(loadings[start:end]))

            modality_contributions = np.array(modality_contributions)
            total = np.sum(modality_contributions)

            if total == 0:
                mct_scores[pc_idx] = 0.0
                continue

            modality_contributions = modality_contributions / total
            mct_scores[pc_idx] = float(np.max(modality_contributions))

        return mct_scores

    @staticmethod
    def geographically_weighted_feature_attribution(
        X: np.ndarray,
        y: np.ndarray,
        coordinates: np.ndarray,
        target_locations: np.ndarray,
        bandwidth: float,
        adaptive: bool = True,
    ) -> np.ndarray:
        n_targets = target_locations.shape[0]
        n_features = X.shape[1]
        attributions = np.zeros((n_targets, n_features))

        for idx in range(n_targets):
            target = target_locations[idx]
            distances = np.sqrt(np.sum((coordinates - target) ** 2, axis=1))

            if adaptive:
                sorted_dists = np.sort(distances)
                k = max(int(0.1 * len(distances)), 20)
                h = sorted_dists[min(k, len(sorted_dists) - 1)]
                h = max(h, 1e-6)
            else:
                h = bandwidth

            weights = np.exp(-distances ** 2 / (2 * h ** 2))
            weights = weights / (np.sum(weights) + 1e-12)

            X_weighted = X * weights[:, np.newaxis]

            feature_importances = np.zeros(n_features)
            for f in range(n_features):
                correlation = np.abs(np.corrcoef(X_weighted[:, f], y)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
                weighted_var = np.average((X[:, f] - np.average(X[:, f], weights=weights)) ** 2,
                                          weights=weights)
                feature_importances[f] = correlation * np.sqrt(weighted_var + 1e-12)

            total = np.sum(feature_importances)
            if total > 0:
                feature_importances = feature_importances / total

            attributions[idx] = feature_importances

        return attributions

    @staticmethod
    def counterfactual_spatial_distance(
        X: np.ndarray,
        predictions: np.ndarray,
        classifier,
        target_class_offset: int = -1,
    ) -> np.ndarray:
        n_samples = X.shape[0]
        csd_values = np.zeros(n_samples)
        unique_classes = np.unique(predictions)

        for idx in range(n_samples):
            current_class = predictions[idx]
            target_class_val = current_class + target_class_offset

            if target_class_val not in unique_classes:
                target_class_val = current_class - target_class_offset
            if target_class_val not in unique_classes:
                csd_values[idx] = float("inf")
                continue

            x_original = X[idx].copy()
            best_delta_norm = float("inf")

            n_features = X.shape[1]
            for step_size in [0.01, 0.05, 0.1, 0.25, 0.5]:
                for feat_idx in range(n_features):
                    for direction in [-1, 1]:
                        x_modified = x_original.copy()
                        for mult in range(1, 50):
                            x_modified[feat_idx] = x_original[feat_idx] + direction * step_size * mult
                            pred = classifier.predict(x_modified.reshape(1, -1))[0]
                            if pred == target_class_val:
                                delta_norm = np.linalg.norm(x_modified - x_original)
                                if delta_norm < best_delta_norm:
                                    best_delta_norm = delta_norm
                                break

            csd_values[idx] = best_delta_norm if best_delta_norm != float("inf") else np.nan

        return csd_values

    @staticmethod
    def attribution_divergence(
        attributions_real: np.ndarray,
        attributions_synthetic: np.ndarray,
    ) -> np.ndarray:
        n_locations = attributions_real.shape[0]
        ad_values = np.zeros(n_locations)

        for idx in range(n_locations):
            p = attributions_real[idx] + 1e-12
            q = attributions_synthetic[idx] + 1e-12
            p = p / np.sum(p)
            q = q / np.sum(q)
            ad_values[idx] = float(entropy(p, q))

        return ad_values

    @staticmethod
    def compute_sfid(real_features: np.ndarray, synthetic_features: np.ndarray) -> float:
        mu_real = np.mean(real_features, axis=0)
        mu_syn = np.mean(synthetic_features, axis=0)
        cov_real = np.cov(real_features, rowvar=False)
        cov_syn = np.cov(synthetic_features, rowvar=False)

        diff = mu_real - mu_syn
        mean_diff_sq = np.dot(diff, diff)

        cov_product = cov_real @ cov_syn
        eigenvalues = np.linalg.eigvalsh(cov_product)
        eigenvalues = np.maximum(eigenvalues, 0)
        sqrt_cov_product_trace = np.sum(np.sqrt(eigenvalues))

        sfid = mean_diff_sq + np.trace(cov_real) + np.trace(cov_syn) - 2 * sqrt_cov_product_trace
        return float(max(sfid, 0.0))

    @staticmethod
    def variogram_fidelity(real_coords: np.ndarray, real_values: np.ndarray,
                           syn_coords: np.ndarray, syn_values: np.ndarray,
                           n_bins: int = 20) -> float:
        def compute_variogram(coords, values, n_bins):
            n = min(len(coords), 2000)
            idx = np.random.choice(len(coords), n, replace=False)
            coords_sub = coords[idx]
            values_sub = values[idx]

            dists = cdist(coords_sub, coords_sub)
            upper_tri = np.triu_indices(n, k=1)
            flat_dists = dists[upper_tri]
            flat_diffs = (values_sub[upper_tri[0]] - values_sub[upper_tri[1]]) ** 2

            bins = np.linspace(0, np.percentile(flat_dists, 90), n_bins + 1)
            gamma = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (flat_dists >= bins[b]) & (flat_dists < bins[b + 1])
                if np.sum(mask) > 0:
                    gamma[b] = 0.5 * np.mean(flat_diffs[mask])
            return gamma

        gamma_real = compute_variogram(real_coords, real_values, n_bins)
        gamma_syn = compute_variogram(syn_coords, syn_values, n_bins)

        mask = (gamma_real > 0) | (gamma_syn > 0)
        if np.sum(mask) == 0:
            return 1.0

        correlation = np.corrcoef(gamma_real[mask], gamma_syn[mask])[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(max(correlation, 0.0))

    @staticmethod
    def morans_i(values: np.ndarray, coordinates: np.ndarray, k_neighbours: int = 8) -> float:
        n = len(values)
        if n < 3:
            return 0.0

        dists = cdist(coordinates, coordinates)
        W = np.zeros((n, n))
        for i in range(n):
            nearest = np.argsort(dists[i])[1:k_neighbours + 1]
            W[i, nearest] = 1.0

        W = (W + W.T) / 2
        z = values - np.mean(values)
        numerator = n * np.sum(W * np.outer(z, z))
        denominator = np.sum(W) * np.sum(z ** 2)

        if denominator == 0:
            return 0.0
        return float(numerator / denominator)
