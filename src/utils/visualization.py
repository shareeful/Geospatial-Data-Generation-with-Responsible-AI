import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Visualizer:
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        })

    def plot_ggi_comparison(self, coverage_original: np.ndarray,
                            coverage_corrected: np.ndarray,
                            coverage_synthetic: np.ndarray,
                            ggi_values: Tuple[float, float, float],
                            ward_labels: List[str]) -> str:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        titles = [
            f"Uncorrected\nGGI = {ggi_values[0]:.2f}",
            f"After Phase 1 correction\nGGI = {ggi_values[1]:.2f}",
            f"Synthetic dataset\nGGI = {ggi_values[2]:.2f}",
        ]
        data_list = [coverage_original, coverage_corrected, coverage_synthetic]
        colors = ["#E8927C", "#7CB8E8", "#7CE89B"]

        for ax, data, title, color in zip(axes, data_list, titles, colors):
            x = np.arange(len(ward_labels))
            ax.bar(x, data, color=color, alpha=0.8)
            ax.axhline(y=np.mean(data), color="red", linestyle="--", linewidth=1, label="Mean")
            ax.set_title(title)
            ax.set_xlabel("Ward")
            ax.set_xticks(x[::3])
            ax.set_xticklabels([ward_labels[i] for i in range(0, len(ward_labels), 3)], rotation=45)
            ax.legend(fontsize=9)

        axes[0].set_ylabel("Coverage ratio")
        path = self.output_dir / "geographic_coverage_equality.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_per_ward_tpr(self, tpr_original: np.ndarray, tpr_corrected: np.ndarray,
                          tpr_synthetic: np.ndarray, seod_values: Tuple[float, float, float]) -> str:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        titles = [
            f"Uncorrected\nsEOD = {seod_values[0]:.2f}",
            f"Phase 1 corrected\nsEOD = {seod_values[1]:.2f}",
            f"Synthetic-trained\nsEOD = {seod_values[2]:.2f}",
        ]
        data_list = [tpr_original, tpr_corrected, tpr_synthetic]

        for ax, data, title in zip(axes, data_list, titles):
            x = np.arange(len(data))
            ax.bar(x, data, color="#5B9BD5", alpha=0.8)
            ax.axhline(y=np.mean(data), color="orange", linestyle="--", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Ward")
            ax.set_ylim(0, 1.0)

        axes[0].set_ylabel("True Positive Rate (High Risk)")
        path = self.output_dir / "per_ward_tpr.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_privacy_utility(self, epsilons: List[float], f1_scores: List[float],
                             oa_scores: List[float], best_single_source_f1: float) -> str:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        eps_labels = [str(e) if e != float("inf") else "∞" for e in epsilons]
        x = np.arange(len(epsilons))

        ax1.plot(x, f1_scores, "o-", color="#2196F3", linewidth=2, markersize=8, label="Macro F1")
        ax2.plot(x, oa_scores, "s-", color="#FF9800", linewidth=2, markersize=8, label="Overall Accuracy (%)")

        ax1.axhline(y=best_single_source_f1, color="red", linestyle="--", linewidth=1.5,
                     label=f"Best single-source (F1={best_single_source_f1:.3f})")

        ax1.axvspan(1.5, 2.5, alpha=0.15, color="green", label="Recommended zone")

        ax1.set_xlabel("Privacy budget ε")
        ax1.set_ylabel("Macro F1-score")
        ax2.set_ylabel("Overall Accuracy (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(eps_labels)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        path = self.output_dir / "privacy_utility_tradeoff.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_pca_loading_matrix(self, loading_matrix: np.ndarray,
                                feature_names: List[str],
                                mct_scores: np.ndarray) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2, 1]})

        n_components = loading_matrix.shape[0]
        im = ax1.imshow(loading_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax1.set_yticks(range(n_components))
        ax1.set_yticklabels([f"PC{i+1}" for i in range(n_components)])
        step = max(1, len(feature_names) // 15)
        ax1.set_xticks(range(0, len(feature_names), step))
        ax1.set_xticklabels([feature_names[i] for i in range(0, len(feature_names), step)],
                            rotation=45, ha="right", fontsize=8)
        ax1.set_title("PCA feature correlation matrix")
        fig.colorbar(im, ax=ax1, shrink=0.8)

        y = np.arange(n_components)
        colors = ["#2196F3" if s >= 0.70 else "#FF9800" for s in mct_scores]
        ax2.barh(y, mct_scores, color=colors)
        ax2.axvline(x=np.mean(mct_scores), color="gray", linestyle="--", label=f"Mean MCT = {np.mean(mct_scores):.2f}")
        ax2.set_yticks(y)
        ax2.set_yticklabels([f"PC{i+1}" for i in range(n_components)])
        ax2.set_xlabel("MCT value")
        ax2.set_title("Modality contribution traceability (MCT)")
        ax2.legend()
        ax2.set_xlim(0.3, 1.0)

        path = self.output_dir / "pca_loading_matrix.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_flood_risk_maps(self, real_map: np.ndarray, synthetic_map: np.ndarray,
                              single_source_map: np.ndarray,
                              oa_values: Tuple[float, float, float]) -> str:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        cmap = mcolors.ListedColormap(["#2E7D32", "#FFC107", "#FF5722", "#B71C1C"])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        titles = [
            f"Real Fused (OA={oa_values[0]:.1f}%)",
            f"Synthetic, ε=5 (OA={oa_values[1]:.1f}%)",
            f"Sentinel-2 only (OA={oa_values[2]:.1f}%)",
        ]

        for ax, data, title in zip(axes, [real_map, synthetic_map, single_source_map], titles):
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")

        path = self.output_dir / "flood_risk_maps.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_gwfa_maps(self, gwfa_real: Dict[str, np.ndarray],
                       gwfa_synthetic: Dict[str, np.ndarray],
                       feature_names: List[str]) -> str:
        n_features = len(feature_names)
        fig, axes = plt.subplots(2, n_features, figsize=(6 * n_features, 10))

        if n_features == 1:
            axes = axes.reshape(-1, 1)

        for col, feat in enumerate(feature_names):
            for row, (label, data_dict) in enumerate([("Real-trained", gwfa_real),
                                                       ("Synthetic-trained", gwfa_synthetic)]):
                im = axes[row, col].imshow(data_dict[feat], cmap="YlOrRd", interpolation="nearest")
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"{feat} importance")
                if col == 0:
                    axes[row, col].set_ylabel(label, fontsize=12)

        fig.colorbar(im, ax=axes, shrink=0.6, label="Feature importance")
        path = self.output_dir / "gwfa_attribution_maps.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_csd_map(self, csd_values: np.ndarray, suds_locations: Optional[np.ndarray] = None) -> str:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(csd_values, cmap="YlGnBu_r", interpolation="nearest")

        if suds_locations is not None:
            ax.scatter(suds_locations[:, 1], suds_locations[:, 0],
                       marker="v", c="red", s=80, label="SuDS recommendations", zorder=5)
            ax.legend()

        ax.set_title("CSD map (High → Moderate transition)\nDarker = more sensitive to reclassification")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Counterfactual Spatial Distance")
        path = self.output_dir / "csd_spatial_map.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_attribution_divergence(self, ad_map: np.ndarray, threshold: float = 0.05) -> str:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(ad_map, cmap="RdYlGn_r", vmin=0.0, vmax=0.15, interpolation="nearest")

        pct_low = np.mean(ad_map < threshold) * 100
        ax.set_title(f"Spatial AD distribution\n{pct_low:.0f}% of pixels with AD < {threshold} (near-identical reasoning)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Attribution Divergence (AD)")

        path = self.output_dir / "attribution_divergence_map.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_adversarial_robustness(self, epsilons_atk: np.ndarray,
                                     real_fgsm: np.ndarray, syn_fgsm: np.ndarray,
                                     real_pgd: np.ndarray, syn_pgd: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epsilons_atk, real_fgsm, "o--", color="#1976D2", label="Real-trained (FGSM)")
        ax.plot(epsilons_atk, syn_fgsm, "o-", color="#D32F2F", label="Synthetic-trained (FGSM)")
        ax.plot(epsilons_atk, real_pgd, "s--", color="#1976D2", alpha=0.6, label="Real-trained (PGD)")
        ax.plot(epsilons_atk, syn_pgd, "s-", color="#D32F2F", alpha=0.6, label="Synthetic-trained (PGD)")

        ax.axvspan(0, 0.04, alpha=0.1, color="green", label="Operational range")
        ax.set_xlabel("Perturbation magnitude ε_atk")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.legend()
        ax.set_title("Adversarial robustness comparison")

        path = self.output_dir / "adversarial_robustness.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_feedback_convergence(self, iterations: List[int], f1_vals: List[float],
                                   seod_vals: List[float], delta_vals: List[float]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(iterations, f1_vals, "o-", color="#2196F3", label="Macro F1")
        ax1b = ax1.twinx()
        ax1b.plot(iterations, seod_vals, "s-", color="#FF5722", label="sEOD")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Macro F1")
        ax1b.set_ylabel("sEOD")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)
        ax1.set_title("Metric trajectory")

        ax2.plot(iterations[1:], delta_vals[1:], "o-", color="#4CAF50")
        ax2.axhline(y=0.005, color="red", linestyle="--", label="Convergence threshold")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Δ (L∞ norm)")
        ax2.set_title("Feedback loop convergence")
        ax2.legend()

        path = self.output_dir / "feedback_convergence.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)

    def plot_ablation_radar(self, conditions: Dict[str, Dict[str, float]], metrics: List[str]) -> str:
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))

        for (cond_name, cond_values), color in zip(conditions.items(), colors):
            values = [cond_values.get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=cond_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.set_title("Ablation study")

        path = self.output_dir / "ablation_radar.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)
