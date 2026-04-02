import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save(fig, name, out="figures"):
    Path(out).mkdir(exist_ok=True)
    fig.savefig(f"{out}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ggi(cov_orig, cov_corr, cov_syn, ggi_vals, out="figures"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    data = [cov_orig, cov_corr, cov_syn]
    titles = [f"Uncorrected\nGGI = {ggi_vals[0]:.2f}",
              f"Phase 1 corrected\nGGI = {ggi_vals[1]:.2f}",
              f"Synthetic\nGGI = {ggi_vals[2]:.2f}"]
    colors = ["#E8927C", "#7CB8E8", "#7CE89B"]
    for ax, d, t, c in zip(axes, data, titles, colors):
        ax.bar(range(len(d)), d, color=c, alpha=0.8)
        ax.axhline(np.mean(d), color="red", ls="--", lw=1)
        ax.set_title(t); ax.set_xlabel("Ward")
    axes[0].set_ylabel("Coverage ratio")
    save(fig, "fig1_ggi", out)


def plot_tpr(tpr_orig, tpr_corr, tpr_syn, seod_vals, out="figures"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, d, s in zip(axes, [tpr_orig, tpr_corr, tpr_syn], seod_vals):
        ax.bar(range(len(d)), d, color="#5B9BD5", alpha=0.8)
        ax.axhline(np.mean(d), color="orange", ls="--", lw=1.5)
        ax.set_title(f"sEOD = {s:.2f}"); ax.set_xlabel("Ward"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("TPR (High Risk)")
    save(fig, "fig2_tpr", out)


def plot_privacy_utility(epsilons, f1s, oas, best_ss, out="figures"):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    x = range(len(epsilons))
    labels = [str(e) if e != float("inf") else "∞" for e in epsilons]
    ax1.plot(x, f1s, "o-", color="#2196F3", lw=2, label="Macro F1")
    ax2.plot(x, oas, "s-", color="#FF9800", lw=2, label="OA (%)")
    ax1.axhline(best_ss, color="red", ls="--", lw=1.5, label=f"Best single-source ({best_ss:.3f})")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Privacy budget ε"); ax1.set_ylabel("Macro F1"); ax2.set_ylabel("OA (%)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    save(fig, "fig3_privacy_utility", out)


def plot_pca_loading(loading, mct, out="figures"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax1.imshow(loading, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax1.set_yticks(range(loading.shape[0]))
    ax1.set_yticklabels([f"PC{i+1}" for i in range(loading.shape[0])])
    ax1.set_title("PCA loading matrix")
    colors = ["#2196F3" if s >= 0.70 else "#FF9800" for s in mct]
    ax2.barh(range(len(mct)), mct, color=colors)
    ax2.axvline(np.mean(mct), color="gray", ls="--")
    ax2.set_yticks(range(len(mct)))
    ax2.set_yticklabels([f"PC{i+1}" for i in range(len(mct))])
    ax2.set_xlabel("MCT"); ax2.set_title("Modality contribution traceability")
    save(fig, "fig4_pca_mct", out)


def plot_ad_map(ad_values, grid_size, out="figures"):
    n = int(np.sqrt(len(ad_values)))
    if n * n != len(ad_values):
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(ad_values.reshape(n, n), cmap="RdYlGn_r", vmin=0, vmax=0.15)
    pct = (ad_values < 0.05).mean() * 100
    ax.set_title(f"Attribution Divergence\n{pct:.0f}% with AD < 0.05")
    ax.axis("off"); fig.colorbar(im, ax=ax, label="AD")
    save(fig, "fig5_ad_map", out)


def plot_ablation(results, out="figures"):
    metrics_keys = ["f1", "seod", "zpr", "mia"]
    labels = list(results.keys())
    n = len(metrics_keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (name, vals) in enumerate(results.items()):
        v = [vals.get(m, 0) for m in metrics_keys] + [vals.get(metrics_keys[0], 0)]
        ax.plot(angles, v, "o-", lw=2, label=name)
        ax.fill(angles, v, alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_keys)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)
    save(fig, "fig6_ablation", out)


def plot_feedback(history, out="figures"):
    iters = [h["iter"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iters, [h["f1"] for h in history], "o-", label="F1")
    ax1b = ax1.twinx()
    ax1b.plot(iters, [h["seod"] for h in history], "s-", color="red", label="sEOD")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("F1"); ax1b.set_ylabel("sEOD")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
    deltas = [h.get("delta", None) for h in history]
    valid = [(i, d) for i, d in zip(iters, deltas) if d is not None]
    if valid:
        ax2.plot([v[0] for v in valid], [v[1] for v in valid], "o-", color="green")
        ax2.axhline(0.005, color="red", ls="--", label="threshold")
        ax2.set_xlabel("Iteration"); ax2.set_ylabel("Δ"); ax2.legend()
    save(fig, "fig7_feedback", out)
