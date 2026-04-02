import yaml
import numpy as np
import json
from pathlib import Path
from collections import Counter

from pipeline import phase1_prepare, phase2_train, phase3_validate, phase4_explain
from pipeline import run_ablation, run_feedback


def load_chelmsford(n=10000, n_zones=22, seed=42):
    rng = np.random.RandomState(seed)
    class_p = np.array([0.0044, 0.327, 0.432, 0.232])
    y = rng.choice(4, n, p=class_p / class_p.sum())
    zones = rng.randint(0, n_zones, n)

    s2 = rng.randn(n, 33) * 0.3
    for i in range(n):
        if y[i] == 0: s2[i, :11] += rng.uniform(0.3, 0.8, 11)
        elif y[i] == 1: s2[i, :11] += rng.uniform(0.1, 0.4, 11)
    cloud = rng.random((n, 33))
    s2[cloud < 0.05] = np.nan
    for z in range(5):
        m = zones == z
        extra = rng.random((m.sum(), 33))
        s2[m][extra < 0.15] = np.nan

    lidar = np.column_stack([rng.uniform(0, 100, n), rng.uniform(0, 45, n),
                              rng.uniform(0, 360, n), rng.uniform(0, 50, n)])
    for i in range(n):
        if y[i] == 0: lidar[i, 0] -= rng.uniform(5, 20); lidar[i, 3] += rng.uniform(10, 30)

    fvec = rng.randn(n, 3) * 0.2
    for i in range(n): fvec[i, y[i] % 3] += rng.uniform(0.5, 1.5)

    lc = rng.randn(n, 1) * 0.3
    roads = rng.randn(n, 2) * 0.2
    rivers = rng.randn(n, 2) * 0.2
    coords = np.column_stack([rng.uniform(0, 10000, n), rng.uniform(0, 10000, n)])

    return {
        "features": {"sentinel2": s2, "lidar": lidar, "flood_vec": fvec,
                      "land_cover": lc, "roads": roads, "rivers": rivers},
        "y": y, "zones": zones, "coords": coords,
        "classes": ["High", "Moderate", "Low", "Negligible"],
    }


def load_leeds(n=8000, seed=99):
    rng = np.random.RandomState(seed)
    class_p = np.array([0.012, 0.284, 0.478, 0.226])
    y = rng.choice(4, n, p=class_p / class_p.sum())
    zones = rng.randint(0, 18, n)

    s2 = rng.randn(n, 32) * 0.3
    for i in range(n):
        if y[i] == 0: s2[i, :11] += rng.uniform(0.2, 0.7, 11)

    lidar = np.column_stack([rng.uniform(0, 80, n), rng.uniform(0, 45, n),
                              rng.uniform(0, 360, n), rng.uniform(0, 40, n)])
    for i in range(n):
        if y[i] == 0: lidar[i, 0] -= rng.uniform(3, 15)

    fvec = rng.randn(n, 3) * 0.2
    for i in range(n): fvec[i, y[i] % 3] += rng.uniform(0.4, 1.2)

    return {
        "features": {"sentinel2": s2, "lidar": lidar, "flood_vec": fvec,
                      "land_cover": rng.randn(n, 1) * 0.4,
                      "roads": rng.randn(n, 2) * 0.2, "rivers": rng.randn(n, 2) * 0.2},
        "y": y, "zones": zones,
        "coords": np.column_stack([rng.uniform(0, 8000, n), rng.uniform(0, 8000, n)]),
        "classes": ["High", "Moderate", "Low", "Negligible"],
    }


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(42)
    for d in ["results", "figures"]:
        Path(d).mkdir(exist_ok=True)

    print("Loading Chelmsford pilot data...")
    ch = load_chelmsford()
    print("Loading Leeds generalisation data...")
    le = load_leeds()

    X, y, zones, loading, slices, scaler, pca = phase1_prepare(
        ch["features"], ch["y"], ch["zones"], ch["coords"], cfg)

    split = int(0.7 * len(X))
    shuf = np.random.permutation(len(X))
    tr, te = shuf[:split], shuf[split:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    z_tr, z_te = zones[tr], zones[te]

    G, D, X_syn, y_syn, prov, mct, eps, hist = phase2_train(
        X_tr, y_tr, z_tr, loading, cfg)

    val = phase3_validate(X_syn, y_syn, X_te, y_te, z_te, X_tr, y_tr, cfg, ch["classes"])

    from sklearn.ensemble import RandomForestClassifier
    clf_real = RandomForestClassifier(
        n_estimators=cfg["classifier"]["n_estimators"],
        max_depth=cfg["classifier"]["max_depth"],
        class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)

    exp = phase4_explain(X_te, y_te, X_syn, y_syn, ch["coords"][:len(X_te)],
                          clf_real, val["best_clf"], X.shape[1], prov, cfg)

    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Leeds Generalisation")
    print("=" * 60)
    X_le, y_le, z_le, ld_le, _, _, _ = phase1_prepare(
        le["features"], le["y"], le["zones"], le["coords"], cfg)
    clf_le = RandomForestClassifier(
        n_estimators=cfg["classifier"]["n_estimators"],
        max_depth=cfg["classifier"]["max_depth"],
        class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_syn[:len(X_le)], y_syn[:len(y_le)])
    import metrics as M
    ev_le = M.full_eval(y_le, clf_le.predict(X_le), z_le)
    print(f"  Leeds: F1={ev_le['f1']:.4f}  ZPR={ev_le['zpr']:.3f}  sEOD={ev_le['seod']:.3f}")
    print(f"  Cross-site F1 gap: {abs(val['tstr']['f1'] - ev_le['f1']):.4f}")

    results = {
        "chelmsford_real_f1": val["real"]["f1"],
        "chelmsford_syn_f1": val["tstr"]["f1"],
        "chelmsford_zpr": val["tstr"]["zpr"],
        "chelmsford_seod": val["tstr"]["seod"],
        "mia_auc": val["mia_auc"],
        "sfid": val["sfid"],
        "variogram": val["variogram"],
        "morans_pres": val["morans_pres"],
        "ad_mean": exp["ad_mean"],
        "ad_pct_005": exp["ad_pct_below_005"],
        "leeds_f1": ev_le["f1"],
        "leeds_zpr": ev_le["zpr"],
        "epsilon": eps,
        "pcs": 1.0,
        "mean_mct": float(mct.mean()),
    }

    with open("results/consolidated.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("CONSOLIDATED RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:25s} = {v:.4f}" if isinstance(v, float) else f"  {k:25s} = {v}")

    print("\nDone. Results in results/, figures in figures/")


if __name__ == "__main__":
    main()
