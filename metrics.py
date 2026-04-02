import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy, ks_2samp
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score


def geographic_gini_index(zone_coverages):
    n = len(zone_coverages)
    c_bar = np.mean(zone_coverages)
    if c_bar == 0:
        return 0.0
    diffs = np.abs(zone_coverages[:, None] - zone_coverages[None, :])
    return float(np.sum(diffs) / (2 * n * n * c_bar))


def spatial_eod(tpr_per_zone):
    return float(np.max(tpr_per_zone) - np.min(tpr_per_zone))


def zonal_parity_ratio(zone_f1, tau=0.10):
    f_bar = np.mean(zone_f1)
    return float(np.mean(np.abs(zone_f1 - f_bar) <= tau))


def provenance_completeness(flags):
    return float(np.mean(flags))


def modality_contribution_traceability(loading_matrix, modality_slices):
    n_pc = loading_matrix.shape[0]
    mct = np.zeros(n_pc)
    for pc in range(n_pc):
        row = np.abs(loading_matrix[pc])
        contribs = np.array([np.sum(row[s]) for s in modality_slices])
        total = np.sum(contribs)
        mct[pc] = float(np.max(contribs) / total) if total > 0 else 0.0
    return mct


def gwfa(X, y, coords, target_coords, adaptive=True, bandwidth=500.0):
    n_targets = target_coords.shape[0]
    n_feat = X.shape[1]
    attr = np.zeros((n_targets, n_feat))
    y_float = y.astype(float)

    for i in range(n_targets):
        d = np.sqrt(np.sum((coords - target_coords[i]) ** 2, axis=1))
        if adaptive:
            k = max(int(0.1 * len(d)), 20)
            h = max(np.sort(d)[min(k, len(d) - 1)], 1e-6)
        else:
            h = bandwidth
        w = np.exp(-d ** 2 / (2 * h ** 2))
        w /= w.sum() + 1e-12

        imp = np.zeros(n_feat)
        for f in range(n_feat):
            r = np.corrcoef(X[:, f] * w, y_float)[0, 1]
            if np.isnan(r):
                r = 0.0
            v = np.average((X[:, f] - np.average(X[:, f], weights=w)) ** 2, weights=w)
            imp[f] = abs(r) * np.sqrt(v + 1e-12)
        total = imp.sum()
        attr[i] = imp / total if total > 0 else imp

    return attr


def counterfactual_spatial_distance(X, preds, clf, offset=-1):
    n = X.shape[0]
    csd = np.full(n, np.nan)
    classes = np.unique(preds)
    steps = [0.01, 0.05, 0.1, 0.25]

    for i in range(n):
        target = preds[i] + offset
        if target not in classes:
            target = preds[i] - offset
        if target not in classes:
            continue

        best = float("inf")
        x0 = X[i]
        for step in steps:
            for f in range(X.shape[1]):
                for sign in [-1.0, 1.0]:
                    xm = x0.copy()
                    for m in range(1, 50):
                        xm[f] = x0[f] + sign * step * m
                        if clf.predict(xm.reshape(1, -1))[0] == target:
                            best = min(best, np.linalg.norm(xm - x0))
                            break
        if best < float("inf"):
            csd[i] = best
    return csd


def attribution_divergence(attr_real, attr_syn):
    n = attr_real.shape[0]
    ad = np.zeros(n)
    for i in range(n):
        p = attr_real[i] + 1e-12
        q = attr_syn[i] + 1e-12
        ad[i] = entropy(p / p.sum(), q / q.sum())
    return ad


def sfid(X_real, X_syn):
    mu_r, mu_s = X_real.mean(0), X_syn.mean(0)
    C_r, C_s = np.cov(X_real, rowvar=False), np.cov(X_syn, rowvar=False)
    diff_sq = np.dot(mu_r - mu_s, mu_r - mu_s)
    eigvals = np.maximum(np.linalg.eigvalsh(C_r @ C_s), 0)
    return float(max(diff_sq + np.trace(C_r) + np.trace(C_s) - 2 * np.sum(np.sqrt(eigvals)), 0))


def variogram_fidelity(coords_r, vals_r, coords_s, vals_s, n_bins=20):
    def _vario(c, v):
        n = min(len(c), 2000)
        idx = np.random.choice(len(c), n, replace=False)
        c, v = c[idx], v[idx]
        d = cdist(c, c)
        ui = np.triu_indices(n, k=1)
        fd, fv = d[ui], (v[ui[0]] - v[ui[1]]) ** 2
        bins = np.linspace(0, np.percentile(fd, 90), n_bins + 1)
        gamma = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (fd >= bins[b]) & (fd < bins[b + 1])
            if mask.any():
                gamma[b] = 0.5 * fv[mask].mean()
        return gamma

    gr, gs = _vario(coords_r, vals_r), _vario(coords_s, vals_s)
    mask = (gr > 0) | (gs > 0)
    if not mask.any():
        return 1.0
    r = np.corrcoef(gr[mask], gs[mask])[0, 1]
    return float(max(r, 0)) if not np.isnan(r) else 0.0


def morans_i(values, coords, k=8):
    n = len(values)
    if n < k + 1:
        return 0.0
    d = cdist(coords, coords)
    W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(d[i])[1:k + 1]
        W[i, nn] = 1.0
    W = (W + W.T) / 2
    z = values - values.mean()
    num = n * np.sum(W * np.outer(z, z))
    den = W.sum() * np.sum(z ** 2)
    return float(num / den) if den != 0 else 0.0


def per_zone_tpr(y_true, y_pred, zones, target_class):
    uz = np.unique(zones)
    tpr = np.zeros(len(uz))
    for i, z in enumerate(uz):
        m = zones == z
        pos = y_true[m] == target_class
        tpr[i] = (y_pred[m][pos] == target_class).mean() if pos.any() else 0.0
    return tpr


def per_zone_f1(y_true, y_pred, zones):
    uz = np.unique(zones)
    return np.array([
        f1_score(y_true[zones == z], y_pred[zones == z], average="macro", zero_division=0)
        for z in uz
    ])


def mia_auc(member_scores, nonmember_scores):
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    return float(roc_auc_score(labels, scores))


def full_eval(y_true, y_pred, zones, target_class=0):
    zf1 = per_zone_f1(y_true, y_pred, zones)
    ztpr = per_zone_tpr(y_true, y_pred, zones, target_class)
    return {
        "oa": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "zpr": zonal_parity_ratio(zf1),
        "seod": spatial_eod(ztpr),
        "zone_f1": zf1,
        "zone_tpr": ztpr,
    }
