"""Statistical verification, bootstrap CIs, BH-FDR, calibration, and L2 linear/ridge probes.

Implements exact scientific controls mandated by CIR_one_run_probe_spec.md:
- Paired group bootstrap CIs (B=2000 default, grouped by ref_id/scene/paraphrase_group)
- Benjamini-Hochberg FDR multiple-testing correction
- Effect sizes (paired Cohen's d_z, Cliff's delta)
- Calibration metrics (ECE, Brier, AUROC, AUPRC, AURC risk-coverage, temperature scaling fit on dev -> frozen on test)
- Deterministic L2-regularized LogisticRegression & Ridge probes in pure PyTorch/NumPy/SciPy:
    fit on train -> sweep C on dev -> freeze best C -> evaluate once on test,
    ALWAYS accompanied by majority-class prior, shuffled-label control, and matched-dim PCA control!
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import optimize, stats  # type: ignore
except ImportError:  # Pure-NumPy fallback for pinned model virtualenvs without scipy
    optimize = None  # type: ignore
    stats = None  # type: ignore


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Exact tie-averaged ranks (1-based) in pure NumPy (matches scipy.stats.rankdata)."""
    if stats is not None:
        return stats.rankdata(a)
    arr = np.asarray(a, dtype=np.float64)
    n = len(arr)
    order = np.argsort(arr, kind="mergesort")
    s = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[j] == s[i]:
            j += 1
        avg_r = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_r
        i = j
    return ranks


def paired_group_bootstrap_ci(
    values_a: np.ndarray,
    values_b: Optional[np.ndarray],
    groups: Sequence[Any],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stat_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, float]:
    """Compute point estimate, paired group bootstrap CI, p-value, and effect size.

    If `values_b` is provided, computes `diff = values_a - values_b` (paired per observation,
    resampled by cluster `groups` so correlated queries sharing a ref/scene stay together).
    """
    a = np.asarray(values_a, dtype=np.float64)
    if values_b is not None:
        b = np.asarray(values_b, dtype=np.float64)
        x = a - b
    else:
        x = a

    mask = np.isfinite(x)
    x = x[mask]
    g_arr = np.asarray(groups, dtype=object)[mask]
    n = int(len(x))
    if n == 0:
        return {"n": 0, "n_groups": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan"), "cohen_dz": float("nan")}

    fn = stat_fn or (lambda arr: float(np.mean(arr)))
    point = fn(x)

    # Group indices by cluster
    uniq_g: Dict[Any, List[int]] = {}
    for idx, g in enumerate(g_arr):
        uniq_g.setdefault(g, []).append(idx)
    group_slices = [np.asarray(idxs, dtype=np.int64) for idxs in uniq_g.values()]
    n_groups = len(group_slices)

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot, dtype=np.float64)
    if n_groups == n:
        # Fast vectorized path when every observation is its own group
        idx_mat = rng.integers(0, n, size=(n_boot, n))
        if stat_fn is None:
            boot_vals = x[idx_mat].mean(axis=1)
        else:
            for bi in range(n_boot):
                boot_vals[bi] = fn(x[idx_mat[bi]])
    else:
        for bi in range(n_boot):
            chosen = rng.integers(0, n_groups, size=n_groups)
            if stat_fn is None:
                s = 0.0
                cnt = 0
                for gi in chosen:
                    sl = group_slices[gi]
                    s += float(x[sl].sum())
                    cnt += len(sl)
                boot_vals[bi] = s / max(cnt, 1)
            else:
                samp = np.concatenate([x[group_slices[gi]] for gi in chosen])
                boot_vals[bi] = fn(samp)

    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))
    # Two-sided bootstrap p-value against null = 0
    p_le = float(np.mean(boot_vals <= 0.0))
    p_ge = float(np.mean(boot_vals >= 0.0))
    p_boot = min(1.0, 2.0 * min(p_le, p_ge))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    dz = float(point / std) if std > 1e-12 else 0.0
    return {
        "n": n,
        "n_groups": n_groups,
        "mean": float(point),
        "std": std,
        "ci_low": lo,
        "ci_high": hi,
        "p_boot": p_boot,
        "cohen_dz": dz,
    }


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR-adjusted q-values preserving original order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    valid = np.isfinite(p)
    out = np.full(m, np.nan, dtype=np.float64)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return out.tolist()
    pv = p[idx]
    k = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * k / np.arange(1, k + 1, dtype=np.float64)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    adj = np.empty(k, dtype=np.float64)
    adj[order] = q
    out[idx] = adj
    return out.tolist()


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric Cliff's delta effect size in [-1, 1]."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # O((n+m) log(n+m)) exact calculation via Mann-Whitney U
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float((2.0 * u) / (len(x) * len(y)) - 1.0)


# ---------------------------------------------------------------------------
# Uncertainty & Calibration Metrics (E15)
# ---------------------------------------------------------------------------

def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(s)
    u = float(ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0)
    return float(u / (n_pos * n_neg))


def binary_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    prec = tp / np.arange(1, len(y_ord) + 1, dtype=np.float64)
    return float((prec * y_ord).sum() / n_pos)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    mask = np.isfinite(p) & np.isfinite(y)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y))
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        cnt = int(in_bin.sum())
        if cnt > 0:
            acc = float(y[in_bin].mean())
            conf = float(p[in_bin].mean())
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    mask = np.isfinite(p) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def risk_coverage_aurc(y_correct: np.ndarray, confidence: np.ndarray) -> Dict[str, Any]:
    """Compute Area Under Risk-Coverage curve (AURC) and risk at fixed coverages."""
    y = np.asarray(y_correct, dtype=np.float64)
    c = np.asarray(confidence, dtype=np.float64)
    mask = np.isfinite(c) & np.isfinite(y)
    y, c = y[mask], c[mask]
    n = len(y)
    if n == 0:
        return {"aurc": float("nan"), "e_aurc": float("nan"), "risk_at_50": float("nan"), "risk_at_80": float("nan")}
    order = np.argsort(-c, kind="mergesort")
    errs = 1.0 - y[order]
    cum_err = np.cumsum(errs)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risks = cum_err / ks
    aurc = float(np.mean(risks))
    # Optimal oracle AURC
    err_rate = float(errs.mean())
    opt_aurc = err_rate + (1.0 - err_rate) * math.log(max(1.0 - err_rate, 1e-12)) if err_rate < 1.0 else 1.0
    e_aurc = float(aurc - opt_aurc)
    r50 = float(risks[max(0, int(round(0.5 * n)) - 1)])
    r80 = float(risks[max(0, int(round(0.8 * n)) - 1)])
    r20 = float(risks[max(0, int(round(0.2 * n)) - 1)])
    return {
        "aurc": aurc,
        "e_aurc": e_aurc,
        "full_risk": float(errs.mean()),
        "risk_at_20": r20,
        "risk_at_50": r50,
        "risk_at_80": r80,
    }


def fit_platt_calibration(dev_scores: np.ndarray, dev_y: np.ndarray) -> Tuple[float, float]:
    """Fit logistic calibration P(y=1|s) = sigmoid(a*s + b) strictly on dev split."""
    s = np.asarray(dev_scores, dtype=np.float64)
    y = np.asarray(dev_y, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    if len(s) < 5 or y.sum() == 0 or y.sum() == len(y):
        prior = float(np.clip(y.mean() if len(y) else 0.5, 1e-4, 1 - 1e-4))
        return 0.0, math.log(prior / (1.0 - prior))
    s_std = float(np.std(s))
    s_mean = float(np.mean(s))
    z = (s - s_mean) / max(s_std, 1e-8)

    def nll(params: np.ndarray) -> Tuple[float, np.ndarray]:
        a, b = params[0], params[1]
        logits = a * z + b
        # Stable log(1 + exp(logits)) - y*logits + tiny L2 reg
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits) + 1e-4 * (a * a + b * b)
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        diff = (p - y) / len(y)
        ga = float(np.dot(diff, z)) + 2e-4 * a
        gb = float(np.sum(diff)) + 2e-4 * b
        return float(loss), np.array([ga, gb], dtype=np.float64)

    if optimize is not None:
        res = optimize.minimize(nll, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        a_z, b_z = float(res.x[0]), float(res.x[1])
    else:
        # Pure-NumPy damped Newton-IRLS for 2-parameter convex logistic regression
        a_z, b_z = 1.0, 0.0
        n_f = float(len(y))
        for _ in range(40):
            logits = np.clip(a_z * z + b_z, -40.0, 40.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            w = np.clip(p * (1.0 - p), 1e-8, None)
            diff = (p - y) / n_f
            g = np.array([float(np.dot(diff, z)) + 2e-4 * a_z, float(np.sum(diff)) + 2e-4 * b_z])
            h00 = float(np.dot(w, z * z)) / n_f + 2e-4
            h01 = float(np.dot(w, z)) / n_f
            h11 = float(np.sum(w)) / n_f + 2e-4
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-14:
                break
            da = (h11 * g[0] - h01 * g[1]) / det
            db = (-h01 * g[0] + h00 * g[1]) / det
            a_z -= da
            b_z -= db
            if abs(da) + abs(db) < 1e-8:
                break
    a = a_z / max(s_std, 1e-8)
    b = b_z - a * s_mean
    return a, b


def apply_platt_calibration(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = np.clip(a * np.asarray(scores, dtype=np.float64) + b, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def false_accept_rate_at_tpr(y_present: np.ndarray, scores: np.ndarray, target_tpr: float = 0.90) -> float:
    """False-accept rate on no-target (y_present==0) at threshold achieving >= target_tpr on present (y_present==1)."""
    y = np.asarray(y_present, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = float(np.quantile(pos, 1.0 - target_tpr))
    return float(np.mean(neg >= thr))


# ---------------------------------------------------------------------------
# Deterministic L2 Linear & Ridge Probes (E04)
# Fit on train -> tune C on dev -> freeze on test (+ prior, shuffled, matched-dim PCA)
# ---------------------------------------------------------------------------

def _standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def _fit_pca_proj(X_train_std: np.ndarray, target_dim: int) -> np.ndarray:
    """Fit truncated SVD/PCA projection strictly on train features to match dimensionality."""
    n, d = X_train_std.shape
    k = min(int(target_dim), n, d)
    if k >= d:
        return np.eye(d, dtype=np.float64)
    _, _, vt = np.linalg.svd(X_train_std, full_matrices=False)
    return vt[:k].T  # [d, k]


def _fit_logreg_l2(X: np.ndarray, y: np.ndarray, n_classes: int, C: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact convex L2-regularized multinomial/binary logistic regression via SciPy L-BFGS-B."""
    n, d = X.shape
    lam = 1.0 / (max(float(C), 1e-8) * max(n, 1))
    if n_classes == 2:
        y_bin = y.astype(np.float64)

        def loss_and_grad(w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
            w = w_flat[:d]
            b = w_flat[d]
            logits = np.clip(X @ w + b, -40.0, 40.0)
            loss = float(np.mean(np.logaddexp(0.0, logits) - y_bin * logits) + 0.5 * lam * np.dot(w, w))
            p = 1.0 / (1.0 + np.exp(-logits))
            diff = (p - y_bin) / n
            gw = X.T @ diff + lam * w
            gb = float(diff.sum())
            return loss, np.concatenate([gw, [gb]])

        res = optimize.minimize(loss_and_grad, np.zeros(d + 1, dtype=np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 200})
        return res.x[:d].reshape(d, 1), res.x[d : d + 1]
    else:
        Y_oh = np.zeros((n, n_classes), dtype=np.float64)
        Y_oh[np.arange(n), y] = 1.0

        def loss_and_grad(w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
            W = w_flat[: d * n_classes].reshape(d, n_classes)
            b = w_flat[d * n_classes :]
            logits = X @ W + b[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            nll = -np.mean(np.sum(Y_oh * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
            loss = float(nll + 0.5 * lam * np.sum(W * W))
            diff = (probs - Y_oh) / n
            gW = X.T @ diff + lam * W
            gb = diff.sum(axis=0)
            return loss, np.concatenate([gW.ravel(), gb])

        res = optimize.minimize(
            loss_and_grad,
            np.zeros((d + 1) * n_classes, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        W = res.x[: d * n_classes].reshape(d, n_classes)
        b = res.x[d * n_classes :]
        return W, b


def _predict_logreg(X: np.ndarray, W: np.ndarray, b: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_classes == 2:
        logits = np.clip((X @ W)[:, 0] + b[0], -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        preds = (p1 >= 0.5).astype(np.int64)
        return preds, p1
    logits = X @ W + b[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return preds, probs


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def run_classification_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: Optional[Sequence[Any]] = None,
    c_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    match_dim: Optional[int] = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit L2 probe on train, select C on dev, freeze on test + prior + shuffled + matched-dim control."""
    y_tr = np.asarray(y_train, dtype=np.int64)
    y_dv = np.asarray(y_dev, dtype=np.int64)
    y_te = np.asarray(y_test, dtype=np.int64)
    uniq = sorted(set(y_tr.tolist()))
    if len(uniq) < 2 or len(y_dv) == 0 or len(y_te) == 0:
        return {"status": "insufficient_classes", "n_train": int(len(y_tr)), "n_dev": int(len(y_dv)), "n_test": int(len(y_te))}

    # Remap labels to 0..K-1
    lmap = {v: i for i, v in enumerate(uniq)}
    tr_mask = np.isin(y_tr, uniq)
    dv_mask = np.isin(y_dv, uniq)
    te_mask = np.isin(y_te, uniq)
    X_tr = np.asarray(X_train[tr_mask], dtype=np.float64)
    X_dv = np.asarray(X_dev[dv_mask], dtype=np.float64)
    X_te = np.asarray(X_test[te_mask], dtype=np.float64)
    y_tr = np.array([lmap[v] for v in y_tr[tr_mask]], dtype=np.int64)
    y_dv = np.array([lmap[v] for v in y_dv[dv_mask]], dtype=np.int64)
    y_te = np.array([lmap[v] for v in y_te[te_mask]], dtype=np.int64)
    n_classes = len(uniq)
    if len(y_dv) == 0 or len(y_te) == 0:
        return {"status": "insufficient_test_or_dev", "n_train": int(len(y_tr)), "n_dev": int(len(y_dv)), "n_test": int(len(y_te))}

    mu, sd = _standardize_fit(X_tr)
    Z_tr = (X_tr - mu) / sd
    Z_dv = (X_dv - mu) / sd
    Z_te = (X_te - mu) / sd

    def _eval_space(A_tr: np.ndarray, A_dv: np.ndarray, A_te: np.ndarray, y_tr_in: np.ndarray) -> Dict[str, Any]:
        best_c = float(c_grid[0])
        best_dev_score = -1.0
        best_W, best_b = None, None
        for c in c_grid:
            W, b = _fit_logreg_l2(A_tr, y_tr_in, n_classes, float(c))
            p_dv, _ = _predict_logreg(A_dv, W, b, n_classes)
            score_dv = _macro_f1(y_dv, p_dv, n_classes)
            if score_dv > best_dev_score:
                best_dev_score = score_dv
                best_c = float(c)
                best_W, best_b = W, b
        p_te, prob_te = _predict_logreg(A_te, best_W, best_b, n_classes)
        correct_vec = (p_te == y_te).astype(np.float64)
        out = {
            "best_C": best_c,
            "dev_macro_f1": float(best_dev_score),
            "test_acc": float(correct_vec.mean()),
            "test_macro_f1": _macro_f1(y_te, p_te, n_classes),
        }
        if n_classes == 2:
            out["test_auroc"] = binary_auroc(y_te, prob_te)
        if test_groups is not None:
            grp = np.asarray(test_groups, dtype=object)[te_mask]
            out["test_acc_ci"] = paired_group_bootstrap_ci(correct_vec, None, grp, n_boot=1000, seed=seed)
        return out

    # 1. Native dimension probe
    native_res = _eval_space(Z_tr, Z_dv, Z_te, y_tr)

    # 2. Majority-class prior
    maj_cls = int(np.bincount(y_tr, minlength=n_classes).argmax())
    prior_acc = float(np.mean(y_te == maj_cls))
    prior_f1 = _macro_f1(y_te, np.full_like(y_te, maj_cls), n_classes)

    # 3. Shuffled-label control (permuted train labels, tuned on dev using same C grid)
    rng = np.random.default_rng(seed)
    y_tr_shuf = rng.permutation(y_tr)
    shuf_res = _eval_space(Z_tr, Z_dv, Z_te, y_tr_shuf)

    res: Dict[str, Any] = {
        "status": "ok",
        "n_train": int(len(y_tr)),
        "n_dev": int(len(y_dv)),
        "n_test": int(len(y_te)),
        "n_classes": n_classes,
        "native_dim": int(X_tr.shape[1]),
        "native": native_res,
        "prior": {"majority_class": maj_cls, "test_acc": prior_acc, "test_macro_f1": prior_f1},
        "shuffled_control": shuf_res,
        "delta_over_prior_acc": float(native_res["test_acc"] - prior_acc),
        "delta_over_shuffled_acc": float(native_res["test_acc"] - shuf_res["test_acc"]),
    }

    # 4. Matched-dimensionality PCA control (when native_dim > match_dim)
    if match_dim is not None and X_tr.shape[1] > match_dim:
        P = _fit_pca_proj(Z_tr, match_dim)
        res["matched_dim"] = int(P.shape[1])
        res["matched_dim_pca"] = _eval_space(Z_tr @ P, Z_dv @ P, Z_te @ P, y_tr)
    else:
        res["matched_dim"] = int(X_tr.shape[1])
        res["matched_dim_pca"] = native_res

    return res


def run_ridge_cardinality_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_grid: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0),
    match_dim: Optional[int] = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """Closed-form Ridge regression probe for F09 Cardinality (1..4): reports MAE, exact acc, and adjacent confusion."""
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_dv = np.asarray(X_dev, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    y_dv = np.asarray(y_dev, dtype=np.float64)
    y_te = np.asarray(y_test, dtype=np.float64)
    if len(y_tr) < 8 or len(y_dv) < 2 or len(y_te) < 2:
        return {"status": "insufficient_samples", "n_train": int(len(y_tr)), "n_dev": int(len(y_dv)), "n_test": int(len(y_te))}

    mu, sd = _standardize_fit(X_tr)
    Z_tr = (X_tr - mu) / sd
    Z_dv = (X_dv - mu) / sd
    Z_te = (X_te - mu) / sd

    def _fit_eval(A_tr: np.ndarray, A_dv: np.ndarray, A_te: np.ndarray, y_t: np.ndarray) -> Dict[str, Any]:
        y_mean = float(y_t.mean())
        yc = y_t - y_mean
        best_a = float(alpha_grid[0])
        best_mae = float("inf")
        best_w = None
        d = A_tr.shape[1]
        AtA = A_tr.T @ A_tr
        Aty = A_tr.T @ yc
        eye = np.eye(d, dtype=np.float64)
        for a in alpha_grid:
            w = np.linalg.solve(AtA + float(a) * eye, Aty)
            pred_dv = A_dv @ w + y_mean
            mae_dv = float(np.mean(np.abs(pred_dv - y_dv)))
            if mae_dv < best_mae:
                best_mae = mae_dv
                best_a = float(a)
                best_w = w
        pred_te = A_te @ best_w + y_mean
        pred_round = np.clip(np.rint(pred_te), 1, 4).astype(np.int64)
        y_te_int = np.rint(y_te).astype(np.int64)
        # Adjacent confusion matrix (counts 1..4)
        adj_err = float(np.mean(np.abs(pred_round - y_te_int) == 1))
        exact_acc = float(np.mean(pred_round == y_te_int))
        conf = {}
        for t_val in sorted(set(y_te_int.tolist())):
            sub = pred_round[y_te_int == t_val]
            conf[str(t_val)] = {str(p_val): int((sub == p_val).sum()) for p_val in range(1, 5)}
        return {
            "best_alpha": best_a,
            "dev_mae": best_mae,
            "test_mae": float(np.mean(np.abs(pred_te - y_te))),
            "test_exact_acc": exact_acc,
            "test_adjacent_error_rate": adj_err,
            "confusion_1_to_4": conf,
        }

    native = _fit_eval(Z_tr, Z_dv, Z_te, y_tr)
    prior_val = float(np.median(y_tr))
    prior_mae = float(np.mean(np.abs(y_te - prior_val)))
    rng = np.random.default_rng(seed)
    shuf = _fit_eval(Z_tr, Z_dv, Z_te, rng.permutation(y_tr))
    out: Dict[str, Any] = {
        "status": "ok",
        "n_train": int(len(y_tr)),
        "n_dev": int(len(y_dv)),
        "n_test": int(len(y_te)),
        "native_dim": int(X_tr.shape[1]),
        "native": native,
        "prior_median": {"value": prior_val, "test_mae": prior_mae},
        "shuffled_control": shuf,
    }
    if match_dim is not None and X_tr.shape[1] > match_dim:
        P = _fit_pca_proj(Z_tr, match_dim)
        out["matched_dim_pca"] = _fit_eval(Z_tr @ P, Z_dv @ P, Z_te @ P, y_tr)
    return out
