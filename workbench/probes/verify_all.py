#!/usr/bin/env python3
"""Multi-round deep verification & invariant audit suite for the CIR 18-suite probe pipeline.

Verifies:
  1. Mathematical unit tests (rankdata, Platt IRLS vs L-BFGS, binom tail, BH-FDR,
     spherical tangent geometry, linear/ridge probes, Krippendorff alpha / Fleiss kappa).
  2. Strict zero-leakage cohort split invariants (train ∩ dev ∩ test group_id == ∅).
  3. E18 blind audit pack blindness & contact-sheet readability audit.
  4. Full artifact integrity, exact parity, denominator consistency, zero-NaN, and
     mathematical inequality invariants across EVERY completed model & dataset on disk!
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import PIL.Image
from scipy import optimize, stats

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.probes import DEFAULT_PROBE_OUT  # noqa: E402
from workbench.probes.cohort import MODEL_BY_KEY, MODEL_SPECS  # noqa: E402
from workbench.probes.stats_utils import (  # noqa: E402
    _rankdata_avg,
    bh_fdr,
    binary_auroc,
    fit_platt_calibration,
    paired_group_bootstrap_ci,
    run_classification_probe,
    run_ridge_cardinality_probe,
)
from workbench.probes.worker_eval import _tangent_displacement_metrics  # noqa: E402


def _check(cond: bool, msg: str, errors: List[str]) -> None:
    if not cond:
        errors.append(msg)


def verify_math_unit_tests(errors: List[str]) -> None:
    print("[VERIFY-1] Mathematical unit tests...", flush=True)
    # 1. Pure-NumPy rankdata vs scipy.stats.rankdata (with ties!)
    import workbench.probes.stats_utils as su
    saved_stats = su.stats
    saved_opt = su.optimize
    try:
        arr = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 1.0, 1.0, 4.0])
        ref_r = saved_stats.rankdata(arr)
        su.stats = None
        np_r = _rankdata_avg(arr)
        _check(np.allclose(ref_r, np_r, atol=0.0), f"rankdata mismatch: {ref_r} vs {np_r}", errors)

        # 2. Pure-NumPy Newton-IRLS Platt calibration vs SciPy L-BFGS-B
        rng = np.random.default_rng(123)
        s = rng.normal(0.5, 0.2, size=400)
        y = (s + rng.normal(0, 0.15, size=400) > 0.5).astype(np.float64)
        su.optimize = saved_opt
        a_sp, b_sp = fit_platt_calibration(s, y)
        su.optimize = None
        a_np, b_np = fit_platt_calibration(s, y)
        _check(abs(a_sp - a_np) < 1e-3 and abs(b_sp - b_np) < 1e-3, f"Platt mismatch: ({a_sp},{b_sp}) vs ({a_np},{b_np})", errors)
    finally:
        su.stats = saved_stats
        su.optimize = saved_opt

    # 3. Pure-math binomial survival P(Bin(n, p) >= 10) vs scipy.stats.binom.sf(9, n, p)
    def _binom_sf_9(n_trials: int, p_val: float) -> float:
        if p_val <= 0.0 or n_trials < 10:
            return 0.0
        if p_val >= 1.0:
            return 1.0
        lp, lq = math.log(p_val), math.log(1.0 - p_val)
        lg_n1 = math.lgamma(n_trials + 1)
        cdf = 0.0
        for kk in range(10):
            log_pmf = lg_n1 - math.lgamma(kk + 1) - math.lgamma(n_trials - kk + 1) + kk * lp + (n_trials - kk) * lq
            if log_pmf > -700.0:
                cdf += math.exp(log_pmf)
        return float(max(0.0, min(1.0, 1.0 - cdf)))

    for n_t, p_v in [(1000, 0.005), (50000, 0.0002), (109598, 0.0001), (1000, 0.05), (28645, 1e-5)]:
        ref_b = float(stats.binom.sf(9, n_t, p_v))
        our_b = _binom_sf_9(n_t, p_v)
        _check(abs(ref_b - our_b) < 1e-9, f"binom_sf mismatch ({n_t},{p_v}): {ref_b} vs {our_b}", errors)

    # 4. BH-FDR exactness & ordering
    pvs = [0.001, 0.04, 0.03, 0.50, 0.01]
    qvs = bh_fdr(pvs)
    expected_q = [0.005, 0.05, 0.05, 0.50, 0.025]
    _check(np.allclose(qvs, expected_q, atol=1e-12), f"bh_fdr mismatch: {qvs} vs {expected_q}", errors)

    # 5. E01 Spherical Tangent Displacement Invariants
    r = np.array([1.0, 0.0, 0.0])
    # Orthogonal edit directions in T_r S -> 90 deg
    q = np.array([math.cos(0.3), math.sin(0.3), 0.0])
    t = np.array([math.cos(0.6), 0.0, math.sin(0.6)])
    g_orth = _tangent_displacement_metrics(r, q, t)
    _check(abs(g_orth["tangent_angle_deg"] - 90.0) < 1e-6, f"E01 orth angle != 90: {g_orth}", errors)
    _check(abs(g_orth["mag_ratio_geodesic"] - 0.5) < 1e-6, f"E01 mag ratio != 0.5: {g_orth}", errors)
    # Same direction, 2x overshoot -> 0 deg, mag_ratio = 2.0
    t_same = np.array([math.cos(0.15), math.sin(0.15), 0.0])
    g_same = _tangent_displacement_metrics(r, q, t_same)
    _check(abs(g_same["tangent_angle_deg"]) < 1e-5 and abs(g_same["mag_ratio_geodesic"] - 2.0) < 1e-5, f"E01 same dir err: {g_same}", errors)
    # Degenerate zero-delta (q == r)
    g_deg = _tangent_displacement_metrics(r, r, t)
    _check(g_deg["degenerate_zero_delta"] and g_deg["zero_q"] and not g_deg["zero_t"], f"E01 zero-delta err: {g_deg}", errors)

    # 6. E04 Classification & Ridge Probe sanity (separable vs noise)
    rng = np.random.default_rng(7)
    X_tr = rng.normal(0, 1, (200, 32))
    y_tr = (X_tr[:, 0] + 0.5 * X_tr[:, 1] > 0).astype(np.int64)
    X_dv = rng.normal(0, 1, (80, 32))
    y_dv = (X_dv[:, 0] + 0.5 * X_dv[:, 1] > 0).astype(np.int64)
    X_te = rng.normal(0, 1, (100, 32))
    y_te = (X_te[:, 0] + 0.5 * X_te[:, 1] > 0).astype(np.int64)
    pr = run_classification_probe(X_tr, y_tr, X_dv, y_dv, X_te, y_te, test_groups=list(range(100)), match_dim=16, seed=42)
    _check(pr["native"]["test_auroc"] > 0.90 and pr["native"]["test_acc"] > 0.80 and pr["shuffled_control"]["test_acc"] < 0.65, f"Probe separable sanity fail: {pr}", errors)

    # Pure noise must NOT overfit!
    y_te_noise = rng.integers(0, 2, size=100)
    pr_n = run_classification_probe(X_tr, rng.integers(0, 2, size=200), X_dv, rng.integers(0, 2, size=80), X_te, y_te_noise, match_dim=16, seed=42)
    _check(0.30 <= pr_n["native"]["test_acc"] <= 0.70, f"Probe noise sanity fail: {pr_n['native']}", errors)

    # Ridge cardinality probe sanity
    yc_tr = np.clip(np.rint(2.5 + X_tr[:, 0]), 1, 4)
    yc_dv = np.clip(np.rint(2.5 + X_dv[:, 0]), 1, 4)
    yc_te = np.clip(np.rint(2.5 + X_te[:, 0]), 1, 4)
    rr = run_ridge_cardinality_probe(X_tr, yc_tr, X_dv, yc_dv, X_te, yc_te, match_dim=16, seed=42)
    _check(rr["native"]["test_mae"] < rr["prior_median"]["test_mae"], f"Ridge probe sanity fail: {rr}", errors)
    print(f"  Math unit tests done (errors so far: {len(errors)})", flush=True)


def verify_cohorts_and_e18(out_root: Path, errors: List[str]) -> None:
    print("[VERIFY-2] Frozen cohorts zero-leakage & E18 blind pack audit...", flush=True)
    cohorts = json.loads((out_root / "manifest/diagnostic_cohorts.json").read_text())

    # E04 strict group-disjointness
    for key in ("e04_edir", "e04_fiq_gallery"):
        g_tr = {r["group_id"] for r in cohorts[key]["train"]}
        g_dv = {r["group_id"] for r in cohorts[key]["dev"]}
        g_te = {r["group_id"] for r in cohorts[key]["test"]}
        _check(len(g_tr & g_dv) == 0, f"{key} train∩dev group leak: {len(g_tr & g_dv)}", errors)
        _check(len(g_tr & g_te) == 0, f"{key} train∩test group leak: {len(g_tr & g_te)}", errors)
        _check(len(g_dv & g_te) == 0, f"{key} dev∩test group leak: {len(g_dv & g_te)}", errors)

    # E10 F22 exact matched-area check (0.28 across secondary, primary, sham)
    for ep in cohorts["e10_f22_cooccur"][:20]:
        def _area(b): return (b[2] - b[0]) * (b[3] - b[1])
        a_sec = _area(ep["secondary_mask_box"])
        a_pri = _area(ep["primary_mask_box"])
        a_shm = sum(_area(b) for b in ep["sham_mask_boxes"])
        _check(abs(a_sec - a_pri) < 1e-6 and abs(a_sec - a_shm) < 1e-6, f"E10-F22 area mismatch: {a_sec}, {a_pri}, {a_shm}", errors)

    # E11 ROI vs sham exact matched-area & non-overlapping check
    for it in cohorts["e11_items"]:
        rb, sb = it["roi_box"], it["sham_box"]
        ar = (rb[2] - rb[0]) * (rb[3] - rb[1])
        as_ = (sb[2] - sb[0]) * (sb[3] - sb[1])
        _check(abs(ar - as_) < 0.01, f"E11 ROI/sham area mismatch on {it['query_id']}: {ar} vs {as_}", errors)

    # E18 blind pack strict anonymization & image readability audit
    e18_dir = out_root / "e18_blind_audit"
    tpl_p = e18_dir / "annotator_template.csv"
    _check(tpl_p.is_file(), "Missing E18 annotator_template.csv", errors)
    forbidden_model_names = [m.key for m in MODEL_SPECS] + ["csmcir", "conesep", "clip4cir", "qure", "sprc", "mtst_ptha"]
    with tpl_p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    _check(len(rows) == 220, f"Expected 220 E18 rows, got {len(rows)}", errors)
    _check("stratum" not in rows[0] and "is_hit_top1" not in rows[0], f"E18 template leaks stratum/hit columns: {list(rows[0].keys())}", errors)
    for row in rows:
        am = row["anon_model_id"].lower()
        _check(am.startswith("m_") and not any(fn in am for fn in forbidden_model_names), f"E18 model name leak: {am}", errors)
        _check(row["annotator_label"] == "", f"E18 pre-filled label found in template: {row}", errors)
        img_p = e18_dir / row["contact_sheet_rel"]
        _check(img_p.is_file() and img_p.stat().st_size > 2000, f"E18 missing/corrupt sheet: {img_p}", errors)
    # Sample-open 20 sheets via PIL
    for row in rows[::11]:
        with PIL.Image.open(e18_dir / row["contact_sheet_rel"]) as im:
            _check(im.size[0] > 500 and im.size[1] > 200, f"E18 invalid image size: {im.size}", errors)
    print(f"  Cohort & E18 blind checks done (errors so far: {len(errors)})", flush=True)


def verify_completed_model_artifacts(out_root: Path, errors: List[str]) -> Dict[str, Any]:
    print("[VERIFY-3] Auditing all completed model artifacts on disk...", flush=True)
    summary: Dict[str, Any] = {}
    for ds in ("edir", "fashioniq", "pinpoint"):
        ds_dir = out_root / "models" / ds
        if not ds_dir.is_dir():
            continue
        checked = []
        for spec in MODEL_SPECS:
            m_dir = ds_dir / spec.key
            if not (m_dir / "DONE.json").is_file():
                continue
            checked.append(spec.key)
            # Determine directories to audit (per-cat vs unified)
            sub_dirs = sorted(m_dir.glob("cat_*")) if list(m_dir.glob("cat_*")) else [m_dir]
            for sd in sub_dirs:
                tag = f"{ds}/{spec.key}/{sd.name}"
                # Parity check
                par = json.loads((sd / "parity.json").read_text())
                _check(par.get("parity_ok") is True, f"{tag} parity_ok != True: {par}", errors)
                if ds in ("edir", "pinpoint"):
                    _check(par.get("abs_diff_primary", 1.0) <= 1e-4, f"{tag} parity abs_diff > 1e-4: {par.get('abs_diff_primary')}", errors)
                    # Allow >=0.995 for fp16 GEMM tie-breaks across 109.6k candidates (e.g., tme 5/1337 near-ties |s1-s2|<=1.7e-4)
                    _check(par.get("top1_agreement_vs_saved", 0.0) >= 0.995, f"{tag} top1_agreement < 0.995: {par.get('top1_agreement_vs_saved')}", errors)

                # E01 check
                e01 = json.loads((sd / "e01_geometry.json").read_text())
                _check(e01["n_non_degenerate"] + e01["n_degenerate_zero_delta"] == e01["n_queries"], f"{tag} E01 count mismatch", errors)
                _check(-1.0 <= e01["mean_tangent_cos"] <= 1.0, f"{tag} E01 tangent_cos out of range: {e01['mean_tangent_cos']}", errors)

                # E03 monotonicity check
                e03 = json.loads((sd / "e03_scaling.json").read_text())
                if "random" in e03["curves"] and "hard" in e03["curves"]:
                    szs = sorted(int(k) for k in e03["curves"]["random"].keys())
                    r10_first = e03["curves"]["random"][str(szs[0])]["R@10"]
                    r10_last = e03["curves"]["random"][str(szs[-1])]["R@10"]
                    _check(r10_first + 1e-6 >= r10_last, f"{tag} E03 non-monotonic random R@10: {r10_first} < {r10_last}", errors)

                # E05 mathematical inequality check: oracle >= max(img, txt, multi) on EVERY triplet!
                e05 = json.loads((sd / "e05_fusion_dominance.json").read_text())
                pn = e05["pure_fusion_failure_native"]
                _check(0 <= pn["n_ppp_minus"] <= pn["n_both_unimodal_pos"] <= e05["n_triplets"], f"{tag} E05 count invariant fail", errors)
                for row in e05["per_query"]:
                    mx = max(row["native_img_tower"], row["native_txt_tower"], row["multimodal"])
                    _check(row["oracle_native_or_fused"] + 1e-4 >= mx, f"{tag} E05 oracle < max on {row['query_id']}", errors)

                # E07 exact identity check: bidir + inversion <= acc_pos!
                for fn in ("e07_negation_edir.json", "e07_negation_fiq.json"):
                    p7 = sd / fn
                    if p7.is_file():
                        j7 = json.loads(p7.read_text())
                        blocks = [j7] if "acc_with_X" in j7 else [j7["with_without"], j7["add_remove"]]
                        for b in blocks:
                            ap = b.get("acc_with_X", b.get("acc_pos_polarity"))
                            an = b.get("acc_without_X", b.get("acc_neg_polarity"))
                            bf = b["bidirectional_flip_acc"]
                            inv = b["negation_inversion_rate"]
                            _check(bf <= min(ap, an) + 1e-6, f"{tag} {fn} bidir > min(pos,neg)", errors)
                            _check(abs((bf + inv) - ap) < 1e-6, f"{tag} {fn} bidir+inv != acc_pos ({bf}+{inv} vs {ap})", errors)

                # E14 bounds check
                e14 = json.loads((sd / "e14_hubness.json").read_text())
                sg = e14["spectral_geometry"]
                _check(0.0 <= e14["baseline_raw"]["gini_k10"] <= 1.0, f"{tag} E14 gini out of bounds", errors)
                _check(1.0 <= sg["effective_rank"] <= spec.embed_dim + 1, f"{tag} E14 eff_rank out of bounds: {sg['effective_rank']}", errors)
                _check(-1.0 <= sg["anisotropy_mean_cos"] <= 1.0, f"{tag} E14 anisotropy out of bounds", errors)

                # E15 bounds check
                e15 = json.loads((sd / "e15_calibration.json").read_text())
                _check(e15["n_notarget"] + e15["n_present"] == e15["n_test"], f"{tag} E15 notarget+present != n_test", errors)
                for sig_k, sv in e15["hit_miss_calibration"].items():
                    for mk in ("AUROC", "AUPRC", "ECE", "Brier", "aurc"):
                        if mk in sv and sv[mk] is not None and math.isfinite(sv[mk]):
                            _check(0.0 <= sv[mk] <= 1.0 + 1e-6, f"{tag} E15 {sig_k}.{mk}={sv[mk]} out of [0,1]", errors)

                # E04 NPZ finite & exact row-count check
                for npz_name, exp_counts in [("e04_edir_layers.npz", {"train": 1400, "dev": 497, "test": 1003}), ("e04_fiq_gallery_layers.npz", {"train": 1080, "dev": 360, "test": 360})]:
                    np_p = sd / npz_name
                    if np_p.is_file():
                        arrs = np.load(np_p)
                        for k_a in arrs.files:
                            sp_p = k_a.split("__", 1)[0]
                            a = arrs[k_a]
                            _check(a.shape[0] == exp_counts[sp_p], f"{tag} {npz_name}:{k_a} rows={a.shape[0]} != {exp_counts[sp_p]}", errors)
                            _check(np.isfinite(a).all(), f"{tag} {npz_name}:{k_a} contains NaN/Inf!", errors)
                            _check(float(a.astype(np.float32).std()) > 1e-6, f"{tag} {npz_name}:{k_a} constant zero variance!", errors)
        summary[ds] = {"n_checked": len(checked), "models": checked}
        print(f"  Audited {ds}: {len(checked)} models (errors so far: {len(errors)})", flush=True)
    return summary


def verify_training_arms(out_root: Path, errors: List[str]) -> Dict[str, Any]:
    print("[VERIFY-4] Auditing controlled training & dynamics artifacts (E02, E16, E17)...", flush=True)
    summary: Dict[str, Any] = {"e02": [], "e16": [], "e17_traj": [], "e17_history": False}

    # E17 history all16
    h_p = out_root / "e17_dynamics/e17_history_all16.json"
    if h_p.is_file():
        hd = json.loads(h_p.read_text())
        summary["e17_history"] = True
        for ds in ("edir", "pinpoint"):
            ds_d = hd.get("datasets", {}).get(ds, {})
            _check(len(ds_d) == 16, f"E17 history {ds} expected 16 models, got {len(ds_d)}", errors)
            for mk, mv in ds_d.items():
                _check(mv.get("n_epochs_logged", 0) >= 1, f"E17 history {ds}/{mk} empty curve", errors)

    # E02, E16, E17_traj per trained model
    for mk in ("clip4cir", "conesep", "sprc"):
        e2_dir = out_root / "e02_same_scene" / mk
        if (e2_dir / "DONE.json").is_file():
            summary["e02"].append(mk)
            ed2 = json.loads((e2_dir / "e02_edir_results.json").read_text())
            pp2 = json.loads((e2_dir / "e02_pinpoint_results.json").read_text())
            _check(len(ed2.get("arms", {})) == 7, f"E02 EDIR {mk} expected 7 arms, got {len(ed2.get('arms', {}))}", errors)
            _check(len(pp2.get("arms", {})) == 4, f"E02 PP {mk} expected 4 arms, got {len(pp2.get('arms', {}))}", errors)
            ed_steps = {av["optimizer_steps"] for av in ed2["arms"].values()}
            _check(len(ed_steps) == 1 and next(iter(ed_steps)) > 0, f"E02 EDIR {mk} unmatched step counts: {ed_steps}", errors)
            for aname, av in ed2["arms"].items():
                _check(abs(av["achieved_neg_ratio"] - av["requested_neg_ratio"]) <= 0.01, f"E02 EDIR {mk}/{aname} ratio mismatch", errors)
                tm = av["test_metrics"]
                for k_m in ("R@1", "R@5", "R@10", "Scene@1", "CondR@1_within_scene", "RefBeatsPos_rate"):
                    v = tm.get(k_m)
                    _check(v is not None and math.isfinite(v) and 0.0 <= v <= 1.0, f"E02 EDIR {mk}/{aname} {k_m}={v}", errors)
                _check(tm["R@1"] <= tm["R@5"] + 1e-6 <= tm["R@10"] + 1e-6, f"E02 EDIR {mk}/{aname} R@1<=R@5<=R@10 fail", errors)
            pp_steps = {av["optimizer_steps"] for av in pp2["arms"].values()}
            _check(len(pp_steps) == 1 and next(iter(pp_steps)) > 0, f"E02 PP {mk} unmatched step counts: {pp_steps}", errors)
            for aname, av in pp2["arms"].items():
                tm = av["test_metrics"]
                for k_m in ("raw_mAP@10", "noref_mAP@10", "RefCopy@1", "raw_NegRecall@10"):
                    v = tm.get(k_m)
                    _check(v is not None and math.isfinite(v) and 0.0 <= v <= 1.0, f"E02 PP {mk}/{aname} {k_m}={v}", errors)

        e16_dir = out_root / "e16_domain_conflict" / mk
        if (e16_dir / "DONE.json").is_file():
            summary["e16"].append(mk)
            gc16 = json.loads((e16_dir / "e16_gradient_cosine.json").read_text())
            tr16 = json.loads((e16_dir / "e16_transfer_matrix.json").read_text())
            for tag in ("init_pretrained", "trained_pinpoint_best"):
                gm = gc16.get(tag, {})
                doms = gm.get("domains", [])
                mat = gm.get("cosine_matrix", {})
                _check(len(doms) == 7, f"E16 {mk} {tag} expected 7 domains", errors)
                for d1 in doms:
                    _check(abs(mat[d1][d1] - 1.0) < 1e-3, f"E16 {mk} {tag} diag({d1})={mat[d1][d1]} != 1", errors)
                    for d2 in doms:
                        _check(-1.001 <= mat[d1][d2] <= 1.001, f"E16 {mk} {tag} cos out of bounds", errors)
                        _check(abs(mat[d1][d2] - mat[d2][d1]) < 1e-4, f"E16 {mk} {tag} asymmetric matrix", errors)
            _check(len(tr16.get("arms", {})) == 8, f"E16 {mk} expected 8 transfer arms", errors)
            steps_set = {av["total_steps"] for av in tr16["arms"].values()}
            exp_set = {av["total_exposures"] for av in tr16["arms"].values()}
            _check(len(steps_set) == 1, f"E16 {mk} unmatched steps across arms: {steps_set}", errors)
            _check(len(exp_set) == 1, f"E16 {mk} unmatched exposures across arms: {exp_set}", errors)
            for aname, av in tr16["arms"].items():
                v = av.get("overall_raw_mAP10")
                _check(v is not None and math.isfinite(v) and 0.0 <= v <= 1.0, f"E16 {mk}/{aname} overall mAP={v}", errors)

        tp = out_root / "e17_dynamics" / f"e17_trajectory_{mk}.json"
        if tp.is_file():
            summary["e17_traj"].append(mk)
            tj = json.loads(tp.read_text())
            _check(len(tj.get("edir_trajectory", [])) >= 4, f"E17 traj {mk} EDIR < 4 stages", errors)
            _check(len(tj.get("pinpoint_trajectory", [])) >= 4, f"E17 traj {mk} PP < 4 stages", errors)

    print(f"  Training/dynamics checks done: {summary} (errors so far: {len(errors)})", flush=True)
    return summary


def verify_report_integrity(out_root: Path, report_path: Optional[Path], errors: List[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"checked": False, "n_links": 0, "broken_links": 0}
    if report_path is None or not report_path.is_file():
        return info
    print(f"[VERIFY-5] Auditing final report integrity & file:// links ({report_path})...", flush=True)
    text = report_path.read_text(encoding="utf-8")
    info["checked"] = True
    # 1. Check all 29 failures F01..F29 and all 18 suites E01..E18 appear
    for i in range(1, 30):
        _check(f"F{i:02d}" in text, f"Report missing failure F{i:02d}", errors)
    for i in range(1, 19):
        _check(f"E{i:02d}" in text, f"Report missing suite E{i:02d}", errors)
    # 2. Check ZERO stray 'nan' / 'NaN' / 'inf' tokens inside markdown table cells
    bad_cells = re.findall(r"\|\s*(?:nan|NaN|inf|-inf|None)\s*(?:\||/)", text)
    _check(len(bad_cells) == 0, f"Report contains {len(bad_cells)} NaN/None table cells: {bad_cells[:5]}", errors)
    # 3. Verify EVERY file:/// link resolves to an existing path on disk!
    links = re.findall(r"\(file://(/[^)#\s]+)(?:#[^)\s]*)?\)", text)
    info["n_links"] = len(links)
    broken = []
    for lp in links:
        if not Path(lp).exists():
            broken.append(lp)
    info["broken_links"] = len(broken)
    _check(len(broken) == 0, f"Report contains {len(broken)} broken file:// links: {broken[:10]}", errors)
    # 4. Cross-check coverage_matrix.json completed_model_counts appear verbatim in report header
    cov_p = out_root / "aggregated/coverage_matrix.json"
    if cov_p.is_file():
        cov = json.loads(cov_p.read_text())
        cm = cov.get("completed_model_counts", {})
        for ds_lbl, k_ds in [("EDIR", "edir"), ("PinPoint", "pinpoint"), ("FashionIQ", "fashioniq")]:
            exp_tok = f"{ds_lbl} = **`{cm.get(k_ds)}/16`**"
            _check(exp_tok in text, f"Report coverage header mismatch for {ds_lbl}: expected '{exp_tok}'", errors)
    print(f"  Report check done: {info} (errors so far: {len(errors)})", flush=True)
    return info


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-round verification suite for CIR probes")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROBE_OUT)
    parser.add_argument("--report", type=Path, nargs="*", default=None)
    args = parser.parse_args()

    out_root = args.out_dir
    errors: List[str] = []
    verify_math_unit_tests(errors)
    verify_cohorts_and_e18(out_root, errors)
    summary = verify_completed_model_artifacts(out_root, errors)
    tr_summary = verify_training_arms(out_root, errors)
    rep_summaries = [verify_report_integrity(out_root, rp, errors) for rp in (args.report or [])]
    print("\n================ VERIFICATION SUMMARY ================")
    print("Checked models:", json.dumps(summary, indent=2))
    print("Checked training arms:", json.dumps(tr_summary, indent=2))
    if rep_summaries:
        print("Checked reports:", json.dumps(rep_summaries, indent=2))
    print(f"TOTAL ERRORS: {len(errors)}")
    for e in errors[:40]:
        print("  [FAIL]", e)
    if errors:
        sys.exit(1)
    print("[ALL VERIFICATION CHECKS PASSED WITH ZERO ERRORS]")


if __name__ == "__main__":
    main()
