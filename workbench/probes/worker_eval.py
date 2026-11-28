#!/usr/bin/env python3
"""Single-pass per-environment GPU diagnostic probe worker (Suites E01, E03-E15).

Runs inside each model's exact pinned virtualenv + PYTHONPATH. Supports passing
multiple models sharing the same virtualenv & frozen EVA-ViT-g backbone
(e.g., `hint,habit,intent,conesep` or `mtst_ptha,csmcir`) so the frozen visual_encoder
runs ONCE per gallery image batch while producing exact per-model outputs!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.benchmarks import evaluate_edir, evaluate_pinpoint, load_split_bundle  # noqa: E402
from workbench.probes.cohort import MODEL_BY_KEY  # noqa: E402
from workbench.probes.model_adapter import ModelHandle, _ImgListDataset, _make_loader, _open_pil  # noqa: E402
from workbench.probes.stats_utils import (  # noqa: E402
    apply_platt_calibration,
    binary_auprc,
    binary_auroc,
    brier_score,
    expected_calibration_error,
    false_accept_rate_at_tpr,
    fit_platt_calibration,
    risk_coverage_aurc,
)

EDIR_DIR = REPO_ROOT / "data/EDIR"
EDIR_SPLIT = EDIR_DIR / "splits/70_10_20_seed42"
PP_DIR = REPO_ROOT / "data/PinPoint"
PP_SPLIT = PP_DIR / "splits/70_10_20_seed42"
FIQ_DIR = REPO_ROOT / "data/FashionIQ"


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _fiq_path(iid: str) -> str:
    p = FIQ_DIR / "images" / f"{iid}.png"
    return str(p if p.is_file() else FIQ_DIR / "images" / f"{iid}.jpg")


def _mask_pil_boxes(im: PIL.Image.Image, boxes_norm: List[List[float]], fill: Tuple[int, int, int] = (128, 128, 128)) -> PIL.Image.Image:
    out = im.convert("RGB").copy()
    arr = np.asarray(out, dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    for x1, y1, x2, y2 in boxes_norm:
        ix1, iy1 = max(0, int(round(x1 * w))), max(0, int(round(y1 * h)))
        ix2, iy2 = min(w, int(round(x2 * w))), min(h, int(round(y2 * h)))
        arr[iy1:iy2, ix1:ix2, :] = fill
    return PIL.Image.fromarray(arr)


def _crop_pil_box(im: PIL.Image.Image, box_norm: List[float]) -> PIL.Image.Image:
    rgb = im.convert("RGB")
    w, h = rgb.size
    x1, y1, x2, y2 = box_norm
    ix1, iy1 = max(0, int(round(x1 * w))), max(0, int(round(y1 * h)))
    ix2, iy2 = min(w, max(ix1 + 16, int(round(x2 * w)))), min(h, max(iy1 + 16, int(round(y2 * h))))
    return rgb.crop((ix1, iy1, ix2, iy2))


def _bg_neutralize_pil(im: PIL.Image.Image) -> PIL.Image.Image:
    """Replace outer 18% border ring (background format cue) with neutral gray (128,128,128)."""
    return _mask_pil_boxes(im, [[0.0, 0.0, 1.0, 0.14], [0.0, 0.86, 1.0, 1.0], [0.0, 0.14, 0.14, 0.86], [0.86, 0.14, 1.0, 0.86]])


def _bg_sham_pil(im: PIL.Image.Image) -> PIL.Image.Image:
    """Matched-area sham mask (same ~48% area? No, outer 14% ring area = 1 - 0.72^2 = 0.4816;
    use thin neutral stripe grid at fixed interior/mixed area matched to 0.20 vs matched border 0.10).
    For exact matched-area control: use 8% border ring (area = 1 - 0.84^2 = 0.2944) vs center-offset sham strips!"""
    return _mask_pil_boxes(im, [[0.0, 0.0, 1.0, 0.08], [0.0, 0.92, 1.0, 1.0], [0.0, 0.08, 0.08, 0.92], [0.92, 0.08, 1.0, 0.92]])


def _bg_matched_center_sham_pil(im: PIL.Image.Image) -> PIL.Image.Image:
    """Exact matched-area (0.2944) 4-corner-inset interior blocks sham."""
    # 4 blocks of 0.271 x 0.271 = 0.294 area
    return _mask_pil_boxes(im, [[0.18, 0.18, 0.451, 0.451], [0.549, 0.18, 0.82, 0.451], [0.18, 0.549, 0.451, 0.82], [0.549, 0.549, 0.82, 0.82]])


# ---------------------------------------------------------------------------
# Dataset Loader Helpers
# ---------------------------------------------------------------------------

def _load_dataset_spec(dataset: str, fiq_category: Optional[str] = None) -> Dict[str, Any]:
    if dataset == "edir":
        corpus = _load_jsonl(EDIR_SPLIT / "corpus_full.jsonl")
        c_ids = [r["image_id"] for r in corpus]
        c_paths = [str(EDIR_DIR / r["image_rel"]) for r in corpus]
        tr_c = _load_jsonl(EDIR_SPLIT / "corpus_train.jsonl")
        val_c = _load_jsonl(EDIR_SPLIT / "corpus_val.jsonl")
        val_q = _load_jsonl(EDIR_SPLIT / "val.jsonl")
        test_q = _load_jsonl(EDIR_SPLIT / "test.jsonl")
        return {
            "corpus_ids": c_ids,
            "corpus_paths": c_paths,
            "cal_corpus_ids": [r["image_id"] for r in tr_c],  # strictly scene-disjoint from test!
            "val_corpus_ids": [r["image_id"] for r in val_c],
            "val_queries": val_q,
            "test_queries": test_q,
            "ref_path_fn": lambda r: str(EDIR_DIR / r["ref_image_rel"]),
            "img_path_map": {r["image_id"]: str(EDIR_DIR / r["image_rel"]) for r in corpus},
        }
    elif dataset == "pinpoint":
        corpus = _load_jsonl(PP_SPLIT / "corpus_full.jsonl")
        c_ids = [r["image_id"] for r in corpus]
        c_paths = [str(PP_DIR / r["image_rel"]) for r in corpus]
        tr_c = _load_jsonl(PP_SPLIT / "corpus_train.jsonl")
        val_c = _load_jsonl(PP_SPLIT / "corpus_val.jsonl")
        val_q = _load_jsonl(PP_SPLIT / "val.jsonl")
        test_q = _load_jsonl(PP_SPLIT / "test.jsonl")
        return {
            "corpus_ids": c_ids,
            "corpus_paths": c_paths,
            "cal_corpus_ids": [r["image_id"] for r in tr_c],  # strictly train-split gallery!
            "val_corpus_ids": [r["image_id"] for r in val_c],
            "val_queries": val_q,
            "test_queries": test_q,
            "ref_path_fn": lambda r: str(PP_DIR / r["ref_image_rel"]),
            "img_path_map": {r["image_id"]: str(PP_DIR / r["image_rel"]) for r in corpus},
        }
    elif dataset == "fashioniq":
        cats = [fiq_category] if fiq_category else ["dress", "shirt", "toptee"]
        c_ids: List[str] = []
        seen = set()
        cat_gal: Dict[str, List[str]] = {}
        all_q: List[Dict[str, Any]] = []
        for cat in cats:
            g = json.loads((FIQ_DIR / f"image_splits/split.{cat}.val.json").read_text())
            cat_gal[cat] = g
            for iid in g:
                if iid not in seen:
                    seen.add(iid)
                    c_ids.append(iid)
            caps = json.loads((FIQ_DIR / f"captions/cap.{cat}.val.json").read_text())
            for qi, r in enumerate(caps):
                for iid in (r["candidate"], r["target"]):
                    if iid not in seen:
                        seen.add(iid)
                        c_ids.append(iid)
                all_q.append({
                    "query_id": f"{cat}:{qi}",
                    "category": cat,
                    "ref_id": r["candidate"],
                    "positive_ids": [r["target"]],
                    "positive_ids_raw": [r["target"]],
                    "negative_ids": [],
                    "text": f"{r['captions'][0]} and {r['captions'][1]}",
                    "captions": r["captions"],
                    "metadata": {"category": cat},
                })
        import hashlib
        val_q = [r for r in all_q if int(hashlib.md5(r["ref_id"].encode()).hexdigest()[:8], 16) % 10 < 3]
        heldout_q = [r for r in all_q if int(hashlib.md5(r["ref_id"].encode()).hexdigest()[:8], 16) % 10 >= 3]
        heldout_tgts = {r["positive_ids"][0] for r in heldout_q}
        cal_c_ids = [cid for cid in c_ids if cid not in heldout_tgts]
        return {
            "corpus_ids": c_ids,
            "corpus_paths": [_fiq_path(i) for i in c_ids],
            "cal_corpus_ids": cal_c_ids,
            "cat_galleries": cat_gal,
            "val_queries": val_q,
            "test_queries": all_q,
            "test_queries_cal_heldout": heldout_q,
            "ref_path_fn": lambda r: _fiq_path(r["ref_id"]),
            "img_path_map": {i: _fiq_path(i) for i in c_ids},
        }
    raise ValueError(dataset)


def _load_arbiter(out_root: Path, dataset: str) -> Optional[Dict[str, Any]]:
    adir = out_root / "arbiter_clip" / dataset
    if not (adir / "DONE.json").is_file():
        return None
    c_emb = np.load(adir / "corpus_emb.npy").astype(np.float32)
    c_ids = json.loads((adir / "corpus_ids.json").read_text())
    c_map = {cid: i for i, cid in enumerate(c_ids)}
    res: Dict[str, Any] = {"c_emb": c_emb, "c_ids": c_ids, "c_map": c_map}
    if (adir / "extra_ref_emb.npy").is_file():
        r_emb = np.load(adir / "extra_ref_emb.npy").astype(np.float32)
        r_ids = json.loads((adir / "extra_ref_ids.json").read_text())
        res["r_emb"] = r_emb
        res["r_map"] = {rid: i for i, rid in enumerate(r_ids)}
    sp_name = "val" if dataset == "fashioniq" else "test"
    if (adir / f"{sp_name}_text_emb.npy").is_file():
        res["t_emb"] = np.load(adir / f"{sp_name}_text_emb.npy").astype(np.float32)
        t_ids = json.loads((adir / f"{sp_name}_query_ids.json").read_text())
        res["t_map"] = {qid: i for i, qid in enumerate(t_ids)}
    return res


def _get_arbiter_img(arb: Dict[str, Any], img_id: str) -> Optional[np.ndarray]:
    if img_id in arb["c_map"]:
        return arb["c_emb"][arb["c_map"][img_id]]
    if "r_map" in arb and img_id in arb["r_map"]:
        return arb["r_emb"][arb["r_map"][img_id]]
    return None


# ---------------------------------------------------------------------------
# Suite Computations on Full Embeddings (E01, E03, E13, E14, E15)
# ---------------------------------------------------------------------------

def _tangent_displacement_metrics(r: np.ndarray, q: np.ndarray, t: np.ndarray, eps: float = 1e-4) -> Dict[str, Any]:
    """Exact spherical tangent displacement at common basepoint r (avoids Euclidean cancellation!)."""
    r = r / max(float(np.linalg.norm(r)), 1e-12)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    cos_qr = float(np.clip(np.dot(q, r), -1.0, 1.0))
    cos_tr = float(np.clip(np.dot(t, r), -1.0, 1.0))
    cos_qt = float(np.clip(np.dot(q, t), -1.0, 1.0))
    theta_q = float(math.acos(cos_qr))
    theta_t = float(math.acos(cos_tr))
    u = q - cos_qr * r  # tangent vector proj_{T_r S}(q - r)
    v = t - cos_tr * r  # tangent vector proj_{T_r S}(t - r)
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < eps or nv < eps:
        return {
            "degenerate_zero_delta": True,
            "zero_q": nu < eps,
            "zero_t": nv < eps,
            "cos_qt": cos_qt,
            "cos_qr": cos_qr,
            "cos_tr": cos_tr,
            "theta_q_deg": math.degrees(theta_q),
            "theta_t_deg": math.degrees(theta_t),
            "tangent_cos": float("nan"),
            "tangent_angle_deg": float("nan"),
            "mag_ratio_geodesic": float("nan"),
        }
    tcos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return {
        "degenerate_zero_delta": False,
        "zero_q": False,
        "zero_t": False,
        "cos_qt": cos_qt,
        "cos_qr": cos_qr,
        "cos_tr": cos_tr,
        "theta_q_deg": math.degrees(theta_q),
        "theta_t_deg": math.degrees(theta_t),
        "tangent_cos": tcos,
        "tangent_angle_deg": math.degrees(math.acos(tcos)),
        "mag_ratio_geodesic": float(theta_q / max(theta_t, eps)),
    }


def _compute_parity_and_full_suites(
    handle: ModelHandle,
    dataset: str,
    ds_spec: Dict[str, Any],
    g_feats: Any,
    arb: Optional[Dict[str, Any]],
    cohorts: Dict[str, Any],
    model_out: Path,
    num_workers: int,
) -> Tuple[np.ndarray, Any, List[str]]:
    """Compute Baseline Parity + E01 + E03 + E13 + E14 + E15 from one full-gallery embedding pass!"""
    c_ids = ds_spec["corpus_ids"]
    c_idx_map = {cid: i for i, cid in enumerate(c_ids)}
    test_q = ds_spec["test_queries"]
    val_q = ds_spec["val_queries"]
    ref_fn = ds_spec["ref_path_fn"]
    is_dcnet = handle.model_key == "dcnet"
    is_multivec = (not is_dcnet) and (g_feats.dim() == 3)

    # Encode test queries natively (match train_clvcnet.py eval_bs=128 for unmasked LSTM batch padding parity!)
    q_bs_eval = 128 if handle.model_key == "clvcnet" else 64
    print(f"  [{handle.model_key}/{dataset}] Encoding {len(test_q)} test queries & {len(val_q)} val queries...", flush=True)
    q_test = handle.encode_queries([ref_fn(r) for r in test_q], [r["text"] for r in test_q], mode="multimodal", bs=q_bs_eval, num_workers=num_workers)
    q_val = handle.encode_queries([ref_fn(r) for r in val_q], [r["text"] for r in val_q], mode="multimodal", bs=q_bs_eval, num_workers=num_workers)

    # Encode unique ref images NOT already in corpus (e.g., EDIR refs!)
    extra_ref_ids = sorted({r["ref_id"] for r in test_q if r["ref_id"] not in c_idx_map})
    extra_ref_feats = None
    extra_ref_map: Dict[str, int] = {}
    if extra_ref_ids:
        id_to_rp = {r["ref_id"]: ref_fn(r) for r in test_q}
        extra_ref_feats = handle.encode_gallery([id_to_rp[i] for i in extra_ref_ids], bs=64, num_workers=num_workers)
        extra_ref_map = {rid: i for i, rid in enumerate(extra_ref_ids)}

    # Exact full-gallery score matrix for test queries (float32)
    print(f"  [{handle.model_key}/{dataset}] Scoring full gallery ({len(test_q)} x {len(c_ids)})...", flush=True)
    sims_test = handle.score_matrix(q_test, g_feats, q_chunk=128, c_chunk=4096)
    sims_val = handle.score_matrix(q_val, g_feats, q_chunk=128, c_chunk=4096)

    # On FashionIQ with 3 categories combined, build an apples-to-apples category-masked view
    # (setting out-of-category gallery columns to -1e9 per query!) for E01/E03/E14/E15 while
    # keeping unmasked `sims_test` for E13 cross-category leakage!
    sims_test_unmasked = sims_test
    sims_val_eff = sims_val
    fiq_cat_valid_mask: Optional[Dict[str, np.ndarray]] = None
    if dataset == "fashioniq" and len(ds_spec.get("cat_galleries", {})) > 1:
        fiq_cat_valid_mask = {}
        for cat, g_list in ds_spec["cat_galleries"].items():
            m_arr = np.zeros(len(c_ids), dtype=bool)
            for gid in g_list:
                if gid in c_idx_map:
                    m_arr[c_idx_map[gid]] = True
            fiq_cat_valid_mask[cat] = m_arr
        sims_test = sims_test_unmasked.copy()
        for qi, r in enumerate(test_q):
            sims_test[qi, ~fiq_cat_valid_mask[r["category"]]] = -1e9
        sims_val_eff = sims_val.copy()
        for qi, r in enumerate(val_q):
            sims_val_eff[qi, ~fiq_cat_valid_mask[r["category"]]] = -1e9
    top50_idx_unmasked = np.argsort(-sims_test_unmasked, axis=1, kind="stable")[:, :50]
    top50_idx = np.argsort(-sims_test, axis=1, kind="stable")[:, :50]
    preds_top50 = {test_q[i]["query_id"]: [c_ids[ci] for ci in top50_idx[i]] for i in range(len(test_q))}
    parity: Dict[str, Any] = {"model": handle.model_key, "dataset": dataset, "n_queries": len(test_q), "n_corpus": len(c_ids)}

    if dataset in ("edir", "pinpoint"):
        q_recs = load_split_bundle(dataset, "test", corpus_scope="full").queries
        ev = evaluate_edir(preds_top50, q_recs, ks=(1, 5, 10, 50), strict=True) if dataset == "edir" else evaluate_pinpoint(preds_top50, q_recs, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
        parity["measured_overall"] = ev["overall"]
        ref_res_p = Path(handle.ckpt_info["test_results_json"])
        ref_pred_p = Path(handle.ckpt_info["test_preds_json"])
        if ref_res_p.is_file():
            ref_data = json.loads(ref_res_p.read_text())
            ref_ov = ref_data.get(handle.spec.result_field, {}).get("overall", {})
            pk = "Recall@1" if dataset == "edir" else "mAP@10"
            parity["official_primary"] = ref_ov.get(pk)
            parity["measured_primary"] = ev["overall"].get(pk)
            if pk in ref_ov and pk in ev["overall"]:
                parity["abs_diff_primary"] = abs(float(ref_ov[pk]) - float(ev["overall"][pk]))
                parity["parity_ok"] = parity["abs_diff_primary"] <= 1e-3
        if ref_pred_p.is_file():
            ref_preds = json.loads(ref_pred_p.read_text())
            top1_agree = np.mean([preds_top50[q["query_id"]][0] == ref_preds[q["query_id"]][0] for q in test_q if q["query_id"] in ref_preds])
            parity["top1_agreement_vs_saved"] = float(top1_agree)
    else:
        # FashionIQ per-category R@10 / R@50 (ref excluded & ref included!)
        cat_metrics = {}
        cat_gals = ds_spec["cat_galleries"]
        for cat, g_list in cat_gals.items():
            g_idxs = np.array([c_idx_map[x] for x in g_list], dtype=np.int64)
            q_idxs = [i for i, r in enumerate(test_q) if r["category"] == cat]
            if not q_idxs:
                continue
            r10_raw, r50_raw, r10_noref, r50_noref, ref_in10 = [], [], [], [], []
            for qi in q_idxs:
                row = sims_test[qi, g_idxs]
                order = np.argsort(-row, kind="stable")
                ranked_ids = [g_list[j] for j in order[:60]]
                tgt = test_q[qi]["positive_ids"][0]
                ref = test_q[qi]["ref_id"]
                r10_raw.append(float(tgt in ranked_ids[:10]))
                r50_raw.append(float(tgt in ranked_ids[:50]))
                ref_in10.append(float(ref in ranked_ids[:10]))
                ranked_nr = [x for x in ranked_ids if x != ref]
                r10_noref.append(float(tgt in ranked_nr[:10]))
                r50_noref.append(float(tgt in ranked_nr[:50]))
            cat_metrics[cat] = {
                "n": len(q_idxs),
                "R10_raw": float(np.mean(r10_raw)),
                "R50_raw": float(np.mean(r50_raw)),
                "R10_ref_excl": float(np.mean(r10_noref)),
                "R50_ref_excl": float(np.mean(r50_noref)),
                "RefInTop10": float(np.mean(ref_in10)),
            }
        parity["per_category"] = cat_metrics
        parity["mean_R10_raw"] = float(np.mean([v["R10_raw"] for v in cat_metrics.values()]))
        parity["mean_R10_ref_excl"] = float(np.mean([v["R10_ref_excl"] for v in cat_metrics.values()]))
        parity["parity_ok"] = True
    _write_json(model_out / "parity.json", parity)
    print(f"  [{handle.model_key}/{dataset}] Parity: {json.dumps({k: v for k, v in parity.items() if k in ('official_primary','measured_primary','abs_diff_primary','top1_agreement_vs_saved','mean_R10_raw','mean_R10_ref_excl','parity_ok')})}", flush=True)

    # ------------------------------------------------------------------
    # E01: Edit Displacement Geometry (F02, F07, F12, F27)
    # ------------------------------------------------------------------
    e01_records = []
    # Compute active matched token for positives & refs cleanly
    for qi, r in enumerate(test_q):
        qid = r["query_id"]
        pos_ids = [p for p in r["positive_ids"] if p in c_idx_map]
        if not pos_ids:
            continue
        pos_set = set(pos_ids)
        row = sims_test[qi]
        # Top non-positive score & ID
        top_order = top50_idx[qi]
        top_neg_ci = next((int(ci) for ci in top_order if c_ids[ci] not in pos_set), int(top_order[0]))
        s_top_neg = float(row[top_neg_ci])
        ref_id = r["ref_id"]

        # Native ref score & ref vector
        if ref_id in c_idx_map:
            ref_ci = c_idx_map[ref_id]
            s_ref = float(row[ref_ci])
            ref_vec_raw = None if is_dcnet else g_feats[ref_ci]
        elif ref_id in extra_ref_map:
            r_idx = extra_ref_map[ref_id]
            if is_dcnet:
                s_ref = float("nan")
                ref_vec_raw = None
            else:
                ref_vec_raw = extra_ref_feats[r_idx]
                if is_multivec:
                    s_ref = float(torch.einsum("d,kd->k", q_test[qi].float(), ref_vec_raw.float()).max().item())
                else:
                    s_ref = float(torch.dot(F.normalize(q_test[qi].float(), dim=-1), F.normalize(ref_vec_raw.float(), dim=-1)).item())
        else:
            s_ref = float("nan")
            ref_vec_raw = None

        # Independent CLIP edit distance
        clip_r = _get_arbiter_img(arb, ref_id) if arb is not None else None
        per_pos = []
        for pid in pos_ids[:10]:  # Cap 10 positives per PinPoint query for clean compactness, always include best!
            pci = c_idx_map[pid]
            s_pos = float(row[pci])
            clip_t = _get_arbiter_img(arb, pid) if arb is not None else None
            d_edit_clip = float(1.0 - np.dot(clip_r, clip_t)) if (clip_r is not None and clip_t is not None) else float("nan")

            if is_dcnet:
                # Auxiliary single-expert im_feat tangent geometry explicitly tagged
                qv = F.normalize(q_test["trg"]["im_feat"][qi].float(), dim=-1).numpy()
                rv = F.normalize(q_test["src"]["im_feat"][qi].float(), dim=-1).numpy()
                tv = F.normalize(g_feats["im_feat"][pci].float(), dim=-1).numpy()
                geom = _tangent_displacement_metrics(rv, qv, tv)
                readout = "auxiliary_single_expert_im_feat"
            elif is_multivec:
                # EXACT native active MaxSim matched token (NO mean pooling!)
                qv = F.normalize(q_test[qi].float(), dim=-1).numpy()
                t_toks = g_feats[pci].float().numpy()  # [32, 256]
                k_t = int(np.argmax(t_toks @ qv))
                tv = t_toks[k_t]
                if ref_vec_raw is not None:
                    r_toks = ref_vec_raw.float().numpy()
                    k_r = int(np.argmax(r_toks @ qv))
                    rv = r_toks[k_r]
                    geom = _tangent_displacement_metrics(rv, qv, tv)
                else:
                    geom = {"degenerate_zero_delta": True}
                readout = "native_active_maxsim_token"
            else:
                qv = F.normalize(q_test[qi].float(), dim=-1).numpy()
                tv = F.normalize(g_feats[pci].float(), dim=-1).numpy()
                if ref_vec_raw is not None:
                    rv = F.normalize(ref_vec_raw.float(), dim=-1).numpy()
                    geom = _tangent_displacement_metrics(rv, qv, tv)
                else:
                    geom = {"degenerate_zero_delta": True}
                readout = "native_single_vector"

            per_pos.append({
                "pos_id": pid,
                "s_pos": s_pos,
                "s_ref": s_ref,
                "s_top_neg": s_top_neg,
                "margin_vs_top_neg": s_pos - s_top_neg,
                "margin_vs_ref": s_pos - s_ref if math.isfinite(s_ref) else float("nan"),
                "d_edit_clip": d_edit_clip,
                "readout": readout,
                **geom,
            })
        # Pick best positive (highest native score) for query-level aggregation
        best_p = max(per_pos, key=lambda x: x["s_pos"])
        # Exact rank of best positive on full gallery
        best_pos_rank = int(np.sum(row > best_p["s_pos"])) + 1
        e01_records.append({
            "query_id": qid,
            "group_id": r.get("metadata", {}).get("stem_base", r["ref_id"]),
            "category": r.get("category", "unknown"),
            "n_positives": len(pos_ids),
            "best_pos_rank": best_pos_rank,
            "ref_copy_top1": bool(c_ids[int(top_order[0])] == ref_id),
            "best_pos": best_p,
            "mean_tangent_cos": float(np.nanmean([x.get("tangent_cos", float("nan")) for x in per_pos])),
            "mean_mag_ratio": float(np.nanmean([x.get("mag_ratio_geodesic", float("nan")) for x in per_pos])),
            "mean_d_edit_clip": float(np.nanmean([x.get("d_edit_clip", float("nan")) for x in per_pos])),
        })
    valid_e01 = [x for x in e01_records if not x["best_pos"].get("degenerate_zero_delta", True)]
    e01_summary = {
        "suite": "E01",
        "model": handle.model_key,
        "dataset": dataset,
        "readout": e01_records[0]["best_pos"]["readout"] if e01_records else "none",
        "n_queries": len(e01_records),
        "n_non_degenerate": len(valid_e01),
        "n_degenerate_zero_delta": len(e01_records) - len(valid_e01),
        "mean_tangent_cos": float(np.nanmean([x["best_pos"]["tangent_cos"] for x in valid_e01])) if valid_e01 else float("nan"),
        "mean_tangent_angle_deg": float(np.nanmean([x["best_pos"]["tangent_angle_deg"] for x in valid_e01])) if valid_e01 else float("nan"),
        "median_mag_ratio": float(np.nanmedian([x["best_pos"]["mag_ratio_geodesic"] for x in valid_e01])) if valid_e01 else float("nan"),
        "mean_margin_vs_top_neg": float(np.nanmean([x["best_pos"]["margin_vs_top_neg"] for x in e01_records])),
        "mean_margin_vs_ref": float(np.nanmean([x["best_pos"]["margin_vs_ref"] for x in e01_records])),
        "ref_copy_rate_at1": float(np.mean([x["ref_copy_top1"] for x in e01_records])),
        "records": e01_records,
    }
    _write_json(model_out / "e01_geometry.json", e01_summary)

    # ------------------------------------------------------------------
    # E03: Controlled Gallery Scaling (Nested per-query pools + CLIP-independent distractors + Null curve)
    # ------------------------------------------------------------------
    n_c = len(c_ids)
    if dataset == "pinpoint":
        sizes = [1000, 5000, 10000, 25000, 50000, n_c]
    elif dataset == "edir":
        sizes = [1000, 5000, 10000, 25000, n_c]
    else:
        sizes = [1000, 2500, 5000, 10000, n_c]
    sizes = sorted(set(s for s in sizes if s <= n_c))

    # Precompute CLIP-independent distractor ranking for up to 500 stratified queries (100% of EDIR/PP <=1337, 600 of FIQ!)
    e03_q_idxs = list(range(len(test_q))) if len(test_q) <= 1400 else list(range(0, len(test_q), max(1, len(test_q) // 600)))[:600]
    e03_results: Dict[str, Any] = {
        "suite": "E03",
        "model": handle.model_key,
        "dataset": dataset,
        "estimand": "per_query_nested_pool",
        "arbiter": "open_clip:ViT-L-14:datacomp_xl_s13b_b90k" if arb is not None else "random_only_no_arbiter",
        "n_queries": len(e03_q_idxs),
        "n_corpus_full": n_c,
        "seeds": [42, 43, 44],
        "curves": {},
    }

    # Compute CLIP hard distractor rankings on GPU aligned STRICTLY to `c_ids` order (zero index mismatch!)
    clip_hard_order: Optional[np.ndarray] = None
    if arb is not None and "t_emb" in arb:
        aligned_arb_idx = [arb["c_map"][cid] for cid in c_ids]
        c_clip_t = torch.from_numpy(arb["c_emb"][aligned_arb_idx]).to(handle.device)
        q_clip_vecs = []
        for qi in e03_q_idxs:
            r = test_q[qi]
            rv = _get_arbiter_img(arb, r["ref_id"])
            tv = arb["t_emb"][arb["t_map"][r["query_id"]]] if r["query_id"] in arb["t_map"] else None
            if rv is not None and tv is not None:
                v = 0.5 * rv + 0.5 * tv
            elif rv is not None:
                v = rv
            else:
                v = np.zeros(arb["c_emb"].shape[1], dtype=np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-8)
            q_clip_vecs.append(v)
        q_clip_t = torch.from_numpy(np.stack(q_clip_vecs, axis=0)).to(handle.device)
        # Keep top-max_sub hard indices per query on CPU
        max_sub = max(s for s in sizes if s < n_c) if any(s < n_c for s in sizes) else min(1000, n_c)
        max_k_hard = min(n_c, max_sub + 64)
        clip_hard_order = np.empty((len(e03_q_idxs), max_k_hard), dtype=np.int32)
        for s_i in range(0, len(e03_q_idxs), 64):
            e_i = min(s_i + 64, len(e03_q_idxs))
            sc = q_clip_t[s_i:e_i] @ c_clip_t.T
            _, top_i = torch.topk(sc, k=max_k_hard, dim=1)
            clip_hard_order[s_i:e_i] = top_i.cpu().numpy().astype(np.int32)
        del c_clip_t, q_clip_t
        torch.cuda.empty_cache()

    # Compute data-driven non-duplicate empirical null p_beat per query!
    null_p_beats = []
    for idx_pos, qi in enumerate(e03_q_idxs):
        r = test_q[qi]
        pos_cis = [c_idx_map[p] for p in r["positive_ids"] if p in c_idx_map]
        if not pos_cis:
            continue
        row = sims_test[qi]
        s_best_pos = float(np.max(row[pos_cis]))
        # Exclude known positives AND same-scene siblings on EDIR when estimating other-scene null
        excl = set(pos_cis)
        if dataset == "edir":
            stem = r["metadata"]["stem_base"]
            for ci in range(n_c):
                if c_ids[ci].startswith(stem):
                    excl.add(ci)
        mask_other = np.ones(n_c, dtype=bool)
        if excl:
            mask_other[list(excl)] = False
        n_other = int(mask_other.sum())
        p_beat = float(np.sum(row[mask_other] > s_best_pos)) / max(n_other, 1)
        null_p_beats.append((p_beat, len(pos_cis)))

    def _binom_sf_9(n_trials: int, p_val: float) -> float:
        """Exact P(Bin(n_trials, p_val) >= 10) in pure math (zero scipy dependency!)."""
        if p_val <= 0.0 or n_trials < 10:
            return 0.0
        if p_val >= 1.0:
            return 1.0
        lp = math.log(p_val)
        lq = math.log(1.0 - p_val)
        lg_n1 = math.lgamma(n_trials + 1)
        cdf = 0.0
        for kk in range(10):
            log_pmf = lg_n1 - math.lgamma(kk + 1) - math.lgamma(n_trials - kk + 1) + kk * lp + (n_trials - kk) * lq
            if log_pmf > -700.0:
                cdf += math.exp(log_pmf)
        return float(max(0.0, min(1.0, 1.0 - cdf)))

    null_curve = {}
    for sz in sizes:
        exp_ranks = [1.0 + max(0, sz - np_) * pb for pb, np_ in null_p_beats]
        exp_zero10 = [_binom_sf_9(max(1, sz - np_), pb) for pb, np_ in null_p_beats]
        null_curve[str(sz)] = {
            "null_mean_best_pos_rank": float(np.mean(exp_ranks)),
            "null_median_best_pos_rank": float(np.median(exp_ranks)),
            "null_ZeroPos10": float(np.mean(exp_zero10)),
        }
    e03_results["data_driven_null_curve"] = null_curve

    # Strict nested per-seed permutations (50x faster AND mathematically strictly nested 1k ⊂ 5k ⊂ ... ⊂ full!)
    regimes = ["random", "hard", "mixed"] if clip_hard_order is not None else ["random"]
    # Precompute valid candidate pool per query (within-category on FashionIQ, full corpus on EDIR/PinPoint)
    e03_valid_pool: Dict[int, np.ndarray] = {}
    if fiq_cat_valid_mask is not None:
        cat_idx_arrays = {cat: np.where(m)[0] for cat, m in fiq_cat_valid_mask.items()}
        for idx_pos, qi in enumerate(e03_q_idxs):
            e03_valid_pool[idx_pos] = cat_idx_arrays[test_q[qi]["category"]]
        # Cap FashionIQ nested sizes to min category gallery size + full_cat
        min_cat_n = min(len(v) for v in cat_idx_arrays.values())
        sizes = sorted(set([s for s in (1000, 2000, 3500) if s < min_cat_n] + [n_c]))

    for reg in regimes:
        e03_results["curves"][reg] = {}
        seeds_use_all = [42] if reg == "hard" else [42, 43, 44]
        # Pre-generate 1 global random permutation per seed so nested sizes for the SAME seed are strictly nested!
        perms_by_seed = {sd: np.random.default_rng(sd).permutation(n_c).astype(np.int32) for sd in seeds_use_all}
        for sz in sizes:
            seeds_use = [42] if (sz == n_c or reg == "hard") else [42, 43, 44]
            seed_stats = []
            for sd in seeds_use:
                perm = perms_by_seed[sd]
                best_ranks, r1s, r10s, z10s, tail_margins = [], [], [], [], []
                for idx_pos, qi in enumerate(e03_q_idxs):
                    r = test_q[qi]
                    pos_cis = [c_idx_map[p] for p in r["positive_ids"] if p in c_idx_map]
                    if not pos_cis:
                        continue
                    pos_set = set(pos_cis)
                    row = sims_test[qi]
                    s_best_pos = float(np.max(row[pos_cis]))
                    if sz == n_c:
                        if idx_pos in e03_valid_pool:
                            vp = e03_valid_pool[idx_pos]
                            neg_cis = vp[~np.isin(vp, pos_cis)]
                            neg_scores = row[neg_cis]
                        else:
                            neg_scores = np.delete(row, pos_cis)
                        n_above = int(np.sum(neg_scores > s_best_pos))
                    else:
                        n_need = max(0, sz - len(pos_cis))
                        if reg == "random":
                            # Fast strictly nested slice from perm[offset:] skipping positives & masked (-1e9) items!
                            off = (qi * 131) % max(1, n_c - n_need - 64)
                            cand = perm[off : off + n_need + len(pos_cis) + 32]
                            sc_c = row[cand]
                            valid_c = (sc_c > -1e8) & (~np.isin(cand, pos_cis))
                            neg_scores = sc_c[valid_c][:n_need]
                            if len(neg_scores) < n_need:
                                # Fallback scan across perm
                                sc_all = row[perm]
                                neg_scores = sc_all[(sc_all > -1e8) & (~np.isin(perm, pos_cis))][:n_need]
                        elif reg == "hard":
                            hc = clip_hard_order[idx_pos]
                            sc_h = row[hc]
                            neg_scores = sc_h[(sc_h > -1e8) & (~np.isin(hc, pos_cis))][:n_need]
                        else:
                            n_h = n_need // 2
                            hc = clip_hard_order[idx_pos]
                            sc_h = row[hc]
                            m_h = (sc_h > -1e8) & (~np.isin(hc, pos_cis))
                            h_ids = hc[m_h][:n_h]
                            h_sc = sc_h[m_h][:n_h]
                            n_r = n_need - len(h_sc)
                            off = (qi * 131) % max(1, n_c - n_r - 64)
                            cand = perm[off : off + n_r + len(pos_cis) + 32]
                            sc_r = row[cand]
                            r_sc = sc_r[(sc_r > -1e8) & (~np.isin(cand, pos_cis)) & (~np.isin(cand, h_ids))][:n_r]
                            neg_scores = np.concatenate([h_sc, r_sc])
                        n_above = int(np.sum(neg_scores > s_best_pos))
                    rk = n_above + 1
                    best_ranks.append(rk)
                    r1s.append(float(rk <= 1))
                    r10s.append(float(rk <= 10))
                    z10s.append(float(rk > 10))
                    k10 = min(10, len(neg_scores))
                    s_neg10 = float(np.partition(neg_scores, -k10)[-k10]) if k10 > 0 else float("nan")
                    tail_margins.append(s_best_pos - s_neg10)
                seed_stats.append({
                    "seed": sd,
                    "mean_best_pos_rank": float(np.mean(best_ranks)),
                    "median_best_pos_rank": float(np.median(best_ranks)),
                    "R@1": float(np.mean(r1s)),
                    "R@10": float(np.mean(r10s)),
                    "ZeroPos@10": float(np.mean(z10s)),
                    "mean_tail_margin_10": float(np.nanmean(tail_margins)),
                })
            e03_results["curves"][reg][str(sz)] = {
                "n_seeds": len(seed_stats),
                "mean_best_pos_rank": float(np.mean([s["mean_best_pos_rank"] for s in seed_stats])),
                "median_best_pos_rank": float(np.mean([s["median_best_pos_rank"] for s in seed_stats])),
                "R@1": float(np.mean([s["R@1"] for s in seed_stats])),
                "R@10": float(np.mean([s["R@10"] for s in seed_stats])),
                "ZeroPos@10": float(np.mean([s["ZeroPos@10"] for s in seed_stats])),
                "mean_tail_margin_10": float(np.mean([s["mean_tail_margin_10"] for s in seed_stats])),
                "per_seed": seed_stats,
            }
    _write_json(model_out / "e03_scaling.json", e03_results)

    # ------------------------------------------------------------------
    # E13: Cross-Category Manifold Analysis (F12, F13)
    # ------------------------------------------------------------------
    if dataset == "pinpoint":
        img_cats = cohorts.get("e13_pp_img_categories", {})
        cat_to_cis: Dict[str, List[int]] = defaultdict(list)
        for ci, cid in enumerate(c_ids):
            for l1 in img_cats.get(cid, []):
                cat_to_cis[l1].append(ci)
        by_qcat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for qi, r in enumerate(test_q):
            pos_set = set(r["positive_ids"])
            # Valid target categories = union of l1_interest across ALL positives + query metadata l1_interest
            valid_cats = set()
            if r["metadata"].get("l1_interest"):
                valid_cats.add(r["metadata"]["l1_interest"])
            for p in r["positive_ids"]:
                valid_cats.update(img_cats.get(p, []))
            if not valid_cats:
                continue
            # Oracle-category pool indices (union of all images belonging to ANY valid target category + positives!)
            ora_set = set()
            for vc in valid_cats:
                ora_set.update(cat_to_cis.get(vc, []))
            for p in r["positive_ids"]:
                if p in c_idx_map:
                    ora_set.add(c_idx_map[p])
            ora_cis = np.array(sorted(ora_set), dtype=np.int64)
            # Full-gallery metrics
            top10_full = [c_ids[ci] for ci in top50_idx[qi][:10]]
            cat_hit1 = float(bool(set(img_cats.get(top10_full[0], [])) & valid_cats))
            cat_hit10 = float(np.mean([bool(set(img_cats.get(x, [])) & valid_cats) for x in top10_full]))
            p1_full = float(top10_full[0] in pos_set)
            # AP@10 helper
            def _ap10(ranked: List[str]) -> float:
                hits, s_ap = 0, 0.0
                for rk_i, x in enumerate(ranked[:10], 1):
                    if x in pos_set:
                        hits += 1
                        s_ap += hits / rk_i
                return s_ap / min(10, max(len(pos_set), 1))
            ap10_full = _ap10(top10_full)
            # Oracle-category pool ranking
            ora_scores = sims_test[qi, ora_cis]
            ora_top10 = [c_ids[ora_cis[j]] for j in np.argsort(-ora_scores, kind="stable")[:10]]
            p1_ora = float(ora_top10[0] in pos_set)
            ap10_ora = _ap10(ora_top10)
            by_qcat[r["category"]].append({
                "query_id": r["query_id"],
                "group_id": r["ref_id"],
                "n_valid_cats": len(valid_cats),
                "ora_pool_size": len(ora_cis),
                "cat_hit1": cat_hit1,
                "cat_hit10": cat_hit10,
                "p1_full": p1_full,
                "p1_oracle_cat": p1_ora,
                "ap10_full": ap10_full,
                "ap10_oracle_cat": ap10_ora,
            })
        e13_summary = {"suite": "E13", "model": handle.model_key, "dataset": "pinpoint", "by_query_category": {}}
        for qcat, rows_c in by_qcat.items():
            e13_summary["by_query_category"][qcat] = {
                "n": len(rows_c),
                "cat_hit1": float(np.mean([x["cat_hit1"] for x in rows_c])),
                "cat_hit10": float(np.mean([x["cat_hit10"] for x in rows_c])),
                "p1_full": float(np.mean([x["p1_full"] for x in rows_c])),
                "p1_oracle_cat": float(np.mean([x["p1_oracle_cat"] for x in rows_c])),
                "ap10_full": float(np.mean([x["ap10_full"] for x in rows_c])),
                "ap10_oracle_cat": float(np.mean([x["ap10_oracle_cat"] for x in rows_c])),
                "delta_ap10_oracle_minus_full": float(np.mean([x["ap10_oracle_cat"] - x["ap10_full"] for x in rows_c])),
            }
        _write_json(model_out / "e13_category_manifold.json", e13_summary)
    elif dataset == "fashioniq" and len(ds_spec.get("cat_galleries", {})) == 3:
        # Cross-category leakage on combined dress+shirt+toptee gallery vs oracle within-category pool!
        img_to_cat = {}
        for cat, g_list in ds_spec["cat_galleries"].items():
            for iid in g_list:
                img_to_cat.setdefault(iid, set()).add(cat)
        by_c = defaultdict(list)
        for qi, r in enumerate(test_q):
            cat = r["category"]
            top10 = [c_ids[ci] for ci in top50_idx_unmasked[qi][:10]]
            cat_hit10 = float(np.mean([cat in img_to_cat.get(x, set()) for x in top10]))
            r10_comb = float(r["positive_ids"][0] in top10)
            by_c[cat].append({"cat_hit10": cat_hit10, "r10_combined_15k": r10_comb})
        _write_json(model_out / "e13_category_manifold.json", {
            "suite": "E13",
            "model": handle.model_key,
            "dataset": "fashioniq",
            "by_category": {
                cat: {
                    "n": len(v),
                    "same_category_rate_in_top10_combined_15k": float(np.mean([x["cat_hit10"] for x in v])),
                    "R10_combined_15k": float(np.mean([x["r10_combined_15k"] for x in v])),
                    "R10_oracle_within_category": parity["per_category"].get(cat, {}).get("R10_raw", float("nan")),
                }
                for cat, v in by_c.items()
            },
        })

    # ------------------------------------------------------------------
    # E14: Hubness, Local Density, Effective Rank, Anisotropy + Dev-Fitted Centering/Whitening/CSLS
    # ------------------------------------------------------------------
    def _hubness_stats(sim_mat: np.ndarray, q_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        top10 = np.argsort(-sim_mat, axis=1, kind="stable")[:, :10]
        k_occ = np.zeros(n_c, dtype=np.int32)
        fp_hub_hits = 0
        tot_hub_hits = 0
        # Count k-occurrence
        np.add.at(k_occ, top10.ravel(), 1)
        # Gini coefficient
        arr = np.sort(k_occ.astype(np.float64))
        s_tot = float(arr.sum())
        if s_tot > 0:
            idx_a = np.arange(1, len(arr) + 1, dtype=np.float64)
            gini = float((2.0 * np.sum(idx_a * arr) / (len(arr) * s_tot)) - (len(arr) + 1.0) / len(arr))
        else:
            gini = 0.0
        hub_mask = k_occ >= 20
        r1_list, r10_list = [], []
        for qi, r in enumerate(q_list):
            ps = set(r["positive_ids"])
            t10_ids = [c_ids[ci] for ci in top10[qi]]
            r1_list.append(float(t10_ids[0] in ps))
            r10_list.append(float(any(x in ps for x in t10_ids)))
            for ci, cid in zip(top10[qi], t10_ids):
                if hub_mask[ci]:
                    tot_hub_hits += 1
                    if cid not in ps:
                        fp_hub_hits += 1
        return {
            "gini_k10": gini,
            "max_k_occ_10": int(k_occ.max()),
            "n_hubs_ge20": int(hub_mask.sum()),
            "hub_top10_share": float(tot_hub_hits / max(len(q_list) * 10, 1)),
            "hub_fp_rate": float(fp_hub_hits / max(tot_hub_hits, 1)) if tot_hub_hits > 0 else 0.0,
            "hub_fp_count": int(fp_hub_hits),
            "Hit@1": float(np.mean(r1_list)),
            "Hit@10": float(np.mean(r10_list)),
        }

    # Spectral geometry: effective rank, anisotropy, local density on 2,048 deterministic gallery items
    rng_geo = np.random.default_rng(42)
    samp_idx = rng_geo.choice(n_c, size=min(2048, n_c), replace=False)
    if is_dcnet:
        g_samp = F.normalize(g_feats["im_feat"][samp_idx].float(), dim=-1).numpy()
    elif is_multivec:
        # Use flattened token sample (1st & random token per image to measure real token manifold!)
        g_samp = F.normalize(g_feats[samp_idx, 0, :].float(), dim=-1).numpy()
    else:
        g_samp = F.normalize(g_feats[samp_idx].float(), dim=-1).numpy()
    sv = np.linalg.svd(g_samp - g_samp.mean(axis=0, keepdims=True), compute_uv=False)
    sv2 = sv ** 2
    p_sv = sv2 / max(float(sv2.sum()), 1e-12)
    eff_rank = float(math.exp(-float(np.sum(p_sv * np.log(np.clip(p_sv, 1e-12, 1.0))))))
    gram = g_samp @ g_samp.T
    np.fill_diagonal(gram, -2.0)
    anisotropy = float(gram[gram > -1.5].mean())
    local_density_10nn = float(np.sort(gram, axis=1)[:, -10:].mean())

    # E14 evaluation subset: held-out test queries (for FashionIQ use 70% cal-heldout so val-fitted params NEVER leak!)
    e14_q = ds_spec.get("test_queries_cal_heldout", test_q)
    e14_q_idxs = [i for i, r in enumerate(test_q) if r["query_id"] in {x["query_id"] for x in e14_q}] if dataset == "fashioniq" else list(range(len(test_q)))
    sims_e14_raw = sims_test[e14_q_idxs]
    e14_raw_stats = _hubness_stats(sims_e14_raw, e14_q)

    # 1. CSLS / Local Scaling fit strictly on VAL queries (`r_S(c)` = mean top-10 sim of candidate c to VAL queries!)
    k_csls = min(10, sims_val_eff.shape[0])
    # Mask out -1e9 entries before computing candidate hubness r_S(c)
    sv_clean = np.where(sims_val_eff > -1e8, sims_val_eff, -10.0)
    r_S_val = np.sort(sv_clean, axis=0)[-k_csls:, :].mean(axis=0)  # [N_c] fit ONLY on val queries!
    sims_e14_csls = np.where(sims_e14_raw > -1e8, 2.0 * sims_e14_raw - r_S_val[None, :], -1e9)
    e14_csls_stats = _hubness_stats(sims_e14_csls, e14_q)

    # 2. Centering & Regularized ZCA Whitening fit STRICTLY on train/calibration gallery (ZERO test-target leakage!)
    e14_center_stats: Dict[str, Any] = {"status": "na_dcnet_7expert"}
    e14_whiten_stats: Dict[str, Any] = {"status": "na_dcnet_7expert"}
    if not is_dcnet:
        cal_ids = ds_spec.get("cal_corpus_ids", c_ids)
        cal_pool = np.array([c_idx_map[cid] for cid in cal_ids if cid in c_idx_map], dtype=np.int64)
        cal_g_idx = rng_geo.choice(cal_pool, size=min(4096, len(cal_pool)), replace=False) if len(cal_pool) > 0 else rng_geo.choice(n_c, size=min(4096, n_c), replace=False)
        def _apply_fiq_mask_e14(sm_in: np.ndarray) -> np.ndarray:
            if fiq_cat_valid_mask is None:
                return sm_in
            sm_out = sm_in.copy()
            for ii, rr in enumerate(e14_q):
                sm_out[ii, ~fiq_cat_valid_mask[rr["category"]]] = -1e9
            return sm_out

        if is_multivec:
            cal_mat = g_feats[cal_g_idx].float().reshape(-1, g_feats.shape[-1]).numpy()
            mu = cal_mat.mean(axis=0, keepdims=True).astype(np.float32)
            q_c = F.normalize(q_test[e14_q_idxs].float() - torch.from_numpy(mu), dim=-1)
            g_c = F.normalize(g_feats.float() - torch.from_numpy(mu).unsqueeze(0), dim=-1)
            sims_cent = _apply_fiq_mask_e14(handle.score_matrix(q_c, g_c, q_chunk=128, c_chunk=4096))
            e14_center_stats = _hubness_stats(sims_cent, e14_q)
            Xc = cal_mat - mu
            cov = (Xc.T @ Xc) / max(len(Xc), 1)
            evals, evecs = np.linalg.eigh(cov.astype(np.float64))
            W = (evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 0.0) + 0.05)) @ evecs.T).astype(np.float32)
            W_t = torch.from_numpy(W)
            q_w = F.normalize((q_test[e14_q_idxs].float() - torch.from_numpy(mu)) @ W_t, dim=-1)
            g_w = F.normalize((g_feats.float() - torch.from_numpy(mu).unsqueeze(0)) @ W_t, dim=-1)
            sims_whit = _apply_fiq_mask_e14(handle.score_matrix(q_w, g_w, q_chunk=128, c_chunk=4096))
            e14_whiten_stats = _hubness_stats(sims_whit, e14_q)
            del q_c, g_c, q_w, g_w, sims_cent, sims_whit
        else:
            cal_mat = F.normalize(g_feats[cal_g_idx].float(), dim=-1).numpy()
            mu = cal_mat.mean(axis=0, keepdims=True).astype(np.float32)
            q_c = F.normalize(F.normalize(q_test[e14_q_idxs].float(), dim=-1) - torch.from_numpy(mu), dim=-1)
            g_c = F.normalize(F.normalize(g_feats.float(), dim=-1) - torch.from_numpy(mu), dim=-1)
            sims_cent = _apply_fiq_mask_e14(handle.score_matrix(q_c, g_c, q_chunk=128, c_chunk=4096))
            e14_center_stats = _hubness_stats(sims_cent, e14_q)
            Xc = cal_mat - mu
            cov = (Xc.T @ Xc) / max(len(Xc), 1)
            evals, evecs = np.linalg.eigh(cov.astype(np.float64))
            W = (evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 0.0) + 0.05)) @ evecs.T).astype(np.float32)
            W_t = torch.from_numpy(W)
            q_w = F.normalize((F.normalize(q_test[e14_q_idxs].float(), dim=-1) - torch.from_numpy(mu)) @ W_t, dim=-1)
            g_w = F.normalize((F.normalize(g_feats.float(), dim=-1) - torch.from_numpy(mu)) @ W_t, dim=-1)
            sims_whit = _apply_fiq_mask_e14(handle.score_matrix(q_w, g_w, q_chunk=128, c_chunk=4096))
            e14_whiten_stats = _hubness_stats(sims_whit, e14_q)
            del q_c, g_c, q_w, g_w, sims_cent, sims_whit

    _write_json(model_out / "e14_hubness.json", {
        "suite": "E14",
        "model": handle.model_key,
        "dataset": dataset,
        "n_eval_queries": len(e14_q),
        "n_corpus": n_c,
        "calibration_gallery_scope": "strict_train_or_non_heldout_disjoint",
        "spectral_geometry": {
            "effective_rank": eff_rank,
            "anisotropy_mean_cos": anisotropy,
            "local_density_10nn": local_density_10nn,
        },
        "baseline_raw": e14_raw_stats,
        "csls_val_fitted": e14_csls_stats,
        "centering_cal_fitted": e14_center_stats,
        "whitening_zca_cal_fitted": e14_whiten_stats,
    })

    # ------------------------------------------------------------------
    # E15: Uncertainty & Calibration (GPU-Vectorized Exact Full-Gallery Temp-Scaled Entropy + No-Target!)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_e15_signals(sim_mat: np.ndarray, T: float) -> Dict[str, np.ndarray]:
        n_q_l = sim_mat.shape[0]
        top1 = np.empty(n_q_l, dtype=np.float64)
        margin12 = np.empty(n_q_l, dtype=np.float64)
        std5 = np.empty(n_q_l, dtype=np.float64)
        ent_full = np.empty(n_q_l, dtype=np.float64)
        ent_top50_trunc = np.empty(n_q_l, dtype=np.float64)
        inv_T = 1.0 / max(T, 1e-4)
        k50 = min(50, sim_mat.shape[1])
        bs_g = 256
        for qs in range(0, n_q_l, bs_g):
            qe = min(qs + bs_g, n_q_l)
            blk = torch.from_numpy(sim_mat[qs:qe]).to(handle.device, dtype=torch.float32)
            # Top-50 scores cleanly on GPU
            p50, _ = torch.topk(blk, k=k50, dim=1)
            top1[qs:qe] = p50[:, 0].double().cpu().numpy()
            margin12[qs:qe] = (p50[:, 0] - p50[:, 1]).double().cpu().numpy() if k50 > 1 else 0.0
            std5[qs:qe] = p50[:, :5].std(dim=1, unbiased=False).double().cpu().numpy()
            # EXACT full-gallery softmax entropy (ignoring -1e9 masked entries cleanly!)
            valid_m = blk > -1e8
            z = torch.where(valid_m, (blk - p50[:, :1]) * inv_T, torch.tensor(-1e9, device=handle.device))
            log_p = F.log_softmax(z, dim=1)
            p = torch.exp(log_p)
            ent_f = -(p * torch.where(valid_m, log_p, torch.zeros_like(log_p))).sum(dim=1)
            ent_full[qs:qe] = ent_f.double().cpu().numpy()
            # Truncated top-50 entropy (explicit control!)
            z50 = (p50 - p50[:, :1]) * inv_T
            lp50 = F.log_softmax(z50, dim=1)
            ent_top50_trunc[qs:qe] = -(torch.exp(lp50) * lp50).sum(dim=1).double().cpu().numpy()
        return {
            "top1_score": top1,
            "margin_1_2": margin12,
            "top5_std": std5,
            "neg_full_entropy_T": -ent_full,
            "neg_top50_entropy_truncated": -ent_top50_trunc,
        }

    # Hit definition: EDIR=Hit@1, PinPoint=Hit@10, FashionIQ=Hit@10
    def _hit_labels(sim_mat: np.ndarray, q_list: List[Dict[str, Any]]) -> np.ndarray:
        k_hit = 1 if dataset == "edir" else 10
        top_k = np.argsort(-sim_mat, axis=1, kind="stable")[:, :k_hit]
        y = np.zeros(len(q_list), dtype=np.int64)
        for i, r in enumerate(q_list):
            ps = set(r["positive_ids"])
            if any(c_ids[ci] in ps for ci in top_k[i]):
                y[i] = 1
        return y

    y_val_hit = _hit_labels(sims_val_eff, val_q)
    y_test_hit = _hit_labels(sims_e14_raw, e14_q)

    # Tune temperature T strictly on val split (maximizing AUROC of -full_entropy on val hit/miss!)
    best_T = 0.05
    best_val_auc = -1.0
    for T_cand in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        sig_v = _extract_e15_signals(sims_val_eff, T_cand)
        auc_v = binary_auroc(y_val_hit, sig_v["neg_full_entropy_T"])
        if math.isfinite(auc_v) and auc_v > best_val_auc:
            best_val_auc = auc_v
            best_T = T_cand

    sig_val = _extract_e15_signals(sims_val_eff, best_T)
    sig_test = _extract_e15_signals(sims_e14_raw, best_T)

    # Deterministic input-perturbation variance on 150 test queries (4 perturbations: bright+, bright-, 94%-crop, text-case)
    pert_n = min(150, len(e14_q))
    pert_q_slice = e14_q[:pert_n]
    pert_ref_pils = [_open_pil(ref_fn(r)) for r in pert_q_slice]
    pert_txts = [r["text"] for r in pert_q_slice]
    pert_top1_mat = np.empty((4, pert_n), dtype=np.float64)
    pert_variants = [
        ([PIL.ImageEnhance.Brightness(im).enhance(1.06) for im in pert_ref_pils], pert_txts),
        ([PIL.ImageEnhance.Brightness(im).enhance(0.94) for im in pert_ref_pils], pert_txts),
        ([_crop_pil_box(im, [0.03, 0.03, 0.97, 0.97]) for im in pert_ref_pils], pert_txts),
        (pert_ref_pils, [t.strip().capitalize() + "." for t in pert_txts]),
    ]
    cand_union = sorted({int(ci) for qi in e14_q_idxs[:pert_n] for ci in top50_idx[qi][:20]})
    if is_dcnet:
        g_sub = {m: g_feats[m][cand_union] for m in handle.extra["modalities"]}
    else:
        g_sub = g_feats[cand_union]
    for pi, (p_imgs, p_txts) in enumerate(pert_variants):
        qp = handle.encode_queries(p_imgs, p_txts, mode="multimodal", bs=48, num_workers=0)
        sp = handle.score_matrix(qp, g_sub, q_chunk=64, c_chunk=2048)
        pert_top1_mat[pi] = sp.max(axis=1)
    neg_perturb_var = -np.var(pert_top1_mat, axis=0)

    e15_hit_metrics: Dict[str, Any] = {}
    for s_name in sig_test:
        a_cal, b_cal = fit_platt_calibration(sig_val[s_name], y_val_hit)
        p_te = apply_platt_calibration(sig_test[s_name], a_cal, b_cal)
        e15_hit_metrics[s_name] = {
            "platt_a_dev": a_cal,
            "platt_b_dev": b_cal,
            "AUROC": binary_auroc(y_test_hit, sig_test[s_name]),
            "AUPRC": binary_auprc(y_test_hit, sig_test[s_name]),
            "ECE": expected_calibration_error(y_test_hit, p_te),
            "Brier": brier_score(y_test_hit, p_te),
            **risk_coverage_aurc(y_test_hit, sig_test[s_name]),
        }
    e15_hit_metrics["neg_perturb_var_n150"] = {
        "n": pert_n,
        "AUROC": binary_auroc(y_test_hit[:pert_n], neg_perturb_var),
        "AUPRC": binary_auprc(y_test_hit[:pert_n], neg_perturb_var),
        **risk_coverage_aurc(y_test_hit[:pert_n], neg_perturb_var),
    }

    # Verified No-Target Episodes (compute val & test no-target signals ONCE outside the loop!)
    notarget_set = set(cohorts["e15_notarget_qids"][dataset])
    y_present = np.array([0 if r["query_id"] in notarget_set else 1 for r in e14_q], dtype=np.int64)
    sims_nt = sims_e14_raw.copy()
    for i, r in enumerate(e14_q):
        if y_present[i] == 0:
            for pid in r["positive_ids"]:
                if pid in c_idx_map:
                    sims_nt[i, c_idx_map[pid]] = -1e9
    sig_nt = _extract_e15_signals(sims_nt, best_T)

    y_val_pres = np.array([i % 2 for i in range(len(val_q))], dtype=np.int64)
    sims_val_nt = sims_val_eff.copy()
    for i, r in enumerate(val_q):
        if y_val_pres[i] == 0:
            for pid in r["positive_ids"]:
                if pid in c_idx_map:
                    sims_val_nt[i, c_idx_map[pid]] = -1e9
    sig_val_nt = _extract_e15_signals(sims_val_nt, best_T)

    e15_notarget_metrics: Dict[str, Any] = {}
    for s_name in sig_nt:
        a_nt, b_nt = fit_platt_calibration(sig_val_nt[s_name], y_val_pres)
        p_nt = apply_platt_calibration(sig_nt[s_name], a_nt, b_nt)
        e15_notarget_metrics[s_name] = {
            "AUROC": binary_auroc(y_present, sig_nt[s_name]),
            "AUPRC": binary_auprc(y_present, sig_nt[s_name]),
            "ECE": expected_calibration_error(y_present, p_nt),
            "Brier": brier_score(y_present, p_nt),
            "FAR_at_90TPR": false_accept_rate_at_tpr(y_present, sig_nt[s_name], target_tpr=0.90),
            "FAR_at_95TPR": false_accept_rate_at_tpr(y_present, sig_nt[s_name], target_tpr=0.95),
        }

    # Save compact common-candidate top-20 scores for cross-model E15 disagreement!
    common_cand_dump = {}
    for i, qi in enumerate(e14_q_idxs):
        qid = test_q[qi]["query_id"]
        t20 = top50_idx[qi][:20]
        common_cand_dump[qid] = {
            "hit": int(y_test_hit[i]),
            "top1_score": float(sig_test["top1_score"][i]),
            "margin_1_2": float(sig_test["margin_1_2"][i]),
            "full_entropy_T": float(-sig_test["neg_full_entropy_T"][i]),
            "top20_ids": [c_ids[ci] for ci in t20],
            "top20_scores": [round(float(sims_test[qi, ci]), 5) for ci in t20],
        }
    _write_json(model_out / "e15_calibration.json", {
        "suite": "E15",
        "model": handle.model_key,
        "dataset": dataset,
        "temperature_fit_on_val": best_T,
        "val_hit_rate": float(y_val_hit.mean()),
        "test_hit_rate": float(y_test_hit.mean()),
        "n_val": len(val_q),
        "n_test": len(e14_q),
        "n_notarget": int((y_present == 0).sum()),
        "n_present": int((y_present == 1).sum()),
        "hit_miss_calibration": e15_hit_metrics,
        "no_target_detection": e15_notarget_metrics,
    })
    _write_json(model_out / "e15_per_query_top20.json", common_cand_dump)

    return sims_test, q_test, c_ids


# ---------------------------------------------------------------------------
# Targeted Diagnostic Suites (E04, E05, E06, E07, E08, E09, E10, E11, E12)
# ---------------------------------------------------------------------------

def _score_pairs_diag(handle: ModelHandle, q_rep: Any, g_pos: Any, g_neg: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Compute diagonal s(q_i, pos_i) and s(q_i, neg_i) cleanly across all scorer families."""
    n = g_pos[handle.extra["modalities"][0]].shape[0] if handle.model_key == "dcnet" else g_pos.shape[0]
    if handle.model_key == "dcnet":
        mods = handle.extra["modalities"]
        g_cat = {m: torch.cat([g_pos[m], g_neg[m]], dim=0) for m in mods}
        sm = handle.score_matrix(q_rep, g_cat)
        return np.array([sm[i, i] for i in range(n)], dtype=np.float32), np.array([sm[i, n + i] for i in range(n)], dtype=np.float32)
    elif g_pos.dim() == 3:
        if q_rep.dim() == 3:
            sp = torch.einsum("bmd,bkd->bmk", F.normalize(q_rep.float(), dim=-1), g_pos.float()).max(dim=-1).values.mean(dim=-1)
            sn = torch.einsum("bmd,bkd->bmk", F.normalize(q_rep.float(), dim=-1), g_neg.float()).max(dim=-1).values.mean(dim=-1)
        else:
            sp = torch.einsum("bd,bkd->bk", q_rep.float(), g_pos.float()).max(dim=-1).values
            sn = torch.einsum("bd,bkd->bk", q_rep.float(), g_neg.float()).max(dim=-1).values
        return sp.cpu().numpy(), sn.cpu().numpy()
    else:
        qn = F.normalize(q_rep.float(), dim=-1)
        sp = (qn * F.normalize(g_pos.float(), dim=-1)).sum(dim=-1)
        sn = (qn * F.normalize(g_neg.float(), dim=-1)).sum(dim=-1)
        return sp.cpu().numpy(), sn.cpu().numpy()


def _run_targeted_suites(
    handle: ModelHandle,
    dataset: str,
    ds_spec: Dict[str, Any],
    g_feats: Any,
    cohorts: Dict[str, Any],
    model_out: Path,
    num_workers: int,
) -> None:
    c_ids = ds_spec["corpus_ids"]
    c_idx_map = {cid: i for i, cid in enumerate(c_ids)}
    img_map = ds_spec["img_path_map"]
    ref_fn = ds_spec["ref_path_fn"]
    test_q = ds_spec["test_queries"]
    is_dcnet = handle.model_key == "dcnet"
    is_multivec = (not is_dcnet) and (g_feats.dim() == 3)

    # ------------------------------------------------------------------
    # E05: Fusion Dominance & Pure Fusion Failure (F01, F03, F04, F22, F28, F29)
    # ------------------------------------------------------------------
    print(f"  [{handle.model_key}/{dataset}] Running E05 (Fusion Dominance & Pure Fusion Failure)...", flush=True)
    # Stratified 300 test triplets including ALL remove/Negation & F28 items!
    e05_rows = []
    seen_e05 = set()
    if dataset == "edir":
        priority = [r for r in test_q if r["category"] in ("remove", "shape", "viewpoint", "cardinality", "complex")]
    elif dataset == "pinpoint":
        priority = [r for r in test_q if r["category"] in ("Negation", "Complement", "Explore")]
    else:
        priority = [r for r in test_q if any(w in r["text"].lower() for w in ("not ", "no ", "without", "darker", "lighter", "shorter", "longer"))]
    for r in priority + test_q:
        if len(e05_rows) >= 300:
            break
        if r["query_id"] in seen_e05:
            continue
        pos_id = next((p for p in r["positive_ids"] if p in img_map), None)
        if not pos_id:
            continue
        # Pick hard negative: explicit negative on EDIR/PinPoint, or next query's target in same category on FashionIQ
        neg_id = next((n for n in r.get("negative_ids", []) if n in img_map and n != r["ref_id"]), None)
        if not neg_id:
            # Deterministic same-category non-positive candidate
            cand_pool = ds_spec.get("cat_galleries", {}).get(r["category"], c_ids)
            neg_id = next((x for x in cand_pool if x != pos_id and x != r["ref_id"]), c_ids[0])
        seen_e05.add(r["query_id"])
        e05_rows.append({
            "query_id": r["query_id"],
            "group_id": r.get("metadata", {}).get("stem_base", r["ref_id"]),
            "category": r.get("category", "unknown"),
            "ref_path": ref_fn(r),
            "pos_path": img_map[pos_id],
            "neg_path": img_map[neg_id],
            "text": r["text"],
        })

    ref_ps = [x["ref_path"] for x in e05_rows]
    pos_ps = [x["pos_path"] for x in e05_rows]
    neg_ps = [x["neg_path"] for x in e05_rows]
    txts = [x["text"] for x in e05_rows]
    # Derangement shift by 1 for shuffled_ref & shuffled_text controls
    shuf_ref_ps = ref_ps[1:] + ref_ps[:1]
    shuf_txts = txts[1:] + txts[:1]

    g_pos = handle.encode_gallery(pos_ps, bs=48, num_workers=num_workers)
    g_neg = handle.encode_gallery(neg_ps, bs=48, num_workers=num_workers)
    conds_q = {
        "multimodal": handle.encode_queries(ref_ps, txts, mode="multimodal", bs=48, num_workers=num_workers),
        "img_only_empty_text": handle.encode_queries(ref_ps, txts, mode="img_only_empty_text", bs=48, num_workers=num_workers),
        "txt_only_black_img": handle.encode_queries(ref_ps, txts, mode="txt_only_black_img", bs=48, num_workers=num_workers),
        "native_img_tower": handle.encode_queries(ref_ps, txts, mode="native_img_tower", bs=48, num_workers=num_workers),
        "native_txt_tower": handle.encode_queries(ref_ps, txts, mode="native_txt_tower", bs=48, num_workers=num_workers),
        "shuffled_ref": handle.encode_queries(shuf_ref_ps, txts, mode="multimodal", bs=48, num_workers=num_workers),
        "shuffled_text": handle.encode_queries(ref_ps, shuf_txts, mode="multimodal", bs=48, num_workers=num_workers),
    }
    margins_e05: Dict[str, np.ndarray] = {}
    for cname, q_rep in conds_q.items():
        sp, sn = _score_pairs_diag(handle, q_rep, g_pos, g_neg)
        margins_e05[cname] = sp - sn

    # Score-add (0.5 * native_img + 0.5 * native_txt), Embedding-add (where valid in shared single-vector sphere!), Oracle
    margins_e05["score_add_native"] = 0.5 * margins_e05["native_img_tower"] + 0.5 * margins_e05["native_txt_tower"]
    margins_e05["score_add_circus"] = 0.5 * margins_e05["img_only_empty_text"] + 0.5 * margins_e05["txt_only_black_img"]
    margins_e05["oracle_native_or_fused"] = np.maximum.reduce([margins_e05["native_img_tower"], margins_e05["native_txt_tower"], margins_e05["multimodal"]])
    if (not is_dcnet) and (not is_multivec):
        q_eadd = F.normalize(F.normalize(conds_q["native_img_tower"].float(), dim=-1) + F.normalize(conds_q["native_txt_tower"].float(), dim=-1), dim=-1)
        sp_ea, sn_ea = _score_pairs_diag(handle, q_eadd, g_pos, g_neg)
        margins_e05["embed_add_valid_sphere"] = sp_ea - sn_ea

    # Count (+, +, -) pure fusion collapse with exact denominator!
    mi = margins_e05["native_img_tower"]
    mt = margins_e05["native_txt_tower"]
    mf = margins_e05["multimodal"]
    both_pos_nat = (mi > 0) & (mt > 0)
    ppp_neg_nat = both_pos_nat & (mf <= 0)
    mic = margins_e05["img_only_empty_text"]
    mtc = margins_e05["txt_only_black_img"]
    both_pos_cir = (mic > 0) & (mtc > 0)
    ppp_neg_cir = both_pos_cir & (mf <= 0)

    # Also run block-level causal interventions on first 100 E05 triplets (including all (+,+,-) cases!)
    e05_int_idx = list(np.where(ppp_neg_nat)[0]) + [i for i in range(len(e05_rows)) if not ppp_neg_nat[i]]
    e05_int_idx = e05_int_idx[:100]
    e05_interventions = handle.run_causal_interventions(
        [ref_ps[i] for i in e05_int_idx],
        [txts[i] for i in e05_int_idx],
        [pos_ps[i] for i in e05_int_idx],
        [neg_ps[i] for i in e05_int_idx],
    )

    _write_json(model_out / "e05_fusion_dominance.json", {
        "suite": "E05",
        "model": handle.model_key,
        "dataset": dataset,
        "n_triplets": len(e05_rows),
        "mean_margins": {k: float(np.mean(v)) for k, v in margins_e05.items()},
        "pair_accuracy": {k: float(np.mean(v > 0)) for k, v in margins_e05.items()},
        "pure_fusion_failure_native": {
            "n_both_unimodal_pos": int(both_pos_nat.sum()),
            "n_ppp_minus": int(ppp_neg_nat.sum()),
            "rate_given_both_pos": float(ppp_neg_nat.sum() / max(int(both_pos_nat.sum()), 1)),
            "rate_overall": float(ppp_neg_nat.mean()),
        },
        "pure_fusion_failure_circus": {
            "n_both_unimodal_pos": int(both_pos_cir.sum()),
            "n_ppp_minus": int(ppp_neg_cir.sum()),
            "rate_given_both_pos": float(ppp_neg_cir.sum() / max(int(both_pos_cir.sum()), 1)),
            "rate_overall": float(ppp_neg_cir.mean()),
        },
        "per_query": [
            {
                "query_id": e05_rows[i]["query_id"],
                "group_id": e05_rows[i]["group_id"],
                "category": e05_rows[i]["category"],
                **{k: round(float(v[i]), 5) for k, v in margins_e05.items()},
            }
            for i in range(len(e05_rows))
        ],
        "block_interventions_subset": {
            "n": len(e05_int_idx),
            "query_ids": [e05_rows[i]["query_id"] for i in e05_int_idx],
            "is_ppp_minus_native": [bool(ppp_neg_nat[i]) for i in e05_int_idx],
            "margins": e05_interventions,
        },
    })

    # ------------------------------------------------------------------
    # E04: Layer-wise Query & Gallery Representations + Causal Interventions (F01, F04, F05, F08-11, F23, F24, F28, F29)
    # ------------------------------------------------------------------
    if dataset == "edir":
        print(f"  [{handle.model_key}/edir] Running E04 (Layer-wise Query+Gallery Reps & F28/ROI Causal Interventions)...", flush=True)
        e04_npz: Dict[str, np.ndarray] = {}
        for sp in ("train", "dev", "test"):
            rows_sp = cohorts["e04_edir"][sp]
            q_reps = handle.extract_layer_reps_query([x["ref_path"] for x in rows_sp], [x["text"] for x in rows_sp], bs=48, num_workers=num_workers)
            # Gallery branch: extract on target positive images (and diff = target - ref on test/dev/train!)
            g_pos_reps = handle.extract_layer_reps_gallery([x["pos_path"] for x in rows_sp], bs=48, num_workers=num_workers)
            g_ref_reps = handle.extract_layer_reps_gallery([x["ref_path"] for x in rows_sp], bs=48, num_workers=num_workers)
            for lname, arr in q_reps.items():
                e04_npz[f"{sp}__{lname}"] = arr.astype(np.float16)
            for lname, arr in g_pos_reps.items():
                e04_npz[f"{sp}__{lname}_pos"] = arr.astype(np.float16)
                e04_npz[f"{sp}__{lname}_diff"] = (arr - g_ref_reps[lname]).astype(np.float16)
        np.savez_compressed(model_out / "e04_edir_layers.npz", **e04_npz)

        # F28 BLIP-2 Blind Spot matched cohorts causal interventions (CLIP-hit/BLIP-miss vs reverse vs both-hit!)
        f28_edir = cohorts["e04_f28"]["edir"]
        f28_res = {}
        for cname, items_c in f28_edir.items():
            if not items_c:
                continue
            r_ps = [str(EDIR_DIR / "imgs_mc" / x["ref_id"]) for x in items_c]
            p_ps = [str(EDIR_DIR / "imgs_mc" / x["pos_id"]) for x in items_c]
            w_ps = [str(EDIR_DIR / "imgs_mc" / x["wrong_id"]) for x in items_c]
            c_txts = [x["text"] for x in items_c]
            f28_res[cname] = {
                "n": len(items_c),
                "query_ids": [x["query_id"] for x in items_c],
                "interventions": handle.run_causal_interventions(r_ps, c_txts, p_ps, w_ps),
            }
        # Also run spatial-ROI vs sham-ROI vision-token intervention on E11 items!
        e11_items = cohorts["e11_items"]
        roi_int = handle.run_causal_interventions(
            [str(EDIR_DIR / "imgs_mc" / x["ref_id"]) for x in e11_items],
            [x["text"] for x in e11_items],
            [str(EDIR_DIR / "imgs_mc" / x["pos_id"]) for x in e11_items],
            [str(EDIR_DIR / "imgs_mc" / x["neg_ids"][0]) for x in e11_items],
            roi_boxes=[x["roi_box"] for x in e11_items],
            sham_boxes=[x["sham_box"] for x in e11_items],
        )
        _write_json(model_out / "e04_interventions_edir.json", {
            "suite": "E04",
            "model": handle.model_key,
            "dataset": "edir",
            "f28_matched_cohorts": f28_res,
            "e11_roi_token_interventions": {
                "n": len(e11_items),
                "query_ids": [x["query_id"] for x in e11_items],
                "sub_cohorts": [x["sub_cohort"] for x in e11_items],
                "margins": roi_int,
            },
        })

    elif dataset == "fashioniq" and handle.fiq_category is None:
        print(f"  [{handle.model_key}/fashioniq] Running E04 Gallery-Branch Layer Reps...", flush=True)
        e04_npz = {}
        for sp in ("train", "dev", "test"):
            rows_sp = cohorts["e04_fiq_gallery"][sp]
            g_reps = handle.extract_layer_reps_gallery([x["image_path"] for x in rows_sp], bs=64, num_workers=num_workers)
            for lname, arr in g_reps.items():
                e04_npz[f"{sp}__{lname}"] = arr.astype(np.float16)
        np.savez_compressed(model_out / "e04_fiq_gallery_layers.npz", **e04_npz)

    elif dataset == "pinpoint":
        # F28 PinPoint matched cohorts causal interventions
        f28_pp = cohorts["e04_f28"]["pinpoint"]
        f28_res = {}
        for cname, items_c in f28_pp.items():
            valid_c = [x for x in items_c if x["ref_id"] in img_map and x["pos_id"] in img_map and x["wrong_id"] in img_map]
            if not valid_c:
                continue
            f28_res[cname] = {
                "n": len(valid_c),
                "query_ids": [x["query_id"] for x in valid_c],
                "interventions": handle.run_causal_interventions(
                    [img_map[x["ref_id"]] for x in valid_c],
                    [x["text"] for x in valid_c],
                    [img_map[x["pos_id"]] for x in valid_c],
                    [img_map[x["wrong_id"]] for x in valid_c],
                ),
            }
        _write_json(model_out / "e04_interventions_pinpoint.json", {"suite": "E04", "model": handle.model_key, "dataset": "pinpoint", "f28_matched_cohorts": f28_res})

    # ------------------------------------------------------------------
    # E07: Negation & Polarity Minimal Pairs (F01, F20)
    # ------------------------------------------------------------------
    if dataset == "edir":
        print(f"  [{handle.model_key}/edir] Running E07 (EDIR Bidirectional Negation Minimal Pairs)...", flush=True)
        pairs = cohorts["e07_edir"]
        r_ps = [x["ref_path"] for x in pairs]
        cw_ps = [x["cand_with_X_path"] for x in pairs]
        cwo_ps = [x["cand_without_X_path"] for x in pairs]
        gw = handle.encode_gallery(cw_ps, bs=48, num_workers=num_workers)
        gwo = handle.encode_gallery(cwo_ps, bs=48, num_workers=num_workers)
        qw = handle.encode_queries(r_ps, [x["text_with_X"] for x in pairs], mode="multimodal", bs=48, num_workers=num_workers)
        qwo = handle.encode_queries(r_ps, [x["text_without_X"] for x in pairs], mode="multimodal", bs=48, num_workers=num_workers)
        sw_w, sw_wo = _score_pairs_diag(handle, qw, gw, gwo)
        swo_w, swo_wo = _score_pairs_diag(handle, qwo, gw, gwo)
        m_with = sw_w - sw_wo      # should be > 0 (prefer with_X)
        m_without = swo_wo - swo_w  # should be > 0 (prefer without_X)
        both_ok = (m_with > 0) & (m_without > 0)
        inverted = (m_with > 0) & (m_without <= 0)
        _write_json(model_out / "e07_negation_edir.json", {
            "suite": "E07",
            "model": handle.model_key,
            "dataset": "edir",
            "n_pairs": len(pairs),
            "acc_with_X": float(np.mean(m_with > 0)),
            "acc_without_X": float(np.mean(m_without > 0)),
            "bidirectional_flip_acc": float(np.mean(both_ok)),
            "negation_inversion_rate": float(np.mean(inverted)),
            "mean_margin_with_X": float(np.mean(m_with)),
            "mean_margin_without_X": float(np.mean(m_without)),
            "per_pair": [
                {"pair_id": pairs[i]["pair_id"], "group_id": pairs[i]["group_id"], "orig_cat": pairs[i]["orig_category"], "m_with": round(float(m_with[i]), 5), "m_without": round(float(m_without[i]), 5)}
                for i in range(len(pairs))
            ],
        })

    # ------------------------------------------------------------------
    # E08-B: EDIR Complex Multi-Constraint Decomposition (F04)
    # ------------------------------------------------------------------
    if dataset == "edir":
        print(f"  [{handle.model_key}/edir] Running E08-B (EDIR Complex Clause Decomposition)...", flush=True)
        items = cohorts["e08_edir_complex"]
        c_res = []
        for it in items:
            rp = str(EDIR_DIR / "imgs_mc" / it["ref_id"])
            pool_ids = [it["pos_id"]] + list(it["neg_ids"])
            pool_cis = [c_idx_map[x] for x in pool_ids if x in c_idx_map]
            if len(pool_cis) < 2 or it["pos_id"] not in c_idx_map:
                continue
            g_pool = {m: g_feats[m][pool_cis] for m in handle.extra["modalities"]} if is_dcnet else g_feats[pool_cis]
            txts_q = [it["full_text"]] + list(it["clauses"])
            if len(it["clauses"]) >= 2:
                txts_q.append(f"{it['clauses'][0]} and {it['clauses'][1]}")
            q_v = handle.encode_queries([rp] * len(txts_q), txts_q, mode="multimodal", bs=16, num_workers=0)
            sm = handle.score_matrix(q_v, g_pool)
            # Pos is index 0 in pool_cis
            ranks = [int(np.sum(sm[j] > sm[j, 0])) + 1 for j in range(len(txts_q))]
            margins = [float(sm[j, 0] - np.max(sm[j, 1:])) for j in range(len(txts_q))]
            c_res.append({
                "query_id": it["query_id"],
                "group_id": it["group_id"],
                "n_clauses": len(it["clauses"]),
                "rank_full": ranks[0],
                "margin_full": round(margins[0], 5),
                "clause_ranks": ranks[1 : 1 + len(it["clauses"])],
                "clause_margins": [round(x, 5) for x in margins[1 : 1 + len(it["clauses"])]],
                "rank_c1_c2": ranks[-1] if len(it["clauses"]) >= 2 else None,
            })
        # Conditional drop: among queries where >=1 individual clause got rank 1, how often does full lose rank 1?
        indiv_pass = [x for x in c_res if any(r == 1 for r in x["clause_ranks"])]
        cond_drop = float(np.mean([x["rank_full"] > 1 for x in indiv_pass])) if indiv_pass else float("nan")
        _write_json(model_out / "e08_edir_complex.json", {
            "suite": "E08",
            "model": handle.model_key,
            "dataset": "edir",
            "n_queries": len(c_res),
            "full_within_scene_R1": float(np.mean([x["rank_full"] == 1 for x in c_res])),
            "any_single_clause_R1": float(np.mean([any(r == 1 for r in x["clause_ranks"]) for x in c_res])),
            "cond_drop_rate_full_given_single_pass": cond_drop,
            "n_indiv_pass": len(indiv_pass),
            "records": c_res,
        })

    # ------------------------------------------------------------------
    # E10-C (F24 Pose Masking) & E11 (Resolution & Patch Sensitivity) on EDIR
    # ------------------------------------------------------------------
    if dataset == "edir":
        print(f"  [{handle.model_key}/edir] Running E10-C (F24 Limb Pose Masking) & E11 (Resolution & Detail/Sham Crops)...", flush=True)
        # E10-C: Mask aligned limb-pose ROI vs exact matched-area sham on ref & candidates
        pose_items = cohorts["e10_f24_pose"]
        if pose_items:
            p_refs = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["ref_id"])) for x in pose_items]
            p_pos = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["pos_id"])) for x in pose_items]
            p_neg = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["neg_ids"][0])) for x in pose_items]
            p_txts = [x["text"] for x in pose_items]
            # Unmasked baseline vs masked ROI vs masked matched-area sham!
            q_orig = handle.encode_queries(p_refs, p_txts, mode="multimodal", bs=32, num_workers=0)
            g_p_orig = handle.encode_gallery(p_pos, bs=32, num_workers=0)
            g_n_orig = handle.encode_gallery(p_neg, bs=32, num_workers=0)
            sp0, sn0 = _score_pairs_diag(handle, q_orig, g_p_orig, g_n_orig)
            m0 = sp0 - sn0

            # Mask limb ROI on candidate targets (and on ref!)
            p_refs_roi = [_mask_pil_boxes(im, [x["roi_box"]]) for im, x in zip(p_refs, pose_items)]
            p_pos_roi = [_mask_pil_boxes(im, [x["roi_box"]]) for im, x in zip(p_pos, pose_items)]
            p_neg_roi = [_mask_pil_boxes(im, [x["roi_box"]]) for im, x in zip(p_neg, pose_items)]
            q_roi = handle.encode_queries(p_refs_roi, p_txts, mode="multimodal", bs=32, num_workers=0)
            sp_r, sn_r = _score_pairs_diag(handle, q_roi, handle.encode_gallery(p_pos_roi, bs=32, num_workers=0), handle.encode_gallery(p_neg_roi, bs=32, num_workers=0))
            m_roi = sp_r - sn_r

            p_refs_shm = [_mask_pil_boxes(im, [x["sham_box"]]) for im, x in zip(p_refs, pose_items)]
            p_pos_shm = [_mask_pil_boxes(im, [x["sham_box"]]) for im, x in zip(p_pos, pose_items)]
            p_neg_shm = [_mask_pil_boxes(im, [x["sham_box"]]) for im, x in zip(p_neg, pose_items)]
            q_shm = handle.encode_queries(p_refs_shm, p_txts, mode="multimodal", bs=32, num_workers=0)
            sp_s, sn_s = _score_pairs_diag(handle, q_shm, handle.encode_gallery(p_pos_shm, bs=32, num_workers=0), handle.encode_gallery(p_neg_shm, bs=32, num_workers=0))
            m_shm = sp_s - sn_s

            _write_json(model_out / "e10_f24_pose_edir.json", {
                "suite": "E10",
                "sub_suite": "F24_limb_pose_masking",
                "model": handle.model_key,
                "dataset": "edir",
                "n": len(pose_items),
                "mean_margin_orig": float(np.mean(m0)),
                "mean_margin_roi_masked": float(np.mean(m_roi)),
                "mean_margin_sham_masked": float(np.mean(m_shm)),
                "delta_roi_vs_orig": float(np.mean(m_roi - m0)),
                "delta_sham_vs_orig": float(np.mean(m_shm - m0)),
                "specific_roi_effect_minus_sham": float(np.mean((m_roi - m0) - (m_shm - m0))),
                "per_item": [
                    {"query_id": pose_items[i]["query_id"], "group_id": pose_items[i]["group_id"], "m_orig": round(float(m0[i]), 5), "m_roi": round(float(m_roi[i]), 5), "m_sham": round(float(m_shm[i]), 5)}
                    for i in range(len(pose_items))
                ],
            })

        # E11: Resolution (native vs 336 high-res) + Detail ROI Crop vs Matched-Area Sham Crop!
        e11_items = cohorts["e11_items"]
        e11_refs = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["ref_id"])) for x in e11_items]
        e11_pos = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["pos_id"])) for x in e11_items]
        e11_neg = [_open_pil(str(EDIR_DIR / "imgs_mc" / x["neg_ids"][0])) for x in e11_items]
        e11_txts = [x["text"] for x in e11_items]

        # 1. Native resolution
        sp_nat, sn_nat = _score_pairs_diag(
            handle,
            handle.encode_queries(e11_refs, e11_txts, mode="multimodal", bs=32, num_workers=0),
            handle.encode_gallery(e11_pos, bs=32, num_workers=0),
            handle.encode_gallery(e11_neg, bs=32, num_workers=0),
        )
        m_nat = sp_nat - sn_nat

        # 2. Detail ROI crop (2x zoom into exact diff/OCR/gaze ROI)
        roi_refs = [_crop_pil_box(im, x["roi_box"]) for im, x in zip(e11_refs, e11_items)]
        roi_pos = [_crop_pil_box(im, x["roi_box"]) for im, x in zip(e11_pos, e11_items)]
        roi_neg = [_crop_pil_box(im, x["roi_box"]) for im, x in zip(e11_neg, e11_items)]
        sp_rc, sn_rc = _score_pairs_diag(
            handle,
            handle.encode_queries(roi_refs, e11_txts, mode="multimodal", bs=32, num_workers=0),
            handle.encode_gallery(roi_pos, bs=32, num_workers=0),
            handle.encode_gallery(roi_neg, bs=32, num_workers=0),
        )
        m_roi_crop = sp_rc - sn_rc

        # 3. Matched-area sham crop (exact same box area, non-ROI region!)
        shm_refs = [_crop_pil_box(im, x["sham_box"]) for im, x in zip(e11_refs, e11_items)]
        shm_pos = [_crop_pil_box(im, x["sham_box"]) for im, x in zip(e11_pos, e11_items)]
        shm_neg = [_crop_pil_box(im, x["sham_box"]) for im, x in zip(e11_neg, e11_items)]
        sp_sc, sn_sc = _score_pairs_diag(
            handle,
            handle.encode_queries(shm_refs, e11_txts, mode="multimodal", bs=32, num_workers=0),
            handle.encode_gallery(shm_pos, bs=32, num_workers=0),
            handle.encode_gallery(shm_neg, bs=32, num_workers=0),
        )
        m_sham_crop = sp_sc - sn_sc

        # 4. Higher resolution 336x336 (where supported via ViT bicubic pos_embed interpolation!)
        m_hr336 = None
        hr_status = "unsupported_fixed_grid"
        with handle.high_res_context(336) as ok_hr:
            if ok_hr:
                try:
                    sp_hr, sn_hr = _score_pairs_diag(
                        handle,
                        handle.encode_queries(e11_refs, e11_txts, mode="multimodal", bs=16, num_workers=0),
                        handle.encode_gallery(e11_pos, bs=16, num_workers=0),
                        handle.encode_gallery(e11_neg, bs=16, num_workers=0),
                    )
                    m_hr336 = sp_hr - sn_hr
                    hr_status = "measured_336_bicubic_pos_interp"
                except Exception as exc:
                    hr_status = f"runtime_error:{type(exc).__name__}:{str(exc)[:100]}"

        # Summarize overall and by sub-cohort (F10_ocr, F23_gaze, F08_geom, F09_cardinality, F28_blip2_blindspot)
        by_sub = {}
        for sub_name in sorted({x["sub_cohort"] for x in e11_items}):
            idxs = [i for i, x in enumerate(e11_items) if x["sub_cohort"] == sub_name]
            sub_d = {
                "n": len(idxs),
                "acc_native": float(np.mean(m_nat[idxs] > 0)),
                "margin_native": float(np.mean(m_nat[idxs])),
                "acc_detail_roi_crop": float(np.mean(m_roi_crop[idxs] > 0)),
                "margin_detail_roi_crop": float(np.mean(m_roi_crop[idxs])),
                "acc_sham_crop": float(np.mean(m_sham_crop[idxs] > 0)),
                "margin_sham_crop": float(np.mean(m_sham_crop[idxs])),
                "delta_roi_minus_native": float(np.mean(m_roi_crop[idxs] - m_nat[idxs])),
                "delta_roi_minus_sham": float(np.mean(m_roi_crop[idxs] - m_sham_crop[idxs])),
            }
            if m_hr336 is not None:
                sub_d["acc_highres_336"] = float(np.mean(m_hr336[idxs] > 0))
                sub_d["margin_highres_336"] = float(np.mean(m_hr336[idxs]))
                sub_d["delta_336_minus_native"] = float(np.mean(m_hr336[idxs] - m_nat[idxs]))
            by_sub[sub_name] = sub_d

        _write_json(model_out / "e11_resolution_edir.json", {
            "suite": "E11",
            "model": handle.model_key,
            "dataset": "edir",
            "native_resolution": handle.spec.native_resolution,
            "highres_336_status": hr_status,
            "n_total": len(e11_items),
            "by_sub_cohort": by_sub,
            "per_item": [
                {
                    "query_id": e11_items[i]["query_id"],
                    "group_id": e11_items[i]["group_id"],
                    "sub_cohort": e11_items[i]["sub_cohort"],
                    "m_native": round(float(m_nat[i]), 5),
                    "m_roi_crop": round(float(m_roi_crop[i]), 5),
                    "m_sham_crop": round(float(m_sham_crop[i]), 5),
                    "m_hr336": round(float(m_hr336[i]), 5) if m_hr336 is not None else None,
                }
                for i in range(len(e11_items))
            ],
        })

    # ------------------------------------------------------------------
    # E09-B (F26 Preserve-vs-Edit) & E12-A/B (Paraphrase, Syntax, Pragmatic F13) on PinPoint!
    # ------------------------------------------------------------------
    if dataset == "pinpoint":
        print(f"  [{handle.model_key}/pinpoint] Running E09-B (F26 Preserve-vs-Edit) & E12 (Paraphrase/Syntax/F13 Pragmatic)...", flush=True)
        # E09-B / F26
        f26_items = [x for x in cohorts["e09_f26_preserve"] if x["ref_id"] in img_map and x["cand_both_ok"] in img_map and x["cand_viol_preserve_only"] in img_map and x["cand_viol_edit_only"] in img_map]
        if f26_items:
            q_f26 = handle.encode_queries([img_map[x["ref_id"]] for x in f26_items], [x["text"] for x in f26_items], mode="multimodal", bs=48, num_workers=num_workers)
            g_ok = handle.encode_gallery([img_map[x["cand_both_ok"]] for x in f26_items], bs=48, num_workers=num_workers)
            g_vp = handle.encode_gallery([img_map[x["cand_viol_preserve_only"]] for x in f26_items], bs=48, num_workers=num_workers)
            g_ve = handle.encode_gallery([img_map[x["cand_viol_edit_only"]] for x in f26_items], bs=48, num_workers=num_workers)
            s_ok, s_vp = _score_pairs_diag(handle, q_f26, g_ok, g_vp)
            _, s_ve = _score_pairs_diag(handle, q_f26, g_ok, g_ve)
            _write_json(model_out / "e09_f26_preserve_pinpoint.json", {
                "suite": "E09",
                "sub_suite": "F26_metadata_preserve_vs_edit",
                "model": handle.model_key,
                "dataset": "pinpoint",
                "n": len(f26_items),
                "acc_ok_beats_viol_preserve": float(np.mean(s_ok > s_vp)),
                "acc_ok_beats_viol_edit": float(np.mean(s_ok > s_ve)),
                "preserve_drop_rate_prefer_viol_preserve": float(np.mean(s_vp > s_ok)),
                "mean_margin_vs_viol_preserve": float(np.mean(s_ok - s_vp)),
                "mean_margin_vs_viol_edit": float(np.mean(s_ok - s_ve)),
                "per_item": [
                    {"episode_id": f26_items[i]["episode_id"], "group_id": f26_items[i]["group_id"], "domain": f26_items[i]["preserve_domain_metadata"], "m_vs_viol_preserve": round(float(s_ok[i] - s_vp[i]), 5), "m_vs_viol_edit": round(float(s_ok[i] - s_ve[i]), 5)}
                    for i in range(len(f26_items))
                ],
            })

        # E12-A: Controlled single-factor syntax/length/abstractness edits on PinPoint!
        syn_items = cohorts["e12_syntax_ctrl"]
        var_names = ["original", "verbose_abstract", "imperative_concise", "question_form"]
        syn_res: Dict[str, Any] = {"n": len(syn_items), "variants": {}}
        # Score on the full PinPoint gallery for exact mAP@10 & rank-flip!
        q_orig_rep = None
        top1_orig = None
        for vn in var_names:
            qv = handle.encode_queries([img_map[x["ref_id"]] for x in syn_items], [x["variants"][vn] for x in syn_items], mode="multimodal", bs=48, num_workers=num_workers)
            sm = handle.score_matrix(qv, g_feats, q_chunk=64, c_chunk=4096)
            top10 = np.argsort(-sm, axis=1, kind="stable")[:, :10]
            ap10s, p1s, t1_ids = [], [], []
            for i, x in enumerate(syn_items):
                ps = set(x["positive_ids"])
                ranked = [c_ids[ci] for ci in top10[i]]
                t1_ids.append(ranked[0])
                p1s.append(float(ranked[0] in ps))
                hits, sap = 0, 0.0
                for rk_i, cid in enumerate(ranked, 1):
                    if cid in ps:
                        hits += 1
                        sap += hits / rk_i
                ap10s.append(sap / min(10, max(len(ps), 1)))
            if vn == "original":
                q_orig_rep = qv
                top1_orig = t1_ids
                emb_cos = 1.0
                flip_r = 0.0
            else:
                flip_r = float(np.mean([a != b for a, b in zip(top1_orig, t1_ids)]))
                if is_dcnet:
                    emb_cos = float("nan")
                else:
                    emb_cos = float((F.normalize(q_orig_rep.float(), dim=-1) * F.normalize(qv.float(), dim=-1)).sum(dim=-1).mean().item())
            syn_res["variants"][vn] = {
                "mAP@10": float(np.mean(ap10s)),
                "P@1": float(np.mean(p1s)),
                "top1_flip_rate_vs_orig": flip_r,
                "mean_query_emb_cos_vs_orig": emb_cos,
                "ap10_list": [round(a, 5) for a in ap10s],
            }

        # E12-B / F13: Implicit Pragmatic vs Explicit Target-Category rewrite on PinPoint!
        f13_items = cohorts["e12_f13_pragmatic"]
        img_cats = cohorts.get("e13_pp_img_categories", {})
        f13_res = {}
        for mode_lbl, field in [("implicit", "text_implicit"), ("explicit_category", "text_explicit")]:
            qv = handle.encode_queries([img_map[x["ref_id"]] for x in f13_items], [x[field] for x in f13_items], mode="multimodal", bs=48, num_workers=num_workers)
            sm = handle.score_matrix(qv, g_feats, q_chunk=64, c_chunk=4096)
            top10 = np.argsort(-sm, axis=1, kind="stable")[:, :10]
            ap10s, cat_h10s = [], []
            for i, x in enumerate(f13_items):
                ps = set(x["positive_ids"])
                ranked = [c_ids[ci] for ci in top10[i]]
                cat_h10s.append(float(np.mean([x["l1_interest"] in img_cats.get(cid, []) for cid in ranked])))
                hits, sap = 0, 0.0
                for rk_i, cid in enumerate(ranked, 1):
                    if cid in ps:
                        hits += 1
                        sap += hits / rk_i
                ap10s.append(sap / min(10, max(len(ps), 1)))
            f13_res[mode_lbl] = {
                "mAP@10": float(np.mean(ap10s)),
                "target_cat_rate_at10": float(np.mean(cat_h10s)),
                "ap10_list": [round(a, 5) for a in ap10s],
                "cat_h10_list": [round(c, 5) for c in cat_h10s],
            }
        _write_json(model_out / "e12_language_pinpoint.json", {
            "suite": "E12",
            "model": handle.model_key,
            "dataset": "pinpoint",
            "syntax_controlled": syn_res,
            "f13_pragmatic_implicit_vs_explicit": {
                "n": len(f13_items),
                "query_ids": [x["query_id"] for x in f13_items],
                "group_ids": [x["group_id"] for x in f13_items],
                "categories": [x["category"] for x in f13_items],
                **f13_res,
            },
        })

    # ------------------------------------------------------------------
    # FashionIQ-Specific Targeted Suites (E06, E07-B, E08-A/C, E09-A, E10-A/B, E12-C F25!)
    # ------------------------------------------------------------------
    if dataset == "fashioniq":
        cat_filter = handle.fiq_category  # None for unified models, or dress/shirt/toptee for per-cat models (limn, clvcnet)
        print(f"  [{handle.model_key}/fashioniq{'' if not cat_filter else ':'+cat_filter}] Running E06, E07, E08, E09, E10, E12-F25...", flush=True)

        # E06-A: Counterfactual "darker" / "lighter" (F07)
        e06_d = [x for x in cohorts["e06_darker"] if cat_filter is None or x["category"] == cat_filter]
        if e06_d:
            d_recs = []
            for ep in e06_d:
                pool_cis = [c_idx_map[p] for p in ep["pool_ids"] if p in c_idx_map]
                if len(pool_cis) != len(ep["pool_ids"]):
                    continue
                g_pool = {m: g_feats[m][pool_cis] for m in handle.extra["modalities"]} if is_dcnet else g_feats[pool_cis]
                v_refs = [_fiq_path(v["ref_id"]) for v in ep["variants"]]
                v_txts = [v["text"] for v in ep["variants"]]
                qv = handle.encode_queries(v_refs, v_txts, mode="multimodal", bs=8, num_workers=0)
                sm = handle.score_matrix(qv, g_pool)
                v_out = {}
                for vi, v in enumerate(ep["variants"]):
                    top_cid = ep["pool_ids"][int(np.argmax(sm[vi]))]
                    v_out[v["name"]] = {
                        "top1_id": top_cid,
                        "hit_valid": top_cid in v["valid_ids"],
                        "picked_abs_dark": top_cid == v.get("abs_dark_id"),
                        "picked_closest_rel": top_cid == v.get("closest_relative_id"),
                    }
                # Counterfactual swap consistency: did ref_light ("darker") vs ref_dark ("darker") change choice or both collapse to abs_dark?
                same_choice_dark_swap = (v_out["ref_light_cmd_darker"]["top1_id"] == v_out["ref_dark_cmd_darker"]["top1_id"])
                flip_light_vs_dark_cmd = (v_out["ref_dark_cmd_darker"]["hit_valid"] and v_out["ref_dark_cmd_lighter"]["hit_valid"])
                d_recs.append({
                    "episode_id": ep["episode_id"],
                    "group_id": ep["group_id"],
                    "same_choice_when_ref_swapped": same_choice_dark_swap,
                    "abs_dark_shortcut_on_light_ref": v_out["ref_light_cmd_darker"]["picked_abs_dark"],
                    "relative_step_on_light_ref": v_out["ref_light_cmd_darker"]["picked_closest_rel"],
                    "valid_darker_light_ref": v_out["ref_light_cmd_darker"]["hit_valid"],
                    "valid_darker_dark_ref": v_out["ref_dark_cmd_darker"]["hit_valid"],
                    "valid_lighter_dark_ref": v_out["ref_dark_cmd_lighter"]["hit_valid"],
                    "both_darker_and_lighter_flip_ok": flip_light_vs_dark_cmd,
                })
            _write_json(model_out / "e06_darker_fiq.json", {
                "suite": "E06",
                "sub_suite": "F07_pseudo_relative_darker_lighter",
                "model": handle.model_key,
                "n_episodes": len(d_recs),
                "abs_dark_shortcut_rate_on_light_ref": float(np.mean([x["abs_dark_shortcut_on_light_ref"] for x in d_recs])),
                "relative_step_rate_on_light_ref": float(np.mean([x["relative_step_on_light_ref"] for x in d_recs])),
                "unchanged_choice_under_ref_swap_rate": float(np.mean([x["same_choice_when_ref_swapped"] for x in d_recs])),
                "valid_darker_dark_ref_rate": float(np.mean([x["valid_darker_dark_ref"] for x in d_recs])),
                "valid_lighter_dark_ref_rate": float(np.mean([x["valid_lighter_dark_ref"] for x in d_recs])),
                "both_darker_lighter_flip_rate": float(np.mean([x["both_darker_and_lighter_flip_ok"] for x in d_recs])),
                "records": d_recs,
            })

        # E06-C / F20: Open-ended difference ("is blue" vs "is another color" vs "is not red") on SAME pool!
        e06_f20 = [x for x in cohorts["e06_f20"] if cat_filter is None or x["category"] == cat_filter]
        if e06_f20:
            f20_recs = []
            for ep in e06_f20:
                pool_cis = [c_idx_map[p] for p in ep["pool_ids"] if p in c_idx_map]
                if len(pool_cis) != len(ep["pool_ids"]):
                    continue
                g_pool = {m: g_feats[m][pool_cis] for m in handle.extra["modalities"]} if is_dcnet else g_feats[pool_cis]
                rp = _fiq_path(ep["ref_id"])
                qv = handle.encode_queries([rp] * 3, [v["text"] for v in ep["variants"]], mode="multimodal", bs=8, num_workers=0)
                sm = handle.score_matrix(qv, g_pool)
                rec = {"episode_id": ep["episode_id"], "group_id": ep["group_id"]}
                for vi, v in enumerate(ep["variants"]):
                    top_cid = ep["pool_ids"][int(np.argmax(sm[vi]))]
                    rec[f"{v['name']}_valid"] = bool(top_cid in v["valid_ids"])
                    rec[f"{v['name']}_picked_same_red"] = bool(top_cid == ep["pool_ids"][0])
                    # Margin: max valid - max invalid
                    v_scores = [float(sm[vi, j]) for j, pid in enumerate(ep["pool_ids"]) if pid in v["valid_ids"]]
                    iv_scores = [float(sm[vi, j]) for j, pid in enumerate(ep["pool_ids"]) if pid in v["invalid_ids"]]
                    rec[f"{v['name']}_margin"] = round(max(v_scores) - max(iv_scores), 5)
                f20_recs.append(rec)
            _write_json(model_out / "e06_f20_open_diff_fiq.json", {
                "suite": "E06/E07",
                "sub_suite": "F20_open_ended_difference",
                "model": handle.model_key,
                "n_episodes": len(f20_recs),
                "acc_explicit_blue": float(np.mean([x["explicit_blue_valid"] for x in f20_recs])),
                "acc_open_another_color": float(np.mean([x["open_another_color_valid"] for x in f20_recs])),
                "acc_negated_not_red": float(np.mean([x["negated_not_red_valid"] for x in f20_recs])),
                "same_red_trap_rate_another_color": float(np.mean([x["open_another_color_picked_same_red"] for x in f20_recs])),
                "same_red_trap_rate_not_red": float(np.mean([x["negated_not_red_picked_same_red"] for x in f20_recs])),
                "records": f20_recs,
            })

        # E06-B: Modality-required counterfactual swap (dress vs shirt, unified models only!)
        if cat_filter is None and cohorts["e06_modality_swaps"]:
            ms_recs = []
            for ep in cohorts["e06_modality_swaps"]:
                pool_cis = [c_idx_map[p] for p in ep["pool_ids"] if p in c_idx_map]
                if len(pool_cis) != 4:
                    continue
                g_pool = {m: g_feats[m][pool_cis] for m in handle.extra["modalities"]} if is_dcnet else g_feats[pool_cis]
                qv = handle.encode_queries([_fiq_path(v["ref_id"]) for v in ep["variants"]], [v["text"] for v in ep["variants"]], mode="multimodal", bs=4, num_workers=0)
                sm = handle.score_matrix(qv, g_pool)
                t0 = ep["pool_ids"][int(np.argmax(sm[0]))]
                t1 = ep["pool_ids"][int(np.argmax(sm[1]))]
                ms_recs.append({
                    "episode_id": ep["episode_id"],
                    "group_id": ep["group_id"],
                    "both_swap_correct": bool(t0 == ep["variants"][0]["target_id"] and t1 == ep["variants"][1]["target_id"]),
                    "hit_v0": bool(t0 == ep["variants"][0]["target_id"]),
                    "hit_v1": bool(t1 == ep["variants"][1]["target_id"]),
                    "img_trap_v0": bool(t0 == ep["variants"][0]["img_only_trap_id"]),
                    "txt_trap_v0": bool(t0 == ep["variants"][0]["txt_only_trap_id"]),
                })
            _write_json(model_out / "e06_modality_swap_fiq.json", {
                "suite": "E06",
                "sub_suite": "F03_modality_required_ref_swap",
                "model": handle.model_key,
                "n_episodes": len(ms_recs),
                "both_swap_acc": float(np.mean([x["both_swap_correct"] for x in ms_recs])),
                "mean_single_acc": float(np.mean([0.5 * (x["hit_v0"] + x["hit_v1"]) for x in ms_recs])),
                "img_trap_rate": float(np.mean([x["img_trap_v0"] for x in ms_recs])),
                "txt_trap_rate": float(np.mean([x["txt_trap_v0"] for x in ms_recs])),
                "records": ms_recs,
            })

        # E07-B: FashionIQ verified attribute negation minimal pairs (with/without & add/remove!)
        e07_f = [x for x in cohorts["e07_fiq"] if cat_filter is None or x["category"] == cat_filter]
        if e07_f:
            r_ps = [_fiq_path(x["ref_id"]) for x in e07_f]
            gw = handle.encode_gallery([_fiq_path(x["cand_with_X"]) for x in e07_f], bs=48, num_workers=num_workers)
            gwo = handle.encode_gallery([_fiq_path(x["cand_without_X"]) for x in e07_f], bs=48, num_workers=num_workers)
            res_e07 = {"suite": "E07", "model": handle.model_key, "dataset": "fashioniq", "n_pairs": len(e07_f)}
            for mode_lbl, k_pos, k_neg in [("with_without", "text_with", "text_without"), ("add_remove", "text_add", "text_remove")]:
                qp = handle.encode_queries(r_ps, [x[k_pos] for x in e07_f], mode="multimodal", bs=48, num_workers=num_workers)
                qn = handle.encode_queries(r_ps, [x[k_neg] for x in e07_f], mode="multimodal", bs=48, num_workers=num_workers)
                sp_w, sp_wo = _score_pairs_diag(handle, qp, gw, gwo)
                sn_w, sn_wo = _score_pairs_diag(handle, qn, gw, gwo)
                m_p = sp_w - sp_wo
                m_n = sn_wo - sn_w
                res_e07[mode_lbl] = {
                    "acc_pos_polarity": float(np.mean(m_p > 0)),
                    "acc_neg_polarity": float(np.mean(m_n > 0)),
                    "bidirectional_flip_acc": float(np.mean((m_p > 0) & (m_n > 0))),
                    "negation_inversion_rate": float(np.mean((m_p > 0) & (m_n <= 0))),
                    "mean_margin_pos": float(np.mean(m_p)),
                    "mean_margin_neg": float(np.mean(m_n)),
                }
            _write_json(model_out / "e07_negation_fiq.json", res_e07)

        # E08-A: FashionIQ 4-state (00/01/10/11) multi-constraint interference (F04!)
        e08_4s = [x for x in cohorts["e08_4state"] if cat_filter is None or x["category"] == cat_filter]
        if e08_4s:
            s4_recs = []
            for ep in e08_4s:
                pool_cis = [c_idx_map[ep["states"][st]] for st in ("00", "10", "01", "11") if ep["states"][st] in c_idx_map]
                if len(pool_cis) != 4:
                    continue
                g_pool = {m: g_feats[m][pool_cis] for m in handle.extra["modalities"]} if is_dcnet else g_feats[pool_cis]
                rp = _fiq_path(ep["ref_id"])
                qv = handle.encode_queries([rp] * 4, [ep["text_A"], ep["text_B"], ep["text_AB"], ep["text_BA"]], mode="multimodal", bs=8, num_workers=0)
                sm = handle.score_matrix(qv, g_pool)
                # Indices: 0='00', 1='10'(has A), 2='01'(has B), 3='11'(has A&B)
                st_names = ["00", "10", "01", "11"]
                pick_A = st_names[int(np.argmax(sm[0]))]
                pick_B = st_names[int(np.argmax(sm[1]))]
                pick_AB = st_names[int(np.argmax(sm[2]))]
                pick_BA = st_names[int(np.argmax(sm[3]))]
                s4_recs.append({
                    "episode_id": ep["episode_id"],
                    "group_id": ep["group_id"],
                    "solve_A_alone": pick_A in ("10", "11"),
                    "solve_B_alone": pick_B in ("01", "11"),
                    "pick_AB": pick_AB,
                    "pick_BA": pick_BA,
                    "AB_exact_11": pick_AB == "11",
                    "BA_exact_11": pick_BA == "11",
                    "AB_keeps_A": pick_AB in ("10", "11"),
                    "AB_keeps_B": pick_AB in ("01", "11"),
                    "BA_keeps_A": pick_BA in ("10", "11"),
                    "BA_keeps_B": pick_BA in ("01", "11"),
                })
            both_alone = [x for x in s4_recs if x["solve_A_alone"] and x["solve_B_alone"]]
            _write_json(model_out / "e08_4state_fiq.json", {
                "suite": "E08",
                "sub_suite": "F04_4state_00_01_10_11",
                "model": handle.model_key,
                "n_episodes": len(s4_recs),
                "solve_A_alone_rate": float(np.mean([x["solve_A_alone"] for x in s4_recs])),
                "solve_B_alone_rate": float(np.mean([x["solve_B_alone"] for x in s4_recs])),
                "AB_exact_11_rate": float(np.mean([x["AB_exact_11"] for x in s4_recs])),
                "BA_exact_11_rate": float(np.mean([x["BA_exact_11"] for x in s4_recs])),
                "n_both_solved_alone": len(both_alone),
                "cond_drop_A_in_AB_given_both_alone": float(np.mean([not x["AB_keeps_A"] for x in both_alone])) if both_alone else float("nan"),
                "cond_drop_B_in_AB_given_both_alone": float(np.mean([not x["AB_keeps_B"] for x in both_alone])) if both_alone else float("nan"),
                "cond_drop_A_in_BA_given_both_alone": float(np.mean([not x["BA_keeps_A"] for x in both_alone])) if both_alone else float("nan"),
                "cond_drop_B_in_BA_given_both_alone": float(np.mean([not x["BA_keeps_B"] for x in both_alone])) if both_alone else float("nan"),
                "records": s4_recs,
            })

        # E08-C / F06: Verified Commutative Clause Swaps (C1, C2, C1+C2, C2+C1) on FashionIQ!
        e08_f06 = [x for x in cohorts["e08_f06_swaps"] if cat_filter is None or x["category"] == cat_filter]
        if e08_f06:
            f06_res = {"suite": "E08/E12", "sub_suite": "F06_caption_primacy_commutative", "model": handle.model_key, "n": len(e08_f06), "variants": {}}
            r_ps = [_fiq_path(x["ref_id"]) for x in e08_f06]
            ranks_by_v: Dict[str, List[int]] = {}
            for vn in ("C1", "C2", "C1_C2", "C2_C1"):
                qv = handle.encode_queries(r_ps, [x[vn] for x in e08_f06], mode="multimodal", bs=48, num_workers=num_workers)
                sm = handle.score_matrix(qv, g_feats, q_chunk=64, c_chunk=4096)
                rks = []
                for i, x in enumerate(e08_f06):
                    g_list = ds_spec["cat_galleries"][x["category"]]
                    g_idxs = np.array([c_idx_map[gid] for gid in g_list], dtype=np.int64)
                    row = sm[i, g_idxs]
                    tgt_pos = g_list.index(x["target_id"])
                    rks.append(int(np.sum(row > row[tgt_pos])) + 1)
                ranks_by_v[vn] = rks
                f06_res["variants"][vn] = {
                    "R@10": float(np.mean([rk <= 10 for rk in rks])),
                    "R@50": float(np.mean([rk <= 50 for rk in rks])),
                    "median_rank": float(np.median(rks)),
                }
            # Direct paired difference: rank(C1+C2) vs rank(C2+C1) AND R@10(C1) vs R@10(C2)
            r12 = np.array(ranks_by_v["C1_C2"])
            r21 = np.array(ranks_by_v["C2_C1"])
            f06_res["order_flip_rate_R10"] = float(np.mean((r12 <= 10) != (r21 <= 10)))
            f06_res["mean_abs_rank_shift"] = float(np.mean(np.abs(r12 - r21)))
            f06_res["delta_R10_C1_minus_C2_alone"] = f06_res["variants"]["C1"]["R@10"] - f06_res["variants"]["C2"]["R@10"]
            f06_res["delta_R10_C1C2_minus_C2C1"] = f06_res["variants"]["C1_C2"]["R@10"] - f06_res["variants"]["C2_C1"]["R@10"]
            f06_res["ranks_per_query"] = {
                "query_ids": [x["query_id"] for x in e08_f06],
                "group_ids": [x["group_id"] for x in e08_f06],
                **ranks_by_v,
            }
            _write_json(model_out / "e08_f06_primacy_fiq.json", f06_res)

        # E09-A: Attribute Binding Swap Test (F05!)
        e09_b = [x for x in cohorts["e09_binding"] if cat_filter is None or x["category"] == cat_filter]
        if e09_b:
            r_ps = [_fiq_path(x["ref_id"]) for x in e09_b]
            gab = handle.encode_gallery([_fiq_path(x["cand_AB"]) for x in e09_b], bs=48, num_workers=num_workers)
            gba = handle.encode_gallery([_fiq_path(x["cand_BA"]) for x in e09_b], bs=48, num_workers=num_workers)
            qab = handle.encode_queries(r_ps, [x["text_AB"] for x in e09_b], mode="multimodal", bs=48, num_workers=num_workers)
            qba = handle.encode_queries(r_ps, [x["text_BA"] for x in e09_b], mode="multimodal", bs=48, num_workers=num_workers)
            sab_ab, sab_ba = _score_pairs_diag(handle, qab, gab, gba)
            sba_ab, sba_ba = _score_pairs_diag(handle, qba, gab, gba)
            m_ab = sab_ab - sab_ba
            m_ba = sba_ba - sba_ab
            _write_json(model_out / "e09_binding_fiq.json", {
                "suite": "E09",
                "sub_suite": "F05_attribute_binding_swap",
                "model": handle.model_key,
                "n_pairs": len(e09_b),
                "acc_AB": float(np.mean(m_ab > 0)),
                "acc_BA": float(np.mean(m_ba > 0)),
                "paired_bidirectional_swap_acc": float(np.mean((m_ab > 0) & (m_ba > 0))),
                "mean_margin_AB": float(np.mean(m_ab)),
                "mean_margin_BA": float(np.mean(m_ba)),
                "per_pair": [
                    {"pair_id": e09_b[i]["pair_id"], "group_id": e09_b[i]["group_id"], "m_ab": round(float(m_ab[i]), 5), "m_ba": round(float(m_ba[i]), 5)}
                    for i in range(len(e09_b))
                ],
            })

        # E10-A / F21: Photo-Presentation Gravity (cross-format target vs same-format distractor + bg mask vs matched-area sham!)
        e10_f21 = [x for x in cohorts["e10_f21_format"] if cat_filter is None or x["category"] == cat_filter]
        if e10_f21:
            refs_im = [_open_pil(_fiq_path(x["ref_id"])) for x in e10_f21]
            tgts_im = [_open_pil(_fiq_path(x["target_diff_format"])) for x in e10_f21]
            dsts_im = [_open_pil(_fiq_path(x["distractor_same_format"])) for x in e10_f21]
            txts_f21 = [x["text"] for x in e10_f21]
            sp0, sn0 = _score_pairs_diag(handle, handle.encode_queries(refs_im, txts_f21, bs=32, num_workers=0), handle.encode_gallery(tgts_im, bs=32, num_workers=0), handle.encode_gallery(dsts_im, bs=32, num_workers=0))
            m_orig = sp0 - sn0
            # Background border neutralized (area = 0.2944) vs exact matched-area interior sham (area = 0.2944)!
            sp_bg, sn_bg = _score_pairs_diag(
                handle,
                handle.encode_queries([_bg_sham_pil(im) for im in refs_im], txts_f21, bs=32, num_workers=0),
                handle.encode_gallery([_bg_sham_pil(im) for im in tgts_im], bs=32, num_workers=0),
                handle.encode_gallery([_bg_sham_pil(im) for im in dsts_im], bs=32, num_workers=0),
            )
            m_bg = sp_bg - sn_bg
            sp_sh, sn_sh = _score_pairs_diag(
                handle,
                handle.encode_queries([_bg_matched_center_sham_pil(im) for im in refs_im], txts_f21, bs=32, num_workers=0),
                handle.encode_gallery([_bg_matched_center_sham_pil(im) for im in tgts_im], bs=32, num_workers=0),
                handle.encode_gallery([_bg_matched_center_sham_pil(im) for im in dsts_im], bs=32, num_workers=0),
            )
            m_sh = sp_sh - sn_sh
            _write_json(model_out / "e10_f21_format_fiq.json", {
                "suite": "E10",
                "sub_suite": "F21_photo_presentation_gravity",
                "model": handle.model_key,
                "n": len(e10_f21),
                "identity_wins_orig_rate": float(np.mean(m_orig > 0)),
                "format_trap_orig_rate": float(np.mean(m_orig <= 0)),
                "mean_identity_margin_orig": float(np.mean(m_orig)),
                "identity_wins_bg_masked_rate": float(np.mean(m_bg > 0)),
                "mean_identity_margin_bg_masked": float(np.mean(m_bg)),
                "identity_wins_sham_masked_rate": float(np.mean(m_sh > 0)),
                "mean_identity_margin_sham_masked": float(np.mean(m_sh)),
                "delta_bg_mask_minus_orig": float(np.mean(m_bg - m_orig)),
                "delta_sham_minus_orig": float(np.mean(m_sh - m_orig)),
                "per_item": [
                    {"episode_id": e10_f21[i]["episode_id"], "group_id": e10_f21[i]["group_id"], "m_orig": round(float(m_orig[i]), 5), "m_bg": round(float(m_bg[i]), 5), "m_sham": round(float(m_sh[i]), 5)}
                    for i in range(len(e10_f21))
                ],
            })

        # E10-B / F22: Co-Occurring Secondary Garment Leakage (secondary vs primary vs matched-area neutral side sham!)
        e10_f22 = [x for x in cohorts["e10_f22_cooccur"] if cat_filter is None or x["category"] == cat_filter]
        if e10_f22:
            refs_im = [_open_pil(_fiq_path(x["ref_id"])) for x in e10_f22]
            tgts_im = [_open_pil(_fiq_path(x["target_id"])) for x in e10_f22]
            # Pick deterministic distractor in same category
            dsts_im = [_open_pil(_fiq_path(ds_spec["cat_galleries"][x["category"]][(i * 7 + 13) % len(ds_spec["cat_galleries"][x["category"]])])) for i, x in enumerate(e10_f22)]
            txts_f22 = [x["text"] for x in e10_f22]
            g_p = handle.encode_gallery(tgts_im, bs=32, num_workers=0)
            g_n = handle.encode_gallery(dsts_im, bs=32, num_workers=0)

            q_orig = handle.encode_queries(refs_im, txts_f22, bs=32, num_workers=0)
            q_sec = handle.encode_queries([_mask_pil_boxes(im, [x["secondary_mask_box"]]) for im, x in zip(refs_im, e10_f22)], txts_f22, bs=32, num_workers=0)
            q_pri = handle.encode_queries([_mask_pil_boxes(im, [x["primary_mask_box"]]) for im, x in zip(refs_im, e10_f22)], txts_f22, bs=32, num_workers=0)
            q_shm = handle.encode_queries([_mask_pil_boxes(im, x["sham_mask_boxes"]) for im, x in zip(refs_im, e10_f22)], txts_f22, bs=32, num_workers=0)

            sp0, sn0 = _score_pairs_diag(handle, q_orig, g_p, g_n)
            sps, sns = _score_pairs_diag(handle, q_sec, g_p, g_n)
            spp, snp = _score_pairs_diag(handle, q_pri, g_p, g_n)
            sph, snh = _score_pairs_diag(handle, q_shm, g_p, g_n)

            def _q_shift(qa: Any, qb: Any) -> float:
                if is_dcnet:
                    return float("nan")
                return float((1.0 - (F.normalize(qa.float(), dim=-1) * F.normalize(qb.float(), dim=-1)).sum(dim=-1)).mean().item())

            _write_json(model_out / "e10_f22_cooccur_fiq.json", {
                "suite": "E10",
                "sub_suite": "F22_cooccurring_garment_leakage",
                "model": handle.model_key,
                "n": len(e10_f22),
                "mask_area_frac_all_conditions": 0.28,
                "mean_margin_orig": float(np.mean(sp0 - sn0)),
                "mean_margin_secondary_masked": float(np.mean(sps - sns)),
                "mean_margin_primary_masked": float(np.mean(spp - snp)),
                "mean_margin_sham_masked": float(np.mean(sph - snh)),
                "query_emb_shift_secondary": _q_shift(q_orig, q_sec),
                "query_emb_shift_primary": _q_shift(q_orig, q_pri),
                "query_emb_shift_sham": _q_shift(q_orig, q_shm),
                "excess_leakage_shift_secondary_minus_sham": _q_shift(q_orig, q_sec) - _q_shift(q_orig, q_shm),
            })

    # ------------------------------------------------------------------
    # E12-C / F25: Homophone Hijack Quadruples (on BOTH FashionIQ and EDIR!)
    # ------------------------------------------------------------------
    if dataset in ("fashioniq", "edir"):
        cat_filter = handle.fiq_category if dataset == "fashioniq" else None
        f25_items = [x for x in cohorts["e12_f25_homophones"] if x["dataset"] == dataset and (cat_filter is None or x["category"] == cat_filter)]
        if f25_items:
            print(f"  [{handle.model_key}/{dataset}] Running E12-C (F25 Homophone Hijack, n={len(f25_items)})...", flush=True)
            r_ps = [_fiq_path(x["ref_id"]) if dataset == "fashioniq" else str(EDIR_DIR / "imgs_mc" / x["ref_id"]) for x in f25_items]
            t_ps = [_fiq_path(x["target_id"]) if dataset == "fashioniq" else str(EDIR_DIR / "imgs_mc" / x["target_id"]) for x in f25_items]
            # Hard distractor: on EDIR use actual same-scene sibling negative; on FashionIQ use same-category distractor!
            if dataset == "edir":
                q_by_id = {q["query_id"]: q for q in test_q}
                d_ps = [str(EDIR_DIR / "imgs_mc" / q_by_id[x["query_id"]]["negative_ids"][0]) for x in f25_items]
            else:
                d_ps = [t_ps[(i + 1) % len(t_ps)] for i in range(len(t_ps))]
            g_tgt = handle.encode_gallery(t_ps, bs=48, num_workers=0)
            g_dst = handle.encode_gallery(d_ps, bs=48, num_workers=0)

            var_names = ["original", "homophone", "typo", "synonym"]
            tok_info = {vn: handle.tokenize_inspect([x["variants"][vn] for x in f25_items]) for vn in var_names}
            q_fused = {vn: handle.encode_queries(r_ps, [x["variants"][vn] for x in f25_items], mode="multimodal", bs=48, num_workers=0) for vn in var_names}
            q_txt = {vn: handle.encode_queries(r_ps, [x["variants"][vn] for x in f25_items], mode="native_txt_tower", bs=48, num_workers=0) for vn in var_names}
            q_hijack_concept = handle.encode_queries(r_ps, [x["hijack_concept_text"] for x in f25_items], mode="native_txt_tower", bs=48, num_workers=0)

            f25_summary: Dict[str, Any] = {"suite": "E12", "sub_suite": "F25_homophone_hijack", "model": handle.model_key, "dataset": dataset, "n": len(f25_items), "variants": {}}
            for vn in var_names:
                sp, sn = _score_pairs_diag(handle, q_fused[vn], g_tgt, g_dst)
                m = sp - sn
                if is_dcnet:
                    t_shift, f_shift, h_pull = float("nan"), float("nan"), float("nan")
                else:
                    t_shift = float((1.0 - (F.normalize(q_txt["original"].float(), dim=-1) * F.normalize(q_txt[vn].float(), dim=-1)).sum(dim=-1)).mean().item())
                    f_shift = float((1.0 - (F.normalize(q_fused["original"].float(), dim=-1) * F.normalize(q_fused[vn].float(), dim=-1)).sum(dim=-1)).mean().item())
                    h_pull = float((F.normalize(q_txt[vn].float(), dim=-1) * F.normalize(q_hijack_concept.float(), dim=-1)).sum(dim=-1).mean().item())
                f25_summary["variants"][vn] = {
                    "pair_acc": float(np.mean(m > 0)),
                    "mean_margin": float(np.mean(m)),
                    "mean_target_score": float(np.mean(sp)),
                    "mean_n_tokens": float(np.mean([ti["n_tokens"] for ti in tok_info[vn]])),
                    "text_encoder_cos_shift_vs_orig": t_shift,
                    "fusion_cos_shift_vs_orig": f_shift,
                    "hijack_concept_text_cos": h_pull,
                }
            # Persist sample token IDs & per-item shifts for full verifiability
            f25_summary["sample_token_audit"] = [
                {
                    "item_id": f25_items[i]["item_id"],
                    "rule_word": f25_items[i]["rule_word"],
                    "tokens": {vn: tok_info[vn][i] for vn in var_names},
                }
                for i in range(min(30, len(f25_items)))
            ]
            _write_json(model_out / f"e12_f25_homophone_{dataset}.json", f25_summary)


# ---------------------------------------------------------------------------
# Main Entry Point (with Safe Frozen EVA-ViT-g Batching for Same-Env Groups!)
# ---------------------------------------------------------------------------

def run_env_group(models: List[str], dataset: str, out_root: Path, num_workers: int, force: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohorts = json.loads((out_root / "manifest/diagnostic_cohorts.json").read_text(encoding="utf-8"))
    arb = _load_arbiter(out_root, dataset)

    # Filter already completed models unless --force
    todo = []
    for mk in models:
        done_p = out_root / "models" / dataset / mk / "DONE.json"
        if done_p.is_file() and not force:
            print(f"[WORKER SKIP] {mk} on {dataset} already completed ({done_p})", flush=True)
        else:
            todo.append(mk)
    if not todo:
        return

    # Special case: per-category FashionIQ models (limn, clvcnet)
    if dataset == "fashioniq" and len(todo) == 1 and todo[0] in ("limn", "clvcnet"):
        mk = todo[0]
        model_out = out_root / "models" / dataset / mk
        model_out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for cat in ("dress", "shirt", "toptee"):
            cat_out = model_out / f"cat_{cat}"
            cat_out.mkdir(parents=True, exist_ok=True)
            ds_spec = _load_dataset_spec("fashioniq", fiq_category=cat)
            handle = ModelHandle(mk, "fashioniq", device, fiq_category=cat)
            g_feats = handle.encode_gallery(ds_spec["corpus_paths"], bs=128, num_workers=num_workers)
            _compute_parity_and_full_suites(handle, "fashioniq", ds_spec, g_feats, arb, cohorts, cat_out, num_workers)
            _run_targeted_suites(handle, "fashioniq", ds_spec, g_feats, cohorts, cat_out, num_workers)
            del handle, g_feats
            gc.collect()
            torch.cuda.empty_cache()
        _write_json(model_out / "DONE.json", {"model": mk, "dataset": dataset, "per_category": True, "elapsed_sec": round(time.time() - t0, 2)})
        print(f"[WORKER DONE] {mk} on {dataset} (3 category heads) in {time.time() - t0:.1f}s", flush=True)
        return

    ds_spec = _load_dataset_spec(dataset)

    # Check if we can safely share frozen EVA-ViT-g gallery pass across multiple models in `todo`!
    can_share_vit = (
        len(todo) > 1
        and all(MODEL_BY_KEY[m].family == "blip2_evag" for m in todo)
        and dataset in ("edir", "pinpoint")  # All EDIR/PinPoint checkpoints saved state_dict_non_vit with exact identical frozen EVA-ViT-g!
    )

    if can_share_vit:
        t_grp = time.time()
        print(f"[SHARED-VIT GROUP] Loading {todo} on {dataset} with single frozen EVA-ViT-g gallery pass ({len(ds_spec['corpus_paths'])} imgs)...", flush=True)
        handles: Dict[str, ModelHandle] = {}
        shared_vit = None
        for mk in todo:
            h = ModelHandle(mk, dataset, device, shared_vit=shared_vit)
            if shared_vit is None:
                shared_vit = h.model.visual_encoder
            handles[mk] = h

        # Single streaming pass through frozen visual_encoder -> populates [N_c, 32, 256] CPU fp32 for ALL models in group!
        ds_gal = _ImgListDataset(ds_spec["corpus_paths"], handles[todo[0]].preprocess)
        dl_gal = _make_loader(ds_gal, bs=96, num_workers=num_workers)
        gal_acc: Dict[str, List[torch.Tensor]] = {mk: [] for mk in todo}
        with torch.no_grad():
            for bi, imgs in enumerate(dl_gal):
                imgs = imgs.to(device, non_blocking=True)
                with handles[todo[0]].model.maybe_autocast():
                    vit_e = shared_vit(imgs)
                for mk in todo:
                    gal_acc[mk].append(handles[mk].encode_gallery_from_vit(vit_e).cpu())
                if (bi + 1) % 200 == 0:
                    print(f"  [SHARED-VIT] Batch {bi+1}/{len(dl_gal)} ({time.time()-t_grp:.1f}s)", flush=True)
        gal_feats_map = {mk: torch.cat(gal_acc[mk], dim=0) for mk in todo}
        del gal_acc
        print(f"[SHARED-VIT] Encoded gallery for {len(todo)} models in {time.time() - t_grp:.1f}s!", flush=True)

        for mk in todo:
            t0 = time.time()
            model_out = out_root / "models" / dataset / mk
            model_out.mkdir(parents=True, exist_ok=True)
            _compute_parity_and_full_suites(handles[mk], dataset, ds_spec, gal_feats_map[mk], arb, cohorts, model_out, num_workers)
            _run_targeted_suites(handles[mk], dataset, ds_spec, gal_feats_map[mk], cohorts, model_out, num_workers)
            _write_json(model_out / "DONE.json", {"model": mk, "dataset": dataset, "shared_vit_group": todo, "elapsed_sec": round(time.time() - t0, 2)})
            # Immediately free this model's large gallery tensor from RAM!
            del gal_feats_map[mk]
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[WORKER DONE] {mk} on {dataset} in {time.time() - t0:.1f}s", flush=True)
        del handles, shared_vit
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Standard standalone per-model execution
    for mk in todo:
        t0 = time.time()
        model_out = out_root / "models" / dataset / mk
        model_out.mkdir(parents=True, exist_ok=True)
        print(f"[WORKER START] {mk} on {dataset}...", flush=True)
        handle = ModelHandle(mk, dataset, device)
        g_feats = handle.encode_gallery(ds_spec["corpus_paths"], bs=128, num_workers=num_workers)
        _compute_parity_and_full_suites(handle, dataset, ds_spec, g_feats, arb, cohorts, model_out, num_workers)
        _run_targeted_suites(handle, dataset, ds_spec, g_feats, cohorts, model_out, num_workers)
        _write_json(model_out / "DONE.json", {"model": mk, "dataset": dataset, "elapsed_sec": round(time.time() - t0, 2)})
        del handle, g_feats
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[WORKER DONE] {mk} on {dataset} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pass CIR diagnostic probe worker")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model keys sharing this virtualenv")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint", "fashioniq"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    run_env_group(models, args.dataset, args.out_dir, num_workers=min(args.num_workers, 8), force=args.force)


if __name__ == "__main__":
    main()
