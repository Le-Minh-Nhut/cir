#!/usr/bin/env python3
"""Deterministic, zero-leakage diagnostic cohort builder for Suites E04, E06-E13, E15.

Strictly executed BEFORE running any model evaluation so every split, minimal pair,
counterfactual swap, ROI/sham box, and no-target episode is pre-registered and frozen.
Uses only official dataset splits/annotations, objective pixel measurements (CIE Lab
luminance, border uniformity, EDIR aligned diff heatmaps), and deterministic regexes
(with provenance explicitly tagged).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.probes.cohort import build_cohort_manifest  # noqa: E402

EDIR_DIR = REPO_ROOT / "data/EDIR"
EDIR_SPLIT = EDIR_DIR / "splits/70_10_20_seed42"
PP_DIR = REPO_ROOT / "data/PinPoint"
PP_SPLIT = PP_DIR / "splits/70_10_20_seed42"
FIQ_DIR = REPO_ROOT / "data/FashionIQ"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _fiq_img_path(img_id: str) -> str:
    p = FIQ_DIR / "images" / f"{img_id}.png"
    if p.is_file():
        return str(p)
    p2 = FIQ_DIR / "images" / f"{img_id}.jpg"
    if p2.is_file():
        return str(p2)
    return str(p)


def _compute_image_pixel_stats(img_path: str) -> Dict[str, float]:
    """Objective pixel stats: foreground Lab luminance L*, dominant hue, border uniformity, aspect ratio."""
    try:
        with PIL.Image.open(img_path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return {"ok": 0.0}
    h, w = rgb.shape[:2]
    # Resize to 128x128 for fast consistent pixel stats
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    # Border ring (outer 10px) -> background uniformity & white-studio score
    border_mask = np.zeros((128, 128), dtype=bool)
    border_mask[:10, :] = True
    border_mask[-10:, :] = True
    border_mask[:, :10] = True
    border_mask[:, -10:] = True
    border_rgb = small[border_mask].astype(np.float32)
    border_std = float(border_rgb.std())
    border_white_frac = float(np.mean((border_rgb[:, 0] > 235) & (border_rgb[:, 1] > 235) & (border_rgb[:, 2] > 235)))

    # Center foreground mask (center 50% excluding near-white studio background)
    center_rgb = small[24:104, 28:100]
    center_L = L[24:104, 28:100]
    non_white = ~((center_rgb[:, :, 0] > 240) & (center_rgb[:, :, 1] > 240) & (center_rgb[:, :, 2] > 240))
    if non_white.sum() >= 100:
        fg_L = float(np.median(center_L[non_white]))
        fg_rgb = center_rgb[non_white]
        fg_frac = float(non_white.mean())
    else:
        fg_L = float(np.median(center_L))
        fg_rgb = center_rgb.reshape(-1, 3)
        fg_frac = 0.0

    # Color stats in HSV on foreground
    fg_hsv = cv2.cvtColor(fg_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat_med = float(np.median(fg_hsv[:, 1]))
    val_med = float(np.median(fg_hsv[:, 2]))
    # Compute pixel fraction matching red / blue
    hue = fg_hsv[:, 0].astype(np.float32)  # 0..179 in OpenCV
    sat = fg_hsv[:, 1].astype(np.float32)
    val = fg_hsv[:, 2].astype(np.float32)
    red_frac = float(np.mean(((hue <= 10) | (hue >= 168)) & (sat >= 65) & (val >= 50)))
    blue_frac = float(np.mean((hue >= 98) & (hue <= 132) & (sat >= 55) & (val >= 45)))
    return {
        "ok": 1.0,
        "w": float(w),
        "h": float(h),
        "aspect": float(h / max(w, 1)),
        "fg_L": fg_L,
        "fg_frac": fg_frac,
        "border_std": border_std,
        "border_white_frac": border_white_frac,
        "sat_med": sat_med,
        "val_med": val_med,
        "red_frac": red_frac,
        "blue_frac": blue_frac,
    }


def _compute_edir_diff_roi_and_sham(ref_path: str, pos_path: str, box_frac: float = 0.38) -> Dict[str, Any]:
    """Locate the edited ROI on aligned EDIR (ref, pos) pairs via Gaussian-smoothed |pos - ref| heatmap,
    plus an exact matched-area lowest-energy sham box! Normalized coords [x1, y1, x2, y2] in [0, 1]."""
    try:
        with PIL.Image.open(ref_path) as im1, PIL.Image.open(pos_path) as im2:
            a = np.asarray(im1.convert("RGB").resize((224, 224), PIL.Image.BILINEAR), dtype=np.float32)
            b = np.asarray(im2.convert("RGB").resize((224, 224), PIL.Image.BILINEAR), dtype=np.float32)
    except Exception:
        return {"roi_box": [0.31, 0.31, 0.69, 0.69], "sham_box": [0.02, 0.02, 0.40, 0.40], "diff_energy_ratio": 1.0}

    diff = np.mean(np.abs(a - b), axis=2)  # [224, 224]
    diff_smooth = cv2.GaussianBlur(diff, (15, 15), 0)
    bh = max(32, int(round(224 * box_frac)))
    bw = bh
    # Integral image for exact O(1) sliding window sum
    integ = cv2.integral(diff_smooth)  # [225, 225]
    # Vectorized window sums
    ys = np.arange(0, 224 - bh + 1, 4)
    xs = np.arange(0, 224 - bw + 1, 4)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sums = integ[yy + bh, xx + bw] - integ[yy, xx + bw] - integ[yy + bh, xx] + integ[yy, xx]
    max_idx = np.unravel_index(int(np.argmax(sums)), sums.shape)
    ry, rx = int(ys[max_idx[0]]), int(xs[max_idx[1]])
    roi_energy = float(sums[max_idx])

    # Find lowest-energy non-overlapping (IoU < 0.05) sham box of EXACT same size (bw, bh)
    best_sham = (0, 0)
    best_sham_energy = float("inf")
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            ix1, iy1 = max(rx, int(x)), max(ry, int(y))
            ix2, iy2 = min(rx + bw, int(x) + bw), min(ry + bh, int(y) + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = inter / float(2 * bw * bh - inter + 1e-6)
            if iou < 0.05:
                en = float(sums[iy, ix])
                if en < best_sham_energy:
                    best_sham_energy = en
                    best_sham = (int(x), int(y))
    sx, sy = best_sham
    return {
        "roi_box": [round(rx / 224.0, 4), round(ry / 224.0, 4), round((rx + bw) / 224.0, 4), round((ry + bh) / 224.0, 4)],
        "sham_box": [round(sx / 224.0, 4), round(sy / 224.0, 4), round((sx + bw) / 224.0, 4), round((sy + bh) / 224.0, 4)],
        "diff_energy_ratio": round(roi_energy / max(best_sham_energy, 1e-3), 3),
        "box_area_frac": round((bw * bh) / float(224 * 224), 4),
    }


def _extract_fiq_image_attrs_from_human_caps() -> Dict[str, Dict[str, Counter]]:
    """Aggregate attribute mentions strictly from official human FashionIQ val captions."""
    attr_patterns = {
        "color_black": r"\bblack\b",
        "color_white": r"\bwhite\b",
        "color_red": r"\b(red|burgundy|maroon|crimson)\b",
        "color_blue": r"\b(blue|navy|teal|denim)\b",
        "color_green": r"\b(green|olive|emerald)\b",
        "color_pink": r"\b(pink|magenta|rose)\b",
        "color_yellow": r"\b(yellow|gold|mustard)\b",
        "color_purple": r"\b(purple|violet|lavender)\b",
        "sleeve_long": r"\b(long sleeve|long sleeves|long-sleeve|full sleeve)\b",
        "sleeve_short": r"\b(short sleeve|short sleeves|short-sleeve)\b",
        "sleeve_none": r"\b(sleeveless|no sleeve|no sleeves|tank top|spaghetti strap|strapless)\b",
        "pat_floral": r"\b(floral|flower|flowers)\b",
        "pat_stripe": r"\b(stripe|striped|stripes)\b",
        "pat_solid": r"\b(solid|plain|no pattern|no print|no graphic|no logo)\b",
        "pat_print": r"\b(print|printed|graphic|logo|text|writing|words|lettering)\b",
        "has_collar": r"\b(collar|collared|polo)\b",
        "has_buttons": r"\b(button|buttons|buttoned|button-up|button down)\b",
        "has_hood": r"\b(hood|hoodie|hooded)\b",
        "len_long": r"\b(longer|maxi|floor length|ankle length|long dress)\b",
        "len_short": r"\b(shorter|mini|knee length|short dress)\b",
    }
    compiled = {k: re.compile(v, re.I) for k, v in attr_patterns.items()}
    img_pos_votes: Dict[str, Counter] = defaultdict(Counter)
    img_neg_votes: Dict[str, Counter] = defaultdict(Counter)
    img_cat: Dict[str, str] = {}

    # Also read train captions for richer verified target mentions (val image IDs only!)
    val_ids_by_cat: Dict[str, set] = {}
    for cat in ("dress", "shirt", "toptee"):
        s = set(json.loads((FIQ_DIR / f"image_splits/split.{cat}.val.json").read_text()))
        val_ids_by_cat[cat] = s
        for iid in s:
            img_cat[iid] = cat
        caps = json.loads((FIQ_DIR / f"captions/cap.{cat}.val.json").read_text())
        for r in caps:
            tgt = r["target"]
            text = f"{r['captions'][0]} ; {r['captions'][1]}"
            # Skip negated clauses when crediting positive target attribute
            for k, rx in compiled.items():
                m = rx.search(text)
                if m:
                    # Check 18-char prefix before match for negation
                    prefix = text[max(0, m.start() - 20) : m.start()].lower()
                    if re.search(r"\b(no|not|without|instead of|rather than|less)\b", prefix):
                        img_neg_votes[tgt][k] += 1
                    else:
                        img_pos_votes[tgt][k] += 1
    return {"pos": img_pos_votes, "neg": img_neg_votes, "cat": img_cat}


def build_all_cohorts(out_root: Path, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    manifest_dir = out_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Freeze Cohort Manifest
    cohort_manifest = build_cohort_manifest(manifest_dir / "cohort_manifest.json")
    print(f"[COHORTS] Wrote cohort_manifest.json ({cohort_manifest['counts']})", flush=True)

    # Load splits
    edir_tr = _load_jsonl(EDIR_SPLIT / "train.jsonl")
    edir_va = _load_jsonl(EDIR_SPLIT / "val.jsonl")
    edir_te = _load_jsonl(EDIR_SPLIT / "test.jsonl")
    pp_tr = _load_jsonl(PP_SPLIT / "train.jsonl")
    pp_va = _load_jsonl(PP_SPLIT / "val.jsonl")
    pp_te = _load_jsonl(PP_SPLIT / "test.jsonl")

    fiq_val: List[Dict[str, Any]] = []
    fiq_galleries: Dict[str, List[str]] = {}
    for cat in ("dress", "shirt", "toptee"):
        fiq_galleries[cat] = json.loads((FIQ_DIR / f"image_splits/split.{cat}.val.json").read_text())
        caps = json.loads((FIQ_DIR / f"captions/cap.{cat}.val.json").read_text())
        for qi, r in enumerate(caps):
            fiq_val.append({
                "query_id": f"{cat}:{qi}",
                "category": cat,
                "ref_id": r["candidate"],
                "target_id": r["target"],
                "captions": r["captions"],
                "text": f"{r['captions'][0]} and {r['captions'][1]}",
            })

    # Pre-aggregate human caption attributes on FashionIQ val
    fiq_attrs = _extract_fiq_image_attrs_from_human_caps()
    pos_v = fiq_attrs["pos"]
    neg_v = fiq_attrs["neg"]

    # ------------------------------------------------------------------
    # E04: Layer-wise Probing Tasks (strictly group-split train/dev/test!)
    # ------------------------------------------------------------------
    neg_rx = re.compile(r"\b(remove|without|no |not |don't|except|erase|delete|eliminate)\b", re.I)
    ocr_rx = re.compile(r"\b(text|word|words|letter|letters|font|watermark|sign|label|logo|title|caption|writing|written)\b", re.I)
    gaze_left_rx = re.compile(r"\b(look(ing|s)?\s+(to\s+the\s+)?left|gaz(ing|e)\s+left|turn(ed|ing)?\s+(head\s+)?left|facing\s+left)\b", re.I)
    gaze_right_rx = re.compile(r"\b(look(ing|s)?\s+(to\s+the\s+)?right|gaz(ing|e)\s+right|turn(ed|ing)?\s+(head\s+)?right|facing\s+right)\b", re.I)
    gaze_front_rx = re.compile(r"\b(look(ing|s)?\s+(at\s+the\s+camera|straight|forward|ahead|directly)|facing\s+(forward|the\s+camera|front))\b", re.I)
    pose_rx = re.compile(r"\b(arm|arms|hand|hands|leg|legs|standing|sitting|seated|pose|posture|raised|crossed|kneeling|jumping|running|walking|pointing|waving|holding)\b", re.I)

    def _parse_count_1to4(text: str) -> Optional[int]:
        t = text.lower()
        hits = []
        if re.search(r"\b(one|single|1|only one|sole)\b", t):
            hits.append(1)
        if re.search(r"\b(two|pair of|couple of|second|2)\b", t):
            hits.append(2)
        if re.search(r"\b(three|trio|third|3)\b", t):
            hits.append(3)
        if re.search(r"\b(four|fourth|4)\b", t):
            hits.append(4)
        return hits[0] if len(hits) == 1 else None

    # EDIR query/gallery probe rows (scene-disjoint train/val/test!)
    def _edir_probe_split(rows: List[Dict[str, Any]], max_n: Optional[int] = None) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            is_neg = int(r["category"] == "remove" or bool(neg_rx.search(r["text"])))
            is_complex = int(r["category"] == "complex")
            is_geom = int(r["category"] in ("shape", "viewpoint"))
            is_ocr = int(bool(ocr_rx.search(r["text"])))
            is_pose = int(bool(pose_rx.search(r["text"])))
            cnt = _parse_count_1to4(r["text"]) if r["category"] == "cardinality" else None
            gaze_lbl = -1
            if gaze_left_rx.search(r["text"]):
                gaze_lbl = 0
            elif gaze_right_rx.search(r["text"]):
                gaze_lbl = 1
            elif gaze_front_rx.search(r["text"]):
                gaze_lbl = 2
            out.append({
                "query_id": r["query_id"],
                "group_id": r["metadata"]["stem_base"],
                "ref_id": r["ref_id"],
                "ref_path": str(EDIR_DIR / r["ref_image_rel"]),
                "pos_id": r["positive_ids"][0],
                "pos_path": str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0]),
                "neg_id": r["negative_ids"][0],
                "neg_path": str(EDIR_DIR / "imgs_mc" / r["negative_ids"][0]),
                "text": r["text"],
                "category": r["category"],
                "labels": {
                    "F01_is_negation": is_neg,
                    "F04_is_complex": is_complex,
                    "F08_is_geom_vs_color": 1 if r["category"] in ("shape", "viewpoint") else (0 if r["category"] in ("color", "material", "texture") else -1),
                    "F09_cardinality_1to4": cnt if cnt is not None else -1,
                    "F10_is_ocr": is_ocr,
                    "F23_gaze_dir": gaze_lbl,
                    "F24_is_pose": is_pose,
                },
            })
        if max_n is not None and len(out) > max_n:
            # Stratified subsample preserving all positive cases of rare labels (gaze, cardinality, ocr, neg)
            must = [x for x in out if x["labels"]["F09_cardinality_1to4"] != -1 or x["labels"]["F23_gaze_dir"] != -1 or x["labels"]["F10_is_ocr"] == 1 or x["labels"]["F01_is_negation"] == 1 or x["labels"]["F08_is_geom_vs_color"] == 1]
            rest = [x for x in out if x not in must]
            rng_local = random.Random(seed)
            rng_local.shuffle(rest)
            out = must + rest[: max(0, max_n - len(must))]
        return out

    e04_edir = {
        "train": _edir_probe_split(edir_tr, max_n=1400),
        "dev": _edir_probe_split(edir_va, max_n=497),
        "test": _edir_probe_split(edir_te, max_n=1003),
    }

    # FashionIQ gallery-branch attribute presence & binding probe (strictly image-ID disjoint 60/20/20!)
    fiq_gal_rows: List[Dict[str, Any]] = []
    for iid, pv in pos_v.items():
        nv = neg_v[iid]
        has_long_sl = 1 if (pv["sleeve_long"] >= 1 and pv["sleeve_none"] == 0 and nv["sleeve_long"] == 0) else (0 if (pv["sleeve_none"] >= 1 and pv["sleeve_long"] == 0) else -1)
        has_pattern = 1 if ((pv["pat_floral"] + pv["pat_stripe"] + pv["pat_print"]) >= 1 and pv["pat_solid"] == 0) else (0 if (pv["pat_solid"] >= 1 and (pv["pat_floral"] + pv["pat_stripe"] + pv["pat_print"]) == 0) else -1)
        is_dark_col = 1 if (pv["color_black"] >= 1 and pv["color_white"] == 0) else (0 if (pv["color_white"] >= 1 and pv["color_black"] == 0) else -1)
        if has_long_sl == -1 and has_pattern == -1 and is_dark_col == -1:
            continue
        fiq_gal_rows.append({
            "image_id": iid,
            "group_id": iid,
            "category": fiq_attrs["cat"].get(iid, "unknown"),
            "image_path": _fiq_img_path(iid),
            "labels": {
                "F01_gallery_long_vs_sleeveless": has_long_sl,
                "F01_gallery_pattern_vs_solid": has_pattern,
                "F01_gallery_black_vs_white": is_dark_col,
            },
        })
    fiq_gal_rows.sort(key=lambda x: hashlib.md5(x["image_id"].encode()).hexdigest())
    n_fg = min(len(fiq_gal_rows), 1800)
    fiq_gal_rows = fiq_gal_rows[:n_fg]
    n_tr_fg = int(0.60 * n_fg)
    n_dv_fg = int(0.20 * n_fg)
    e04_fiq_gallery = {
        "train": fiq_gal_rows[:n_tr_fg],
        "dev": fiq_gal_rows[n_tr_fg : n_tr_fg + n_dv_fg],
        "test": fiq_gal_rows[n_tr_fg + n_dv_fg :],
    }

    # F28 BLIP-2 Blind Spot matched cohorts from saved official EDIR & PinPoint predictions!
    def _build_f28_cohort(dataset: str, test_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = REPO_ROOT / f"workbench/artifacts/benchmark_runs/{dataset}/70_10_20_seed42"
        blip_models = ["sprc_pretrain", "mtst_ptha_blip2", "csmcir_blip2", "hint_blip2", "habit_blip2", "intent_blip2", "conesep_blip2", "airknow_blip2", "tme_blip2", "qure_vitl"]
        clip_models = ["clip4cir_rn50x4", "encoder_vitb32", "tgcir_vitb16", "limn_vitl14"]
        preds_b = {m: json.loads((base / m / "test_predictions_top50.json").read_text()) for m in blip_models}
        preds_c = {m: json.loads((base / m / "test_predictions_top50.json").read_text()) for m in clip_models}
        clip_hit_blip_miss, blip_hit_clip_miss, both_hit = [], [], []
        for r in test_rows:
            qid = r["query_id"]
            pos_set = set(r["positive_ids"])
            if dataset == "edir":
                b_hits = sum(1 for m in blip_models if preds_b[m][qid][0] in pos_set)
                c_hits = sum(1 for m in clip_models if preds_c[m][qid][0] in pos_set)
            else:
                b_hits = sum(1 for m in blip_models if any(x in pos_set for x in preds_b[m][qid][:10]))
                c_hits = sum(1 for m in clip_models if any(x in pos_set for x in preds_c[m][qid][:10]))
            # Wrong top-1 candidate from conesep (representative BLIP-2) or clip4cir
            wrong_b = next((x for x in preds_b["conesep_blip2"][qid] if x not in pos_set), r["negative_ids"][0] if r.get("negative_ids") else preds_b["conesep_blip2"][qid][0])
            entry = {
                "query_id": qid,
                "group_id": r["metadata"]["stem_base"] if dataset == "edir" else r["ref_id"],
                "ref_id": r["ref_id"],
                "pos_id": r["positive_ids"][0],
                "wrong_id": wrong_b,
                "text": r["text"],
                "category": r["category"],
                "blip_hits": b_hits,
                "clip_hits": c_hits,
            }
            if b_hits == 0 and c_hits >= 1:
                clip_hit_blip_miss.append(entry)
            elif c_hits == 0 and b_hits >= 5:
                blip_hit_clip_miss.append(entry)
            elif b_hits >= 5 and c_hits >= 2:
                both_hit.append(entry)
        rng_l = random.Random(seed)
        rng_l.shuffle(blip_hit_clip_miss)
        rng_l.shuffle(both_hit)
        k_match = max(len(clip_hit_blip_miss), 40)
        return {
            "clip_hit_blip_miss": clip_hit_blip_miss,
            "blip_hit_clip_miss": blip_hit_clip_miss[:k_match],
            "matched_both_hit": both_hit[:k_match],
        }

    e04_f28 = {
        "edir": _build_f28_cohort("edir", edir_te),
        "pinpoint": _build_f28_cohort("pinpoint", pp_te),
    }

    # Pre-compute pixel stats for images referenced in fiq_val (up to 4,500 images -> ~4 seconds!)
    print("[COHORTS] Computing objective CIE Lab / HSV / border pixel stats on FashionIQ val sample...", flush=True)
    fiq_pixel_stats: Dict[str, Dict[str, float]] = {}
    for cat in ("dress", "shirt", "toptee"):
        cat_val_qs = [r for r in fiq_val if r["category"] == cat]
        ref_tgt_order = []
        seen_c = set()
        for r in cat_val_qs:
            for iid in (r["ref_id"], r["target_id"]):
                if iid not in seen_c:
                    seen_c.add(iid)
                    ref_tgt_order.append(iid)
        for iid in fiq_galleries[cat]:
            if iid not in seen_c:
                seen_c.add(iid)
                ref_tgt_order.append(iid)
        for iid in ref_tgt_order[:2500]:
            st = _compute_image_pixel_stats(_fiq_img_path(iid))
            if st.get("ok", 0.0) > 0.5:
                st["category"] = cat
                fiq_pixel_stats[iid] = st

    # ------------------------------------------------------------------
    # E06: Counterfactual Reference Swapping (F07 darker/lighter, F03 modality-req, F20 open-ended)
    # ------------------------------------------------------------------
    e06_darker_episodes: List[Dict[str, Any]] = []
    for cat in ("dress", "shirt", "toptee"):
        cat_imgs = [(iid, st) for iid, st in fiq_pixel_stats.items() if st["category"] == cat and st["fg_frac"] >= 0.15]
        very_dark = [iid for iid, st in cat_imgs if st["fg_L"] <= 20.0]
        dark = [iid for iid, st in cat_imgs if 25.0 <= st["fg_L"] <= 42.0]
        mid = [iid for iid, st in cat_imgs if 48.0 <= st["fg_L"] <= 64.0]
        light = [iid for iid, st in cat_imgs if st["fg_L"] >= 72.0]
        rng.shuffle(very_dark); rng.shuffle(dark); rng.shuffle(mid); rng.shuffle(light)
        n_ep = min(40, len(very_dark) // 2, len(dark) // 2, len(mid) // 2, len(light) // 2)
        for ei in range(n_ep):
            # Counterfactual pair sharing the EXACT SAME 4-item candidate pool:
            # Pool = [abs_dark_distractor (L<=20), cand_dark (25..42), cand_mid (48..64), cand_light (>=72)]
            # Ref A = ref_light (>=72): "is slightly darker" -> cand_mid is relative step darker, cand_dark/very_dark also darker, cand_light NOT darker
            # Ref B = ref_dark (25..42): "is darker" -> ONLY abs_dark_distractor is darker!
            # Ref C = ref_mid (48..64): "is darker" -> dark & very_dark valid; "is lighter" -> light valid!
            ref_light = light[2 * ei]
            cand_light = light[2 * ei + 1]
            ref_mid = mid[2 * ei]
            cand_mid = mid[2 * ei + 1]
            ref_dark = dark[2 * ei]
            cand_dark = dark[2 * ei + 1]
            cand_vdark = very_dark[2 * ei]
            pool = [cand_vdark, cand_dark, cand_mid, cand_light]
            e06_darker_episodes.append({
                "episode_id": f"e06_darker_{cat}_{ei}",
                "group_id": f"{cat}_{ei}",
                "category": cat,
                "pool_ids": pool,
                "pool_L": [round(fiq_pixel_stats[x]["fg_L"], 1) for x in pool],
                "variants": [
                    {
                        "name": "ref_light_cmd_darker",
                        "ref_id": ref_light,
                        "ref_L": round(fiq_pixel_stats[ref_light]["fg_L"], 1),
                        "text": "is darker",
                        "valid_ids": [cand_mid, cand_dark, cand_vdark],
                        "closest_relative_id": cand_mid,
                        "abs_dark_id": cand_vdark,
                        "invalid_ids": [cand_light],
                    },
                    {
                        "name": "ref_dark_cmd_darker",
                        "ref_id": ref_dark,
                        "ref_L": round(fiq_pixel_stats[ref_dark]["fg_L"], 1),
                        "text": "is darker",
                        "valid_ids": [cand_vdark],
                        "closest_relative_id": cand_vdark,
                        "abs_dark_id": cand_vdark,
                        "invalid_ids": [cand_mid, cand_light],
                    },
                    {
                        "name": "ref_dark_cmd_lighter",
                        "ref_id": ref_dark,
                        "ref_L": round(fiq_pixel_stats[ref_dark]["fg_L"], 1),
                        "text": "is lighter",
                        "valid_ids": [cand_mid, cand_light],
                        "closest_relative_id": cand_mid,
                        "invalid_ids": [cand_vdark],
                    },
                    {
                        "name": "ref_mid_cmd_darker",
                        "ref_id": ref_mid,
                        "ref_L": round(fiq_pixel_stats[ref_mid]["fg_L"], 1),
                        "text": "is darker",
                        "valid_ids": [cand_vdark, cand_dark],
                        "closest_relative_id": cand_dark,
                        "abs_dark_id": cand_vdark,
                        "invalid_ids": [cand_light],
                    },
                ],
            })

    # E06-C / E07 / F20 Open-Ended Difference: "is blue" vs "is another color" vs "is not red" on SAME pool!
    # Select verified RED garments (pixel red_frac >= 0.25 OR human caption color_red) and BLUE garments and OTHER colors
    e06_f20_episodes: List[Dict[str, Any]] = []
    for cat in ("dress", "shirt", "toptee"):
        red_imgs = [iid for iid, st in fiq_pixel_stats.items() if st["category"] == cat and (st["red_frac"] >= 0.22 or pos_v[iid]["color_red"] >= 1) and pos_v[iid]["color_blue"] == 0 and st["blue_frac"] < 0.05]
        blue_imgs = [iid for iid, st in fiq_pixel_stats.items() if st["category"] == cat and (st["blue_frac"] >= 0.22 or pos_v[iid]["color_blue"] >= 1) and pos_v[iid]["color_red"] == 0 and st["red_frac"] < 0.05]
        other_imgs = [iid for iid, st in fiq_pixel_stats.items() if st["category"] == cat and st["red_frac"] < 0.04 and st["blue_frac"] < 0.04 and pos_v[iid]["color_red"] == 0 and pos_v[iid]["color_blue"] == 0 and (pos_v[iid]["color_green"] >= 1 or pos_v[iid]["color_yellow"] >= 1 or pos_v[iid]["color_black"] >= 1 or pos_v[iid]["color_white"] >= 1)]
        rng.shuffle(red_imgs); rng.shuffle(blue_imgs); rng.shuffle(other_imgs)
        n_ep = min(35, len(red_imgs) // 2, len(blue_imgs), len(other_imgs) // 2)
        for ei in range(n_ep):
            ref_red = red_imgs[2 * ei]
            cand_red = red_imgs[2 * ei + 1]
            cand_blue = blue_imgs[ei]
            cand_oth1 = other_imgs[2 * ei]
            cand_oth2 = other_imgs[2 * ei + 1]
            pool = [cand_red, cand_blue, cand_oth1, cand_oth2]
            e06_f20_episodes.append({
                "episode_id": f"f20_{cat}_{ei}",
                "group_id": f"f20_{cat}_{ei}",
                "category": cat,
                "ref_id": ref_red,
                "pool_ids": pool,
                "pool_roles": ["red_same_as_ref", "blue", "other_1", "other_2"],
                "variants": [
                    {"name": "explicit_blue", "text": "is blue", "valid_ids": [cand_blue], "invalid_ids": [cand_red, cand_oth1, cand_oth2]},
                    {"name": "open_another_color", "text": "is another color", "valid_ids": [cand_blue, cand_oth1, cand_oth2], "invalid_ids": [cand_red]},
                    {"name": "negated_not_red", "text": "is not red", "valid_ids": [cand_blue, cand_oth1, cand_oth2], "invalid_ids": [cand_red]},
                ],
            })

    # E06-B Modality-Required Counterfactual Swap (F03): same text "is black" (or "is red"), swap ref between dress & shirt/toptee!
    e06_modality_swaps: List[Dict[str, Any]] = []
    black_dress = [i for i, st in fiq_pixel_stats.items() if st["category"] == "dress" and pos_v[i]["color_black"] >= 1 and pos_v[i]["color_white"] == 0]
    white_dress = [i for i, st in fiq_pixel_stats.items() if st["category"] == "dress" and pos_v[i]["color_white"] >= 1 and pos_v[i]["color_black"] == 0]
    black_shirt = [i for i, st in fiq_pixel_stats.items() if st["category"] == "shirt" and pos_v[i]["color_black"] >= 1 and pos_v[i]["color_white"] == 0]
    white_shirt = [i for i, st in fiq_pixel_stats.items() if st["category"] == "shirt" and pos_v[i]["color_white"] >= 1 and pos_v[i]["color_black"] == 0]
    rng.shuffle(black_dress); rng.shuffle(white_dress); rng.shuffle(black_shirt); rng.shuffle(white_shirt)
    for ei in range(min(40, len(black_dress), len(white_dress) // 2, len(black_shirt), len(white_shirt) // 2)):
        # Pool has: [black_dress, white_dress_cand, black_shirt, white_shirt_cand]
        # Ref=white_dress + "is black" -> target MUST be black_dress (needs BOTH img=dress AND txt=black!)
        # Ref=white_shirt + "is black" -> target MUST be black_shirt!
        pool = [black_dress[ei], white_dress[2 * ei + 1], black_shirt[ei], white_shirt[2 * ei + 1]]
        e06_modality_swaps.append({
            "episode_id": f"e06_modreq_{ei}",
            "group_id": f"modreq_{ei}",
            "pool_ids": pool,
            "variants": [
                {"name": "ref_white_dress_to_black", "ref_id": white_dress[2 * ei], "text": "is black", "target_id": black_dress[ei], "img_only_trap_id": white_dress[2 * ei + 1], "txt_only_trap_id": black_shirt[ei]},
                {"name": "ref_white_shirt_to_black", "ref_id": white_shirt[2 * ei], "text": "is black", "target_id": black_shirt[ei], "img_only_trap_id": white_shirt[2 * ei + 1], "txt_only_trap_id": black_dress[ei]},
            ],
        })

    # ------------------------------------------------------------------
    # E07: Negation & Polarity Minimal Pairs (F01, F20)
    # ------------------------------------------------------------------
    # E07-A: EDIR remove & addition queries -> exact polarity inversion minimal pairs!
    e07_edir_pairs: List[Dict[str, Any]] = []
    for r in edir_te:
        if r["category"] not in ("remove", "addition"):
            continue
        txt = r["text"].strip()
        # Construct clean syntactic polarity mirror
        if r["category"] == "remove":
            # Target has X removed; ref (or sibling) has X. Flip "remove/without/no/delete/erase" -> "add/with/keep"
            inv = re.sub(r"\b(remove|delete|erase|eliminate)\b", "keep and emphasize", txt, count=1, flags=re.I)
            if inv.lower() == txt.lower():
                inv = re.sub(r"\bwithout\b", "with", txt, count=1, flags=re.I)
            if inv.lower() == txt.lower():
                inv = re.sub(r"\bno\b", "with", txt, count=1, flags=re.I)
            if inv.lower() == txt.lower():
                inv = f"Keep the object instead of: {txt}"
            e07_edir_pairs.append({
                "pair_id": f"e07_edir_{r['query_id']}",
                "group_id": r["metadata"]["stem_base"],
                "orig_category": "remove",
                "ref_id": r["ref_id"],
                "ref_path": str(EDIR_DIR / r["ref_image_rel"]),
                "cand_without_X": r["positive_ids"][0],
                "cand_without_X_path": str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0]),
                "cand_with_X": r["negative_ids"][0],
                "cand_with_X_path": str(EDIR_DIR / "imgs_mc" / r["negative_ids"][0]),
                "text_without_X": txt,
                "text_with_X": inv,
            })
        else:
            inv = re.sub(r"\b(add|adding|include|with)\b", "remove", txt, count=1, flags=re.I)
            if inv.lower() == txt.lower():
                inv = f"Without any added object: remove {txt}"
            e07_edir_pairs.append({
                "pair_id": f"e07_edir_{r['query_id']}",
                "group_id": r["metadata"]["stem_base"],
                "orig_category": "addition",
                "ref_id": r["ref_id"],
                "ref_path": str(EDIR_DIR / r["ref_image_rel"]),
                "cand_with_X": r["positive_ids"][0],
                "cand_with_X_path": str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0]),
                "cand_without_X": r["negative_ids"][0],
                "cand_without_X_path": str(EDIR_DIR / "imgs_mc" / r["negative_ids"][0]),
                "text_with_X": txt,
                "text_without_X": inv,
            })

    # E07-B: FashionIQ verified attribute minimal pairs (fixed ref + (has_X, no_X), exact with/without & add/remove!)
    e07_fiq_pairs: List[Dict[str, Any]] = []
    attr_defs = [
        ("sleeves", "sleeve_long", "sleeve_none", "with long sleeves", "without sleeves", "add long sleeves", "remove the sleeves"),
        ("floral_print", "pat_floral", "pat_solid", "with floral print", "without floral print", "add floral pattern", "remove the floral pattern"),
        ("stripes", "pat_stripe", "pat_solid", "with stripes", "without stripes", "add stripes", "remove the stripes"),
        ("graphic_logo", "pat_print", "pat_solid", "with a graphic print", "without any print or logo", "add graphic text print", "remove the print and logo"),
        ("collar", "has_collar", "sleeve_none", "with a collar", "without a collar", "add a collar", "remove the collar"),
    ]
    for attr_name, pos_k, neg_k, w_txt, wo_txt, add_txt, rem_txt in attr_defs:
        for cat in ("dress", "shirt", "toptee"):
            c_with = [i for i in fiq_galleries[cat] if pos_v[i][pos_k] >= 1 and pos_v[i][neg_k] == 0 and neg_v[i][pos_k] == 0]
            c_without = [i for i in fiq_galleries[cat] if pos_v[i][neg_k] >= 1 and pos_v[i][pos_k] == 0 and neg_v[i][neg_k] == 0]
            rng.shuffle(c_with); rng.shuffle(c_without)
            n_p = min(15, len(c_with) // 2, len(c_without))
            for pi in range(n_p):
                ref_i = c_with[2 * pi]
                cand_w = c_with[2 * pi + 1]
                cand_wo = c_without[pi]
                e07_fiq_pairs.append({
                    "pair_id": f"e07_fiq_{attr_name}_{cat}_{pi}",
                    "group_id": f"{attr_name}_{cat}_{pi}",
                    "category": cat,
                    "attribute": attr_name,
                    "ref_id": ref_i,
                    "cand_with_X": cand_w,
                    "cand_without_X": cand_wo,
                    "text_with": w_txt,
                    "text_without": wo_txt,
                    "text_add": add_txt,
                    "text_remove": rem_txt,
                })

    # ------------------------------------------------------------------
    # E08: Multi-Constraint Interference (F04 4-state 00/01/10/11 + EDIR complex + F06 C1/C2 swap)
    # ------------------------------------------------------------------
    # E08-A: FashionIQ 4-state (00, 01, 10, 11) pools for (A=color, B=sleeve/pattern)!
    e08_4state: List[Dict[str, Any]] = []
    ab_combos = [
        ("black", "color_black", "white", "color_white", "long sleeves", "sleeve_long", "sleeveless", "sleeve_none"),
        ("white", "color_white", "black", "color_black", "long sleeves", "sleeve_long", "sleeveless", "sleeve_none"),
        ("black", "color_black", "red", "color_red", "floral pattern", "pat_floral", "solid plain", "pat_solid"),
        ("blue", "color_blue", "white", "color_white", "stripes", "pat_stripe", "solid plain", "pat_solid"),
    ]
    for a_lbl, a_pos, a_neg_lbl, a_neg, b_lbl, b_pos, b_neg_lbl, b_neg in ab_combos:
        for cat in ("dress", "shirt", "toptee"):
            s11 = [i for i in fiq_galleries[cat] if pos_v[i][a_pos] >= 1 and pos_v[i][b_pos] >= 1 and pos_v[i][a_neg] == 0 and pos_v[i][b_neg] == 0]
            s10 = [i for i in fiq_galleries[cat] if pos_v[i][a_pos] >= 1 and pos_v[i][b_neg] >= 1 and pos_v[i][a_neg] == 0 and pos_v[i][b_pos] == 0]
            s01 = [i for i in fiq_galleries[cat] if pos_v[i][a_neg] >= 1 and pos_v[i][b_pos] >= 1 and pos_v[i][a_pos] == 0 and pos_v[i][b_neg] == 0]
            s00 = [i for i in fiq_galleries[cat] if pos_v[i][a_neg] >= 1 and pos_v[i][b_neg] >= 1 and pos_v[i][a_pos] == 0 and pos_v[i][b_pos] == 0]
            rng.shuffle(s11); rng.shuffle(s10); rng.shuffle(s01); rng.shuffle(s00)
            n_ep = min(15, len(s11), len(s10), len(s01), len(s00) // 2)
            for ei in range(n_ep):
                ref_i = s00[2 * ei]
                c00 = s00[2 * ei + 1]
                c10 = s10[ei]
                c01 = s01[ei]
                c11 = s11[ei]
                e08_4state.append({
                    "episode_id": f"e08_4s_{a_lbl}_{b_lbl.replace(' ','_')}_{cat}_{ei}",
                    "group_id": f"{cat}_{a_lbl}_{b_lbl}_{ei}",
                    "category": cat,
                    "ref_id": ref_i,
                    "attr_A": a_lbl,
                    "attr_B": b_lbl,
                    "states": {"00": c00, "10": c10, "01": c01, "11": c11},
                    "pool_ids": [c00, c10, c01, c11],
                    "text_A": f"is {a_lbl}",
                    "text_B": f"has {b_lbl}",
                    "text_AB": f"is {a_lbl} and has {b_lbl}",
                    "text_BA": f"has {b_lbl} and is {a_lbl}",
                })

    # E08-B: EDIR complex decomposed into 2-3 individual clauses
    e08_edir_complex: List[Dict[str, Any]] = []
    for r in edir_te:
        if r["category"] != "complex":
            continue
        # Strip standard prefixes and split on ", and " / " and " / ", with "
        clean = re.sub(r"^(I want to see a picture that shows|Look for|Find me a picture that has|A similar scene with|With)\s+", "", r["text"].strip(), flags=re.I)
        parts = [p.strip(" .,;") for p in re.split(r",\s+and\s+|\s+and\s+|,\s+but\s+|,\s+with\s+", clean) if len(p.strip()) >= 8]
        if len(parts) >= 2:
            e08_edir_complex.append({
                "query_id": r["query_id"],
                "group_id": r["metadata"]["stem_base"],
                "ref_id": r["ref_id"],
                "pos_id": r["positive_ids"][0],
                "neg_ids": r["negative_ids"],
                "full_text": r["text"],
                "clauses": parts[:3],
            })
    e08_edir_complex = e08_edir_complex[:100]

    # E08-C / F06: FashionIQ verified meaning-preserving commutative (C1, C2) pairs
    non_comm_rx = re.compile(r"\b(it|its|them|they|that|this|those|these|also|instead|too|same|other|then|after|before|while|but)\b", re.I)
    e08_f06_swaps: List[Dict[str, Any]] = []
    for r in fiq_val:
        c1, c2 = r["captions"][0].strip(), r["captions"][1].strip()
        if len(c1.split()) < 3 or len(c2.split()) < 3:
            continue
        # Strictly verify meaning-preserving commutativity (no cross-clause anaphora or discourse connectors)
        if non_comm_rx.search(c1) or non_comm_rx.search(c2):
            continue
        # Require both clauses to express distinct visual attributes
        if c1.lower() == c2.lower():
            continue
        e08_f06_swaps.append({
            "query_id": r["query_id"],
            "group_id": f"{r['category']}_{r['ref_id']}",
            "category": r["category"],
            "ref_id": r["ref_id"],
            "target_id": r["target_id"],
            "C1": c1,
            "C2": c2,
            "C1_C2": f"{c1} and {c2}",
            "C2_C1": f"{c2} and {c1}",
        })
    rng.shuffle(e08_f06_swaps)
    e08_f06_swaps = e08_f06_swaps[:300]

    # ------------------------------------------------------------------
    # E09: Attribute Binding Swap Test (F05 + F26 metadata-only preserve/edit)
    # ------------------------------------------------------------------
    # E09-A: Swapped color-of-garment vs color-of-print/text/stripes on FashionIQ!
    e09_binding_pairs: List[Dict[str, Any]] = []
    # Mine captions explicitly stating "<c1> ... with <c2> (print|text|writing|logo|stripes|pattern|flowers|trim|collar|sleeves)"
    bind_rx = re.compile(
        r"\b(black|white|red|blue|pink|green|yellow|purple)\b[^.;,]{0,25}\bwith\b[^.;,]{0,20}\b(black|white|red|blue|pink|green|yellow|purple)\b[^.;,]{0,20}\b(print|text|writing|logo|stripes|stripe|pattern|flowers|floral|trim|collar|sleeves|dots|design)\b",
        re.I,
    )
    by_binding: Dict[Tuple[str, str, str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    for r in fiq_val:
        txt = f"{r['captions'][0]} ; {r['captions'][1]}"
        m = bind_rx.search(txt)
        if m:
            c1, c2, part = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
            if c1 != c2:
                by_binding[(r["category"], c1, c2, part)].append((r["query_id"], r["ref_id"], r["target_id"]))
    # Also group by coarse part family (print/text/writing/logo/design -> "print", stripes/stripe -> "stripes", etc.)
    part_norm = {"print": "print", "text": "print", "writing": "print", "logo": "print", "design": "print", "stripes": "stripes", "stripe": "stripes", "flowers": "floral", "floral": "floral", "pattern": "pattern", "trim": "trim", "collar": "collar", "sleeves": "sleeves", "dots": "dots"}
    by_bind_coarse: Dict[Tuple[str, str, str, str], List[Tuple[str, str, str]]] = defaultdict(list)
    for (cat, c1, c2, part), items in by_binding.items():
        by_bind_coarse[(cat, c1, c2, part_norm.get(part, part))].extend(items)
    seen_sym = set()
    for (cat, c1, c2, part), items_ab in by_bind_coarse.items():
        if (cat, c2, c1, part) in by_bind_coarse and (cat, min(c1, c2), max(c1, c2), part) not in seen_sym:
            seen_sym.add((cat, min(c1, c2), max(c1, c2), part))
            items_ba = by_bind_coarse[(cat, c2, c1, part)]
            for idx in range(min(len(items_ab), len(items_ba), 20)):
                q1, ref1, tgt_ab = items_ab[idx]
                q2, ref2, tgt_ba = items_ba[idx]
                if tgt_ab == tgt_ba:
                    continue
                e09_binding_pairs.append({
                    "pair_id": f"e09_bind_{cat}_{c1}_{c2}_{part}_{idx}",
                    "group_id": f"{cat}_{c1}_{c2}_{part}_{idx}",
                    "category": cat,
                    "c1": c1,
                    "c2": c2,
                    "part": part,
                    "ref_id": ref1,
                    "cand_AB": tgt_ab,
                    "cand_BA": tgt_ba,
                    "text_AB": f"is {c1} with {c2} {part}",
                    "text_BA": f"is {c2} with {c1} {part}",
                })

    # E09-B / F26: Preserve-vs-Edit using ONLY official PinPoint l1_interest (Women's Fashion vs Men's Fashion)
    # metadata (zero gender inference from appearance!).
    e09_f26_preserve: List[Dict[str, Any]] = []
    pp_wf = [r for r in pp_te if r["metadata"].get("l1_interest") == "Women's Fashion" and r["category"] in ("Swap", "Context Fit")]
    pp_mf = [r for r in pp_te if r["metadata"].get("l1_interest") == "Men's Fashion" and r["category"] in ("Swap", "Context Fit")]
    # Build pools of verified positive images per domain + attribute/color keyword so viol_preserve matches edit word but flips l1_interest!
    attr_words = ["black", "white", "red", "blue", "green", "brown", "grey", "gray", "beige", "pink", "denim", "leather", "jacket", "shirt", "pants", "shoes", "coat", "sweater", "casual", "formal", "summer", "winter", "long", "short", "dark", "light"]
    mf_by_kw: Dict[str, List[str]] = defaultdict(list)
    wf_by_kw: Dict[str, List[str]] = defaultdict(list)
    for r in pp_mf + [x for x in pp_tr if x["metadata"].get("l1_interest") == "Men's Fashion"]:
        t = r["text"].lower()
        for kw in attr_words:
            if re.search(rf"\b{kw}\b", t) and len(mf_by_kw[kw]) < 20:
                mf_by_kw[kw].extend(r["positive_ids"][:1])
    for r in pp_wf + [x for x in pp_tr if x["metadata"].get("l1_interest") == "Women's Fashion"]:
        t = r["text"].lower()
        for kw in attr_words:
            if re.search(rf"\b{kw}\b", t) and len(wf_by_kw[kw]) < 20:
                wf_by_kw[kw].extend(r["positive_ids"][:1])
    for dom_name, rows_dom, opp_by_kw in [("Women's Fashion", pp_wf, mf_by_kw), ("Men's Fashion", pp_mf, wf_by_kw)]:
        for r in rows_dom:
            t = r["text"].lower()
            matched_kws = [kw for kw in attr_words if re.search(rf"\b{kw}\b", t) and opp_by_kw[kw]]
            if not matched_kws or not r["negative_ids"]:
                continue
            kw = matched_kws[0]
            opp_cand = next((x for x in opp_by_kw[kw] if x not in set(r["positive_ids"])), None)
            same_dom_viol_edit = next((x for x in r["negative_ids"] if x != r["ref_id"]), None)
            if opp_cand and same_dom_viol_edit:
                e09_f26_preserve.append({
                    "episode_id": f"f26_{r['query_id']}",
                    "group_id": r["ref_id"],
                    "query_id": r["query_id"],
                    "preserve_domain_metadata": dom_name,
                    "matched_edit_kw": kw,
                    "ref_id": r["ref_id"],
                    "text": r["text"],
                    "cand_both_ok": r["positive_ids"][0],
                    "cand_viol_preserve_only": opp_cand,
                    "cand_viol_edit_only": same_dom_viol_edit,
                })
    e09_f26_preserve = e09_f26_preserve[:120]

    # ------------------------------------------------------------------
    # E10: Visual Context Masking (F21 Photo-Format, F22 Co-Occurring Garment, F24 Limb Pose)
    # ------------------------------------------------------------------
    # E10-A / F21: Studio white-background flat-lay vs non-white model-worn pairs on FashionIQ!
    e10_f21_format: List[Dict[str, Any]] = []
    for cat in ("dress", "shirt", "toptee"):
        studio_white = [i for i, st in fiq_pixel_stats.items() if st["category"] == cat and st["border_white_frac"] >= 0.72 and st["border_std"] <= 26.0]
        natural_scene = [i for i, st in fiq_pixel_stats.items() if st["category"] == cat and st["border_white_frac"] <= 0.35 and st["border_std"] >= 28.0]
        studio_set, nat_set = set(studio_white), set(natural_scene)
        # Find real FashionIQ val queries where ref and target differ in photo presentation format!
        cross_fmt = [r for r in fiq_val if r["category"] == cat and ((r["ref_id"] in studio_set and r["target_id"] in nat_set) or (r["ref_id"] in nat_set and r["target_id"] in studio_set))]
        for r in cross_fmt[:40]:
            ref_is_studio = r["ref_id"] in studio_set
            same_fmt_pool = studio_white if ref_is_studio else natural_scene
            same_fmt_distractor = next((x for x in same_fmt_pool if x != r["ref_id"] and x != r["target_id"]), None)
            if same_fmt_distractor:
                e10_f21_format.append({
                    "episode_id": f"e10_f21_{r['query_id']}",
                    "group_id": f"{cat}_{r['ref_id']}",
                    "query_id": r["query_id"],
                    "category": cat,
                    "ref_id": r["ref_id"],
                    "ref_format": "studio_white" if ref_is_studio else "natural_scene",
                    "target_diff_format": r["target_id"],
                    "distractor_same_format": same_fmt_distractor,
                    "text": r["text"],
                })

    # E10-B / F22: Co-occurring secondary garment masking on FashionIQ toptee/shirt (bottom 35% pants/skirt vs top 35% vs matched-area neutral side sham!)
    e10_f22_cooccur: List[Dict[str, Any]] = []
    for r in fiq_val:
        if r["category"] not in ("toptee", "shirt"):
            continue
        st = fiq_pixel_stats.get(r["ref_id"])
        if st and st["aspect"] >= 1.15:  # Tall portrait model-worn shots contain lower garment!
            e10_f22_cooccur.append({
                "episode_id": f"e10_f22_{r['query_id']}",
                "group_id": f"{r['category']}_{r['ref_id']}",
                "query_id": r["query_id"],
                "category": r["category"],
                "ref_id": r["ref_id"],
                "target_id": r["target_id"],
                "text": r["text"],
                # Normalized mask boxes [x1, y1, x2, y2], all EXACTLY 0.28 image area!
                "secondary_mask_box": [0.10, 0.65, 0.90, 1.00],  # area = 0.80 * 0.35 = 0.28
                "primary_mask_box": [0.10, 0.20, 0.90, 0.55],    # area = 0.80 * 0.35 = 0.28
                "sham_mask_boxes": [[0.00, 0.00, 0.14, 1.00], [0.86, 0.00, 1.00, 1.00]],  # area = 2 * 0.14 = 0.28 (exact match!)
            })
    rng.shuffle(e10_f22_cooccur)
    e10_f22_cooccur = e10_f22_cooccur[:120]

    # E10-C / F24: Limb Pose Masking on EDIR pose queries using exact pixel-diff ROI + matched-area sham!
    print("[COHORTS] Computing aligned EDIR diff ROIs + matched-area shams for E10-C (F24) and E11 (F08/F09/F10/F23/F28)...", flush=True)
    e10_f24_pose: List[Dict[str, Any]] = []
    for r in edir_te:
        if pose_rx.search(r["text"]):
            rp = str(EDIR_DIR / r["ref_image_rel"])
            pp = str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0])
            boxes = _compute_edir_diff_roi_and_sham(rp, pp, box_frac=0.38)
            e10_f24_pose.append({
                "episode_id": f"e10_f24_{r['query_id']}",
                "group_id": r["metadata"]["stem_base"],
                "query_id": r["query_id"],
                "ref_id": r["ref_id"],
                "pos_id": r["positive_ids"][0],
                "neg_ids": r["negative_ids"],
                "text": r["text"],
                **boxes,
            })

    # ------------------------------------------------------------------
    # E11: Resolution & Patch Sensitivity (F08, F09, F10, F23, F28)
    # ------------------------------------------------------------------
    e11_items: List[Dict[str, Any]] = []
    seen_e11 = set()
    f28_edir_qids = {x["query_id"] for x in e04_f28["edir"]["clip_hit_blip_miss"]}

    def _add_e11(r: Dict[str, Any], tag: str) -> None:
        if r["query_id"] in seen_e11:
            return
        seen_e11.add(r["query_id"])
        rp = str(EDIR_DIR / r["ref_image_rel"])
        pp = str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0])
        boxes = _compute_edir_diff_roi_and_sham(rp, pp, box_frac=0.45)
        e11_items.append({
            "query_id": r["query_id"],
            "group_id": r["metadata"]["stem_base"],
            "sub_cohort": tag,
            "category": r["category"],
            "ref_id": r["ref_id"],
            "pos_id": r["positive_ids"][0],
            "neg_ids": r["negative_ids"],
            "text": r["text"],
            **boxes,
        })

    for r in edir_te:
        if ocr_rx.search(r["text"]) and len([x for x in e11_items if x["sub_cohort"] == "F10_ocr"]) < 55:
            _add_e11(r, "F10_ocr")
    for r in edir_te:
        if (gaze_left_rx.search(r["text"]) or gaze_right_rx.search(r["text"]) or gaze_front_rx.search(r["text"]) or re.search(r"\b(gaze|look(ing)?\s+(left|right|away|down|up|at))\b", r["text"], re.I)) and len([x for x in e11_items if x["sub_cohort"] == "F23_gaze"]) < 35:
            _add_e11(r, "F23_gaze")
    for r in edir_te:
        if r["category"] in ("shape", "viewpoint") and len([x for x in e11_items if x["sub_cohort"] == "F08_geom"]) < 40:
            _add_e11(r, "F08_geom")
    for r in edir_te:
        if r["category"] == "cardinality" and len([x for x in e11_items if x["sub_cohort"] == "F09_cardinality"]) < 30:
            _add_e11(r, "F09_cardinality")
    for r in edir_te:
        if r["query_id"] in f28_edir_qids and len([x for x in e11_items if x["sub_cohort"] == "F28_blip2_blindspot"]) < 30:
            _add_e11(r, "F28_blip2_blindspot")

    # ------------------------------------------------------------------
    # E12: Paraphrase, Syntax, Pragmatic Rewriting & Homophone Hijack (F06, F11, F13, F25)
    # ------------------------------------------------------------------
    # E12-A: Controlled single-factor edits on PinPoint test queries (distinct from natural 258 paraphrase groups)
    e12_syntax_ctrl: List[Dict[str, Any]] = []
    for r in pp_te[:120]:
        orig = r["text"].strip()
        e12_syntax_ctrl.append({
            "query_id": r["query_id"],
            "group_id": r["ref_id"],
            "ref_id": r["ref_id"],
            "positive_ids": r["positive_ids"],
            "negative_ids": r["negative_ids"],
            "category": r["category"],
            "variants": {
                "original": orig,
                "verbose_abstract": f"I would love to explore an aesthetic visual vibe and stylistic approach that captures the following concept: {orig}",
                "imperative_concise": re.sub(r"^(can you |could you |please |i want to see |show me |find me )", "", orig, flags=re.I),
                "question_form": f"Can you show me {orig[0].lower() + orig[1:] if len(orig) > 1 else orig}?",
            },
        })

    # E12-B / F13: Pragmatic Implicit vs Explicit Target Category on PinPoint Complement / Explore / Context Fit
    cat_explicit_hint = {
        "Women's Fashion": "women's clothing outfit",
        "Men's Fashion": "men's clothing outfit",
        "Home Decor": "interior home decor item",
        "Beauty": "makeup beauty hairstyle look",
        "Art": "artwork illustration",
        "Food and Drinks": "food dish or drink",
        "Vehicles": "car or vehicle",
        "Animals": "animal photo",
        "Wedding": "wedding attire or decoration",
        "Travel": "travel destination or outfit",
        "DIY and Crafts": "craft project",
        "Entertainment": "movie or pop culture image",
    }
    e12_f13_pragmatic: List[Dict[str, Any]] = []
    for r in pp_te:
        if r["category"] not in ("Complement", "Explore", "Context Fit"):
            continue
        l1 = r["metadata"].get("l1_interest", "")
        if l1 not in cat_explicit_hint:
            continue
        e12_f13_pragmatic.append({
            "query_id": r["query_id"],
            "group_id": r["ref_id"],
            "ref_id": r["ref_id"],
            "category": r["category"],
            "l1_interest": l1,
            "positive_ids": r["positive_ids"],
            "negative_ids": r["negative_ids"],
            "text_implicit": r["text"],
            "text_explicit": f"{r['text']} (specifically a {cat_explicit_hint[l1]})",
        })
    rng.shuffle(e12_f13_pragmatic)
    e12_f13_pragmatic = e12_f13_pragmatic[:150]

    # E12-C / F25: Homophone Hijack Quadruples (original, homophone, 1-char typo, clean synonym)
    homophone_rules = [
        # (regex, orig_word, homophone, char_typo, synonym, hijack_concept_text)
        (r"\bsee[\s-]through\b", "see-through", "sea-through", "see-thrugh", "sheer transparent", "ocean sea water waves"),
        (r"\bplain\b", "plain", "plane", "plian", "simple solid", "airplane jet aircraft"),
        (r"\bcollar\b", "collar", "color", "colar", "neckline", "rainbow colorful paint"),
        (r"\bwaist\b", "waist", "waste", "wiast", "midsection", "trash garbage waste"),
        (r"\bflare\b", "flare", "flair", "flaer", "wide bell", "bright fire signal flare"),
        (r"\bflared\b", "flared", "flaired", "flarred", "wide bell", "bright fire signal flare"),
        (r"\bbow\b", "bow", "bough", "bwo", "ribbon knot", "tree branch bough"),
        (r"\btee\b", "tee", "tea", "teee", "t-shirt", "cup of hot tea"),
        (r"\bred\b", "red", "read", "rde", "crimson", "reading a book"),
        (r"\bblue\b", "blue", "blew", "bluu", "azure", "wind blowing"),
        (r"\bknit\b", "knit", "nit", "knti", "woven sweater", "tiny bug"),
        (r"\bflower\b", "flower", "flour", "flowre", "blossom", "white baking flour powder"),
        (r"\bfloral\b", "floral", "floural", "flroal", "flowery", "white baking flour powder"),
        (r"\blace\b", "lace", "lase", "lcae", "eyelet trim", "laser light beam"),
        (r"\bjean\b", "jean", "gene", "jaen", "denim", "DNA helix"),
        (r"\bjeans\b", "jeans", "genes", "jaens", "denim pants", "DNA helix"),
        (r"\bpair\b", "pair", "pear", "piar", "couple of two", "green pear fruit"),
        (r"\bwood\b", "wood", "would", "wodo", "timber", "text word"),
        (r"\bsun\b", "sun", "son", "snu", "sunlight", "young boy child"),
        (r"\bnight\b", "night", "knight", "nihgt", "dark evening", "medieval armored knight"),
    ]
    e12_f25_homophones: List[Dict[str, Any]] = []
    # Mine from FashionIQ val + EDIR test (cap 8 per rule for balanced coverage!)
    rule_counts: Counter = Counter()
    for src_ds, rows_src in [("fashioniq", fiq_val), ("edir", edir_te)]:
        for r in rows_src:
            txt = r["text"]
            for pat, ow, homo, typo, syn, hijack in homophone_rules:
                if rule_counts[ow] >= 8:
                    continue
                if re.search(pat, txt, re.I):
                    rule_counts[ow] += 1
                    e12_f25_homophones.append({
                        "item_id": f"f25_{src_ds}_{r['query_id']}_{ow}",
                        "dataset": src_ds,
                        "query_id": r["query_id"],
                        "group_id": f"{src_ds}_{r['ref_id']}",
                        "category": r["category"],
                        "ref_id": r["ref_id"],
                        "target_id": r["target_id"] if src_ds == "fashioniq" else r["positive_ids"][0],
                        "rule_word": ow,
                        "hijack_concept_text": hijack,
                        "variants": {
                            "original": txt,
                            "homophone": re.sub(pat, homo, txt, count=1, flags=re.I),
                            "typo": re.sub(pat, typo, txt, count=1, flags=re.I),
                            "synonym": re.sub(pat, syn, txt, count=1, flags=re.I),
                        },
                    })
                    break

    # ------------------------------------------------------------------
    # E13: PinPoint Multi-Category Target Mapping (via official qrels + metadata)
    # ------------------------------------------------------------------
    img_to_l1: Dict[str, set] = defaultdict(set)
    for split_rows in (pp_tr, pp_va, pp_te):
        for r in split_rows:
            l1 = r["metadata"].get("l1_interest")
            if l1:
                img_to_l1[r["ref_id"]].add(l1)
                for p in r["positive_ids"]:
                    img_to_l1[p].add(l1)
    e13_pp_img_categories = {k: sorted(v) for k, v in img_to_l1.items()}

    # ------------------------------------------------------------------
    # E15: Verified No-Target Episode Split (deterministic 50% held-out query IDs)
    # ------------------------------------------------------------------
    def _split_half(ids: List[str], tag: str) -> List[str]:
        s = sorted(ids, key=lambda x: hashlib.md5(f"{tag}:{x}".encode()).hexdigest())
        return s[: len(s) // 2]

    e15_notarget_qids = {
        "edir": _split_half([r["query_id"] for r in edir_te], "e15_edir"),
        "pinpoint": _split_half([r["query_id"] for r in pp_te], "e15_pinpoint"),
        "fashioniq": _split_half([r["query_id"] for r in fiq_val], "e15_fiq"),
    }

    # Persist frozen diagnostic cohorts cleanly by suite
    cohorts = {
        "schema_version": "1.0.0",
        "seed": seed,
        "summary_counts": {
            "e04_edir_train": len(e04_edir["train"]),
            "e04_edir_dev": len(e04_edir["dev"]),
            "e04_edir_test": len(e04_edir["test"]),
            "e04_fiq_gal_train": len(e04_fiq_gallery["train"]),
            "e04_fiq_gal_dev": len(e04_fiq_gallery["dev"]),
            "e04_fiq_gal_test": len(e04_fiq_gallery["test"]),
            "e04_f28_edir_clip_hit_blip_miss": len(e04_f28["edir"]["clip_hit_blip_miss"]),
            "e04_f28_pp_clip_hit_blip_miss": len(e04_f28["pinpoint"]["clip_hit_blip_miss"]),
            "e06_darker_episodes": len(e06_darker_episodes),
            "e06_f20_episodes": len(e06_f20_episodes),
            "e06_modality_swaps": len(e06_modality_swaps),
            "e07_edir_pairs": len(e07_edir_pairs),
            "e07_fiq_pairs": len(e07_fiq_pairs),
            "e08_4state_episodes": len(e08_4state),
            "e08_edir_complex": len(e08_edir_complex),
            "e08_f06_swaps": len(e08_f06_swaps),
            "e09_binding_pairs": len(e09_binding_pairs),
            "e09_f26_preserve": len(e09_f26_preserve),
            "e10_f21_format": len(e10_f21_format),
            "e10_f22_cooccur": len(e10_f22_cooccur),
            "e10_f24_pose": len(e10_f24_pose),
            "e11_items": len(e11_items),
            "e12_syntax_ctrl": len(e12_syntax_ctrl),
            "e12_f13_pragmatic": len(e12_f13_pragmatic),
            "e12_f25_homophones": len(e12_f25_homophones),
            "e13_pp_categorized_images": len(e13_pp_img_categories),
        },
        "e04_edir": e04_edir,
        "e04_fiq_gallery": e04_fiq_gallery,
        "e04_f28": e04_f28,
        "e06_darker": e06_darker_episodes,
        "e06_f20": e06_f20_episodes,
        "e06_modality_swaps": e06_modality_swaps,
        "e07_edir": e07_edir_pairs,
        "e07_fiq": e07_fiq_pairs,
        "e08_4state": e08_4state,
        "e08_edir_complex": e08_edir_complex,
        "e08_f06_swaps": e08_f06_swaps,
        "e09_binding": e09_binding_pairs,
        "e09_f26_preserve": e09_f26_preserve,
        "e10_f21_format": e10_f21_format,
        "e10_f22_cooccur": e10_f22_cooccur,
        "e10_f24_pose": e10_f24_pose,
        "e11_items": e11_items,
        "e12_syntax_ctrl": e12_syntax_ctrl,
        "e12_f13_pragmatic": e12_f13_pragmatic,
        "e12_f25_homophones": e12_f25_homophones,
        "e13_pp_img_categories": e13_pp_img_categories,
        "e15_notarget_qids": e15_notarget_qids,
    }
    out_file = manifest_dir / "diagnostic_cohorts.json"
    out_file.write_text(json.dumps(cohorts, indent=2), encoding="utf-8")
    print(f"[COHORTS] Wrote {out_file} | counts={json.dumps(cohorts['summary_counts'], indent=2)}", flush=True)
    return cohorts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & freeze deterministic diagnostic probe cohorts")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_all_cohorts(args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
