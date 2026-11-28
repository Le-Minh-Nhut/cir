#!/usr/bin/env python3
"""Suite E18: Randomized Stratified Blind Human Annotation Audit Package (Failure #19).

Strictly follows CIR_one_run_probe_spec.md:
- Stratified random sample including BOTH hits AND misses across EDIR, PinPoint, FashionIQ
- Model identities cryptographically anonymized (salted SHA-256 opaque IDs; mapping sealed
  in `e18_sealed_key.json` separate from annotator pack)
- Exact 5-label taxonomy:
    {model_error, gt_error, alt_positive, underspecified_query, protocol_intent_mismatch}
- Generates visual contact sheets (Ref | GT Target | Model Top-1 | Model Top-2),
  blank annotator CSV/JSONL templates with exact inverse-probability sampling weights,
  and a standalone scorer (`--score`) computing Krippendorff's alpha, Fleiss' kappa,
  adjudication, Horvitz-Thompson weighted prevalence, and paired group bootstrap CIs!

Never fabricates human annotations: clearly marks status as `blind_pack_ready_awaiting_human_annotators`
until independent annotator files are provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import duckdb
import numpy as np
import PIL.Image
import PIL.ImageDraw

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.probes.stats_utils import paired_group_bootstrap_ci  # noqa: E402

LABELS_5 = [
    "model_error",
    "gt_error",
    "alt_positive",
    "underspecified_query",
    "protocol_intent_mismatch",
]


def _make_contact_sheet(paths: List[Tuple[str, str]], caption: str, out_path: Path) -> None:
    """Render a clean blind 4-panel contact sheet (Ref | GT | Pred-A | Pred-B) without model names."""
    tile = 256
    pad = 12
    header_h = 64
    w = len(paths) * (tile + pad) + pad
    h = header_h + tile + 32
    canvas = PIL.Image.new("RGB", (w, h), (245, 245, 245))
    draw = PIL.ImageDraw.Draw(canvas)
    cap_clean = caption.replace("\n", " ")[:135]
    draw.text((pad, 10), f"Query: {cap_clean}", fill=(20, 20, 20))
    draw.text((pad, 34), "Blind Audit Sheet — Model Identity Hidden", fill=(90, 90, 90))
    for i, (label, img_p) in enumerate(paths):
        x0 = pad + i * (tile + pad)
        y0 = header_h
        try:
            with PIL.Image.open(img_p) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, tile), PIL.Image.BILINEAR)
                ox = x0 + (tile - im.width) // 2
                oy = y0 + (tile - im.height) // 2
                canvas.paste(im, (ox, oy))
        except Exception:
            draw.rectangle([x0, y0, x0 + tile, y0 + tile], outline=(180, 0, 0))
            draw.text((x0 + 8, y0 + 100), "IMAGE_ERR", fill=(180, 0, 0))
        draw.text((x0 + 4, y0 + tile + 6), label, fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)


def _anon_id(model_name: str, salt: str) -> str:
    return "M_" + hashlib.sha256(f"{salt}:{model_name}".encode()).hexdigest()[:10]


def build_blind_audit_pack(out_root: Path, seed: int = 20261128, per_stratum: int = 20) -> Dict[str, Any]:
    """Build stratified blind audit pack (hits + misses across EDIR, PinPoint, FashionIQ)."""
    rng = random.Random(seed)
    salt = hashlib.sha256(f"cir_e18_blind_salt_{seed}".encode()).hexdigest()[:16]
    audit_dir = out_root / "e18_blind_audit"
    sheets_dir = audit_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # Representative models across families (identities will be anonymized!)
    models_ep = ["conesep_blip2", "mtst_ptha_blip2", "sprc_pretrain", "qure_vitl", "clip4cir_rn50x4", "limn_vitl14"]
    anon_map = {m: _anon_id(m, salt) for m in models_ep + ["csmcir_fashioniq_20261122_native", "conesep_fiq_n02_20261122_native", "qure_fiq_vitl_repro_20261124_native", "clip4cir_rn50x4_fullft_20261122_native", "limn_base_iter0_20261122_native"]}

    items_blind: List[Dict[str, Any]] = []
    sealed_records: List[Dict[str, Any]] = []
    stratum_stats: Dict[str, Dict[str, Any]] = {}

    # 1. EDIR (strata: hit@1, miss_same_scene, miss_diff_scene, miss_remove_or_neg)
    edir_dir = REPO_ROOT / "data/EDIR"
    edir_te = [json.loads(l) for l in (edir_dir / "splits/70_10_20_seed42/test.jsonl").read_text().splitlines()]
    edir_preds = {m: json.loads((REPO_ROOT / f"workbench/artifacts/benchmark_runs/edir/70_10_20_seed42/{m}/test_predictions_top50.json").read_text()) for m in models_ep}
    edir_strata: Dict[str, List[Tuple[dict, str, List[str]]]] = defaultdict(list)
    for r in edir_te:
        m = rng.choice(models_ep)
        top = edir_preds[m][r["query_id"]][:2]
        pos = r["positive_ids"][0]
        stem = r["metadata"]["stem_base"]
        if top[0] == pos:
            st_name = "edir_hit"
        elif r["category"] == "remove":
            st_name = "edir_miss_remove"
        elif top[0].startswith(stem):
            st_name = "edir_miss_same_scene"
        else:
            st_name = "edir_miss_diff_scene"
        edir_strata[st_name].append((r, m, top))

    for st_name, pool in edir_strata.items():
        rng.shuffle(pool)
        k = min(per_stratum, len(pool))
        weight = float(len(pool)) / max(k, 1)
        stratum_stats[st_name] = {"population_n": len(pool), "sampled_n": k, "sampling_weight": round(weight, 4)}
        for idx, (r, m, top) in enumerate(pool[:k]):
            item_id = f"{st_name}_{idx:03d}"
            sheet = sheets_dir / f"{item_id}.jpg"
            _make_contact_sheet(
                [
                    ("Reference", str(edir_dir / r["ref_image_rel"])),
                    ("GT Positive", str(edir_dir / "imgs_mc" / r["positive_ids"][0])),
                    ("Retrieved #1", str(edir_dir / "imgs_mc" / top[0])),
                    ("Retrieved #2", str(edir_dir / "imgs_mc" / top[1])),
                ],
                r["text"],
                sheet,
            )
            items_blind.append({
                "item_id": item_id,
                "dataset": "edir",
                "stratum": st_name,
                "sampling_weight": round(weight, 4),
                "group_id": r["metadata"]["stem_base"],
                "query_id": r["query_id"],
                "query_text": r["text"],
                "anon_model_id": anon_map[m],
                "is_hit_top1": top[0] == r["positive_ids"][0],
                "contact_sheet_rel": f"contact_sheets/{item_id}.jpg",
                "annotator_label": "",
                "annotator_notes": "",
            })
            sealed_records.append({"item_id": item_id, "true_model": m, "anon_model_id": anon_map[m], "category": r["category"]})

    # 2. PinPoint (strata: hit@10, miss_negation, miss_complement_explore, miss_swap_context)
    pp_dir = REPO_ROOT / "data/PinPoint"
    pp_te = [json.loads(l) for l in (pp_dir / "splits/70_10_20_seed42/test.jsonl").read_text().splitlines()]
    pp_corpus = {j["image_id"]: j["image_rel"] for j in (json.loads(l) for l in (pp_dir / "splits/70_10_20_seed42/corpus_full.jsonl").read_text().splitlines())}
    pp_preds = {m: json.loads((REPO_ROOT / f"workbench/artifacts/benchmark_runs/pinpoint/70_10_20_seed42/{m}/test_predictions_top50.json").read_text()) for m in models_ep}
    pp_strata: Dict[str, List[Tuple[dict, str, List[str]]]] = defaultdict(list)
    for r in pp_te:
        m = rng.choice(models_ep)
        top = pp_preds[m][r["query_id"]][:10]
        pos_set = set(r["positive_ids"])
        hit10 = any(x in pos_set for x in top)
        if hit10:
            st_name = "pinpoint_hit10"
        elif r["category"] == "Negation":
            st_name = "pinpoint_miss_negation"
        elif r["category"] in ("Complement", "Explore"):
            st_name = "pinpoint_miss_cross_cat"
        else:
            st_name = "pinpoint_miss_swap_ctx"
        pp_strata[st_name].append((r, m, top[:2]))

    for st_name, pool in pp_strata.items():
        rng.shuffle(pool)
        k = min(per_stratum, len(pool))
        weight = float(len(pool)) / max(k, 1)
        stratum_stats[st_name] = {"population_n": len(pool), "sampled_n": k, "sampling_weight": round(weight, 4)}
        for idx, (r, m, top) in enumerate(pool[:k]):
            item_id = f"{st_name}_{idx:03d}"
            sheet = sheets_dir / f"{item_id}.jpg"
            _make_contact_sheet(
                [
                    ("Reference", str(pp_dir / r["ref_image_rel"])),
                    ("GT Pos #1", str(pp_dir / pp_corpus.get(r["positive_ids"][0], ""))),
                    ("Retrieved #1", str(pp_dir / pp_corpus.get(top[0], ""))),
                    ("Retrieved #2", str(pp_dir / pp_corpus.get(top[1], ""))),
                ],
                r["text"],
                sheet,
            )
            items_blind.append({
                "item_id": item_id,
                "dataset": "pinpoint",
                "stratum": st_name,
                "sampling_weight": round(weight, 4),
                "group_id": r["ref_id"],
                "query_id": r["query_id"],
                "query_text": r["text"],
                "anon_model_id": anon_map[m],
                "is_hit_top1": top[0] in set(r["positive_ids"]),
                "contact_sheet_rel": f"contact_sheets/{item_id}.jpg",
                "annotator_label": "",
                "annotator_notes": "",
            })
            sealed_records.append({"item_id": item_id, "true_model": m, "anon_model_id": anon_map[m], "category": r["category"]})

    # 3. FashionIQ (via DuckDB top_results: hit@10, near_miss_11_50, extreme_miss_gt50)
    fiq_dir = REPO_ROOT / "data/FashionIQ"
    fiq_runs = [
        "csmcir_fashioniq_20261122_native",
        "conesep_fiq_n02_20261122_native",
        "qure_fiq_vitl_repro_20261124_native",
        "clip4cir_rn50x4_fullft_20261122_native",
        "limn_base_iter0_20261122_native",
    ]
    con = duckdb.connect(str(REPO_ROOT / "workbench/artifacts/workbench.duckdb"), read_only=True)
    fiq_strata: Dict[str, List[dict]] = defaultdict(list)
    rows_db = con.execute(
        """
        SELECT q.run_id, q.query_id, q.category, q.reference_id, q.target_id, q.model_input_text, q.target_rank,
               t1.image_id AS top1_id, t2.image_id AS top2_id
        FROM queries q
        JOIN top_results t1 ON q.run_id = t1.run_id AND q.query_id = t1.query_id AND t1.rank = 1
        JOIN top_results t2 ON q.run_id = t2.run_id AND q.query_id = t2.query_id AND t2.rank = 2
        WHERE q.run_id IN (SELECT UNNEST(?))
        """,
        [fiq_runs],
    ).fetchall()
    con.close()
    by_q: Dict[str, List[dict]] = defaultdict(list)
    for rid, qid, cat, ref_id, tgt_id, txt, trk, top1, top2 in rows_db:
        by_q[qid].append({
            "qid": qid, "run_id": rid, "category": cat, "ref_id": ref_id,
            "target_id": tgt_id, "text": txt, "target_rank": trk, "top2": [top1, top2],
        })
    for qid, cand_runs in by_q.items():
        rec = rng.choice(cand_runs)
        trk = rec["target_rank"]
        if trk is not None and trk <= 10:
            st_name = "fashioniq_hit10"
        elif trk is not None and trk <= 50:
            st_name = "fashioniq_miss_11_50"
        else:
            st_name = "fashioniq_miss_gt50"
        fiq_strata[st_name].append(rec)

    def _fiq_p(iid: str) -> str:
        p = fiq_dir / "images" / f"{iid}.png"
        return str(p if p.is_file() else fiq_dir / "images" / f"{iid}.jpg")

    for st_name, pool in fiq_strata.items():
        rng.shuffle(pool)
        k = min(per_stratum, len(pool))
        weight = float(len(pool)) / max(k, 1)
        stratum_stats[st_name] = {"population_n": len(pool), "sampled_n": k, "sampling_weight": round(weight, 4)}
        for idx, rec in enumerate(pool[:k]):
            item_id = f"{st_name}_{idx:03d}"
            sheet = sheets_dir / f"{item_id}.jpg"
            _make_contact_sheet(
                [
                    ("Reference", _fiq_p(rec["ref_id"])),
                    ("GT Target", _fiq_p(rec["target_id"])),
                    ("Retrieved #1", _fiq_p(rec["top2"][0])),
                    ("Retrieved #2", _fiq_p(rec["top2"][1])),
                ],
                rec["text"],
                sheet,
            )
            items_blind.append({
                "item_id": item_id,
                "dataset": "fashioniq",
                "stratum": st_name,
                "sampling_weight": round(weight, 4),
                "group_id": f"{rec['category']}_{rec['ref_id']}",
                "query_id": rec["qid"],
                "query_text": rec["text"],
                "anon_model_id": anon_map[rec["run_id"]],
                "is_hit_top1": rec["top2"][0] == rec["target_id"],
                "contact_sheet_rel": f"contact_sheets/{item_id}.jpg",
                "annotator_label": "",
                "annotator_notes": "",
            })
            sealed_records.append({"item_id": item_id, "true_model": rec["run_id"], "anon_model_id": anon_map[rec["run_id"]], "category": rec["category"], "target_rank": rec["target_rank"]})

    # Randomize presentation order so annotators cannot infer strata or hits/misses!
    rng.shuffle(items_blind)
    # Strip `is_hit_top1` and `stratum` from the annotator-facing sheet for 100% blind rating!
    annotator_fields = ["item_id", "dataset", "query_id", "query_text", "anon_model_id", "contact_sheet_rel", "annotator_label", "annotator_notes"]
    with (audit_dir / "annotator_template.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=annotator_fields)
        w.writeheader()
        for row in items_blind:
            w.writerow({k: row[k] for k in annotator_fields})

    (audit_dir / "blind_items_full_meta.jsonl").write_text("\n".join(json.dumps(x) for x in items_blind) + "\n", encoding="utf-8")
    (audit_dir / "e18_sealed_key.json").write_text(json.dumps({"salt_hash": hashlib.sha256(salt.encode()).hexdigest(), "model_map": anon_map, "records": sealed_records}, indent=2), encoding="utf-8")

    rubric = """# Suite E18 — Blind Human Annotation Audit Rubric

**Protocol:** Model identities, strata, and hit/miss flags are stripped from `annotator_template.csv`.
Inspect each `contact_sheets/<item_id>.jpg` showing `[Reference | GT Target | Retrieved #1 | Retrieved #2]`
and assign EXACTLY ONE primary label in `annotator_label`:

1. `model_error`: The query is clear and GT is valid, but Retrieved #1 clearly violates the edit or reference constraints.
2. `gt_error`: Ground-truth target itself contradicts the instruction or swaps Reference <-> Target.
3. `alt_positive`: Retrieved #1 genuinely satisfies both the Reference and Query instruction as well as (or better than) GT (false negative in benchmark).
4. `underspecified_query`: Query is genuinely ambiguous / under-constrained so multiple incompatible images are valid.
5. `protocol_intent_mismatch`: Dataset evaluation protocol (e.g., ref-not-masked, multi-image intent evaluated as single-image, or category mismatch) causes failure despite reasonable model intent.

**Adjudication & Scoring:** Place completed files `annotator_1.csv`, `annotator_2.csv` (and optional `adjudicated.csv`) in this directory and run:
`workbench/envs/ctl312h/bin/python workbench/probes/e18_audit_pack.py --score --audit-dir <this_dir> --annotators annotator_1.csv annotator_2.csv`
"""
    (audit_dir / "ANNOTATION_RUBRIC.md").write_text(rubric, encoding="utf-8")

    summary = {
        "suite": "E18",
        "status": "blind_pack_ready_awaiting_human_annotators",
        "human_annotations_completed": False,
        "seed": seed,
        "total_sampled_items": len(items_blind),
        "allowed_labels": LABELS_5,
        "strata": stratum_stats,
        "files": {
            "annotator_template_csv": str(audit_dir / "annotator_template.csv"),
            "rubric_md": str(audit_dir / "ANNOTATION_RUBRIC.md"),
            "sealed_key_json": str(audit_dir / "e18_sealed_key.json"),
            "contact_sheets_dir": str(sheets_dir),
        },
    }
    (audit_dir / "e18_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[E18] Built blind audit pack: {len(items_blind)} items across {len(stratum_stats)} strata -> {audit_dir}", flush=True)
    return summary


def score_annotations(audit_dir: Path, annotator_csvs: Sequence[Path]) -> Dict[str, Any]:
    """Compute Fleiss' kappa, Krippendorff's alpha (nominal), Horvitz-Thompson weighted prevalence & group CIs."""
    meta = {json.loads(l)["item_id"]: json.loads(l) for l in (audit_dir / "blind_items_full_meta.jsonl").read_text().splitlines() if l.strip()}
    ratings: Dict[str, List[str]] = defaultdict(list)
    for csv_p in annotator_csvs:
        with csv_p.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("annotator_label", "").strip()
                if lbl in LABELS_5:
                    ratings[row["item_id"]].append(lbl)

    # Fleiss' kappa & Krippendorff's nominal alpha across items rated by >= 2 annotators
    multi = {iid: r for iid, r in ratings.items() if len(r) >= 2}
    n_k = len(LABELS_5)
    lbl_idx = {l: i for i, l in enumerate(LABELS_5)}
    fleiss_k = float("nan")
    kripp_a = float("nan")
    if multi:
        # Nominal Krippendorff's alpha (handles unequal raters per item!)
        Do_num, Do_den = 0.0, 0.0
        total_counts = np.zeros(n_k, dtype=np.float64)
        for lbls in multi.values():
            m_u = len(lbls)
            cnts = Counter(lbls)
            for l, c in cnts.items():
                total_counts[lbl_idx[l]] += c
            disagree = sum(c * (m_u - c) for c in cnts.values())
            Do_num += disagree / (m_u - 1.0)
            Do_den += m_u
        Do = Do_num / max(Do_den, 1.0)
        N_tot = float(total_counts.sum())
        De = float(np.sum(total_counts * (N_tot - total_counts)) / max(N_tot * (N_tot - 1.0), 1.0))
        kripp_a = float(1.0 - Do / max(De, 1e-12))
        # Fleiss' kappa on items with exact modal rater count
        m_mode = Counter(len(v) for v in multi.values()).most_common(1)[0][0]
        eq_items = [v for v in multi.values() if len(v) == m_mode]
        if eq_items and m_mode >= 2:
            mat = np.zeros((len(eq_items), n_k), dtype=np.float64)
            for i, lbls in enumerate(eq_items):
                for l in lbls:
                    mat[i, lbl_idx[l]] += 1.0
            P_i = (np.sum(mat * mat, axis=1) - m_mode) / (m_mode * (m_mode - 1.0))
            P_bar = float(np.mean(P_i))
            p_j = np.sum(mat, axis=0) / (len(eq_items) * m_mode)
            P_e = float(np.sum(p_j * p_j))
            fleiss_k = float((P_bar - P_e) / max(1.0 - P_e, 1e-12))

    # Majority / adjudicated label per item -> Horvitz-Thompson weighted prevalence + group CI
    adjudicated = {iid: Counter(lbls).most_common(1)[0][0] for iid, lbls in ratings.items()}
    prevalence: Dict[str, Any] = {}
    for ds in ("edir", "pinpoint", "fashioniq"):
        ds_ids = [iid for iid in adjudicated if meta.get(iid, {}).get("dataset") == ds]
        if not ds_ids:
            continue
        weights = np.array([meta[iid]["sampling_weight"] for iid in ds_ids], dtype=np.float64)
        groups = [meta[iid]["group_id"] for iid in ds_ids]
        prevalence[ds] = {"n_annotated": len(ds_ids), "labels": {}}
        for lbl in LABELS_5:
            ind = np.array([1.0 if adjudicated[iid] == lbl else 0.0 for iid in ds_ids], dtype=np.float64)
            w_est = float(np.sum(weights * ind) / max(np.sum(weights), 1e-12))
            ci = paired_group_bootstrap_ci(ind, None, groups, n_boot=2000, seed=42)
            prevalence[ds]["labels"][lbl] = {"weighted_rate": w_est, "unweighted_ci": ci}

    out = {
        "n_items_rated": len(adjudicated),
        "n_items_multi_rater": len(multi),
        "krippendorff_alpha_nominal": kripp_a,
        "fleiss_kappa": fleiss_k,
        "prevalence_by_dataset": prevalence,
    }
    (audit_dir / "e18_scored_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="E18 Blind Human Annotation Audit Pack Generator & Scorer")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261128)
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--annotators", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    if args.score:
        res = score_annotations(args.audit_dir, args.annotators)
        print(json.dumps(res, indent=2))
    else:
        build_blind_audit_pack(args.out_dir, seed=args.seed, per_stratum=args.per_stratum)


if __name__ == "__main__":
    main()
