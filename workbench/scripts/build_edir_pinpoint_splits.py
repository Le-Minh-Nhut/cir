#!/usr/bin/env python3
"""Deterministic, leakage-audited 70/10/20 train/val/test split & canonical export builder
for EDIR and PinPoint.

Exports:
  1) Versioned ID manifests & plain-text ID lists (tracked in git):
     workbench/docs/reviews/20261123-edir-pinpoint-splits/manifests/
  2) Self-contained JSONL split bundles + SHA256 manifest (zero external deps to read):
     data/EDIR/splits/70_10_20_seed42/
     data/PinPoint/splits/70_10_20_seed42/
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb

REPO_ROOT = Path("/home/heheboiz/data/cir")
EDIR_DIR = REPO_ROOT / "data" / "EDIR"
PINPOINT_DIR = REPO_ROOT / "data" / "PinPoint"
TRACKED_MANIFEST_DIR = (
    REPO_ROOT / "workbench" / "docs" / "reviews" / "20261123-edir-pinpoint-splits" / "manifests"
)

SEED = 42
SPLIT_TAG = "70_10_20_seed42"
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


def write_json(path: Path, payload: Dict[str, Any]) -> str:
    return write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sha256_file(path)


def write_id_list(path: Path, ids: Sequence[str]) -> str:
    return write_text(path, "\n".join(str(x) for x in ids) + ("\n" if ids else ""))


def stem_base(img_name: str) -> str:
    s = Path(img_name).stem
    parts = s.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return s


def normalize_pinpoint_qid(qid: str) -> str:
    if qid.startswith("query_"):
        return qid[6:].zfill(5)
    return str(qid).zfill(5)


def build_and_export_edir(seed: int = SEED) -> Dict[str, Any]:
    img_dir = EDIR_DIR / "imgs_mc"
    queries = [json.loads(l) for l in (EDIR_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = [json.loads(l) for l in (EDIR_DIR / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    instances = [json.loads(l) for l in (EDIR_DIR / "instances.jsonl").read_text(encoding="utf-8").splitlines()]
    inst_by_qid = {x["qid"]: x for x in instances}
    q_by_id = {q["id"]: q for q in queries}

    # Verify every image exists on disk
    for q in queries:
        assert (img_dir / q["image"]).is_file(), f"Missing EDIR ref image: {q['image']}"
    for c in corpus:
        assert (img_dir / c["image"]).is_file(), f"Missing EDIR corpus image: {c['image']}"

    # Group queries by stem_base (exact connected components of ref + pos + neg)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for q in queries:
        groups[stem_base(q["image"])].append(q)

    rng = random.Random(seed)
    by_subcat: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = defaultdict(list)
    for base, qs in sorted(groups.items()):
        subcat = Counter(q["metadata"]["subcategory"] for q in qs).most_common(1)[0][0]
        by_subcat[subcat].append((base, qs))

    split_qids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_bases: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for subcat in sorted(by_subcat.keys()):
        items = list(by_subcat[subcat])
        rng.shuffle(items)
        total_q = sum(len(qs) for _, qs in items)
        targets = {
            "train": round(total_q * RATIOS["train"]),
            "val": round(total_q * RATIOS["val"]),
            "test": total_q - round(total_q * RATIOS["train"]) - round(total_q * RATIOS["val"]),
        }
        counts = {"train": 0, "val": 0, "test": 0}
        for base, qs in items:
            best_sp = max(
                ("train", "val", "test"),
                key=lambda sp: (targets[sp] - counts[sp]) / max(targets[sp], 1),
            )
            counts[best_sp] += len(qs)
            split_bases[best_sp].append(base)
            for q in qs:
                split_qids[best_sp].append(q["id"])

    for sp in ("train", "val", "test"):
        split_qids[sp].sort(key=lambda x: int(x))
        split_bases[sp].sort()

    # Verify strict 0% leakage across all dimensions
    sets = {}
    for sp in ("train", "val", "test"):
        refs, poss, negs, bases = set(), set(), set(), set()
        for qid in split_qids[sp]:
            q = q_by_id[qid]
            ins = inst_by_qid[qid]
            refs.add(q["image"])
            bases.add(stem_base(q["image"]))
            for p in ins["pos"]:
                poss.add(p)
                bases.add(stem_base(p))
            for n in ins["neg"]:
                negs.add(n)
                bases.add(stem_base(n))
        sets[sp] = {"ref": refs, "pos": poss, "neg": negs, "base": bases}

    leakage = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leakage[f"{a}_vs_{b}"] = {k: len(sets[a][k] & sets[b][k]) for k in ("ref", "pos", "neg", "base")}
        assert all(v == 0 for v in leakage[f"{a}_vs_{b}"].values()), f"EDIR leak: {leakage}"

    # Export canonical JSONL files under data/EDIR/splits/70_10_20_seed42/
    split_dir = EDIR_DIR / "splits" / SPLIT_TAG
    split_dir.mkdir(parents=True, exist_ok=True)
    file_sha256: Dict[str, str] = {}

    corpus_full_recs = [
        {"image_id": c["image"], "image_rel": f"imgs_mc/{c['image']}"}
        for c in corpus
    ]
    file_sha256["corpus_full.jsonl"] = write_jsonl(split_dir / "corpus_full.jsonl", corpus_full_recs)

    split_corpus_counts: Dict[str, int] = {}
    for sp in ("train", "val", "test"):
        q_records = []
        sp_cand_set = set()
        for qid in split_qids[sp]:
            q = q_by_id[qid]
            ins = inst_by_qid[qid]
            pos_list = list(ins["pos"])
            neg_list = [n for n in ins["neg"] if n not in set(pos_list)]
            sp_cand_set.update(pos_list)
            sp_cand_set.update(neg_list)
            q_records.append(
                {
                    "query_id": str(qid),
                    "normalized_query_id": str(qid),
                    "ref_id": q["image"],
                    "ref_image_rel": f"imgs_mc/{q['image']}",
                    "ref2_id": None,
                    "ref2_image_rel": None,
                    "is_single": True,
                    "text": q["text"],
                    "category": q["metadata"]["subcategory"],
                    "positive_ids": pos_list,
                    "negative_ids": neg_list,
                    "positive_ids_raw": list(ins["pos"]),
                    "negative_ids_raw": list(ins["neg"]),
                    "metadata": {
                        "stem_base": stem_base(q["image"]),
                        "subcategory": q["metadata"]["subcategory"],
                    },
                }
            )
        file_sha256[f"{sp}.jsonl"] = write_jsonl(split_dir / f"{sp}.jsonl", q_records)
        file_sha256[f"{sp}_ids.txt"] = write_id_list(split_dir / f"{sp}_ids.txt", split_qids[sp])
        write_id_list(TRACKED_MANIFEST_DIR / f"edir_{sp}_ids.txt", split_qids[sp])

        sp_corpus_recs = [c for c in corpus_full_recs if c["image_id"] in sp_cand_set]
        split_corpus_counts[sp] = len(sp_corpus_recs)
        file_sha256[f"corpus_{sp}.jsonl"] = write_jsonl(split_dir / f"corpus_{sp}.jsonl", sp_corpus_recs)

    subcat_table = {
        sp: dict(sorted(Counter(q_by_id[qid]["metadata"]["subcategory"] for qid in split_qids[sp]).items()))
        for sp in ("train", "val", "test")
    }

    manifest = {
        "dataset": "EDIR",
        "source": "EDIR-BENCH/EDIR",
        "split_tag": SPLIT_TAG,
        "seed": seed,
        "ratios": RATIOS,
        "strategy": "group_by_stem_base_stratified_by_15_subcategories",
        "counts": {sp: len(split_qids[sp]) for sp in ("train", "val", "test")},
        "group_counts": {sp: len(split_bases[sp]) for sp in ("train", "val", "test")},
        "corpus_counts": {"full": len(corpus_full_recs), **split_corpus_counts},
        "leakage_audit": leakage,
        "subcategory_distribution": subcat_table,
        "file_sha256": file_sha256,
        "splits": split_qids,
    }
    write_json(split_dir / "manifest.json", manifest)
    write_json(TRACKED_MANIFEST_DIR / f"edir_{SPLIT_TAG}.json", manifest)

    # Remove superseded top-level file if present
    legacy = EDIR_DIR / f"splits_{SPLIT_TAG}.json"
    if legacy.exists():
        legacy.unlink()

    return manifest


def build_and_export_pinpoint(seed: int = SEED) -> Dict[str, Any]:
    img_root = PINPOINT_DIR / "images"
    idx_lines = [
        l.strip()
        for l in (PINPOINT_DIR / "index_signatures.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    def img_rel(sig: str) -> str:
        return f"images/{sig[:2]}/{sig}.jpg"

    def img_exists(sig: str) -> bool:
        return bool(sig) and len(sig) >= 6 and (PINPOINT_DIR / img_rel(sig)).is_file()

    existing_corpus_sigs = [s for s in idx_lines if img_exists(s)]
    missing_corpus_sigs = [s for s in idx_lines if not img_exists(s)]
    existing_set = set(existing_corpus_sigs)

    con = duckdb.connect()
    pq = str(PINPOINT_DIR / "pinpoint_licensed.parquet")
    rows = con.execute(
        f"""
        SELECT query_id, query_image_signature, query_image_signature2,
               is_single, query_category, l1_interest,
               instruction, length_category, is_question, is_imperative,
               skin_tone_bucket, positive_candidates, negative_candidates
        FROM read_parquet('{pq}')
        ORDER BY query_id
        """
    ).fetchall()

    single_rows = [r for r in rows if r[3]]
    multi_rows = [r for r in rows if not r[3]]

    # Group single-image queries by query_image_signature
    groups: Dict[str, List[tuple]] = defaultdict(list)
    for r in single_rows:
        groups[r[1]].append(r)

    hub_refs = sorted([sig for sig, rs in groups.items() if len(rs) >= 50], key=lambda s: -len(groups[s]))
    non_hub_refs = sorted([sig for sig in groups if sig not in hub_refs])

    total_n = len(single_rows)
    targets = {
        "train": round(total_n * RATIOS["train"]),
        "val": round(total_n * RATIOS["val"]),
        "test": total_n - round(total_n * RATIOS["train"]) - round(total_n * RATIOS["val"]),
    }

    cat_total = Counter(r[4] for r in single_rows)
    cat_targets = {sp: {c: cat_total[c] * RATIOS[sp] for c in cat_total} for sp in ("train", "val", "test")}

    split_refs: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_cat_counts = {sp: Counter() for sp in ("train", "val", "test")}
    split_pos_sets = {sp: set() for sp in ("train", "val", "test")}

    def add_group(sp: str, sig: str) -> None:
        rs = groups[sig]
        split_refs[sp].append(sig)
        split_counts[sp] += len(rs)
        for r in rs:
            split_cat_counts[sp][r[4]] += 1
            split_pos_sets[sp].update(p for p in r[11] if p)

    for sig in hub_refs:
        add_group("train", sig)

    rng = random.Random(seed)
    rng.shuffle(non_hub_refs)
    group_pos = {sig: {p for r in rs for p in r[11] if p} for sig, rs in groups.items()}
    group_cats = {sig: Counter(r[4] for r in rs) for sig, rs in groups.items()}

    for sig in non_hub_refs:
        n_g = len(groups[sig])
        g_pos = group_pos[sig]
        g_cats = group_cats[sig]
        best_sp = None
        best_score = -1e18
        for sp in ("train", "val", "test"):
            rem_ratio = (targets[sp] - split_counts[sp]) / max(targets[sp], 1)
            cat_bonus = sum(
                (cat_targets[sp][c] - split_cat_counts[sp][c]) / max(cat_targets[sp][c], 1.0) * cnt
                for c, cnt in g_cats.items()
            ) / max(n_g, 1)
            other_pos = set()
            for osp in ("train", "val", "test"):
                if osp != sp:
                    other_pos |= split_pos_sets[osp]
            overlap_frac = (len(g_pos & other_pos) / max(len(g_pos), 1)) if g_pos else 0.0
            score = 3.0 * rem_ratio + 1.5 * cat_bonus - 0.6 * overlap_frac
            if score > best_score:
                best_score = score
                best_sp = sp
        assert best_sp is not None
        add_group(best_sp, sig)

    split_qids: Dict[str, List[str]] = {}
    for sp in ("train", "val", "test"):
        qids = [r[0] for sig in split_refs[sp] for r in groups[sig]]
        qids.sort()
        split_qids[sp] = qids

    multi_qids = sorted(r[0] for r in multi_rows)

    row_by_qid = {r[0]: r for r in rows}

    # Clean candidate helper (preserves deterministic order, strips None/dups/missing/pos-neg conflicts)
    clean_stats = {
        "null_neg_removed": 0,
        "dup_pos_removed": 0,
        "dup_neg_removed": 0,
        "pos_neg_conflict_removed": 0,
        "missing_disk_neg_removed": 0,
        "single_with_nonnull_q2_count": 0,
    }

    def format_record(r: tuple) -> Dict[str, Any]:
        (
            qid, q1, q2_raw, is_single, cat, l1,
            instr, len_cat, is_q, is_imp, skin, pos_raw, neg_raw,
        ) = r
        assert img_exists(q1), f"Missing PinPoint q1 image for {qid}: {q1}"
        # Preserve raw q2 string for bit-exact linguistic-sensitivity grouping in evaluate.py
        has_valid_q2 = bool(q2_raw) and str(q2_raw).lower() != "none"
        if is_single and has_valid_q2:
            clean_stats["single_with_nonnull_q2_count"] += 1
        ref2_id = q2_raw if (not is_single and has_valid_q2) else None
        if ref2_id is not None:
            assert img_exists(ref2_id), f"Missing PinPoint q2 image for {qid}: {ref2_id}"

        pos_clean: List[str] = []
        pos_seen = set()
        for p in (pos_raw or []):
            if not p:
                continue
            if p in pos_seen:
                clean_stats["dup_pos_removed"] += 1
                continue
            assert p in existing_set, f"Missing PinPoint pos image on disk: {p}"
            pos_seen.add(p)
            pos_clean.append(p)

        neg_clean: List[str] = []
        neg_seen = set()
        for n in (neg_raw or []):
            if not n:
                clean_stats["null_neg_removed"] += 1
                continue
            if n in pos_seen:
                clean_stats["pos_neg_conflict_removed"] += 1
                continue
            if n in neg_seen:
                clean_stats["dup_neg_removed"] += 1
                continue
            if n not in existing_set:
                clean_stats["missing_disk_neg_removed"] += 1
                continue
            neg_seen.add(n)
            neg_clean.append(n)

        return {
            "query_id": str(qid),
            "normalized_query_id": normalize_pinpoint_qid(str(qid)),
            "ref_id": str(q1),
            "ref_image_rel": img_rel(str(q1)),
            "ref2_id": ref2_id,
            "ref2_image_rel": img_rel(ref2_id) if ref2_id else None,
            "is_single": bool(is_single),
            "text": str(instr),
            "category": str(cat),
            "positive_ids": pos_clean,
            "negative_ids": neg_clean,
            "positive_ids_raw": list(pos_raw or []),
            "negative_ids_raw": list(neg_raw or []),
            "metadata": {
                "query_category": str(cat),
                "l1_interest": str(l1),
                "length_category": str(len_cat),
                "is_question": bool(is_q),
                "is_imperative": bool(is_imp),
                "skin_tone_bucket": str(skin),
                "query_image_signature2_raw": q2_raw if has_valid_q2 else None,
            },
        }

    # Export canonical JSONL files under data/PinPoint/splits/70_10_20_seed42/
    split_dir = PINPOINT_DIR / "splits" / SPLIT_TAG
    split_dir.mkdir(parents=True, exist_ok=True)
    file_sha256: Dict[str, str] = {}

    corpus_full_recs = [
        {"image_id": sig, "image_rel": img_rel(sig)}
        for sig in existing_corpus_sigs
    ]
    file_sha256["corpus_full.jsonl"] = write_jsonl(split_dir / "corpus_full.jsonl", corpus_full_recs)

    all_splits_map = {**split_qids, "multi_image_eval": multi_qids}
    split_corpus_counts: Dict[str, int] = {}
    exported_records: Dict[str, List[Dict[str, Any]]] = {}

    for sp_name, qids in all_splits_map.items():
        recs = [format_record(row_by_qid[qid]) for qid in qids]
        exported_records[sp_name] = recs
        file_sha256[f"{sp_name}.jsonl"] = write_jsonl(split_dir / f"{sp_name}.jsonl", recs)
        file_sha256[f"{sp_name}_ids.txt"] = write_id_list(split_dir / f"{sp_name}_ids.txt", qids)
        write_id_list(TRACKED_MANIFEST_DIR / f"pinpoint_{sp_name}_ids.txt", qids)

        sp_img_set = set()
        for rec in recs:
            sp_img_set.add(rec["ref_id"])
            if rec["ref2_id"]:
                sp_img_set.add(rec["ref2_id"])
            sp_img_set.update(rec["positive_ids"])
            sp_img_set.update(rec["negative_ids"])
        sp_corpus_recs = [c for c in corpus_full_recs if c["image_id"] in sp_img_set]
        split_corpus_counts[sp_name] = len(sp_corpus_recs)
        file_sha256[f"corpus_{sp_name}.jsonl"] = write_jsonl(split_dir / f"corpus_{sp_name}.jsonl", sp_corpus_recs)

    # Leakage audit on cleaned sets
    sets = {}
    for sp in ("train", "val", "test"):
        refs = {rec["ref_id"] for rec in exported_records[sp]}
        poss = {p for rec in exported_records[sp] for p in rec["positive_ids"]}
        negs = {n for rec in exported_records[sp] for n in rec["negative_ids"]}
        sets[sp] = {"ref": refs, "pos": poss, "neg": negs}

    leakage = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leakage[f"{a}_vs_{b}"] = {
            "ref_overlap": len(sets[a]["ref"] & sets[b]["ref"]),
            "pos_overlap": len(sets[a]["pos"] & sets[b]["pos"]),
            "pos_overlap_pct_of_b": round(100.0 * len(sets[a]["pos"] & sets[b]["pos"]) / max(len(sets[b]["pos"]), 1), 2),
            "neg_overlap": len(sets[a]["neg"] & sets[b]["neg"]),
            "neg_overlap_pct_of_b": round(100.0 * len(sets[a]["neg"] & sets[b]["neg"]) / max(len(sets[b]["neg"]), 1), 2),
        }
        assert leakage[f"{a}_vs_{b}"]["ref_overlap"] == 0, "PinPoint ref leak!"

    missing_set = set(missing_corpus_sigs)
    missing_by_split = {
        sp: sum(
            1 for qid in qids
            if any(m in (row_by_qid[qid][12] or []) for m in missing_set)
        )
        for sp, qids in all_splits_map.items()
    }

    manifest = {
        "dataset": "PinPoint",
        "source": "pinpoint_licensed.parquet",
        "split_tag": SPLIT_TAG,
        "seed": seed,
        "ratios": RATIOS,
        "strategy": "single_image_group_by_ref_pin_top5_hubs_to_train_stratified_min_pos_overlap",
        "pinned_hub_refs_in_train": {s: len(groups[s]) for s in hub_refs},
        "counts": {sp: len(split_qids[sp]) for sp in ("train", "val", "test")},
        "unique_ref_counts": {sp: len(split_refs[sp]) for sp in ("train", "val", "test")},
        "multi_image_eval_count": len(multi_qids),
        "corpus_counts": {
            "index_signatures_total": len(idx_lines),
            "full_existing": len(corpus_full_recs),
            "missing_403_excluded": missing_corpus_sigs,
            **split_corpus_counts,
        },
        "category_distribution": {sp: dict(sorted(split_cat_counts[sp].items())) for sp in ("train", "val", "test")},
        "cleaning_audit": clean_stats,
        "leakage_audit": leakage,
        "missing_403_neg_affected_queries": missing_by_split,
        "file_sha256": file_sha256,
        "splits": split_qids,
        "multi_image_eval": multi_qids,
    }
    write_json(split_dir / "manifest.json", manifest)
    write_json(TRACKED_MANIFEST_DIR / f"pinpoint_{SPLIT_TAG}.json", manifest)

    legacy = PINPOINT_DIR / f"splits_{SPLIT_TAG}.json"
    if legacy.exists():
        legacy.unlink()

    return manifest


def main() -> None:
    edir_m = build_and_export_edir(SEED)
    pp_m = build_and_export_pinpoint(SEED)

    print("=== EDIR EXPORTED ===")
    print("Counts:", edir_m["counts"], "Corpus:", edir_m["corpus_counts"])
    print("Leakage:", json.dumps(edir_m["leakage_audit"]))
    print("Files:", sorted(edir_m["file_sha256"].keys()))

    print("\n=== PINPOINT EXPORTED ===")
    print("Counts:", pp_m["counts"], "Multi-img:", pp_m["multi_image_eval_count"])
    print("Corpus:", pp_m["corpus_counts"])
    print("Cleaning:", json.dumps(pp_m["cleaning_audit"]))
    print("Leakage:", json.dumps(pp_m["leakage_audit"]))
    print("Missing 403 affected:", pp_m["missing_403_neg_affected_queries"])
    print("Files:", sorted(pp_m["file_sha256"].keys()))


if __name__ == "__main__":
    main()
