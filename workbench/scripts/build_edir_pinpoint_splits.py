#!/usr/bin/env python3
"""Deterministic, leakage-audited 70/10/20 train/val/test split generator for EDIR and PinPoint."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb

REPO_ROOT = Path("/home/heheboiz/data/cir")
EDIR_DIR = REPO_ROOT / "data" / "EDIR"
PINPOINT_DIR = REPO_ROOT / "data" / "PinPoint"
OUT_DIR = REPO_ROOT / "workbench" / "docs" / "reviews" / "20261123-edir-pinpoint-splits" / "manifests"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.10, "test": 0.20}


def stem_base(img_name: str) -> str:
    s = Path(img_name).stem
    parts = s.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return s


def build_edir_split(seed: int = SEED) -> dict[str, Any]:
    queries = [json.loads(l) for l in (EDIR_DIR / "queries.jsonl").read_text().splitlines()]
    instances = [json.loads(l) for l in (EDIR_DIR / "instances.jsonl").read_text().splitlines()]
    inst_by_qid = {x["qid"]: x for x in instances}

    # Group queries by stem_base (which equals the exact connected components of ref+pos+neg)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        groups[stem_base(q["image"])].append(q)

    # Assign each group a primary stratification key = tuple of sorted subcategories (usually length 1)
    rng = random.Random(seed)
    by_subcat: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for base, qs in sorted(groups.items()):
        # Majority (or first) subcategory for stratification
        subcat = Counter(q["metadata"]["subcategory"] for q in qs).most_common(1)[0][0]
        by_subcat[subcat].append((base, qs))

    split_qids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_bases: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for subcat in sorted(by_subcat.keys()):
        items = list(by_subcat[subcat])
        rng.shuffle(items)
        # Sort multi-query groups first slightly randomized so counts hit targets closely
        total_q = sum(len(qs) for _, qs in items)
        targets = {
            "train": round(total_q * RATIOS["train"]),
            "val": round(total_q * RATIOS["val"]),
            "test": total_q - round(total_q * RATIOS["train"]) - round(total_q * RATIOS["val"]),
        }
        counts = {"train": 0, "val": 0, "test": 0}
        for base, qs in items:
            # Pick split with largest remaining fractional deficit
            best_sp = max(
                ("train", "val", "test"),
                key=lambda sp: (targets[sp] - counts[sp]) / max(targets[sp], 1),
            )
            counts[best_sp] += len(qs)
            split_bases[best_sp].append(base)
            for q in qs:
                split_qids[best_sp].append(q["id"])

    # Sort IDs cleanly
    for sp in ("train", "val", "test"):
        split_qids[sp].sort(key=lambda x: int(x))
        split_bases[sp].sort()

    # Leakage verification
    q_by_id = {q["id"]: q for q in queries}
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
        leakage[f"{a}_vs_{b}"] = {
            k: len(sets[a][k] & sets[b][k]) for k in ("ref", "pos", "neg", "base")
        }
        assert all(v == 0 for v in leakage[f"{a}_vs_{b}"].values()), f"EDIR leak detected: {leakage}"

    subcat_table = {}
    for sp in ("train", "val", "test"):
        subcat_table[sp] = dict(sorted(Counter(q_by_id[qid]["metadata"]["subcategory"] for qid in split_qids[sp]).items()))

    return {
        "dataset": "EDIR",
        "source": "EDIR-BENCH/EDIR",
        "seed": seed,
        "ratios": RATIOS,
        "strategy": "group_by_stem_base_stratified_by_15_subcategories",
        "counts": {sp: len(split_qids[sp]) for sp in ("train", "val", "test")},
        "group_counts": {sp: len(split_bases[sp]) for sp in ("train", "val", "test")},
        "leakage_audit": leakage,
        "subcategory_distribution": subcat_table,
        "splits": split_qids,
    }


def build_pinpoint_split(seed: int = SEED) -> dict[str, Any]:
    con = duckdb.connect()
    pq = str(PINPOINT_DIR / "pinpoint_licensed.parquet")
    rows = con.execute(
        f"""
        SELECT query_id, query_image_signature, query_image_signature2,
               is_single, query_category, l1_interest,
               positive_candidates, negative_candidates
        FROM read_parquet('{pq}')
        ORDER BY query_id
        """
    ).fetchall()

    single_rows = [r for r in rows if r[3]]
    multi_rows = [r for r in rows if not r[3]]

    # Group single-image queries by query_image_signature
    groups: dict[str, list[tuple]] = defaultdict(list)
    for r in single_rows:
        groups[r[1]].append(r)

    # Identify top-5 giant hub refs (>= 50 queries each) and pin them to train
    hub_refs = sorted([sig for sig, rs in groups.items() if len(rs) >= 50], key=lambda s: -len(groups[s]))
    non_hub_refs = sorted([sig for sig in groups if sig not in hub_refs])

    total_n = len(single_rows)
    targets = {
        "train": round(total_n * RATIOS["train"]),
        "val": round(total_n * RATIOS["val"]),
        "test": total_n - round(total_n * RATIOS["train"]) - round(total_n * RATIOS["val"]),
    }

    # Global category targets
    cat_total = Counter(r[4] for r in single_rows)
    cat_targets = {
        sp: {c: cat_total[c] * RATIOS[sp] for c in cat_total}
        for sp in ("train", "val", "test")
    }

    split_refs: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_cat_counts = {sp: Counter() for sp in ("train", "val", "test")}
    split_pos_sets = {sp: set() for sp in ("train", "val", "test")}

    def add_group(sp: str, sig: str) -> None:
        rs = groups[sig]
        split_refs[sp].append(sig)
        split_counts[sp] += len(rs)
        for r in rs:
            split_cat_counts[sp][r[4]] += 1
            split_pos_sets[sp].update(r[6])

    for sig in hub_refs:
        add_group("train", sig)

    # Deterministic shuffle of non-hub refs, then sort largest-first (with deterministic tie-break)
    rng = random.Random(seed)
    rng.shuffle(non_hub_refs)
    # Precompute pos sets per group
    group_pos = {sig: {p for r in rs for p in r[6]} for sig, rs in groups.items()}
    group_cats = {sig: Counter(r[4] for r in rs) for sig, rs in groups.items()}

    # Greedy assignment balancing: (1) overall & category ratio targets, (2) minimizing cross-split pos overlap
    for sig in non_hub_refs:
        n_g = len(groups[sig])
        g_pos = group_pos[sig]
        g_cats = group_cats[sig]
        best_sp = None
        best_score = -1e18
        for sp in ("train", "val", "test"):
            rem_ratio = (targets[sp] - split_counts[sp]) / max(targets[sp], 1)
            # Category deficit bonus
            cat_bonus = sum(
                (cat_targets[sp][c] - split_cat_counts[sp][c]) / max(cat_targets[sp][c], 1.0) * cnt
                for c, cnt in g_cats.items()
            ) / max(n_g, 1)
            # Penalty for pos overlap with OTHER splits
            other_pos = set()
            for osp in ("train", "val", "test"):
                if osp != sp:
                    other_pos |= split_pos_sets[osp]
            overlap_frac = (len(g_pos & other_pos) / max(len(g_pos), 1)) if g_pos else 0.0
            # Combined score: keep ratio & category balance tight while penalizing cross-split pos overlap
            score = 3.0 * rem_ratio + 1.5 * cat_bonus - 0.6 * overlap_frac
            if score > best_score:
                best_score = score
                best_sp = sp
        assert best_sp is not None
        add_group(best_sp, sig)

    split_qids: dict[str, list[str]] = {}
    for sp in ("train", "val", "test"):
        qids = [r[0] for sig in split_refs[sp] for r in groups[sig]]
        qids.sort()
        split_qids[sp] = qids

    multi_qids = sorted(r[0] for r in multi_rows)

    # Leakage audit
    row_by_qid = {r[0]: r for r in single_rows}
    sets = {}
    for sp in ("train", "val", "test"):
        refs, poss, negs = set(), set(), set()
        for qid in split_qids[sp]:
            r = row_by_qid[qid]
            refs.add(r[1])
            poss.update(r[6])
            negs.update(r[7])
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

    # Check where the 2 missing 403 hard-negative images appear
    missing_sigs = {"1269714db704dc89eb1fcfe18fbc8c4e", "a5b5625b26942bd4d5cd5513b235bc83"}
    missing_by_split = {}
    for sp in ("train", "val", "test"):
        aff = [qid for qid in split_qids[sp] if any(m in row_by_qid[qid][7] for m in missing_sigs)]
        missing_by_split[sp] = len(aff)

    return {
        "dataset": "PinPoint",
        "source": "pinpoint_licensed.parquet",
        "seed": seed,
        "ratios": RATIOS,
        "strategy": "single_image_group_by_ref_pin_top5_hubs_to_train_stratified_min_pos_overlap",
        "pinned_hub_refs_in_train": {s: len(groups[s]) for s in hub_refs},
        "counts": {sp: len(split_qids[sp]) for sp in ("train", "val", "test")},
        "unique_ref_counts": {sp: len(split_refs[sp]) for sp in ("train", "val", "test")},
        "multi_image_eval_count": len(multi_qids),
        "category_distribution": {sp: dict(sorted(split_cat_counts[sp].items())) for sp in ("train", "val", "test")},
        "leakage_audit": leakage,
        "missing_403_neg_affected_queries": missing_by_split,
        "splits": split_qids,
        "multi_image_eval": multi_qids,
    }


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    edir_manifest = build_edir_split(SEED)
    pp_manifest = build_pinpoint_split(SEED)

    edir_out = OUT_DIR / "edir_70_10_20_seed42.json"
    pp_out = OUT_DIR / "pinpoint_70_10_20_seed42.json"

    edir_sha = write_json(edir_out, edir_manifest)
    pp_sha = write_json(pp_out, pp_manifest)

    # Also copy convenient local copies into data/EDIR and data/PinPoint
    write_json(EDIR_DIR / "splits_70_10_20_seed42.json", edir_manifest)
    write_json(PINPOINT_DIR / "splits_70_10_20_seed42.json", pp_manifest)

    print("=== EDIR SUMMARY ===")
    print("SHA256:", edir_sha)
    print("Counts:", edir_manifest["counts"], "Groups:", edir_manifest["group_counts"])
    print("Leakage:", json.dumps(edir_manifest["leakage_audit"]))
    print("Subcats train/val/test sample:", {k: edir_manifest["subcategory_distribution"][k]["remove"] for k in ("train", "val", "test")})

    print("\n=== PINPOINT SUMMARY ===")
    print("SHA256:", pp_sha)
    print("Counts:", pp_manifest["counts"], "Unique refs:", pp_manifest["unique_ref_counts"], "Multi-img eval:", pp_manifest["multi_image_eval_count"])
    print("Pinned hubs:", pp_manifest["pinned_hub_refs_in_train"])
    print("Categories:", json.dumps(pp_manifest["category_distribution"]))
    print("Leakage:", json.dumps(pp_manifest["leakage_audit"], indent=2))
    print("Missing 403 neg affected queries:", pp_manifest["missing_403_neg_affected_queries"])


if __name__ == "__main__":
    main()
