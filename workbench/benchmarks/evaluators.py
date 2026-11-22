"""Standardized, deterministic evaluators and top-K ranking for EDIR and PinPoint.

PinPoint metrics match ``pinpoint-dataset/evaluate.py`` & ``utils/metrics.py``
bit-for-bit (when ``use_raw_gt=True``), plus per-category breakdown.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from workbench.benchmarks.schema import CIRQueryRecord


def rank_embeddings(
    query_embeddings: np.ndarray,
    query_ids: Sequence[str],
    corpus_embeddings: np.ndarray,
    corpus_ids: Sequence[str],
    top_k: int = 50,
    l2_normalize: bool = True,
    exclude_ids_per_query: Optional[Mapping[str, Iterable[str]]] = None,
    chunk_size: int = 512,
) -> Dict[str, List[str]]:
    """Deterministic top-K cosine/dot-product ranking using stable tie-breaking.

    Note:
        - For **EDIR**: reference images are NOT in corpus, so ``exclude_ids_per_query=None``.
        - For **PinPoint**: reference images ARE intentionally inside the corpus (and
          explicitly marked as hard negatives for 4,217 queries), matching
          ``pinpoint-dataset/run_retrieval.py``, so keep ``exclude_ids_per_query=None``.
    """
    q_emb = np.asarray(query_embeddings, dtype=np.float32)
    c_emb = np.asarray(corpus_embeddings, dtype=np.float32)
    if q_emb.ndim != 2 or c_emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got q={q_emb.shape}, c={c_emb.shape}")
    if q_emb.shape[0] != len(query_ids):
        raise ValueError(f"query_embeddings rows ({q_emb.shape[0]}) != len(query_ids) ({len(query_ids)})")
    if c_emb.shape[0] != len(corpus_ids):
        raise ValueError(f"corpus_embeddings rows ({c_emb.shape[0]}) != len(corpus_ids) ({len(corpus_ids)})")

    if l2_normalize:
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        c_norm = np.linalg.norm(c_emb, axis=1, keepdims=True)
        q_emb = q_emb / np.maximum(q_norm, 1e-12)
        c_emb = c_emb / np.maximum(c_norm, 1e-12)

    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    k = min(int(top_k), len(corpus_ids))
    results: Dict[str, List[str]] = {}

    for start in range(0, len(query_ids), chunk_size):
        end = min(start + chunk_size, len(query_ids))
        sims = q_emb[start:end] @ c_emb.T  # [B, N_corpus]
        if exclude_ids_per_query:
            for row_i, qid in enumerate(query_ids[start:end]):
                excl = exclude_ids_per_query.get(qid)
                if excl:
                    for cid in excl:
                        idx = corpus_id_to_idx.get(cid)
                        if idx is not None:
                            sims[row_i, idx] = -np.inf
        # Stable descending sort: deterministic tie-breaking by corpus order
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[start:end]):
            results[str(qid)] = [str(corpus_ids[c_i]) for c_i in order[row_i]]

    return results


def _lookup_pred(predictions: Mapping[str, Any], rec: CIRQueryRecord) -> Optional[List[str]]:
    val = predictions.get(rec.query_id)
    if val is None:
        val = predictions.get(rec.normalized_query_id)
    if val is None:
        return None
    if isinstance(val, dict):
        return list(val.get("retrieved_items", []))
    return list(val)


def evaluate_edir(
    predictions: Mapping[str, Any],
    records: Sequence[CIRQueryRecord],
    ks: Sequence[int] = (1, 5, 10, 50),
) -> Dict[str, Any]:
    """Evaluate ranked predictions on EDIR (overall + per-15-subcategory Recall@K & MRR)."""
    overall_hits: Dict[int, List[float]] = {k: [] for k in ks}
    overall_rr: List[float] = []
    by_cat_hits: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: {k: [] for k in ks})
    by_cat_rr: Dict[str, List[float]] = defaultdict(list)

    matched = 0
    for rec in records:
        retrieved = _lookup_pred(predictions, rec)
        if retrieved is None:
            continue
        matched += 1
        pos_set = set(rec.positive_ids)

        # Reciprocal rank (within returned top-K)
        rr = 0.0
        for rank_idx, cid in enumerate(retrieved, start=1):
            if cid in pos_set:
                rr = 1.0 / rank_idx
                break
        overall_rr.append(rr)
        by_cat_rr[rec.category].append(rr)

        for k in ks:
            hit = 1.0 if any(cid in pos_set for cid in retrieved[:k]) else 0.0
            overall_hits[k].append(hit)
            by_cat_hits[rec.category][k].append(hit)

    overall: Dict[str, Any] = {
        "matched_queries": matched,
        "total_queries": len(records),
        "MRR": float(np.mean(overall_rr)) if overall_rr else 0.0,
    }
    for k in ks:
        overall[f"Recall@{k}"] = float(np.mean(overall_hits[k])) if overall_hits[k] else 0.0

    per_subcategory: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(by_cat_hits.keys()):
        cat_metrics: Dict[str, Any] = {
            "count": len(by_cat_rr[cat]),
            "MRR": float(np.mean(by_cat_rr[cat])) if by_cat_rr[cat] else 0.0,
        }
        for k in ks:
            cat_metrics[f"Recall@{k}"] = float(np.mean(by_cat_hits[cat][k])) if by_cat_hits[cat][k] else 0.0
        per_subcategory[cat] = cat_metrics

    # Macro-average across subcategories (important because `complex` has 800 vs 300 for other 14)
    macro: Dict[str, float] = {}
    if per_subcategory:
        macro["MRR"] = float(np.mean([m["MRR"] for m in per_subcategory.values()]))
        for k in ks:
            macro[f"Recall@{k}"] = float(np.mean([m[f"Recall@{k}"] for m in per_subcategory.values()]))

    return {
        "overall": overall,
        "macro_subcategory": macro,
        "per_subcategory": per_subcategory,
    }


def _pp_metrics_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> Dict[str, float]:
    """Exact replica of ``pinpoint-dataset/utils/metrics.py:calculate_metrics_at_k``."""
    retrieved_k = retrieved[:k]
    if not relevant:
        return {"precision": 0.0, "recall": 0.0, "ap": 0.0}
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved_k if item in relevant_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant)
    ap = 0.0
    hits_so_far = 0
    for i, item in enumerate(retrieved_k):
        if item in relevant_set:
            hits_so_far += 1
            ap += hits_so_far / (i + 1)
    ap = ap / min(len(relevant), k)
    return {"precision": precision, "recall": recall, "ap": ap}


def _pp_neg_recall_at_k(retrieved: Sequence[str], negative: Sequence[str], k: int) -> float:
    """Exact replica of ``pinpoint-dataset/utils/metrics.py:calculate_neg_recall_at_k``."""
    if not negative:
        return 0.0
    negative_set = set(negative)
    retrieved_k = retrieved[:k]
    neg_hits = sum(1 for item in retrieved_k if item in negative_set)
    return neg_hits / min(len(negative), k)


def _aggregate_pp_bucket(bucket: Dict[str, Any], ks: Sequence[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total_queries": bucket["matched"]}
    for k in ks:
        ks_str = str(k)
        out[f"mAP@{ks_str}"] = float(np.mean(bucket["ap"][k])) if bucket["ap"][k] else 0.0
        out[f"mAP@{ks_str}_noNeg"] = float(np.mean(bucket["ap_no_neg"][k])) if bucket["ap_no_neg"][k] else 0.0
        out[f"precision@{ks_str}"] = float(np.mean(bucket["prec"][k])) if bucket["prec"][k] else 0.0
        out[f"recall@{ks_str}"] = float(np.mean(bucket["rec"][k])) if bucket["rec"][k] else 0.0
        out[f"NegRecall@{ks_str}"] = float(np.mean(bucket["neg_rec"][k])) if bucket["neg_rec"][k] else 0.0

    if 10 in ks:
        out["delta_mAP@10_noNeg"] = out["mAP@10_noNeg"] - out["mAP@10"]
        out["pct_increase_mAP@10_noNeg"] = (
            out["delta_mAP@10_noNeg"] / out["mAP@10"] if out["mAP@10"] > 0 else 0.0
        )

    ranges, stds = [], []
    for scores in bucket["variations"].values():
        if len(scores) > 1:
            ranges.append(max(scores) - min(scores))
            stds.append(float(np.std(scores)))
    out["ling_sens_range"] = float(np.mean(ranges)) if ranges else 0.0
    out["ling_sens_std"] = float(np.mean(stds)) if stds else 0.0
    return out


def evaluate_pinpoint(
    predictions: Mapping[str, Any],
    records: Sequence[CIRQueryRecord],
    ks: Sequence[int] = (1, 5, 10, 50),
    use_raw_gt: bool = True,
) -> Dict[str, Any]:
    """Evaluate ranked predictions on PinPoint (overall + per-query_category).

    When ``use_raw_gt=True`` (default), uses ``positive_ids_raw`` & ``negative_ids_raw``
    so metrics match ``pinpoint-dataset/evaluate.py`` bit-for-bit.
    """
    def new_bucket() -> Dict[str, Any]:
        return {
            "matched": 0,
            "ap": {k: [] for k in ks},
            "ap_no_neg": {k: [] for k in ks},
            "prec": {k: [] for k in ks},
            "rec": {k: [] for k in ks},
            "neg_rec": {k: [] for k in ks},
            "variations": defaultdict(list),
        }

    overall_b = new_bucket()
    cat_b: Dict[str, Dict[str, Any]] = defaultdict(new_bucket)

    for rec in records:
        retrieved = _lookup_pred(predictions, rec)
        if retrieved is None:
            continue

        relevant = list(rec.positive_ids_raw) if (use_raw_gt and rec.positive_ids_raw) else list(rec.positive_ids)
        negative = list(rec.negative_ids_raw) if (use_raw_gt and rec.negative_ids_raw) else list(rec.negative_ids)

        neg_set = set(negative)
        retrieved_no_neg = [item for item in retrieved if item not in neg_set]

        # Linguistic sensitivity key matches pinpoint-dataset/evaluate.py:137-140
        img1 = rec.ref_id
        img2_raw = rec.metadata.get("query_image_signature2_raw", rec.ref2_id)
        var_key = (img1, img2_raw) if img2_raw else (img1, None)

        for bucket in (overall_b, cat_b[rec.category]):
            bucket["matched"] += 1
            for k in ks:
                rk = _pp_metrics_at_k(retrieved, relevant, k)
                bucket["ap"][k].append(rk["ap"])
                bucket["prec"][k].append(rk["precision"])
                bucket["rec"][k].append(rk["recall"])
                bucket["neg_rec"][k].append(_pp_neg_recall_at_k(retrieved, negative, k))

                rk_nn = _pp_metrics_at_k(retrieved_no_neg, relevant, k)
                bucket["ap_no_neg"][k].append(rk_nn["ap"])
            p10 = _pp_metrics_at_k(retrieved, relevant, 10)["precision"]
            bucket["variations"][var_key].append(p10)

    return {
        "overall": _aggregate_pp_bucket(overall_b, ks),
        "per_category": {cat: _aggregate_pp_bucket(b, ks) for cat, b in sorted(cat_b.items())},
    }
