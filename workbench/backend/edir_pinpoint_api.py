"""FastAPI router serving verified EDIR & PinPoint in-domain + cross-benchmark transfer results.

Strictly read-only over:
  - Canonical splits: data/{EDIR,PinPoint}/splits/70_10_20_seed42/
  - In-Domain runs:   workbench/artifacts/benchmark_runs/{edir,pinpoint}/70_10_20_seed42/
  - Transfer runs:    workbench/artifacts/transfer_evals/{fashioniq_to_pinpoint,cirr_to_edir,fashioniq_to_edir}/

Uses mtime-keyed in-memory caching so live background runs appear automatically as soon as
they finish writing `test_results.json` + `test_predictions_top50.json`, with <10ms warm latency.
"""

from __future__ import annotations

import csv
import io
import json
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from workbench.benchmarks import evaluate_edir, evaluate_pinpoint, load_split_bundle
from workbench.benchmarks.evaluators import _pp_metrics_at_k, _pp_neg_recall_at_k
from workbench.benchmarks.schema import CIRQueryRecord, DEFAULT_SPLIT_TAG, SplitBundle
from workbench.benchmarks.trainers.run_benchmark_suite import MODEL_QUEUE
from workbench.backend.storage import get_analysis_input, list_runs

REPO_ROOT = Path(__file__).resolve().parents[2]
INDOMAIN_ROOT = REPO_ROOT / "workbench" / "artifacts" / "benchmark_runs"
TRANSFER_ROOT = REPO_ROOT / "workbench" / "artifacts" / "transfer_evals"

MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "clip4cir": {"display": "CLIP4Cir", "family": "CLIP", "backbone": "RN50x4 + Combiner"},
    "encoder": {"display": "ENCODER", "family": "CLIP", "backbone": "OpenCLIP ViT-B/32"},
    "tgcir": {"display": "TG-CIR", "family": "CLIP", "backbone": "CLIP ViT-B/16"},
    "limn": {"display": "LIMN", "family": "CLIP", "backbone": "OpenCLIP ViT-L/14"},
    "sprc": {"display": "SPRC", "family": "BLIP-2", "backbone": "BLIP-2 Q-Former (ViT-g / ViT-L)"},
    "mtst_ptha": {"display": "MTST-PTHA", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g"},
    "hint": {"display": "HINT", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g (iLearn)"},
    "intent": {"display": "INTENT", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g (iLearn)"},
    "habit": {"display": "HABIT", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g (iLearn)"},
    "conesep": {"display": "ConeSep", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g (iLearn)"},
    "airknow": {"display": "AirKnow", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g (iLearn)"},
    "csmcir": {"display": "CSMCIR", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g"},
    "tme": {"display": "TME", "family": "BLIP-2", "backbone": "BLIP-2 EVA-ViT-g"},
    "clvcnet": {"display": "CLVC-Net", "family": "Classic", "backbone": "ResNet50 + LSTM"},
    "dcnet": {"display": "DCNet", "family": "Classic", "backbone": "ResNet50 + GloVe 300d"},
}

UNAVAILABLE_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "fiq_transfer": {
        "mtst_ptha": "Tác giả không phát hành checkpoint FashionIQ (chỉ công bố bản CIRR).",
    },
    "cirr_transfer": {
        "tme": "Link Kaggle chính chủ (`cirr-0.0-best`) trả HTTP 404.",
        "airknow": "Tác giả không phát hành checkpoint CIRR trên HuggingFace.",
        "conesep": "Tác giả không phát hành checkpoint CIRR trên HuggingFace.",
        "clvcnet": "Paper năm 2021 (trước CIRR), không chạy benchmark CIRR.",
        "dcnet": "Paper năm 2021 (trước CIRR), không chạy benchmark CIRR.",
    },
}

PINPOINT_SLICE_DEFS: List[Tuple[str, str, str]] = [
    ("all", "Toàn bộ PinPoint", "Tất cả 1,337 test queries trên 23 lĩnh vực"),
    ("fashion_all", "Fashion All", "Women's + Men's + Children's Fashion"),
    ("fashion_no_comp", "Fashion (trừ Complement)", "Fashion thuần chỉnh sửa thuộc tính / bối cảnh / phủ định"),
    ("fashion_swap", "Fashion · Swap", "Thay đổi thuộc tính trang phục (gần FashionIQ nhất)"),
    ("fashion_ctx", "Fashion · Context Fit", "Đặt trang phục vào bối cảnh / phong cách mới"),
    ("fashion_neg", "Fashion · Negation", "Phủ định thuộc tính trang phục"),
    ("fashion_comp", "Fashion · Complement", "Phối đồ / phụ kiện bổ trợ khác chủng loại"),
    ("non_fashion", "Non-Fashion", "20 lĩnh vực ngoài thời trang (Nội thất, Nghệ thuật, DIY, ...)"),
]

PINPOINT_PUBLISHED_BASELINES = [
    {"label": "GPT-5 Reranked (ref)", "map10": 0.1837},
    {"label": "Qwen2.5-VL-7B ZS (ref)", "map10": 0.1380},
]

SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {spec["key"]: spec for spec in MODEL_QUEUE}

edir_pinpoint_router = APIRouter(prefix="/api/benchmarks", tags=["edir-pinpoint-benchmarks"])

_CACHE_LOCK = threading.RLock()
_BUNDLE_CACHE: Dict[str, SplitBundle] = {}
_IMAGE_MAP_CACHE: Dict[str, Dict[str, Path]] = {}
_JSON_CACHE: Dict[str, Tuple[int, Any]] = {}
_SUMMARY_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int], ...], Dict[str, Any]]] = {}


def _validate_dataset(dataset: str) -> str:
    key = dataset.strip().lower()
    if key not in ("edir", "pinpoint"):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_dataset", "message": f"Dataset không hợp lệ: {dataset!r} (chỉ nhận 'edir' hoặc 'pinpoint')."},
        )
    return key


def _get_bundle(dataset: str) -> SplitBundle:
    with _CACHE_LOCK:
        cached = _BUNDLE_CACHE.get(dataset)
        if cached is not None:
            return cached
        try:
            bundle = load_split_bundle(dataset, "test", corpus_scope="full", split_tag=DEFAULT_SPLIT_TAG)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"error": "split_unavailable", "message": f"Không tải được canonical split cho {dataset}: {exc}"},
            ) from exc
        _BUNDLE_CACHE[dataset] = bundle
        return bundle


def _get_image_map(dataset: str) -> Dict[str, Path]:
    with _CACHE_LOCK:
        cached = _IMAGE_MAP_CACHE.get(dataset)
        if cached is not None:
            return cached
        bundle = _get_bundle(dataset)
        img_map: Dict[str, Path] = {}
        for rec in bundle.corpus:
            img_map[rec.image_id] = rec.image_path(bundle.data_dir)
        for q in bundle.queries:
            img_map[q.ref_id] = q.ref_path(bundle.data_dir)
            if q.ref2_id and q.ref2_image_rel:
                r2 = q.ref2_path(bundle.data_dir)
                if r2 is not None:
                    img_map[q.ref2_id] = r2
            for pid in (*q.positive_ids, *q.positive_ids_raw, *q.negative_ids, *q.negative_ids_raw):
                if isinstance(pid, str) and pid and pid not in img_map:
                    img_map[pid] = bundle.resolve_image_path(pid)
        _IMAGE_MAP_CACHE[dataset] = img_map
        return img_map


def _clean_unique_ids(ids: Any) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in ids or ():
        if isinstance(item, str) and item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _image_url(dataset: str, image_id: str) -> str:
    return f"/api/benchmarks/{dataset}/images/{urllib.parse.quote(image_id, safe='')}"


def _read_json_cached(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    key = str(path)
    with _CACHE_LOCK:
        hit = _JSON_CACHE.get(key)
        if hit is not None and hit[0] == mtime_ns:
            return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # File may currently be mid-write by a background evaluator; treat gracefully as pending.
        return None
    with _CACHE_LOCK:
        _JSON_CACHE[key] = (mtime_ns, data)
    return data


def _artifact_paths(dataset: str, model_key: str, regime: str) -> Tuple[Path, Path]:
    spec = SPEC_BY_KEY.get(model_key)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_model", "message": f"Model không hợp lệ: {model_key!r}"},
        )
    if regime == "indomain":
        base = INDOMAIN_ROOT / dataset / DEFAULT_SPLIT_TAG / spec["run_id"]
    elif regime == "fiq_transfer":
        base = TRANSFER_ROOT / f"fashioniq_to_{dataset}" / model_key
    elif regime == "cirr_transfer":
        if dataset != "edir":
            raise HTTPException(
                status_code=400,
                detail={"error": "unsupported_regime", "message": "CIRR Transfer chỉ áp dụng trên EDIR."},
            )
        base = TRANSFER_ROOT / "cirr_to_edir" / model_key
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_regime", "message": f"Regime không hợp lệ: {regime!r}"},
        )
    return base / "test_results.json", base / "test_predictions_top50.json"


def _pinpoint_slice_match(q: CIRQueryRecord, slice_id: str) -> bool:
    is_fashion = "Fashion" in str(q.metadata.get("l1_interest", ""))
    if slice_id == "all":
        return True
    if slice_id == "fashion_all":
        return is_fashion
    if slice_id == "fashion_no_comp":
        return is_fashion and q.category != "Complement"
    if slice_id == "fashion_swap":
        return is_fashion and q.category == "Swap"
    if slice_id == "fashion_ctx":
        return is_fashion and q.category == "Context Fit"
    if slice_id == "fashion_neg":
        return is_fashion and q.category == "Negation"
    if slice_id == "fashion_comp":
        return is_fashion and q.category == "Complement"
    if slice_id == "non_fashion":
        return not is_fashion
    raise HTTPException(
        status_code=400,
        detail={"error": "invalid_slice", "message": f"Slice không hợp lệ: {slice_id!r}"},
    )


def _compute_pinpoint_slices(preds: Dict[str, List[str]], queries: Tuple[CIRQueryRecord, ...]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for slice_id, _, _ in PINPOINT_SLICE_DEFS:
        q_sub = [q for q in queries if _pinpoint_slice_match(q, slice_id)]
        sub_preds = {q.query_id: preds[q.query_id] for q in q_sub if q.query_id in preds}
        if not q_sub or len(sub_preds) != len(q_sub):
            continue
        res = evaluate_pinpoint(sub_preds, tuple(q_sub), ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
        out[slice_id] = res["overall"]
    return out


def _signature_for_dataset(dataset: str) -> Tuple[Tuple[str, int], ...]:
    regimes = ("indomain", "fiq_transfer", "cirr_transfer") if dataset == "edir" else ("indomain", "fiq_transfer")
    items: List[Tuple[str, int]] = []
    for spec in MODEL_QUEUE:
        for reg in regimes:
            res_p, pred_p = _artifact_paths(dataset, spec["key"], reg)
            for p in (res_p, pred_p):
                try:
                    mtime = p.stat().st_mtime_ns if p.is_file() else 0
                except OSError:
                    mtime = 0
                items.append((str(p), mtime))
    return tuple(items)


def _build_edir_summary() -> Dict[str, Any]:
    bundle = _get_bundle("edir")
    subcat_counts: Dict[str, int] = {}
    for q in bundle.queries:
        subcat_counts[q.category] = subcat_counts.get(q.category, 0) + 1
    subcategories = [{"id": k, "count": subcat_counts[k]} for k in sorted(subcat_counts)]

    models_out: List[Dict[str, Any]] = []
    completed = {"indomain": 0, "cirr_transfer": 0, "fiq_transfer": 0}
    expected = {
        "indomain": len(MODEL_QUEUE),
        "cirr_transfer": len(MODEL_QUEUE) - len(UNAVAILABLE_CHECKPOINTS["cirr_transfer"]),
        "fiq_transfer": len(MODEL_QUEUE) - len(UNAVAILABLE_CHECKPOINTS["fiq_transfer"]),
    }

    for spec in MODEL_QUEUE:
        key = spec["key"]
        meta = MODEL_METADATA[key]
        in_res_p, in_pred_p = _artifact_paths("edir", key, "indomain")
        cirr_res_p, cirr_pred_p = _artifact_paths("edir", key, "cirr_transfer")
        fiq_res_p, fiq_pred_p = _artifact_paths("edir", key, "fiq_transfer")

        in_res = _read_json_cached(in_res_p)
        cirr_res = _read_json_cached(cirr_res_p)
        fiq_res = _read_json_cached(fiq_res_p)

        s0_raw = (in_res or {}).get("stage0_zeroshot_test") or (in_res or {}).get("stage0_init_test") or {}
        stage0 = s0_raw.get("overall") if isinstance(s0_raw, dict) else None

        if in_res and spec["result_field"] in in_res:
            completed["indomain"] += 1
            in_metrics = in_res[spec["result_field"]]
            best_ep = in_res.get("best_epoch") if in_res.get("best_epoch") is not None else in_res.get("best_stage2_epoch")
            best_val = in_res.get("best_val_score") if in_res.get("best_val_score") is not None else in_res.get("best_stage2_val_score")
            indomain = {
                "status": "ready",
                "has_predictions": in_pred_p.is_file(),
                "best_epoch": best_ep,
                "best_val_score": best_val,
                "finished_at": in_res.get("finished_at"),
                "overall": in_metrics.get("overall"),
                "macro_subcategory": in_metrics.get("macro_subcategory"),
                "per_subcategory": in_metrics.get("per_subcategory"),
            }
        else:
            indomain = {"status": "pending", "has_predictions": False}

        def _pack_transfer(reg_name: str, res_obj: Optional[Dict[str, Any]], pred_path: Path) -> Dict[str, Any]:
            if key in UNAVAILABLE_CHECKPOINTS[reg_name]:
                return {
                    "status": "unavailable",
                    "has_predictions": False,
                    "unavailable_reason": UNAVAILABLE_CHECKPOINTS[reg_name][key],
                }
            if res_obj and "test_metrics" in res_obj:
                completed[reg_name] += 1
                tm = res_obj["test_metrics"]
                return {
                    "status": "ready",
                    "has_predictions": pred_path.is_file(),
                    "variant": res_obj.get("variant"),
                    "finished_at": res_obj.get("finished_at"),
                    "elapsed_sec": res_obj.get("elapsed_sec"),
                    "overall": tm.get("overall"),
                    "macro_subcategory": tm.get("macro_subcategory"),
                    "per_subcategory": tm.get("per_subcategory"),
                }
            return {"status": "pending", "has_predictions": False}

        cirr = _pack_transfer("cirr_transfer", cirr_res, cirr_pred_p)
        fiq = _pack_transfer("fiq_transfer", fiq_res, fiq_pred_p)

        in_r1 = (indomain.get("overall") or {}).get("Recall@1")
        cirr_r1 = (cirr.get("overall") or {}).get("Recall@1")
        fiq_r1 = (fiq.get("overall") or {}).get("Recall@1")
        s0_r1 = (stage0 or {}).get("Recall@1")

        models_out.append({
            "key": key,
            "run_id": spec["run_id"],
            "display": meta["display"],
            "family": meta["family"],
            "backbone": meta["backbone"],
            "stage0": stage0,
            "indomain": indomain,
            "cirr_transfer": cirr,
            "fiq_transfer": fiq,
            "deltas": {
                "indomain_minus_cirr_r1": (in_r1 - cirr_r1) if (in_r1 is not None and cirr_r1 is not None) else None,
                "indomain_minus_fiq_r1": (in_r1 - fiq_r1) if (in_r1 is not None and fiq_r1 is not None) else None,
                "indomain_minus_s0_r1": (in_r1 - s0_r1) if (in_r1 is not None and s0_r1 is not None) else None,
            },
        })

    models_out.sort(key=lambda m: ((m["indomain"].get("overall") or {}).get("Recall@1", -1.0)), reverse=True)
    return {
        "dataset": "edir",
        "split_tag": DEFAULT_SPLIT_TAG,
        "corpus_size": len(bundle.corpus),
        "test_query_count": len(bundle.queries),
        "primary_metric": "Recall@1",
        "completion": {"completed": completed, "expected": expected},
        "subcategories": subcategories,
        "models": models_out,
        "unavailable": UNAVAILABLE_CHECKPOINTS,
    }


def _build_pinpoint_summary() -> Dict[str, Any]:
    bundle = _get_bundle("pinpoint")
    slice_meta = [
        {
            "id": sid,
            "label": label,
            "description": desc,
            "count": sum(1 for q in bundle.queries if _pinpoint_slice_match(q, sid)),
        }
        for sid, label, desc in PINPOINT_SLICE_DEFS
    ]
    cat_counts: Dict[str, int] = {}
    for q in bundle.queries:
        cat_counts[q.category] = cat_counts.get(q.category, 0) + 1
    categories = [{"id": k, "count": cat_counts[k]} for k in sorted(cat_counts)]

    models_out: List[Dict[str, Any]] = []
    completed = {"indomain": 0, "fiq_transfer": 0}
    expected = {
        "indomain": len(MODEL_QUEUE),
        "fiq_transfer": len(MODEL_QUEUE) - len(UNAVAILABLE_CHECKPOINTS["fiq_transfer"]),
    }

    for spec in MODEL_QUEUE:
        key = spec["key"]
        meta = MODEL_METADATA[key]
        in_res_p, in_pred_p = _artifact_paths("pinpoint", key, "indomain")
        fiq_res_p, fiq_pred_p = _artifact_paths("pinpoint", key, "fiq_transfer")

        in_res = _read_json_cached(in_res_p)
        in_preds = _read_json_cached(in_pred_p)
        fiq_res = _read_json_cached(fiq_res_p)
        fiq_preds = _read_json_cached(fiq_pred_p)

        s0_raw = (in_res or {}).get("stage0_zeroshot_test") or (in_res or {}).get("stage0_init_test") or {}
        stage0 = s0_raw.get("overall") if isinstance(s0_raw, dict) else None

        if in_res and spec["result_field"] in in_res:
            completed["indomain"] += 1
            in_metrics = in_res[spec["result_field"]]
            slices = _compute_pinpoint_slices(in_preds, bundle.queries) if isinstance(in_preds, dict) else {"all": in_metrics.get("overall")}
            best_ep = in_res.get("best_epoch") if in_res.get("best_epoch") is not None else in_res.get("best_stage2_epoch")
            best_val = in_res.get("best_val_score") if in_res.get("best_val_score") is not None else in_res.get("best_stage2_val_score")
            indomain = {
                "status": "ready",
                "has_predictions": isinstance(in_preds, dict),
                "best_epoch": best_ep,
                "best_val_score": best_val,
                "finished_at": in_res.get("finished_at"),
                "overall": in_metrics.get("overall"),
                "per_category": in_metrics.get("per_category"),
                "slices": slices,
            }
        else:
            indomain = {"status": "pending", "has_predictions": False, "slices": {}}

        if key in UNAVAILABLE_CHECKPOINTS["fiq_transfer"]:
            fiq: Dict[str, Any] = {
                "status": "unavailable",
                "has_predictions": False,
                "unavailable_reason": UNAVAILABLE_CHECKPOINTS["fiq_transfer"][key],
                "slices": {},
            }
        elif fiq_res and "test_metrics" in fiq_res:
            completed["fiq_transfer"] += 1
            tm = fiq_res["test_metrics"]
            slices = _compute_pinpoint_slices(fiq_preds, bundle.queries) if isinstance(fiq_preds, dict) else {"all": tm.get("overall")}
            fiq = {
                "status": "ready",
                "has_predictions": isinstance(fiq_preds, dict),
                "variant": fiq_res.get("variant"),
                "finished_at": fiq_res.get("finished_at"),
                "elapsed_sec": fiq_res.get("elapsed_sec"),
                "overall": tm.get("overall"),
                "per_category": tm.get("per_category"),
                "slices": slices,
            }
        else:
            fiq = {"status": "pending", "has_predictions": False, "slices": {}}

        slice_deltas: Dict[str, Optional[float]] = {}
        for sid, _, _ in PINPOINT_SLICE_DEFS:
            a = (indomain["slices"].get(sid) or {}).get("mAP@10")
            b = (fiq["slices"].get(sid) or {}).get("mAP@10")
            slice_deltas[sid] = (a - b) if (a is not None and b is not None) else None

        models_out.append({
            "key": key,
            "run_id": spec["run_id"],
            "display": meta["display"],
            "family": meta["family"],
            "backbone": meta["backbone"],
            "stage0": stage0,
            "indomain": indomain,
            "fiq_transfer": fiq,
            "slice_deltas": slice_deltas,
        })

    models_out.sort(key=lambda m: ((m["indomain"].get("overall") or {}).get("mAP@10", -1.0)), reverse=True)
    return {
        "dataset": "pinpoint",
        "split_tag": DEFAULT_SPLIT_TAG,
        "corpus_size": len(bundle.corpus),
        "test_query_count": len(bundle.queries),
        "primary_metric": "mAP@10",
        "completion": {"completed": completed, "expected": expected},
        "slices": slice_meta,
        "categories": categories,
        "published_baselines": PINPOINT_PUBLISHED_BASELINES,
        "models": models_out,
        "unavailable": UNAVAILABLE_CHECKPOINTS,
    }


def _get_summary(dataset: str) -> Dict[str, Any]:
    sig = _signature_for_dataset(dataset)
    with _CACHE_LOCK:
        cached = _SUMMARY_CACHE.get(dataset)
        if cached is not None and cached[0] == sig:
            return cached[1]
    summary = _build_edir_summary() if dataset == "edir" else _build_pinpoint_summary()
    with _CACHE_LOCK:
        _SUMMARY_CACHE[dataset] = (sig, summary)
    return summary


def _score_single_query(dataset: str, q: CIRQueryRecord, ranked: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if ranked is None:
        return None
    if dataset == "edir":
        pos_list = list(q.positive_ids)
        pos_set = set(pos_list)
        best_rank = next((i for i, cid in enumerate(ranked, start=1) if cid in pos_set), None)
        rr = (1.0 / best_rank) if best_rank is not None else 0.0
        top1 = ranked[0] if ranked else None
        return {
            "primary_score": 1.0 if best_rank == 1 else 0.0,
            "best_gt_rank": best_rank,
            "rr": rr,
            "hit1": best_rank == 1,
            "hit10": best_rank is not None and best_rank <= 10,
            "hit50": best_rank is not None and best_rank <= 50,
            "top1_image_id": top1,
            "top1_is_gt": top1 in pos_set if top1 else False,
            "top1_is_ref": top1 == q.ref_id if top1 else False,
            "top1_is_hneg": False,
        }
    pos_list = list(q.positive_ids_raw or q.positive_ids)
    neg_list = list(q.negative_ids_raw or q.negative_ids)
    pos_set = set(pos_list)
    neg_set = set(neg_list)
    best_rank = next((i for i, cid in enumerate(ranked, start=1) if cid in pos_set), None)
    m10 = _pp_metrics_at_k(ranked, pos_list, 10)
    neg_rec10 = _pp_neg_recall_at_k(ranked, neg_list, 10)
    top1 = ranked[0] if ranked else None
    return {
        "primary_score": float(m10["ap"]),
        "ap10": float(m10["ap"]),
        "p10": float(m10["precision"]),
        "neg_recall10": float(neg_rec10),
        "best_gt_rank": best_rank,
        "hit1": top1 in pos_set if top1 else False,
        "hit10": best_rank is not None and best_rank <= 10,
        "hit50": best_rank is not None and best_rank <= 50,
        "top1_image_id": top1,
        "top1_is_gt": top1 in pos_set if top1 else False,
        "top1_is_ref": top1 == q.ref_id if top1 else False,
        "top1_is_hneg": top1 in neg_set if top1 else False,
    }


def _load_preds_for(dataset: str, model_key: str, regime: str) -> Tuple[Optional[Dict[str, List[str]]], Optional[Dict[str, Any]]]:
    res_p, pred_p = _artifact_paths(dataset, model_key, regime)
    res_obj = _read_json_cached(res_p)
    pred_obj = _read_json_cached(pred_p)
    return (pred_obj if isinstance(pred_obj, dict) else None), (res_obj if isinstance(res_obj, dict) else None)


@edir_pinpoint_router.get("/overview")
def benchmarks_overview() -> Dict[str, Any]:
    edir = _get_summary("edir")
    pinpoint = _get_summary("pinpoint")

    edir_remove_vals = [
        (m["indomain"].get("per_subcategory") or {}).get("remove", {}).get("Recall@1")
        for m in edir["models"]
        if m["indomain"].get("status") == "ready"
    ]
    edir_remove_vals = [v for v in edir_remove_vals if v is not None]
    pp_above_gpt5 = sum(
        1 for m in pinpoint["models"]
        if ((m["indomain"].get("overall") or {}).get("mAP@10") or 0.0) > 0.1837
    )

    availability_rows = []
    for spec in MODEL_QUEUE:
        k = spec["key"]
        meta = MODEL_METADATA[k]
        e_m = next(m for m in edir["models"] if m["key"] == k)
        p_m = next(m for m in pinpoint["models"] if m["key"] == k)
        availability_rows.append({
            "key": k,
            "display": meta["display"],
            "family": meta["family"],
            "backbone": meta["backbone"],
            "edir_indomain": e_m["indomain"]["status"],
            "pinpoint_indomain": p_m["indomain"]["status"],
            "fiq_to_pinpoint": p_m["fiq_transfer"]["status"],
            "fiq_variant": p_m["fiq_transfer"].get("variant") or e_m["fiq_transfer"].get("variant"),
            "fiq_reason": p_m["fiq_transfer"].get("unavailable_reason"),
            "cirr_to_edir": e_m["cirr_transfer"]["status"],
            "cirr_variant": e_m["cirr_transfer"].get("variant"),
            "cirr_reason": e_m["cirr_transfer"].get("unavailable_reason"),
            "fiq_to_edir": e_m["fiq_transfer"]["status"],
        })

    return {
        "split_tag": DEFAULT_SPLIT_TAG,
        "edir": {
            "corpus_size": edir["corpus_size"],
            "test_query_count": edir["test_query_count"],
            "completion": edir["completion"],
            "top3": [
                {
                    "key": m["key"],
                    "display": m["display"],
                    "family": m["family"],
                    "recall1": (m["indomain"].get("overall") or {}).get("Recall@1"),
                    "mrr": (m["indomain"].get("overall") or {}).get("MRR"),
                }
                for m in edir["models"][:3]
                if m["indomain"].get("status") == "ready"
            ],
            "remove_r1_min": min(edir_remove_vals) if edir_remove_vals else None,
            "remove_r1_max": max(edir_remove_vals) if edir_remove_vals else None,
        },
        "pinpoint": {
            "corpus_size": pinpoint["corpus_size"],
            "test_query_count": pinpoint["test_query_count"],
            "completion": pinpoint["completion"],
            "top3": [
                {
                    "key": m["key"],
                    "display": m["display"],
                    "family": m["family"],
                    "map10": (m["indomain"].get("overall") or {}).get("mAP@10"),
                    "p1": (m["indomain"].get("overall") or {}).get("precision@1"),
                }
                for m in pinpoint["models"][:3]
                if m["indomain"].get("status") == "ready"
            ],
            "models_beating_gpt5": pp_above_gpt5,
            "gpt5_reference_map10": 0.1837,
        },
        "availability": availability_rows,
    }


@edir_pinpoint_router.get("/common-failures")
def benchmarks_common_failures(
    protocol: str = Query(default="fashioniq_original_split"),
    models: str = Query(default=""),
    preset: str = Query(default="fail80"),
    k: int = Query(default=10, ge=1, le=50),
    category: str = Query(default=""),
    slice: str = Query(default="all"),
    q: str = Query(default=""),
    sort: str = Query(default="fail_desc"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=60),
) -> Dict[str, Any]:
    return _build_common_failures_payload(
        protocol=protocol,
        models_param=models,
        preset=preset,
        k=k,
        category=category,
        slice_id=slice,
        q_text=q,
        sort=sort,
        offset=offset,
        limit=limit,
    )


@edir_pinpoint_router.get("/common-failures/export.csv")
def benchmarks_common_failures_csv(
    protocol: str = Query(default="fashioniq_original_split"),
    models: str = Query(default=""),
    preset: str = Query(default="fail80"),
    k: int = Query(default=10, ge=1, le=50),
    category: str = Query(default=""),
    slice: str = Query(default="all"),
    q: str = Query(default=""),
    sort: str = Query(default="fail_desc"),
) -> PlainTextResponse:
    data = _build_common_failures_payload(
        protocol=protocol,
        models_param=models,
        preset=preset,
        k=k,
        category=category,
        slice_id=slice,
        q_text=q,
        sort=sort,
        offset=0,
        limit=10000,
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "protocol", "query_id", "category", "l1_interest", "text",
        f"fail_count@{k}", "scored_models", f"fail_rate@{k}", "ref_copy_count", "best_rank", "median_rank",
    ])
    for it in data["items"]:
        writer.writerow([
            protocol, it["query_id"], it["category"], it.get("l1_interest") or "", it["text"],
            it["fail_count"], it["scored_models"], f"{it['fail_rate']:.4f}", it["ref_copy_count"],
            it["best_rank"] if it["best_rank"] is not None else ">50",
            it["median_rank"] if it["median_rank"] is not None else ">50",
        ])
    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="common_failures_{protocol}_{preset}_k{k}.csv"'},
    )


@edir_pinpoint_router.get("/{dataset}/summary")
def benchmark_summary(dataset: str) -> Dict[str, Any]:
    ds = _validate_dataset(dataset)
    return _get_summary(ds)


@edir_pinpoint_router.get("/{dataset}/queries")
def benchmark_queries(
    dataset: str,
    model: str = Query(default=""),
    regime: str = Query(default="indomain"),
    compare_model: str = Query(default=""),
    compare_regime: str = Query(default=""),
    category: str = Query(default=""),
    slice: str = Query(default="all"),
    outcome: str = Query(default="all"),
    q: str = Query(default=""),
    sort: str = Query(default="worst"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=60),
) -> Dict[str, Any]:
    ds = _validate_dataset(dataset)
    bundle = _get_bundle(ds)
    summary = _get_summary(ds)

    chosen_model = model.strip().lower() or summary["models"][0]["key"]
    preds, res_obj = _load_preds_for(ds, chosen_model, regime)
    if preds is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "predictions_not_available",
                "message": f"Chưa có predictions top-50 cho {chosen_model} ({regime}) trên {ds}.",
            },
        )

    cmp_preds: Optional[Dict[str, List[str]]] = None
    cmp_model_clean = compare_model.strip().lower()
    cmp_regime_clean = compare_regime.strip().lower()
    if cmp_model_clean and cmp_regime_clean:
        cmp_preds, _ = _load_preds_for(ds, cmp_model_clean, cmp_regime_clean)

    q_lower = q.strip().lower()
    cat_filter = category.strip()
    outcome_filter = outcome.strip().lower()
    if outcome_filter not in ("all", "hit1", "hit10", "miss10", "miss50", "ref_copy", "improved", "regressed"):
        raise HTTPException(status_code=400, detail={"error": "invalid_outcome", "message": f"Outcome không hợp lệ: {outcome!r}"})

    # Pre-filter by slice, category, and search text first so outcome facets reflect active slice/search.
    rows: List[Tuple[CIRQueryRecord, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    outcome_counts = {"all": 0, "hit1": 0, "hit10": 0, "miss10": 0, "miss50": 0, "ref_copy": 0, "improved": 0, "regressed": 0}
    for rec in bundle.queries:
        if ds == "pinpoint" and not _pinpoint_slice_match(rec, slice):
            continue
        if cat_filter and rec.category != cat_filter:
            continue
        if q_lower:
            hay = f"{rec.query_id} {rec.text} {rec.category} {rec.metadata.get('l1_interest', '')}".lower()
            if q_lower not in hay:
                continue
        pe = _score_single_query(ds, rec, preds.get(rec.query_id))
        if pe is None:
            continue
        ce = _score_single_query(ds, rec, cmp_preds.get(rec.query_id)) if cmp_preds is not None else None

        outcome_counts["all"] += 1
        if pe["hit1"]:
            outcome_counts["hit1"] += 1
        if pe["hit10"]:
            outcome_counts["hit10"] += 1
        else:
            outcome_counts["miss10"] += 1
        if not pe["hit50"]:
            outcome_counts["miss50"] += 1
        if pe["top1_is_ref"]:
            outcome_counts["ref_copy"] += 1
        if ce is not None:
            delta = pe["primary_score"] - ce["primary_score"]
            if delta > 1e-9:
                outcome_counts["improved"] += 1
            elif delta < -1e-9:
                outcome_counts["regressed"] += 1

        keep = True
        if outcome_filter == "hit1":
            keep = bool(pe["hit1"])
        elif outcome_filter == "hit10":
            keep = bool(pe["hit10"])
        elif outcome_filter == "miss10":
            keep = not bool(pe["hit10"])
        elif outcome_filter == "miss50":
            keep = not bool(pe["hit50"])
        elif outcome_filter == "ref_copy":
            keep = bool(pe["top1_is_ref"])
        elif outcome_filter == "improved":
            keep = ce is not None and (pe["primary_score"] - ce["primary_score"]) > 1e-9
        elif outcome_filter == "regressed":
            keep = ce is not None and (pe["primary_score"] - ce["primary_score"]) < -1e-9
        if keep:
            rows.append((rec, pe, ce))

    def _rank_num(ev: Dict[str, Any]) -> int:
        return ev["best_gt_rank"] if ev["best_gt_rank"] is not None else 999

    if sort == "worst":
        rows.sort(key=lambda t: (t[1]["primary_score"], -_rank_num(t[1]), t[0].query_id))
    elif sort == "best":
        rows.sort(key=lambda t: (-t[1]["primary_score"], _rank_num(t[1]), t[0].query_id))
    elif sort == "delta_desc":
        rows.sort(key=lambda t: (-(t[1]["primary_score"] - (t[2]["primary_score"] if t[2] else 0.0)), t[0].query_id))
    elif sort == "delta_asc":
        rows.sort(key=lambda t: ((t[1]["primary_score"] - (t[2]["primary_score"] if t[2] else 0.0)), t[0].query_id))
    else:
        rows.sort(key=lambda t: t[0].query_id)

    page = rows[offset : offset + limit]
    items = []
    for rec, pe, ce in page:
        gt_ids = _clean_unique_ids(rec.positive_ids_raw or rec.positive_ids) if ds == "pinpoint" else _clean_unique_ids(rec.positive_ids)
        hn_ids = _clean_unique_ids(rec.negative_ids_raw or rec.negative_ids) if ds == "pinpoint" else []
        items.append({
            "query_id": rec.query_id,
            "text": rec.text,
            "category": rec.category,
            "l1_interest": rec.metadata.get("l1_interest"),
            "ref_image_id": rec.ref_id,
            "ref_image_url": _image_url(ds, rec.ref_id),
            "gt_count": len(gt_ids),
            "hard_neg_count": len(hn_ids),
            "primary": pe,
            "compare": ce,
            "delta": (pe["primary_score"] - ce["primary_score"]) if ce is not None else None,
        })

    return {
        "dataset": ds,
        "model": chosen_model,
        "regime": regime,
        "variant": (res_obj or {}).get("variant"),
        "compare_active": cmp_preds is not None,
        "compare_model": cmp_model_clean if cmp_preds is not None else None,
        "compare_regime": cmp_regime_clean if cmp_preds is not None else None,
        "total_queries": len(bundle.queries),
        "filtered_total": len(rows),
        "offset": offset,
        "limit": limit,
        "outcome_counts": outcome_counts,
        "items": items,
    }


@edir_pinpoint_router.get("/{dataset}/queries/{query_id}")
def benchmark_query_detail(
    dataset: str,
    query_id: str,
    model: str = Query(default=""),
    regime: str = Query(default="indomain"),
    compare_model: str = Query(default=""),
    compare_regime: str = Query(default=""),
    top_k: int = Query(default=12, ge=1, le=50),
) -> Dict[str, Any]:
    ds = _validate_dataset(dataset)
    bundle = _get_bundle(ds)
    summary = _get_summary(ds)

    rec = next((q for q in bundle.queries if q.query_id == query_id), None)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "query_not_found", "message": f"Không tìm thấy query {query_id!r} trong {ds}."},
        )

    chosen_model = model.strip().lower() or summary["models"][0]["key"]
    p_preds, p_res = _load_preds_for(ds, chosen_model, regime)
    if p_preds is None or query_id not in p_preds:
        raise HTTPException(
            status_code=404,
            detail={"error": "predictions_not_available", "message": f"Chưa có kết quả dự đoán cho {chosen_model} ({regime})."},
        )

    c_preds: Optional[Dict[str, List[str]]] = None
    c_res: Optional[Dict[str, Any]] = None
    cmp_m = compare_model.strip().lower()
    cmp_r = compare_regime.strip().lower()
    if cmp_m and cmp_r:
        c_preds, c_res = _load_preds_for(ds, cmp_m, cmp_r)

    gt_ids = _clean_unique_ids(rec.positive_ids_raw or rec.positive_ids) if ds == "pinpoint" else _clean_unique_ids(rec.positive_ids)
    hn_ids = _clean_unique_ids(rec.negative_ids_raw or rec.negative_ids) if ds == "pinpoint" else []
    gt_set = set(gt_ids)
    hn_set = set(hn_ids)

    p_list = p_preds[query_id]
    c_list = c_preds[query_id] if (c_preds is not None and query_id in c_preds) else None
    p_rank_map = {cid: i for i, cid in enumerate(p_list, start=1)}
    c_rank_map = {cid: i for i, cid in enumerate(c_list, start=1)} if c_list is not None else {}

    def _build_candidates(ranked: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "rank": idx,
                "image_id": cid,
                "image_url": _image_url(ds, cid),
                "is_gt": cid in gt_set,
                "is_ref": cid == rec.ref_id,
                "is_hard_neg": cid in hn_set,
            }
            for idx, cid in enumerate(ranked[:top_k], start=1)
        ]

    # Cross-model consensus across all completed models on this exact query
    all_models_eval: List[Dict[str, Any]] = []
    regimes_to_scan = ("indomain", "cirr_transfer", "fiq_transfer") if ds == "edir" else ("indomain", "fiq_transfer")
    for m in summary["models"]:
        mk = m["key"]
        for rg in regimes_to_scan:
            pr, _ = _load_preds_for(ds, mk, rg)
            if pr is None or query_id not in pr:
                continue
            ev = _score_single_query(ds, rec, pr[query_id])
            if ev is None:
                continue
            all_models_eval.append({
                "model": mk,
                "display": m["display"],
                "family": m["family"],
                "regime": rg,
                "primary_score": ev["primary_score"],
                "best_gt_rank": ev["best_gt_rank"],
                "hit1": ev["hit1"],
                "hit10": ev["hit10"],
                "top1_is_ref": ev["top1_is_ref"],
                "top1_is_hneg": ev["top1_is_hneg"],
            })

    return {
        "dataset": ds,
        "query": {
            "query_id": rec.query_id,
            "text": rec.text,
            "category": rec.category,
            "l1_interest": rec.metadata.get("l1_interest"),
            "length_category": rec.metadata.get("length_category"),
            "is_imperative": rec.metadata.get("is_imperative"),
            "is_question": rec.metadata.get("is_question"),
            "ref_image_id": rec.ref_id,
            "ref_image_url": _image_url(ds, rec.ref_id),
        },
        "ground_truths": [
            {
                "image_id": gid,
                "image_url": _image_url(ds, gid),
                "primary_rank": p_rank_map.get(gid),
                "compare_rank": c_rank_map.get(gid) if c_list is not None else None,
            }
            for gid in gt_ids
        ],
        "hard_negatives": [
            {
                "image_id": nid,
                "image_url": _image_url(ds, nid),
                "is_ref": nid == rec.ref_id,
                "primary_rank": p_rank_map.get(nid),
                "compare_rank": c_rank_map.get(nid) if c_list is not None else None,
            }
            for nid in hn_ids
        ],
        "primary": {
            "model": chosen_model,
            "display": MODEL_METADATA[chosen_model]["display"],
            "regime": regime,
            "variant": (p_res or {}).get("variant"),
            "metrics": _score_single_query(ds, rec, p_list),
            "candidates": _build_candidates(p_list),
        },
        "compare": (
            {
                "model": cmp_m,
                "display": MODEL_METADATA[cmp_m]["display"],
                "regime": cmp_r,
                "variant": (c_res or {}).get("variant"),
                "metrics": _score_single_query(ds, rec, c_list),
                "candidates": _build_candidates(c_list),
            }
            if c_list is not None
            else None
        ),
        "all_models": all_models_eval,
    }


@edir_pinpoint_router.get("/{dataset}/images/{image_id:path}")
def benchmark_image(dataset: str, image_id: str):
    ds = _validate_dataset(dataset)
    img_map = _get_image_map(ds)
    path = img_map.get(image_id)
    if path is None or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error": "image_not_found", "message": f"Ảnh không tồn tại trong corpus {ds}: {image_id!r}"},
        )
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@edir_pinpoint_router.get("/{dataset}/export.csv")
def benchmark_export_csv(dataset: str, slice: str = Query(default="all")) -> PlainTextResponse:
    ds = _validate_dataset(dataset)
    summary = _get_summary(ds)
    buf = io.StringIO()
    writer = csv.writer(buf)

    if ds == "edir":
        writer.writerow([
            "model", "family", "backbone", "stage0_R@1",
            "fiq_variant", "fiq_R@1", "fiq_R@10",
            "cirr_variant", "cirr_R@1", "cirr_R@10",
            "indomain_best_ep", "indomain_R@1", "indomain_R@5", "indomain_R@10", "indomain_R@50", "indomain_MRR", "indomain_Macro15_R@1",
            "delta_indomain_minus_cirr_R@1",
        ])
        for m in summary["models"]:
            s0 = m.get("stage0") or {}
            fq = (m["fiq_transfer"].get("overall") or {})
            cr = (m["cirr_transfer"].get("overall") or {})
            ind = (m["indomain"].get("overall") or {})
            mac = (m["indomain"].get("macro_subcategory") or {})
            writer.writerow([
                m["key"], m["family"], m["backbone"], s0.get("Recall@1", ""),
                m["fiq_transfer"].get("variant") or m["fiq_transfer"].get("status"), fq.get("Recall@1", ""), fq.get("Recall@10", ""),
                m["cirr_transfer"].get("variant") or m["cirr_transfer"].get("status"), cr.get("Recall@1", ""), cr.get("Recall@10", ""),
                m["indomain"].get("best_epoch", ""), ind.get("Recall@1", ""), ind.get("Recall@5", ""), ind.get("Recall@10", ""), ind.get("Recall@50", ""), ind.get("MRR", ""), mac.get("Recall@1", ""),
                m["deltas"].get("indomain_minus_cirr_r1", ""),
            ])
    else:
        valid_slices = {s["id"] for s in summary["slices"]}
        if slice not in valid_slices:
            raise HTTPException(status_code=400, detail={"error": "invalid_slice", "message": f"Slice không hợp lệ: {slice!r}"})
        writer.writerow([
            "model", "family", "backbone", "slice", "stage0_all_mAP@10",
            "fiq_variant", "fiq_mAP@10", "fiq_P@1", "fiq_NegRecall@10",
            "indomain_best_ep", "indomain_mAP@10", "indomain_P@1", "indomain_P@10", "indomain_NegRecall@10", "indomain_mAP@10_noNeg",
            "delta_indomain_minus_fiq_mAP@10",
        ])
        for m in summary["models"]:
            s0 = m.get("stage0") or {}
            fq = (m["fiq_transfer"].get("slices") or {}).get(slice) or {}
            ind = (m["indomain"].get("slices") or {}).get(slice) or {}
            writer.writerow([
                m["key"], m["family"], m["backbone"], slice, s0.get("mAP@10", ""),
                m["fiq_transfer"].get("variant") or m["fiq_transfer"].get("status"), fq.get("mAP@10", ""), fq.get("precision@1", ""), fq.get("NegRecall@10", ""),
                m["indomain"].get("best_epoch", ""), ind.get("mAP@10", ""), ind.get("precision@1", ""), ind.get("precision@10", ""), ind.get("NegRecall@10", ""), ind.get("mAP@10_noNeg", ""),
                (m.get("slice_deltas") or {}).get(slice, ""),
            ])

    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="cir_{ds}_{slice}.csv"'},
    )


# =============================================================================
# Unified Common Failures Explorer (EDIR + PinPoint + FashionIQ 3 Protocols)
# Zero-click default: automatically aggregates ALL ready models of the chosen
# protocol and returns common failures immediately (optional filtering on top).
# =============================================================================

COMMON_FAILURE_PROTOCOLS: List[Dict[str, Any]] = [
    {"id": "pinpoint_indomain", "group": "PinPoint", "label": "PinPoint · In-Domain", "dataset": "pinpoint", "regime": "indomain"},
    {"id": "pinpoint_fiq_transfer", "group": "PinPoint", "label": "PinPoint · FashionIQ Transfer", "dataset": "pinpoint", "regime": "fiq_transfer"},
    {"id": "edir_indomain", "group": "EDIR", "label": "EDIR · In-Domain", "dataset": "edir", "regime": "indomain"},
    {"id": "edir_cirr_transfer", "group": "EDIR", "label": "EDIR · CIRR Transfer", "dataset": "edir", "regime": "cirr_transfer"},
    {"id": "edir_fiq_transfer", "group": "EDIR", "label": "EDIR · FashionIQ Transfer", "dataset": "edir", "regime": "fiq_transfer"},
    {"id": "fashioniq_original_split", "group": "FashionIQ", "label": "FashionIQ · Original Split", "dataset": "fashioniq", "regime": "fashioniq_original_split"},
    {"id": "fashioniq_full_gallery_ref_excluded", "group": "FashionIQ", "label": "FashionIQ · Full Gallery (Ref-Excl)", "dataset": "fashioniq", "regime": "fashioniq_full_gallery_ref_excluded"},
    {"id": "fashioniq_val_split", "group": "FashionIQ", "label": "FashionIQ · Val Split", "dataset": "fashioniq", "regime": "fashioniq_val_split"},
    {"id": "fashioniq_all_protocols", "group": "FashionIQ", "label": "FashionIQ · Gộp cả 3 protocol (khác gallery)", "dataset": "fashioniq", "regime": "all"},
]

_CF_PROTOCOL_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in COMMON_FAILURE_PROTOCOLS}
_FIQ_CACHE: Dict[str, Tuple[Tuple[str, ...], Dict[str, Any]]] = {}
_CF_BENCH_CACHE: Dict[Tuple[str, str], Tuple[Tuple[Tuple[str, int], ...], Dict[str, Any]]] = {}


def _load_bench_protocol_matrix(ds: str, regime: str, protocol: str) -> Dict[str, Any]:
    """Build & mtime-cache per-query ranks across all ready models for an EDIR/PinPoint protocol."""
    sig = _signature_for_dataset(ds)
    ck = (ds, regime)
    with _CACHE_LOCK:
        hit = _CF_BENCH_CACHE.get(ck)
        if hit is not None and hit[0] == sig:
            return hit[1]

    bundle = _get_bundle(ds)
    summary = _get_summary(ds)
    categories = summary["subcategories"] if ds == "edir" else summary["categories"]
    slices_meta = summary["slices"] if ds == "pinpoint" else []
    all_models = []
    preds_by_model: Dict[str, Dict[str, List[str]]] = {}
    for m in summary["models"]:
        if not m[regime].get("has_predictions"):
            continue
        pr, _ = _load_preds_for(ds, m["key"], regime)
        if pr is None:
            continue
        all_models.append({
            "key": m["key"],
            "display": m["display"],
            "family": m["family"],
            "protocol_id": protocol,
        })
        preds_by_model[m["key"]] = pr

    base_queries = []
    for rec in bundle.queries:
        gt_ids = _clean_unique_ids(rec.positive_ids_raw or rec.positive_ids) if ds == "pinpoint" else _clean_unique_ids(rec.positive_ids)
        gt_set = set(gt_ids)
        first_gt = gt_ids[0] if gt_ids else ""
        ranks_map: Dict[str, Optional[int]] = {}
        ref_copy_map: Dict[str, bool] = {}
        scores_map: Dict[str, float] = {}
        for m_item in all_models:
            mk = m_item["key"]
            ranked = preds_by_model[mk].get(rec.query_id) or []
            br = next((i for i, cid in enumerate(ranked, start=1) if cid in gt_set), None)
            ranks_map[mk] = br
            ref_copy_map[mk] = bool(ranked and ranked[0] == rec.ref_id)
            if ds == "pinpoint":
                pos_raw = list(rec.positive_ids_raw or rec.positive_ids)
                scores_map[mk] = float(_pp_metrics_at_k(ranked, pos_raw, 10)["ap"])
            else:
                scores_map[mk] = (1.0 / br) if br is not None else 0.0
        base_queries.append({
            "rec": rec,
            "query_id": rec.query_id,
            "text": rec.text,
            "category": rec.category,
            "l1_interest": rec.metadata.get("l1_interest"),
            "ref_image_id": rec.ref_id,
            "ref_image_url": _image_url(ds, rec.ref_id),
            "gt_image_id": first_gt,
            "gt_image_url": _image_url(ds, first_gt) if first_gt else "",
            "ranks": ranks_map,
            "ref_copies": ref_copy_map,
            "scores": scores_map,
        })

    payload = {
        "models": all_models,
        "queries": base_queries,
        "categories": categories,
        "slices": slices_meta,
    }
    with _CACHE_LOCK:
        _CF_BENCH_CACHE[ck] = (sig, payload)
    return payload


def _load_fashioniq_protocol_matrix(protocol_filter: str) -> Dict[str, Any]:
    """Load per-query ranks across all DuckDB FashionIQ runs for a given protocol (cached)."""
    runs = list_runs()
    if protocol_filter == "all":
        chosen = [r for r in runs if str(r.get("protocol_id", "")).startswith("fashioniq_")]
    else:
        chosen = [r for r in runs if r.get("protocol_id") == protocol_filter]
    chosen.sort(key=lambda r: (str(r.get("model_id", "")), str(r.get("run_id", ""))))
    run_ids = tuple(str(r["run_id"]) for r in chosen)
    with _CACHE_LOCK:
        hit = _FIQ_CACHE.get(protocol_filter)
        if hit is not None and hit[0] == run_ids:
            return hit[1]

    if not run_ids:
        payload = {"models": [], "queries": [], "categories": []}
        with _CACHE_LOCK:
            _FIQ_CACHE[protocol_filter] = (run_ids, payload)
        return payload

    _meta, run_maps = get_analysis_input(list(run_ids), k=10)
    models_meta = []
    for r in chosen:
        mid = str(r.get("model_id") or r["run_id"])
        models_meta.append({
            "key": str(r["run_id"]),
            "display": mid,
            "family": str(r.get("protocol_id", "")).replace("fashioniq_", ""),
            "protocol_id": str(r.get("protocol_id", "")),
        })

    # Union of all query IDs across selected runs
    q_info: Dict[str, Dict[str, Any]] = {}
    for rid in run_ids:
        rmap = run_maps.get(rid, {})
        for qid, row in rmap.items():
            if qid not in q_info:
                cat = str(row.get("category") or "all")
                ref_id = str(row.get("reference_id") or "")
                tgt_id = str(row.get("target_id") or "")
                raw_caps = row.get("raw_captions") or []
                text = " / ".join(str(c) for c in raw_caps if c) if raw_caps else str(row.get("model_input_text") or "")
                q_info[qid] = {
                    "query_id": qid,
                    "text": text,
                    "category": cat,
                    "l1_interest": f"FashionIQ · {cat}",
                    "ref_image_id": ref_id,
                    "ref_image_url": f"/api/images/{cat}/{urllib.parse.quote(ref_id, safe='')}" if ref_id else "",
                    "gt_image_id": tgt_id,
                    "gt_image_url": f"/api/images/{cat}/{urllib.parse.quote(tgt_id, safe='')}" if tgt_id else "",
                    "ranks": {},
                    "ref_copies": {},
                    "scores": {},
                }
            tr = row.get("target_rank")
            q_info[qid]["ranks"][rid] = int(tr) if tr is not None else None
            q_info[qid]["ref_copies"][rid] = False
            q_info[qid]["scores"][rid] = (1.0 / int(tr)) if (tr is not None and int(tr) > 0) else 0.0

    cat_counts: Dict[str, int] = {}
    for q in q_info.values():
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    categories = [{"id": c, "count": cat_counts[c]} for c in sorted(cat_counts)]
    payload = {
        "models": models_meta,
        "queries": list(q_info.values()),
        "categories": categories,
    }
    with _CACHE_LOCK:
        _FIQ_CACHE[protocol_filter] = (run_ids, payload)
    return payload


def _build_common_failures_payload(
    protocol: str,
    models_param: str,
    preset: str,
    k: int,
    category: str,
    slice_id: str,
    q_text: str,
    sort: str,
    offset: int,
    limit: int,
) -> Dict[str, Any]:
    spec = _CF_PROTOCOL_BY_ID.get(protocol)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_protocol", "message": f"Protocol không hợp lệ: {protocol!r}"},
        )
    if preset not in ("fail80", "all_fail", "fail50", "disagree", "ref_copy50", "all"):
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_preset", "message": f"Preset không hợp lệ: {preset!r}"},
        )

    ds = spec["dataset"]
    regime = spec["regime"]

    # Build protocol summary list with live ready model counts
    available_protocols = []
    fiq_runs = list_runs()
    for pdef in COMMON_FAILURE_PROTOCOLS:
        if pdef["dataset"] in ("edir", "pinpoint"):
            sm = _get_summary(pdef["dataset"])
            n_ready = sum(1 for m in sm["models"] if m[pdef["regime"]].get("has_predictions"))
        elif pdef["regime"] == "all":
            n_ready = sum(1 for r in fiq_runs if str(r.get("protocol_id", "")).startswith("fashioniq_"))
        else:
            n_ready = sum(1 for r in fiq_runs if r.get("protocol_id") == pdef["regime"])
        available_protocols.append({**pdef, "ready_models": n_ready})

    # Collect per-query ranks across all ready models in this protocol
    slices_meta: List[Dict[str, Any]] = []
    if ds == "fashioniq":
        fiq_data = _load_fashioniq_protocol_matrix(regime)
        all_models = fiq_data["models"]
        categories = fiq_data["categories"]
        base_queries = fiq_data["queries"]
    else:
        bm_data = _load_bench_protocol_matrix(ds, regime, protocol)
        all_models = bm_data["models"]
        categories = bm_data["categories"]
        slices_meta = bm_data["slices"]
        if ds == "pinpoint" and slice_id != "all":
            base_queries = [r for r in bm_data["queries"] if _pinpoint_slice_match(r["rec"], slice_id)]
        else:
            base_queries = bm_data["queries"]

    all_model_keys = [m["key"] for m in all_models]
    if models_param.strip():
        requested = [x.strip() for x in models_param.split(",") if x.strip()]
        selected_keys = [k_m for k_m in all_model_keys if k_m in set(requested)]
        if not selected_keys:
            selected_keys = list(all_model_keys)
    else:
        selected_keys = list(all_model_keys)

    selected_models = [m for m in all_models if m["key"] in set(selected_keys)]
    n_sel = len(selected_keys)
    q_lower = q_text.strip().lower()
    cat_filter = category.strip()

    preset_counts = {"all": 0, "all_fail": 0, "fail80": 0, "fail50": 0, "disagree": 0, "ref_copy50": 0}
    cat_stats: Dict[str, Dict[str, int]] = {}
    evaluated_rows: List[Dict[str, Any]] = []

    for row in base_queries:
        if q_lower:
            hay = f"{row['query_id']} {row['text']} {row['category']} {row.get('l1_interest') or ''}".lower()
            if q_lower not in hay:
                continue

        scored_models = 0
        fail_count = 0
        hit_count = 0
        ref_copy_count = 0
        finite_ranks: List[int] = []
        per_model = []

        for m_item in selected_models:
            mk = m_item["key"]
            if mk not in row["ranks"]:
                continue
            scored_models += 1
            rk = row["ranks"][mk]
            failed = rk is None or rk > k
            if failed:
                fail_count += 1
            else:
                hit_count += 1
            if rk is not None:
                finite_ranks.append(rk)
            rc = bool(row["ref_copies"].get(mk, False))
            if rc:
                ref_copy_count += 1
            per_model.append({
                "key": mk,
                "display": m_item["display"],
                "rank": rk,
                "failed": failed,
                "ref_copy": rc,
                "score": row["scores"].get(mk, 0.0),
            })

        if scored_models == 0:
            continue

        fail_rate = fail_count / scored_models
        ref_copy_rate = ref_copy_count / scored_models
        is_all_fail = fail_count == scored_models
        is_fail80 = fail_rate >= 0.8 - 1e-9
        is_fail50 = fail_rate >= 0.5 - 1e-9
        is_disagree = hit_count >= 1 and fail_count >= 1 and (hit_count <= max(2, int(round(scored_models * 0.25))))
        is_ref_copy50 = ref_copy_rate >= 0.5 - 1e-9

        # Track category common-failure stats BEFORE category filter so bars stay informative
        c_id = row["category"]
        cs = cat_stats.setdefault(c_id, {"total": 0, "all_fail": 0, "fail80": 0})
        cs["total"] += 1
        if is_all_fail:
            cs["all_fail"] += 1
        if is_fail80:
            cs["fail80"] += 1

        if cat_filter and c_id != cat_filter:
            continue

        preset_counts["all"] += 1
        if is_all_fail:
            preset_counts["all_fail"] += 1
        if is_fail80:
            preset_counts["fail80"] += 1
        if is_fail50:
            preset_counts["fail50"] += 1
        if is_disagree:
            preset_counts["disagree"] += 1
        if is_ref_copy50:
            preset_counts["ref_copy50"] += 1

        keep = (
            preset == "all"
            or (preset == "all_fail" and is_all_fail)
            or (preset == "fail80" and is_fail80)
            or (preset == "fail50" and is_fail50)
            or (preset == "disagree" and is_disagree)
            or (preset == "ref_copy50" and is_ref_copy50)
        )
        if not keep:
            continue

        best_rank = min(finite_ranks) if finite_ranks else None
        median_rank = sorted(finite_ranks)[len(finite_ranks) // 2] if finite_ranks else None
        mean_score = sum(pm["score"] for pm in per_model) / len(per_model)
        evaluated_rows.append({
            "query_id": row["query_id"],
            "text": row["text"],
            "category": c_id,
            "l1_interest": row.get("l1_interest"),
            "ref_image_id": row["ref_image_id"],
            "ref_image_url": row["ref_image_url"],
            "gt_image_id": row["gt_image_id"],
            "gt_image_url": row["gt_image_url"],
            "scored_models": scored_models,
            "fail_count": fail_count,
            "hit_count": hit_count,
            "fail_rate": fail_rate,
            "ref_copy_count": ref_copy_count,
            "best_rank": best_rank,
            "median_rank": median_rank,
            "mean_score": mean_score,
            "per_model": per_model,
        })

    if sort == "fail_desc":
        evaluated_rows.sort(key=lambda r: (-r["fail_rate"], -r["fail_count"], r["mean_score"], r["query_id"]))
    elif sort == "ref_copy_desc":
        evaluated_rows.sort(key=lambda r: (-r["ref_copy_count"], -r["fail_rate"], r["query_id"]))
    elif sort == "best":
        evaluated_rows.sort(key=lambda r: (r["fail_rate"], -r["mean_score"], r["query_id"]))
    else:
        evaluated_rows.sort(key=lambda r: r["query_id"])

    category_breakdown = sorted(
        [
            {
                "category": c,
                "total": st["total"],
                "all_fail": st["all_fail"],
                "fail80": st["fail80"],
                "fail80_rate": (st["fail80"] / st["total"]) if st["total"] > 0 else 0.0,
            }
            for c, st in cat_stats.items()
        ],
        key=lambda x: (-x["fail80_rate"], -x["fail80"], x["category"]),
    )

    page = evaluated_rows[offset : offset + limit]
    return {
        "protocol": protocol,
        "protocol_label": spec["label"],
        "dataset": ds,
        "regime": regime,
        "k": k,
        "preset": preset,
        "available_protocols": available_protocols,
        "all_models": all_models,
        "selected_models": selected_keys,
        "selected_model_count": n_sel,
        "slices": slices_meta,
        "categories": categories,
        "category_breakdown": category_breakdown,
        "preset_counts": preset_counts,
        "filtered_total": len(evaluated_rows),
        "offset": offset,
        "limit": limit,
        "items": page,
    }
