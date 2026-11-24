#!/usr/bin/env python3
"""Independent bit-exact verifier & master comparison report generator across:
  - PinPoint: Base Zero-Shot vs FashionIQ Transfer (14/15) vs In-Domain Fine-tuned (15/15)
    sliced cleanly across All (1,337), Fashion All (351), Fashion excl. Complement (338),
    Fashion-Swap (197), Fashion-Context Fit (107), Fashion-Negation (29),
    Fashion-Complement (13), and Non-Fashion (986).
  - EDIR: Base Zero-Shot vs FashionIQ Transfer (14/15) vs CIRR Transfer (10/15) vs In-Domain Fine-tuned (15/15)
    across Overall R@1/5/10/50, MRR, Macro-15, and all 15 edit subcategories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path("/home/heheboiz/data/cir")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.benchmarks import evaluate_edir, evaluate_pinpoint, load_split_bundle  # noqa: E402
from workbench.benchmarks.trainers.run_benchmark_suite import MODEL_QUEUE  # noqa: E402

DOCS_DIR = REPO / "workbench" / "docs" / "reviews" / "20261123-edir-pinpoint-splits"
INDOMAIN_ROOT = REPO / "workbench" / "artifacts" / "benchmark_runs"
TRANSFER_ROOT = REPO / "workbench" / "artifacts" / "transfer_evals"
SPLIT_TAG = "70_10_20_seed42"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_pinpoint(preds: Dict[str, List[str]], queries: Tuple[Any, ...]) -> Dict[str, Dict[str, Any]]:
    """Compute exact PinPoint metrics across all 8 clean evaluation slices from 1 top-50 prediction set."""
    slices = {
        "all": list(queries),
        "fashion_all": [q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", ""))],
        "fashion_no_comp": [
            q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", "")) and q.category != "Complement"
        ],
        "fashion_swap": [
            q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", "")) and q.category == "Swap"
        ],
        "fashion_ctx": [
            q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", "")) and q.category == "Context Fit"
        ],
        "fashion_neg": [
            q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", "")) and q.category == "Negation"
        ],
        "fashion_comp": [
            q for q in queries if "Fashion" in str(q.metadata.get("l1_interest", "")) and q.category == "Complement"
        ],
        "non_fashion": [q for q in queries if "Fashion" not in str(q.metadata.get("l1_interest", ""))],
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, q_sub in slices.items():
        sub_preds = {q.query_id: preds[q.query_id] for q in q_sub}
        res = evaluate_pinpoint(sub_preds, tuple(q_sub), ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
        out[name] = res["overall"]
    return out


def verify_and_generate_pinpoint() -> Tuple[Path, Dict[str, Any]]:
    bundle = load_split_bundle("pinpoint", "test", corpus_scope="full", split_tag=SPLIT_TAG)
    assert len(bundle.queries) == 1337
    max_diff = 0.0
    rows_summary: List[Dict[str, Any]] = []

    for spec in MODEL_QUEUE:
        key = spec["key"]
        in_dir = INDOMAIN_ROOT / "pinpoint" / SPLIT_TAG / spec["run_id"]
        in_res = _load_json(in_dir / "test_results.json")
        in_preds = _load_json(in_dir / "test_predictions_top50.json")
        fiq_dir = TRANSFER_ROOT / "fashioniq_to_pinpoint" / key
        fiq_res = _load_json(fiq_dir / "test_results.json")
        fiq_preds = _load_json(fiq_dir / "test_predictions_top50.json")

        in_slices = None
        if in_res and in_preds:
            in_slices = _slice_pinpoint(in_preds, bundle.queries)
            stored = in_res[spec["result_field"]]["overall"]
            for mk in ("mAP@10", "precision@1", "precision@10", "NegRecall@10", "mAP@10_noNeg"):
                diff = abs(in_slices["all"][mk] - stored[mk])
                max_diff = max(max_diff, diff)
                assert diff < 1e-12, f"PinPoint In-Domain mismatch {key} {mk}: {diff}"

        fiq_slices = None
        if fiq_res and fiq_preds:
            fiq_slices = _slice_pinpoint(fiq_preds, bundle.queries)
            stored = fiq_res["test_metrics"]["overall"]
            for mk in ("mAP@10", "precision@1", "precision@10", "NegRecall@10", "mAP@10_noNeg"):
                diff = abs(fiq_slices["all"][mk] - stored[mk])
                max_diff = max(max_diff, diff)
                assert diff < 1e-12, f"PinPoint FIQ-Transfer mismatch {key} {mk}: {diff}"

        s0 = ((in_res or {}).get("stage0_zeroshot_test") or (in_res or {}).get("stage0_init_test") or {}).get("overall", {})
        rows_summary.append({
            "key": key,
            "run_id": spec["run_id"],
            "s0": s0,
            "fiq_variant": (fiq_res or {}).get("variant", "N/A (no FIQ ckpt)"),
            "fiq": fiq_slices,
            "indomain": in_slices,
        })

    rows_summary.sort(key=lambda r: (r["indomain"]["all"]["mAP@10"] if r["indomain"] else -1.0), reverse=True)

    lines = [
        f"# So Sánh Đối Chiếu 3 Cột & Phân Rã Fashion Trên PinPoint (`{SPLIT_TAG}`)",
        "",
        f"- **Corpus chuẩn duy nhất:** `109,598` ảnh | **Test queries:** `1,337` (`Fashion All = 351`, `Fashion excl. Complement = 338`, `Fashion-Swap = 197`, `Fashion-Context Fit = 107`, `Fashion-Negation = 29`, `Fashion-Complement = 13`, `Non-Fashion = 986`).",
        f"- **Kiểm chứng độc lập bit-exact từ `test_predictions_top50.json`:** `max_abs_diff = {max_diff:.2e}` (khớp 100%).",
        "- **Ghi chú checkpoint không phát hành:** `mtst_ptha` không có checkpoint `FashionIQ` chính chủ từ tác giả; 4 model iLearn (`habit`, `intent`, `conesep`: `N0.2`; `airknow`: `N0.5`) là bản train kèm noise chính chủ; `limn` & `clvcnet` là 3-head ensemble (`dress + shirt + toptee`).",
        "",
        "## 1. Tổng Quan 3 Cột Trên Toàn Bộ PinPoint (`1,337` queries — `mAP@10`)",
        "",
        "| Model | FIQ Variant | Base Zero-Shot `mAP@10` | **FashionIQ Transfer `mAP@10`** | **In-Domain PinPoint `mAP@10`** | Delta (`In-Domain - FIQ`) | FIQ `NegRec@10` | In-Domain `NegRec@10` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_summary:
        s0_m = f"{r['s0']['mAP@10']:.4f}" if "mAP@10" in r["s0"] else "-"
        fiq_m = f"**{r['fiq']['all']['mAP@10']:.4f}**" if r["fiq"] else "N/A"
        in_m = f"**{r['indomain']['all']['mAP@10']:.4f}**" if r["indomain"] else "-"
        delta = f"{(r['indomain']['all']['mAP@10'] - r['fiq']['all']['mAP@10']):+.4f}" if (r["indomain"] and r["fiq"]) else "-"
        fiq_nr = f"{r['fiq']['all']['NegRecall@10']:.4f}" if r["fiq"] else "-"
        in_nr = f"{r['indomain']['all']['NegRecall@10']:.4f}" if r["indomain"] else "-"
        lines.append(f"| **{r['key']}** | `{r['fiq_variant']}` | {s0_m} | {fiq_m} | {in_m} | {delta} | {fiq_nr} | {in_nr} |")

    lines.extend([
        "",
        "## 2. Lát Cắt Sân Nhà `PinPoint-Fashion` vs `Non-Fashion` (Checkpoint FashionIQ Transfer — `mAP@10`)",
        "",
        "| Model | Variant | `All` (1337) | `Non-Fashion` (986) | `Fashion All` (351) | **`Fashion excl. Comp` (338)** | **`Fashion-Swap` (197)** | `Fashion-Ctx` (107) | `Fashion-Neg` (29) | `Fashion-Comp` (13) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in rows_summary:
        if not r["fiq"]:
            lines.append(f"| **{r['key']}** | `{r['fiq_variant']}` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        f = r["fiq"]
        lines.append(
            f"| **{r['key']}** | `{r['fiq_variant']}` | {f['all']['mAP@10']:.4f} | {f['non_fashion']['mAP@10']:.4f} | "
            f"{f['fashion_all']['mAP@10']:.4f} | **{f['fashion_no_comp']['mAP@10']:.4f}** | **{f['fashion_swap']['mAP@10']:.4f}** | "
            f"{f['fashion_ctx']['mAP@10']:.4f} | {f['fashion_neg']['mAP@10']:.4f} | {f['fashion_comp']['mAP@10']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Lát Cắt Tương Ứng Của Bản **In-Domain PinPoint Fine-tuned** (`mAP@10`)",
        "",
        "| Model | `All` (1337) | `Non-Fashion` (986) | `Fashion All` (351) | **`Fashion excl. Comp` (338)** | **`Fashion-Swap` (197)** | `Fashion-Ctx` (107) | `Fashion-Neg` (29) | `Fashion-Comp` (13) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in rows_summary:
        if not r["indomain"]:
            continue
        d = r["indomain"]
        lines.append(
            f"| **{r['key']}** | {d['all']['mAP@10']:.4f} | {d['non_fashion']['mAP@10']:.4f} | "
            f"{d['fashion_all']['mAP@10']:.4f} | **{d['fashion_no_comp']['mAP@10']:.4f}** | **{d['fashion_swap']['mAP@10']:.4f}** | "
            f"{d['fashion_ctx']['mAP@10']:.4f} | {d['fashion_neg']['mAP@10']:.4f} | {d['fashion_comp']['mAP@10']:.4f} |"
        )

    out_path = DOCS_DIR / "TRANSFER_AND_INDOMAIN_PINPOINT.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, {"max_diff": max_diff, "models": len(rows_summary)}


def verify_and_generate_edir() -> Tuple[Path, Dict[str, Any]]:
    bundle = load_split_bundle("edir", "test", corpus_scope="full", split_tag=SPLIT_TAG)
    assert len(bundle.queries) == 1003
    max_diff = 0.0
    rows_summary: List[Dict[str, Any]] = []

    for spec in MODEL_QUEUE:
        key = spec["key"]
        in_dir = INDOMAIN_ROOT / "edir" / SPLIT_TAG / spec["run_id"]
        in_res = _load_json(in_dir / "test_results.json")
        in_preds = _load_json(in_dir / "test_predictions_top50.json")
        fiq_dir = TRANSFER_ROOT / "fashioniq_to_edir" / key
        fiq_res = _load_json(fiq_dir / "test_results.json")
        fiq_preds = _load_json(fiq_dir / "test_predictions_top50.json")
        cirr_dir = TRANSFER_ROOT / "cirr_to_edir" / key
        cirr_res = _load_json(cirr_dir / "test_results.json")
        cirr_preds = _load_json(cirr_dir / "test_predictions_top50.json")

        for tag, res_obj, pred_obj, field in (
            ("indomain", in_res, in_preds, spec["result_field"]),
            ("fiq", fiq_res, fiq_preds, "test_metrics"),
            ("cirr", cirr_res, cirr_preds, "test_metrics"),
        ):
            if res_obj and pred_obj:
                recomp = evaluate_edir(pred_obj, bundle.queries, ks=(1, 5, 10, 50), strict=True)
                stored = res_obj[field]["overall"]
                for mk in ("Recall@1", "Recall@5", "Recall@10", "Recall@50", "MRR"):
                    diff = abs(recomp["overall"][mk] - stored[mk])
                    max_diff = max(max_diff, diff)
                    assert diff < 1e-12, f"EDIR {tag} mismatch {key} {mk}: {diff}"

        s0 = ((in_res or {}).get("stage0_zeroshot_test") or (in_res or {}).get("stage0_init_test") or {}).get("overall", {})
        cirr_note = (cirr_res or {}).get("variant")
        if not cirr_note:
            cirr_note = {
                "tme": "N/A (Kaggle 404)",
                "airknow": "N/A (no CIRR release)",
                "conesep": "N/A (no CIRR release)",
                "clvcnet": "N/A (2021 pre-CIRR)",
                "dcnet": "N/A (2021 pre-CIRR)",
            }.get(key, "N/A")

        rows_summary.append({
            "key": key,
            "s0": s0,
            "fiq_variant": (fiq_res or {}).get("variant", "N/A (no FIQ ckpt)"),
            "fiq": (fiq_res or {}).get("test_metrics"),
            "cirr_variant": cirr_note,
            "cirr": (cirr_res or {}).get("test_metrics"),
            "indomain": (in_res or {}).get(spec["result_field"]),
        })

    rows_summary.sort(key=lambda r: (r["indomain"]["overall"]["Recall@1"] if r["indomain"] else -1.0), reverse=True)

    lines = [
        f"# So Sánh Đối Chiếu 4 Cột Trên EDIR (`{SPLIT_TAG}`)",
        "",
        f"- **Corpus chuẩn duy nhất:** `28,645` ảnh | **Test queries:** `1,003` (chia đều 15 phép chỉnh sửa độc lập).",
        f"- **Kiểm chứng độc lập bit-exact từ `test_predictions_top50.json`:** `max_abs_diff = {max_diff:.2e}` (khớp 100%).",
        "- **Ghi chú checkpoint không phát hành:** `CIRR`: `tme` (Kaggle 404), `airknow` & `conesep` (tác giả không up bản CIRR), `clvcnet` & `dcnet` (paper 2021 không chạy CIRR). `FashionIQ`: `mtst_ptha` (tác giả chỉ công bố bản CIRR).",
        "",
        "## 1. Tổng Quan 4 Cột Trên EDIR (`Recall@1` / `Recall@10` / `MRR`)",
        "",
        "| Model | Base Zero-Shot `R@1` | **FIQ Transfer `R@1` (`R@10`)** | **CIRR Transfer `R@1` (`R@10`)** | **In-Domain EDIR `R@1` (`R@10`)** | Delta (`In-Domain - CIRR`) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows_summary:
        s0_s = f"{r['s0']['Recall@1']*100:.2f}%" if "Recall@1" in r["s0"] else "-"
        fiq_s = f"{r['fiq']['overall']['Recall@1']*100:.2f}% ({r['fiq']['overall']['Recall@10']*100:.1f}%)" if r["fiq"] else "N/A"
        cirr_s = f"**{r['cirr']['overall']['Recall@1']*100:.2f}%** ({r['cirr']['overall']['Recall@10']*100:.1f}%)" if r["cirr"] else f"`{r['cirr_variant']}`"
        in_s = f"**{r['indomain']['overall']['Recall@1']*100:.2f}%** ({r['indomain']['overall']['Recall@10']*100:.1f}%)" if r["indomain"] else "-"
        delta = (
            f"{(r['indomain']['overall']['Recall@1'] - r['cirr']['overall']['Recall@1'])*100:+.2f}%"
            if (r["indomain"] and r["cirr"])
            else "-"
        )
        lines.append(f"| **{r['key']}** | {s0_s} | {fiq_s} | {cirr_s} | {in_s} | {delta} |")

    # Section 2: CIRR Transfer 15-subcategory Recall@1 matrix
    n_total = len(rows_summary)
    n_cirr = sum(1 for r in rows_summary if r["cirr"])
    n_fiq = sum(1 for r in rows_summary if r["fiq"])
    subcats = sorted(rows_summary[0]["indomain"]["per_subcategory"].keys())
    lines.extend(["", f"## 2. Ma Trận `Recall@1` Theo 15 Phép Chỉnh Sửa — **CIRR Transfer (`{n_cirr}/{n_total}` model)**", ""])
    lines.append("| Model | Variant | " + " | ".join(subcats) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in subcats) + "|")
    for r in rows_summary:
        if not r["cirr"]:
            continue
        ps = r["cirr"]["per_subcategory"]
        vals = [f"{ps[c]['Recall@1']*100:.1f}%" for c in subcats]
        lines.append(f"| **{r['key']}** | `{r['cirr_variant']}` | " + " | ".join(vals) + " |")

    lines.extend(["", f"## 3. Ma Trận `Recall@1` Theo 15 Phép Chỉnh Sửa — **FashionIQ Transfer (`{n_fiq}/{n_total}` model)**", ""])
    lines.append("| Model | Variant | " + " | ".join(subcats) + " |")
    lines.append("|---|---|" + "|".join("---:" for _ in subcats) + "|")
    for r in rows_summary:
        if not r["fiq"]:
            continue
        ps = r["fiq"]["per_subcategory"]
        vals = [f"{ps[c]['Recall@1']*100:.1f}%" for c in subcats]
        lines.append(f"| **{r['key']}** | `{r['fiq_variant']}` | " + " | ".join(vals) + " |")

    out_path = DOCS_DIR / "TRANSFER_AND_INDOMAIN_EDIR.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, {"max_diff": max_diff, "models": len(rows_summary)}


def main() -> None:
    p_path, p_stat = verify_and_generate_pinpoint()
    e_path, e_stat = verify_and_generate_edir()
    print(f"[REPORT OK] {p_path} (max_diff={p_stat['max_diff']:.2e})")
    print(f"[REPORT OK] {e_path} (max_diff={e_stat['max_diff']:.2e})")


if __name__ == "__main__":
    main()
