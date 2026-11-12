"""Controlled native CLVC-Net R@1/R@10/R@50 run with per-query ranks and full provenance.

Two interpreters, on purpose:

* CONTROLLER — `workbench/envs/ctl312h/bin/python`. Orchestrates only: spawns the worker, merges the
  worker's per-category files into `summary.json`. The gate refuses `clvc39-la` as composition
  controller, so the worker is a subprocess of this process.
* WORKER — `workbench/envs/clvc39-la/bin/python`, the interpreter the recorded 30.0208 command used
  (`BLOCKED_MODELS_VERIFICATION.vi.md:215`). It unpickles the author's whole-module checkpoints and
  executes the pinned `test.test`.

Per-query evidence without reimplementing the evaluator: pinned `test.py:138` calls
`np.argsort(-sims[i, :])[:50]` once per query, so a scoped wrapper on `numpy.argsort` records the
similarity row AND the exact ordering the native ranking used, while `test.test` still returns its
own numbers. Recomputed recalls must equal the native ones exactly or the run fails.

Batch size travels only through `params.batch_size` (`test.py:96,112`). The pinned source is never
edited and no env var is needed: the wrapper passes the namespace through the native signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]          # .../cir
RAW = HERE.parents[1] / "raw" / "clvc_native"   # .../20261112-parity-repair/raw/clvc_native
SOURCE_ROOT = REPO / "workbench/third_party/CLVCNet"
CHECKPOINT_ROOT = (REPO / "workbench/artifacts/checkpoint_selection/20261106T002617+0700"
                   "/checkpoints/clvc")
DATASET_ROOT = REPO / "data/FashionIQ"
MODEL_PYTHON = REPO / "workbench/envs/clvc39-la/bin/python"

CATEGORIES = ("dress", "shirt", "toptee")
FIXED_BATCH = 32  # declared before the run: the value the 30.0208 artifact recorded
PROBE_BATCH = 1
PROBE_CATEGORY = "dress"  # batch-1 sensitivity pass, one category (cost, see CLVC_NATIVE.md)

MASK_CONSTANT = -10e10  # test.py:135
GALLERY_CONSTRUCTION = (
    "datasets.FashionIQ.get_test_targets() with gallery_all=False (datasets.py:233-257): the gallery "
    "is the order-preserving union of every query's candidate (source) and target image, in first-"
    "appearance order -- NOT the full split image list, NOT per-query candidates only")
REFERENCE_POLICY = (
    "the query's own source image IS in the gallery and is masked to -10e10 before ranking "
    "(test.py:134-135), i.e. excluded by score, not by removal")
TIE_POLICY = (
    "np.argsort(-sims[i, :]) with numpy default kind='quicksort' (test.py:138). NOT stable: equal "
    "scores take an arbitrary positional order, so a query whose target ties with another gallery "
    "image is scored by that arbitrary order. target_rank here is the native positional rank; "
    "tied_with_target > 0 marks a tie-decided rank")
TEXT_PATH = (
    "no text correction anywhere: captions reach the model as "
    "'<BOS> {c0} <AND> {c1} <EOS>' (datasets.py:173-175) and are tokenised by the checkpoint's own "
    "SimpleVocab (text_model.py:23-27)")
TRANSFORM = ("Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet 0.485/0.456/0.406, "
             "0.229/0.224/0.225) -- the native FashionIQ val pipeline, train.py:90-96")
BATCH_PLUMBING = (
    "params.batch_size, read at test.py:96 (query loop) and test.py:112 (target/gallery loop); "
    "the loop flushes at len(imgs) >= params.batch_size. Passed as a SimpleNamespace attribute -- "
    "pinned test.py has no CLI, so this is the only argument path, and the checkpoint/pinned files "
    "stay byte-identical")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------- worker (3.9) ----

def execute(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms

    sys.path.insert(0, str(REPO))
    from workbench.backend.registry import model_by_id
    from workbench.replay.common import load_trusted_whole_model

    entry = model_by_id("clvc_net")
    variant = entry["checkpoint_variants"][0]
    declared = {m["filename"]: m["expected_sha256"] for m in variant["bundle_members"]}

    sys.path.insert(0, str(SOURCE_ROOT))
    import datasets  # pinned upstream
    import test as native_test  # pinned upstream

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    out_root = Path(args.output_dir)
    pass_dir = out_root / args.pass_label
    pass_dir.mkdir(parents=True, exist_ok=True)

    real_argsort = np.argsort
    capture: list[dict] = []
    capturing = False

    def recording_argsort(a, *rest, **kwargs):
        order = real_argsort(a, *rest, **kwargs)
        if capturing:
            capture.append({"row": np.array(a, copy=True), "order": np.array(order, copy=True)})
        return order

    np.argsort = recording_argsort

    for category in args.categories:
        local_path = CHECKPOINT_ROOT / f"{category}_local_model.pt"
        global_path = CHECKPOINT_ROOT / f"{category}_global_model.pt"
        local_model = load_trusted_whole_model(local_path, declared[local_path.name], torch)
        global_model = load_trusted_whole_model(global_path, declared[global_path.name], torch)
        local_model = local_model.to("cuda").eval()
        global_model = global_model.to("cuda").eval()

        testset = datasets.FashionIQ(
            path=str(DATASET_ROOT).rstrip("/") + "/",
            gallery_all=False, name=category, split="val", transform=transform)

        params = SimpleNamespace(batch_size=args.batch_size)
        capture.clear()
        capturing = True
        started = time.monotonic()
        raw = native_test.test(params, local_model, global_model, testset, category)
        wall = time.monotonic() - started
        capturing = False

        native = dict(raw)
        # `test.test` already populated both caches (test.py:84-85); rebuilding here would decode
        # every query image a second time for nothing.
        queries = testset.test_queries
        targets = testset.test_targets
        gallery_ids = [t["target_img_id"] for t in targets]
        gallery_count = len(gallery_ids)
        assert len(capture) == len(queries), (len(capture), len(queries))

        records = []
        for i, query in enumerate(queries):
            order = capture[i]["order"]
            row = capture[i]["row"]  # == -sims[i, :]
            target_pos = gallery_ids.index(query["target_img_id"])
            reference_pos = gallery_ids.index(query["source_img_id"])
            hits = np.nonzero(order == target_pos)[0]
            rank = int(hits[0]) if len(hits) else None
            target_score = float(-row[target_pos])
            records.append({
                "qid": f"{category}:{i:05d}:{testset.images[query['source_img_id']]}"
                       f"->{testset.images[query['target_img_id']]}",
                "category": category,
                "query_index": i,
                "candidate_id": testset.images[query["source_img_id"]],
                "target_id": testset.images[query["target_img_id"]],
                "reference_gallery_index": reference_pos,
                "target_gallery_index": target_pos,
                "target_rank": rank,
                "score_at_target_rank": target_score,
                "score_at_prev_rank": float(-row[int(order[rank - 1])]) if rank else None,
                "tied_with_target": int(np.count_nonzero(row == row[target_pos])) - 1,
                # The mask puts the reference at the BOTTOM, not out of the array, so a check
                # against the full ordering is always true and says nothing. What matters is where
                # the mask actually left it: its positional rank in the native ordering.
                "reference_rank": int(np.nonzero(order == reference_pos)[0][0]),
                "reference_masked_last": bool(order[-1] == reference_pos),
                "batch_size": args.batch_size,
                "tie_rule": "argsort_quicksort_positional",
            })

        recomputed = {}
        for k in (1, 10, 50):
            hit = sum(1 for r in records if r["target_rank"] is not None and r["target_rank"] < k)
            recomputed[f"r{k}"] = 100.0 * hit / len(records)
        mismatched = [k for k in recomputed
                      if abs(recomputed[k] - native[f"{category}_{k}"]) > 1e-9]

        payload = {
            "category": category,
            "query_count": len(records),
            "gallery_count": gallery_count,
            "mask_policy": f"reference score set to {MASK_CONSTANT} (test.py:135); {REFERENCE_POLICY}",
            "tie_policy": TIE_POLICY,
            "gallery_construction": GALLERY_CONSTRUCTION,
            "text_path": TEXT_PATH,
            "transform": TRANSFORM,
            "batch_size_argument_plumbing": BATCH_PLUMBING,
            "batch_size": args.batch_size,
            "native_recalls": {f"r{k}": native[f"{category}_r{k}"] for k in (1, 10, 50)},
            "recomputed_recalls": recomputed,
            "recomputed_matches_native": not mismatched,
            "mismatched_keys": mismatched,
            "queries_with_tie_at_target": sum(1 for r in records if r["tied_with_target"] > 0),
            "queries_with_target_rank_none": sum(1 for r in records if r["target_rank"] is None),
            "references_masked_last": sum(1 for r in records if r["reference_masked_last"]),
            "min_reference_rank": min(r["reference_rank"] for r in records),
            "checkpoint_digests": {p.name: sha256_file(p) for p in (local_path, global_path)},
            "checkpoint_root": str(CHECKPOINT_ROOT),
            "source_commit": entry["upstream_commit_sha"],
            "source_root": str(SOURCE_ROOT),
            "source_tree_status": source_status(entry),
            "dataset_root": str(DATASET_ROOT),
            "annotation_sha256": {
                name: sha256_file(DATASET_ROOT / rel) for name, rel in (
                    (f"cap.{category}.val.json", f"captions/cap.{category}.val.json"),
                    (f"split.{category}.val.json", f"image_splits/split.{category}.val.json"))},
            "interpreter": {"worker": sys.executable, "worker_version": platform.python_version(),
                            "controller": args.controller},
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "numpy_version": np.__version__,
            "device": "cuda",
            "wall_seconds": round(wall, 3),
            "native_evaluator": "test.py:test (pinned, unmodified)",
        }
        (pass_dir / f"{category}.json").write_text(json.dumps(payload, indent=2) + "\n")
        with (pass_dir / f"{category}.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(json.dumps({"pass": args.batch_size, "category": category,
                          "native": payload["native_recalls"],
                          "recomputed": recomputed,
                          "agree": payload["recomputed_matches_native"],
                          "wall_s": payload["wall_seconds"]}), flush=True)
        if mismatched:
            raise SystemExit(f"recomputing from captured ranks disagrees with native for "
                             f"{category}: {mismatched}")
        del local_model, global_model
        torch.cuda.empty_cache()

    return 0


def source_status(entry: dict) -> dict:
    """The repo's own vendored-source check.

    `git -C <vendored dir>` resolves to the OUTER repository here, because the vendored upstream has
    no `.git` of its own: it is pinned by a digest manifest. Asking the same helper the replay path
    uses keeps this run's claim identical to the gate's.
    """
    from workbench.backend.vendored_source import source_provenance

    provenance = source_provenance(entry, SOURCE_ROOT)
    return {"verified": provenance.get("verified"), "state": provenance.get("state"),
            "expected_commit": provenance.get("expected_commit"),
            "actual_commit": provenance.get("actual_commit"),
            "dirty": provenance.get("dirty"), "error": provenance.get("error"),
            "provenance_digest": provenance.get("provenance_digest")}


# ----------------------------------------------------------- controller (3.12) ----

def fresh_label(out_root: Path, label: str) -> str:
    """Never overwrite an earlier pass: a colliding label gets a `_r2`/`_r3` suffix.

    Prior evidence in this directory is what the verdict rests on, so a second run of the same size
    must land beside the first, not on top of it.
    """
    if not (out_root / label).exists():
        return label
    index = 2
    while (out_root / f"{label}_r{index}").exists():
        index += 1
    return f"{label}_r{index}"


def controller(args: argparse.Namespace) -> int:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    # `--extra-pass SIZE[:CATEGORY]` exists so the two controls behind the verdict below are
    # reproducible from the committed script: batch 48 (the size CHECKPOINT_SELECTION_REPORT.md:1698
    # recorded for run1) and a batch-32 repeat (to separate batch effects from run-to-run drift).
    # Without the flag the run is exactly the deliverable: batch 32 on all three categories, then
    # batch 1 on one.
    passes = [] if args.summarise_only else [
        (FIXED_BATCH, CATEGORIES, f"batch{FIXED_BATCH}"),
        (PROBE_BATCH, (PROBE_CATEGORY,), f"batch{PROBE_BATCH}")]
    # `--summarise-only` suppresses the two default passes (already on disk) but still RUNS any
    # `--extra-pass`, which is how the controls below were produced without redoing the baseline.
    for spec in args.extra_pass or ():
        size, _, category = spec.partition(":")
        passes.append((int(size), (category,) if category else CATEGORIES, f"batch{size}"))

    for batch_size, categories, label in passes:
        label = fresh_label(out_root, label)
        command = [str(MODEL_PYTHON), str(Path(__file__).resolve()), "--execute",
                   "--output-dir", str(out_root), "--batch-size", str(batch_size),
                   "--pass-label", label, "--controller", sys.executable]
        for category in categories:
            command += ["--category", category]
        print(f"[controller] {sys.executable} -> subprocess: {' '.join(command)}", flush=True)
        started = time.monotonic()
        completed = subprocess.run(command, cwd=str(REPO))
        print(f"[controller] pass {label} rc={completed.returncode} "
              f"wall={time.monotonic() - started:.1f}s", flush=True)
        if completed.returncode != 0:
            return completed.returncode

    summary = {
        "declared_before_run": {
            "fixed_batch_size": FIXED_BATCH,
            "fixed_batch_rationale": "the value the 30.0208 artifact recorded; declared before the "
                                     "run and not chosen to match a target",
            "probe_batch_size": PROBE_BATCH,
            "probe_category": PROBE_CATEGORY,
            "extra_controls": list(args.extra_pass or ()),
        },
        "mask_policy": f"{MASK_CONSTANT} (test.py:135); reference excluded by score, not by removal",
        "gallery_construction": GALLERY_CONSTRUCTION,
        "reference_policy": REFERENCE_POLICY,
        "tie_policy": TIE_POLICY,
        "text_path": TEXT_PATH,
        "transform": TRANSFORM,
        "batch_size_argument_plumbing": BATCH_PLUMBING,
        "controller_interpreter": {"path": sys.executable, "version": platform.python_version()},
        "worker_interpreter": {"path": str(MODEL_PYTHON)},
        "passes": {},
    }

    # Every pass directory on disk, so a re-run of one control never hides another.
    for pass_dir in sorted(p for p in out_root.glob("batch*") if p.is_dir()):
        files = sorted(f for f in pass_dir.glob("*.json") if f.name != "summary.json")
        if not files:
            continue
        per_category = {}
        batch_sizes = set()
        for path in files:
            data = json.loads(path.read_text())
            batch_sizes.add(data["batch_size"])
            per_category[data["category"]] = {
                "query_count": data["query_count"], "gallery_count": data["gallery_count"],
                "native_recalls": data["native_recalls"],
                "recomputed_recalls": data["recomputed_recalls"],
                "recomputed_matches_native": data["recomputed_matches_native"],
                "queries_with_tie_at_target": data["queries_with_tie_at_target"],
                "wall_seconds": data["wall_seconds"],
                "per_query_file": str(path.with_suffix(".jsonl")),
            }
        macro = {k: sum(v["native_recalls"][k] for v in per_category.values()) / len(per_category)
                 for k in ("r1", "r10", "r50")}
        macro_recomputed = {k: sum(v["recomputed_recalls"][k] for v in per_category.values())
                            / len(per_category) for k in ("r1", "r10", "r50")}
        summary["passes"][pass_dir.name] = {
            "batch_size": sorted(batch_sizes)[0],
            "categories": per_category,
            "macro_native": macro,
            "macro_recomputed": macro_recomputed,
            "macro_matches_recomputed": all(abs(macro[k] - macro_recomputed[k]) < 1e-9
                                            for k in macro),
            "category_count": len(per_category),
            "total_query_count": sum(v["query_count"] for v in per_category.values()),
        }

    # Batch sensitivity: every non-baseline pass against batch32, per shared category.
    base_name = f"batch{FIXED_BATCH}"
    base = summary["passes"].get(base_name)
    if base:
        sensitivity = {}
        for name, pass_row in summary["passes"].items():
            if name == base_name:
                continue
            per_category = {}
            for category, row in pass_row["categories"].items():
                if category not in base["categories"]:
                    continue
                base_row = base["categories"][category]
                base_rows = [json.loads(line) for line in
                             Path(base_row["per_query_file"]).read_text().splitlines() if line]
                rows = [json.loads(line) for line in
                        Path(row["per_query_file"]).read_text().splitlines() if line]
                assert [r["qid"] for r in base_rows] == [r["qid"] for r in rows], \
                    "query set differs between passes"
                moved = [{"qid": a["qid"], "rank_batch32": a["target_rank"],
                          "rank_above": a["target_rank"],
                          "rank_this_pass": b["target_rank"]}
                         for a, b in zip(base_rows, rows)
                         if a["target_rank"] != b["target_rank"]]
                score_deltas = [abs(a["score_at_target_rank"] - b["score_at_target_rank"])
                                for a, b in zip(base_rows, rows)]
                per_category[category] = {
                    "recall_delta": {k: row["native_recalls"][k] - base_row["native_recalls"][k]
                                     for k in ("r1", "r10", "r50")},
                    "max_abs_delta_r1_r10_r50": max(
                        abs(row["native_recalls"][k] - base_row["native_recalls"][k])
                        for k in ("r1", "r10", "r50")),
                    "max_abs_score_delta_at_target": max(score_deltas, default=0.0),
                    "moved_ranks": len(moved),
                    "moved_rank_fraction": round(len(moved) / len(rows), 6),
                    "moved_rank_examples": moved[:20],
                    "compared_queries": len(rows),
                }
            sensitivity[name] = {"batch_size": pass_row["batch_size"],
                                 "categories": per_category,
                                 "scope_note": "score comparison is at the target rank only; full "
                                               "similarity rows are not persisted"}
        summary["batch_sensitivity_vs_batch32"] = sensitivity

    # Historical comparison, values taken from the two artifacts read-only.
    history = {}
    for label, path, rows_key in (
            ("run1_29.9707", REPO / "workbench/artifacts/checkpoint_selection/"
                                   "20261106T002617+0700/measurements/clvc_run1.json", "categories"),
            ("run3_30.0208", REPO / "workbench/docs/reviews/20261111-master-audit/raw/"
                                   "clvc_run3cat_gpu.json", "results")):
        if not path.is_file():
            continue
        old = json.loads(path.read_text())
        old_rows = {r["category"]: r for r in old[rows_key]}
        entry = {"artifact": str(path.relative_to(REPO)),
                 "macro_r10": old["aggregate"]["macro_r10"],
                 "macro_r50": old["aggregate"]["macro_r50"],
                 "per_category": {}}
        if base:
            for category, row in old_rows.items():
                if category not in base["categories"]:
                    continue
                new_row = base["categories"][category]
                queries = new_row["query_count"]
                entry["per_category"][category] = {
                    "history_r1": row["r1"], "history_r10": row["r10"],
                    "history_r50": row["r50"],
                    "delta_r10": new_row["native_recalls"]["r10"] - row["r10"],
                    "delta_r50": new_row["native_recalls"]["r50"] - row["r50"],
                    "history_r1_hits": round(row["r1"] * queries / 100, 3),
                    "history_r10_hits": round(row["r10"] * queries / 100, 3),
                    "history_r50_hits": round(row["r50"] * queries / 100, 3),
                    "new_r1_hits": round(new_row["native_recalls"]["r1"] * queries / 100, 3),
                    "new_r10_hits": round(new_row["native_recalls"]["r10"] * queries / 100, 3),
                    "new_r50_hits": round(new_row["native_recalls"]["r50"] * queries / 100, 3),
                }
            if len(entry["per_category"]) == len(CATEGORIES):
                entry["macro_r10_matches"] = abs(entry["macro_r10"] - base["macro_native"]["r10"]) < 1e-12
                entry["macro_r50_matches"] = abs(entry["macro_r50"] - base["macro_native"]["r50"]) < 1e-12
        history[label] = entry
    summary["historical_comparison"] = history

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[controller] wrote", out_root / "summary.json", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW)
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--pass-label", default=None,
                        help="internal: subdirectory name for this pass")
    parser.add_argument("--controller", default="",
                        help="internal: path of the controlling interpreter, recorded for provenance")
    parser.add_argument("--extra-pass", action="append", dest="extra_pass", metavar="SIZE[:CATEGORY]",
                        help="additional batch pass (controls for the verdict); default: none")
    parser.add_argument("--execute", action="store_true",
                        help="internal: run the native evaluation in this process (model env only)")
    parser.add_argument("--summarise-only", action="store_true",
                        help="merge the pass directories already on disk without running anything")
    args = parser.parse_args(argv)
    args.categories = tuple(args.categories) if args.categories else CATEGORIES
    if not args.pass_label:
        args.pass_label = f"batch{args.batch_size}"
    if args.execute:
        return execute(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
