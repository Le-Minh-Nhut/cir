"""Source-faithful replay for the CLVC-Net FashionIQ evaluator.

Why a wrapper exists: the pinned `test.py` defines `test(params, local_model, global_model, testset, dataname)`
and nothing else — no CLI, no `__main__`, and no checkpoint loader. This module supplies the
missing orchestration shell and calls pinned `test.test(params, local_model, global_model, testset, dataname)`
without altering ranking, metric arithmetic, gallery construction, or feature extraction.

Exact signature and attributes read by pinned `test.py:6-149`:
- Signature: `def test(params, local_model, global_model, testset, dataname)` (5 arguments)
- `params.batch_size`: int (lines 96, 112; batch accumulation threshold)
- `local_model.eval()`: method (line 81)
- `global_model.eval()`: method (line 82)
- `local_model.compose_img_text(imgs, mods)[1]`: tensor (line 100)
- `global_model.compose_img_text(imgs, mods)[0]`: tensor (line 100)
- `local_model.extract_img_feature(imgs)[1]`: tensor (line 116)
- `global_model.extract_img_feature(imgs)[1]`: tensor (line 116)
- `testset.get_test_queries()`: method (line 84)
- `testset.get_test_targets()`: method (line 85)
- `dataname`: str (line 148; category label 'dress', 'shirt', 'toptee')
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import (
    ReplayResult,
    cache_requirement,
    run_cache_root,
    load_trusted_whole_model,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

CHECKPOINT_VARIANT = model_by_id("clvc_net")["checkpoint_variants"][0]

MODEL = model_by_id("clvc_net")
PROTOCOL = "fashioniq_val_split"

# The pinned trainer saves whole pickled modules per category: local_model.pt and global_model.pt
# (train.py:274-275). 6 files total across 3 categories.
EXPECTED_CATEGORY_FILES = {
    cat: (f"{cat}_local_model.pt", f"{cat}_global_model.pt")
    for cat in CATEGORIES
}


def validate_checkpoint(checkpoint_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    """Bind the six staged files to the registry's own declaration.

    The registry already records `checkpoint_mapping_status: VERIFIED_METADATA` plus one
    `expected_sha256` per member, and all six are reproducible from the staged bytes
    (`bench` evidence in the audit). Emitting a hard-coded "mapping unresolved" string regardless of
    those bytes made a verified bundle read as unverified, so the CLI could never reach the evaluator
    even when every declared digest matched. The check is now the digests themselves.
    """
    blockers: list[str] = []
    declared = {member["filename"]: member["expected_sha256"] for member in CHECKPOINT_VARIANT["bundle_members"]}
    for cat in categories:
        local_name, global_name = EXPECTED_CATEGORY_FILES[cat]
        for name in (local_name, global_name):
            path = checkpoint_root / name
            if not path.is_file() or path.stat().st_size == 0:
                blockers.append(f"CLVC-Net checkpoint missing or empty for {cat}: {path}")
                continue
            expected = declared.get(name)
            if not expected:
                blockers.append(f"CLVC-Net checkpoint has no declared digest: {name}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                blockers.append(
                    f"CLVC-Net checkpoint digest mismatch for {name}: expected {expected[:16]}..., found {actual[:16]}...")
    if not blockers and CHECKPOINT_VARIANT.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(
            f"CLVC-Net checkpoint mapping is unresolved: registry status="
            f"{CHECKPOINT_VARIANT.get('checkpoint_mapping_status')}")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"CLVC-Net source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"CLVC-Net dataset input missing: {dataset_root / relative}")
        cat_resized = dataset_root / "resized_image" / category
        if not cat_resized.is_dir():
            blockers.append(f"CLVC-Net dataset input missing: {cat_resized}")
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_checkpoint(checkpoint_root, categories)
        + validate_dataset(dataset_root, categories)
    )


def select_capture_rows(rows: list[dict[str, Any]], sample_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    """Select explicit capture IDs, preserving requested order and refusing ambiguous rows."""
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("CLVC capture selector needs non-empty unique sample IDs")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ValueError("CLVC capture rows need unique string sample_id values")
        by_id[sample_id] = row
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"CLVC capture IDs are absent from native rows: {missing[:10]}")
    return [by_id[sample_id] for sample_id in sample_ids]

def _install_clvc_observer(local_model: Any, global_model: Any, testset: Any,
                           capture_root: Path, selected_queries: set[str],
                           selected_gallery: set[str], invocation_id: str) -> tuple[dict[str, Any], Any, Any]:
    """Observe pinned evaluator tensors and features without changing model outputs."""
    import hashlib
    import numpy as np
    import torch.nn.functional as functional
    raw_queries = list(testset.ref_captions)
    query_ids = ["clvc:" + hashlib.sha256(json.dumps(
        {"category": testset.name, **row}, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest() for row in raw_queries]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("CLVC native query source rows are ambiguous")
    targets = list(testset.get_test_targets())
    gallery_ids = [testset.images[row["target_img_id"]] for row in targets]
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("CLVC native gallery IDs are ambiguous")
    if not selected_queries <= set(query_ids) or not selected_gallery <= set(gallery_ids):
        raise ValueError("CLVC selected native IDs are absent from the pinned dataset")
    state: dict[str, Any] = {"query_cursor": 0, "gallery_cursor": 0,
        "arrays": {key: [] for key in ("query_ids", "gallery_ids", "references", "targets",
            "query_local", "query_global", "gallery_local", "gallery_global",
            "selected_query_ids", "selected_query_pixels", "selected_gallery_ids",
            "selected_gallery_pixels", "selected_local_lengths", "selected_global_lengths",
            "selected_local_tokens", "selected_global_tokens")}}
    original_argsort = np.argsort
    state["native_order"] = []
    state["native_scores"] = []

    def observed_argsort(values: Any, *args: Any, **kwargs: Any) -> Any:
        order = original_argsort(values, *args, **kwargs)
        if np.ndim(values) == 1 and np.shape(values) == (len(gallery_ids),):
            state["native_scores"].append(np.asarray(values).copy())
            state["native_order"].append(np.asarray(order).copy())
        return order

    np.argsort = observed_argsort
    originals: list[tuple[Any, Any, Any, Any]] = []
    handles: list[Any] = []

    def install(model: Any, tower: str) -> None:
        original_compose = model.compose_img_text
        original_extract = model.extract_img_feature
        original_encoded = model.text_model.forward_encoded_texts
        originals.append((model, original_compose, original_extract, original_encoded))
        token_data: dict[str, Any] = {"lengths": None, "padded": None}

        def encoded(texts: Any) -> Any:
            token_data["lengths"] = np.asarray([len(text) for text in texts], dtype=np.int64)
            token_data["padded"] = None
            return original_encoded(texts)

        def embedding_input(module: Any, inputs: Any) -> None:
            token_data["padded"] = inputs[0].detach().cpu().numpy()

        handles.append(model.text_model.embedding_layer.register_forward_pre_hook(embedding_input))

        def compose(images: Any, texts: Any) -> Any:
            output = original_compose(images, texts)
            first = state["query_cursor"]
            ids = query_ids[first:first + len(texts)]
            if len(ids) != len(texts) or token_data["padded"] is None or token_data["lengths"] is None:
                raise RuntimeError("CLVC native query/token capture order mismatch")
            rows = output[1] if tower == "local" else output[0]
            state["arrays"][f"query_{tower}"].append(
                functional.normalize(rows, dim=-1).detach().cpu().numpy())
            if tower == "local":
                state["arrays"]["query_ids"].extend(ids)
                state["arrays"]["references"].extend(row["candidate"] for row in raw_queries[first:first + len(ids)])
                state["arrays"]["targets"].extend(row["target"] for row in raw_queries[first:first + len(ids)])
                keep = [i for i, sample_id in enumerate(ids) if sample_id in selected_queries]
                if keep:
                    state["arrays"]["selected_query_ids"].extend(ids[i] for i in keep)
                    state["arrays"]["selected_query_pixels"].append(images[keep].detach().cpu().numpy())
            keep = [i for i, sample_id in enumerate(ids) if sample_id in selected_queries]
            if keep:
                state["arrays"][f"selected_{tower}_tokens"].append(token_data["padded"][:, keep])
                state["arrays"][f"selected_{tower}_lengths"].append(token_data["lengths"][keep])
            if tower == "global":
                state["query_cursor"] += len(texts)
            return output

        def extract(images: Any) -> Any:
            output = original_extract(images)
            rows = functional.normalize(output[1], dim=-1).detach().cpu().numpy()
            state["arrays"][f"gallery_{tower}"].append(rows)
            if tower == "local":
                first = state["gallery_cursor"]
                ids = gallery_ids[first:first + len(images)]
                if len(ids) != len(images):
                    raise RuntimeError("CLVC native gallery capture order mismatch")
                state["arrays"]["gallery_ids"].extend(ids)
                keep = [i for i, image_id in enumerate(ids) if image_id in selected_gallery]
                if keep:
                    state["arrays"]["selected_gallery_ids"].extend(ids[i] for i in keep)
                    state["arrays"]["selected_gallery_pixels"].append(images[keep].detach().cpu().numpy())
                state["gallery_cursor"] += len(images)
            return output

        model.compose_img_text = compose
        model.extract_img_feature = extract
        model.text_model.forward_encoded_texts = encoded

    install(local_model, "local")
    install(global_model, "global")

    def restore() -> None:
        for handle in handles:
            handle.remove()
        np.argsort = original_argsort
        for model, compose, extract, encoded in originals:
            model.compose_img_text = compose
            model.extract_img_feature = extract
            model.text_model.forward_encoded_texts = encoded


    def finalize(metrics: dict[str, Any]) -> Path:
        arrays = state["arrays"]
        if state["query_cursor"] != len(query_ids) or state["gallery_cursor"] != len(gallery_ids):
            raise RuntimeError("CLVC native observer did not cover exact query/gallery order")
        query_local = np.concatenate(arrays["query_local"])
        query_global = np.concatenate(arrays["query_global"])
        gallery_local = np.concatenate(arrays["gallery_local"])
        gallery_global = np.concatenate(arrays["gallery_global"])
        queries = np.concatenate((query_local, query_global), axis=1)
        gallery = np.concatenate((gallery_local, gallery_global), axis=1)
        scores = np.stack(state["native_scores"])
        if scores.shape != (len(query_ids), len(gallery_ids)):
            raise RuntimeError("CLVC native evaluator score capture shape mismatch")
        gallery_index = {sample_id: index for index, sample_id in enumerate(arrays["gallery_ids"])}
        references = np.asarray(arrays["references"], dtype=np.str_)
        targets = np.asarray(arrays["targets"], dtype=np.str_)
        order = np.stack(state["native_order"])
        if (order.shape != scores.shape or len(state["native_order"]) != len(query_ids) or
                len(state["native_scores"]) != len(query_ids)):
            raise RuntimeError("CLVC native evaluator score/sorter capture coverage mismatch")
        target_indices = np.asarray([gallery_index[str(target)] for target in targets])
        ranks = np.argmax(order == target_indices[:, None], axis=1) + 1
        if not arrays["selected_query_pixels"] or not arrays["selected_gallery_pixels"]:
            raise RuntimeError("CLVC G1 selection must contain query and gallery rows")
        expected_metrics = {f"{testset.name}_r{k}": float(100 * np.mean(ranks <= k))
                            for k in (1, 10, 50)}
        for metric_name, expected in expected_metrics.items():
            observed = metrics.get(metric_name)
            if observed is None or not np.isclose(float(observed), expected, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"CLVC native capture metric mismatch for {metric_name}: "
                                   f"native={observed!r}, recomputed={expected!r}")
        def join_tokens(chunks: list[np.ndarray]) -> np.ndarray:
            width = max(chunk.shape[0] for chunk in chunks)
            return np.concatenate([
                np.pad(chunk, ((0, width - chunk.shape[0]), (0, 0))) for chunk in chunks
            ], axis=1)
        payload = {
            "query_ids": np.asarray(arrays["query_ids"], dtype=np.str_),
            "gallery_ids": np.asarray(arrays["gallery_ids"], dtype=np.str_),
            "query_features": queries, "gallery_features": gallery,
            "query_local_features": query_local, "query_global_features": query_global,
            "gallery_local_features": gallery_local, "gallery_global_features": gallery_global,
            "scores": scores.astype(np.float32, copy=False), "native_order": order,
            "rank_qids": np.asarray(arrays["query_ids"], dtype=np.str_), "ranks": ranks,
            "reference_ids": references, "target_ids": targets,
            "selected_query_ids": np.asarray(arrays["selected_query_ids"], dtype=np.str_),
            "selected_gallery_ids": np.asarray(arrays["selected_gallery_ids"], dtype=np.str_),
            "selected_query_pixels": np.concatenate(arrays["selected_query_pixels"]),
            "selected_gallery_pixels": np.concatenate(arrays["selected_gallery_pixels"]),
            "selected_local_tokens": join_tokens(arrays["selected_local_tokens"]),
            "selected_global_tokens": join_tokens(arrays["selected_global_tokens"]),
            "selected_local_lengths": np.concatenate(arrays["selected_local_lengths"]),
            "selected_global_lengths": np.concatenate(arrays["selected_global_lengths"]),
            "metadata": np.asarray(json.dumps({"schema": "cir-clvc-native-capture-v1",
                "category": testset.name, "query_token_padding": "zero padded; no attention mask",
                "invocation_id": invocation_id,
                "metrics": metrics}, sort_keys=True)),
        }
        from workbench.replay.clvc_tme_parity import _write_streaming_npz
        path = capture_root / f"{testset.name}.npz"
        _write_streaming_npz(path, payload)
        return path

    return state, restore, finalize

def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any]:
    import torch
    import torchvision.transforms as transforms

    source_str = str(source_root)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    datasets = __import__("datasets")
    test_module = __import__("test")
    return torch, datasets, test_module

def _cpu_device_patch(torch_module: Any) -> None:
    """ADAPTED RUNTIME: make the pinned `.cuda()` calls device-neutral for a CPU smoke.

    The evaluator hard-codes `.cuda()` at `test.py:28,47,99,115` and `text_model.py:96,108`.
    Nothing about weights, module structure, arithmetic or ranking changes -- only where tensors
    live. This is a device parameterisation, NOT the author's code path, and every artifact that
    uses it must say so.
    """
    if getattr(torch_module.Tensor.cuda, "_workbench_device_neutral", False):
        return
    original = torch_module.Tensor.cuda

    def cuda(self, *args, **kwargs):  # noqa: ANN001 - mirrors Tensor.cuda's signature loosely
        return self

    setattr(cuda, "_workbench_device_neutral", True)
    setattr(cuda, "_workbench_original", original)
    torch_module.Tensor.cuda = cuda


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    batch_size: int = 32,
    device: str | None = None,
    capture_root: Path | None = None,
    selected_queries: set[str] | None = None,
    selected_gallery: set[str] | None = None,
    sample_queries: set[str] | None = None,
    invocation_id: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, test_module = _load_replay_modules(source_root)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapted = target_device == "cpu"
    if adapted:
        _cpu_device_patch(torch)

    import torchvision.transforms as transforms
    # The NATIVE FashionIQ validation pipeline, copied from `train.py:90-96` (`load_dataset`, the
    # `args.dataset == 'fashioniq'` branch, `split='val'`):
    #
    #     Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize
    #
    # This wrapper used `Resize(224)`, which is the *fashion200k* branch (`train.py:113-117`) rather
    # than the FashionIQ one. Both produce a (3, 224, 224) tensor, so nothing failed visibly: the
    # short side was scaled to 224 and then center-cropped to 224, i.e. a tighter crop of a smaller
    # resample than the evaluator these weights were validated under. On a 300x500 fixture the two
    # pipelines differ by up to 4.24 in normalized units while matching in shape, which is exactly
    # the kind of mismatch that shows up as an unexplained Recall gap rather than an error.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results: list[ReplayResult] = []
    # A shared root is reused ONLY when its manifest matches THIS run's description; see
    # `replay/tgcir.py`. CLVC-Net's constructor is per-category and does NOT write
    # `test_queries_<cat>.pkl`, so `warm_replay_cache.py` does not warm it (its `cache_files` is
    # empty by design). Pointing WORKBENCH_REPLAY_CACHE_ROOT at such a root would therefore be
    # refused; the normal path here is the fresh isolated one and reports `rebuilt`.
    cache_metadata: dict[str, Any] = {}
    with run_cache_root(
            dataset_root,
            requirement=cache_requirement(
                dataset_root, model_id="clvc_net", transform=transform,
                checkpoint={"checkpoint_id": CHECKPOINT_VARIANT["checkpoint_id"],
                            "expected_sha256": None},
                state={"seed": None, "eval_mode": True, "noise_ratio": 0}),
            record=cache_metadata) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        params = SimpleNamespace(batch_size=batch_size)

        for category in categories:
            local_name, global_name = EXPECTED_CATEGORY_FILES[category]
            local_path = checkpoint_root / local_name
            global_path = checkpoint_root / global_name
            declared = {member["filename"]: member["expected_sha256"]
                        for member in CHECKPOINT_VARIANT["bundle_members"]}

            # Whole pickled modules: unpickling is authorized only against the declared digest.
            local_model = load_trusted_whole_model(local_path, declared[local_name], torch)
            global_model = load_trusted_whole_model(global_path, declared[global_name], torch)
            local_model = local_model.to(target_device)
            global_model = global_model.to(target_device)
            local_model.eval()
            global_model.eval()

            testset = datasets.FashionIQ(
                path=root_str,
                gallery_all=False,
                name=category,
                split="val",
                transform=transform,
            )
            original_ref_captions = testset.ref_captions
            original_get_test_targets = testset.get_test_targets
            if sample_queries is not None:
                import hashlib
                import json

                full_targets = original_get_test_targets()
                selected_rows = []
                for row in original_ref_captions:
                    query_id = "clvc:" + hashlib.sha256(json.dumps(
                        {"category": category, **row}, sort_keys=True, separators=(",", ":")
                    ).encode()).hexdigest()
                    if query_id in sample_queries:
                        selected_rows.append(row)
                if not selected_rows:
                    raise ValueError(f"CLVC sample has no queries for category {category}")
                testset.ref_captions = selected_rows
                testset.get_test_targets = lambda: full_targets

            observer = None
            try:
                if capture_root is not None:
                    if selected_queries is None or selected_gallery is None:
                        raise ValueError("CLVC native capture requires explicit query/gallery IDs")
                    _, restore_observer, finalize_observer = _install_clvc_observer(
                        local_model, global_model, testset, capture_root,
                        selected_queries, selected_gallery, invocation_id or "")
                    observer = (restore_observer, finalize_observer)
                recalls = dict(test_module.test(params, local_model, global_model, testset, category))
            finally:
                if observer is not None:
                    observer[0]()
            if observer is not None:
                observer[1](recalls)
            results.append(
                ReplayResult(
                    model_id="clvc_net",
                    checkpoint_id=f"author_drive_{category}",
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "local_checkpoint": str(local_path),
                        "global_checkpoint": str(global_path),
                        "source_commit": MODEL["upstream_commit_sha"],
                        "native_evaluator": "test.py:test",
                        "runtime_label": "ADAPTED_DEVICE_NEUTRAL" if adapted else "NATIVE_CUDA",
                        "device": target_device,
                        "adaptation": (
                            "torch.Tensor.cuda() neutralised so the pinned `.cuda()` sites "
                            "(test.py:28,47,99,115; text_model.py:96,108) run on CPU; weights, "
                            "architecture, arithmetic and ranking untouched"
                        ) if adapted else None,
                        "cache_manifest": dict(cache_metadata),
                    },
                )
            )

    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--capture-selection", type=Path, default=None)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--invocation-id", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sample-queries", type=Path, default=None,
                        help="JSON array of native CLVC query IDs to evaluate over the full pair-union gallery")
    args = parser.parse_args(argv)
    if (args.capture_root is None) != (args.capture_selection is None):
        parser.error("--capture-root and --capture-selection must be provided together")
    if args.capture_root is not None and not args.invocation_id:
        parser.error("capture requires pre-run --invocation-id")
    capture_queries: set[str] | None = None
    capture_gallery: set[str] | None = None
    if args.capture_selection is not None:
        try:
            selection = json.loads(args.capture_selection.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --capture-selection: {exc}")
        if not isinstance(selection, dict) or set(selection) != {"queries", "gallery"}:
            parser.error("capture selection must contain exactly queries and gallery")
        if not all(isinstance(value, list) and all(isinstance(item, str) for item in value)
                   for value in selection.values()):
            parser.error("capture selection values must be string lists")
        capture_queries = set(selection["queries"])
        capture_gallery = set(selection["gallery"])
        if not capture_queries or not capture_gallery or len(capture_queries) > 32 or len(capture_gallery) > 32:
            parser.error("capture selection needs non-empty query/gallery sets of at most 32 IDs")
    sample_queries = None
    if args.sample_queries is not None:
        try:
            sample_queries = json.loads(args.sample_queries.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --sample-queries: {exc}")
        if (not isinstance(sample_queries, list) or not sample_queries or
                any(not isinstance(value, str) or not value for value in sample_queries) or
                len(sample_queries) != len(set(sample_queries))):
            parser.error("--sample-queries must be a non-empty unique string array")
        sample_queries = set(sample_queries)

    selected_categories = tuple(args.categories) if args.categories else CATEGORIES
    blockers = preflight(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        selected_categories,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=selected_categories,
        batch_size=args.batch_size,
        device=args.device,
        capture_root=args.capture_root,
        selected_queries=capture_queries,
        selected_gallery=capture_gallery,
        sample_queries=sample_queries,
        invocation_id=args.invocation_id,
    )
    payload = {
        "model_id": "clvc_net",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "invocation_id": args.invocation_id,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
