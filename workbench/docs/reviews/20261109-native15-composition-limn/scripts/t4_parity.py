"""Task 4: G1 input parity + G2 forward/scoring parity for LIMN.

Two phases, run under the interpreter each side actually uses in production:

  phase `native`  (LIMN interpreter)  — pinned `datasets.py` / `model.py` / `test.py`; records
                  identities, tokens, decoded tensors and native features for a fixed real sample,
                  and the population identities the adapter phase will re-encode.
  phase `adapter` (CONTROL interpreter) — `workbench.composition.model_limn.Adapter`, i.e. the exact
                  gate + subprocess-worker path the CIRCUS runner uses. Compares against the native
                  artifacts and scores/ranks the whole category population.

Splitting on the interpreter is not cosmetic: the adapter refuses to run when its own interpreter is
the controller (`UNSAFE_INTERPRETER`), so "same process" would mean a path production never takes.

Tolerances: declared in INPUT_FORWARD_PARITY.vi.md before this ran.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")

import numpy as np

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.operator_config import resolve_config

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn")
RAW = OUT / "raw"

TOL = {
    "forward_max_abs": 1e-5,
    "forward_max_rel": 1e-4,
    "batch_max_abs": 1e-4,
    "batch_max_rel": 1e-3,
    "score_max_abs": 1e-6,
    "tie_epsilon": 1e-6,
}


def correct_text(text: str, correction: dict[str, str]) -> str:
    """Verbatim copy of `datasets.FashionIQ.correct_text`; used only to re-derive the model text."""
    import string

    translation = str.maketrans({key: " " for key in string.punctuation})
    tokens = str(text).lower().translate(translation).strip().split()
    return " ".join(correction.get(word, word) for word in tokens)

# --------------------------------------------------------------------------- v2 evidence writer

#: The declared standard this producer emits from now on. Written FIRST in
#: `workbench/docs/reviews/20261112-parity-repair/V2_STANDARD.md`, then here, then read by
#: `parity_gate._component_checks`. Both phases declare it: the gate refuses a pair whose two sides
#: disagree, and the block itself is written by the phase that holds BOTH arrays.
STANDARD_V2 = "parity-absolute-relative/v2"
COMPONENT_FULL_LIMIT = 4096
COMPONENT_HEAD = 512

def _component_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()

def component_scope(name: str, dimension: str, actual: np.ndarray, reference: np.ndarray,
                    *, atol: float, rtol: float) -> dict:
    """The v2 per-component record for ONE aligned comparison, from arrays already in memory.

    `abs(adapter - native) <= atol + rtol*abs(native)`, native as the reference, applied element by
    element here -- `max_abs`/`max_rel` are never used to infer it. Chunk rule and every field:
    `V2_STANDARD.md`. Above `COMPONENT_FULL_LIMIT` components the record is the first `COMPONENT_HEAD`
    plus every violating component, with a sha256 over the full arrays so the chunk is provable.
    """
    a = np.ascontiguousarray(actual, dtype=np.float64).ravel()
    b = np.ascontiguousarray(reference, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise SystemExit(f"{name}: {a.shape} vs {b.shape} are not aligned; refusing to record a "
                         "component block for arrays that do not correspond")
    delta = np.abs(a - b)
    denominator = atol + rtol * np.abs(b)
    total = int(b.size)
    normalized = np.zeros(total, dtype=np.float64)
    np.divide(delta, denominator, out=normalized, where=denominator > 0.0)
    # `~(delta <= denominator)` counts a NaN as a violation, so a non-finite component is recorded as
    # an offender instead of silently reading clean; the gate refuses the block as malformed.
    offenders = [int(index) for index in np.flatnonzero(~(delta <= denominator))]
    if total <= COMPONENT_FULL_LIMIT:
        emitted, coverage = list(range(total)), "full"
    else:
        emitted, coverage = sorted(set(range(COMPONENT_HEAD)) | set(offenders)), "chunked"
    return {
        "dimension": dimension,
        "shape": list(actual.shape),
        "components_total": total,
        "components_emitted": len(emitted),
        "coverage": coverage,
        "indices": emitted,
        "reference": [float(b[index]) for index in emitted],
        "actual": [float(a[index]) for index in emitted],
        "sha256": {"reference": _component_digest(b), "actual": _component_digest(a),
                   "delta": _component_digest(delta)},
        "mismatch_count": len(offenders),
        "max_abs": float(delta.max()) if total else 0.0,
        "worst_normalized": float(normalized.max()) if total else 0.0,
        "offenders": [{"index": index, "reference": float(b[index]), "actual": float(a[index]),
                       "normalized": float(normalized[index])} for index in offenders],
    }

def component_block(scopes: dict[str, dict]) -> dict:
    """The `component_parity` document, as recorded beside `relation_state` at the top level."""
    return {"standard": STANDARD_V2,
            "declared_rule": "abs(adapter - native) <= atol + rtol * abs(native)",
            "reference_side": "native",
            "chunk_rule": (f"full when components_total <= {COMPONENT_FULL_LIMIT}, else the first "
                           f"{COMPONENT_HEAD} components plus every component whose normalized "
                           "violation exceeds 1"),
            "scopes": scopes}


# --------------------------------------------------------------------------- native phase

def native_phase(category: str, config) -> dict:
    from workbench.replay import limn as replay

    import torch

    checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "limn"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "LIMN"
    problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
    if problems:
        raise SystemExit(f"preflight blocked: {problems}")

    began = time.time()
    with replay.native_category(category, checkpoint_root, dataset_root, source_root) as loaded:
        model, dataset, native_test = loaded["model"], loaded["dataset"], loaded["native_test"]
        images = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
        captions = json.loads((dataset_root / "captions" / f"cap.{category}.val.json").read_text())
        correction = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json").read_text())

        gallery_positions = [entry["target_img_id"] for entry in dataset.test_targets]
        gallery_ids = [images[position] for position in gallery_positions]
        rebuilt: list[str] = []
        for row in captions:
            for key in ("candidate", "target"):
                if row[key] not in rebuilt:
                    rebuilt.append(row[key])

        native_texts = [entry["mod"]["str"] for entry in dataset.test_queries]
        composed = [f"{correct_text(row['captions'][0], correction)} and "
                    f"{correct_text(row['captions'][1], correction)}" for row in captions]
        native_refs = [images[entry["source_img_id"]] for entry in dataset.test_queries]
        native_targets = [images[entry["target_img_id"]] for entry in dataset.test_queries]

        # Fixed small REAL sample covering all three categories plus caption edge cases and the
        # final short batch: first row, first row whose FIRST caption is capitalized, first row
        # whose second caption is shorter than 12 characters, and the last row.
        capitalized = next((i for i, row in enumerate(captions)
                            if row["captions"][0][:1].isupper()), 0)
        short_half = next((i for i, row in enumerate(captions)
                           if len(row["captions"][1]) < 12), 0)
        chosen = sorted({0, capitalized, short_half, len(captions) - 1})

        identity = {
            "query_count_native": len(dataset.test_queries),
            "query_count_annotations": len(captions),
            "gallery_count_native": len(dataset.test_targets),
            "gallery_count_rebuilt_from_annotations": len(rebuilt),
            "gallery_order_identical": gallery_ids == rebuilt,
            "reference_ids_identical": native_refs == [row["candidate"] for row in captions],
            "target_ids_identical": native_targets == [row["target"] for row in captions],
            "model_text_identical_to_author_composition": native_texts == composed,
            "model_text_mismatches": [
                {"index": i, "native": native_texts[i], "recomposed": composed[i]}
                for i in range(len(composed)) if native_texts[i] != composed[i]][:5],
            "duplicate_caption_pairs_within_category":
                len(captions) - len({(row["candidate"], composed[i]) for i, row in enumerate(captions)}),
            "reference_equals_target_rows": sum(1 for row in captions if row["candidate"] == row["target"]),
            "tokenizer": {
                "callable": f"{getattr(model.backbone.tokenizer, '__module__', '?')}."
                            f"{getattr(model.backbone.tokenizer, '__qualname__', type(model.backbone.tokenizer).__name__)}",
                "vocab_size": int(model.backbone.clip.token_embedding.num_embeddings),
                "context_length": int(model.backbone.clip.positional_embedding.shape[0]),
                "source": "open_clip.get_tokenizer('ViT-L-14') inside pinned Backbone.__init__",
            },
            "transform": {
                "image_size": (list(model.backbone.clip.visual.image_size)
                               if isinstance(model.backbone.clip.visual.image_size, (tuple, list))
                               else int(model.backbone.clip.visual.image_size)),
                "mean": [float(v) for v in model.backbone.clip.visual.image_mean],
                "std": [float(v) for v in model.backbone.clip.visual.image_std],
                "source": "open_clip.image_transform(..., is_train=False) in replay.limn.load_native",
            },
            "model_in_eval": not model.training,
            "dataset_type_index": int(dataset.dataset_type),
            "sample_indices": chosen,
        }
        identity["sample_tokens"] = [
            {"index": i, "text": native_texts[i],
             "token_ids": model.backbone.tokenizer([native_texts[i]]).numpy().tolist()[0]}
            for i in chosen]

        from PIL import Image

        def image_path(name: str) -> str:
            # NOT `dataset.image_dir`: that is the isolated replay cache, removed when this context
            # exits, so the adapter phase could never open it. The cache only symlinks the real
            # iLearn files, and those are what the native loader read.
            return str(dataset_root / "resized_image" / category / f"{name}.jpg")

        sample_inputs = {
            "category": category,
            "queries": [[image_path(native_refs[i]), image_path(native_targets[i])]
                        for i in chosen],
            "query_texts": [native_texts[i] for i in chosen],
            "reference_ids": [native_refs[i] for i in chosen],
            "target_ids": [native_targets[i] for i in chosen],
            "sample_indices": chosen,
            "full_query_paths": [image_path(name) for name in native_refs],
            "full_query_texts": native_texts,
            "full_reference_ids": native_refs,
            "full_target_ids": native_targets,
            "gallery_ids": gallery_ids,
            "gallery_paths": [image_path(name) for name in gallery_ids],
            "dataset_type_index": int(dataset.dataset_type),
        }
        modes: dict[str, int] = {}
        for pair in sample_inputs["queries"]:
            for path in pair:
                with Image.open(path) as handle:
                    modes[handle.mode] = modes.get(handle.mode, 0) + 1
        identity["sample_pil_modes"] = modes
        identity["sample_convert_to_rgb_is_noop"] = set(modes) <= {"RGB"}

        source_tensor = np.stack([dataset.test_queries[i]["source_img_data"].numpy() for i in chosen])
        target_tensor = np.stack([dataset.test_queries[i]["target_img_data"].numpy() for i in chosen])
        identity["native_tensor"] = {"dtype": str(source_tensor.dtype),
                                     "shape": list(source_tensor.shape),
                                     "layout": "CHW float32 (ToTensor + Normalize)"}

        with torch.no_grad():
            batch = torch.from_numpy(source_tensor).float().cuda()
            kinds = torch.ones(batch.shape[0]).cuda() * dataset.dataset_type
            query_features = model.compose_query(batch, sample_inputs["query_texts"], kinds)[0]
            target_batch = torch.from_numpy(target_tensor).float().cuda()
            target_kinds = torch.ones(target_batch.shape[0]).cuda() * dataset.dataset_type
            target_features = model.compose_target(target_batch, target_kinds)[0]

        np.savez(RAW / f"g2_{category}_native.npz",
                 native_query=query_features.float().cpu().numpy(),
                 native_target=target_features.float().cpu().numpy(),
                 source_tensor=source_tensor, target_tensor=target_tensor)
        (RAW / f"g2_{category}_inputs.json").write_text(json.dumps(sample_inputs), encoding="utf-8")

        # Native ranks for the whole population, from the SAME loaded objects the aggregate used,
        # cross-checked against the native metric so a wrong rank set cannot be compared to.
        from types import SimpleNamespace

        metrics = dict(native_test.test(SimpleNamespace(local_rank=0,
                                                        batch_size=replay.resolved_batch_size()),
                                        model, dataset, category))
        identity["native_metrics_recheck"] = metrics
        del model, dataset
    torch.cuda.empty_cache()
    identity["phase_seconds"] = round(time.time() - began, 1)
    return identity


# --------------------------------------------------------------------------- adapter phase

def adapter_phase(category: str, config) -> dict:
    from workbench.composition.model_limn import Adapter

    inputs = json.loads((RAW / f"g2_{category}_inputs.json").read_text())
    native = np.load(RAW / f"g2_{category}_native.npz")
    began = time.time()
    adapter = Adapter(config)
    adapter.work_root = RAW / "parity-work" / category
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    try:
        adapter_queries = adapter.encode_queries(
            [pair[0] for pair in inputs["queries"]], inputs["query_texts"],
            [category] * len(inputs["queries"]))
        adapter_targets = adapter.encode_targets(
            [pair[1] for pair in inputs["queries"]], [category] * len(inputs["queries"]))

        def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
            delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
            cosines = [(row @ ref) / (np.linalg.norm(row) * np.linalg.norm(ref))
                       for row, ref in zip(actual.astype(np.float64), reference.astype(np.float64))]
            return {"max_abs": float(delta.max()), "max_rel": float((delta / scale).max()),
                    "min_cosine": float(min(cosines)), "bitwise_equal": bool(np.array_equal(actual, reference))}

        forward = {
            "query": compare(adapter_queries, native["native_query"]),
            "target": compare(adapter_targets, native["native_target"]),
            "adapter_dtype": str(adapter_queries.dtype),
            "adapter_shape": list(adapter_queries.shape),
        }
        for side in ("query", "target"):
            forward[side]["within_tolerance"] = (
                forward[side]["max_abs"] <= TOL["forward_max_abs"]
                and forward[side]["max_rel"] <= TOL["forward_max_rel"])
        result["sample_forward"] = forward
        # v2: the per-component record for the SAME aligned arrays the aggregates above summarise.
        # Both reference arrays are the native side's, so the rule's "native is the reference" holds.
        # Additive: a v1 reader ignores the key; `max_abs`/`max_rel` stay exactly as they were and are
        # never used to INFER the componentwise result.
        result["component_parity"] = component_block({
            "sample_forward.query": component_scope(
                "sample_forward.query", "forward_parity", adapter_queries, native["native_query"],
                atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"]),
            "sample_forward.target": component_scope(
                "sample_forward.target", "forward_parity", adapter_targets, native["native_target"],
                atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"]),
        })

        singles = np.concatenate([
            adapter.encode_queries([path], [text], [category])
            for path, text in zip([pair[0] for pair in inputs["queries"]], inputs["query_texts"])])
        result["batch_size_1_vs_n"] = compare(singles, adapter_queries)
        result["batch_size_1_vs_n"]["within_tolerance"] = (
            result["batch_size_1_vs_n"]["max_abs"] <= TOL["batch_max_abs"]
            and result["batch_size_1_vs_n"]["max_rel"] <= TOL["batch_max_rel"])

        # Cache on/off: the worker restarts and re-encodes the same sample; a cached model must
        # produce the same features as a freshly loaded one.
        adapter.restart()
        after_restart = adapter.encode_queries(
            [pair[0] for pair in inputs["queries"]], inputs["query_texts"],
            [category] * len(inputs["queries"]))
        result["cache_off_restart"] = compare(after_restart, adapter_queries)
        result["cache_off_restart"]["within_tolerance"] = (
            result["cache_off_restart"]["max_abs"] <= TOL["batch_max_abs"]
            and result["cache_off_restart"]["max_rel"] <= TOL["batch_max_rel"])

        # Full population through the production worker path.
        population_queries = adapter.encode_queries(
            inputs["full_query_paths"], inputs["full_query_texts"],
            [category] * len(inputs["full_query_texts"]))
        population_gallery = adapter.encode_targets(
            inputs["gallery_paths"], [category] * len(inputs["gallery_paths"]))
        np.savez(RAW / f"g2_{category}_population.npz",
                 queries=population_queries, gallery=population_gallery)
        result["population"] = {
            "query_count": int(population_queries.shape[0]),
            "gallery_count": int(population_gallery.shape[0]),
            "finite": bool(np.isfinite(population_queries).all()
                           and np.isfinite(population_gallery).all()),
        }
    finally:
        adapter.close()

    result["scoring_ranking"] = score_and_rank(category, inputs, config)
    result["phase_seconds"] = round(time.time() - began, 1)
    return result


def _native_population_ranks(category: str) -> list[int] | None:
    path = OUT / f"limn_native.{category}.jsonl"
    if not path.is_file():
        return None
    return [json.loads(line)["target_rank"] for line in path.read_text().splitlines() if line.strip()]


def score_and_rank(category: str, inputs: dict, config) -> dict:
    """G2 scoring/ranking: native convention vs the adapter's own score/rank, same features."""
    from workbench.composition.model_limn import Adapter

    blob = np.load(RAW / f"g2_{category}_population.npz")
    # float32 END TO END, because pinned `test.py` normalizes the float32 feature arrays in place and
    # then dots them: the native ranking is computed on float32 similarities. Comparing a float64
    # recompute against the adapter's float32 path would confound the tie convention with a dtype
    # difference, which is exactly the kind of rank drift this gate must not accept silently.
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    queries_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    gallery_n = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    unmasked = queries_n.dot(gallery_n.T)                     # what score() must reproduce
    sims = unmasked.copy()
    position = {name: index for index, name in enumerate(inputs["gallery_ids"])}
    reference_positions = [position[name] for name in inputs["full_reference_ids"]]
    target_positions = [position[name] for name in inputs["full_target_ids"]]
    for row, reference in enumerate(reference_positions):
        sims[row, reference] = -10e10
    # Pinned `test.py` convention.
    native_style = [int(np.flatnonzero(np.argsort(-sims[row]) == target)[0]) + 1
                    for row, target in enumerate(target_positions)]
    # The SAME argsort convention WITHOUT the reference mask. The mask is a property of the native
    # metric (the reference is not retrievable there), while CIRCUS is reference-eligible, so the
    # only thing the two rank FUNCTIONS can be compared on is the unmasked matrix. Anything else
    # would report the deliberate mask difference as a tie-convention difference.
    argsort_unmasked = [int(np.flatnonzero(np.argsort(-unmasked[row]) == target)[0]) + 1
                        for row, target in enumerate(target_positions)]

    # The adapter's own convention, same matrices, same dtype.
    adapter = Adapter(config)
    adapter.work_root = RAW / "parity-work" / f"{category}-rank"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    try:
        positions = [[position] for position in target_positions]
        # LIMN's `score` requires scope -> full-gallery matrices (a Mapping), not a bare matrix.
        adapter_style = adapter.rank(queries, {category: gallery}, positions,
                                     [category] * len(positions))
    finally:
        adapter.close()

    # The adapter's score() against the native dot product. Compared on the UNMASKED matrix: the
    # -10e10 reference mask belongs to test.py's metric, not to score(), and comparing against the
    # masked row would report a 1e11 "delta" that measures the mask.
    pair_count = min(64, len(target_positions))
    adapter2 = Adapter(config)
    adapter2.work_root = RAW / "parity-work" / f"{category}-score"
    adapter2.work_root.mkdir(parents=True, exist_ok=True)
    try:
        scores = adapter2.score(queries[:pair_count], {category: gallery}, [category] * pair_count)
    finally:
        adapter2.close()
    score_delta = np.abs(scores.astype(np.float64) - unmasked[:pair_count].astype(np.float64))

    native_ranks = _native_population_ranks(category)
    unmasked_differences = [i for i, (a, b) in enumerate(zip(adapter_style, argsort_unmasked))
                            if a != b]
    mask_explained = [i for i, (a, b) in enumerate(zip(adapter_style, native_style)) if a != b]
    rankings = {"query_count": len(native_style),
                "gallery_count": len(inputs["gallery_ids"]),
                # tie-convention comparison: adapter vs argsort, BOTH unmasked
                "adapter_vs_argsort_unmasked_identical": not unmasked_differences,
                "adapter_vs_argsort_unmasked_differences": unmasked_differences[:30],
                "adapter_vs_argsort_unmasked_difference_count": len(unmasked_differences),
                # the mask difference is a protocol difference, quantified rather than hidden
                "mask_explained_rank_shift_count": len(mask_explained),
                "mask_explained_rank_shift_all_positive": all(
                    adapter_style[i] - native_style[i] >= 0 for i in mask_explained),
                "mask_shift_is_exactly_one": all(
                    adapter_style[i] - native_style[i] == 1 for i in mask_explained),
                "adapter_style_recall": {f"r{k}": 100 * sum(1 for r in adapter_style if r <= k) / len(adapter_style)
                                        for k in (1, 10, 50)},
                "native_style_recall": {f"r{k}": 100 * sum(1 for r in native_style if r <= k) / len(native_style)
                                        for k in (1, 10, 50)},
                "adapter_vs_native_masked_identical": adapter_style == native_style,
                "score_max_abs_delta_vs_dot": float(score_delta.max()),
                "score_pairs_compared": pair_count}
    if native_ranks is not None:
        differing = [i for i, (a, b) in enumerate(zip(native_style, native_ranks)) if a != b]
        unexplained = []
        for index in differing:
            row = sims[index]
            low, high = sorted((native_style[index], native_ranks[index]))
            between = np.argsort(-row)[low - 1:high]
            if not np.all(np.abs(row[between] - row[target_positions[index]]) <= TOL["tie_epsilon"]):
                unexplained.append(index)
        rankings.update({
            "native_aggregate_ranks_compared": len(native_ranks),
            "rank_differences": len(differing),
            "rank_difference_indices_head": differing[:20],
            "unexplained_rank_differences": len(unexplained),
            "first_unexplained_index": unexplained[0] if unexplained else None,
            "max_abs_rank_delta": int(max([abs(a - b) for a, b in
                                          zip(native_style, native_ranks)] or [0])),
        })
    else:
        rankings["native_aggregate_ranks_compared"] = None
    return rankings


def refresh_metrics_block(categories: tuple[str, ...]) -> None:
    """Re-derive the `metrics_parity` block on the artifacts this script just rewrote.

    `main()` writes each native artifact from scratch, so the hit-count block added by
    `20261112-parity-repair/scripts/limn_hits.py` would be lost. Deriving it needs only stored data,
    so it is re-run here rather than left to a manual step that will be forgotten.
    """
    import importlib.util

    path = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261112-parity-repair"
                "/scripts/limn_hits.py")
    if not path.is_file():
        print(f"# metrics block NOT refreshed: {path} is missing; "
              "`metrics_parity` will report missing", flush=True)
        return
    spec = importlib.util.spec_from_file_location("limn_hits", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for line in module.refresh(categories):
        print(f"# metrics: {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    phase = argv[argv.index("--phase") + 1] if "--phase" in argv else "native"
    categories = CATEGORIES
    if "--category" in argv:
        categories = (argv[argv.index("--category") + 1],)
    config = resolve_config()
    RAW.mkdir(parents=True, exist_ok=True)

    for category in categories:
        target = OUT / f"g1_g2.{category}.{phase}.json"
        print(f"# {category}/{phase}: start", flush=True)
        if phase == "native":
            payload = {"category": category, "phase": phase, "tolerances": TOL,
                       "standard": STANDARD_V2,
                       "layer1_identity": native_phase(category, config)}
        elif phase == "adapter":
            body = adapter_phase(category, config)
            components = body.pop("component_parity", None)
            payload = {"category": category, "phase": phase, "tolerances": TOL,
                       "standard": STANDARD_V2,
                       "layer2_4_forward": body}
            # Beside the body, at the top level: `parity_gate._component_checks` reads the WHOLE
            # document, exactly as it reads `relation_state` there.
            if components is not None:
                payload["component_parity"] = components
        else:
            raise SystemExit(f"unknown phase {phase!r}; use native or adapter")
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"# {category}/{phase}: wrote {target.name}", flush=True)
        if phase == "native":
            print(json.dumps({
                "gallery_order_identical": payload["layer1_identity"]["gallery_order_identical"],
                "reference_ids_identical": payload["layer1_identity"]["reference_ids_identical"],
                "target_ids_identical": payload["layer1_identity"]["target_ids_identical"],
                "model_text_identical": payload["layer1_identity"]["model_text_identical_to_author_composition"],
                "sample_indices": payload["layer1_identity"]["sample_indices"],
                "rgb_noop": payload["layer1_identity"]["sample_convert_to_rgb_is_noop"],
            }, indent=1))
        else:
            print(json.dumps({
                "sample_query": payload["layer2_4_forward"]["sample_forward"]["query"],
                "sample_target": payload["layer2_4_forward"]["sample_forward"]["target"],
                "batch": payload["layer2_4_forward"]["batch_size_1_vs_n"],
                "restart": payload["layer2_4_forward"]["cache_off_restart"],
                "rankings": payload["layer2_4_forward"]["scoring_ranking"],
            }, indent=1))
    # Both phases rewrite at least one side of the pair, so the block is re-derived after the loop.
    refresh_metrics_block(categories)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
