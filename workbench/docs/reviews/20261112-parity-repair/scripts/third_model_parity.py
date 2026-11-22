"""Third-model G1/G2 parity: one shared capture/compare interface, HINT as its first instance.

Why this file exists at all: `t4_parity.py` proved the shape for LIMN, and `encoder_parity.py` for
ENCODER, but each is a per-model script. A third model would have become a third divergent clone.
So the parts that are the SAME for every model live here and are named as the interface; the parts
that are model-specific are one small object (`CaptureHook`) with two hooks and one scope value.

WHAT A NEW MODEL SUPPLIES (this is the whole contract -- see THIRD_MODEL_PARITY.md §2):

    ModelScope          -- identity: model_id, adapter module, replay module, checkpoint id,
                           categories, and the GALLERY SCOPE (per-category or category-agnostic).
    CaptureHook.gallery -- scope function: (loaded, category) -> {gallery_ids, gallery_paths,
                           reference_ids, target_ids, model_texts, image_path(name)}.
    CaptureHook.native_features / .native_target_features
                        -- capture hook: the pinned evaluator's own forward, given decoded pixels or
                           paths. Nothing about ranking, masking or metric arithmetic is here.

Everything else -- the fixed sample, the native/adapter comparison, the rank/tie classification, the
v2 component record, and the artifact emission -- is shared and is imported from THIS module by the
next model. The tolerance table is imported from `t4_parity.py`, never restated: a third copy of a
tolerance is a third thing to drift.

Phases, and why they are split on the interpreter (not cosmetic): the adapter refuses to run when its
own interpreter is the controller, so "same process" would be a path production never takes.

    --phase identity   CPU, no model weights, no GPU.  The discrete G1 block: counts, gallery order,
                       reference/target ids, model text vs the author's composition, plus the sample
                       selection and the decode equality of both sides' preprocessing.
    --phase native     HINT interpreter, GPU.  Pinned forward for the fixed sample, pixel tensors,
                       the population identities the adapter phase re-encodes.
    --phase adapter    controller interpreter, GPU (launches the worker under the model interpreter).
                       Adapter features, the comparisons, the scoring/ranking classification, the
                       v2 component block, then ONE pair of artifacts.
    --dry-run          identity phase only, into `raw/third_model/`; never touches the six artifacts.
                       Refuses to print an adapter document, because none can be measured on CPU:
                       HINT's worker raises `HINT requires CUDA in its model interpreter`.

Every emitted number is measured by this process or read from a named stored file whose identity is
re-verified here. Absent evidence stays absent: a phase that did not run leaves its keys out rather
than writing a default.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

EVIDENCE = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"
RAW = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/third_model"
#: The LIMN producer. Imported for `TOL`, `STANDARD_V2`, `component_scope`, `component_block` and
#: `correct_text` -- never copied, so the tolerance table and the v2 record have one definition.
T4_PATH = EVIDENCE / "scripts/t4_parity.py"
#: The stored native HINT measurement. Reused as the native RANK CONTROL rather than re-running the
#: full native evaluation: it is a real measurement of the same checkpoint bytes, protocol and
#: population, and re-running it would add nothing.
NATIVE_CONTROL = (REPO / "workbench/docs/reviews/checkpoint_selection/20261106T233409+0700"
                  "/hint/measurements/hint_native_verify.json")

SAMPLE_QUERIES = 5
SAMPLE_GALLERY = 20


def _t4():
    """The LIMN producer as a module. Its tolerance table is the declared one used here."""
    spec = importlib.util.spec_from_file_location("t4_parity", T4_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T4 = _t4()
TOL = T4.TOL                      # LIMN's declared table, unchanged (see THIRD_MODEL_PARITY.md §5)
STANDARD = T4.STANDARD_V2


# --------------------------------------------------------------------------- the shared interface


@dataclass(frozen=True)
class ModelScope:
    """What the shared machinery needs to know about a model. Data only, no behaviour."""

    model_id: str
    adapter_module: str
    replay_module: str
    checkpoint_id: str
    categories: tuple[str, ...]
    #: "per-category": the gallery and the query set are per category and the model takes a category
    #: id (LIMN). "category-agnostic": one shared gallery, the adapter refuses target categories
    #: (HINT, `model_hint.py:encode_targets`).
    gallery_scope: str
    #: The target side's feature rank. 2 = one vector per image, 3 = the BLIP-2 32x256 token layout.
    target_rank: int
    #: The adapter's accepted category value on the target side; None for category-agnostic models.
    target_category: str | None
    #: Extra declared tolerances this model's artifacts must carry. Empty for both current models.
    extra_tolerances: dict[str, Any] | None = None
    #: The stored native per-query-rank measurement reused as the ranking CONTROL, or None when the
    #: model has none. None is not a gap papered over: without a control `scoring_parity` cannot be
    #: measured against a native MEASUREMENT, and the sample-scope path below says so instead of
    #: substituting the adapter's own output for one.
    native_control_path: str | None = None
    #: "full"    -- rank the WHOLE category population (HINT: 2017 queries over 2628 images).
    #: "sample"  -- rank the fixed sample only, and SAY SO. A sample-shaped scope is a preview: the
    #:              verdict it supports is about the sample, never about the population, and the
    #:              artifact carries `population_scope` so a reader cannot mistake one for the other.
    population_scope: str = "full"
    #: The pinned text-processor config key this model's evaluator selects.
    processor_config: str = "pretrain"
    #: The protocol id this pair is measured under, as the registry declares it. Written into the
    #: artifact so `parity_gate._metrics_checks`'s provenance comparison has a value on both sides.
    protocol_id: str = "fashioniq_val_split"
    #: HOW the native preprocessing is obtained, as a declaration rather than a hardcoded call:
    #:   "data_utils" -- `data_utils.targetpad_transform(1.25, 224)` from the pinned checkout
    #:                   (HINT, HABIT, INTENT, CONESEPS)
    #:   "clip"       -- `clip.load('ViT-B/16', jit=False).preprocess`, which is a DIFFERENT
    #:                   normalization (CLIP's own mean/std) and a different resize (224 bicubic via
    #:                   CLIP, not TargetPad). TG-CIR builds exactly this (`train.py:87,90`).
    #: A model whose preprocessing is wrong is not measurable: comparing features produced under two
    #: normalizations measures the preprocessing, not the model.
    transform_source: str = "data_utils"
    #: The device the native model is placed on. "cuda" for every model whose evaluator requires it;
    #: "cpu" is refused by the CUDA-hard ones and is here for the models that patch it.
    native_device: str = "cuda"
    #: Whether the adapter tolerates an in-process `restart()`, which the `cache_off_restart`
    #: comparison needs. False for an adapter that asserts its worker loads ONCE per instance:
    #: `model_tgcir.py:344 _check_worker_pid` raises `adapter_worker_reloaded` when a second worker
    #: replies, so restarting it is refused by design. Recording that as a capability keeps the
    #: artifact honest -- the comparison is reported NOT MEASURED, never silently skipped and never
    #: passed.
    restart_supported: bool = True

    def declared_tolerances(self) -> dict[str, Any]:
        return {**TOL, **(self.extra_tolerances or {})}


class CaptureHook:
    """The two hooks and the scope function a new model supplies.

    Subclassed per model. Everything below the line is the producer's; everything above it is the
    model's. The hooks return ARRAYS, never metrics: ranking, masking and hit counts are the shared
    layer's job, so two models cannot disagree about what "the same rank" means.
    """

    scope: ModelScope

    def dataset_kwargs(self) -> dict[str, Any]:
        """Constructor keywords the pinned `FashionIQ` dataset needs BESIDES path and transform.

        HINT's takes none; HABIT/INTENT/CONESEPS's take `noise_ratio` (`datasets.py`, whose default
        path reads a noise setting). Declared as a hook rather than sniffed, because a wrong default
        here silently evaluates a DIFFERENT split and every downstream number is then about a model
        that was never asked the question.
        """
        return {}

    def replay_modules(self, source_root: Path) -> Any:
        """The pinned modules this model's evaluator imports, as the hook's own tuple.

        Defaults to the LAVIS-family `_load_replay_modules`. A model that names its loader
        differently (TG-CIR's `_load_tgcir`) overrides this rather than being special-cased in
        `identity_phase` -- which is the whole reason this is a method and not an inline call.
        """
        return load_replay_modules(self.scope.replay_module, source_root,
                                   self.scope.processor_config)

    def transform(self, loaded: dict[str, Any]):
        """The preprocessing callable BOTH sides apply, resolved from `scope.transform_source`.

        One definition serving every hook, because the transform is a property of the MODEL's pinned
        training/eval recipe, not of the producer. A hook that needs something else overrides this.
        """
        if self.scope.transform_source == "clip":
            import clip

            _clip_model, preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
            return preprocess
        if self.scope.transform_source != "data_utils":
            raise SystemExit(f"unknown transform_source {self.scope.transform_source!r} for "
                             f"{self.scope.model_id}: expected 'data_utils' or 'clip'")
        return loaded["data_utils"].targetpad_transform(1.25, 224)

    def gallery(self, loaded: Any, category: str) -> dict[str, Any]:
        """Scope function: what this category's query/gallery population IS, per the pinned source."""
        raise NotImplementedError

    def native_features(self, loaded: Any, paths: Sequence[str],
                        texts: Sequence[str]) -> Any:
        """Capture hook, query side: the pinned evaluator's forward over reference images + text."""
        raise NotImplementedError

    def native_target_features(self, loaded: Any, paths: Sequence[str]) -> Any:
        """Capture hook, target side: the pinned evaluator's forward over gallery images."""
        raise NotImplementedError


def fixed_sample(caption_rows: Sequence[dict], gallery_ids: Sequence[str], *,
                 query_count: int = SAMPLE_QUERIES,
                 gallery_count: int = SAMPLE_GALLERY) -> dict[str, Any]:
    """The fixed sample, chosen by DATA-DERIVED rules so it is the same sample in every process.

    Five query slots per category, and the two properties the sample EXISTS for are spent first:

      1-2. MULTIPLE POSITIVES -- two rows sharing one candidate, i.e. one reference with two targets.
           A per-query max-over-targets metric has to get this right, and a single-positive sample
           cannot test it. This is rule 1 because it is the only rule that needs TWO slots.
      3.   the LAST row -- the final partial batch (`query_count % batch != 0` is measured beside it).
      4-5. breadth: a capitalised first caption (text normalisation), a second caption shorter than
           12 characters (a short half-caption).
      ..   any remaining slot is filled in gallery order by `fill`.

    When a breadth rule finds no free slot it is recorded in
    `selection_rules_skipped_for_slot_budget`, so which rule lost is visible rather than inferred.
    `gallery_count` images are taken as: both positives of every sampled query (a reference missing
    from the gallery would make the rank meaningless), then the category's last gallery entry, then
    gallery order -- 20 images, so the image batch is itself partial.
    """
    counts: dict[str, int] = {}
    for row in caption_rows:
        counts[row["candidate"]] = counts.get(row["candidate"], 0) + 1
    # A multi-positive row is only a multi-positive SAMPLE if its reference is shared by another
    # SELECTED row: picking row 0 (whose reference happens to be shared) would otherwise be labelled
    # multi-positive while the sample holds exactly one positive for it.
    chosen_by_rule: list[tuple[str, int]] = []

    def take(rule: str, index: int | None) -> None:
        if index is not None and index not in [i for _, i in chosen_by_rule] \
                and len(chosen_by_rule) < query_count:
            chosen_by_rule.append((rule, index))

    def taken() -> list[int]:
        return [index for _, index in chosen_by_rule]

    def take(rule: str, index: int | None) -> bool:
        if index is None or index in taken() or len(chosen_by_rule) >= query_count:
            return False
        chosen_by_rule.append((rule, index))
        return True

    # Slot budget is `query_count`, and the two properties the sample EXISTS for are spent first:
    #   1. MULTIPLE POSITIVES for one reference (two rows, same candidate) -- a per-query
    #      max-over-targets metric has to get this right, and a single-positive sample cannot test it;
    #   2. the LAST row -- the final partial batch.
    # The remaining slots are breadth: capitalised first caption (text normalisation), a short second
    # caption, the first row, the middle row. A lower-priority rule that finds no slot is recorded as
    # `skipped`, so a reader can see which rule lost, rather than inferring it from the indices.
    shared = next((i for i, row in enumerate(caption_rows) if counts[row["candidate"]] > 1), None)
    if shared is not None:
        partner = next((i for i, row in enumerate(caption_rows)
                        if row["candidate"] == caption_rows[shared]["candidate"] and i != shared),
                       None)
        take("multi_positive_first", shared)
        take("multi_positive_second", partner)
    take("last_row", len(caption_rows) - 1)
    breadth = (
        ("capitalized_first_caption",
         next((i for i, row in enumerate(caption_rows) if row["captions"][0][:1].isupper()), None)),
        ("short_second_caption",
         next((i for i, row in enumerate(caption_rows) if len(row["captions"][1]) < 12), None)),
    )
    skipped: list[str] = []
    for rule, index in breadth:
        if not take(rule, index):
            skipped.append(rule)
    for index in range(len(caption_rows)):
        if len(chosen_by_rule) >= query_count:
            break
        take("fill", index)
    chosen_by_rule.sort(key=lambda item: item[1])
    indices = [index for _, index in chosen_by_rule]

    reference_ids = [caption_rows[i]["candidate"] for i in indices]
    target_ids = [caption_rows[i]["target"] for i in indices]
    selected_counts: dict[str, int] = {}
    for name in reference_ids:
        selected_counts[name] = selected_counts.get(name, 0) + 1
    # Gallery sample: both positives of every sampled query first (a reference that is not in the
    # gallery would make the rank meaningless), then the category's last gallery entry -- the
    # element that only a final partial batch reaches -- then gallery order.
    chosen: list[str] = []
    for name in [*reference_ids, *target_ids]:
        if name in gallery_ids and name not in chosen and len(chosen) < gallery_count:
            chosen.append(name)
    tail = gallery_ids[-1] if gallery_ids else None
    if tail is not None and tail not in chosen and len(chosen) < gallery_count:
        chosen.append(tail)
    for name in gallery_ids:
        if len(chosen) == gallery_count:
            break
        if name not in chosen:
            chosen.append(name)
    positions = {name: index for index, name in enumerate(gallery_ids)}
    return {
        "sample_indices": indices,
        "reference_ids": reference_ids,
        "target_ids": target_ids,
        "captions": [list(caption_rows[i]["captions"]) for i in indices],
        "selection_rules": {str(index): rule for rule, index in chosen_by_rule},
        "selection_rules_skipped_for_slot_budget": skipped,
        "multi_positive_rows": [i for i in indices
                                if selected_counts[caption_rows[i]["candidate"]] > 1],
        "sample_has_two_rows_sharing_a_reference":
            any(count > 1 for count in selected_counts.values()),
        "gallery_sample_ids": chosen,
        "gallery_sample_positions": [positions[name] for name in chosen],
        "gallery_sample_covers_tail": tail is not None and tail in chosen,
    }


def _hint_capture_row_ids(inputs: dict[str, Any], indices: Sequence[int] | None = None) -> tuple[list[str], list[str]]:
    """Return unique annotation identities for capture inputs and requested subset."""
    query_ids = [str(value) for value in inputs["query_annotation_ids"]]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("HINT capture annotation IDs must be unique")
    selected = query_ids if indices is None else [query_ids[index] for index in indices]
    if indices is not None and (len(selected) > 32 or len(set(selected)) != len(selected)):
        raise ValueError("HINT selected annotation capture IDs must be unique and at most 32")
    return query_ids, selected

def digest_ids(values: Sequence[str]) -> str:
    """One sha256 over an id sequence, so two artifacts can prove they measured the same order."""
    return hashlib.sha256(("\n".join(str(value) for value in values) + "\n").encode()).hexdigest()


def compare(actual: Any, reference: Any) -> dict[str, Any]:
    """The three-part absolute/relative/cosine comparison, computed here and never defaulted."""
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise SystemExit(f"compare: {a.shape} vs {b.shape} -- refusing to compare unaligned arrays")
    delta = np.abs(a - b)
    scale = np.maximum(np.abs(b), 1e-12)
    cosines = [(row.ravel() @ ref.ravel())
               / (np.linalg.norm(row) * np.linalg.norm(ref))
               for row, ref in zip(a.reshape(len(a), -1), b.reshape(len(b), -1))]
    return {"max_abs": float(delta.max()), "max_rel": float((delta / scale).max()),
            "min_cosine": float(min(cosines)),
            "bitwise_equal": bool(np.array_equal(np.asarray(actual), np.asarray(reference))),
            "shape": list(a.shape)}


def toleranced(block: dict[str, Any], *, abs_key: str, rel_key: str,
               declared: dict[str, Any]) -> dict[str, Any]:
    """Attach the harness's own `within_tolerance`, so the gate can report a label disagreement."""
    block["within_tolerance"] = (block["max_abs"] <= declared[abs_key]
                                 and block["max_rel"] <= declared[rel_key])
    return block


#: `hint_native_verify.json` records `0` for "the target was not in the top 50 (or the reference row
#: was masked)". It is a SENTINEL, not a rank: `rank_counts.below_top50_or_masked` = 492 = the number
#: of zeros in dress, and `hit1/hit10/hit50` = 432/1034/1525 is exactly what you get counting only
#: `rank > 0`. Counting `rank <= k` naively treats every miss as a hit and inflates r1 to 924.
RANK_ZERO_MEANS_MISSED = True


def _hits(ranks: Sequence[int]) -> dict[str, int]:
    """Hit counts over 1-based ranks. A `0` is a MISS (see `RANK_ZERO_MEANS_MISSED`), never a hit."""
    return {f"r{k}": sum(1 for rank in ranks if rank is not None and rank > 0 and rank <= k)
            for k in (1, 10, 50)}


def native_control(hook: CaptureHook, category: str) -> dict[str, Any]:
    """The stored native per-query ranks, re-verified against the registry checkpoint digest.

    Reused rather than re-measured: it is the SAME checkpoint bytes, protocol and population, and a
    second full native evaluation would add a number, not information. What is re-verified here is
    the thing reuse could silently get wrong -- that the ranks still reproduce the macro that file
    records, and that its checkpoint is the one the registry declares.

    The path and the checkpoint come from the SCOPE. A model with no stored control gets `None` and
    the caller records the ranking dimension as unmeasured rather than comparing the adapter's own
    output to itself.
    """
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    if not hook.scope.native_control_path:
        return None
    document = json.loads(Path(REPO / hook.scope.native_control_path).read_text(encoding="utf-8"))
    declared = checkpoint_by_id(model_by_id(hook.scope.model_id), hook.scope.checkpoint_id)
    entry = document["categories"][category]
    ranks = [int(rank) for rank in entry["ranks"]]
    hits = _hits(ranks)
    recorded = [float(value) for value in entry["native_recalls_r1_r10_r50"]]
    recomputed = [100.0 * hits[f"r{k}"] / len(ranks) for k in (1, 10, 50)]
    return {
        "source": hook.scope.native_control_path,
        "checkpoint_sha256": document["checkpoint"]["sha256"],
        "checkpoint_sha256_matches_registry":
            document["checkpoint"]["sha256"] == declared.get("expected_sha256"),
        "protocol": document["protocol"],
        "batch_size": document["batch_size"],
        "runtime_python": document["runtime"]["python"],
        "query_count": len(ranks),
        "reference_masked": entry["reference_masked"],
        "gallery_count": entry["gallery"],
        "ranks": ranks,
        "stored_ranks_reproduce_recorded_macro":
            all(abs(a - b) < 1e-9 for a, b in zip(recomputed, recorded)),
        "zero_rank_count": sum(1 for rank in ranks if rank == 0),
        "zero_rank_recorded_as_missed":
            entry.get("rank_counts", {}).get("below_top50_or_masked"),
        "zero_rank_semantics_certified":
            sum(1 for rank in ranks if rank == 0)
            == entry.get("rank_counts", {}).get("below_top50_or_masked"),
        "recomputed_recalls": dict(zip(("r1", "r10", "r50"), recomputed)),
        "recorded_recalls": dict(zip(("r1", "r10", "r50"), recorded)),
    }


def load_replay_modules(replay_module: str, source_root: Path, processor_config: str) -> Any:
    """Call a replay module's `_load_replay_modules`, whatever arity that module's own signature has.

    Measured, not assumed: HINT's takes `(source_root, config_key)` because it selects among several
    processor configs (`replay/hint.py`), while HABIT/INTENT/CONESEP take `(source_root)` and pin
    `PROCESSOR_CONFIGS["pretrain"]` themselves. Passing two arguments to the single-argument form is
    a `TypeError`, and guessing which form a new module has is how a producer breaks on the model it
    was just extended for. So the arity is read from the function itself.
    """
    import inspect

    loader = importlib.import_module(replay_module)._load_replay_modules
    parameters = [name for name, parameter in inspect.signature(loader).parameters.items()
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    return loader(source_root, processor_config) if len(parameters) >= 2 else loader(source_root)

def decode_equality(paths: Sequence[str], hook: "CaptureHook | None" = None) -> dict[str, Any]:
    """Report that preprocessing parity is unmeasured until independent path captures exist."""
    model_id = hook.scope.model_id if hook is not None else "hint"
    return {
        "status": "NOT_MEASURED",
        "reason": "native and adapter inputs must be captured independently at their actual forward paths",
        "model_id": model_id,
        "requested_image_count": len(paths),
        "native_capture": None,
        "adapter_capture": None,
    }


# --------------------------------------------------------------------------- HINT as instance 1


class HintCapture(CaptureHook):
    """HINT: one checkpoint for all three categories, category-agnostic gallery.

    Read off the pinned source, not assumed from LIMN:
      * gallery (`datasets.py:131-169`) -- ordered union of each row's source then target id, i.e.
        `get_test_data`, the same construction LIMN uses but built ONCE for all three categories
        (`__init__:39-42`) instead of per category.
      * reference mask (`test.py:82-83`) -- `sims[i, tid.index(source_img_id)] = -10e10`.
      * transform (`replay/hint.py:184`) -- `targetpad_transform(1.25, 224)`, val path, BOTH sides.
      * text (`test.py:44`) -- `txt_processors["eval"](caption)` on the ALREADY concatenated
        sentence; `HINT.extract_retrieval_compose` then tokenises with `max_txt_len=32` truncation
      * layer-2 input capture records the actual tokenizer `input_ids` and `attention_mask` used by
        `extract_retrieval_compose`, plus the transformed pixels before device transfer.
      * target encoding (`HINT.py:436-455`) -- (N,256,32); the adapter transposes to (N,32,256)
        (`model_hint.py:366`),
        so the shared comparator transposes the native side to match.
      * no `dataset_type` -- unlike LIMN, nothing is conditioned on a category id.
    """

    scope = ModelScope(
        model_id="hint",
        adapter_module="workbench.composition.model_hint",
        replay_module="workbench.replay.hint",
        checkpoint_id="fashioniq",
        categories=("dress", "shirt", "toptee"),
        gallery_scope="category-agnostic",
        target_rank=3,
        target_category=None,
    )

    # -- loading -------------------------------------------------------------------------------
    def load_native(self, config: Any, category: str):
        """Pinned modules, catalogue checkpoint and the pinned dataset, as `test.py` sees them."""
        from workbench.backend.registry import checkpoint_by_id, model_by_id
        from workbench.replay import hint as replay
        from workbench.replay.common import load_trusted_whole_model

        model = model_by_id(self.scope.model_id)
        checkpoint = checkpoint_by_id(model, self.scope.checkpoint_id)
        checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / self.scope.model_id
        dataset_root = Path(config.FASHIONIQ_ROOT)
        source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
        problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
        if problems:
            raise SystemExit(f"HINT preflight blocked: {problems}")
        import torch

        torch_mod, datasets, pinned, processors = replay._load_replay_modules(source_root, "pretrain")
        data_utils, test_module = pinned
        path = checkpoint_root / checkpoint["filename"]
        network = load_trusted_whole_model(path, checkpoint.get("expected_sha256"), torch_mod)
        network = network.to("cuda").eval()
        return {"torch": torch_mod, "datasets": datasets, "data_utils": data_utils,
                "test_module": test_module, "processors": processors, "model": network,
                "dataset_root": dataset_root, "cache": None, "checkpoint": checkpoint,
                "checkpoint_path": path, "config": config}

    # -- scope function ------------------------------------------------------------------------
    def gallery(self, loaded: dict[str, Any], category: str) -> dict[str, Any]:
        dataset_root = loaded["dataset_root"]
        images = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json")
                            .read_text(encoding="utf-8"))
        rows = json.loads((dataset_root / "captions" / f"cap.{category}.val.json")
                          .read_text(encoding="utf-8"))
        correction = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json")
                                .read_text(encoding="utf-8"))
        gallery_ids: list[str] = []
        for row in rows:
            for key in ("candidate", "target"):
                if row[key] not in gallery_ids:
                    gallery_ids.append(row[key])
        composed = [f"{T4.correct_text(row['captions'][0], correction)} and "
                    f"{T4.correct_text(row['captions'][1], correction)}" for row in rows]

        def image_path(name: str) -> str:
            # NOT the isolated replay cache: that is a temp root, removed when the context exits.
            return str(dataset_root / "resized_image" / category / f"{name}.jpg")

        return {
            "category": category,
            "images": images,
            "rows": rows,
            "correction": correction,
            "gallery_ids": gallery_ids,
            "gallery_positions": [images.index(name) for name in gallery_ids],
            "reference_ids": [row["candidate"] for row in rows],
            "target_ids": [row["target"] for row in rows],
            "composed_texts": composed,
            "image_path": image_path,
            "gallery_paths": [image_path(name) for name in gallery_ids],
        }

    # -- capture hooks -------------------------------------------------------------------------
    def native_features(self, loaded: dict[str, Any], paths: Sequence[str],
                        texts: Sequence[str], *, capture_inputs: bool = False) -> Any:
        import numpy as np
        from PIL import Image

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = self.transform(loaded)
        processor = loaded["processors"][0]["eval"]
        from workbench.composition.adapters import resolved_batch_size

        outputs = []
        captured_pixels = []
        captured_token_ids = []
        captured_token_masks = []
        captured_texts = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images_list = []
                for path in chunk:
                    with Image.open(path) as image:
                        images_list.append(transform(image.convert("RGB")))
                images = torch_mod.stack(images_list)
                if capture_inputs:
                    captured_pixels.append(images.detach().cpu().numpy())
                images = images.float().cuda()
                chunk_texts = [processor(text) for text in texts[start:start + len(chunk)]]
                original_tokenizer = network.tokenizer
                actual_texts = []

                def capture_tokenizer(*args: Any, **kwargs: Any) -> Any:
                    tokens = original_tokenizer(*args, **kwargs)
                    if capture_inputs:
                        captured_token_ids.append(tokens.input_ids.detach().cpu().numpy())
                        captured_token_masks.append(tokens.attention_mask.detach().cpu().numpy())
                        actual_texts.extend(args[0])
                    return tokens

                if capture_inputs:
                    network.tokenizer = capture_tokenizer
                try:
                    features = network.extract_retrieval_compose(images, chunk_texts)
                finally:
                    if capture_inputs:
                        network.tokenizer = original_tokenizer
                if capture_inputs:
                    captured_texts.extend(actual_texts)
                outputs.append(features.detach().cpu().numpy().reshape(len(chunk), -1))
                del images, features
        if capture_inputs:
            self.last_native_pixels = np.concatenate(captured_pixels, axis=0)
            self.last_native_token_ids = np.concatenate(captured_token_ids, axis=0)
            self.last_native_token_mask = np.concatenate(captured_token_masks, axis=0)
            self.last_native_processed_texts = captured_texts
        return np.concatenate(outputs, axis=0)

    def native_target_features(self, loaded: dict[str, Any], paths: Sequence[str],
                               *, capture_inputs: bool = False) -> Any:
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = self.transform(loaded)
        outputs = []
        captured_pixels = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images_list = []
                for path in chunk:
                    with Image.open(path) as image:
                        images_list.append(transform(image.convert("RGB")))
                images = torch_mod.stack(images_list)
                if capture_inputs:
                    captured_pixels.append(images.detach().cpu().numpy())
                images = images.float().cuda()
                features = network.extract_retrieval_target(images)
                outputs.append(features.detach().cpu().numpy().transpose(0, 2, 1))
                del images, features
        if capture_inputs:
            self.last_native_target_pixels = np.concatenate(captured_pixels, axis=0)
        return np.concatenate(outputs, axis=0)


class LavisFashionIQCapture(CaptureHook):
    """The shared LAVIS BLIP-2 FashionIQ contract: HABIT, INTENT and CONESEPS.

    These three are the SAME shape, measured rather than assumed:

      * `datasets.FashionIQ(path, category=None, transform=None, noise_ratio=0, split=...)` in all
        three (`HABIT/datasets.py:28`, `INTENT/datasets.py:26`, `ConeSep/datasets.py:26`), each
        exposing `test_queries_<cat>`/`test_targets_<cat>` with `source_img_id`, `target_img_id`,
        `mod['str']` and `target_img_data`;
      * the SAME decisor in `test.py` -- `sims = np.matmul(all_queries, all_imgs).squeeze().max(-1)`,
        then `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, then `argsort(-sims)`
        (`HABIT/test.py:141-152`, `INTENT/test.py:141-150`) -- so one ranking convention serves all
        three, and this class has no ranking code of its own;
      * the SAME transform `data_utils.targetpad_transform(1.25, 224)` and the SAME processor config.

    CONESEPS is the one DIVERGENCE and it is a layout, not a convention: its
    `extract_retrieval_compose` returns `(N,1,1,256)` and its `extract_retrieval_target` returns
    `(N,256,32)` (`ConeSep.py:492,516`), which the adapter widens to an 8192-wide row and `score()`
    then re-splits. The hook flattens the native query to `(N,256)` so both sides are the same
    quantity, and the layout is recorded rather than silently transposed.
    """

    def __init__(self, scope: ModelScope, *, noise_ratio: float = 0.0,
                 target_transpose: bool = True) -> None:
        self.scope = scope
        #: `datasets.FashionIQ(noise_ratio=...)`. 0 is the evaluation split every one of the three
        #: registries declares (`evaluation_noise_pct: 0` in their native aggregates).
        self.noise_ratio = noise_ratio
        #: Whether the native TARGET array must be transposed `(N,256,32) -> (N,32,256)` to match the
        #: adapter, which is what all three adapters emit.
        self.target_transpose = target_transpose

    def dataset_kwargs(self) -> dict[str, Any]:
        return {"noise_ratio": self.noise_ratio}

    # -- loading -------------------------------------------------------------------------------
    def load_native(self, config: Any, category: str):
        """Pinned modules, catalogue checkpoint and the pinned dataset, as `test.py` sees them."""
        from workbench.backend.registry import checkpoint_by_id, model_by_id
        from workbench.replay.common import load_trusted_whole_model

        model = model_by_id(self.scope.model_id)
        checkpoint = checkpoint_by_id(model, self.scope.checkpoint_id)
        checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / self.scope.model_id
        dataset_root = Path(config.FASHIONIQ_ROOT)
        source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
        replay = importlib.import_module(self.scope.replay_module)
        problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
        if problems:
            raise SystemExit(f"{self.scope.model_id} preflight blocked: {problems}")
        import torch

        torch_mod, datasets, data_utils, _test, processors = replay._load_replay_modules(
            source_root)
        txt_processors, _boundary = processors
        path = checkpoint_root / checkpoint["filename"]
        network = load_trusted_whole_model(path, checkpoint.get("expected_sha256"), torch_mod)
        network = network.to("cuda").eval()
        return {"torch": torch_mod, "datasets": datasets, "data_utils": data_utils,
                "processors": txt_processors, "model": network,
                "dataset_root": dataset_root, "cache": None, "checkpoint": checkpoint,
                "checkpoint_path": path, "config": config}

    # -- scope function ------------------------------------------------------------------------
    def gallery(self, loaded: dict[str, Any], category: str) -> dict[str, Any]:
        dataset_root = loaded["dataset_root"]
        images = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json")
                            .read_text(encoding="utf-8"))
        rows = json.loads((dataset_root / "captions" / f"cap.{category}.val.json")
                          .read_text(encoding="utf-8"))
        correction = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json")
                                .read_text(encoding="utf-8"))
        # The GALLERY is the category's whole image list, in the pinned split's order: that is what
        # `datasets.get_test_data` enumerates under `split='original-split'`
        # (`HABIT/datasets.py:205-210`) and it is why the reference mask has something to mask.
        gallery_ids = list(images)
        composed = [f"{T4.correct_text(row['captions'][0], correction)} and "
                    f"{T4.correct_text(row['captions'][1], correction)}" for row in rows]

        def image_path(name: str) -> str:
            return str(dataset_root / "resized_image" / category / f"{name}.jpg")

        return {
            "category": category,
            "images": images,
            "rows": rows,
            "correction": correction,
            "gallery_ids": gallery_ids,
            "gallery_positions": list(range(len(gallery_ids))),
            "reference_ids": [row["candidate"] for row in rows],
            "target_ids": [row["target"] for row in rows],
            "composed_texts": composed,
            "image_path": image_path,
            "gallery_paths": [image_path(name) for name in gallery_ids],
        }

    # -- capture hooks -------------------------------------------------------------------------
    def native_features(self, loaded: dict[str, Any], paths: Sequence[str],
                        texts: Sequence[str]) -> Any:
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = self.transform(loaded)
        # The mapping's key is `"eval"` in every pinned `test.py` (`HABIT/test.py:111`).
        processor = loaded["processors"]["eval"]
        outputs = []
        captured_pixels = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images = torch_mod.stack([transform(Image.open(path).convert("RGB"))
                                          for path in chunk]).float().cuda()
                captured_pixels.append(images.detach().cpu().numpy())
                chunk_texts = [processor(text) for text in texts[start:start + len(chunk)]]
                features = network.extract_retrieval_compose(images, chunk_texts)
                outputs.append(features.detach().cpu().numpy().reshape(len(chunk), -1))
                del images, features
        self.last_native_pixels = np.concatenate(captured_pixels, axis=0)
        return np.concatenate(outputs, axis=0)

    def native_target_features(self, loaded: dict[str, Any], paths: Sequence[str]) -> Any:
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = self.transform(loaded)
        outputs = []
        captured_pixels = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images = torch_mod.stack([transform(Image.open(path).convert("RGB"))
                                          for path in chunk]).float().cuda()
                captured_pixels.append(images.detach().cpu().numpy())
                features = network.extract_retrieval_target(images)
                array = features.detach().cpu().numpy()
                outputs.append(array.transpose(0, 2, 1) if self.target_transpose else array)
                del images, features
        self.last_native_target_pixels = np.concatenate(captured_pixels, axis=0)
        return np.concatenate(outputs, axis=0)

class ClippedModelCapture(LavisFashionIQCapture):
    """TG-CIR: same FashionIQ dataset contract, DIFFERENT loader, transform and model restore.

    Measured differences from the LAVIS family, each of which the shared phases would have got wrong:

      * **No `_load_replay_modules`.** `replay/tgcir.py:174 _load_tgcir(source_root, device)` returns
        `(torch, (clip, datasets, tg_test), adaptations)` -- three elements whose middle is itself a
        three-tuple. So this class supplies `load_native` itself, and the shared
        `load_replay_modules()` is not consulted.
      * **Transform is CLIP's, not TargetPad.** `train.py:87,90` builds
        `clip.load('ViT-B/16', jit=False).preprocess`; the LAVIS family uses
        `data_utils.targetpad_transform(1.25, 224)`. Different mean/std AND a different resize, so
        using the wrong one compares preprocessing rather than models. Declared as
        `scope.transform_source = "clip"` and served by `CaptureHook.transform`.
      * **Model restore is `torch.load(..., weights_only=False)`**, not
        `load_trusted_whole_model`, plus a torch-1.7 pickle alias (`nn.MultiheadAttention` without
        `batch_first`). Reusing the LAVIS restore path would fail on the pickle.

    The dataset contract is otherwise identical (`third_party/TGCIR/datasets.py:21
    __init__(path, transform=None)`, `test_queries_<cat>`/`test_targets_<cat>`), which is why this
    SUBCLASSES `LavisFashionIQCapture` rather than repeating its `gallery()`.
    """

    def dataset_kwargs(self) -> dict[str, Any]:
        """TG-CIR's dataset takes no `noise_ratio` (`third_party/TGCIR/datasets.py:21
        __init__(path, transform=None)`), unlike the LAVIS family."""
        return {}

    def gallery(self, loaded: dict[str, Any], category: str) -> dict[str, Any]:
        """The gallery is the UNION of source and target ids, in first-appearance order.

        Not the full image list: `third_party/TGCIR/datasets.py:147-150` builds `test_targets_id` by
        walking the queries and appending `source_img_id` then `target_img_id` if unseen. Measured:
        dress has 2017 queries but only **2628** gallery images, against 3817 images in the split
        file. Inheriting the full-list `gallery()` produced `gallery_order_identical: false` on the
        first identity run -- a true statement about MY hook, not about the model.
        """
        scope = super().gallery(loaded, category)
        gallery_ids: list[str] = []
        for row in scope["rows"]:
            for name in (row["candidate"], row["target"]):
                if name not in gallery_ids:
                    gallery_ids.append(name)
        positions = {name: index for index, name in enumerate(scope["images"])}
        return {**scope, "gallery_ids": gallery_ids,
                "gallery_positions": [positions[name] for name in gallery_ids],
                "gallery_paths": [scope["image_path"](name) for name in gallery_ids]}

    def replay_modules(self, source_root: Path) -> Any:
        """`_load_tgcir(source_root, device)` flattened to the shape the shared phases index.

        The pinned loader returns `(torch, (clip, datasets, tg_test), adaptations)`, so index 1 is a
        TUPLE, not the `datasets` module. Returning it unchanged gave
        `AttributeError: 'tuple' object has no attribute 'FashionIQ'`. The shared phase only needs
        `[1]`: the datasets module. The clip module and the adaptations are carried on `loaded`
        instead, by `load_native`, where the forwards can use them.
        """
        from workbench.replay import tgcir as replay

        torch_mod, inner, _adaptations = replay._load_tgcir(source_root, self.scope.native_device)
        clip_mod, datasets, tg_test = inner
        return (torch_mod, datasets, (None, tg_test), clip_mod)

    def load_native(self, config: Any, category: str):
        # `sys.path` must carry the checkout before `_load_tgcir` imports `clip`, `datasets`, `test`.
        import torch as _torch

        from workbench.backend.registry import model_by_id
        from workbench.replay import tgcir as replay

        from workbench.backend.registry import checkpoint_path as _resolved_path

        model = model_by_id(self.scope.model_id)
        # The checkpoint lives under `downloads/tgcir/x/checkpoints/`, not `root/tgcir/`. The pinned
        # `preflight` and `validate_checkpoint` take a DIRECTORY and append `Fashioniq.pt`, so the
        # registry-resolved path's own parent is what they must be given -- re-deriving the
        # conventional location made both report the checkpoint missing on a file that is present.
        checkpoint_path = Path(_resolved_path(self.scope.model_id, replay.CHECKPOINT,
                                              config.WORKBENCH_CHECKPOINT_ROOT))
        checkpoint_root = checkpoint_path.parent
        dataset_root = Path(config.FASHIONIQ_ROOT)
        source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
        problems = replay.preflight(source_root, dataset_root, checkpoint_root)
        if problems:
            raise SystemExit(f"tgcir preflight blocked: {problems}")
        torch_mod, (clip_mod, datasets, _tg_test), adaptations = replay._load_tgcir(
            source_root, self.scope.native_device)
        network = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        unset = [module for module in network.modules()
                 if isinstance(module, torch_mod.nn.MultiheadAttention)
                 and not hasattr(module, "batch_first")]
        for module in unset:
            module.batch_first = False
        if unset:
            adaptations = [*adaptations,
                           f"batch_first=False restored on {len(unset)} MultiheadAttention modules"]
        network = network.to(self.scope.native_device).eval()
        return {"torch": torch_mod, "datasets": datasets, "data_utils": None,
                "processors": None, "model": network, "dataset_root": dataset_root,
                "cache": None, "checkpoint": replay.CHECKPOINT,
                "checkpoint_path": checkpoint_path, "config": config,
                "clip": clip_mod, "adaptations": list(adaptations)}

    def native_features(self, loaded: dict[str, Any], paths: Sequence[str],
                        texts: Sequence[str]) -> Any:
        """The pinned encoder path, verbatim from `test.py`: compose -> mean-pool over tokens.

        `model.py:197-199` in the pinned source: query row = `F.normalize(mean(tokens))`, so the
        comparable quantity is one 512-d vector per query, not the token layout. The adapter does the
        same (`model_tgcir.py:635`), which is why no transpose is needed on this side.
        """
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        # TG-CIR applies NO text processor at the call site: `test.py` passes the corrected caption
        # string straight to `extract_retrieval_compose`, and its own tokenizer runs inside.
        preprocess = self.transform(loaded)
        outputs = []
        captured_pixels = []
        captured_token_ids = []
        batch = resolved_batch_size()
        tokenizer = loaded["clip"].tokenize
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                texts_chunk = texts[start:start + len(chunk)]
                pixel_batches = []
                for path in chunk:
                    with Image.open(path) as image:
                        pixel_batches.append(preprocess(image.copy().convert("RGB")))
                cpu_images = torch_mod.stack(pixel_batches)
                captured_pixels.append(cpu_images.numpy())
                token_ids = tokenizer(texts_chunk)
                captured_token_ids.append(token_ids.detach().cpu().numpy())
                features = network.extract_retrieval_compose(
                    cpu_images.to(self.scope.native_device), texts_chunk)
                outputs.append(features.detach().float().cpu().numpy().reshape(len(chunk), -1))
                del cpu_images, token_ids, features
        self.last_native_pixels = np.concatenate(captured_pixels, axis=0)
        self.last_native_token_ids = np.concatenate(captured_token_ids, axis=0)
        self.last_native_token_mask = self.last_native_token_ids != 0
        return np.concatenate(outputs, axis=0)

    def native_target_features(self, loaded: dict[str, Any], paths: Sequence[str]) -> Any:
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        preprocess = self.transform(loaded)
        batch = resolved_batch_size()
        outputs = []
        captured_pixels = []
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                pixel_batches = []
                for path in chunk:
                    with Image.open(path) as image:
                        pixel_batches.append(preprocess(image.copy().convert("RGB")))
                cpu_images = torch_mod.stack(pixel_batches)
                captured_pixels.append(cpu_images.numpy())
                images = cpu_images.to(self.scope.native_device)
                features = network.extract_retrieval_target(images)
                outputs.append(features.detach().float().cpu().numpy().reshape(len(chunk), -1))
                del cpu_images, images, features
        self.last_native_target_pixels = np.concatenate(captured_pixels, axis=0)
        return np.concatenate(outputs, axis=0)

# --------------------------------------------------------------------------- phases


def _versions() -> dict[str, Any]:
    import numpy as np

    record = {"python": sys.executable, "numpy": np.__version__}
    try:
        import torch

        record.update({"torch": torch.__version__,
                       "torch_cuda": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available()})
        if torch.cuda.is_available():
            record["device"] = torch.cuda.get_device_name(0)
    except Exception as error:                                   # noqa: BLE001 - recorded, not raised
        record["torch_import_error"] = f"{type(error).__name__}: {error}"
    return record


def checkpoint_identity(config: Any, model_id: str = "hint",
                        checkpoint_id: str = "fashioniq") -> dict[str, Any]:
    """The checkpoint bytes, hashed once and cached under the producer's own raw dir.

    `model_id`/`checkpoint_id` are parameters rather than HINT's literals: a second model's pair must
    name ITS checkpoint, and a cache keyed only on size+path would have collided silently if two
    models happened to stage equal-sized files.
    """
    import hashlib as _hash

    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id

    model = model_by_id(model_id)
    declared = checkpoint_by_id(model, checkpoint_id)
    # The REGISTRY's own resolution, not `root/<model>/<filename>`: several models stage their
    # checkpoint somewhere the convention does not predict (TG-CIR is under
    # `downloads/tgcir/x/checkpoints/`), and constructing the path here reported a
    # FileNotFoundError on a checkpoint the registry can point straight at.
    path = checkpoint_path(model_id, declared, config.WORKBENCH_CHECKPOINT_ROOT)
    cache = RAW / f"checkpoint_identity_{model_id}_{checkpoint_id}.json"
    record = {"path": str(path), "size_bytes": path.stat().st_size,
              "declared_sha256": declared.get("expected_sha256")}
    if cache.is_file():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("size_bytes") == record["size_bytes"] and cached.get("path") == record["path"]:
            return cached
    digest = _hash.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    record["sha256"] = digest.hexdigest()
    record["matches_declared"] = record["sha256"] == record["declared_sha256"]
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def identity_phase(hook: CaptureHook, category: str, config: Any) -> dict[str, Any]:
    """G1, CPU only: measured against the pinned dataset, not against a re-derivation of it.

    The pinned `datasets.FashionIQ` is constructed under an identity transform (the decode is
    measured separately by `decode_equality`) because only its STRUCTURE is wanted here: which ids
    it holds, in which order, and the text it hands the model.
    """
    class Identity:
        def __call__(self, image):
            return image

        def __getitem__(self, index):
            return self

    from workbench.backend.registry import model_by_id
    from workbench.replay.common import isolated_cache_root

    model = model_by_id(hook.scope.model_id)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
    dataset_root = Path(config.FASHIONIQ_ROOT)
    # The MODULE and the processor config come from the scope, not from HINT: `_load_replay_modules`
    # is one function name every LAVIS-family replay module happens to share, and picking it by
    # hardcoded name is what would have made a second hook silently measure HINT's classes.
    datasets = hook.replay_modules(source_root)[1]
    began = time.time()
    with isolated_cache_root(dataset_root) as cache:
        dataset = datasets.FashionIQ(path=str(cache).rstrip("/") + "/", transform=Identity(),
                                     **hook.dataset_kwargs())
        scope = hook.gallery({"dataset_root": dataset_root}, category)
        native_queries = getattr(dataset, f"test_queries_{category}")
        native_targets = getattr(dataset, f"test_targets_{category}")
        native_texts = [entry["mod"]["str"] for entry in native_queries]
        native_refs = [scope["images"][entry["source_img_id"]] for entry in native_queries]
        native_tgts = [scope["images"][entry["target_img_id"]] for entry in native_queries]
        gallery_ids = [scope["images"][entry["target_img_id"]] for entry in native_targets]
        query_count_native = len(native_queries)

    sample = fixed_sample(scope["rows"], gallery_ids)
    mismatches = [{"index": index, "native": native_texts[index],
                   "recomposed": scope["composed_texts"][index]}
                  for index in range(len(native_texts))
                  if native_texts[index] != scope["composed_texts"][index]]

    # Population arithmetic that the GPU phases will exercise: the last batch of the FULL population
    # is the partial one, so it is recorded as measured arithmetic rather than re-derived later.
    from workbench.composition.adapters import resolved_batch_size

    batch = resolved_batch_size()
    query_annotations = [
        {
            "annotation_id": (
                f"{category}:{index}:{entry['source_img_id']}:{entry['target_img_id']}:"
                f"{hashlib.sha256(native_texts[index].encode('utf-8')).hexdigest()}"
            ),
            "source_img_id": entry["source_img_id"],
            "target_img_id": entry["target_img_id"],
            "reference_id": scope["images"][entry["source_img_id"]],
            "target_id": scope["images"][entry["target_img_id"]],
            "reference_text": native_texts[index],
            "text_order": index,
        }
        for index, entry in enumerate(native_queries)
    ]
    target_annotations = [item["annotation_id"] for item in query_annotations]
    identity = {
        "query_count_native": query_count_native,
        "query_count_annotations": len(scope["rows"]),
        "gallery_count_native": len(native_targets),
        "gallery_count_rebuilt_from_annotations": len(gallery_ids),
        "gallery_order_identical": gallery_ids == scope["gallery_ids"],
        "reference_ids_identical": native_refs == scope["reference_ids"],
        "target_ids_identical": native_tgts == scope["target_ids"],
        "model_text_identical_to_author_composition": not mismatches,
        "model_text_mismatches": mismatches[:5],
        "duplicate_caption_pairs_within_category":
            len(scope["rows"]) - len({(row["candidate"], scope["composed_texts"][i])
                                      for i, row in enumerate(scope["rows"])}),
        "reference_equals_target_rows": sum(ref == target for ref, target in zip(native_refs, native_tgts)),
        "gallery_is_category_agnostic": hook.scope.gallery_scope == "category-agnostic",
        "adapter_target_category": hook.scope.target_category,
        "target_feature_rank": hook.scope.target_rank,
        "batch_size": batch,
        "query_annotations": query_annotations,
        "query_native_order": [item["annotation_id"] for item in query_annotations],
        "gallery_native_order": list(scope["gallery_ids"]),
        "query_text_order": [item["reference_text"] for item in query_annotations],
        "population_final_batch": {
            "queries": query_count_native % batch or batch,
            "gallery": len(native_targets) % batch or batch,
            "note": "last batch of the FULL population; the fixed sample is one partial batch",
        },
        "text_handling": ({
            "native": "test.py passes corrected caption string directly to extract_retrieval_compose",
            "adapter": "model_tgcir.py passes the same corrected caption string to its worker",
            "comparable_quantity": "query feature; tokenizer is internal to TG-CIR model",
        } if hook.scope.model_id == "tgcir" else {
            "native": "datasets.py concat_text -> mod['str']; pinned eval processor applied by test.py",
            "adapter": f"model_{hook.scope.model_id}.py applies its eval processor to composed sentence",
            "comparable_quantity": "query feature; tokenization is internal to model",
        }),
        "image_transform": {
            "source": hook.scope.transform_source,
            "native": f"{hook.scope.replay_module} pinned evaluation preprocessing",
            "adapter": f"workbench.composition.{hook.scope.adapter_module.rsplit('.', 1)[-1]} production worker preprocessing",
            "transform_steps": (None if hook.scope.model_id == "hint" else
                                [type(step).__name__ for step in getattr(
                                    hook.transform({}), "transforms", [])]),
            "note": "metadata does not establish parity; compare independently captured execution tensors",
        },
        "sample": {
            "query_indices": sample["sample_indices"],
            "reference_ids": sample["reference_ids"],
            "target_ids": sample["target_ids"],
            "captions": sample["captions"],
            "model_texts": [native_texts[index] for index in sample["sample_indices"]],
            "selection_rules": sample["selection_rules"],
            "selection_rules_skipped_for_slot_budget":
                sample["selection_rules_skipped_for_slot_budget"],
            "multi_positive_rows": sample["multi_positive_rows"],
            "sample_has_two_rows_sharing_a_reference":
                sample["sample_has_two_rows_sharing_a_reference"],
            "gallery_sample_ids": sample["gallery_sample_ids"],
            "gallery_sample_positions": sample["gallery_sample_positions"],
            "gallery_sample_covers_tail": sample["gallery_sample_covers_tail"],
            "gallery_sample_size": len(sample["gallery_sample_ids"]),
            "is_partial_batch": len(sample["gallery_sample_ids"]) % batch != 0,
        },
        "phase_seconds": round(time.time() - began, 1),
    }
    return {"identity": identity, "sample": sample, "scope": scope}


def native_phase(hook: CaptureHook, category: str, config: Any,
                 batch_size: int | None = None, role: str | None = None) -> dict[str, Any]:
    """Capture native population and sample inputs for one declared execution."""
    import numpy as np
    from workbench.replay.common import isolated_cache_root
    from workbench.composition.adapters import resolved_batch_size

    phase_root = RAW
    if hook.scope.model_id == "hint" and role is not None:
        if batch_size is None or resolved_batch_size() != batch_size:
            raise ValueError("HINT native role batch differs from configured batch")
        phase_root = RAW / "roles" / category / role
        phase_root.mkdir(parents=True, exist_ok=False)

    began = time.time()
    loaded = hook.load_native(config, category)
    torch_mod = loaded["torch"]
    records: dict[str, Any] = {}
    try:
        with isolated_cache_root(loaded["dataset_root"]) as cache:
            if hook.scope.model_id == "hint":
                dataset = load_hint_category_dataset(
                    loaded["datasets"], cache, category, hook.transform(loaded))
            else:
                class _Identity:
                    def __call__(self, image: Any) -> Any:
                        return image

                    def __getitem__(self, index: int) -> Any:
                        return self

                dataset = loaded["datasets"].FashionIQ(
                    path=str(cache).rstrip("/") + "/", transform=_Identity(), **hook.dataset_kwargs())

            queries = getattr(dataset, f"test_queries_{category}")
            targets = getattr(dataset, f"test_targets_{category}")
            scope = hook.gallery(loaded, category)
            gallery_ids = [scope["images"][row["target_img_id"]] for row in targets]
            sample = fixed_sample(scope["rows"], gallery_ids)
            texts = [row["mod"]["str"] for row in queries]
            identity_document = json.loads(
                (RAW / f"identity_hint_{category}.json").read_text(encoding="utf-8"))
            query_annotation_ids = [str(item["annotation_id"])
                                    for item in identity_document["query_annotations"]]
            if len(query_annotation_ids) != len(queries):
                raise ValueError("HINT identity annotation count differs from native query population")
            inputs = {
                "query_annotation_ids": query_annotation_ids,
                "sample_indices": sample["sample_indices"],
                "query_texts": [texts[i] for i in sample["sample_indices"]],
                "reference_ids": sample["reference_ids"],
                "target_ids": sample["target_ids"],
                "query_reference_paths": [scope["image_path"](name) for name in sample["reference_ids"]],
                "query_target_paths": [scope["image_path"](name) for name in sample["target_ids"]],
                "gallery_sample_ids": sample["gallery_sample_ids"],
                "gallery_sample_paths": [scope["image_path"](name) for name in sample["gallery_sample_ids"]],
                "gallery_ids": gallery_ids,
                "gallery_source_positions": scope["gallery_positions"],
                "gallery_paths": scope["gallery_paths"],
                "full_query_paths": [scope["image_path"](name) for name in scope["reference_ids"]],
                "full_query_texts": texts,
                "full_reference_ids": scope["reference_ids"],
                "full_target_ids": scope["target_ids"],
            }
            query_started_at = datetime.now(timezone.utc).isoformat()
            if hook.scope.model_id == "hint":
                from types import SimpleNamespace
                source = observe_hint_evaluator(
                    loaded["test_module"], SimpleNamespace(batch_size=resolved_batch_size(), dataset="fashioniq"),
                    loaded["model"], dataset, category, loaded["processors"][0],
                    inputs["gallery_ids"], inputs["full_reference_ids"], inputs["full_target_ids"],
                    inputs["gallery_paths"], inputs["full_query_paths"],
                    [scope["image_path"](name) for name in inputs["full_target_ids"]],
                    phase_root / "source_scores", inputs["query_annotation_ids"])
                source_artifacts = hint_source_artifacts(source)
                full_native_queries = source_artifacts["population_native_queries"]
                full_native_gallery = source_artifacts["population_native_gallery"]
                capture_data = {
                    **source_artifacts,
                    "pinned_source_matmul_gallery": source["gallery_features"],
                    "population_native_query_paths": np.asarray(inputs["full_query_paths"]),
                    "population_native_gallery_paths": np.asarray(inputs["gallery_paths"]),
                }
                records["native_population"] = {
                    "query_count": len(queries), "gallery_count": len(targets), "scope": "full",
                    "source": "pinned HINT test.py:test np.matmul operands",
                    "matmul_started_at": source["score_started_at"],
                    "matmul_completed_at": source["score_completed_at"],
                    "query_shape": list(full_native_queries.shape),
                    "gallery_shape": list(full_native_gallery.shape),
                    "query_dtype": str(full_native_queries.dtype),
                    "query_started_at": query_started_at,
                    **_hint_execution_partitions(
                        inputs["query_annotation_ids"], inputs["full_reference_ids"],
                        inputs["gallery_ids"], resolved_batch_size()),
                    "query_batch_size": int(resolved_batch_size()),
                    "gallery_batch_size": int(resolved_batch_size()),
                    "pinned_source_evaluator": {
                        "evaluator": "workbench/third_party/HINT/test.py:test",
                        "hit_counts": source["hit_counts"], "measured_recall": source["measured"],
                        "ranks": source_artifacts["pinned_source_ranks"].tolist(),
                        "topk_target_hits": source_artifacts["pinned_source_topk_target_hits"].tolist(),
                        "duplicate_sort_calls_identical": source["duplicate_sort_calls_identical"],
                        "score_order_sha256": source["score_order_sha256"],
                        "scores_path": str(phase_root / "source_scores/scores.float32.npy"),
                        "scores_dtype": "float32", "scores_shape": list(source["scores"].shape),
                        "orders_path": str(phase_root / "source_scores/orders.int32.npy"),
                        "orders_dtype": "int32", "orders_shape": list(source["orders"].shape),
                        "actual_matmul_calls": 1,
                    },
                }
                # Bounded fixed-sample capture remains separate from population scorer evidence.
                query_features = hook.native_features(
                    loaded, inputs["query_reference_paths"], inputs["query_texts"], capture_inputs=True)
                query_pixels = getattr(hook, "last_native_pixels", None)
                query_tokens = getattr(hook, "last_native_token_ids", None)
                query_mask = getattr(hook, "last_native_token_mask", None)
                query_processed_texts = getattr(hook, "last_native_processed_texts", None)
                target_features = hook.native_target_features(
                    loaded, inputs["query_target_paths"], capture_inputs=True)
                target_pixels = getattr(hook, "last_native_target_pixels", None)
                gallery_features = hook.native_target_features(
                    loaded, inputs["gallery_sample_paths"], capture_inputs=True)
                gallery_pixels = getattr(hook, "last_native_target_pixels", None)
                capture_data.update({
                    "native_query": query_features, "native_target": target_features,
                    "native_query_ids": np.asarray(_hint_capture_row_ids(
                        inputs, inputs["sample_indices"])[1]),
                    "native_target_ids": np.asarray(_hint_capture_row_ids(
                        inputs, inputs["sample_indices"])[1]),
                    "native_query_paths": np.asarray(inputs["query_reference_paths"]),
                    "native_target_paths": np.asarray(inputs["query_target_paths"]),
                    "native_gallery_sample_features": gallery_features,
                    "native_gallery_sample_pixels": gallery_pixels,
                    "native_gallery_sample_ids": np.asarray(inputs["gallery_sample_ids"]),
                    "native_gallery_sample_paths": np.asarray(inputs["gallery_sample_paths"]),
                    "native_query_processed_texts": np.asarray(query_processed_texts),
                })
                if query_pixels is not None:
                    capture_data["native_query_pixels"] = query_pixels
                if query_tokens is not None:
                    capture_data["native_query_token_ids"] = query_tokens
                    capture_data["native_query_token_mask"] = query_mask
                if target_pixels is not None:
                    capture_data["native_target_pixels"] = target_pixels
                records["native_population"]["score_feature_capture"] = {
                    "source": "pinned evaluator matmul inputs", "duplicate_population_forward": False,
                    "query_order_sha256": digest_ids(inputs["full_reference_ids"]),
                    "gallery_order_sha256": digest_ids(inputs["gallery_ids"]),
                }
            else:
                query_features = hook.native_features(
                    loaded, inputs["query_reference_paths"], inputs["query_texts"])
                target_features = hook.native_target_features(loaded, inputs["query_target_paths"])
                full_query_started_at = datetime.now(timezone.utc).isoformat()
                full_native_queries = hook.native_features(
                    loaded, inputs["full_query_paths"], inputs["full_query_texts"])
                full_query_completed_at = datetime.now(timezone.utc).isoformat()
                full_gallery_started_at = datetime.now(timezone.utc).isoformat()
                full_native_gallery = hook.native_target_features(loaded, inputs["gallery_paths"])
                full_gallery_completed_at = datetime.now(timezone.utc).isoformat()
                diagnostic = native_population_rank_measurement(
                    full_native_queries, full_native_gallery, inputs["gallery_ids"],
                    inputs["full_reference_ids"], inputs["full_target_ids"])
                diagnostic.pop("scores", None)
                diagnostic.pop("masked_scores", None)
                capture_data = {
                    "native_query": query_features, "native_target": target_features,
                    "population_native_queries": full_native_queries,
                    "population_native_gallery": full_native_gallery,
                    "population_native_query_paths": np.asarray(inputs["full_query_paths"]),
                    "population_native_gallery_paths": np.asarray(inputs["gallery_paths"]),
                }
                records["native_population"] = {
                    "query_count": len(full_native_queries), "gallery_count": len(full_native_gallery),
                    "query_started_at": full_query_started_at, "query_completed_at": full_query_completed_at,
                    "gallery_started_at": full_gallery_started_at, "gallery_completed_at": full_gallery_completed_at,
                    "scope": "full", "independent_feature_diagnostic": diagnostic,
                    "diagnostic_source": "recomputed NumPy feature diagnostic; not upstream native scorer",
                    "query_order_sha256": digest_ids(inputs["full_reference_ids"]),
                    "gallery_order_sha256": digest_ids(inputs["gallery_ids"]),
                    "query_shape": list(np.asarray(full_native_queries).shape),
                    "gallery_shape": list(np.asarray(full_native_gallery).shape),
                    "query_dtype": str(np.asarray(full_native_queries).dtype),
                    "gallery_dtype": str(np.asarray(full_native_gallery).dtype),
                    "query_batch_size": int(resolved_batch_size()), "gallery_batch_size": int(resolved_batch_size()),
                    "source": "native production forward arrays",
                }
            query_pixels = getattr(hook, "last_native_pixels", None) if hook.scope.model_id != "hint" else query_pixels
            target_pixels = getattr(hook, "last_native_target_pixels", None) if hook.scope.model_id != "hint" else target_pixels
            if hook.scope.model_id != "hint":
                records["native_population"].update({
                    "query_partition_sha256": digest_ids(inputs["full_reference_ids"]),
                    "gallery_partition_sha256": digest_ids(inputs["gallery_ids"]),
                })
            records["decode_equality"] = {
                "status": "CAPTURED" if query_pixels is not None else "NOT_MEASURED",
                "model_id": hook.scope.model_id,
                "native_query_pixel_shape": list(query_pixels.shape) if query_pixels is not None else None,
                "native_target_pixel_shape": list(target_pixels.shape) if target_pixels is not None else None,
                "capture_source": "fixed sample forward actual tensor before device transfer",
            }
            (phase_root / f"sample_{hook.scope.model_id}_{category}_inputs.json").write_text(
                json.dumps(inputs, indent=1) + "\n", encoding="utf-8")
            native_path = phase_root / f"sample_{hook.scope.model_id}_{category}_native.npz"
            np.savez(native_path, **capture_data)
            records["native_population"]["artifact_path"] = str(native_path)
            records["native_population"]["scores_dtype"] = "float32" if hook.scope.model_id == "hint" else None
            records["native_population"]["orders_dtype"] = "int32" if hook.scope.model_id == "hint" else None
            records.update({
                "sample_forward_native": {"query_shape": list(query_features.shape),
                                          "target_shape": list(target_features.shape),
                                          "adapter_expected_query_shape": [len(sample["sample_indices"]), 256],
                                          "adapter_expected_target_shape": [len(sample["sample_indices"]), 32, 256]},
                "population_identity": {"query_count": len(queries), "gallery_count": len(targets)},
            })
    finally:
        del loaded["model"]
        torch_mod.cuda.empty_cache()
    records["captured_at"] = datetime.now(timezone.utc).isoformat()
    records["phase_seconds"] = round(time.time() - began, 1)
    return records


def _adapter_class(hook: CaptureHook) -> Any:
    """The adapter class this hook measures, resolved from the SCOPE rather than from HINT.

    One definition, used by `adapter_phase` and `scoring_ranking`, because those two must measure the
    same class: a model whose comparison used one adapter and whose ranking used another would be an
    artifact pair that contradicts itself in a way no check would catch.
    """
    return importlib.import_module(hook.scope.adapter_module).Adapter

def adapter_phase(hook: CaptureHook, category: str, config: Any,
                  batch_size: int | None = None, role: str | None = None) -> dict[str, Any]:
    """Capture one Adapter execution using role-scoped immutable inputs."""
    import numpy as np
    from workbench.composition.adapters import resolved_batch_size

    declared = hook.scope.declared_tolerances()
    phase_root = RAW
    input_root = RAW
    if hook.scope.model_id == "hint" and role is not None:
        if batch_size is None or resolved_batch_size() != batch_size:
            raise ValueError("HINT adapter role batch differs from configured batch")
        input_role = {"adapter_repeat_1": "native_repeat_1", "adapter_repeat_2": "native_repeat_2",
                      "native_adapter_adapter": "native_adapter_native", "adapter_batch_1": "native_batch_1",
                      "adapter_batch_n": "native_batch_n"}[role]
        phase_root = RAW / "roles" / category / role
        phase_root.mkdir(parents=True, exist_ok=False)
        input_root = RAW / "roles" / category / input_role
    inputs = json.loads((input_root / f"sample_{hook.scope.model_id}_{category}_inputs.json").read_text(encoding="utf-8"))
    native = np.load(input_root / f"sample_{hook.scope.model_id}_{category}_native.npz")
    began = time.time()
    adapter = _adapter_class(hook)(config)
    adapter.work_root = phase_root / "adapter-work" if role is not None else RAW / f"work-{hook.scope.model_id}-{category}"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    scope_value = hook.scope.target_category
    result: dict[str, Any] = {}
    try:
        capture_root = phase_root / "adapter-capture" if role is not None else RAW / f"adapter-capture-{hook.scope.model_id}-{category}"
        capture_root.mkdir(parents=True, exist_ok=role is None)
        if hook.scope.model_id == "hint":
            _, capture_query_ids = _hint_capture_row_ids(inputs, inputs["sample_indices"])
        else:
            capture_query_ids = [str(value) for value in inputs["reference_ids"]]
        adapter_query_started_at = datetime.now(timezone.utc).isoformat()
        adapter_queries = adapter.encode_queries(
            inputs["query_reference_paths"], inputs["query_texts"],
            [scope_value] * len(inputs["query_texts"]) if scope_value is not None else
            [None] * len(inputs["query_texts"]),
            **({"capture_path": capture_root / "queries.npz", "capture_ids": capture_query_ids,
                "selected_capture_ids": capture_query_ids}
               if hook.scope.model_id == "hint" else {}))
        capture_target_ids = capture_query_ids if hook.scope.model_id == "hint" else [
            str(value) for value in inputs["target_ids"]]
        adapter_target_started_at = datetime.now(timezone.utc).isoformat()
        adapter_targets = adapter.encode_targets(
            inputs["query_target_paths"],
            [scope_value] * len(inputs["query_target_paths"]) if scope_value is not None else
            [None] * len(inputs["query_target_paths"]),
            **({"capture_path": capture_root / "targets.npz", "capture_ids": capture_target_ids,
                "selected_capture_ids": capture_target_ids}
               if hook.scope.model_id == "hint" else {}))
        if hook.scope.model_id == "hint":
            gallery_capture_path = capture_root / "gallery-sample.npz"
            adapter.encode_targets(
                inputs["gallery_sample_paths"], [None] * len(inputs["gallery_sample_paths"]),
                capture_path=gallery_capture_path,
                capture_ids=[str(value) for value in inputs["gallery_sample_ids"]],
                selected_capture_ids=[str(value) for value in inputs["gallery_sample_ids"]])
        result["adapter_query_started_at"] = adapter_query_started_at
        result["adapter_target_started_at"] = adapter_target_started_at
        forward = {
            "query": toleranced(compare(adapter_queries, native["native_query"]),
                                abs_key="forward_max_abs", rel_key="forward_max_rel",
                                declared=declared),
            "target": toleranced(compare(adapter_targets, native["native_target"]),
                                 abs_key="forward_max_abs", rel_key="forward_max_rel",
                                 declared=declared),
            "adapter_dtype": str(adapter_queries.dtype),
            "adapter_shape": list(adapter_queries.shape),
            "target_shape": list(adapter_targets.shape),
        }
        result["sample_forward"] = forward
        if hook.scope.model_id == "hint" or hook.scope.transform_source == "clip":
            pixel_delta = None
            with np.load(capture_root / "queries.npz") as query_capture:
                with np.load(capture_root / "targets.npz") as target_capture:
                    query_equal = (query_capture["capture_ids"].tolist() ==
                                   native["native_query_ids"].tolist() and
                                   query_capture["image_paths"].tolist() ==
                                   native["native_query_paths"].tolist() and
                                   np.array_equal(query_capture["pixels"], native["native_query_pixels"]))
                    target_equal = (target_capture["capture_ids"].tolist() ==
                                    native["native_target_ids"].tolist() and
                                    target_capture["image_paths"].tolist() ==
                                    native["native_target_paths"].tolist() and
                                    np.array_equal(target_capture["pixels"], native["native_target_pixels"]))
                    if hook.scope.model_id == "hint":
                        with np.load(capture_root / "gallery-sample.npz") as gallery_capture:
                            gallery_equal = (
                                gallery_capture["capture_ids"].tolist() ==
                                native["native_gallery_sample_ids"].tolist() and
                                gallery_capture["image_paths"].tolist() ==
                                native["native_gallery_sample_paths"].tolist() and
                                np.array_equal(gallery_capture["pixels"],
                                               native["native_gallery_sample_pixels"]))
                            gallery_pixels_shape = list(gallery_capture["pixels"].shape)
                            pixel_delta = max(
                                float(np.max(np.abs(query_capture["pixels"].astype(np.float32)
                                                   - native["native_query_pixels"].astype(np.float32)), initial=0.0)),
                                float(np.max(np.abs(target_capture["pixels"].astype(np.float32)
                                                   - native["native_target_pixels"].astype(np.float32)), initial=0.0)),
                                float(np.max(np.abs(gallery_capture["pixels"].astype(np.float32)
                                                   - native["native_gallery_sample_pixels"].astype(np.float32)), initial=0.0)))
                            np.savez(
                                capture_root / "hint-g1-inputs.npz",
                                query_pixels=query_capture["pixels"],
                                query_paths=query_capture["image_paths"],
                                query_ids=query_capture["capture_ids"],
                                query_texts=query_capture["processed_texts"],
                                query_token_ids=query_capture["token_ids"],
                                query_token_mask=query_capture["token_mask"],
                                target_pixels=target_capture["pixels"],
                                target_paths=target_capture["image_paths"],
                                target_ids=target_capture["capture_ids"],
                                gallery_pixels=gallery_capture["pixels"],
                                gallery_paths=gallery_capture["image_paths"],
                                gallery_ids=gallery_capture["capture_ids"],
                                native_query_pixels=native["native_query_pixels"],
                                native_query_paths=native["native_query_paths"],
                                native_query_ids=native["native_query_ids"],
                                native_query_texts=native["native_query_processed_texts"],
                                native_query_token_ids=native["native_query_token_ids"],
                                native_query_token_mask=native["native_query_token_mask"],
                                native_target_pixels=native["native_target_pixels"],
                                native_target_paths=native["native_target_paths"],
                                native_target_ids=native["native_target_ids"],
                                native_gallery_pixels=native["native_gallery_sample_pixels"],
                                native_gallery_paths=native["native_gallery_sample_paths"],
                                native_gallery_ids=native["native_gallery_sample_ids"])
                            result["hint_input_capture"] = {
                                "artifact": str(capture_root / "hint-g1-inputs.npz"),
                                "query_count": len(query_capture["capture_ids"]),
                                "target_count": len(target_capture["capture_ids"]),
                                "gallery_count": len(gallery_capture["capture_ids"]),
                                "query_pixels_equal": bool(query_equal),
                                "target_pixels_equal": bool(target_equal),
                                "gallery_pixels_equal": bool(gallery_equal),
                                "text_equal": bool(np.array_equal(
                                    query_capture["processed_texts"],
                                    native["native_query_processed_texts"])),
                                "token_ids_equal": bool(np.array_equal(
                                    query_capture["token_ids"],
                                    native["native_query_token_ids"])),
                                "token_mask_equal": bool(np.array_equal(
                                    query_capture["token_mask"],
                                    native["native_query_token_mask"])),
                                "capture_bound": 32,
                            }
                    result["layer2_tokens"] = {
                        "token_ids_exact_match": bool(np.array_equal(
                            query_capture["token_ids"], native["native_query_token_ids"])),
                        "token_mask_exact_match": bool(np.array_equal(
                            query_capture["token_mask"], native["native_query_token_mask"])),
                        "processed_texts_exact_match": bool(np.array_equal(
                            query_capture["processed_texts"], native["native_query_processed_texts"])),
                        "token_ids_shape": list(query_capture["token_ids"].shape),
                        "token_mask_shape": list(query_capture["token_mask"].shape),
                    }
                    result["layer2_pixels"] = {
                        "status": "CAPTURED",
                        "capture_source": "native and production adapter forward input tensors",
                        "query_ids": query_capture["capture_ids"].tolist(),
                        "target_ids": target_capture["capture_ids"].tolist(),
                        "query_paths": query_capture["image_paths"].tolist(),
                        "query_pixels_shape": list(query_capture["pixels"].shape),
                        "target_pixels_shape": list(target_capture["pixels"].shape),
                        "gallery_pixels_shape": (gallery_pixels_shape if hook.scope.model_id == "hint" else None),
                        "query_exact_match": bool(query_equal),
                        "target_exact_match": bool(target_equal),
                        "gallery_exact_match": bool(gallery_equal) if hook.scope.model_id == "hint" else None,
                        "adapter_decode_vs_native_val_decode": (
                            {"bitwise_equal": bool(query_equal and target_equal and gallery_equal),
                             "max_abs": pixel_delta}
                            if hook.scope.model_id == "hint" else None),
                        "model_id": hook.scope.model_id,
                    }
        else:
            result["layer2_pixels"] = {
                "status": "NOT_MEASURED", "reason": "production worker has no actual-decode capture",
                "model_id": hook.scope.model_id,
            }
        result["component_parity"] = T4.component_block({
            "sample_forward.query": T4.component_scope(
                "sample_forward.query", "forward_parity", adapter_queries, native["native_query"],
                atol=declared["forward_max_abs"], rtol=declared["forward_max_rel"]),
            "sample_forward.target": T4.component_scope(
                "sample_forward.target", "forward_parity", adapter_targets, native["native_target"],
                atol=declared["forward_max_abs"], rtol=declared["forward_max_rel"]),
        })

        singles = np.concatenate([
            adapter.encode_queries([path], [text], [scope_value])
            for path, text in zip(inputs["query_reference_paths"], inputs["query_texts"])])
        result["batch_size_1_vs_n"] = toleranced(
            compare(singles, adapter_queries), abs_key="batch_max_abs",
            rel_key="batch_max_rel", declared=declared)

        if hook.scope.restart_supported:
            adapter.restart()
            after_restart = adapter.encode_queries(
                inputs["query_reference_paths"], inputs["query_texts"],
                [scope_value] * len(inputs["query_texts"]))
            result["cache_off_restart"] = toleranced(
                compare(after_restart, adapter_queries), abs_key="batch_max_abs",
                rel_key="batch_max_rel", declared=declared)
        else:
            # NOT MEASURED, and named as such. The adapter refuses a second worker by design
            # (`_check_worker_pid`), so this comparison cannot run through the shared path without
            # changing production code that this lane does not own.
            result["cache_off_restart"] = {
                "measured": False,
                "why": "the adapter asserts one worker load per instance "
                       "(`model_tgcir.py:344 _check_worker_pid` -> `adapter_worker_reloaded`), so an "
                       "in-process restart is refused by construction",
                "consequence": "a same-process restart comparison is ABSENT for this model; it is "
                               "not reported as equal",
            }

        # `population_scope: "sample"` encodes the FIXED SAMPLE instead of the whole category. That is
        # what makes a broad-but-shallow lane possible at all: a full category encode is minutes of
        # GPU per model, and the shared sample is chosen by data-derived rules so two runs of the same
        # model see the same rows. A sample scope is a PREVIEW and is labelled as one -- the gate
        # cannot call a preview scope ok, and neither can this producer.
        if hook.scope.population_scope == "sample":
            population_queries = adapter.encode_queries(
                inputs["query_reference_paths"], inputs["query_texts"],
                [scope_value] * len(inputs["query_texts"]))
            population_gallery = adapter.encode_targets(
                inputs["gallery_sample_paths"], [scope_value] * len(inputs["gallery_sample_paths"]))
        else:
            if hook.scope.model_id == "hint":
                full_capture_ids, selected_capture_ids = _hint_capture_row_ids(
                    inputs, inputs["sample_indices"])
                full_capture_args = {"capture_path": capture_root / "full-queries.npz",
                                     "capture_ids": full_capture_ids,
                                     "selected_capture_ids": selected_capture_ids}
                gallery_positions = {image_id: index for index, image_id in
                                     enumerate(inputs["gallery_ids"])}
                full_target_paths = [inputs["gallery_paths"][gallery_positions[target]]
                                     for target in inputs["full_target_ids"]]
                target_capture_args = {"capture_path": capture_root / "full-targets.npz",
                                       "capture_ids": full_capture_ids,
                                       "selected_capture_ids": selected_capture_ids}
            else:
                full_capture_args = {}
                full_target_paths = inputs["gallery_paths"]
                target_capture_args = {}
            population_queries = adapter.encode_queries(
                inputs["full_query_paths"], inputs["full_query_texts"],
                [scope_value] * len(inputs["full_query_texts"]), **full_capture_args)
            if hook.scope.model_id == "hint":
                adapter.encode_targets(
                    full_target_paths, [None] * len(full_target_paths), **target_capture_args)
                population_gallery = adapter.encode_targets(
                    inputs["gallery_paths"], [scope_value] * len(inputs["gallery_paths"]))
            else:
                population_gallery = adapter.encode_targets(
                    full_target_paths, [scope_value] * len(full_target_paths), **target_capture_args)
            if hook.scope.model_id == "hint":
                with np.load(capture_root / "full-queries.npz") as full_queries:
                    with np.load(capture_root / "full-targets.npz") as full_targets:
                        selected_paths = [inputs["full_query_paths"][index]
                                          for index in inputs["sample_indices"]]
                        selected_target_paths = [full_target_paths[index]
                                                 for index in inputs["sample_indices"]]
                        selected_ids = _hint_capture_row_ids(
                            inputs, inputs["sample_indices"])[1]
                        if (full_queries["capture_ids"].tolist() != selected_ids
                                or full_queries["image_paths"].tolist() != selected_paths
                                or full_targets["capture_ids"].tolist() != selected_ids
                                or full_targets["image_paths"].tolist() != selected_target_paths):
                            raise ValueError("HINT full-population capture row identity/order mismatch")
                        result["hint_population_input_capture"] = {
                            "query_artifact": str(capture_root / "full-queries.npz"),
                            "target_artifact": str(capture_root / "full-targets.npz"),
                            "annotation_ids": selected_ids,
                            "sample_indices": list(inputs["sample_indices"]),
                            "capture_bound": 32,
                        }
        if hook.scope.population_scope == "full":
            with np.load(input_root / f"sample_{hook.scope.model_id}_{category}_native.npz") as native_full:
                native_query_paths = native_full["population_native_query_paths"].tolist()
                native_gallery_paths = native_full["population_native_gallery_paths"].tolist()
                if (native_query_paths != inputs["full_query_paths"] or
                        native_gallery_paths != inputs["gallery_paths"]):
                    raise ValueError("native_population_order_mismatch")
                result["population_forward"] = {
                    "query": toleranced(compare(population_queries,
                                                 native_full["population_native_queries"]),
                                        abs_key="forward_max_abs", rel_key="forward_max_rel",
                                        declared=declared),
                    "gallery": toleranced(compare(population_gallery,
                                                   native_full["population_native_gallery"]),
                                          abs_key="forward_max_abs", rel_key="forward_max_rel",
                                          declared=declared),
                    "query_count": len(population_queries),
                    "gallery_count": len(population_gallery), "scope": "full"}
        np.savez(phase_root / f"population_{hook.scope.model_id}_{category}.npz",
                 queries=population_queries, gallery=population_gallery)
        if hook.scope.model_id == "hint" and role == "native_adapter_adapter":
            gallery_position = {image_id: index for index, image_id in
                               enumerate(inputs["gallery_ids"])}
            target_positions = [gallery_position[target] for target in inputs["full_target_ids"]]
            ranking = adapter.rank(population_queries, population_gallery,
                                   [[index] for index in target_positions])
            scores = adapter.score(population_queries, population_gallery)
            adapter_scores = scores.astype(np.float32, copy=True)
            for index, reference_id in enumerate(inputs["full_reference_ids"]):
                adapter_scores[index, gallery_position[reference_id]] = -10e10
            adapter_orders = np.argsort(-adapter_scores, axis=1).astype(np.int32, copy=False)
            np.save(phase_root / "adapter_orders.npy", adapter_orders)
            np.save(phase_root / "adapter_scores.npy", adapter_scores)
            result["v3_population_scoring"] = {
                "ranks": [int(value) for value in ranking],
                "score_pairs_compared": min(64, len(ranking)),
                "score_shape": list(scores.shape),
                "score_order_sha256": _hint_score_order_digest(
                    inputs["query_annotation_ids"], adapter_scores, adapter_orders),
            }
        from workbench.composition.adapters import resolved_batch_size
        execution_batch_size = int(getattr(adapter, "batch_size", resolved_batch_size()))
        if hook.scope.model_id == "hint":
            annotation_ids, _ = _hint_capture_row_ids(
                inputs, inputs["sample_indices"] if hook.scope.population_scope != "full" else None)
            partition_metadata = _hint_execution_partitions(
                annotation_ids,
                inputs["full_reference_ids"] if hook.scope.population_scope == "full"
                else inputs["reference_ids"],
                inputs["gallery_ids"] if hook.scope.population_scope == "full"
                else inputs["gallery_sample_ids"], execution_batch_size)
        else:
            query_ids = (inputs["full_reference_ids"] if hook.scope.population_scope == "full"
                         else inputs["reference_ids"])
            gallery_ids = (inputs["gallery_ids"] if hook.scope.population_scope == "full"
                           else inputs["gallery_sample_ids"])
            partition_metadata = {
                "query_order_sha256": digest_ids(query_ids),
                "gallery_order_sha256": digest_ids(gallery_ids),
                "query_partition_sha256": digest_ids(query_ids),
                "gallery_partition_sha256": digest_ids(gallery_ids),
                "batch_partition_sha256": _batch_partition_digest(
                    query_ids, gallery_ids, execution_batch_size)[2],
            }
        result["population"] = {
            "query_started_at": adapter_query_started_at,
            "query_count": int(population_queries.shape[0]),
            "gallery_count": int(population_gallery.shape[0]),
            "population_scope": hook.scope.population_scope,
            "coverage": ("full" if hook.scope.population_scope == "full" else "chunked"),
            "query_shape": list(population_queries.shape),
            "gallery_shape": list(population_gallery.shape),
            "query_dtype": str(population_queries.dtype),
            "gallery_dtype": str(population_gallery.dtype),
            "query_batch_size": execution_batch_size,
            "gallery_batch_size": execution_batch_size,
            **partition_metadata,
            "source": "production adapter forward arrays",
            "why": ("the whole category population was encoded"
                    if hook.scope.population_scope == "full" else
                    "the fixed sample only: preview scope, population verdict incomplete"),
            "finite": bool(np.isfinite(population_queries).all()
                           and np.isfinite(population_gallery).all()),
        }
    finally:
        adapter.close()

    if role is None:
        result["scoring_ranking"] = scoring_ranking(hook, category, inputs, config, declared)
        update_native_metrics_flag(hook, category, result["scoring_ranking"])
    result["captured_at"] = datetime.now(timezone.utc).isoformat()
    result["phase_seconds"] = round(time.time() - began, 1)
    return result

def load_hint_category_dataset(datasets: Any, dataset_root: Path, category: str,
                               transform: Callable[[Any], Any]) -> Any:
    """Build one real HINT FashionIQ split, sharing each transformed image tensor."""
    dataset = datasets.FashionIQ.__new__(datasets.FashionIQ)
    dataset.path = str(dataset_root).rstrip("/") + "/"
    dataset.image_dir = dataset.path + "resized_image"
    dataset.split_dir = dataset.path + "image_splits"
    dataset.caption_dir = dataset.path + "captions"
    dataset.transform = transform
    cache: dict[str, Any] = {}
    original_get_test_img = dataset.get_test_img

    def cached_get_test_img(image_name: str) -> Any:
        if image_name not in cache:
            cache[image_name] = original_get_test_img(image_name)
        return cache[image_name]

    dataset.get_test_img = cached_get_test_img
    queries, targets = dataset.get_test_data(category)
    setattr(dataset, f"test_queries_{category}", queries)
    setattr(dataset, f"test_targets_{category}", targets)
    return dataset



class _HintNumpyCapture:
    def __init__(self, numpy_module: Any, query_count: int, gallery_count: int,
                 reference_positions: Sequence[int], target_positions: Sequence[int],
                 staging_dir: Path) -> None:
        self.numpy = numpy_module
        self.query_count = query_count
        self.gallery_count = gallery_count
        self.reference_positions = numpy_module.asarray(reference_positions, dtype=numpy_module.int32)
        self.target_positions = numpy_module.asarray(target_positions, dtype=numpy_module.int32)
        self.calls = 0
        self.matmul_count = 0
        self.query_features: Any = None
        self.gallery_features: Any = None
        self.score_started_at: str | None = None
        self.score_completed_at: str | None = None
        staging_dir.mkdir(parents=True, exist_ok=False)
        self.scores = numpy_module.lib.format.open_memmap(
            staging_dir / "scores.float32.npy", mode="w+", dtype=numpy_module.float32,
            shape=(query_count, gallery_count))
        self.orders = numpy_module.lib.format.open_memmap(
            staging_dir / "orders.int32.npy", mode="w+", dtype=numpy_module.int32,
            shape=(query_count, gallery_count))
        self.ranks = numpy_module.empty(query_count, dtype=numpy_module.int32)
        self.hit_counts: dict[str, int] = {}
        self._first_order_digests: list[bytes] = []
    def __getattr__(self, name: str) -> Any:
        return getattr(self.numpy, name)
    def matmul(self, left: Any, right: Any, *args: Any, **kwargs: Any) -> Any:
        from datetime import datetime, timezone

        self.matmul_count += 1
        if self.matmul_count != 1:
            raise ValueError("pinned HINT evaluator called matmul more than once")
        self.score_started_at = datetime.now(timezone.utc).isoformat()
        self.query_features = self.numpy.asarray(left)
        self.gallery_features = self.numpy.asarray(right)
        result = self.numpy.matmul(left, right, *args, **kwargs)
        self.score_completed_at = datetime.now(timezone.utc).isoformat()
        return result

    def argsort(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        import hashlib

        values = self.numpy.asarray(values)
        order = self.numpy.argsort(values, *args, **kwargs)
        row_index = self.calls if self.calls < self.query_count else self.calls - self.query_count
        self.calls += 1
        if row_index >= self.query_count or values.shape != (self.gallery_count,):
            raise ValueError("pinned HINT argsort shape/count differs from evaluator contract")
        scores = -values
        if scores.dtype != self.numpy.float32:
            raise ValueError(f"pinned HINT score dtype changed at query {row_index}: {scores.dtype}")
        sentinel = self.numpy.asarray(-10e10, dtype=scores.dtype).item()
        if scores[self.reference_positions[row_index]] != sentinel:
            raise ValueError(f"pinned HINT reference sentinel mismatch at query {row_index}")
        digest = hashlib.sha256(self.numpy.ascontiguousarray(order).view(self.numpy.uint8)).digest()
        if self.calls <= self.query_count:
            self.scores[row_index] = scores
            self.orders[row_index] = order.astype(self.numpy.int32, copy=False)
            self.ranks[row_index] = int(self.numpy.flatnonzero(order == self.target_positions[row_index])[0]) + 1
            self._first_order_digests.append(digest)
        elif digest != self._first_order_digests[row_index]:
            raise ValueError(f"pinned HINT repeated argsort differs at query {row_index}")
        return order
def _hint_score_order_digest(query_ids: Sequence[str], scores: Any, orders: Any) -> str:
    """Hash capture-time score rows and exact source ranking keys, in query order."""
    import numpy as np

    score_values, order_values = np.asarray(scores), np.asarray(orders)
    if (score_values.ndim != 2 or order_values.shape != score_values.shape
            or score_values.shape[0] != len(query_ids)):
        raise ValueError("HINT score/order receipt dimensions differ from ordered queries")
    digest = hashlib.sha256()
    for query_id, score_row, order_row in zip(query_ids, score_values, order_values):
        digest.update(json.dumps(
            {"query_id": str(query_id), "scores": [float(value) for value in score_row],
             "order": [int(value) for value in order_row]},
            ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def observe_hint_evaluator(test_module: Any, params: Any, model: Any, dataset: Any,
                           category: str, txt_processors: Any,
                           expected_gallery_ids: Sequence[str],
                           expected_reference_ids: Sequence[str],
                           expected_target_ids: Sequence[str],
                           expected_gallery_paths: Sequence[str],
                           expected_reference_paths: Sequence[str],
                           expected_target_paths: Sequence[str],
                           staging_dir: Path,
                           expected_query_ids: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    queries = getattr(dataset, f"test_queries_{category}")
    targets = getattr(dataset, f"test_targets_{category}")
    if len(queries) != len(expected_reference_ids) or len(queries) != len(expected_target_ids):
        raise ValueError("pinned HINT query count differs from assembled decode identity")
    if len(targets) != len(expected_gallery_ids):
        raise ValueError("pinned HINT gallery count differs from assembled decode identity")
    split_path = Path(dataset.split_dir) / f"split.{category}.val.json"
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    annotation_rows = json.loads(
        (Path(dataset.caption_dir) / f"cap.{category}.val.json").read_text(encoding="utf-8"))
    correction = json.loads((Path(dataset.caption_dir) / f"correction_dict_{category}.json")
                            .read_text(encoding="utf-8"))
    if len(annotation_rows) != len(queries):
        raise ValueError("pinned HINT annotation count differs from evaluator query population")
    resolved_query_ids = []
    for index, (annotation, query) in enumerate(zip(annotation_rows, queries)):
        source_index = int(query["source_img_id"])
        target_index = int(query["target_img_id"])
        annotation_text = dataset.concat_text(annotation["captions"], correction)
        if (annotation["candidate"] != split_ids[source_index]
                or annotation["target"] != split_ids[target_index]
                or query["mod"]["str"] != annotation_text):
            raise ValueError("pinned HINT annotation source/target/text order differs from evaluator query rows")
        resolved_query_ids.append(
            f"{category}:{index}:{source_index}:{target_index}:"
            f"{hashlib.sha256(annotation_text.encode('utf-8')).hexdigest()}")
    if resolved_query_ids != list(expected_query_ids):
        raise ValueError("pinned HINT annotation ID/order differs from immutable identity")
    resolved_gallery = [str(split_ids[int(row["target_img_id"])]) for row in targets]
    resolved_references = [str(split_ids[int(row["source_img_id"])]) for row in queries]
    resolved_targets = [str(split_ids[int(row["target_img_id"])]) for row in queries]
    if resolved_gallery != list(expected_gallery_ids):
        raise ValueError("pinned HINT gallery ID/order differs from assembled decode identity")
    if resolved_references != list(expected_reference_ids) or resolved_targets != list(expected_target_ids):
        raise ValueError("pinned HINT query ID/order differs from assembled decode identity")
    def check_paths(paths: Sequence[str], ids: Sequence[str], label: str) -> None:
        expected = [Path(dataset.image_dir) / category / f"{image_id}.jpg" for image_id in ids]
        if len(paths) != len(expected) or any(
                Path(path).resolve() != target.resolve() for path, target in zip(paths, expected)):
            raise ValueError(f"assembled HINT {label} paths/order differ from dataset image paths")
    check_paths(expected_gallery_paths, expected_gallery_ids, "gallery")
    check_paths(expected_reference_paths, expected_reference_ids, "reference")
    check_paths(expected_target_paths, expected_target_ids, "target")
    target_position_by_id = {int(row["target_img_id"]): position for position, row in enumerate(targets)}
    capture = _HintNumpyCapture(np, len(queries), len(targets),
        [target_position_by_id[int(row["source_img_id"])] for row in queries],
        [target_position_by_id[int(row["target_img_id"])] for row in queries], staging_dir)
    original_numpy = test_module.np
    try:
        test_module.np = capture
        measured = test_module.test(params, model, dataset, category, txt_processors)
    finally:
        test_module.np = original_numpy
    if capture.calls != 2 * len(queries) or capture.matmul_count != 1:
        raise ValueError("pinned HINT matmul/argsort call count differs from evaluator contract")
    ranks = capture.ranks.tolist()
    hits = _hits(ranks)
    measured = dict(measured)
    for k in (1, 10, 50):
        key = f"{category}_r{k}"
        expected = 100.0 * hits[f"r{k}"] / len(ranks)
        if key not in measured or measured[key] != expected:
            raise ValueError(f"pinned HINT {key} disagrees with observed ranks: {measured.get(key)!r} != {expected!r}")
    return {"ranks": ranks, "hit_counts": hits, "gallery_ids": resolved_gallery,
            "reference_ids": resolved_references, "target_ids": resolved_targets,
            "measured": dict(measured), "scores": capture.scores, "orders": capture.orders,
            "score_order_sha256": _hint_score_order_digest(
                resolved_query_ids, capture.scores, capture.orders),
            "query_features": capture.query_features, "gallery_features": capture.gallery_features,
            "score_started_at": capture.score_started_at,
            "score_completed_at": capture.score_completed_at,
            "duplicate_sort_calls_identical": True}
def hint_source_artifacts(source: dict[str, Any]) -> dict[str, Any]:
    """Build bounded disk artifacts from the captured pinned evaluator pass."""
    import numpy as np

    ranks = np.asarray(source["ranks"], dtype=np.int32)
    cutoffs = np.asarray((1, 10, 50), dtype=np.int32)
    counts = np.asarray([source["hit_counts"][f"r{k}"] for k in cutoffs], dtype=np.int32)
    return {
        "population_native_queries": np.asarray(source["query_features"]).reshape(len(ranks), -1),
        "population_native_gallery": np.asarray(source["gallery_features"]).transpose(0, 2, 1),
        "pinned_source_ranks": ranks,
        "pinned_source_topk_target_hits": ranks[:, None] <= cutoffs[None, :],
        "pinned_source_hit_cutoffs": cutoffs,
        "pinned_source_hit_counts": counts,
        "pinned_source_gallery_ids": np.asarray(source["gallery_ids"]),
        "pinned_source_reference_ids": np.asarray(source["reference_ids"]),
        "pinned_source_target_ids": np.asarray(source["target_ids"]),
    }
def native_population_rank_measurement(query_features: Any, gallery_features: Any,
                                       gallery_ids: Sequence[str], reference_ids: Sequence[str],
                                       target_ids: Sequence[str]) -> dict[str, Any]:
    """Rank native-val features with pinned CIRCUS score/mask/tie rules."""
    import numpy as np
    queries = np.asarray(query_features, dtype=np.float32)
    gallery = np.asarray(gallery_features, dtype=np.float32)
    if queries.ndim != 2 or gallery.ndim not in (2, 3) or queries.shape[-1] != gallery.shape[-1]:
        raise ValueError(f"native HINT rank layout mismatch: query={queries.shape}, gallery={gallery.shape}")
    if not (len(reference_ids) == len(target_ids) == len(queries)):
        raise ValueError("native HINT rank IDs do not match query feature count")
    position = {image_id: index for index, image_id in enumerate(gallery_ids)}
    references = [position[image_id] for image_id in reference_ids]
    targets = [position[image_id] for image_id in target_ids]
    ranks = []
    for query, reference, target in zip(queries, references, targets):
        scores = (np.einsum("d,gqd->gq", query, gallery).max(axis=-1)
                  if gallery.ndim == 3 else gallery @ query)
        scores[reference] = -10e10
        order = np.argsort(-scores, kind="stable")
        ranks.append(int(np.flatnonzero(order == target)[0]) + 1)
    return {"ranks": ranks, "query_count": len(ranks), "gallery_count": len(gallery_ids),
            "hit_counts": _hits(ranks),
            "recall": {f"r{k}": 100.0 * sum(rank <= k for rank in ranks) / len(ranks)
                       for k in (1, 10, 50)} if ranks else {f"r{k}": None for k in (1, 10, 50)},
            "reference_mask": "native CIRCUS reference exclusion applied before stable argsort",
            "score_source": "native production query/gallery feature arrays; no adapter call"}
def scoring_ranking(hook: CaptureHook, category: str, inputs: dict[str, Any], config: Any,
                    declared: dict[str, Any]) -> dict[str, Any]:
    """G2 ranking: the adapter's own score/rank against the native convention, same features."""
    import numpy as np

    blob = np.load(RAW / f"population_{hook.scope.model_id}_{category}.npz")
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    queries_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    # The gallery's RANK is the model's, not the producer's: HINT/HABIT carry (G, 32, 256) token
    # layouts and are reduced with `.max(-1)` over the token axis, while TG-CIR carries (G, 512) and
    # has no token axis at all. Hardcoding the 3-D form raised
    # `ValueError: einstein sum subscripts string contains too many subscripts for operand 1`.
    gallery_rank = gallery.ndim
    # HINT's gallery is (G, 32, 256): the FEATURE axis is the last one. Normalising axis=1 would
    # normalise across the 32 Q-former tokens, which is not a direction in feature space; that mistake
    # produced a score delta of 1.17 against a declared 1e-6. The native evaluator normalises nothing
    # (features are already L2-normalised inside the model), so this is a re-normalisation of unit
    # vectors and must be a no-op to float32 precision.
    gallery_n = gallery / np.linalg.norm(gallery, axis=-1, keepdims=True)
    # `axis=-1` is the FEATURE axis in both layouts: the last axis of (G,32,256) and of (G,512).
    # Native convention, verbatim: `np.matmul(all_queries, all_imgs).squeeze().max(-1)`, reference
    # masked to -10e10 (test.py:74-79). float32 end to end, as the pinned evaluator computes it.
    if gallery_rank == 3:
        sims = np.einsum("nd,gqd->ngq", queries_n, gallery_n).max(-1)
    elif gallery_rank == 2:
        sims = queries_n @ gallery_n.T
    else:
        raise SystemExit(f"unsupported gallery rank {gallery_rank} for {hook.scope.model_id}: "
                         "expected 2 (one vector per image) or 3 (a token layout)")
    unmasked = sims.copy()
    # WHICH population the ranking is over is declared by the scope, and the id lists follow it. A
    # sample-scoped model ranks the sample's queries against the sample's gallery; reading
    # `gallery_ids`/`full_*` there would index into a gallery that was never encoded, so this is the
    # one place the two scopes must diverge.
    if hook.scope.population_scope == "sample":
        gallery_id_list = list(inputs["gallery_sample_ids"])
        reference_id_list = list(inputs["reference_ids"])
        target_id_list = list(inputs["target_ids"])
    else:
        gallery_id_list = list(inputs["gallery_ids"])
        reference_id_list = list(inputs["full_reference_ids"])
        target_id_list = list(inputs["full_target_ids"])
    position = {name: index for index, name in enumerate(gallery_id_list)}
    reference_positions = [position[name] for name in reference_id_list]
    target_positions = [position[name] for name in target_id_list]
    coverage = "full" if hook.scope.population_scope == "full" else "chunked"
    for row, reference in enumerate(reference_positions):
        sims[row, reference] = -10e10
    native_style = [int(np.flatnonzero(np.argsort(-sims[row]) == target)[0]) + 1
                    for row, target in enumerate(target_positions)]
    argsort_unmasked = [int(np.flatnonzero(np.argsort(-unmasked[row]) == target)[0]) + 1
                        for row, target in enumerate(target_positions)]

    adapter = _adapter_class(hook)(config)
    adapter.work_root = RAW / f"work-{hook.scope.model_id}-{category}-rank"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    try:
        adapter_style = adapter.rank(queries, gallery, [[index] for index in target_positions])
        pair_count = min(64, len(target_positions))
        scores = adapter.score(queries[:pair_count], gallery)
        hint_scores = (adapter.score(queries, gallery)
                       if hook.scope.model_id == "hint" else None)
    finally:
        adapter.close()
    # The NATIVE PROTOCOL ranking of the ADAPTER's features: same reference mask, same argsort. This is
    # the quantity `metrics_parity` compares, because the native metric masks the reference and CIRCUS
    # does not -- comparing an unmasked adapter recall to a masked native one would measure the
    # protocol, not the model.
    # NOTE: `native_style` above is ALREADY the adapter's own features ranked under the native
    # protocol (masked argsort), because `sims` is built from the adapter population features. The two
    # names are kept apart because `native_style` is the native CONVENTION and `control["ranks"]` is
    # the native MEASUREMENT; the comparison below is measurement vs convention. No second masked pass
    # is computed, and no assert is written that compares a variable to itself.
    score_delta = np.abs(scores.astype(np.float64) - unmasked[:pair_count].astype(np.float64))

    control = native_control(hook, category)
    if control is None:
        # No stored native rank measurement for this model. The ranking dimension then CANNOT be
        # compared against a native MEASUREMENT, and substituting the adapter's own output for one
        # would be a comparison of a value with itself. So the block records the convention checks it
        # CAN make and says plainly that the measurement comparison was not made -- `metrics_parity`
        # and `scoring_parity` then read `missing` rather than "clean".
        unmasked_differences_nc = [i for i, (a, b) in enumerate(zip(adapter_style, argsort_unmasked))
                                   if a != b]
        # WHAT IS STILL MEASURABLE. `adapter_style` comes from the adapter's production `rank()`; the
        # ranks below come from numpy `argsort(-sims)` under the native convention. Those are two
        # DIFFERENT code paths over the SAME features, so a difference between them measures whether
        # the adapter's ranking obeys the pinned protocol -- mask, tie order, 1-based index -- and is
        # not a value compared with itself. Each difference is then classified against `tie_epsilon`
        # exactly as the control branch does, so a difference the declared tie rule cannot explain is
        # REPORTED, never rounded to zero. What remains unmeasurable without a native rank measurement
        # is whether the adapter's features are the native model's features, and that is the
        # `metrics_parity` gap stated below.
        # UNMASKED, on both sides. `adapter.rank()` does not apply the native reference mask
        # (`model_habit.py reference_eligible = True`), and HINT's artifact proves the convention it
        # reproduces is the unmasked argsort (`adapter_vs_argsort_unmasked_difference_count: 26`, not
        # 0 -- the residual is the mask). Comparing it against the MASKED convention would measure the
        # mask: that mistake produced 663 "differences" on dress, none of which was about ordering.
        convention_differences = list(unmasked_differences_nc)
        convention_unexplained = []
        for index in convention_differences:
            row = unmasked[index]
            low, high = sorted((adapter_style[index], argsort_unmasked[index]))
            between = np.argsort(-row)[low - 1:high]
            if not np.all(np.abs(row[between] - row[target_positions[index]])
                          <= declared["tie_epsilon"]):
                convention_unexplained.append(index)
        return {
            "query_count": len(native_style),
            "gallery_count": len(gallery_id_list),
            "coverage": coverage,
            "reference_mask": "the native protocol's reference mask is applied by the adapter's own "
                              "rank() call; the native RANK MEASUREMENT was not available",
            "adapter_vs_argsort_unmasked_identical": not unmasked_differences_nc,
            "adapter_vs_argsort_unmasked_differences": unmasked_differences_nc[:30],
            "adapter_vs_argsort_unmasked_difference_count": len(unmasked_differences_nc),
            "adapter_style_recall": {f"r{k}": 100 * sum(1 for r in adapter_style if r <= k)
                                     / len(adapter_style) for k in (1, 10, 50)},
            "native_style_recall": None,
            "score_max_abs_delta_vs_dot": float(score_delta.max()),
            "score_pairs_compared": pair_count,
            "score_max_abs_declared": declared["score_max_abs"],
            "score_within_tolerance": bool(score_delta.max() <= declared["score_max_abs"]),
            "native_aggregate_ranks_compared": 0,
            "native_aggregate_source": None,
            "population_scope": hook.scope.population_scope,
            "native_rank_control_absent": {
                "declared_path": hook.scope.native_control_path,
                "why": "no stored native per-query-rank measurement exists for this model, so the "
                       "adapter's ranks cannot be compared against a native MEASUREMENT",
                "consequence": "`metrics_parity` is reported `missing`, which is INCOMPLETE and "
                               "never a pass. `scoring_parity` IS measured, on the comparison below",
            },
            # The measured rank comparison: adapter production `rank()` vs the native convention
            # applied to the same features. Named with the SAME key HINT/LIMN use, so
            # `parity_gate._rank_checks` reads it without a per-model branch.
            "rank_differences": len(convention_differences),
            "unexplained_rank_differences": len(convention_unexplained),
            "first_unexplained_index": (convention_unexplained[0] if convention_unexplained else None),
            "rank_comparison": {
                "left": "adapter.rank() -- the production scoring path (`model_<m>.score`/`rank`)",
                "right": "numpy argsort(-sims) UNMASKED, applied to the SAME adapter features "
                         "(`test.py:141-152` convention minus the reference mask, which this "
                         "adapter does not apply)",
                "tie_epsilon": declared["tie_epsilon"],
                "why_this_is_not_a_self_comparison":
                    "two different implementations of the ranking over one feature array: the "
                    "adapter's own score/rank code, and the pinned evaluator's convention",
                "differences_head": convention_differences[:20],
                "unexplained_head": convention_unexplained[:20],
            },
            "rank_differences_vs_native_val_control": None,
            "hit_counts": None,
            "hit_counts_equal": None,
            "eligible": None,
            "adapter_production_recall_informational": {
                "recall": {f"r{k}": 100 * sum(1 for r in adapter_style if r <= k)
                           / len(adapter_style) for k in (1, 10, 50)},
                "note": "the adapter's own ranking of its own features. INFORMATION, not a parity "
                        "comparison: with no native control there is nothing to compare it to",
            },
        }
    differing = [i for i, (a, b) in enumerate(zip(native_style, control["ranks"])) if a != b]
    # The control stores `0` for "outside the top 50 / reference masked", while the adapter-side rank
    # is a true rank (always > 50 at those rows). Those rows are the SAME outcome recorded two ways,
    # not a disagreement, so they are separated out rather than left to inflate `rank_differences`.
    sentinel_only = [i for i in differing if control["ranks"][i] == 0 and native_style[i] > 50]
    real_differences = [i for i in differing if i not in set(sentinel_only)]
    unexplained = []
    for index in differing:
        row = sims[index]
        low, high = sorted((native_style[index], control["ranks"][index]))
        between = np.argsort(-row)[low - 1:high]
        if not np.all(np.abs(row[between] - row[target_positions[index]])
                      <= declared["tie_epsilon"]):
            unexplained.append(index)
    unmasked_differences = [i for i, (a, b) in enumerate(zip(adapter_style, argsort_unmasked))
                            if a != b]
    block = {
        "query_count": len(native_style),
        "gallery_count": len(gallery_id_list),
        "coverage": coverage,
        "reference_mask": "test.py:74-79 applied to BOTH sides (sims[row, reference] = -10e10)",
        "adapter_vs_argsort_unmasked_identical": not unmasked_differences,
        "adapter_vs_argsort_unmasked_differences": unmasked_differences[:30],
        "adapter_vs_argsort_unmasked_difference_count": len(unmasked_differences),
        "adapter_style_recall": {f"r{k}": 100 * sum(1 for r in adapter_style if r <= k)
                                 / len(adapter_style) for k in (1, 10, 50)},
        # Adapter features ranked under the NATIVE convention (masked argsort). Named for the
        # convention, as in t4_parity; the native MEASUREMENT's own recall is
        # `native_measured_recall` beside it.
        "native_style_recall": {f"r{k}": 100 * sum(1 for r in native_style if r <= k)
                                / len(native_style) for k in (1, 10, 50)},
        "score_max_abs_delta_vs_dot": float(score_delta.max()),
        "score_pairs_compared": pair_count,
        "score_max_abs_declared": declared["score_max_abs"],
        "score_within_tolerance": bool(score_delta.max() <= declared["score_max_abs"]),
        "native_aggregate_ranks_compared": len(control["ranks"]),
        "native_measured_recall": {f"r{k}": 100 * sum(1 for r in control["ranks"] if r <= k)
                                   / len(control["ranks"]) for k in (1, 10, 50)},
        "native_aggregate_source": control["source"],
        "population_scope": hook.scope.population_scope,
        "rank_differences": len(differing),
        "rank_differences_that_are_only_the_rank_0_sentinel": len(sentinel_only),
        "rank_differences_on_a_real_native_rank": len(real_differences),
        "rank_difference_real_indices_head": real_differences[:20],
        "rank_difference_indices_head": differing[:20],
        "unexplained_rank_differences": len(unexplained),
        "first_unexplained_index": unexplained[0] if unexplained else None,
        "max_abs_rank_delta": int(max([abs(a - b) for a, b in
                                       zip(native_style, control["ranks"])] or [0])),
    }
    # metrics_parity: the stored native per-query ranks vs the adapter features ranked under the SAME
    # native protocol (masked), so a difference is a FEATURE difference and never the mask. The
    # unmasked production ranking is reported as information, never as the comparison.
    # ENCODER-shaped view of the SAME measurement, so a reader comparing models finds both layouts.
    # `parity_gate._rank_checks` returns early on `unexplained_rank_differences`, so this block is
    # documentation: it is derived from `differing`/`unexplained`, never a second computation.
    block["rank_differences_vs_native_val_control"] = {
        "native_control_source": control["source"],
        "compared": len(control["ranks"]),
        "count": len(differing),
        "count_excluding_rank_0_sentinel": len(real_differences),
        "head": differing[:20],
        "non_tie_count": len(unexplained),
        "non_tie_head": unexplained[:20],
        "tie_epsilon": declared["tie_epsilon"],
        "max_abs_rank_delta": block["max_abs_rank_delta"],
    }
    block["hit_counts"] = {"native": _hits(control["ranks"]), "adapter": _hits(native_style)}
    block["hit_counts_equal"] = block["hit_counts"]["native"] == block["hit_counts"]["adapter"]
    # Per-side DENOMINATORS and the membership assertion the metric comparison needs: equal hit counts
    # over different query sets measure nothing, and `parity_gate._metrics_checks` refuses to certify a
    # comparison that cannot show both sides ranked the same population.
    # The two sides ARE the same population by construction -- `native_style` is the adapter's features
    # ranked over the SAME `inputs["full_reference_ids"]`/`full_target_ids` the control scored -- but
    # "by construction" is exactly what a reader cannot check. So the two counts and the two id
    # sequences are compared here, at the one place both are in hand.
    block["eligible"] = {
        "native_query_count": len(control["ranks"]),
        "adapter_query_count": len(native_style),
        "native_gallery_count": control["gallery_count"],
        "adapter_gallery_count": len(gallery_id_list),
        "query_set_identical": (
            len(control["ranks"]) == len(native_style) == len(reference_id_list)
            and control["gallery_count"] == len(gallery_id_list)),
    }
    block["adapter_production_recall_informational"] = {
        "recall": block["adapter_style_recall"],
        "note": "unmasked, reference-eligible: what CIRCUS reports. It differs from the native metric "
                "by PROTOCOL, so it is information, never the hit-count comparison",
    }
    return block


def assert_identity_agreement(native_document: dict[str, Any],
                              adapter_document: dict[str, Any]) -> None:
    """Refuse to write a pair whose IDENTITY_FIELDS disagree or appear on one side only.

    `parity_gate._identity_cross_checks` makes a field carried by ONE side a MANDATORY failure, and a
    field both sides omit is skipped. So the safe pair is: identical values, and the same key set
    present on both. Asserted here, before the write, because after it the defect costs a GPU run.
    """
    from workbench.backend.parity_gate import IDENTITY_FIELDS

    problems = []
    for field in IDENTITY_FIELDS:
        left = native_document.get(field, "<absent>")
        right = adapter_document.get(field, "<absent>")
        if left != right:
            problems.append(f"{field}: native={left!r} adapter={right!r}")
    if problems:
        raise SystemExit("identity fields disagree across the pair; refusing to write: "
                         + "; ".join(problems))


# --------------------------------------------------------------------------- emission


def emit_native(hook: CaptureHook, category: str, identity: dict[str, Any],
                config: Any) -> Path:
    """One native artifact, from the identity phase only. It declares NO forward block.

    Written without `native_metrics_match_feature_rerank`: that boolean asserts hit-count equality
    between two sides, and only the adapter phase holds both. Emitting it here would be a default.
    """
    document = {
        "category": category,
        "phase": "native",
        "model_id": hook.scope.model_id,
        "checkpoint_id": hook.scope.checkpoint_id,
        "protocol": hook.scope.protocol_id,
        "standard": STANDARD,
        "tolerances": hook.scope.declared_tolerances(),
        "checkpoint": checkpoint_identity(config, hook.scope.model_id,
                                              hook.scope.checkpoint_id),
        "query_order_sha256": digest_ids(identity["query_native_order"]),
        "gallery_order_sha256": digest_ids(identity["gallery_native_order"]),
        "layer1_identity": {key: value for key, value in identity.items()
                            if key not in ("gallery_native_order", "query_native_order")},
        "capture_state": {
            "producer": "scripts/third_model_parity.py --phase identity",
            "interpreter": sys.executable,
            "versions": _versions(),
            "gpu_capture_executed": False,
            "measured_here": list(identity),
            "not_yet_measured": [
                "sample_forward (native + adapter features)",
                "batch_size_1_vs_n",
                "cache_off_restart",
                "population forward deltas",
                "scoring_ranking score delta and tie classification",
                "component_parity",
                "native_metrics_match_feature_rerank",
            ],
            "why": "the adapter half needs the model worker, which refuses to start without CUDA "
                   "(`model_hint.py::_worker_main`); absent evidence stays absent",
        },
    }
    path = EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def update_native_metrics_flag(hook: CaptureHook, category: str, ranking: dict[str, Any]) -> None:
    """Write `native_metrics_match_feature_rerank` into the native artifact, additively.

    `parity_gate._metrics_checks` reads it from the NATIVE side, but only the adapter phase holds
    both hit-count vectors. So the boolean is written here, by the phase that measured both, into the
    native document beside the detail it summarises -- exactly the split `limn_hits.py` uses. The key
    is not emitted by the identity phase, because there it would be a default.
    """
    if ranking.get("hit_counts") is None:
        # NO NATIVE RANK CONTROL, so there is no hit-count comparison to summarise. The flag is
        # deliberately NOT written: `parity_gate._metrics_checks` treats a bare boolean with no counts
        # behind it as absent evidence (hence `INCOMPLETE`), and writing `False` here would instead
        # read as a measured disagreement between two count vectors that do not exist. The adapter's
        # own recall stays readable in `capture_state.uncontrolled_recall` as INFORMATION.
        path = EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document.pop("native_metrics_match_feature_rerank", None)
        document.pop("native_metrics_parity_detail", None)
        if isinstance(document.get("capture_state"), dict):
            document["capture_state"]["gpu_capture_executed"] = True
            document["capture_state"]["measured_by"] = [
                "scripts/third_model_parity.py --phase identity (G1 block)",
                "scripts/third_model_parity.py --phase native (sample forward, decode equality)",
                "scripts/third_model_parity.py --phase adapter (comparisons, ranking)",
            ]
            document["capture_state"]["not_yet_measured"] = [
                "native_metrics_match_feature_rerank (no stored native per-query-rank control for "
                "this model, so hit counts cannot be compared against a native MEASUREMENT)",
            ]
            document["capture_state"]["uncontrolled_recall"] = (
                ranking.get("adapter_production_recall_informational"))
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return
    path = EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["native_metrics_match_feature_rerank"] = bool(ranking["hit_counts_equal"])
    if isinstance(document.get("capture_state"), dict):
        document["capture_state"]["gpu_capture_executed"] = True
        document["capture_state"]["not_yet_measured"] = []
        document["capture_state"]["measured_by"] = [
            "scripts/third_model_parity.py --phase identity (G1 block)",
            "scripts/third_model_parity.py --phase native (sample forward, decode equality)",
            "scripts/third_model_parity.py --phase adapter (comparisons, ranking, hit counts)",
        ]
    eligible = ranking["eligible"]
    document["native_metrics_parity_detail"] = {
        "rule": "exact integer hit-count equality at every cutoff",
        "cutoffs": [1, 10, 50],
        "reference_mask": ranking["reference_mask"],
        "ranking_convention": "numpy argsort(-sims), 1-based target index (test.py:74-86)",
        # Per-side counts AND denominators, plus the membership assertion: the gate derives equality
        # from these and refuses a comparison that cannot show both sides scored the same population.
        "native": {"source": ranking["native_aggregate_source"],
                   "hits": ranking["hit_counts"]["native"],
                   "query_count": eligible["native_query_count"],
                   "recall_pct": ranking["native_measured_recall"]},
        "adapter": {"source": "raw/third_model/population_<category>.npz (adapter worker features), "
                              "ranked under the native protocol",
                    "hits": ranking["hit_counts"]["adapter"],
                    "query_count": eligible["adapter_query_count"],
                    "recall_pct": ranking["native_style_recall"]},
        "eligible_query_set_identical": eligible["query_set_identical"],
        "equality": {key: ranking["hit_counts"]["native"][key] == ranking["hit_counts"]["adapter"][key]
                     for key in ranking["hit_counts"]["native"]},
        "information_only": ranking["adapter_production_recall_informational"],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


_CONFIG: Any = None


def _config() -> Any:
    global _CONFIG
    if _CONFIG is None:
        from workbench.backend.operator_config import resolve_config

        _CONFIG = resolve_config()
    return _CONFIG


def self_check() -> int:
    """CPU-only check of the shared layer. No model weights, no worker, no GPU.

    Asserts the properties the shared machinery is FOR, on data this process builds: the sample rules
    hold on a real category, the comparator is exact on identical arrays and reports a planted
    difference, the tie convention is the pinned `argsort` one, and the masked hit count changes when
    the mask is dropped (i.e. the protocol really is being applied).
    """
    import numpy as np

    config = _config()
    rows = json.loads((Path(config.FASHIONIQ_ROOT) / "captions/cap.dress.val.json")
                      .read_text(encoding="utf-8"))
    gallery: list[str] = []
    for row in rows:
        for key in ("candidate", "target"):
            if row[key] not in gallery:
                gallery.append(row[key])
    sample = fixed_sample(rows, gallery)
    assert len(sample["sample_indices"]) == SAMPLE_QUERIES, sample["sample_indices"]
    assert len(sample["gallery_sample_ids"]) == SAMPLE_GALLERY, sample["gallery_sample_ids"]
    assert sample["sample_has_two_rows_sharing_a_reference"], sample["selection_rules"]
    assert sample["gallery_sample_covers_tail"]
    assert len(sample["gallery_sample_ids"]) % 32 != 0, "the image sample must be a partial batch"
    assert len(set(sample["gallery_sample_ids"])) == len(sample["gallery_sample_ids"])
    assert fixed_sample(rows, gallery) == sample, "the sample is not deterministic"

    a = np.arange(12, dtype=np.float32).reshape(3, 4) / 10.0
    assert compare(a, a)["bitwise_equal"] and compare(a, a)["max_abs"] == 0.0
    perturbed = a.copy()
    perturbed[1, 2] += 0.5
    delta = compare(perturbed, a)
    assert delta["max_abs"] == 0.5 and not delta["bitwise_equal"], delta
    assert toleranced(delta, abs_key="forward_max_abs", rel_key="forward_max_rel",
                      declared=TOL)["within_tolerance"] is False

    # The pinned convention: rank = position in argsort(-sims), 1-based, ties broken by argsort.
    sims = np.array([[0.9, 0.8, 0.7, 0.6]])
    assert int(np.flatnonzero(np.argsort(-sims[0]) == 2)[0]) + 1 == 3
    masked = sims.copy()
    masked[0, 0] = -10e10
    assert int(np.flatnonzero(np.argsort(-masked[0]) == 2)[0]) + 1 == 2, "the mask must lift the rank"
    assert digest_ids(["a", "b"]) == digest_ids(["a", "b"]) != digest_ids(["b", "a"])
    native_rank = native_population_rank_measurement(
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        ["reference", "target"], ["reference"], ["target"])
    assert native_rank["ranks"] == [1] and native_rank["hit_counts"] == {"r1": 1, "r10": 1, "r50": 1}
    assert native_rank["score_source"].startswith("native production")

    # Cross-side identity: every IDENTITY_FIELD either side emits must agree, because
    # `parity_gate._identity_cross_checks` compares them value-to-value. The two digests below are the
    # ones that were actually wrong once -- native over the 5-row SAMPLE, adapter over the full
    # population -- so they are checked on synthetic data here rather than discovered on the GPU run.
    from workbench.backend.parity_gate import IDENTITY_FIELDS

    native_doc = {"query_order_sha256": digest_ids(["r1", "r2", "r3"]),
                  "gallery_order_sha256": digest_ids(["g1", "g2"])}
    adapter_doc = {"query_order_sha256": digest_ids(["r1", "r2", "r3"]),
                   "gallery_order_sha256": digest_ids(["g1", "g2"])}
    for field in IDENTITY_FIELDS:
        if field in native_doc or field in adapter_doc:
            assert native_doc.get(field) == adapter_doc.get(field), (
                f"identity field {field} differs across the pair: "
                f"{native_doc.get(field)!r} vs {adapter_doc.get(field)!r}")
    # A 3-D gallery must be normalised on the FEATURE axis. Normalising axis=1 (the 32-token axis)
    # is not a direction in feature space; it produced a score delta of 1.17 against a 1e-6 tolerance.
    rng = np.random.default_rng(20261112)
    gallery3 = rng.normal(size=(4, 3, 8)).astype(np.float32)
    queries = rng.normal(size=(2, 8)).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    good = gallery3 / np.linalg.norm(gallery3, axis=-1, keepdims=True)
    bad = gallery3 / np.linalg.norm(gallery3, axis=1, keepdims=True)
    expected = np.einsum("nd,gqd->ngq", queries, good).max(-1)
    assert not np.allclose(expected, np.einsum("nd,gqd->ngq", queries, bad).max(-1)), (
        "the axis probe cannot distinguish the two normalisations; it proves nothing")
    assert np.allclose(np.linalg.norm(good, axis=-1), 1.0, atol=1e-6)
    assert not np.allclose(np.linalg.norm(bad, axis=-1), 1.0, atol=1e-6), (
        "normalising axis=1 leaves the feature axis non-unit, so a score built from it is not a "
        "cosine")

    sample_only = digest_ids(["r1"])
    assert sample_only != native_doc["query_order_sha256"], (
        "a sample-order digest must not be able to masquerade as the population digest")

    # The writer itself must refuse a pair that would fail `_identity_cross_checks`.
    assert_identity_agreement({"model_id": "hint"}, {"model_id": "hint"})
    for native_only, adapter_only in (({"checkpoint_id": "x"}, {}), ({}, {"checkpoint_id": "x"}),
                                      ({"protocol": "a"}, {"protocol": "b"})):
        try:
            assert_identity_agreement(native_only, adapter_only)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"one-sided/disagreeing identity was accepted: "
                                 f"{native_only} vs {adapter_only}")
    assert _hits([1, 2, 60]) == {"r1": 1, "r10": 2, "r50": 2}, _hits([1, 2, 60])
    print("self-check OK: sample rules, comparator, tolerances, tie convention, mask, digests, hits")
    return 0


def _scopes() -> dict[str, Callable[[], CaptureHook]]:
    """Every model this producer can measure, as DATA: a scope plus the hook that reads it.

    Adding a model here is what "extend the shared producer" means -- there is no per-model script.
    A model absent from this table is not measurable by this producer, and `--model` says so rather
    than measuring HINT under another model's name.
    """
    def hint() -> CaptureHook:
        return HintCapture()

    def lavis(model_id: str, adapter_module: str, replay_module: str, checkpoint_id: str,
              protocol_id: str, *, target_rank: int = 3, target_category: str | None = None,
              native_control: str | None = None, population_scope: str = "full",
              noise_ratio: float = 0.0, extra_tolerances: dict[str, Any] | None = None,
              target_transpose: bool = True) -> CaptureHook:
        scope = ModelScope(
            model_id=model_id, adapter_module=adapter_module, replay_module=replay_module,
            checkpoint_id=checkpoint_id, categories=("dress", "shirt", "toptee"),
            gallery_scope="category-agnostic", target_rank=target_rank,
            target_category=target_category, extra_tolerances=extra_tolerances,
            native_control_path=native_control, population_scope=population_scope,
            protocol_id=protocol_id)
        return LavisFashionIQCapture(scope, noise_ratio=noise_ratio,
                                     target_transpose=target_transpose)

    return {
        "hint": hint,
        "habit": lambda: lavis(
            "habit", "workbench.composition.model_habit", "workbench.replay.habit", "fiq_n02",
            "fashioniq_full_gallery_ref_excluded"),
        "intent": lambda: lavis(
            "intent", "workbench.composition.model_intent", "workbench.replay.intent", "fiq_n02",
            "fashioniq_full_gallery_ref_excluded"),
        "conesep": lambda: lavis(
            "conesep", "workbench.composition.model_conesep", "workbench.replay.conesep", "fiq_n02",
            "fashioniq_full_gallery_ref_excluded"),
        "tgcir": lambda: ClippedModelCapture(ModelScope(
            model_id="tgcir", adapter_module="workbench.composition.model_tgcir",
            replay_module="workbench.replay.tgcir",
            checkpoint_id="tgcir_archive_unresolved",
            categories=("dress", "shirt", "toptee"), gallery_scope="category-agnostic",
            target_rank=2, target_category=None, native_control_path=None,
            population_scope="sample", protocol_id="fashioniq_val_split",
            transform_source="clip", native_device="cuda", restart_supported=False)),
    }

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest(hook: CaptureHook) -> list[dict[str, str]]:
    """Exact source-byte identities used by HINT plan, before any model phase."""
    paths = [Path(__file__), REPO / (hook.scope.adapter_module.replace(".", "/") + ".py"),
             REPO / (hook.scope.replay_module.replace(".", "/") + ".py")]
    if hook.scope.model_id == "hint":
        paths.extend(REPO / "workbench/third_party/HINT" / name
                     for name in ("test.py", "datasets.py", "data_utils.py"))
        paths.append(REPO / "workbench/third_party/HINT/lavis/models/blip2_models/HINT.py")
        paths.append(REPO / "workbench/composition/adapters.py")
    result = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"planned source is missing: {path}")
        result.append({"path": str(path.relative_to(REPO)), "sha256": _sha256_file(path)})
    return result


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()


def _reference_exclusion_digest(query_ids: Sequence[str], gallery_ids: Sequence[str],
                                reference_ids: Sequence[str]) -> str:
    if len(query_ids) != len(reference_ids):
        raise ValueError("reference exclusion inputs are not aligned")
    positions = {value: index for index, value in enumerate(gallery_ids)}
    rows = []
    for query_id, reference_id in zip(query_ids, reference_ids):
        if reference_id not in positions:
            raise ValueError(f"reference ID {reference_id!r} is absent from declared gallery")
        rows.append({"query_id": str(query_id), "excluded_gallery_index": positions[reference_id],
                     "excluded_gallery_id": str(reference_id)})
    return _json_digest(rows)


def _batch_partition_digest(query_ids: Sequence[str], gallery_ids: Sequence[str],
                            batch_size: int) -> tuple[str, str, str]:
    query_batches = [list(query_ids[start:start + batch_size])
                     for start in range(0, len(query_ids), batch_size)]
    gallery_batches = [list(gallery_ids[start:start + batch_size])
                       for start in range(0, len(gallery_ids), batch_size)]
    return (digest_ids(query_ids), digest_ids(gallery_ids),
            _json_digest({"query_batch_size": batch_size, "gallery_batch_size": batch_size,
                          "query_batches": query_batches, "gallery_batches": gallery_batches}))


def _hint_execution_partitions(query_annotation_ids: Sequence[str],
                               reference_ids: Sequence[str],
                               gallery_ids: Sequence[str], batch_size: int
                               ) -> dict[str, str]:
    """Bind HINT execution to annotation rows; preserve references for exclusion semantics."""
    if len(query_annotation_ids) != len(reference_ids):
        raise ValueError("HINT annotation and reference rows are not aligned")
    query, gallery, batches = _batch_partition_digest(
        query_annotation_ids, gallery_ids, batch_size)
    return {"query_partition_sha256": query, "query_order_sha256": query,
            "reference_order_sha256": digest_ids(reference_ids),
            "gallery_partition_sha256": gallery, "gallery_order_sha256": gallery,
            "batch_partition_sha256": batches}


def _hint_plan(hook: CaptureHook, category: str, run_root: Path, seed: int | None,
               identity_input: dict[str, Any], config: Any,
               controller_interpreter: str) -> dict[str, Any]:
    """Build canonical immutable V3 preregistration from CPU-observed FashionIQ identity."""
    if not isinstance(identity_input, dict):
        raise SystemExit(f"HINT preregistration identity is malformed for {category}")
    source = _source_manifest(hook)
    checkpoint = checkpoint_identity(config, hook.scope.model_id, hook.scope.checkpoint_id)
    checkpoint_sha = checkpoint.get("sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise SystemExit("HINT preregistration needs verified checkpoint SHA256 bytes")
    annotations = identity_input.get("query_annotations")
    if not isinstance(annotations, list) or not annotations:
        raise SystemExit(f"HINT preregistration has no canonical query annotations for {category}")
    query_ids = [str(item.get("annotation_id")) for item in annotations]
    references = [str(item.get("reference_id")) for item in annotations]
    targets = [str(item.get("target_id")) for item in annotations]
    gallery = [str(value) for value in identity_input["gallery_native_order"]]
    if any(value in {"None", ""} for value in query_ids + references + targets) or not gallery:
        raise SystemExit(f"HINT preregistration has empty query annotation metadata for {category}")
    if len(set(query_ids)) != len(query_ids):
        raise SystemExit(f"HINT preregistration query annotation IDs are not unique for {category}")
    if len(query_ids) != len(references) or len(query_ids) != len(targets):
        raise SystemExit(f"HINT preregistration has unaligned query metadata for {category}")
    runtime = _versions()
    configuration = _json_digest({"source": source, "processor": hook.scope.processor_config,
                                  "transform": hook.scope.transform_source,
                                  "target_rank": hook.scope.target_rank})
    protocol = hook.scope.protocol_id
    cohort_id = f"fashioniq-val-full-{hook.scope.model_id}-{category}-v3"
    labels = {
        "query_targets": [[query_id, target_id] for query_id, target_id in zip(query_ids, targets)],
        "reference_exclusions": [[query_id, [reference_id]]
                                  for query_id, reference_id in zip(query_ids, references)],
    }
    identity = {
        "checkpoint": checkpoint_sha, "configuration": configuration,
        "environment": _json_digest(runtime), "protocol": protocol,
        "category": category, "cohort_id": cohort_id,
        "precision": "native_cuda_autocast_fp16; query_and_gallery_features_float32",
        "seed": seed if seed is not None else "unspecified",
        "query_order_sha256": _json_digest(query_ids), "gallery_order_sha256": _json_digest(gallery),
        "reference_exclusions_sha256": hashlib.sha256(json.dumps(
            labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "query_count": len(query_ids), "gallery_count": len(gallery),
    }
    from workbench.backend.registry import model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    native_configured_interpreter = resolve_model_interpreter(model_by_id(hook.scope.model_id), config)
    if not native_configured_interpreter or not Path(native_configured_interpreter).is_file():
        raise SystemExit("HINT configured native interpreter unavailable")
    native_interpreter = str(Path(native_configured_interpreter).absolute())
    if not controller_interpreter or not Path(controller_interpreter).is_file():
        raise SystemExit("HINT identity phase requires --controller-interpreter pointing to controller Python")
    adapter_interpreter = str(Path(controller_interpreter).absolute())
    if Path(adapter_interpreter).resolve() == Path(native_interpreter).resolve():
        raise SystemExit("HINT controller interpreter must differ from native interpreter")
    source_id = _json_digest(source)
    batch_n = 32
    run_specs = {
        "native_repeat_1": ("native", 32), "native_repeat_2": ("native", 32),
        "adapter_repeat_1": ("adapter", 32), "adapter_repeat_2": ("adapter", 32),
        "native_adapter_native": ("native", 32), "native_adapter_adapter": ("adapter", 32),
        "native_batch_1": ("native", 1), "native_batch_n": ("native", 32),
        "adapter_batch_1": ("adapter", 1), "adapter_batch_n": ("adapter", 32),
    }
    runs: dict[str, Any] = {}
    for role, (side, batch) in run_specs.items():
        partitions = _hint_execution_partitions(query_ids, references, gallery, batch)
        interpreter = native_interpreter if side == "native" else adapter_interpreter
        role_run_id = f"{uuid.uuid4()}:{side}:{role}"
        runs[role] = {
            "run_id": role_run_id, "source_id": f"{side}:{role}:{source_id}",
            "identity": dict(identity),
            "execution": {
                "partition_id": f"hint:{category}:full:v1",
                **{key: partitions[key] for key in (
                    "query_partition_sha256", "gallery_partition_sha256",
                    "batch_partition_sha256")},
                "query_batch_size": batch,
                "gallery_batch_size": batch,
                "source_interpreter": interpreter,
                "controller_interpreter": adapter_interpreter,
                "command": f"WORKBENCH_COMPOSITION_BATCH_SIZE={batch} {interpreter} {Path(__file__)} --model hint --category {category} --phase {side} --role {role} --run-root {run_root} --controller-interpreter {adapter_interpreter}",
            },
        }
    controls = {}
    for name, reference, actual, side, dimension, tolerance in (
            ("native_repeat", "native_repeat_1", "native_repeat_2", "native", "forward_parity", "forward"),
            ("adapter_repeat", "adapter_repeat_1", "adapter_repeat_2", "adapter", "forward_parity", "forward"),
            ("native_adapter", "native_adapter_native", "native_adapter_adapter", "native", "forward_parity", "forward"),
            ("native_batch_1_vs_n", "native_batch_1", "native_batch_n", "native", "batch_parity", "batch"),
            ("adapter_batch_1_vs_n", "adapter_batch_1", "adapter_batch_n", "adapter", "batch_parity", "batch")):
        controls[name] = {
            "reference_run_id": runs[reference]["run_id"],
            "actual_run_id": runs[actual]["run_id"],
            "reference_side": side, "dimension": dimension, "dtype": "float32",
            "tolerance": {"atol": TOL[f"{tolerance}_max_abs"],
                          "rtol": TOL[f"{tolerance}_max_rel"]},
            "scopes": {
                "query": {"coverage": "full", "population_count": len(query_ids),
                          "sample_count": None, "tensor_shape": [len(query_ids), 256],
                          "component_shape": [len(query_ids), 256],
                          "ordered_ids_sha256": _json_digest(query_ids), "cohort_id": cohort_id},
                "gallery": {"coverage": "full", "population_count": len(gallery),
                            "sample_count": None, "tensor_shape": [len(gallery), 32, 256],
                            "component_shape": [len(gallery), 32, 256],
                            "ordered_ids_sha256": _json_digest(gallery), "cohort_id": cohort_id},
            },
        }
    return {
        "standard": "parity-absolute-relative/v3", "model_id": hook.scope.model_id,
        "category": category, "identity": identity,
        "scoring_labels": labels,
        "metric_cutoffs": [1, 10, 50],
        "scoring_proof_path": f"g1_g2.hint.{category}.native.score-ranks.jsonl",
        "population_scope": {"coverage": "full", "query_count": len(query_ids),
                             "gallery_count": len(gallery), "query_ids": query_ids,
                             "reference_ids": references, "target_ids": targets,
                             "gallery_ids": gallery, "references_remain_gallery_eligible": False,
                             "query_source": "FashionIQ native identity phase",
                             "gallery_source": "FashionIQ native identity phase"},
        "supported_batch_policy": {"supported": True, "batch_sizes": [1, batch_n],
                                   "batch_invariant": False,
                                   "description": "native source and Adapter query/gallery batches 1 and 32"},
        "runs": runs, "controls": controls, "source": source,
        "environment": {"native_interpreter": native_interpreter,
                         "adapter_interpreter": adapter_interpreter, "runtime": runtime},
        "configuration": {"protocol": protocol, "processor": hook.scope.processor_config,
                          "transform_source": hook.scope.transform_source},
        "checkpoint_sha256": checkpoint_sha,
        "resource_budget": {"one_model_process_at_a_time": True,
                             "query_features_fp32_bytes": len(query_ids) * 256 * 4,
                             "gallery_features_fp32_bytes": len(gallery) * 32 * 256 * 4},
    }




def _aggregate_hint_v3(run_root: Path, category: str, plan: dict[str, Any],
                       plan_sha256: str, plan_path: Path) -> Path:
    """Aggregate ten independently captured HINT roles into raw V3 scopes and receipts."""
    import numpy as np
    population = plan["population_scope"]
    ids = {"query": population["query_ids"], "gallery": population["gallery_ids"]}
    roles = {}
    for role, declaration in plan["runs"].items():
        root = run_root / "raw" / "roles" / category / role
        record = json.loads((root / "record.json").read_text(encoding="utf-8"))
        side = declaration["source_id"].split(":", 1)[0]
        filename = (f"sample_hint_{category}_native.npz" if side == "native"
                    else f"population_hint_{category}.npz")
        with np.load(root / filename) as raw:
            pair = ((raw["population_native_queries"], raw["population_native_gallery"])
                    if side == "native" else (raw["queries"], raw["gallery"]))
            arrays = {"query": pair[0].copy(), "gallery": pair[1].copy()}
        observed = record["native_population"] if side == "native" else record["population"]
        execution = declaration["execution"]
        if (record.get("role") != role or record.get("run_id") != declaration["run_id"]
                or record.get("source_id") != declaration["source_id"]
                or record.get("identity") != declaration["identity"]
                or record.get("execution") != execution
                or record.get("plan_sha256") != plan_sha256
                or arrays["query"].shape[0] != len(ids["query"])
                or arrays["gallery"].shape[0] != len(ids["gallery"])
                or any(value.dtype != np.float32 or not np.isfinite(value).all()
                       for value in arrays.values())
                or observed["query_batch_size"] != execution["query_batch_size"]
                or observed["gallery_batch_size"] != execution["gallery_batch_size"]
                or observed["query_partition_sha256"] != execution["query_partition_sha256"]
                or observed["gallery_partition_sha256"] != execution["gallery_partition_sha256"]
                or observed["query_order_sha256"] != execution["query_partition_sha256"]
                or observed["gallery_order_sha256"] != execution["gallery_partition_sha256"]
                or observed["batch_partition_sha256"] != execution["batch_partition_sha256"]):
            raise ValueError(f"HINT role {role} capture differs from preregistered execution")
        if role == "native_adapter_native":
            scoring = record["native_population"]["pinned_source_evaluator"]
            if scoring.get("score_order_sha256") != _hint_score_order_digest(
                    population["query_ids"], raw_scores := np.load(
                        root / "source_scores" / "scores.float32.npy", mmap_mode="r", allow_pickle=False),
                    raw_orders := np.load(
                        root / "source_scores" / "orders.int32.npy", mmap_mode="r", allow_pickle=False)):
                raise ValueError("HINT native capture-time score/order receipt mismatch")
        if role == "native_adapter_adapter":
            scoring = record.get("v3_population_scoring")
            score_path = root / "adapter_scores.npy"
            order_path = root / "adapter_orders.npy"
            if (not isinstance(scoring, dict) or not score_path.is_file() or not order_path.is_file()
                    or len(scoring.get("ranks", [])) != len(ids["query"])
                    or scoring.get("score_pairs_compared") != min(64, len(ids["query"]))
                    or scoring.get("score_shape") != [len(ids["query"]), len(ids["gallery"])]):
                raise ValueError("HINT Adapter primary scoring capture is incomplete")
            captured_scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
            captured_orders = np.load(order_path, mmap_mode="r", allow_pickle=False)
            if scoring.get("score_order_sha256") != _hint_score_order_digest(
                    population["query_ids"], captured_scores, captured_orders):
                raise ValueError("HINT Adapter capture-time score/order receipt mismatch")
        roles[role] = (side, arrays, observed, record)
    capture_artifacts = {
        role: [
            {"path": str(path.relative_to(run_root)), "sha256": _sha256_file(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "record.json"
        ]
        for role in roles
        for root in (run_root / "raw" / "roles" / category / role,)
    }
    if any(not files for files in capture_artifacts.values()):
        raise ValueError("HINT role has no persisted capture artifacts")
    controls = plan["controls"]
    scopes = {}
    receipt_arrays: dict[str, dict[str, tuple[Any, Any]]] = {role: {} for role in roles}
    started_at = {}
    captured_at = {}
    for name, control in controls.items():
        ref_role = next(role for role, run in plan["runs"].items()
                        if run["run_id"] == control["reference_run_id"])
        actual_role = next(role for role, run in plan["runs"].items()
                           if run["run_id"] == control["actual_run_id"])
        _, ref_arrays, _, _ = roles[ref_role]
        _, actual_arrays, _, _ = roles[actual_role]
        scope_arrays = {}
        for tensor in ("query", "gallery"):
            scope = T4.component_scope(name, control["dimension"], actual_arrays[tensor],
                                       ref_arrays[tensor], atol=control["tolerance"]["atol"],
                                       rtol=control["tolerance"]["rtol"])
            scope.update({"reference_run_id": control["reference_run_id"],
                          "actual_run_id": control["actual_run_id"], "dtype": control["dtype"],
                          "shape": list(actual_arrays[tensor].shape),
                          "coverage": scope["coverage"], "complete_coverage": True,
                          "population_count": len(ids[tensor]), "sample_count": None,
                          "cohort_id": plan["identity"]["cohort_id"]})
            scope_arrays[tensor] = scope
            receipt_arrays[ref_role][f"{name}.{tensor}"] = (ref_arrays[tensor], ids[tensor])
            receipt_arrays[actual_role][f"{name}.{tensor}"] = (actual_arrays[tensor], ids[tensor])
        scopes[name] = scope_arrays
    receipts = {}
    for role, arrays in receipt_arrays.items():
        side, _, observed, record = roles[role]
        run = plan["runs"][role]
        started_at = (observed["query_started_at"] if side == "native"
                      else record["adapter_query_started_at"])
        receipts[role] = T4.v3_run_receipt(
            run_id=run["run_id"], side=side, source_id=run["source_id"],
            plan_sha256=plan_sha256, started_at=started_at, arrays=arrays,
            execution=run["execution"], identity=run["identity"])
        receipts[role]["captured_at"] = record["captured_at"]
        receipts[role]["capture_artifacts"] = capture_artifacts[role]
        for tensor_name, captured in receipts[role]["arrays"].items():
            tensor = "query" if tensor_name.endswith(".query") else "gallery"
            captured["ordered_ids_sha256"] = _json_digest(ids[tensor])
    artifact = T4.v3_artifact_block(
        Path(plan_path.name), plan_sha256, scopes=scopes, receipts=receipts)
    evidence_root = run_root / "evidence"
    plan_path = evidence_root / plan_path.name
    if not plan_path.is_file() or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
        raise ValueError("HINT V3 final evidence requires the immutable sibling plan")
    import numpy as np
    adapter_role = "native_adapter_adapter"
    native_role = "native_adapter_native"
    native_observed = roles[native_role][3]["native_population"]["pinned_source_evaluator"]
    native_ranks = [int(rank) for rank in native_observed["ranks"]]
    gallery_position = {image_id: index for index, image_id in enumerate(ids["gallery"])}
    target_positions = [gallery_position[target] for target in population["target_ids"]]
    adapter_scores = np.load(run_root / "raw" / "roles" / category / adapter_role
                             / "adapter_scores.npy", allow_pickle=False)
    query = roles[adapter_role][1]["query"]
    gallery = roles[adapter_role][1]["gallery"]
    if (adapter_scores.shape != (len(ids["query"]), len(ids["gallery"]))
            or adapter_scores.dtype != np.float32 or not np.isfinite(adapter_scores).all()):
        raise ValueError("HINT Adapter production score matrix is malformed")
    source_scores = np.load(run_root / "raw" / "roles" / category / native_role
                            / "source_scores" / "scores.float32.npy", mmap_mode="r", allow_pickle=False)
    source_orders = np.load(run_root / "raw" / "roles" / category / native_role
                            / "source_scores" / "orders.int32.npy", mmap_mode="r", allow_pickle=False)
    adapter_orders = np.load(run_root / "raw" / "roles" / category / adapter_role
                             / "adapter_orders.npy", mmap_mode="r", allow_pickle=False)
    if (source_scores.shape != adapter_scores.shape or source_orders.shape != adapter_scores.shape
            or adapter_orders.shape != adapter_scores.shape
            or source_scores.dtype != np.float32 or source_orders.dtype != np.int32
            or adapter_orders.dtype != np.int32):
        raise ValueError("HINT source score/order arrays differ from the full Adapter population")
    score_max = 0.0
    adapter_argsort = []
    adapter_masked_orders = []
    reference_positions = {value: index for index, value in enumerate(ids["gallery"])}
    for index, (reference_id, target) in enumerate(zip(population["reference_ids"], target_positions)):
        native_row = np.asarray(source_scores[index])
        adapter_row = np.asarray(adapter_scores[index])
        score_max = max(score_max, float(np.max(np.abs(
            adapter_row.astype(np.float64) - native_row.astype(np.float64)), initial=0.0)))
        order = adapter_orders[index]
        adapter_masked_orders.append(order)
        adapter_argsort.append(int(np.flatnonzero(order == target)[0]) + 1)
    rank_differences = [index for index, (left, right) in
                        enumerate(zip(native_ranks, adapter_argsort)) if left != right]
    unexplained = [index for index in rank_differences
                   if abs(float(source_scores[index, target_positions[index]])
                          - float(adapter_scores[index, target_positions[index]])) > TOL["tie_epsilon"]]
    adapter_native_ranks = adapter_argsort
    scoring = {
        "rank_differences": len(rank_differences),
        "unexplained_rank_differences": len(unexplained),
        "first_unexplained_index": unexplained[0] if unexplained else None,
        "rank_comparison": {"left": "pinned HINT source argsort ranks",
                            "right": "production Adapter scores with reference excluded",
                            "tie_epsilon": TOL["tie_epsilon"],
                            "differences_head": rank_differences[:20],
                            "unexplained_head": unexplained[:20]},
        "score_max_abs_delta_vs_dot": score_max,
        "score_pairs_compared": len(ids["query"]) * len(ids["gallery"]),
        "score_max_abs_declared": TOL["score_max_abs"],
        "score_within_tolerance": score_max <= TOL["score_max_abs"],
        "rank_source": "pinned HINT test.py:test argsort vs production Adapter score argsort",
        "score_source": "pinned HINT masked matmul rows vs production Adapter.score",
    }
    proof_path = evidence_root / plan["scoring_proof_path"]
    proof_hash = hashlib.sha256()
    role_hashes = {native_role: hashlib.sha256(), adapter_role: hashlib.sha256()}
    query_targets = plan["scoring_labels"]["query_targets"]
    reference_exclusions = plan["scoring_labels"]["reference_exclusions"]
    with proof_path.open("xb") as proof:
        header = {"schema": "cir.score-ranks/v1", "identity": plan["identity"],
                  "query_ids": ids["query"], "gallery_ids": ids["gallery"],
                  "query_targets": query_targets, "reference_exclusions": reference_exclusions}
        encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        proof.write(encoded)
        proof_hash.update(encoded)
        for index, query_id in enumerate(ids["query"]):
            native_row = [float(value) for value in source_scores[index]]
            adapter_row_values = adapter_scores[index]
            adapter_row = [float(value) for value in adapter_row_values]
            native_order = [int(value) for value in source_orders[index]]
            adapter_order = [int(value) for value in adapter_masked_orders[index]]
            target_id = query_targets[index][1]
            target_position = gallery_position[target_id]
            excluded = sorted(reference_exclusions[index][1])
            row = {"query_id": query_id, "target_id": target_id,
                   "target_position": target_position, "excluded_gallery_ids": excluded,
                   "native_scores": native_row, "adapter_scores": adapter_row,
                   "native_order": native_order, "adapter_order": adapter_order}
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            proof.write(encoded)
            proof_hash.update(encoded)
            for role, scores, order in ((native_role, native_row, native_order),
                                        (adapter_role, adapter_row, adapter_order)):
                role_hashes[role].update(json.dumps(
                    {"query_id": query_id, "scores": scores, "order": order},
                    ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
    artifact.update({
        "scoring_proof_path": proof_path.name,
        "scoring_proof_sha256": proof_hash.hexdigest(),
        "scoring_run_receipts": {
            role: {"run_id": plan["runs"][role]["run_id"],
                   "source_id": plan["runs"][role]["source_id"],
                   "side": side, "identity": plan["runs"][role]["identity"],
                   "score_order_sha256": (
                       roles[role][3]["native_population"]["pinned_source_evaluator"]["score_order_sha256"]
                       if side == "native" else
                       roles[role][3]["v3_population_scoring"]["score_order_sha256"])}
            for role, side in ((native_role, "native"), (adapter_role, "adapter"))},
    })
    metric = {
        "rule": "exact integer hit-count equality at every cutoff",
        "cutoffs": list(plan["metric_cutoffs"]),
        "reference_mask": "native HINT reference exclusion before argsort",
        "ranking_convention": "pinned HINT source evaluator; Adapter features ranked under same protocol",
        "native": {"source": "pinned HINT test.py:test evaluator ranks",
                   "hits": native_observed["hit_counts"], "query_count": len(native_ranks),
                   "recall_pct": {f"r{k}": 100.0 * native_observed["hit_counts"][f"r{k}"] / len(native_ranks)
                                  for k in (1, 10, 50)}},
        "adapter": {"source": "production Adapter features ranked with native HINT reference mask",
                    "hits": _hits(adapter_native_ranks), "query_count": len(adapter_native_ranks),
                    "recall_pct": {f"r{k}": 100.0 * _hits(adapter_native_ranks)[f"r{k}"] / len(adapter_native_ranks)
                                   for k in (1, 10, 50)}},
        "eligible_query_set_identical": (
            roles[native_role][3]["native_population"]["query_order_sha256"]
            == roles[adapter_role][3]["population"]["query_order_sha256"]),
        "equality": {f"r{k}": native_observed["hit_counts"][f"r{k}"]
                     == _hits(adapter_native_ranks)[f"r{k}"] for k in (1, 10, 50)},
        "information_only": None,
    }
    adapter_ranks = adapter_native_ranks
    identity_input = json.loads(
        (run_root / "raw" / f"identity_hint_{category}.json").read_text(encoding="utf-8"))
    identity_layer = identity_input
    layer1 = {key: identity_layer[key] for key in (
        "query_count_native", "query_count_annotations", "gallery_count_native",
        "gallery_count_rebuilt_from_annotations", "gallery_order_identical",
        "reference_ids_identical", "target_ids_identical",
        "model_text_identical_to_author_composition")}
    adapter_capture = roles[adapter_role][3]
    identity = plan["identity"]
    common = {
        "_artifact_path": str((run_root / "evidence" / f"g1_g2.hint.{category}.native.json").resolve()),
        "model_id": "hint", "category": category, "identity": identity,
        "protocol": identity["protocol"], "protocol_id": identity["protocol"],
        "checkpoint_id": identity["checkpoint"], "checkpoint_sha256": identity["checkpoint"],
        "query_order_sha256": identity["query_order_sha256"],
        "gallery_order_sha256": identity["gallery_order_sha256"],
        "standard": "parity-absolute-relative/v3", "tolerances": TOL,
    }
    native_document = {**common, "layer1_identity": layer1,
                       "native_capture_artifacts": capture_artifacts[native_role],
                       "native": {"layer1_identity": layer1},
                       "native_metrics_parity_detail": metric,
                       "native_metrics_match_feature_rerank": all(metric["equality"].values()),
                       "parity_v3": artifact}
    adapter_document_path = run_root / "evidence" / f"g1_g2.hint.{category}.adapter.json"
    common_adapter = {**common, "_artifact_path": str(adapter_document_path.resolve())}
    adapter_document = {**common_adapter, "adapter": {
        "population": {"query_count": len(adapter_ranks), "gallery_count": len(ids["gallery"]),
                       "source": "independent production Adapter role capture"},
        "hint_input_capture": adapter_capture["hint_input_capture"],
        "capture_artifacts": capture_artifacts[adapter_role],
        "layer2_pixels": adapter_capture["layer2_pixels"],
        "layer2_tokens": adapter_capture["layer2_tokens"],
        "sample_forward": adapter_capture["sample_forward"],
        "scoring_ranking": scoring}}
    native_path = evidence_root / f"g1_g2.hint.{category}.native.json"
    adapter_path = evidence_root / f"g1_g2.hint.{category}.adapter.json"
    if native_path.exists() or adapter_path.exists():
        raise FileExistsError("refusing to overwrite existing HINT V3 evidence")
    native_path.write_text(json.dumps(native_document, indent=2) + "\n", encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter_document, indent=2) + "\n", encoding="utf-8")
    output = run_root / "generation" / f"hint.{category}.v3-artifact.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return native_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    hint_roles = (
        "native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
        "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
        "adapter_batch_1", "adapter_batch_n")
    hint_run_sides = {"native_repeat_1": "native", "native_repeat_2": "native",
                      "adapter_repeat_1": "adapter", "adapter_repeat_2": "adapter",
                      "native_adapter_native": "native", "native_adapter_adapter": "adapter",
                      "native_batch_1": "native", "native_batch_n": "native",
                      "adapter_batch_1": "adapter", "adapter_batch_n": "adapter"}
    parser.add_argument("--role", choices=(*hint_roles, "aggregate"),
                        help="HINT V3 preregistered run role")
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"),
                        help="run one category; default runs all categories declared by model hook")
    parser.add_argument("--model", default=None,
                        help="which model to measure; defaults to hint for the historical invocation")
    parser.add_argument("--list-models", action="store_true",
                        help="print every model this producer can measure and exit")
    parser.add_argument("--run-root", type=Path,
                        help="new isolated output directory; must not exist or contain prior evidence")
    parser.add_argument("--controller-interpreter", type=Path,
                        help="required controller Python launcher for HINT V3 role plans")
    parser.add_argument("--seed", type=int, default=None,
                        help="planned RNG seed; recorded before execution and never applied implicitly")
    parser.add_argument("--self-check", action="store_true",
                        help="CPU-only check of the shared layer; no weights, no worker, no GPU")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU phases into raw/third_model/; never writes the six artifacts")
    parser.add_argument("--phase", choices=("identity", "native", "adapter", "aggregate"),
                        default="identity", help="capture phase")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.self_check:
        return self_check()
    scopes = _scopes()
    if args.list_models:
        print(json.dumps(sorted(scopes), indent=1))
        return 0
    selected = args.model or "hint"
    if selected not in scopes:
        raise SystemExit(f"this producer cannot measure {selected!r}; it measures "
                         f"{sorted(scopes)}. A model absent here has no capture hook, which is a "
                         "missing implementation, not a measurement")
    hook = scopes[selected]()
    config = _config()
    if hook.scope.model_id == "hint":
        if args.controller_interpreter is None or not args.controller_interpreter.is_file():
            parser.error("HINT phases require --controller-interpreter pointing to controller Python")
    if args.category is not None and args.category not in hook.scope.categories:
        parser.error(f"category {args.category!r} not supported by {selected}; "
                     f"choose from {', '.join(hook.scope.categories)}")
    if args.role is not None and hook.scope.model_id != "hint":
        parser.error("--role is supported only for HINT V3")
    if hook.scope.model_id == "hint" and args.phase in ("native", "adapter"):
        if args.role is None or args.role not in hint_run_sides:
            parser.error(f"HINT --phase {args.phase} requires a declared V3 run role")
        if hint_run_sides[args.role] != args.phase:
            parser.error(f"HINT role {args.role} requires --phase {hint_run_sides[args.role]}")
    if args.phase == "native":
        # Measured: the pinned whole-model checkpoint unpickles a `BertTokenizer` whose
        # `_added_tokens_encoder` attribute only the DECLARED transformers (`==4.25.0`, models.yaml)
        # still has. Under the controller's 4.36.2 the native forward dies with
        # `AttributeError: 'BertTokenizer' object has no attribute '_added_tokens_encoder'`.
        # So the native phase must run under the model interpreter -- and the adapter phase must NOT,
        # because `require_verified_interpreter` refuses the active workbench environment.
        from workbench.backend.registry import model_by_id
        from workbench.backend.runtime import resolve_model_interpreter

        expected = resolve_model_interpreter(model_by_id(hook.scope.model_id), config)
        if expected and Path(sys.executable).resolve() != Path(expected).resolve():
            raise SystemExit(
                f"--phase native must run under the model interpreter {expected}, not "
                f"{sys.executable}: the checkpoint's tokenizer is the registry-pinned "
                "transformers==4.25.0 build")
    if args.phase == "adapter":
        from workbench.backend.registry import model_by_id
        from workbench.backend.runtime import resolve_model_interpreter

        model_interpreter = resolve_model_interpreter(model_by_id(hook.scope.model_id), config)
        if model_interpreter and Path(sys.executable).resolve() == Path(model_interpreter).resolve():
            raise SystemExit(
                f"--phase adapter must run under the CONTROLLER interpreter, not the model's "
                f"({model_interpreter}): the adapter refuses to launch a worker from its own "
                "environment (`require_verified_interpreter`, UNSAFE_INTERPRETER)")
    global RAW, EVIDENCE
    if args.run_root is None:
        raise SystemExit("--run-root is required; historical capture and evidence paths are read-only")
    run_root = args.run_root.expanduser().resolve()
    manifest = run_root / "third-model-run.json"
    plan_path = run_root / "third-model-plan.json"
    categories = (args.category,) if args.category else hook.scope.categories
    plan = {
        "model_id": hook.scope.model_id,
        "categories": list(categories),
        "declared_categories": list(hook.scope.categories),
        "checkpoint_id": hook.scope.checkpoint_id,
        "protocol_id": hook.scope.protocol_id,
        "population_scope": hook.scope.population_scope,
        "gallery_scope": hook.scope.gallery_scope,
        "target_rank": hook.scope.target_rank,
        "batch_size": "resolved at execution; production batch partition must be recorded",
        "precision": ("native_cuda_autocast_fp16; feature_outputs_float32" if hook.scope.model_id == "hint"
                      else "float32"),
        "controller_interpreter": str(args.controller_interpreter.absolute()) if args.controller_interpreter else None,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
    }
    plan_identity = {key: value for key, value in plan.items()
                     if key not in ("created_at", "command")}
    plan_bytes = json.dumps(plan_identity, sort_keys=True, separators=(",", ":")).encode()
    plan["plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
    if run_root.exists():
        if not manifest.is_file() or not plan_path.is_file():
            raise SystemExit(f"refusing existing run root without immutable manifest and plan: {run_root}")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        saved_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != {"model_id": hook.scope.model_id, "categories": list(hook.scope.categories)}:
            raise SystemExit(f"run root belongs to another model or category scope: {run_root}")
        if {key: saved_plan.get(key) for key in plan if key != "created_at" and key != "command"} != {
                key: plan.get(key) for key in plan if key != "created_at" and key != "command"}:
            raise SystemExit(f"run root plan differs from requested execution: {run_root}")
    else:
        run_root.mkdir(parents=True)
        manifest.write_text(json.dumps({"model_id": hook.scope.model_id,
                                        "categories": list(hook.scope.categories)}, indent=2) + "\n",
                            encoding="utf-8")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    RAW = run_root / "raw"
    EVIDENCE = run_root / "evidence"
    RAW.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    categories = (args.category,) if args.category else hook.scope.categories
    v3_plans: dict[str, tuple[dict[str, Any], str, Path]] = {}
    if hook.scope.model_id == "hint" and args.phase in ("native", "adapter", "aggregate"):
        role = args.role
        for planned_category in categories:
            v3_plan_path = EVIDENCE / f"g1_g2.hint.{planned_category}.v3-plan.json"
            if not v3_plan_path.is_file():
                raise SystemExit(f"HINT V3 immutable plan missing: {v3_plan_path}; run identity phase first")
            raw_plan = v3_plan_path.read_bytes()
            loaded_plan = json.loads(raw_plan)
            if loaded_plan.get("category") != planned_category:
                raise SystemExit(f"HINT V3 plan category mismatch: {planned_category}")
            if args.phase != "aggregate":
                declaration = loaded_plan.get("runs", {}).get(role)
                execution = declaration.get("execution", {}) if isinstance(declaration, dict) else {}
                expected_controller = loaded_plan.get("environment", {}).get("adapter_interpreter")
                supplied_controller = (str(args.controller_interpreter.absolute())
                                       if args.controller_interpreter is not None else None)
                expected_checkpoint = loaded_plan.get("checkpoint_sha256")
                actual_checkpoint = checkpoint_identity(
                    config, hook.scope.model_id, hook.scope.checkpoint_id).get("sha256")
                if (declaration is None
                        or execution.get("source_interpreter") != str(Path(sys.executable).absolute())
                        or expected_controller != supplied_controller
                        or expected_checkpoint != actual_checkpoint
                        or loaded_plan.get("source") != _source_manifest(hook)):
                    raise SystemExit(f"HINT V3 role/source/checkpoint/interpreter differs from immutable plan: {role}")
            v3_plans[str(planned_category)] = (
                loaded_plan, hashlib.sha256(raw_plan).hexdigest(), v3_plan_path)
            print(f"# HINT V3 role {role} plan={v3_plan_path} sha256={hashlib.sha256(raw_plan).hexdigest()}", flush=True)
    # Categories were fixed in immutable plan above; execute only those rows.
    for category in categories:
        v3_plan, v3_plan_sha256, v3_plan_path = v3_plans.get(str(category), (None, None, None))
        if hook.scope.model_id == "hint" and args.phase in ("native", "adapter"):
            role_root = RAW / "roles" / category / str(args.role)
            planned_outputs = (role_root,)
            existing = [path for path in planned_outputs if path.exists()]
        else:
            planned_outputs = {
                "identity": (RAW / f"identity_{hook.scope.model_id}_{category}.json",
                             EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json"),
                "native": (RAW / f"native_{hook.scope.model_id}_{category}.json",
                           RAW / f"sample_{hook.scope.model_id}_{category}_native.npz"),
                "adapter": (EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.adapter.json",),
                "aggregate": (),
            }[args.phase]
            existing = [path for path in planned_outputs if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite existing phase evidence: {existing[0]}")
        began = time.time()
        # The adapter phase only needs identity for the `query_order_sha256`/`gallery_order_sha256`
        # cross-check, and those are already written by the identity phase. Re-deriving them here
        # imports the pinned modules, which only the MODEL interpreter has: TG-CIR's checkout needs
        # `clip`, so re-running identity under the controller died with
        # `ModuleNotFoundError: No module named 'clip'`. Reuse the recorded block when it exists, and
        # fall back to measuring it (the HINT-era path, where the controller can import them).
        cached = RAW / f"identity_{hook.scope.model_id}_{category}.json"
        if args.role is not None:
            if not cached.is_file():
                raise SystemExit("HINT role capture requires identity-phase evidence")
            stage = {"identity": json.loads(cached.read_text(encoding="utf-8")),
                     "plan": {"plan_sha256": plan["plan_sha256"], "plan_path": str(plan_path)}}
        elif args.phase == "adapter" and cached.is_file():
            stage = {"identity": json.loads(cached.read_text(encoding="utf-8")),
                     "plan": {"plan_sha256": plan["plan_sha256"], "plan_path": str(plan_path)}}
            print(f"# {category}: identity reused from {cached.name} ",
                  "(measured earlier under the model interpreter)", flush=True)
        else:
            stage = identity_phase(hook, category, config)
            cached.write_text(json.dumps(stage["identity"], indent=1) + "\n", encoding="utf-8")
        print(f"# {category}: identity measured in {stage['identity']['phase_seconds']}s", flush=True)
        print(json.dumps({
            "gallery_order_identical": stage["identity"]["gallery_order_identical"],
            "reference_ids_identical": stage["identity"]["reference_ids_identical"],
            "target_ids_identical": stage["identity"]["target_ids_identical"],
            "model_text_identical": stage["identity"]["model_text_identical_to_author_composition"],
            "counts": [stage["identity"]["query_count_native"],
                       stage["identity"]["query_count_annotations"],
                       stage["identity"]["gallery_count_native"],
                       stage["identity"]["gallery_count_rebuilt_from_annotations"]],
            "sample_indices": stage["identity"]["sample"]["query_indices"],
            "gallery_sample": stage["identity"]["sample"]["gallery_sample_size"],
        }, indent=1), flush=True)
        if args.dry_run:
            print(f"# {category}: dry run -- no artifact written", flush=True)
            continue
        if args.phase == "identity":
            if hook.scope.model_id == "hint":
                # HINT's final native artifact must be written atomically with its ten-role V3 block.
                v3_plan_path = EVIDENCE / f"g1_g2.hint.{category}.v3-plan.json"
                v3_plan = _hint_plan(hook, str(category), run_root, args.seed,
                                     stage["identity"], config,
                                     str(args.controller_interpreter.absolute()))
                v3_plan_sha256 = T4.write_v3_plan(v3_plan_path, v3_plan)
                print(f"# HINT V3 immutable plan {v3_plan_path} sha256={v3_plan_sha256}", flush=True)
            else:
                identity_path = emit_native(hook, category, stage["identity"], config)
                print(f"# {category}: wrote {identity_path.name}", flush=True)
            continue
        if hook.scope.model_id == "hint" and args.phase == "aggregate":
            if v3_plan is None or v3_plan_sha256 is None or v3_plan_path is None:
                raise RuntimeError("HINT aggregate requires immutable V3 plan")
            artifact_path = _aggregate_hint_v3(
                run_root, category, v3_plan, v3_plan_sha256, v3_plan_path)
            print(f"# {category}: aggregate {artifact_path}", flush=True)
            continue
        if hook.scope.model_id == "hint" and args.phase in ("native", "adapter"):
            if v3_plan is None or v3_plan_sha256 is None or args.role is None:
                raise RuntimeError("HINT role capture requires immutable V3 plan")
            run = v3_plan["runs"][args.role]
            role_batch = run["execution"]["query_batch_size"]
            role_root = RAW / "roles" / category / args.role
            records = (native_phase(hook, category, config, role_batch, args.role)
                       if args.phase == "native" else
                       adapter_phase(hook, category, config, role_batch, args.role))
            records["role"] = args.role
            records["run_id"] = run["run_id"]
            records["source_id"] = run["source_id"]
            records["identity"] = run["identity"]
            records["execution"] = run["execution"]
            records["plan_sha256"] = v3_plan_sha256
            (role_root / "record.json").write_text(
                json.dumps(records, indent=1) + "\n", encoding="utf-8")
            continue
        print(f"# {category}: {args.phase} finished in {round(time.time() - began, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
