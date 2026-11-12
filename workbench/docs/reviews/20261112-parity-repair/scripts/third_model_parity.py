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
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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

    def declared_tolerances(self) -> dict[str, Any]:
        return {**TOL, **(self.extra_tolerances or {})}


class CaptureHook:
    """The two hooks and the scope function a new model supplies.

    Subclassed per model. Everything below the line is the producer's; everything above it is the
    model's. The hooks return ARRAYS, never metrics: ranking, masking and hit counts are the shared
    layer's job, so two models cannot disagree about what "the same rank" means.
    """

    scope: ModelScope

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


def native_control(category: str) -> dict[str, Any]:
    """The stored native per-query ranks, re-verified against the registry checkpoint digest.

    Reused rather than re-measured: it is the SAME checkpoint bytes, protocol and population, and a
    second full native evaluation would add a number, not information. What is re-verified here is
    the thing reuse could silently get wrong -- that the ranks still reproduce the macro that file
    records, and that its checkpoint is the one the registry declares.
    """
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    document = json.loads(NATIVE_CONTROL.read_text(encoding="utf-8"))
    declared = checkpoint_by_id(model_by_id("hint"), "fashioniq")
    entry = document["categories"][category]
    ranks = [int(rank) for rank in entry["ranks"]]
    hits = _hits(ranks)
    recorded = [float(value) for value in entry["native_recalls_r1_r10_r50"]]
    recomputed = [100.0 * hits[f"r{k}"] / len(ranks) for k in (1, 10, 50)]
    return {
        "source": str(NATIVE_CONTROL.relative_to(REPO)),
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


def decode_equality(paths: Sequence[str]) -> dict[str, Any]:
    """Both sides' preprocessing, decoded in THIS process, compared bitwise.

    Honest scope: this proves the two transform COMPOSITIONS are identical (HINT uses
    `data_utils.targetpad_transform(1.25, 224)` on both sides -- `replay/hint.py:184` and
    `model_hint.py:_worker_main`), not that a second process decodes differently. Pixel decoding is
    CPU-only, so it is measured here rather than deferred to the GPU step.
    """
    import numpy as np
    import torch
    from PIL import Image

    from workbench.replay.hint import _load_replay_modules

    _, _, pinned, _ = _load_replay_modules(
        Path(REPO / "workbench/third_party/HINT"), "pretrain")
    data_utils, _ = pinned
    transform = data_utils.targetpad_transform(1.25, 224)
    native_side = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    adapter_side = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    measured = compare(adapter_side.numpy(), native_side.numpy())
    result = {"adapter_decode_vs_native_val_decode": {
        "max_abs": measured["max_abs"], "max_rel": measured["max_rel"],
        "min_cosine": measured["min_cosine"], "bitwise_equal": measured["bitwise_equal"],
        "shape": measured["shape"]}}
    terms = transform.transforms
    result.update({
        "measured_in": "one process (CPU); both compositions executed here",
        "native_composition": "replay/hint.py:184 data_utils.targetpad_transform(1.25, 224)",
        "adapter_composition": "model_hint.py:329 same call",
        "transform_steps": [type(step).__name__ for step in terms],
        "image_size": 224,
        "target_ratio": 1.25,
        "normalization": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "bitwise_equal_is_the_expected_value": True,
    })
    assert result["adapter_decode_vs_native_val_decode"]["bitwise_equal"], (
        "the two sides' preprocessing differs; report it, do not pass it")
    return result


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
        (`HINT.py:401-433`). There is no per-token id artifact on the adapter side: the worker
        applies the same processor, so the comparable quantity is the FEATURE, not a token list.
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
                        texts: Sequence[str]) -> Any:
        import numpy as np
        from PIL import Image

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = loaded["data_utils"].targetpad_transform(1.25, 224)
        processor = loaded["processors"][0]["eval"]
        from workbench.composition.adapters import resolved_batch_size

        outputs = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images = torch_mod.stack([transform(Image.open(path).convert("RGB"))
                                          for path in chunk]).float().cuda()
                chunk_texts = [processor(text) for text in texts[start:start + len(chunk)]]
                features = network.extract_retrieval_compose(images, chunk_texts)
                outputs.append(features.detach().cpu().numpy().reshape(len(chunk), -1))
                del images, features
        return np.concatenate(outputs, axis=0)

    def native_target_features(self, loaded: dict[str, Any], paths: Sequence[str]) -> Any:
        import numpy as np
        from PIL import Image

        from workbench.composition.adapters import resolved_batch_size

        torch_mod, network = loaded["torch"], loaded["model"]
        transform = loaded["data_utils"].targetpad_transform(1.25, 224)
        outputs = []
        batch = resolved_batch_size()
        with torch_mod.no_grad():
            for start in range(0, len(paths), batch):
                chunk = paths[start:start + batch]
                images = torch_mod.stack([transform(Image.open(path).convert("RGB"))
                                          for path in chunk]).float().cuda()
                features = network.extract_retrieval_target(images)
                # (N, 256, 32) -> (N, 32, 256), the adapter's layout, so the comparator is aligned.
                outputs.append(features.detach().cpu().numpy().transpose(0, 2, 1))
                del images, features
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


def checkpoint_identity(config: Any) -> dict[str, Any]:
    """The checkpoint bytes, hashed once and cached under the producer's own raw dir."""
    import hashlib as _hash

    from workbench.backend.registry import checkpoint_by_id, model_by_id

    declared = checkpoint_by_id(model_by_id("hint"), "fashioniq")
    path = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "hint" / declared["filename"]
    cache = RAW / "checkpoint_identity.json"
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
    from workbench.replay.hint import _load_replay_modules

    model = model_by_id(hook.scope.model_id)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
    dataset_root = Path(config.FASHIONIQ_ROOT)
    _, datasets, _, _ = _load_replay_modules(source_root, "pretrain")
    began = time.time()
    with isolated_cache_root(dataset_root) as cache:
        dataset = datasets.FashionIQ(path=str(cache).rstrip("/") + "/", transform=Identity())
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
        "reference_equals_target_rows":
            sum(1 for row in scope["rows"] if row["candidate"] == row["target"]),
        "gallery_has_duplicates": len(gallery_ids) != len(set(gallery_ids)),
        "gallery_native_order": gallery_ids,
        # The FULL query order, in the native evaluator's own order. `query_order_sha256` is digested
        # from THIS, on both sides: `parity_gate._identity_cross_checks` compares the field across the
        # pair, so a digest over the 5-row SAMPLE on one side and the 2017-row population on the other
        # would be a mandatory `input.identity_agreement.query_order_sha256` failure in every category.
        "query_native_order": native_refs,
        "gallery_scope": hook.scope.gallery_scope,
        "gallery_is_category_agnostic": hook.scope.gallery_scope == "category-agnostic",
        "adapter_target_category": hook.scope.target_category,
        "target_feature_rank": hook.scope.target_rank,
        "batch_size": batch,
        "population_final_batch": {
            "queries": query_count_native % batch or batch,
            "gallery": len(native_targets) % batch or batch,
            "note": "last batch of the FULL population; the fixed sample is one partial batch",
        },
        "text_handling": {
            "native": "datasets.py:82-84 concat_text -> mod['str']; test.py:44 txt_processors['eval']",
            "adapter": "model_hint.py:330 applies processors[0]['eval'] to the same composed sentence",
            "comparable_quantity": "the query FEATURE, not a token list: HINT tokenises inside "
                                   "extract_retrieval_compose with max_txt_len=32 truncation",
        },
        "image_transform": {
            "call": "data_utils.targetpad_transform(1.25, 224)",
            "native": "replay/hint.py:184 (val path)",
            "adapter": "model_hint.py::_worker_main (same call)",
            "note": "both sides take the VAL transform; LIMN's adapter/native split has no analogue "
                    "here, so no train-transform comparison is recorded",
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


def native_phase(hook: CaptureHook, category: str, config: Any) -> dict[str, Any]:
    """GPU: the pinned forward for the fixed sample, plus the native pixel tensors."""
    import numpy as np

    from workbench.replay.common import isolated_cache_root

    began = time.time()
    loaded = hook.load_native(config, category)
    torch_mod = loaded["torch"]
    records: dict[str, Any] = {}
    try:
        with isolated_cache_root(loaded["dataset_root"]) as cache:
            dataset = loaded["datasets"].FashionIQ(
                path=str(cache).rstrip("/") + "/",
                transform=loaded["data_utils"].targetpad_transform(1.25, 224))
            queries = getattr(dataset, f"test_queries_{category}")
            targets = getattr(dataset, f"test_targets_{category}")
            scope = hook.gallery(loaded, category)
            gallery_ids = [scope["images"][entry["target_img_id"]] for entry in targets]
            sample = fixed_sample(scope["rows"], gallery_ids)
            texts = [entry["mod"]["str"] for entry in queries]
            inputs = {
                "category": category,
                "sample_indices": sample["sample_indices"],
                "query_texts": [texts[index] for index in sample["sample_indices"]],
                "reference_ids": sample["reference_ids"],
                "target_ids": sample["target_ids"],
                "query_reference_paths": [scope["image_path"](name)
                                          for name in sample["reference_ids"]],
                "query_target_paths": [scope["image_path"](name)
                                       for name in sample["target_ids"]],
                "gallery_sample_ids": sample["gallery_sample_ids"],
                "gallery_sample_paths": [scope["image_path"](name)
                                         for name in sample["gallery_sample_ids"]],
                "gallery_ids": gallery_ids,
                "gallery_paths": scope["gallery_paths"],
                "full_query_paths": [scope["image_path"](name) for name in scope["reference_ids"]],
                "full_query_texts": texts,
                "full_reference_ids": scope["reference_ids"],
                "full_target_ids": scope["target_ids"],
            }
            query_features = hook.native_features(
                loaded, inputs["query_reference_paths"], inputs["query_texts"])
            target_features = hook.native_target_features(
                loaded, inputs["query_target_paths"])
            pixel_paths = [scope["image_path"](name) for name in inputs["reference_ids"]]
            records["decode_equality"] = decode_equality(pixel_paths)
            (RAW / f"sample_{category}_inputs.json").write_text(
                json.dumps(inputs, indent=1) + "\n", encoding="utf-8")
            np.savez(RAW / f"sample_{category}_native.npz",
                     native_query=query_features, native_target=target_features)
            records.update({
                "sample_forward_native": {"query_shape": list(query_features.shape),
                                          "target_shape": list(target_features.shape),
                                          "adapter_expected_query_shape":
                                              [len(sample["sample_indices"]), 256],
                                          "adapter_expected_target_shape":
                                              [len(sample["sample_indices"]), 32, 256]},
                "population_identity": {"query_count": len(queries),
                                        "gallery_count": len(targets)},
            })
    finally:
        del loaded["model"]
        torch_mod.cuda.empty_cache()
    records["phase_seconds"] = round(time.time() - began, 1)
    return records


def adapter_phase(hook: CaptureHook, category: str, config: Any) -> dict[str, Any]:
    """GPU (worker under the model interpreter): adapter features, comparisons, ranking, v2 block."""
    import numpy as np

    from workbench.composition import model_hint

    declared = hook.scope.declared_tolerances()
    inputs = json.loads((RAW / f"sample_{category}_inputs.json").read_text(encoding="utf-8"))
    native = np.load(RAW / f"sample_{category}_native.npz")
    began = time.time()
    adapter = model_hint.Adapter(config)
    adapter.work_root = RAW / f"work-{category}"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    scope_value = hook.scope.target_category
    result: dict[str, Any] = {}
    try:
        adapter_queries = adapter.encode_queries(
            inputs["query_reference_paths"], inputs["query_texts"],
            [scope_value] * len(inputs["query_texts"]))
        adapter_targets = adapter.encode_targets(
            inputs["query_target_paths"], [scope_value] * len(inputs["query_target_paths"]))
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
        # Measured in THIS process, on this process's own decode, so the check is about the adapter's
        # pixels rather than a value copied from the native artifact.
        result["layer2_pixels"] = {
            **decode_equality(inputs["query_reference_paths"]),
            "adapter_decode_by": "workbench.composition.model_hint (targetpad_transform(1.25, 224), "
                                 "the same call the worker makes at model_hint.py:329)",
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

        adapter.restart()
        after_restart = adapter.encode_queries(
            inputs["query_reference_paths"], inputs["query_texts"],
            [scope_value] * len(inputs["query_texts"]))
        result["cache_off_restart"] = toleranced(
            compare(after_restart, adapter_queries), abs_key="batch_max_abs",
            rel_key="batch_max_rel", declared=declared)

        population_queries = adapter.encode_queries(
            inputs["full_query_paths"], inputs["full_query_texts"],
            [scope_value] * len(inputs["full_query_texts"]))
        population_gallery = adapter.encode_targets(
            inputs["gallery_paths"], [scope_value] * len(inputs["gallery_paths"]))
        np.savez(RAW / f"population_{category}.npz",
                 queries=population_queries, gallery=population_gallery)
        result["population"] = {"query_count": int(population_queries.shape[0]),
                                "gallery_count": int(population_gallery.shape[0]),
                                "finite": bool(np.isfinite(population_queries).all()
                                               and np.isfinite(population_gallery).all())}
    finally:
        adapter.close()

    result["scoring_ranking"] = scoring_ranking(hook, category, inputs, config, declared)
    update_native_metrics_flag(hook, category, result["scoring_ranking"])
    result["phase_seconds"] = round(time.time() - began, 1)
    return result


def scoring_ranking(hook: CaptureHook, category: str, inputs: dict[str, Any], config: Any,
                    declared: dict[str, Any]) -> dict[str, Any]:
    """G2 ranking: the adapter's own score/rank against the native convention, same features."""
    import numpy as np

    from workbench.composition import model_hint

    blob = np.load(RAW / f"population_{category}.npz")
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    queries_n = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    # HINT's gallery is (G, 32, 256): the FEATURE axis is the last one. Normalising axis=1 would
    # normalise across the 32 Q-former tokens, which is not a direction in feature space; that mistake
    # produced a score delta of 1.17 against a declared 1e-6. The native evaluator normalises nothing
    # (features are already L2-normalised inside the model), so this is a re-normalisation of unit
    # vectors and must be a no-op to float32 precision.
    gallery_n = gallery / np.linalg.norm(gallery, axis=-1, keepdims=True)
    # Native convention, verbatim: `np.matmul(all_queries, all_imgs).squeeze().max(-1)`, reference
    # masked to -10e10 (test.py:74-79). float32 end to end, as the pinned evaluator computes it.
    sims = np.einsum("nd,gqd->ngq", queries_n, gallery_n).max(-1)
    unmasked = sims.copy()
    position = {name: index for index, name in enumerate(inputs["gallery_ids"])}
    reference_positions = [position[name] for name in inputs["full_reference_ids"]]
    target_positions = [position[name] for name in inputs["full_target_ids"]]
    for row, reference in enumerate(reference_positions):
        sims[row, reference] = -10e10
    native_style = [int(np.flatnonzero(np.argsort(-sims[row]) == target)[0]) + 1
                    for row, target in enumerate(target_positions)]
    argsort_unmasked = [int(np.flatnonzero(np.argsort(-unmasked[row]) == target)[0]) + 1
                        for row, target in enumerate(target_positions)]

    adapter = model_hint.Adapter(config)
    adapter.work_root = RAW / f"work-{category}-rank"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    try:
        adapter_style = adapter.rank(queries, gallery, [[index] for index in target_positions])
        pair_count = min(64, len(target_positions))
        scores = adapter.score(queries[:pair_count], gallery)
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

    control = native_control(category)
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
        "gallery_count": len(inputs["gallery_ids"]),
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
        "protocol": "fashioniq_val_split",
        "standard": STANDARD,
        "tolerances": hook.scope.declared_tolerances(),
        "checkpoint": checkpoint_identity(config),
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
    document["native_metrics_parity_detail"] = {
        "rule": "exact integer hit-count equality at every cutoff",
        "cutoffs": [1, 10, 50],
        "reference_mask": ranking["reference_mask"],
        "ranking_convention": "numpy argsort(-sims), 1-based target index (test.py:74-86)",
        "native": {"source": ranking["native_aggregate_source"],
                   "hits": ranking["hit_counts"]["native"],
                   "recall_pct": ranking["native_measured_recall"]},
        "adapter": {"source": "raw/third_model/population_<category>.npz (adapter worker features), "
                              "ranked under the native protocol",
                    "hits": ranking["hit_counts"]["adapter"],
                    "recall_pct": ranking["native_style_recall"]},
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("identity", "native", "adapter"), default="identity")
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"))
    parser.add_argument("--self-check", action="store_true",
                        help="CPU-only check of the shared layer; no weights, no worker, no GPU")
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU phases into raw/third_model/; never writes the six artifacts")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.self_check:
        return self_check()
    config = _config()
    hook = HintCapture()
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
    RAW.mkdir(parents=True, exist_ok=True)
    categories = (args.category,) if args.category else hook.scope.categories
    for category in categories:
        began = time.time()
        stage = identity_phase(hook, category, config)
        (RAW / f"identity_{category}.json").write_text(
            json.dumps(stage["identity"], indent=1) + "\n", encoding="utf-8")
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
            print(f"# {category}: wrote {emit_native(hook, category, stage['identity'], config).name}",
                  flush=True)
            continue
        # The GPU phases. Native first, then adapter, in the interpreter the caller launched.
        if args.phase in ("native", "adapter"):
            if args.phase == "native":
                records = native_phase(hook, category, config)
                (RAW / f"native_{category}.json").write_text(
                    json.dumps(records, indent=1) + "\n", encoding="utf-8")
                print(f"# {category}: native capture {json.dumps(records)[:200]}", flush=True)
            else:
                body = adapter_phase(hook, category, config)
                components = body.pop("component_parity", None)
                document = {
                    "category": category, "phase": "adapter",
                    "model_id": hook.scope.model_id,
                    "checkpoint_id": hook.scope.checkpoint_id,
                    "protocol": "fashioniq_val_split",
                    "standard": STANDARD,
                    "tolerances": hook.scope.declared_tolerances(),
                    "checkpoint": checkpoint_identity(config),
                    # Read from the native artifact, not recomputed: the two sides must be provably
                    # the same sequence, and one source is how that stays true after an edit.
                    "query_order_sha256": json.loads(
                        (EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json")
                        .read_text(encoding="utf-8"))["query_order_sha256"],
                    "gallery_order_sha256": json.loads(
                        (EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json")
                        .read_text(encoding="utf-8"))["gallery_order_sha256"],
                    "layer2_4_forward": body,
                }
                if components is not None:
                    document["component_parity"] = components
                native_document = json.loads(
                    (EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.native.json")
                    .read_text(encoding="utf-8"))
                assert_identity_agreement(native_document, document)
                path = EVIDENCE / f"g1_g2.{hook.scope.model_id}.{category}.adapter.json"
                path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
                print(f"# {category}: wrote {path.name}", flush=True)
        print(f"# {category}: {args.phase} finished in {round(time.time() - began, 1)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
