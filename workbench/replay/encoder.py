"""Source-faithful replay for the ENCODER FashionIQ evaluator.

Why a wrapper exists: the pinned `test.py` defines `test(params, model, testset, category)`
and nothing else — no CLI, no `__main__`, and no checkpoint loader. This module supplies the
missing orchestration shell and calls pinned `test.test(params, model, testset, category)`
without altering ranking, metric arithmetic, gallery construction, or feature extraction.

Exact attributes read by pinned `test.py:6-83`:
- `params.local_rank`: int (lines 26, 43; disables tqdm progress bar when nonzero)
- `params.batch_size`: int (lines 29, 45; batch accumulation threshold)
- `model.eval()`: method (line 7)
- `model.extract_retrieval_compose(imgs, mods)`: method (line 33)
- `model.extract_retrieval_target(imgs)`: method (line 49)
- `testset.test_queries_<cat>` and `testset.test_targets_<cat>`: lists (lines 10-15)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
# ONE constant, imported rather than copied: `Adapter.prepare_identity` pins the composition worker
# with this seed and the native path below pins its own construction with it, so the two can never
# drift into measuring different relations. (No cycle: `composition.model_encoder` imports replay
# only inside `_worker_main`, after the module is loaded.)
from workbench.composition.model_encoder import RELATION_EXPECTED_DIGEST, RELATION_SEED
from workbench.replay.common import (
    OBSERVATION_OUT_ENV,
    FeatureRecorder,
    ReplayResult,
    enable_deterministic_inference,
    load_trusted_whole_model,
    run_cache_root,
    macro_aggregate,
    seeded_dataset_construction,
    sha256_file,
    source_pin_status,
)
from workbench.replay.tensor_spool import disk_backed_images

SCORE_OBSERVATION_ENV = "WORKBENCH_ENCODER_SCORE_OBSERVATION_OUT"


G1_OBSERVATION_ENV = "WORKBENCH_ENCODER_G1_OBSERVATION_OUT"


class _NativeArgsortCapture:
    """Capture score rows/order passed through the pinned ENCODER evaluator."""

    def __init__(self, np_module: Any, destination: Path, reference_positions: list[int],
                 target_positions: list[int], gallery_ids: list[str], reference_ids: list[str],
                 target_ids: list[str]) -> None:
        self._np = np_module
        self._argsort = np_module.argsort
        self.destination = destination
        destination.mkdir(parents=True, exist_ok=True)
        self.query_count = len(target_positions)
        self.gallery_ids = gallery_ids
        self.reference_ids = reference_ids
        self.target_ids = target_ids
        self.reference_positions = reference_positions
        self.target_positions = target_positions
        self.query_texts: list[str] = []
        self.query_paths: list[str] = []
        self.gallery_paths: list[str] = []
        self.image_manifest_paths: list[str] = []
        self.image_sha256: dict[str, str] = {}
        self.scores = np_module.lib.format.open_memmap(
            destination / "native_scores.float32.npy", mode="w+", dtype=np_module.float32,
            shape=(self.query_count, len(gallery_ids)))
        self.orders = np_module.lib.format.open_memmap(
            destination / "native_orders.int32.npy", mode="w+", dtype=np_module.int32,
            shape=(self.query_count, len(gallery_ids)))
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._np, name)

    def argsort(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        row = self._np.asarray(values)
        if row.ndim != 1 or self.calls >= self.query_count:
            raise ValueError("unexpected ENCODER argsort shape or call count")
        scores = -row
        if scores.dtype != self._np.float32 or scores.shape != (len(self.gallery_ids),):
            raise ValueError("unexpected ENCODER source score dtype or gallery shape")
        if not self._np.isfinite(scores).all():
            raise ValueError(f"non-finite ENCODER source score row {self.calls}")
        reference = self.reference_positions[self.calls]
        if scores[reference] != self._np.asarray(-10e10, dtype=scores.dtype).item():
            raise ValueError(f"ENCODER source reference mask missing at query {self.calls}")
        order = self._argsort(values, *args, **kwargs)
        self.scores[self.calls] = scores
        self.orders[self.calls] = order
        self.calls += 1
        return order

    def finish(self, category: str, recalls: dict[str, float]) -> dict[str, Any]:
        if self.calls != self.query_count:
            raise ValueError(f"ENCODER source argsort calls {self.calls} != {self.query_count}")
        self.scores.flush()
        self.orders.flush()
        score_sha = sha256_file(self.destination / "native_scores.float32.npy")
        order_sha = sha256_file(self.destination / "native_orders.int32.npy")
        self.scores_sha256 = score_sha
        ranks = self._np.empty(self.query_count, dtype=self._np.int32)
        for index, target in enumerate(self.target_positions):
            ranks[index] = self._np.flatnonzero(self.orders[index] == target)[0] + 1
        hits = {str(k): int((ranks <= k).sum()) for k in (1, 10, 50)}
        for k, count in hits.items():
            observed = float(recalls[f"{category}_r{k}"])
            if observed != 100.0 * count / self.query_count:
                raise ValueError(f"ENCODER source hit count mismatch R@{k}")
        self._np.savez(
            self.destination / "source_ids.npz",
            category=self._np.asarray(category),
            gallery_ids=self._np.asarray(self.gallery_ids),
            reference_ids=self._np.asarray(self.reference_ids),
            target_ids=self._np.asarray(self.target_ids),
            query_texts=self._np.asarray(self.query_texts),
            query_paths=self._np.asarray(self.query_paths),
            gallery_paths=self._np.asarray(self.gallery_paths),
            reference_positions=self._np.asarray(self.reference_positions),
            target_positions=self._np.asarray(self.target_positions),
            image_manifest_paths=self._np.asarray(self.image_manifest_paths),
            image_manifest_sha256=self._np.asarray([self.image_sha256[path] for path in self.image_manifest_paths]))
        ids_sha = sha256_file(self.destination / "source_ids.npz")
        return {
            "status": "captured",
            "score_path": str(self.destination / "native_scores.float32.npy"),
            "score_shape": list(self.scores.shape), "score_dtype": "float32",
            "score_sha256": score_sha,
            "order_path": str(self.destination / "native_orders.int32.npy"),
            "order_shape": list(self.orders.shape), "order_dtype": "int32",
            "order_sha256": order_sha,
            "ids_path": str(self.destination / "source_ids.npz"),
            "ids_sha256": ids_sha,
            "target_ranks": ranks.tolist(), "hit_counts": hits,
        }


def _compare_adapter_scores(capture: _NativeArgsortCapture, adapter_scores: Any,
                            destination: Path) -> dict[str, Any]:
    import numpy as np

    scores = np.asarray(adapter_scores)
    shape = tuple(capture.scores.shape)
    if scores.dtype != np.float32:
        raise ValueError(f"ENCODER Adapter score dtype {scores.dtype} != float32")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite ENCODER Adapter score matrix")
    if scores.shape != shape:
        raise ValueError(f"ENCODER Adapter scores shape {scores.shape} != native {shape}")
    destination.mkdir(parents=True, exist_ok=True)
    adapter_scores_map = np.lib.format.open_memmap(
        destination / "adapter_scores.float32.npy", mode="w+", dtype=np.float32, shape=shape)
    adapter_orders = np.lib.format.open_memmap(
        destination / "adapter_orders.int32.npy", mode="w+", dtype=np.int32, shape=shape)
    adapter_ranks = np.empty(shape[0], dtype=np.int32)
    source_ranks = np.empty(shape[0], dtype=np.int32)
    topk_hit_flips = {str(k): [] for k in (1, 10, 50)}
    candidate_topk_flips = {str(k): [] for k in (1, 10, 50)}
    max_delta = 0.0
    rank_flips = []
    for row in range(shape[0]):
        source = capture.scores[row]
        adapter = scores[row].copy()
        reference = capture.reference_positions[row]
        if source[reference] != np.asarray(-10e10, dtype=source.dtype).item():
            raise ValueError(f"ENCODER native reference mask changed at query {row}")
        adapter[reference] = np.asarray(-10e10, dtype=adapter.dtype).item()
        order = np.argsort(-adapter)
        adapter_scores_map[row] = adapter
        adapter_orders[row] = order
        target = capture.target_positions[row]
        source_rank = int(np.flatnonzero(capture.orders[row] == target)[0]) + 1
        adapter_rank = int(np.flatnonzero(order == target)[0]) + 1
        source_ranks[row] = source_rank
        adapter_ranks[row] = adapter_rank
        if source_rank != adapter_rank:
            rank_flips.append(row)
        for k in candidate_topk_flips:
            limit = int(k)
            if set(capture.orders[row, :limit]) != set(order[:limit]):
                candidate_topk_flips[k].append(row)
        max_delta = max(max_delta, float(np.max(np.abs(source - adapter))))
        for k in topk_hit_flips:
            if (source_rank <= int(k)) != (adapter_rank <= int(k)):
                topk_hit_flips[k].append(row)
    adapter_scores_map.flush()
    adapter_orders.flush()
    source_hits = {str(k): int((source_ranks <= k).sum()) for k in (1, 10, 50)}
    adapter_hits = {str(k): int((adapter_ranks <= k).sum()) for k in (1, 10, 50)}
    return {
        "status": "measured",
        "score_path": str(destination / "adapter_scores.float32.npy"),
        "score_sha256": sha256_file(destination / "adapter_scores.float32.npy"),
        "order_path": str(destination / "adapter_orders.int32.npy"),
        "order_sha256": sha256_file(destination / "adapter_orders.int32.npy"),
        "score_max_abs_delta": max_delta,
        "source_hits": source_hits, "adapter_hits": adapter_hits,
        "hit_counts_equal": source_hits == adapter_hits,
        "diagnostic_aggregate_status": "hit_count_difference" if source_hits != adapter_hits else "hit_counts_equal",
        "score_max_abs_delta": max_delta,
        "all_target_ranks_equal": not rank_flips,
        "all_topk_memberships_equal": not any(candidate_topk_flips.values()),
        "source_target_ranks": source_ranks.tolist(),
        "adapter_target_ranks": adapter_ranks.tolist(),
        "source_target_topk_hits": {str(k): (source_ranks <= k).tolist() for k in (1, 10, 50)},
        "adapter_target_topk_hits": {str(k): (adapter_ranks <= k).tolist() for k in (1, 10, 50)},
        "target_rank_flip_count": len(rank_flips), "target_rank_flip_query_indices": rank_flips,
        "target_topk_hit_flip_query_indices": topk_hit_flips,
        "candidate_topk_membership_flip_query_indices": candidate_topk_flips,
    }


MODEL = model_by_id("encoder")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
DEFAULT_CHECKPOINT = MODEL["checkpoint_variants"][0]
PROTOCOL = "fashioniq_val_split"
AUXILIARY_ASSET_NAME = "open_clip_pytorch_model.bin"


#: Checkpoint entries the released ENCODER artifact carries but the pinned model does not declare.
#: Evidence, all reproducible (see `docs/reviews/R16_ENCODER_KEY_COMPAT.md`):
#:   * `BindingDecoder.__init__` declares no `fc` attribute at all (`model_try2.py:89-116`), and no
#:     forward path in the pinned checkout reads `binding_decoder.fc` -- a whole-tree search finds
#:     no reference;
#:   * loading the real checkpoint reports `missing_keys: []`, so every parameter the model needs
#:     received a value, and all 415 consumed tensors match their parameter's shape;
#:   * perturbing these two tensors leaves `BindingDecoder.binding` output bit-identical
#:     (max absolute difference 0.0).
#: These are therefore weights for a layer that does not exist in this revision, which is a
#: statement about the artifacts rather than about any score.
_UNUSED_CHECKPOINT_KEYS = frozenset({
    "backbone.binding_decoder.fc.0.weight",
    "backbone.binding_decoder.fc.0.bias",
})

#: The released checkpoint those keys were observed in. They are not permitted for any other
#: artifact: a different file could carry them for a different reason.
_UNUSED_KEYS_CHECKPOINT_SHA = "0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173"


def _unused_checkpoint_keys(encoder_model, variant: dict[str, Any], source_root: Path) -> frozenset[str]:
    """Checkpoint entries this model may ignore, or nothing.

    An extra tensor is only tolerable when all four of these hold, because each one rules out a
    different way the allowance could hide a real problem:

    * the artifact is the exact released file this was observed in, by its registered digest -- so
      an unrelated or tampered checkpoint cannot borrow the allowance;
    * the registry's own mapping for that variant is verified metadata, not an assumption;
    * the pinned source really is the pinned, clean checkout, so the "the model does not read it"
      claim is about the code that will actually run;
    * the module does not declare those names, so they are genuinely unconsumed rather than being
      silently dropped while a real parameter goes unloaded.

    Everything else keeps the strict behaviour: a missing parameter, a shape mismatch, or any other
    unexpected key still refuses the run.
    """
    if variant.get("expected_sha256") != _UNUSED_KEYS_CHECKPOINT_SHA:
        return frozenset()
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        return frozenset()
    ok, _reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        return frozenset()
    declared = set(encoder_model.state_dict())
    if not _UNUSED_CHECKPOINT_KEYS.isdisjoint(declared):
        # The module declares one of them: then it is not an unused extra, it is a parameter whose
        # value this run would silently drop, and the strict path must refuse it.
        return frozenset()
    return _UNUSED_CHECKPOINT_KEYS


def validate_checkpoint(checkpoint_root: Path, checkpoint_id: str | None = None) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(checkpoint_id) if checkpoint_id else DEFAULT_CHECKPOINT
    if variant is None:
        return [f"ENCODER checkpoint variant unknown: {checkpoint_id}"]
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"ENCODER checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"ENCODER checkpoint is an empty placeholder: {path}")
        return blockers
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(
            f"ENCODER checkpoint mapping is unverified: {variant['filename']} has status "
            f"{variant.get('checkpoint_mapping_status')}; author Drive mapping unverified"
        )
    expected_sha = variant.get("expected_sha256")
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(
                f"ENCODER checkpoint sha256 mismatch: expected {expected_sha}, found {actual_sha}"
            )
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"ENCODER source pin invalid: {reason}"]


def validate_auxiliary_assets(source_root: Path) -> list[str]:
    blockers: list[str] = []
    asset_path = source_root / AUXILIARY_ASSET_NAME
    if not asset_path.is_file() or asset_path.stat().st_size == 0:
        blockers.append(
            f"ENCODER auxiliary asset missing: LAION CLIP ViT-B/32 open_clip weights required at {asset_path}"
        )
    return blockers


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"ENCODER dataset input missing: {dataset_root / relative}")
        cat_resized = dataset_root / "resized_image" / category
        if not cat_resized.is_dir():
            blockers.append(f"ENCODER dataset input missing: {cat_resized}")
        # `datasets.py` builds a FRESH cache with `train_init_process`, which opens
        # `cap.<category>.train.json` and the correction dictionary for all three categories BEFORE
        # reading any validation file (`train_init_process` at :59-73, called from `__init__`).
        # `isolated_cache_root` gives that constructor an empty temp root every run, so a cache in the
        # canonical root does NOT satisfy it: without the train captions the constructor raises
        # FileNotFoundError at dataset construction, long after an expensive model load.
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/correction_dict_{category}.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"{'ENCODER'} dataset input missing: {dataset_root / relative}")
    # The evaluated subset and the constructor's input set are different questions:
    # `datasets.py:95-107` (the module `datasets1.py` restores) runs `train_init_process` and then
    # `get_test_data` for all three categories, with no category argument in its signature.
    from workbench.replay.common import constructor_input_blockers

    return list(dict.fromkeys(
        blockers + constructor_input_blockers(
            dataset_root, lambda path: f"ENCODER dataset input missing: {path}")))


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    checkpoint_id: str | None = None,
) -> list[str]:
    blockers = (validate_source(source_root) + validate_auxiliary_assets(source_root)
                + validate_checkpoint(checkpoint_root, checkpoint_id)
                + validate_dataset(dataset_root, categories))
    return blockers


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    import open_clip
    import torch

    source_str = str(source_root)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    datasets = __import__("datasets")
    test_module = __import__("test")
    model_try2 = __import__("model_try2")
    return torch, open_clip, datasets, test_module, model_try2



def _assert_checkpoint_loaded(encoder_model, report, variant: dict[str, Any], source_root: Path) -> None:
    """Refuse a partially-loaded model; allow only the entries known to be unused, and say so.

    Split out of `evaluate_categories` so the decision is directly testable: a regression that only
    exercised `_unused_checkpoint_keys` would still pass if the loader stopped calling it.
    """
    missing = [key for key in getattr(report, "missing_keys", [])]
    permitted = _unused_checkpoint_keys(encoder_model, variant, source_root)
    unexpected = [key for key in getattr(report, "unexpected_keys", []) if key not in permitted]
    ignored = sorted(key for key in getattr(report, "unexpected_keys", []) if key in permitted)
    if ignored:
        # Recorded, never silent: a reader must be able to see which tensors were skipped and why,
        # the same way the completion proof records the rest of the run's provenance.
        print("[INFO] ENCODER checkpoint keys present but unused by the pinned model: "
              f"{ignored} (justified in _unused_checkpoint_keys; every declared parameter loaded)",
              file=sys.stderr)
    if missing or unexpected:
        raise ValueError(
            "ENCODER checkpoint does not match the model: "
            f"{len(missing)} missing parameter(s) (first: {missing[0] if missing else '-'}), "
            f"{len(unexpected)} unexpected key(s) (first: {unexpected[0] if unexpected else '-'}). "
            "Refusing to evaluate a partially-loaded model.")

def _relation_record(encoder_model: Any, destination: Path | None, *,
                     drawn: bool = True) -> dict[str, Any]:
    """Persist the relation tensor the native construction just drew, and return its record.

    `learnable_relation` is not in `state_dict()`, so these bytes plus their digest are the only
    record of the state this category's query AND gallery features were both computed with. Written
    beside the run's output when the parent named one (`WORKBENCH_OBSERVATION_OUT`), because a record
    inside a temp tree is deleted at the end of the run that needs it afterwards.
    """
    import hashlib

    import numpy as np

    relation = getattr(getattr(getattr(encoder_model, "backbone", None), "binding_decoder", None),
                       "learnable_relation", None)
    if relation is None:
        # Honest absence, in the same vocabulary the composition side uses: this object exposes no
        # relation tensor, so nothing about a relation is claimed (and the parity gate reports
        # `not_recorded` rather than a pass). The pinned `Encoder` always declares it, so this path is
        # reachable only for an object shaped like one but not one.
        return {"status": "not_recorded",
                "reason": "the constructed model exposes no `backbone.binding_decoder."
                          "learnable_relation`, so no relation tensor was drawn or read here"}
    relation = relation.detach().float().cpu().contiguous()
    digest = hashlib.sha256(relation.numpy().tobytes()).hexdigest()
    record = {
        "status": "recorded",
        "seed": RELATION_SEED if drawn else None,
        "sha256": digest,
        "expected_sha256": RELATION_EXPECTED_DIGEST,
        "matches_pinned_digest": digest == RELATION_EXPECTED_DIGEST,
        "in_state_dict": False,
        "seed_placement": ("torch.manual_seed(seed) immediately before Encoder(...) construction"
                           if drawn else
                           "not applicable: this checkpoint is a whole pickled model, so the tensor "
                           "came from that file rather than from a seeded `randn`"),
        # Says exactly what the pin is: a declared policy, because the released checkpoint carries no
        # relation and the authors' draw is not recoverable from any artifact.
        "policy": ("declared pin, not a restoration: the released checkpoint does not contain "
                   "`learnable_relation`, so the drawn tensor is reproducible under this seed and is "
                   "not claimed to be the authors'"),
        "tensor": "backbone.binding_decoder.learnable_relation",
        "shape": list(relation.shape), "dtype": str(relation.dtype),
        "path": None,
    }
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"relation-seed{RELATION_SEED}.npy"
        np.save(path, relation.numpy(), allow_pickle=False)
        Path(str(path) + ".json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        record["path"] = str(path)
    return record


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    checkpoint_id: str | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, open_clip, datasets, test_module, model_try2 = _load_replay_modules(source_root)

    variant = CHECKPOINTS.get(checkpoint_id) if checkpoint_id else DEFAULT_CHECKPOINT
    if variant is None:
        raise ValueError(f"unknown checkpoint variant: {checkpoint_id}")
    checkpoint_path = checkpoint_root / variant["filename"]

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # The relation pin, at the construction site and immediately before it.
    #
    # POLICY, not restoration: `BindingDecoder.learnable_relation` is drawn by
    # `nn.Parameter(torch.randn(N_p, dim)).cuda()` inside `Encoder.__init__`
    # (`model_try2.py:93`). `.cuda()` on a Parameter returns a plain Tensor, so the attribute is not
    # registered, is absent from `state_dict()` and from the released 417-key checkpoint, and NO
    # checkpoint contains it -- the tensor the authors evaluated with is not recoverable from any
    # artifact. This pin (the SAME `RELATION_SEED` the CIRCUS worker uses, imported so the two cannot
    # drift) is therefore a declared policy that makes the draw reproducible and comparable, and the
    # drawn bytes plus their digest are written beside the run's output. Without it the native path
    # drew from an unpinned global RNG, so the historical numbers could not be reproduced from the
    # code that produced them.
    torch.manual_seed(RELATION_SEED)
    # Hyperparameters matching evaluate_model.py defaults for FashionIQ: {P=3, wc=2, N_p=2}
    encoder_model = model_try2.Encoder(
        hidden_dim=512,
        dropout=0.5,
        local_token_num=3,
        t=0.1,
        wc=2,
        N_p=2,
        weighted=True,
    )

    ckpt_data = torch.load(checkpoint_path, map_location="cpu")
    drawn = True
    if isinstance(ckpt_data, dict):
        # `strict=False` is upstream's choice and it is kept, but its RESULT is no longer discarded:
        # a checkpoint wrapped in the wrong schema loads nothing and `load_state_dict` returns
        # normally, so the wrapper used to evaluate a randomly-initialised model and print plausible
        # Recall. Two checks make that impossible to miss:
        #
        #   * the module's own parameter set must be covered. The author's file is a flat
        #     `OrderedDict` of `loss_T`/`local_weight`/`backbone.*` entries (verified from the
        #     checkpoint's `fashioniq/data.pkl` header), i.e. a full state_dict, so every parameter
        #     the model declares should be present.
        #   * entries the module does not declare are refused, except where the released artifact is
        #     known to carry them AND the pinned source provably never reads them (see
        #     `_unused_checkpoint_keys`).
        report = encoder_model.load_state_dict(ckpt_data, strict=False)
        _assert_checkpoint_loaded(encoder_model, report, variant, source_root)
    else:
        encoder_model = ckpt_data
        drawn = False

    del ckpt_data
    encoder_model = encoder_model.to(target_device)
    encoder_model.eval()
    # The per-query export may recompute a category's features when the aggregate did not capture
    # them, and must agree to the hit count, so the forward pass is pinned as reproducible as CUDA
    # permits rather than merely equivalent.
    enable_deterministic_inference(torch)

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(source_root / AUXILIARY_ASSET_NAME),
    )
    del clip_model
    transform = [preprocess_train, preprocess_val]

    results: list[ReplayResult] = []
    # When the parent asked for an observation, record the features of THIS evaluation. The
    # per-query export then ranks these instead of running a second pass, which cannot reproduce
    # this one (measured: two processes, same seed and deterministic kernels, different feature
    # hashes).
    observation_out = os.environ.get(OBSERVATION_OUT_ENV, "").strip()
    # The drawn tensor, recorded from the CONSTRUCTED model and written beside the run's output (the
    # per-query payload's own directory) so it outlives the temp tree. `drawn` is False when the
    # checkpoint was a whole pickled model: then this tensor came from that file, not from `randn`.
    relation_destination = Path(os.environ[SCORE_OBSERVATION_ENV]).parent         if os.environ.get(SCORE_OBSERVATION_ENV, "").strip() else         (Path(observation_out).parent if observation_out else None)
    relation_state = _relation_record(encoder_model, relation_destination, drawn=drawn)
    captured_features: dict[str, tuple] = {}
    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        # The query inputs are baked in HERE, through the stochastic training transform, so this
        # construction must be reproducible for the per-query export -- which rebuilds the dataset in
        # its own process -- to observe the same inputs. See `seeded_dataset_construction`.
        with disk_backed_images(dataset_root, cache_root, datasets) as tensor_storage:
            with seeded_dataset_construction():
                dataset = datasets.FashionIQ(path=root_str, transform=transform, split="val-split")
            params = SimpleNamespace(batch_size=batch_size, local_rank=0)
            score_observation_out = os.environ.get(SCORE_OBSERVATION_ENV, "").strip()
            if score_observation_out and len(categories) != 1:
                raise ValueError("ENCODER source score capture requires exactly one category")
            if score_observation_out and variant["checkpoint_id"] != DEFAULT_CHECKPOINT["checkpoint_id"]:
                raise ValueError("ENCODER source score capture currently requires the Adapter's default checkpoint")
            for category in categories:
                source_capture = None
                original_numpy = None
                if score_observation_out:
                    import numpy as np

                    queries = getattr(dataset, f"test_queries_{category}")
                    targets = getattr(dataset, f"test_targets_{category}")
                    image_ids = json.loads((dataset_root / "image_splits" /
                                            f"split.{category}.val.json").read_text(encoding="utf-8"))
                    gallery_ids = [image_ids[row["target_img_id"]] for row in targets]
                    gallery_positions = {image_id: index for index, image_id in enumerate(gallery_ids)}
                    reference_positions = [gallery_positions[image_ids[row["source_img_id"]]] for row in queries]
                    target_positions = [gallery_positions[image_ids[row["target_img_id"]]] for row in queries]
                    capture_root = Path(score_observation_out)
                    if capture_root.exists() and (not capture_root.is_dir() or any(capture_root.iterdir())):
                        raise FileExistsError(f"ENCODER capture output is not empty: {capture_root}")
                    source_capture = _NativeArgsortCapture(
                        np, capture_root, reference_positions, target_positions, gallery_ids,
                        [image_ids[row["source_img_id"]] for row in queries],
                        [image_ids[row["target_img_id"]] for row in queries])
                    source_capture.query_texts = [row["mod"]["str"] for row in queries]
                    image_root = Path(dataset.image_dir).resolve() / category
                    source_capture.query_paths = [str((image_root / f"{image_ids[row['source_img_id']]}.jpg").resolve())
                                                  for row in queries]
                    source_capture.gallery_paths = [str((image_root / f"{image_id}.jpg").resolve())
                                                    for image_id in gallery_ids]
                    source_capture.image_manifest_paths = sorted(set(source_capture.query_paths)
                                                                 | set(source_capture.gallery_paths))
                    if any(not Path(path).is_file() for path in source_capture.image_manifest_paths):
                        raise FileNotFoundError("ENCODER capture image disappeared before digest")
                    source_capture.image_sha256 = {path: sha256_file(Path(path))
                                                   for path in source_capture.image_manifest_paths}
                    if any(not Path(path).is_file() for path in source_capture.image_manifest_paths):
                        raise FileNotFoundError("ENCODER capture image disappeared before digest")
                    original_numpy = test_module.np
                    test_module.np = source_capture
                g1_root = os.environ.get(G1_OBSERVATION_ENV, "").strip()
                query_calls = target_calls = 0
                original_tokenizer: Any = None
                original_compose: Any = None
                original_target: Any = None
                capture_source: Any = source_capture
                token_batches: list[Any] = []
                if g1_root:
                    if source_capture is None:
                        raise ValueError("ENCODER G1 capture requires source score identity capture")
                    g1_root_path = Path(g1_root)
                    if g1_root_path.exists() and any(g1_root_path.iterdir()):
                        raise FileExistsError(f"ENCODER G1 output is not empty: {g1_root_path}")
                    g1_root_path.mkdir(parents=True, exist_ok=True)
                    tokenizer = getattr(getattr(encoder_model, "backbone", None), "tokenizer", None)
                    if tokenizer is None:
                        raise RuntimeError("ENCODER G1 capture requires native backbone tokenizer")
                    original_tokenizer = tokenizer
                    original_compose = encoder_model.extract_retrieval_compose
                    original_target = encoder_model.extract_retrieval_target
                    assert callable(original_tokenizer)
                    assert callable(original_compose)
                    assert callable(original_target)

                    def capture_tokens(texts: Any) -> Any:
                        tokens = original_tokenizer(texts)
                        token_batches.append(tokens.detach().cpu().numpy())
                        return tokens

                    def capture_compose(images: Any, texts: list[str]) -> Any:
                        nonlocal query_calls
                        index = query_calls
                        start = index * batch_size
                        stop = start + len(images)
                        pixels = images.detach().cpu().numpy()
                        output = original_compose(images, texts)
                        if len(token_batches) != 1:
                            raise RuntimeError("ENCODER G1 tokenizer/query batch alignment failed")
                        np.savez(g1_root_path / f"query-{index:05d}.npz",
                                 pixels=pixels,
                                 image_paths=np.asarray(capture_source.query_paths[start:stop]),
                                 reference_ids=np.asarray(capture_source.reference_ids[start:stop]),
                                 target_ids=np.asarray(capture_source.target_ids[start:stop]),
                                 reference_positions=np.asarray(capture_source.reference_positions[start:stop]),
                                 target_positions=np.asarray(capture_source.target_positions[start:stop]),
                                 texts=np.asarray(texts), input_ids=token_batches.pop())
                        query_calls += 1
                        return output

                    def capture_target(images: Any) -> Any:
                        nonlocal target_calls
                        index = target_calls
                        start = index * batch_size
                        stop = start + len(images)
                        pixels = images.detach().cpu().numpy()
                        np.savez(g1_root_path / f"gallery-{index:05d}.npz",
                                 pixels=pixels,
                                 image_paths=np.asarray(capture_source.gallery_paths[start:stop]),
                                 gallery_ids=np.asarray(capture_source.gallery_ids[start:stop]))
                        target_calls += 1
                        return original_target(images)

                    encoder_model.backbone.tokenizer = capture_tokens
                    encoder_model.extract_retrieval_compose = capture_compose
                    encoder_model.extract_retrieval_target = capture_target
                original_stack = torch.stack

                def spool_bounded_stack(*args: Any, **kwargs: Any) -> Any:
                    batch = original_stack(*args, **kwargs)
                    tensor_storage.evict()
                    return batch

                torch.stack = spool_bounded_stack
                native_eval_succeeded = False
                try:
                    with FeatureRecorder(encoder_model) as recorder:
                        recalls = dict(test_module.test(params, encoder_model, dataset, category))
                    native_eval_succeeded = True
                finally:
                    torch.stack = original_stack
                    if original_numpy is not None:
                        test_module.np = original_numpy
                    if g1_root:
                        encoder_model.extract_retrieval_compose = original_compose
                        encoder_model.extract_retrieval_target = original_target
                        encoder_model.backbone.tokenizer = original_tokenizer
                        if native_eval_succeeded:
                            if query_calls != (len(capture_source.query_paths) + batch_size - 1) // batch_size:
                                raise ValueError("ENCODER G1 query capture count mismatch")
                            if target_calls != (len(capture_source.gallery_paths) + batch_size - 1) // batch_size:
                                raise ValueError("ENCODER G1 gallery capture count mismatch")
                score_capture_record = (source_capture.finish(category, recalls)
                                        if source_capture is not None else None)
                provenance = {"checkpoint": str(checkpoint_path),
                              "checkpoint_sha256": variant.get("expected_sha256"),
                              "source_commit": MODEL["upstream_commit_sha"],
                              "tensor_storage": tensor_storage}
                if score_capture_record is not None:
                    provenance["native_score_capture"] = score_capture_record
                results.append(ReplayResult(
                    model_id="encoder", checkpoint_id=variant["checkpoint_id"], category=category,
                    protocol=PROTOCOL, r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"), r50=recalls.get(f"{category}_r50"),
                    complete=True, provenance=provenance))
                if observation_out:
                    captured_features[category] = recorder.features()

    if observation_out and captured_features:
        import numpy as np

        arrays = {f"{category}:{kind}": batch
                  for category, (queries, targets) in captured_features.items() if queries is not None
                  for kind, batch in (("queries", queries), ("targets", targets))}
        if arrays:
            destination = Path(observation_out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, **arrays)

    aggregate = macro_aggregate(results)
    # The relation this evaluation's features were computed with travels with the result, in the
    # document the wrapper prints and the parent parses, so an artifact can never describe a
    # measurement without stating its relation (and `parity_gate._relation_checks` can compare it
    # with the composition side).
    for result in results:
        result.provenance["relation_state"] = relation_state
    aggregate["relation_state"] = relation_state
    return results, aggregate


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    if os.environ.get(SCORE_OBSERVATION_ENV, "").strip():
        capture_root = Path(os.environ[SCORE_OBSERVATION_ENV])
        if capture_root.exists() and (not capture_root.is_dir() or any(capture_root.iterdir())):
            parser.error(f"ENCODER capture output is not empty: {capture_root}")
        if args.output is not None and args.output.exists():
            parser.error(f"ENCODER result output already exists: {args.output}")

    selected_categories = tuple(args.categories) if args.categories else CATEGORIES
    blockers = preflight(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        selected_categories,
        checkpoint_id=args.checkpoint_id,
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
        checkpoint_id=args.checkpoint_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = {
        "model_id": "encoder",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
