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

import json
import os
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
    return (
        validate_source(source_root)
        + validate_auxiliary_assets(source_root)
        + validate_checkpoint(checkpoint_root, checkpoint_id)
        + validate_dataset(dataset_root, categories)
    )


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

    encoder_model = encoder_model.to(target_device)
    encoder_model.eval()
    # The per-query export may recompute a category's features when the aggregate did not capture
    # them, and must agree to the hit count, so the forward pass is pinned as reproducible as CUDA
    # permits rather than merely equivalent.
    enable_deterministic_inference(torch)

    clip_path = str(source_root)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(source_root / AUXILIARY_ASSET_NAME),
    )
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
    relation_state = _relation_record(
        encoder_model,
        Path(observation_out).parent if observation_out else None,
        drawn=drawn)
    captured_features: dict[str, tuple] = {}
    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        # The query inputs are baked in HERE, through the stochastic training transform, so this
        # construction must be reproducible for the per-query export -- which rebuilds the dataset in
        # its own process -- to observe the same inputs. See `seeded_dataset_construction`.
        with seeded_dataset_construction():
            dataset = datasets.FashionIQ(path=root_str, transform=transform, split="val-split")
        params = SimpleNamespace(batch_size=batch_size, local_rank=0)

        for category in categories:
            with FeatureRecorder(encoder_model) as recorder:
                recalls = dict(test_module.test(params, encoder_model, dataset, category))
            if observation_out:
                captured_features[category] = recorder.features()
            results.append(
                ReplayResult(
                    model_id="encoder",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": variant.get("expected_sha256"),
                        "source_commit": MODEL["upstream_commit_sha"],
                    },
                )
            )

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
