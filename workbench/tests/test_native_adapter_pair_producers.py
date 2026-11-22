from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.scripts import conesep_native_adapter_pair as conesep_pair
from workbench.scripts import intent_native_adapter_pair as intent_pair


@pytest.mark.parametrize(("module", "model_id", "source_name"), [
    (conesep_pair, "conesep", "ConeSep.py"),
    (intent_pair, "intent", "blip2_cir.py"),
])
def test_pair_producer_consumes_aligned_native_and_adapter_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module, model_id: str,
    source_name: str,
) -> None:
    output = tmp_path / model_id
    source_root = tmp_path / "source"
    (source_root / "lavis/models/blip2_models").mkdir(parents=True)
    (source_root / "test.py").write_text("source\n")
    (source_root / "lavis/models/blip2_models" / source_name).write_text("model\n")
    checkpoint_root = tmp_path / "checkpoints"
    variant = module.CHECKPOINTS["fiq_n02"]
    checkpoint_root.mkdir()
    (checkpoint_root / variant["filename"]).write_bytes(b"checkpoint")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    queries = np.asarray(["0:dress_ref:dress_target"])
    gallery = np.asarray(["dress_ref", "dress_target", "dress_other"])
    scores = np.asarray([[-1e10, 0.9, 0.1]], dtype=np.float32)
    orders = np.argsort(-scores, axis=1)
    refs = np.asarray([0], dtype=np.int32)
    targets = np.asarray([1], dtype=np.int32)
    native_root, adapter_root = output / "native", output / "adapter"
    native_root.mkdir(parents=True)
    adapter_root.mkdir(parents=True)
    np.savez(native_root / "dress.scores.npz", scores=scores, orders=orders, query_ids=queries,
        gallery_ids=gallery, reference_positions=refs, target_positions=targets)
    np.savez(adapter_root / "adapter_pair_scores.npz", scores=scores, orders=orders, query_ids=queries,
        gallery_ids=gallery, reference_positions=refs, target_positions=targets,
        adapter_forward_repeat_equal=np.asarray([True]), adapter_b1_bn_equal=np.asarray([True]))
    comparison_fields = {"rank_equal": np.asarray([3]), "abs_score_delta": np.zeros_like(scores)}
    comparison_fields["top10_membership_equal" if model_id == "conesep" else "top10_equal"] = np.asarray([3])
    np.savez(adapter_root / "comparison.npz", **comparison_fields)
    receipt = {"matmul_calls": 1, "identical_repeat_passes": 1}
    (native_root / "dress.json").write_text(json.dumps(receipt))
    (adapter_root / "input_comparison.json").write_text(json.dumps({"pixels": True, "tokens": True}))
    # ReplayResult is not needed by the producer beyond its serialized result fields.
    result = SimpleNamespace(__dict__={"r1": 1.0})
    monkeypatch.setattr(module, "require_verified_model_interpreter", lambda *args, **kwargs: "/verified/python")
    monkeypatch.setattr(module, "_controller_adapter", lambda *args, **kwargs: {"query_count": 1, "queries": 1})
    def native_stage(command, **kwargs):
        assert command[:4] == ["/verified/python", "-m", module.__name__, "--native-stage"]
        args = dict(zip(command[4::2], command[5::2]))
        assert args["--checkpoint-root"] == str(checkpoint_root)
        assert args["--dataset-root"] == str(dataset_root)
        assert args["--source-root"] == str(source_root)
        assert args["--output"] == str(output)
        assert args["--batch-size"] == "64"
        if "--category" in args:
            assert args["--category"] == "dress"
        stage_root = output / "native"
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / "native-stage.json").write_text(json.dumps({"native_pid": -1}))
    monkeypatch.setattr(module.subprocess, "run", native_stage)
    seal = module.run(checkpoint_root, dataset_root, source_root, output)
def test_habit_resume_uses_verified_native_archive_without_native_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workbench.scripts import habit_native_adapter_pair as habit_pair

    source_root, checkpoint_root = tmp_path / "source", tmp_path / "checkpoints"
    dataset_root = tmp_path / "dataset"
    (source_root / "lavis/models/blip2_models").mkdir(parents=True)
    (dataset_root / "image_splits").mkdir(parents=True)
    (dataset_root / "captions").mkdir()
    checkpoint_root.mkdir()
    source_files = (source_root / "test.py",
                    source_root / "lavis/models/blip2_models/HABIT.py")
    for path in source_files:
        path.write_text("source\n")
    checkpoint = checkpoint_root / "HABIT-FIQ_N0.2.pt"
    checkpoint.write_bytes(b"checkpoint")
    population_files = [dataset_root / "image_splits" / f"split.{category}.val.json"
                        for category in ("dress", "shirt", "toptee")]
    population_files += [dataset_root / "captions" / f"cap.{category}.val.json"
                         for category in ("dress", "shirt", "toptee")]
    for path in population_files:
        path.write_text("[]\n")
    archive = tmp_path / "archive/native"
    archive.mkdir(parents=True)
    manifest = {"role": "fresh_native_primary", "partition": "FashionIQ original-split dress.val",
        "batch_size": 32, "precision": "float32",
        "checkpoint_sha256": habit_pair.sha256(checkpoint),
        "source_sha256": {str(path.relative_to(source_root)): habit_pair.sha256(path)
                          for path in source_files},
        "population_files_sha256": {str(path.relative_to(dataset_root)): habit_pair.sha256(path)
                                     for path in population_files}}
    (archive / "manifest.json").write_text(json.dumps(manifest))
    (archive / "native-stage.json").write_text(json.dumps({"native_pid": -1}))
    (archive / "dress.json").write_text("{}")
    (archive / "dress.scores.npz").write_bytes(b"native scores")
    output = tmp_path / "resume"
    calls = []

    def controller(*args, **kwargs):
        calls.append(kwargs["native_root"])
        score_path = output / "adapter/adapter_pair_scores.npz"
        score_path.parent.mkdir(parents=True)
        score_path.write_bytes(b"adapter scores")
        return {"query_count": 2017}

    monkeypatch.setattr(habit_pair, "require_verified_model_interpreter", lambda *a, **kw: "/python")
    monkeypatch.setattr(habit_pair, "_controller_adapter", controller)
    monkeypatch.setattr(habit_pair.subprocess, "run", lambda *a, **kw: pytest.fail("native rerun"))
    result = habit_pair.run(checkpoint_root, dataset_root, source_root, output, 32,
                            resume_native_root=archive)

    assert calls == [archive]
    assert result["native_archive"] == str(archive.resolve())
    assert result["native_source_manifest_sha256"] == habit_pair.sha256(archive / "manifest.json")
    assert result["native_scores_sha256"] == habit_pair.sha256(archive / "dress.scores.npz")
    assert (output / "seal.json").is_file()
