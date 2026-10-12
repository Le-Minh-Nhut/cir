"""Every scientific input must be in the evaluation fingerprint, through the real stage graph.

These drive the production `evaluation` stage — the same construction `pipeline.py all` uses —
and mutate one scientific input at a time. A mutation that leaves the fingerprint unchanged is a
resume that would silently reuse a result computed from different data.

The fixtures follow the real shape: a category layout, a checkpoint, correction dictionaries,
resized image roots, prepared annotation caches, and registered auxiliary assets.
"""
from __future__ import annotations


import subprocess
from pathlib import Path

import pytest

from workbench.backend import registry
from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import pipeline

CATEGORIES = ("dress", "shirt", "toptee")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A workspace whose layout satisfies one model's real preparation contract."""
    ws_pin: dict[str, str] = {}
    repo, data, checkpoints, third_party, results = (
        tmp_path / name for name in ("repo", "FashionIQ", "checkpoints", "third_party", "results"))
    for directory in (repo, data, checkpoints, third_party, results):
        directory.mkdir()

    # Canonical FashionIQ: captions, splits, images, correction dictionaries, resized images.
    for category in CATEGORIES:
        (data / "captions").mkdir(exist_ok=True)
        (data / "image_splits").mkdir(exist_ok=True)
        (data / "images").mkdir(exist_ok=True)
        (data / "captions" / f"cap.{category}.val.json").write_text(
            '[{"target": "img1", "candidate": "img0", "captions": ["a", "b"]}]')
        (data / "image_splits" / f"split.{category}.val.json").write_text('["img0", "img1"]')
        (data / "captions" / f"correction_dict_{category}.json").write_text('{"a": "b"}')
        (data / "resized_image" / category).mkdir(parents=True, exist_ok=True)
        (data / "resized_image" / category / "img0.jpg").write_bytes(b"\xff\xd8\xff\xe0image-v1")
        (data / "images" / f"img{0}.png").write_bytes(b"\x89PNG\r\n\x1a\nimage-v1")
        (data / "images" / f"img{1}.png").write_bytes(b"\x89PNG\r\n\x1a\nimage-v1")

    # The evaluator implementation is itself an input, and its provenance is part of the
    # evidence: a real pinned checkout, so an unreadable source is not mistaken for the norm.
    source = third_party / "IlearnModel"
    (source / "src").mkdir(parents=True)
    (source / "src" / "evaluate.py").write_text("print('evaluator')\n")
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    ws_pin["head"] = head

    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path / "data", FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=checkpoints, WORKBENCH_RESULTS_ROOT=results,
        WORKBENCH_THIRD_PARTY_ROOT=third_party)
    return {"repo": repo, "data": data, "checkpoints": checkpoints,
            "third_party": third_party, "results": results, "config": config,
            "tmp": tmp_path, "head": ws_pin.get("head")}


def _model(model_id: str, *, layout: str, extra: dict | None = None) -> dict:
    model = {
        "model_id": model_id,
        "source_dir": "IlearnModel",
        "fashioniq_layout": layout,
        "checkpoint_variants": [{"checkpoint_id": "ckpt", "filename": "model.pt",
                                 "checkpoint_mapping_status": "VERIFIED_METADATA"}],
        "checkpoint_bundles": [],
        "environment_required": False,
        "native_protocol": "fashioniq_original_split",
        "supported_protocols": ["fashioniq_original_split"],
        "upstream_commit_sha": None,
        "preparation_contract": "manual_ilearn_resized",
    }
    model.update(extra or {})
    return model


def _stage_for(ws: dict, monkeypatch: pytest.MonkeyPatch, model: dict) -> pipeline.Stage:
    """Build the real evaluation stage for *model* in this workspace."""
    fake_registry = {"models": [model], "protocols": {
        "fashioniq_original_split": {"literature_split_label": "original"}}}
    monkeypatch.setattr(pipeline, "load_registry", lambda: fake_registry)
    monkeypatch.setattr(registry, "load_registry", lambda: fake_registry)
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: (
        model if model_id == model["model_id"] else () or (_ for _ in ()).throw(KeyError())))
    monkeypatch.setattr(registry, "auxiliary_model_ids", lambda: set())
    monkeypatch.setattr(registry, "auxiliary_assets_for_model", lambda model_id: [])
    monkeypatch.setattr(pipeline, "evaluation_evidence",
                        lambda *args: (lambda record: (True, None, None)))
    args = pipeline.parser_for().parse_args(
        ["all", "--model", model["model_id"], "--dataset-root", str(ws["data"])])
    return next(stage for stage in pipeline.all_stages(args, ws["config"])
                if stage.name == "evaluation")


def _checkpoint(ws: dict, model: dict) -> Path:
    path = ws["checkpoints"] / model["model_id"] / "model.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint-v1-weights")
    return path


def _fingerprint(ws: dict, monkeypatch: pytest.MonkeyPatch, model: dict) -> str:
    stage = _stage_for(ws, monkeypatch, model)
    return pipeline.stage_input_fingerprint_info(stage, ws["config"])["digest"]


def _assert_mutation_invalidates(ws: dict, monkeypatch: pytest.MonkeyPatch, model: dict,
                                 mutate, *, label: str) -> None:
    before = _fingerprint(ws, monkeypatch, model)
    mutate()
    after = _fingerprint(ws, monkeypatch, model)
    assert after != before, f"{label} changed without invalidating the evaluation fingerprint"


# ---------------------------------------------------------------------- data identity

def test_data_id_01_raw_image_content_change_invalidates(tmp_path: Path, ws: dict,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A rewritten gallery image changes what the model sees, so the fingerprint must change."""
    model = _model("ilearn_raw", layout="fashioniq_standard")
    _checkpoint(ws, model)
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: (ws["data"] / "images" / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\nimage-v2"),
        label="raw image content")


def test_data_id_02_resized_image_change_invalidates(ws: dict,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """iLearn evaluators open `resized_image/<category>/<id>.jpg`, so those bytes are inputs."""
    model = _model("ilearn_resized", layout="fashioniq_ilearn_resized")
    _checkpoint(ws, model)
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: (ws["data"] / "resized_image" / "dress" / "img0.jpg").write_bytes(
            b"\xff\xd8\xff\xe0image-v2"),
        label="resized image content")


def test_data_id_03_correction_dictionary_change_invalidates(ws: dict,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """The correction dictionary rewrites the query text, so it changes the query."""
    model = _model("ilearn_correction", layout="fashioniq_ilearn_resized")
    _checkpoint(ws, model)
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: (ws["data"] / "captions" / "correction_dict_dress.json").write_text('{"a": "c"}'),
        label="correction dictionary")


def test_data_id_04_prepared_pickle_change_invalidates(ws: dict,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """DCNet reads prepared GloVe annotations; regenerating them changes the query set."""
    model = _model("dcnet_like", layout="fashioniq_dcnet")
    _checkpoint(ws, model)
    for category in CATEGORIES:
        (ws["data"] / "captions" / f"cap.{category}.glove.val.pkl").write_bytes(b"glove-v1")
    (ws["data"] / "resized_images").mkdir(parents=True, exist_ok=True)
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: (ws["data"] / "captions" / "cap.dress.glove.val.pkl").write_bytes(b"glove-v2"),
        label="prepared GloVe pickle")


def test_data_id_05_auxiliary_caption_change_invalidates(tmp_path: Path, ws: dict,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A registered auxiliary asset placed in the checkout is an evaluation input."""
    asset = ws["third_party"] / "IlearnModel" / "COT_ours2" / "fashioniq"
    asset.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        (asset / f"{category}_cot_val.json").write_text('{"img1": "cot-v1"}')

    model = _model("aux_model", layout="fashioniq_standard")
    _checkpoint(ws, model)
    monkeypatch.setattr(registry, "auxiliary_assets_for_model", lambda model_id: [
        {"model_id": model_id, "family": "COT", "destination_scope": "model_source",
         "destination_path": f"COT_ours2/fashioniq/{category}_cot_val.json",
         "required_for": "official_evaluation"}
        for category in CATEGORIES])
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: (asset / "dress_cot_val.json").write_text('{"img1": "cot-v2"}'),
        label="auxiliary caption")


def test_data_id_06_checkpoint_change_invalidates(ws: dict,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """The checkpoint is the model; a different file is a different experiment."""
    model = _model("ilearn_ckpt", layout="fashioniq_standard")
    checkpoint = _checkpoint(ws, model)
    _assert_mutation_invalidates(
        ws, monkeypatch, model,
        lambda: checkpoint.write_bytes(b"checkpoint-v2-weights"),
        label="checkpoint content")


def test_data_id_07_preprocessing_recipe_change_invalidates(
        ws: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the declared layout or extensions changes the experiment, with no file moved."""
    model = _model("ilearn_recipe", layout="fashioniq_standard")

    class _ChangingRegistry:
        """The registry returns the same model with a different recipe on the second read."""

        def __init__(self, first: dict, second: dict) -> None:
            self.models = [first]

        def swap(self) -> None:
            self.models = [self.models[0]]

    fake = {"models": [model], "protocols": {}}
    monkeypatch.setattr(pipeline, "load_registry", lambda: fake)
    monkeypatch.setattr(registry, "load_registry", lambda: fake)
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: fake["models"][0])
    monkeypatch.setattr(registry, "auxiliary_model_ids", lambda: set())
    monkeypatch.setattr(registry, "auxiliary_assets_for_model", lambda model_id: [])
    monkeypatch.setattr(pipeline, "evaluation_evidence",
                        lambda *args: (lambda record: (True, None, None)))
    _checkpoint(ws, model)

    stage = _stage_for(ws, monkeypatch, model)
    before = pipeline.stage_input_fingerprint_info(stage, ws["config"])["digest"]

    model["fashioniq_image_extensions"] = [".png"]
    after = pipeline.stage_input_fingerprint_info(stage, ws["config"])["digest"]
    assert after != before, "a preprocessing recipe change must invalidate the stage"

    model["validation_caption_policy"] = "normalized_ordered_and"
    third = pipeline.stage_input_fingerprint_info(stage, ws["config"])["digest"]
    assert third != after, "the caption policy is part of the recipe"


def test_data_id_08_dataset_symlink_retarget_invalidates(ws: dict,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared dataset link that points somewhere else is a different dataset."""
    other = ws["tmp"] / "OtherFashionIQ"
    (other / "images").mkdir(parents=True)
    (other / "images" / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\nother")
    link = ws["third_party"] / "IlearnModel" / "fashionIQ_dataset"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(ws["data"], target_is_directory=True)

    model = _model("link_model", layout="fashioniq_standard",
                   extra={"canonical_dataset_link": "fashionIQ_dataset"})
    _checkpoint(ws, model)

    def make_stage() -> pipeline.Stage:
        return _stage_for(ws, monkeypatch, model)

    before = pipeline.stage_input_fingerprint_info(make_stage(), ws["config"])["digest"]
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    after = pipeline.stage_input_fingerprint_info(make_stage(), ws["config"])["digest"]
    assert after != before, "a retargeted dataset link must invalidate the stage"


# -------------------------------------------------------------------- resume behaviour

def test_data_id_09_unchanged_verified_inputs_still_resume(ws: dict,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """The stricter identity must not defeat resume: identical inputs still skip."""
    model = _model("ilearn_resume", layout="fashioniq_standard")
    _checkpoint(ws, model)
    stage = _stage_for(ws, monkeypatch, model)
    state, launched = ws["tmp"] / "state.json", []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)

    args = dict(apply=True, authorized_capabilities={"allow_gpu_eval", "allow_preparation"},
                config=ws["config"], state_path=state, resume=True)
    assert pipeline.run_stages([stage], **args) == 0
    assert len(launched) == 1
    assert pipeline.run_stages([stage], **args) == 0
    assert len(launched) == 1, "unchanged verified inputs must still be skipped"


def test_data_id_10_a_missing_declared_input_refuses_verified_resume(
        ws: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared input that is gone is not evidence of an unchanged run."""
    model = _model("ilearn_missing", layout="fashioniq_standard")
    _checkpoint(ws, model)
    stage = _stage_for(ws, monkeypatch, model)
    state, launched = ws["tmp"] / "state.json", []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)
    args = dict(apply=True, authorized_capabilities={"allow_gpu_eval", "allow_preparation"},
                config=ws["config"], state_path=state, resume=True)

    assert pipeline.run_stages([stage], **args) == 0
    (ws["data"] / "captions" / "cap.dress.val.json").unlink()
    assert pipeline.run_stages([stage], **args) == 0
    record = pipeline.load_pipeline_state(state)["stages"][pipeline.stage_key(stage)]
    assert record["input_fingerprint_evidence"]["weak"] is True
    assert "missing" in record["resume_refused_reason"].lower()
    assert len(launched) == 2, "a missing input must force a rerun"


def test_data_id_12_a_shared_input_change_invalidates_both_models(
        ws: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two models reading the same file both invalidate; the fingerprint records the shared path."""
    model_a = _model("shared_a", layout="fashioniq_standard")
    model_b = _model("shared_b", layout="fashioniq_standard")
    _checkpoint(ws, model_a)
    _checkpoint(ws, model_b)

    before_a = _fingerprint(ws, monkeypatch, model_a)
    before_b = _fingerprint(ws, monkeypatch, model_b)
    (ws["data"] / "captions" / "cap.dress.val.json").write_text(
        '[{"target": "img1", "candidate": "img0", "captions": ["a", "c"]}]')
    assert _fingerprint(ws, monkeypatch, model_a) != before_a
    assert _fingerprint(ws, monkeypatch, model_b) != before_b

    # A model that does not read the file is unaffected by it.
    unrelated = _model("unrelated", layout="fashioniq_standard")
    _checkpoint(ws, unrelated)
    stage = _stage_for(ws, monkeypatch, unrelated)
    paths = {str(path) for path in stage.input_paths}
    assert str(ws["data"] / "captions" / "cap.dress.val.json") in paths
    # And its own checkpoint is the only checkpoint it names.
    assert str(ws["checkpoints"] / "shared_a" / "model.pt") not in paths


# ------------------------------------------------------- inaccessible and external content

def test_an_inaccessible_directory_is_not_content_verified(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory the walk cannot enter has contents the fingerprint never saw.

    `os.walk` skips an unreadable directory silently, so a tree holding a file nobody could read
    was reportable as fully content-verified. The failure is now recorded and the tree is partial.
    """
    root = tmp_path / "tree"
    (root / "visible").mkdir(parents=True)
    (root / "visible" / "a.txt").write_text("a")
    blocked = root / "blocked"
    blocked.mkdir()
    (blocked / "secret.bin").write_bytes(b"unreadable-content")

    real_walk = pipeline.os.walk

    def walk_with_a_failure(top, *, followlinks=False, onerror=None):
        # Enter the visible part, then report the blocked directory exactly as the kernel would.
        yield from real_walk(root / "visible", followlinks=False, onerror=onerror)
        assert onerror is not None, "the walk must be given an error callback"
        onerror(OSError(13, "Permission denied", str(blocked)))

    monkeypatch.setattr(pipeline.os, "walk", walk_with_a_failure)
    inventory, category, files, total, truncated = pipeline._walk_evidence(root)

    assert category == pipeline.TREE_PARTIAL, category
    assert category not in pipeline.VERIFIED_TREE_CATEGORIES
    assert not truncated, "an inaccessible directory is partial evidence, not a budget problem"
    assert inventory is not None


def test_a_symlink_leaving_the_root_is_not_verified_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An external target's bytes are never read by this walk, so it cannot count as verified.

    Counting the link as verified would let a file outside the tree change without invalidating a
    resumed stage.
    """
    root = tmp_path / "tree"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "photo.png"
    target.write_bytes(b"external-v1")
    (root / "img.png").symlink_to(target)

    inventory, category, files, total, truncated = pipeline._walk_evidence(root)
    assert category == pipeline.TREE_PARTIAL, category
    assert category not in pipeline.VERIFIED_TREE_CATEGORIES

    # And the recorded entry says why.
    entry_categories = [
        item["category"]
        for item in pipeline.stage_input_fingerprint_info(
            pipeline.Stage("link", input_paths=(root,)))["inputs"]
        if item["category"] in {pipeline.TREE_PARTIAL, pipeline.TREE_CONTENT_VERIFIED}
    ]
    assert entry_categories == [pipeline.TREE_PARTIAL], entry_categories


def test_a_symlink_inside_the_root_is_verified(tmp_path: Path) -> None:
    """A link that stays inside is covered by the target's own entry in the same walk."""
    root = tmp_path / "tree"
    (root / "real").mkdir(parents=True)
    (root / "real" / "img.png").write_bytes(b"internal")
    (root / "img.png").symlink_to(root / "real" / "img.png")

    inventory, category, files, total, truncated = pipeline._walk_evidence(root)
    assert category == pipeline.TREE_CONTENT_VERIFIED, category
    assert files == 2, files
