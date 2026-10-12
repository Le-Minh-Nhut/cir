from __future__ import annotations

from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry
from workbench.scripts import pipeline


def _config(tmp_path: Path) -> WorkbenchConfig:
    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path, CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ", WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party")


def test_preparation_stages_follow_declared_links_without_duplicate_writers(
        tmp_path: Path) -> None:
    models = load_registry()["models"]
    declaring = [model for model in models if model.get("canonical_dataset_link")]
    args = pipeline.parser_for().parse_args(
        ["all", "--all-models", "--apply", "--allow-preparation"])
    config = _config(tmp_path)

    stages = pipeline.all_stages(args, config)
    preparations = {stage.model_id: stage for stage in stages if stage.name == "preparation"}
    assert len(preparations) == len(models), "pipeline must retain per-model preparation status"
    assert len([stage for stage in preparations.values() if stage.output_paths]) == len(
        {model["source_dir"] + "/" + model["canonical_dataset_link"] for model in declaring})
    assert {stage.model_id for stage in preparations.values() if stage.output_paths} == {
        next(model["model_id"] for model in declaring
             if model["source_dir"] + "/" + model["canonical_dataset_link"] == path)
        for path in {model["source_dir"] + "/" + model["canonical_dataset_link"] for model in declaring}
    }

    for model in declaring:
        stage = preparations[model["model_id"]]
        expected = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] / model["canonical_dataset_link"]
        matching_writers = [item for item in preparations.values() if expected in item.output_paths]
        assert len(matching_writers) == 1, f"{expected} must have one writer"
        owner = matching_writers[0]
        if stage.model_id == owner.model_id:
            assert stage.output_paths == (expected,)
            assert "--execute" in stage.commands[0] and "--allow-preparation" in stage.commands[0]
        else:
            assert stage.output_paths == ()
            assert f"preparation:{owner.model_id}" in stage.dependencies
            assert "--execute" not in stage.commands[0]

    clip_models = [model for model in declaring if model["source_dir"] == "CLIP4Cir"]
    assert len(clip_models) == 2
    clip_link = config.WORKBENCH_THIRD_PARTY_ROOT / "CLIP4Cir" / clip_models[0]["canonical_dataset_link"]
    assert sum(clip_link in stage.output_paths for stage in preparations.values()) == 1


def test_preparation_status_stages_are_read_only_without_apply_authorization(tmp_path: Path) -> None:
    models = load_registry()["models"]
    args = pipeline.parser_for().parse_args(["all", "--all-models"])
    stages = pipeline.all_stages(args, _config(tmp_path))
    preparations = [stage for stage in stages if stage.name == "preparation"]

    declaring = [model for model in models if model.get("canonical_dataset_link")]
    assert len(preparations) == len(models)
    assert all(not stage.required_capabilities for stage in preparations)
    assert all("--execute" not in stage.commands[0] for stage in preparations)
    writers = [stage for stage in preparations if stage.output_paths]
    assert len(writers) == len({model["source_dir"] + "/" + model["canonical_dataset_link"]
                                for model in declaring})
