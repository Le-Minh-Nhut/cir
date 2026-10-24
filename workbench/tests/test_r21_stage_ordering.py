from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import pipeline


#: The capabilities a parsed invocation actually granted, named as the runner sees them.
_CAPABILITY_FLAGS = ("allow_network", "allow_large_downloads", "allow_env_install",
                     "allow_preparation", "allow_gpu_eval", "allow_gpu_preparation",
                     "allow_index_write")


def _granted(args) -> set[str]:
    return {flag for flag in _CAPABILITY_FLAGS if getattr(args, flag, False)}


def _fixture_config(tmp_path: Path) -> WorkbenchConfig:
    """Every workspace root under the test's own directory: no host path is consulted."""
    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path,
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ",
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
    )


def _stub_host_probe(monkeypatch) -> None:
    from workbench.backend import runtime

    monkeypatch.setattr(runtime, "verify_environment", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime, "_cached_gpu_capability", lambda *args, **kwargs: None)


def _applied_tme_stages(tmp_path: Path, monkeypatch, *, allow_gpu_eval: bool):
    """The cold-authorization shape: `--apply` with preparation authorized."""
    _stub_host_probe(monkeypatch)
    argv = ["all", "--model", "tme", "--apply", "--allow-preparation",
            "--allow-gpu-preparation", "--continue-on-error"]
    if allow_gpu_eval:
        argv.append("--allow-gpu-eval")
    args = pipeline.parser_for().parse_args(argv)
    return args, pipeline.all_stages(args, _fixture_config(tmp_path))


def _tme_stages(tmp_path: Path, monkeypatch):
    # Keep all workspace paths temporary. Stub the only host-runtime probe in stage construction;
    # this fixture needs neither a TME environment nor CUDA.
    from workbench.backend import runtime

    monkeypatch.setattr(runtime, "verify_environment", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(runtime, "_cached_gpu_capability", lambda *args, **kwargs: None)
    config = WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path,
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ",
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
    )
    args = pipeline.parser_for().parse_args(["all", "--model", "tme"])
    return args, pipeline.all_stages(args, config)


def test_preparation_follows_same_model_prerequisites(tmp_path, monkeypatch):
    _, stages = _tme_stages(tmp_path, monkeypatch)
    indices = {(stage.name, stage.model_id): index for index, stage in enumerate(stages)}
    model_ids = {stage.model_id for stage in stages if stage.name == "preparation"}

    failures = []
    for model_id in model_ids:
        preparation = indices[("preparation", model_id)]
        for prerequisite in ("auxiliary-assets", "gpu-verification", "checkpoint"):
            if indices[(prerequisite, model_id)] >= preparation:
                failures.append(f"preparation:{model_id} must follow {prerequisite}:{model_id}")
    assert not failures, "; ".join(failures)


def test_tme_stage_selection_is_a_permutation_of_current_stages(tmp_path, monkeypatch):
    _, stages = _tme_stages(tmp_path, monkeypatch)
    actual = Counter((stage.name, stage.model_id) for stage in stages)
    expected = Counter({
        ("doctor", None),
        ("dataset", None),
        ("sync", "tme"),
        ("environment", "tme"),
        ("preparation", "tme"),
        ("checkpoint", "tme"),
        ("auxiliary-assets", "tme"),
        ("gpu-verification", "tme"),
        ("runtime-preflight", "tme"),
        ("evaluation", "tme"),
        ("validation-index", None),
    })
    assert actual == expected


def test_dependencies_never_point_forward_to_stages_in_this_list(tmp_path, monkeypatch):
    _, stages = _tme_stages(tmp_path, monkeypatch)
    indices = {(stage.name, stage.model_id): index for index, stage in enumerate(stages)}

    for index, stage in enumerate(stages):
        for dependency in stage.dependencies:
            name, separator, model_id = dependency.partition(":")
            dependency_index = indices.get((name, model_id if separator else None))
            # Dependencies excluded from this selection are external to this list.
            if dependency_index is not None:
                assert dependency_index < index, (
                    f"{dependency} must appear before {stage.name}:{stage.model_id}"
                )


def test_unauthorized_gpu_verification_is_not_preparation_dependency(tmp_path, monkeypatch):
    args, stages = _tme_stages(tmp_path, monkeypatch)
    preparation = next(stage for stage in stages if stage.name == "preparation" and stage.model_id == "tme")
    gpu_verification = next(stage for stage in stages if stage.name == "gpu-verification" and stage.model_id == "tme")

    # Guard the reorder: unauthorized GPU verification must not block a non-GPU-authorized prep.
    assert not args.allow_gpu_eval
    assert "allow_gpu_preparation" not in preparation.required_capabilities
    assert "allow_gpu_eval" in gpu_verification.required_capabilities
    assert "gpu-verification:tme" not in preparation.dependencies


def test_an_ungranted_evaluation_permission_cannot_cancel_authorized_preparation(tmp_path, monkeypatch):
    """The leak this guards against: `gpu-verification` needs `allow_gpu_eval`, and
    `BLOCKED_AUTHORIZATION_REQUIRED` counts as an UNUSABLE dependency, so depending on it while it
    is unauthorized would mark preparation `SKIPPED_DEPENDENCY` -- withholding an EVALUATION
    permission would silently cancel preparation the operator explicitly authorized.

    The earlier guard used a read-only invocation, where preparation carries no capabilities at all,
    so it could never reach the branch that adds the edge. This one authorizes the run the way a
    cold TME preparation actually is authorized.
    """
    args, stages = _applied_tme_stages(tmp_path, monkeypatch, allow_gpu_eval=False)
    preparation = next(stage for stage in stages if stage.name == "preparation" and stage.model_id == "tme")
    gpu_verification = next(stage for stage in stages if stage.name == "gpu-verification" and stage.model_id == "tme")

    # The run really is in the dangerous configuration: preparation IS authorized, and the GPU stage
    # really does carry a capability this invocation did not grant.
    assert args.apply and args.allow_preparation and args.allow_gpu_preparation
    assert not args.allow_gpu_eval
    assert "allow_gpu_preparation" in preparation.required_capabilities
    assert "allow_gpu_eval" in gpu_verification.required_capabilities
    # Therefore the edge must be absent, or the leak is live.
    assert "gpu-verification:tme" not in preparation.dependencies, \
        "an ungranted allow_gpu_eval would skip preparation as SKIPPED_DEPENDENCY"


def test_granted_gpu_verification_is_ordered_before_cuda_preparation(tmp_path, monkeypatch):
    """The other half: when the operator DID authorize it, the verified capability precedes the
    CUDA-only generation rather than being dropped from the ordering."""
    args, stages = _applied_tme_stages(tmp_path, monkeypatch, allow_gpu_eval=True)
    preparation = next(stage for stage in stages if stage.name == "preparation" and stage.model_id == "tme")
    indices = {(stage.name, stage.model_id): index for index, stage in enumerate(stages)}

    assert args.allow_gpu_eval
    assert "gpu-verification:tme" in preparation.dependencies
    assert indices[("gpu-verification", "tme")] < indices[("preparation", "tme")]


def test_preparation_does_not_depend_on_a_checkpoint_it_never_reads(tmp_path, monkeypatch):
    """No declared preparation generator consumes a checkpoint: every one reads the dataset or the
    pinned source. `checkpoint` carries `allow_large_downloads`/`allow_network`, so an edge to it
    would let a withheld DOWNLOAD permission cancel preparation the operator authorized.
    """
    for model_id in ("limn", "tme"):
        args = pipeline.parser_for().parse_args(
            ["all", "--model", model_id, "--apply", "--allow-preparation", "--continue-on-error"])
        stages = pipeline.all_stages(args, _fixture_config(tmp_path))
        preparation = next(stage for stage in stages
                           if stage.name == "preparation" and stage.model_id == model_id)
        assert f"checkpoint:{model_id}" not in preparation.dependencies, model_id

        # And the stage the edge would have pointed at really does demand permissions the operator
        # withheld -- which is what made the edge harmful rather than merely redundant.
        checkpoint_stages = [stage for stage in stages
                             if stage.name == "checkpoint" and stage.model_id == model_id]
        for checkpoint in checkpoint_stages:
            assert not (checkpoint.required_capabilities <= _granted(args)), checkpoint.name
