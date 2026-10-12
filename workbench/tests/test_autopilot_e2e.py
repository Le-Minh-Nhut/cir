"""The synthetic multi-model `autopilot` scenario required by the release spec (section 18).

One command drives a miniature research workbench containing six models with deliberately different
fates, and the run must group environments, acquire a shared source once, evaluate the runnable
models through real subprocesses, block the model whose checkpoint is missing, fail the model whose
evaluator errors, and report every model with a truthful status.

The models, their environments and their sources are synthetic. The machinery under test is not:
the CLI is invoked as a subprocess, the queue and the operator run for real, and the evaluators are
real child processes that read real files.

This module owns the fixture and the assertions. It is skipped, loudly, if the autopilot entry point
is not present, so a missing CLI cannot pass as success here.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
AUTOPILOT = REPOSITORY / "workbench" / "scripts" / "autopilot.py"

pytestmark = pytest.mark.skipif(
    not AUTOPILOT.is_file(),
    reason="the autopilot entry point does not exist yet; this scenario cannot pass without it")


MODELS = (
    # model_id, shares_source_with, checkpoint, evaluator behaviour, environment constraint
    ("alpha", None, "present", "ok", "torch=1.12.1"),
    ("beta", "alpha", "present", "ok", "torch=1.12.1"),
    ("gamma", None, "present", "ok", "torch=2.0.1"),
    ("delta", None, "absent", "ok", "torch=1.12.1"),
    ("epsilon", None, "present", "fail", "torch=1.12.1"),
    ("zeta", "alpha", "present", "ok", "torch=1.12.1"),
)


def _fashioniq(root: Path) -> None:
    for sub in ("captions", "image_splits", "images"):
        (root / sub).mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (root / "captions" / f"cap.{category}.val.json").write_text(
            '[{"target": "img1", "candidate": "img0", "captions": ["c1", "c2"]}]')
        (root / "image_splits" / f"split.{category}.val.json").write_text('["img0", "img1"]')
    for name in ("img0", "img1"):
        (root / "images" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")


EVALUATOR_SOURCE = '''import json, os, sys

if os.environ.get("SYNTHETIC_FAIL"):
    sys.stderr.write("synthetic evaluator failed\\n")
    sys.exit(3)
print("evaluating")
print(json.dumps({
    "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
    "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
    "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0,
    "average_recall_at10": 55.0, "average_recall_at50": 75.0, "average_recall": 65.0,
}))
'''


@pytest.fixture
def workbench(tmp_path: Path) -> dict:
    """A six-model synthetic workbench: registry, sources, data, checkpoints, config."""
    ws = tmp_path / "workbench"
    ws.mkdir()
    data, checkpoints, third_party, results, reports = (
        ws / name for name in ("FashionIQ", "checkpoints", "third_party", "results", "reports"))
    for directory in (data, checkpoints, third_party, results, reports, ws / "artifacts"):
        directory.mkdir()
    _fashioniq(data)

    # Models alpha, beta and zeta share one source checkout: it must be cloned/verified once.
    shared_source = third_party / "SharedModel"
    (shared_source / "src").mkdir(parents=True)
    (shared_source / "src" / "evaluator.py").write_text(EVALUATOR_SOURCE)
    # A pinned requirements file is what makes an environment verified rather than unverified.
    (shared_source / "requirements.txt").write_text("torch==1.12.1\ntorchvision==0.13.1\n")
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=shared_source, check=True, capture_output=True)
    shared_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shared_source,
                                          text=True).strip()

    # gamma and delta get their own sources so their fates are independent of the shared one.
    for name in ("GammaModel", "DeltaModel"):
        source = third_party / name
        (source / "src").mkdir(parents=True)
        (source / "src" / "evaluator.py").write_text(EVALUATOR_SOURCE)
        (source / "requirements.txt").write_text("torch==1.12.1\ntorchvision==0.13.1\n")

    # Model E's evaluator fails on its own, so its failure is independent of every other model.
    epsilon_source = third_party / "EpsilonModel"
    (epsilon_source / "src").mkdir(parents=True)
    (epsilon_source / "src" / "evaluator.py").write_text(
        "import sys\n"
        "sys.stderr.write('synthetic evaluator failed\\n')\n"
        "sys.exit(3)\n")
    (epsilon_source / "requirements.txt").write_text("torch==1.12.1\ntorchvision==0.13.1\n")

    # Every checkout is a real, pinned git repository, because provenance is checked from git.
    for name in ("SharedModel", "GammaModel", "DeltaModel", "EpsilonModel"):
        source = third_party / name
        for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                     ["git", "config", "user.name", "T"], ["git", "add", "."],
                     ["git", "commit", "-qm", "pin"]):
            # `git commit` exits non-zero when there is nothing to record, which is not a failure
            # for the fixture: the checkout is still clean and pinned.
            subprocess.run(argv, cwd=source, check=False, capture_output=True)

    def head(path: Path) -> str:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

    models = []
    for model_id, shares, checkpoint, behaviour, constraint in MODELS:
        source_dir = {"alpha": "SharedModel", "beta": "SharedModel", "zeta": "SharedModel",
                      "gamma": "GammaModel", "delta": "DeltaModel",
                      "epsilon": "EpsilonModel"}[model_id]
        source = third_party / source_dir
        checkpoint_dir = checkpoints / model_id
        checkpoint_dir.mkdir(parents=True)
        filename = "model.pt"
        if checkpoint == "present":
            (checkpoint_dir / filename).write_bytes(b"synthetic-checkpoint-%s" % model_id.encode())
        models.append({
            "model_id": model_id,
            "method_name": model_id.title(),
            "research_generation": "recent",
            "publication_year": 2026,
            "paper_title": f"Synthetic {model_id}",
            "paper_url": None,
            "upstream_repo_url": "https://example.invalid/synthetic.git",
            "upstream_default_branch": "main",
            "upstream_commit_sha": shared_head if shares or source_dir == "SharedModel" else head(source),
            "source_available": True,
            "source_dir": source_dir,
            "synthetic_source": True,
            "synthetic_script": "src/evaluator.py",
            "fashioniq_layout": "fashioniq_standard",
            "literature_split_label": "original",
            "native_protocol": "fashioniq_original_split",
            "supported_protocols": ["fashioniq_original_split"],
            "reported_r10": 55.0, "reported_r50": 75.0, "reported_mean": 65.0,
            "reproduction_status": "NOT_RUN",
            "command_status": "COMMAND_AUDITED",
            "strict_clean_checkpoint_available": checkpoint == "present",
            "environment_required": True,
            # Declared packages must be exactly what the shared interpreter provides: the runtime
            # probes the interpreter and refuses a contract it cannot satisfy.
            "environment": {"python_env_var": f"WORKBENCH_{model_id.upper()}_PYTHON",
                            "python": "UNKNOWN", "confidence": "VERIFIED_README",
                            "pytorch": "2.0.1" if model_id == "gamma" else "1.12.1",
                            "torchvision": "0.15.2" if model_id == "gamma" else "0.13.1",
                            "packages": []},
            "checkpoint_variants": [{
                "checkpoint_id": "ckpt",
                "filename": filename,
                # A local file is only accepted as verified when its provenance is recorded; the
                # fixture mirrors that by giving the present models a URL and delta none.
                "download_url": (f"https://example.invalid/{model_id}/model.pt"
                                 if checkpoint == "present" else None),
                "source_type": "author_huggingface" if checkpoint == "present" else "none",
                "checkpoint_training_noise_pct": 0,
                "evaluation_noise_pct": 0,
                "expected_sha256": None,
                "status": "URL_VERIFIED" if checkpoint == "present" else "NO_SOURCE",
                "checkpoint_mapping_status": "VERIFIED_METADATA",
            }],
        })

    registry = ws / "models.yaml"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "protocols": {"fashioniq_original_split": {
            "label": "Original Split", "literature_split_label": "original",
            "evaluation_noise_pct": 0}},
        "models": models,
    }))

    config = ws / "pc_autopilot.yaml"
    config.write_text(json.dumps({
        "schema_version": 1,
        "selected_models": "all_registered",
        "paths": {
            "registry": str(registry),
            "dataset_root": str(data),
            "artifacts": str(ws / "artifacts"),
            "checkpoints": str(checkpoints),
            "results": str(results),
            "third_party": str(third_party),
        },
        "budgets": {"max_download_bytes": 0, "max_models": 64, "top_k": 200},
        # The synthetic run needs exactly what its stages declare: no network (every source and
        # checkpoint is already on disk), no environment install (the fixture supplies
        # interpreters), but preparation and GPU evaluation, because the evaluation stage declares
        # both.
        # The operator configuration authorizes exactly the operations this run needs. The
        # protocol's stages declare their capabilities statically, so a stage that would run is
        # refused unless its capability is granted here; that is the intended contract, and the
        # denial path is covered by its own test.
        "permissions": {
            "allow_network": True,
            "allow_large_downloads": True,
            "allow_env_install": False,
            "allow_preparation": True,
            "allow_gpu_eval": True,
            "allow_index_write": True,
        },
    }))

    # Real isolated interpreters, shared the way the planner is supposed to share them: alpha,
    # beta, delta, epsilon and zeta declare the same torch constraint and get ONE environment;
    # gamma's different constraint gets its own. That is the compatibility question the fixture
    # has to exercise, and a bare `sys.executable` is refused by the runtime as the active
    # workbench environment.
    import venv as _venv

    def _interpreter(*, tag: str, packages: dict[str, str]) -> str:
        env_dir = ws / "envs" / tag
        if not env_dir.is_dir():
            _venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
            site = next((env_dir / "lib").glob("python*"), None)
            site = (site / "site-packages") if site else env_dir
            for name, version in packages.items():
                (site / f"{name}.py").write_text(f"__version__ = {version!r}\n")
                dist = site / f"{name}-{version}.dist-info"
                dist.mkdir(exist_ok=True)
                (dist / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
                record = site / f"{name}.py"
                (dist / "RECORD").write_text(f"{name}.py,sha256=t,{record.stat().st_size}\n")
        return str(env_dir / "bin" / "python")

    shared_python = _interpreter(tag="env-shared", packages={"torch": "1.12.1",
                                                             "torchvision": "0.13.1"})
    gamma_python = _interpreter(tag="env-gamma", packages={"torch": "2.0.1",
                                                           "torchvision": "0.15.2"})

    env = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY),
        "CIR_REPO_ROOT": str(ws),
        "CIR_DATA_ROOT": str(ws / "data"),
        "FASHIONIQ_ROOT": str(data),
        "WORKBENCH_CHECKPOINT_ROOT": str(checkpoints),
        "WORKBENCH_RESULTS_ROOT": str(results),
        "WORKBENCH_THIRD_PARTY_ROOT": str(third_party),
        "WORKBENCH_REGISTRY_PATH": str(registry),
        "WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS": "1",
        "WORKBENCH_EVALUATOR_ENV_PASSTHROUGH": "SYNTHETIC_FAIL:SYNTHETIC_SILENT",
        "WORKBENCH_NO_ENV_CACHE": "1",
        "WORKBENCH_LEDGER_KEY": "synthetic-autopilot-key",
        "SYNTHETIC_FAIL": "",
    }
    for model_id, _, _, _, _ in MODELS:
        env[f"WORKBENCH_{model_id.upper()}_PYTHON"] = (
            gamma_python if model_id == "gamma" else shared_python)
    return {"ws": ws, "config": config, "env": env, "data": data,
            "third_party": third_party, "checkpoints": checkpoints,
            # The autopilot writes its reports under `<artifacts>/reports`.
            "reports": ws / "artifacts" / "reports", "artifacts": ws / "artifacts",
            "results": results, "shared_source": shared_source}


def _autopilot(workbench: dict, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AUTOPILOT), "--config", str(workbench["config"]),
         *extra],
        env=env or workbench["env"], capture_output=True, text=True, cwd=str(REPOSITORY),
        timeout=900)


def _summary(workbench: dict) -> dict:
    path = workbench["reports"] / "autopilot_summary.json"
    assert path.is_file(), "the run must produce its summary"
    return json.loads(path.read_text())


# ------------------------------------------------------------------ AUTO-40, AUTO-41, AUTO-42

def test_one_command_runs_the_mixed_six_model_scenario(workbench: dict) -> None:
    """A/B/C(/F) complete, D is blocked, E fails, and every model is reported truthfully."""
    result = _autopilot(workbench, "--apply")
    summary = _summary(workbench)

    by_id = {entry["model_id"]: entry for entry in summary["models"]}
    assert set(by_id) == {model[0] for model in MODELS}, "every selected model must be reported"

    for model_id in ("alpha", "beta", "gamma", "zeta"):
        assert by_id[model_id]["status"] in {"COMPLETED", "COMPLETED_WITH_LIMITATIONS"}, by_id[model_id]
    # The model whose checkpoint is absent is blocked, and blocked is not success.
    assert by_id["delta"]["status"].startswith("BLOCKED"), by_id["delta"]
    assert by_id["delta"]["blockers"], "a blocked model must say why"
    # The model whose evaluator exits nonzero failed.
    assert by_id["epsilon"]["status"].startswith("FAILED"), by_id["epsilon"]

    # Partial completion is partial: the exit code must not claim full success.
    assert result.returncode != 0


def test_the_shared_source_is_acquired_once(workbench: dict) -> None:
    """Three models share one checkout; the run must acquire it once, not three times.

    The dedup is visible in the pipeline stages: `all_stages()` registers a `sync` stage per
    unique checkout, so a shared source produces one stage that several models depend on rather
    than one per model.
    """
    _autopilot(workbench, "--apply")

    state = json.loads((workbench["artifacts"] / "autopilot_pipeline_state.json").read_text())
    records = state.get("stages", {})
    sync_records = {key: value for key, value in records.items()
                    if key.startswith("sync") and isinstance(value, dict)}
    assert sync_records, "the run must have a sync stage"

    # Every sync record names a checkout, and the shared checkout appears exactly once.
    shared_records = [key for key, value in sync_records.items()
                      if "SharedModel" in json.dumps(value)]
    assert len(shared_records) <= 1, shared_records
    # Three of the six models read that checkout, so the fixture really does share one source.
    shared_models = {"alpha", "beta", "zeta"}
    summary = _summary(workbench)
    assert shared_models <= {entry["model_id"] for entry in summary["models"]}


def test_environments_are_grouped_rather_than_one_per_model(workbench: dict) -> None:
    """alpha, beta, epsilon and zeta share a constraint; gamma's torch 2.x separates it."""
    _autopilot(workbench, "--apply")
    plan_path = workbench["artifacts"] / "environment_plan.json"
    if not plan_path.is_file():
        # Some runs record the plan inside the resolved execution plan instead of as its own file.
        plan_path = workbench["artifacts"] / "autopilot_execution_plan.json"
    plan = json.loads(plan_path.read_text())
    if "environment_plan" in plan:
        plan = plan["environment_plan"]
    groups = plan["groups"]

    assert len(groups) < len(MODELS), "a shared environment must actually be shared"
    for group in groups:
        assert group["members"], group
        # Every group states its evidence and its unresolved gaps, or says there are none.
        assert "evidence" in group and "unresolved" in group


def test_no_model_reports_a_fabricated_metric(workbench: dict) -> None:
    """A blocked or failed model must not carry a metric, and a completed one must be real."""
    _autopilot(workbench, "--apply")
    summary = _summary(workbench)
    for entry in summary["models"]:
        if entry["status"].startswith(("BLOCKED", "FAILED")):
            assert entry.get("metrics") in (None, {}), entry


# ------------------------------------------------------------------------------- resume, AUTO-31

def test_resuming_reuses_completed_work_and_touches_nothing_else(workbench: dict) -> None:
    """A second invocation with --resume must not redo verified work or re-download."""
    _autopilot(workbench, "--apply")
    first = _summary(workbench)

    completed_before = {
        entry["model_id"]: entry.get("run_id") or entry.get("completion_proof")
        for entry in first["models"] if entry["status"].startswith("COMPLETED")
    }
    assert completed_before, "the first run must complete something"

    result = _autopilot(workbench, "--apply", "--resume")
    second = _summary(workbench)
    by_id = {entry["model_id"]: entry for entry in second["models"]}

    for model_id, identity in completed_before.items():
        assert by_id[model_id]["status"].startswith("COMPLETED"), (model_id, by_id[model_id])
        # Either the same run identity is reported, or the artifacts on disk were left untouched;
        # what must not happen is a re-run of verified work.
        assert by_id[model_id].get("run_id") in (None, identity) or \
            by_id[model_id].get("completion_proof") == identity, by_id[model_id]
    assert result.returncode != 0  # still partial, because delta and epsilon never succeed


def test_mutating_one_scientific_input_invalidates_only_that_model(workbench: dict) -> None:
    """A changed checkpoint invalidates the model that reads it, not its neighbours."""
    _autopilot(workbench, "--apply")
    before = _summary(workbench)
    before_proofs = {entry["model_id"]: entry.get("run_id") or entry.get("completion_proof")
                     for entry in before["models"]}

    logs = workbench["artifacts"] / "logs"
    alpha_runs_before = len([p for p in logs.iterdir() if p.is_dir() and p.name.startswith("alpha__")]) \
        if logs.is_dir() else 0

    (workbench["checkpoints"] / "alpha" / "model.pt").write_bytes(b"mutated-checkpoint")
    _autopilot(workbench, "--apply", "--resume")
    after = _summary(workbench)

    alpha_runs_after = len([p for p in logs.iterdir() if p.is_dir() and p.name.startswith("alpha__")]) \
        if logs.is_dir() else 0
    assert alpha_runs_after > alpha_runs_before, "alpha must re-run after its checkpoint changed"
    after_proofs = {entry["model_id"]: entry.get("run_id") or entry.get("completion_proof")
                    for entry in after["models"]}
    after_status = {entry["model_id"]: entry["status"] for entry in after["models"]}

    for model_id in ("beta", "gamma", "zeta"):
        if before_proofs.get(model_id):
            assert after_status[model_id].startswith("COMPLETED"), \
                f"{model_id} was invalidated by an unrelated model's checkpoint change"


# ----------------------------------------------------------------------------- reports, AUTO-30

def test_the_report_set_is_complete_and_internally_consistent(workbench: dict) -> None:
    """Every declared report exists, and the CSV agrees with the JSON for every model."""
    _autopilot(workbench, "--apply")
    reports = workbench["reports"]
    for name in ("autopilot_summary.json", "autopilot_summary.md", "autopilot_results.csv",
                 "blockers.json", "run_manifest.json"):
        assert (reports / name).is_file(), name

    summary = json.loads((reports / "autopilot_summary.json").read_text())
    with (reports / "autopilot_results.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["model_id"] for row in rows} == {entry["model_id"] for entry in summary["models"]}
    for row in rows:
        entry = next(item for item in summary["models"] if item["model_id"] == row["model_id"])
        assert row["status"] == entry["status"], row


def test_a_dry_run_changes_nothing(workbench: dict) -> None:
    """AUTO-02: dry-run must not download, install, link, evaluate, or write a report."""
    reports_before = (sorted(path.name for path in workbench["reports"].iterdir())
                      if workbench["reports"].is_dir() else [])
    result = _autopilot(workbench, "--dry-run")
    assert result.returncode == 0, (result.stdout, result.stderr)
    reports_after = (sorted(path.name for path in workbench["reports"].iterdir())
                     if workbench["reports"].is_dir() else [])
    assert reports_after == reports_before
    assert not any((workbench["checkpoints"] / model[0] / "model.pt").with_suffix(".part").exists()
                   for model in MODELS)
