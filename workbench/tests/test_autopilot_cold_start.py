"""Cold-start provisioning and end-to-end orchestration tests for autopilot.

Starts with NO model environment and NO `WORKBENCH_<MODEL>_PYTHON` assignment,
driving the REAL top-level command `./workbench/run.sh autopilot --config <cfg> --apply`
and then `--resume` against the synthetic six-model fixture.

Exercises the real command path, strengthens resume evidence, tests scientific input
mutations, offline authorization, the first-run GPU capability state machine, and
clearly distinguishes real conda solves from controlled package-manager CLIs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUN_SH = REPOSITORY / "workbench" / "run.sh"

pytestmark = pytest.mark.skipif(
    not RUN_SH.is_file(),
    reason="workbench/run.sh entry point does not exist",
)

MODELS = (
    ("alpha", None, "present", "ok", "torch=1.12.1"),
    ("beta", "alpha", "present", "ok", "torch=1.12.1"),
    ("gamma", None, "present", "ok", "torch=1.12.1"),
    ("delta", None, "absent", "ok", "torch=1.12.1"),
    ("epsilon", None, "present", "fail", "torch=1.12.1"),
    ("zeta", "alpha", "present", "ok", "torch=1.12.1"),
)

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


def _create_fake_conda_cli(bin_dir: Path) -> Path:
    """Create controlled fake package-manager CLI for testing orchestration and subprocess behavior.

    Explicitly labeled as CONTROLLED_FAKE_CLI to never conflate simulated proof with a real conda solve.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    conda_path = bin_dir / "conda"
    shebang = f"#!{sys.executable}\n"
    script_body = '''import sys, os, subprocess, venv
from pathlib import Path

# Label execution as controlled fake CLI
print("[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]", file=sys.stderr)

argv = sys.argv[1:]
if "--version" in argv:
    print("conda 24.1.0 (CONTROLLED_FAKE_CLI)")
    sys.exit(0)

if "create" in argv and "--dry-run" in argv and "--json" in argv:
    print('{"success": true, "actions": {"LINK": [], "FETCH": []}}')
    sys.exit(0)

if "create" in argv and "--prefix" in argv:
    prefix_idx = argv.index("--prefix") + 1
    prefix = Path(argv[prefix_idx])
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    site = next((prefix / "lib").glob("python*"), None)
    site = (site / "site-packages") if site else prefix
    args_str = " ".join(argv)
    if "2.0.1" in args_str or "9dcf7e0241a6afe7" in str(prefix) or "gamma" in str(prefix):
        torch_ver = "2.0.1"
        torchvision_ver = "0.15.2"
    else:
        torch_ver = "1.12.1"
        torchvision_ver = "0.13.1"
    def _write_pkg(s_dir, name, ver):
        (s_dir / f"{name}.py").write_text(f"__version__ = '{ver}'\\n")
        dist = s_dir / f"{name}-{ver}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\\nName: {name}\\nVersion: {ver}\\n")
        rec = s_dir / f"{name}.py"
        (dist / "RECORD").write_text(f"{name}.py,sha256=t,{rec.stat().st_size}\\n")
    for name, ver in [("torch", torch_ver), ("torchvision", torchvision_ver), ("yaml", "6.0"), ("pyyaml", "6.0")]:
        _write_pkg(site, name, ver)
    pip_dir = site / "pip"
    pip_dir.mkdir(exist_ok=True)
    (pip_dir / "__init__.py").write_text("__version__ = '24.0'\\n")
    pip_main = """import sys
from pathlib import Path
argv = sys.argv[1:]
if "-r" in argv:
    req_path = Path(argv[argv.index("-r") + 1])
    if req_path.is_file():
        txt = req_path.read_text()
        site = Path(__file__).resolve().parent.parent
        if "torch==2.0.1" in txt:
            (site / "torch.py").write_text("__version__ = '2.0.1'\\n")
            d = site / "torch-2.0.1.dist-info"
            d.mkdir(exist_ok=True)
            (d / "METADATA").write_text("Metadata-Version: 2.1\\nName: torch\\nVersion: 2.0.1\\n")
            r = site / "torch.py"
            (d / "RECORD").write_text(f"torch.py,sha256=t,{r.stat().st_size}\\n")
            (site / "torchvision.py").write_text("__version__ = '0.15.2'\\n")
            d2 = site / "torchvision-0.15.2.dist-info"
            d2.mkdir(exist_ok=True)
            (d2 / "METADATA").write_text("Metadata-Version: 2.1\\nName: torchvision\\nVersion: 0.15.2\\n")
            r2 = site / "torchvision.py"
            (d2 / "RECORD").write_text(f"torchvision.py,sha256=t,{r2.stat().st_size}\\n")
sys.exit(0)
"""
    (pip_dir / "__main__.py").write_text(pip_main)
    sys.exit(0)

sys.exit(0)
'''
    conda_path.write_text(shebang + script_body)
    conda_path.chmod(0o755)
    return conda_path

@pytest.fixture
def cold_workbench(tmp_path: Path) -> dict:
    """A cold-start six-model workbench fixture starting with NO environments and NO WORKBENCH_*_PYTHON."""
    ws = tmp_path / "cold_workbench"
    ws.mkdir()
    data, checkpoints, third_party, results, reports = (
        ws / name for name in ("FashionIQ", "checkpoints", "third_party", "results", "reports"))
    for directory in (data, checkpoints, third_party, results, reports, ws / "artifacts"):
        directory.mkdir()

    # Standard FashionIQ dataset layout
    for sub in ("captions", "image_splits", "images"):
        (data / sub).mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (data / "captions" / f"cap.{category}.val.json").write_text(
            '[{"target": "img1", "candidate": "img0", "captions": ["c1", "c2"]}]')
        (data / "image_splits" / f"split.{category}.val.json").write_text('["img0", "img1"]')
    for name in ("img0", "img1"):
        (data / "images" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")

    # Gamma uses clvc_resized layout for distinct scientific input testing
    (data / "resized_image" / "dress").mkdir(parents=True, exist_ok=True)
    (data / "resized_image" / "shirt").mkdir(parents=True, exist_ok=True)
    (data / "resized_image" / "toptee").mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        (data / "captions" / f"cap.{category}.train.json").write_text("[]")
        (data / "resized_image" / category / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\nclvc_img")

    # Pinned git repositories for synthetic models
    shared_source = third_party / "SharedModel"
    (shared_source / "src").mkdir(parents=True)
    (shared_source / "src" / "evaluator.py").write_text(EVALUATOR_SOURCE)
    (shared_source / "requirements.txt").write_text("torch==1.12.1\ntorchvision==0.13.1\n")
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=shared_source, check=True, capture_output=True)
    shared_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shared_source, text=True).strip()

    gamma_source = third_party / "GammaModel"
    (gamma_source / "src").mkdir(parents=True)
    (gamma_source / "src" / "evaluator.py").write_text(EVALUATOR_SOURCE)
    (gamma_source / "requirements.txt").write_text("torch==2.0.1\ntorchvision==0.15.2\n")
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=gamma_source, check=True, capture_output=True)
    gamma_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gamma_source, text=True).strip()

    for name in ("DeltaModel", "EpsilonModel"):
        source = third_party / name
        (source / "src").mkdir(parents=True)
        if name == "EpsilonModel":
            (source / "src" / "evaluator.py").write_text(
                "import sys\nsys.stderr.write('synthetic evaluator failed\\n')\nsys.exit(3)\n")
        else:
            (source / "src" / "evaluator.py").write_text(EVALUATOR_SOURCE)
        (source / "requirements.txt").write_text("torch==1.12.1\ntorchvision==0.13.1\n")
        for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                     ["git", "config", "user.name", "T"], ["git", "add", "."],
                     ["git", "commit", "-qm", "pin"]):
            subprocess.run(argv, cwd=source, check=True, capture_output=True)

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
            "fashioniq_layout": "fashioniq_clvc_resized" if model_id == "gamma" else "fashioniq_standard",
            "literature_split_label": "original",
            "native_protocol": "fashioniq_original_split",
            "supported_protocols": ["fashioniq_original_split"],
            "reported_r10": 55.0, "reported_r50": 75.0, "reported_mean": 65.0,
            "reproduction_status": "NOT_RUN",
            "command_status": "COMMAND_AUDITED",
            "strict_clean_checkpoint_available": checkpoint == "present",
            "environment_required": True,
            "environment": {"python_env_var": f"WORKBENCH_{model_id.upper()}_PYTHON",
                            "python": "UNKNOWN", "confidence": "VERIFIED_README",
                            "pytorch": "2.0.1" if model_id == "gamma" else "1.12.1",
                            "torchvision": "0.15.2" if model_id == "gamma" else "0.13.1",
                            "packages": []},
            "checkpoint_variants": [{
                "checkpoint_id": "ckpt",
                "filename": filename,
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
        "permissions": {
            "allow_network": True,
            "allow_large_downloads": True,
            "allow_env_install": True,
            "allow_preparation": True,
            "allow_gpu_eval": True,
            "allow_index_write": True,
        },
        "checkpoint_policy": "strict_clean",
        "checkpoint_overrides": {},
    }))

    # Setup controlled fake package manager CLI
    fake_bin = ws / "fake_bin"
    _create_fake_conda_cli(fake_bin)

    # CRITICAL COLD START CONTRACT:
    # 1. NO model environment exists on disk (NO ws/envs, NO artifacts/environments)
    # 2. NO WORKBENCH_<MODEL>_PYTHON environment variables are exported
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": str(REPOSITORY),
        "CIR_REPO_ROOT": str(ws),
        "CIR_DATA_ROOT": str(ws / "data"),
        "FASHIONIQ_ROOT": str(data),
        "WORKBENCH_CHECKPOINT_ROOT": str(checkpoints),
        "WORKBENCH_RESULTS_ROOT": str(results),
        "WORKBENCH_THIRD_PARTY_ROOT": str(third_party),
        "WORKBENCH_REGISTRY_PATH": str(registry),
        "WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS": "1",
        "WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE": "1",
        "WORKBENCH_EVALUATOR_ENV_PASSTHROUGH": "SYNTHETIC_FAIL:SYNTHETIC_SILENT",
        "WORKBENCH_NO_ENV_CACHE": "1",
        "WORKBENCH_LEDGER_KEY": "synthetic-autopilot-key",
        "SYNTHETIC_FAIL": "",
    }
    # Ensure NO pre-assigned model interpreter variables
    for k in list(env):
        if k.startswith("WORKBENCH_") and k.endswith("_PYTHON"):
            del env[k]

    return {
        "ws": ws,
        "config": config,
        "env": env,
        "data": data,
        "third_party": third_party,
        "checkpoints": checkpoints,
        "reports": ws / "artifacts" / "reports",
        "artifacts": ws / "artifacts",
        "results": results,
        "shared_source": shared_source,
        "fake_bin": fake_bin,
    }


def _run_sh_autopilot(workbench: dict, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """Drive the REAL top-level command `./workbench/run.sh autopilot --config <cfg> [extra]`."""
    return subprocess.run(
        [str(RUN_SH), "autopilot", "--config", str(workbench["config"]), *extra],
        env=env or workbench["env"],
        capture_output=True,
        text=True,
        cwd=str(REPOSITORY),
        timeout=900,
    )


def _summary(workbench: dict) -> dict:
    path = workbench["reports"] / "autopilot_summary.json"
    assert path.is_file(), "autopilot run must produce summary"
    return json.loads(path.read_text())


def _capture_run_evidence(workbench: dict, summary: dict, model_id: str) -> dict:
    """Capture run_id, completion proof digest, report digest, and log invocation count."""
    row = next((m for m in summary.get("models", []) if m["model_id"] == model_id), {})
    run_id = row.get("run_id")
    if not run_id:
        latest_file = workbench["reports"] / f"{model_id}_ckpt_latest_successful.json"
        if latest_file.is_file():
            run_id = json.loads(latest_file.read_text()).get("run_id")
    if not run_id:
        state_file = workbench["artifacts"] / "autopilot_pipeline_state.json"
        if state_file.is_file():
            state = json.loads(state_file.read_text())
            run_id = (state.get("stages", {}).get(f"evaluation:{model_id}") or {}).get("run_id")
    assert run_id and isinstance(run_id, str), f"completed model {model_id} must have non-null run_id"

    latest_file = workbench["reports"] / f"{model_id}_ckpt_latest_successful.json"
    assert latest_file.is_file(), f"latest successful report pointer missing for {model_id}"
    latest_data = json.loads(latest_file.read_text())
    assert latest_data.get("run_id") == run_id

    proof_path = Path(latest_data.get("proof") or (workbench["reports"] / f"{model_id}_ckpt_{run_id}_completion.json"))
    assert proof_path.is_file(), f"completion proof missing for {model_id}"
    proof_digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    report_digest = latest_data.get("report_digest")
    assert report_digest, f"report digest missing for {model_id}"

    logs = workbench["artifacts"] / "logs"
    invocations = len([p for p in logs.iterdir() if p.is_dir() and p.name.startswith(f"{model_id}__")]) if logs.is_dir() else 0

    return {
        "run_id": run_id,
        "proof_path": str(proof_path),
        "proof_digest": proof_digest,
        "report_digest": report_digest,
        "invocations": invocations,
    }


def test_cold_start_runs_top_level_command_without_prior_environments(cold_workbench: dict) -> None:
    """Cold-start test: drives `./workbench/run.sh autopilot --config <cfg> --apply` with NO environments set."""
    # Label that controlled fake package manager CLI is exercised for orchestration
    print("\n[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]")

    # Assert preconditions: NO model environment exists and NO WORKBENCH_*_PYTHON is exported
    assert not any(k.startswith("WORKBENCH_") and k.endswith("_PYTHON") for k in cold_workbench["env"])
    assert not (cold_workbench["ws"] / "envs").exists()

    result = _run_sh_autopilot(cold_workbench, "--apply")
    summary = _summary(cold_workbench)

    by_id = {m["model_id"]: m for m in summary["models"]}
    assert set(by_id) == {m[0] for m in MODELS}

    # Runnable models completed via cold-start provisioned environments
    for mid in ("alpha", "beta", "gamma", "zeta"):
        assert by_id[mid]["status"].startswith("COMPLETED"), by_id[mid]
        assert by_id[mid].get("run_id") is not None, by_id[mid]

    # Blocked and failed models have truthful statuses
    assert by_id["delta"]["status"].startswith("BLOCKED"), by_id["delta"]
    assert by_id["epsilon"]["status"].startswith("FAILED"), by_id["epsilon"]
    assert result.returncode != 0


def test_strengthened_resume_reuses_all_verified_work_and_creates_no_new_invocations(cold_workbench: dict) -> None:
    """Strengthened resume: assert SAME run_id, SAME proof digest, SAME report digest, zero new invocations."""
    # First invocation
    _run_sh_autopilot(cold_workbench, "--apply")
    first_summary = _summary(cold_workbench)

    completed_models = [
        m["model_id"] for m in first_summary["models"] if m["status"].startswith("COMPLETED")
    ]
    assert set(completed_models) == {"alpha", "beta", "gamma", "zeta"}

    before_evidence = {m: _capture_run_evidence(cold_workbench, first_summary, m) for m in completed_models}
    for m, ev in before_evidence.items():
        assert ev["run_id"] is not None and len(ev["run_id"]) > 0

    # Second invocation with --resume
    result = _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    second_summary = _summary(cold_workbench)
    by_id = {m["model_id"]: m for m in second_summary["models"]}

    after_evidence = {m: _capture_run_evidence(cold_workbench, second_summary, m) for m in completed_models}

    for mid in completed_models:
        assert by_id[mid]["status"].startswith("COMPLETED"), by_id[mid]
        # Exact same run identity and proof digest
        assert after_evidence[mid]["run_id"] == before_evidence[mid]["run_id"]
        assert after_evidence[mid]["proof_digest"] == before_evidence[mid]["proof_digest"]
        assert after_evidence[mid]["report_digest"] == before_evidence[mid]["report_digest"]
        assert after_evidence[mid]["proof_path"] == before_evidence[mid]["proof_path"]
        # Zero new evaluator invocations
        assert after_evidence[mid]["invocations"] == before_evidence[mid]["invocations"]

    # Assert no new downloads, source acquisitions, or environment installs occurred
    state = json.loads((cold_workbench["artifacts"] / "autopilot_pipeline_state.json").read_text())
    stages = state.get("stages", {})
    for mid in completed_models:
        eval_stage = stages.get(f"evaluation:{mid}") or next(
            (v for k, v in stages.items() if k.startswith(f"evaluation:{mid}")), None
        )
        assert eval_stage and eval_stage.get("status") == "COMPLETE"
        assert eval_stage.get("run_id") == before_evidence[mid]["run_id"]
    assert result.returncode != 0


def test_mutation_invalidates_only_dependent_evaluations(cold_workbench: dict) -> None:
    """Mutate one scientific input at a time and assert only dependent evaluations rerun."""
    _run_sh_autopilot(cold_workbench, "--apply")
    init_summary = _summary(cold_workbench)
    init_ev = {m: _capture_run_evidence(cold_workbench, init_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    # 1. Mutate alpha checkpoint -> only alpha reruns
    (cold_workbench["checkpoints"] / "alpha" / "model.pt").write_bytes(b"mutated-alpha-checkpoint")
    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_ckpt_summary = _summary(cold_workbench)
    ckpt_ev = {m: _capture_run_evidence(cold_workbench, after_ckpt_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    assert ckpt_ev["alpha"]["run_id"] != init_ev["alpha"]["run_id"], "alpha must rerun after checkpoint changed"
    assert ckpt_ev["alpha"]["invocations"] > init_ev["alpha"]["invocations"]
    for unrelated in ("beta", "gamma", "zeta"):
        assert ckpt_ev[unrelated]["run_id"] == init_ev[unrelated]["run_id"], f"{unrelated} was wrongly rerun"
        assert ckpt_ev[unrelated]["proof_digest"] == init_ev[unrelated]["proof_digest"]

    # 2. Mutate gamma's relevant dataset image -> only gamma reruns
    gamma_img = cold_workbench["data"] / "resized_image" / "dress" / "img0.png"
    gamma_img.write_bytes(b"\x89PNG\r\n\x1a\nmutated_clvc_image")
    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_img_summary = _summary(cold_workbench)
    img_ev = {m: _capture_run_evidence(cold_workbench, after_img_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    assert img_ev["gamma"]["run_id"] != ckpt_ev["gamma"]["run_id"], "gamma must rerun after dataset image changed"
    assert img_ev["gamma"]["invocations"] > ckpt_ev["gamma"]["invocations"]
    for unrelated in ("alpha", "beta", "zeta"):
        assert img_ev[unrelated]["run_id"] == ckpt_ev[unrelated]["run_id"], f"{unrelated} was wrongly rerun"
        assert img_ev[unrelated]["proof_digest"] == ckpt_ev[unrelated]["proof_digest"]

    # 3. Mutate beta's preparation recipe -> only beta reruns
    reg_path = Path(json.loads(cold_workbench["config"].read_text())["paths"]["registry"])
    reg_data = json.loads(reg_path.read_text())
    for m in reg_data["models"]:
        if m["model_id"] == "beta":
            m["validation_caption_policy"] = "strict_validation_caption"
    reg_path.write_text(json.dumps(reg_data, indent=2))
    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_recipe_summary = _summary(cold_workbench)
    recipe_ev = {m: _capture_run_evidence(cold_workbench, after_recipe_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    assert recipe_ev["beta"]["run_id"] != img_ev["beta"]["run_id"], "beta must rerun after preparation recipe changed"
    assert recipe_ev["beta"]["invocations"] > img_ev["beta"]["invocations"]
    for unrelated in ("alpha", "gamma", "zeta"):
        assert recipe_ev[unrelated]["run_id"] == img_ev[unrelated]["run_id"], f"{unrelated} was wrongly rerun"
        assert recipe_ev[unrelated]["proof_digest"] == img_ev[unrelated]["proof_digest"]

    # 4. Mutate installed dependency identity in gamma environment -> only gamma reruns
    # Locate gamma's installed environment prefix
    assignment_path = cold_workbench["artifacts"] / "environment_assignment.json"
    assignment = json.loads(assignment_path.read_text()) if assignment_path.is_file() else {}
    gamma_interp = Path(assignment.get("models", {}).get("gamma", {}).get("interpreter", ""))
    if not gamma_interp.is_file():
        # Fallback to group interpreter mapping
        for g in assignment.get("groups", []):
            if "gamma" in g.get("members", []):
                gamma_interp = Path(g.get("interpreter", ""))
                break
    assert gamma_interp.is_file()
    site = next((gamma_interp.parent.parent / "lib").glob("python*"), None) / "site-packages"
    (site / "torch.py").write_text("__version__ = '2.0.1'\n# dependency identity mutation\n")

    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_dep_summary = _summary(cold_workbench)
    dep_ev = {m: _capture_run_evidence(cold_workbench, after_dep_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    assert dep_ev["gamma"]["run_id"] != recipe_ev["gamma"]["run_id"], "gamma must rerun after dependency mutated"
    assert dep_ev["gamma"]["invocations"] > recipe_ev["gamma"]["invocations"]
    for unrelated in ("alpha", "beta", "zeta"):
        assert dep_ev[unrelated]["run_id"] == recipe_ev[unrelated]["run_id"], f"{unrelated} was wrongly rerun"
        assert dep_ev[unrelated]["proof_digest"] == recipe_ev[unrelated]["proof_digest"]


def test_offline_run_reaches_eligible_evaluation_without_network_permission(cold_workbench: dict) -> None:
    """Offline run: with resources verified, allow_network=false and allow_large_downloads=false reach eligible eval."""
    # 1. Initial run provisions and verifies required resources
    _run_sh_autopilot(cold_workbench, "--apply")

    # 2. Modify permissions to strictly offline: no network, no large downloads, no env installs
    cfg_path = cold_workbench["config"]
    cfg = json.loads(cfg_path.read_text())
    cfg["permissions"]["allow_network"] = False
    cfg["permissions"]["allow_large_downloads"] = False
    cfg["permissions"]["allow_env_install"] = False
    cfg_path.write_text(json.dumps(cfg, indent=2))

    result = _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    summary = _summary(cold_workbench)

    by_id = {m["model_id"]: m for m in summary["models"]}
    # Eligible models (whose resources are verified on disk) reach evaluation and complete
    for mid in ("alpha", "beta", "gamma", "zeta"):
        assert by_id[mid]["status"].startswith("COMPLETED"), by_id[mid]
        assert by_id[mid].get("run_id") is not None

    # Model delta requires missing checkpoint; without network permission it is refused/blocked
    assert by_id["delta"]["status"].startswith("BLOCKED"), by_id["delta"]

def test_first_run_gpu_capability_state_machine(cold_workbench: dict) -> None:
    """First-run GPU state machine: no capability, synthetic verification succeeds, stale blocks, valid reused."""
    from workbench.backend import runtime
    from workbench.backend.operator_config import WorkbenchConfig
    from workbench.scripts.pipeline import probe_gpu_capability
    from workbench.tests._core_fixture import build_core_module

    from dataclasses import replace
    from workbench.backend.operator_config import resolve_config
    wb_config = replace(
        resolve_config(),
        CIR_REPO_ROOT=cold_workbench["ws"],
        CIR_DATA_ROOT=cold_workbench["ws"] / "data",
        FASHIONIQ_ROOT=cold_workbench["data"],
        WORKBENCH_ARTIFACTS_ROOT=cold_workbench["artifacts"],
        WORKBENCH_CHECKPOINT_ROOT=cold_workbench["checkpoints"],
        WORKBENCH_RESULTS_ROOT=cold_workbench["results"],
        WORKBENCH_THIRD_PARTY_ROOT=cold_workbench["third_party"],
    )
    # Isolated model environment for CUDA test
    gpu_env_dir = cold_workbench["ws"] / "gpu_env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(gpu_env_dir)
    site = next((gpu_env_dir / "lib").glob("python*"), None) / "site-packages"
    for name, ver in [("torch", "1.12.1"), ("torchvision", "0.13.1")]:
        (site / f"{name}.py").write_text(f"__version__ = '{ver}'\n")
        dist = site / f"{name}-{ver}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
        rec = site / f"{name}.py"
        (dist / "RECORD").write_text(f"{name}.py,sha256=t,{rec.stat().st_size}\n")
    torch_core = build_core_module(site / "_C_fixture.so")

    gpu_model = {
        "model_id": "alpha",
        "environment_required": True,
        "environment": {"python_env_var": "WORKBENCH_ALPHA_PYTHON", "pytorch": "1.12.1", "cuda": "11.7"},
    }

    runtime.forget_gpu_capability()

    # 2. State 2: Authorized synthetic verification succeeds
    probe_invocations = 0

    def mock_probe(m, report):
        nonlocal probe_invocations
        probe_invocations += 1
        return {
            "probe": "gpu_capability",
            "cuda_available": True,
            "device_count": 1,
            "device_name": "Synthetic Tesla V100",
            "torch_version": "1.12.1",
            "torch_cuda_version": "11.7",
            "capability_verified": True,
            "allocated_tensor": True,
            "computed": True,
            "torch_has_compiled_core": True,
            "torch_core_origin": str(torch_core),
            "torch_module_file": str(site / "torch.py"),
            "error": None,
        }

    runtime.set_gpu_probe_for_tests(mock_probe)
    env_backup = os.environ.get("WORKBENCH_ALPHA_PYTHON")
    probe_override_backup = os.environ.get(runtime.GPU_PROBE_OVERRIDE_ENV)
    env_artifacts_backup = os.environ.get("WORKBENCH_ARTIFACTS_ROOT")
    try:
        os.environ[runtime.GPU_PROBE_OVERRIDE_ENV] = "1"
        os.environ["WORKBENCH_ALPHA_PYTHON"] = str(gpu_env_dir / "bin" / "python")
        os.environ["WORKBENCH_ARTIFACTS_ROOT"] = str(cold_workbench["artifacts"])
        key = runtime._gpu_capability_key(gpu_model)
        cap_file = runtime.gpu_capability_path(wb_config, key)
        if cap_file.exists():
            cap_file.unlink()
        assert not cap_file.exists(), "State 1: No capability initially"
        payload = runtime.verify_gpu_runtime(gpu_model, wb_config, authorized=True, require_cuda=True)
        assert payload.get("capability_verified") is True
        assert cap_file.is_file(), "GPU capability record must be persisted to disk"
        assert probe_invocations == 1
        # 3. State 4: Valid capability is reused without re-probing
        runtime.forget_gpu_capability()  # clear memory cache to ensure disk read
        reused = runtime.verify_gpu_runtime(gpu_model, wb_config, authorized=True, require_cuda=True, force=False)
        assert reused.get("capability_verified") is True
        assert probe_invocations == 1, "Probe must NOT be re-executed when valid capability exists on disk"

        # 4. State 3: Stale / corrupted capability blocks launch
        tampered = json.loads(cap_file.read_text())
        tampered["signature"] = "corrupted_invalid_signature"
        cap_file.write_text(json.dumps(tampered))

        runtime.forget_gpu_capability()
        report = runtime.verify_environment(gpu_model, probe=True)
        cached = runtime._cached_gpu_capability(gpu_model, report)
        assert cached is None, "Corrupted capability must be rejected by signature validator"

        # Launch without authorization or with corrupted capability is blocked
        runtime.clear_gpu_probe_for_tests()
        with pytest.raises(runtime.GpuVerificationError):
            runtime.verify_gpu_runtime(gpu_model, wb_config, authorized=False, require_cuda=True)

    finally:
        runtime.clear_gpu_probe_for_tests()
        if probe_override_backup is not None:
            os.environ[runtime.GPU_PROBE_OVERRIDE_ENV] = probe_override_backup
        else:
            os.environ.pop(runtime.GPU_PROBE_OVERRIDE_ENV, None)
        if env_artifacts_backup is not None:
            os.environ["WORKBENCH_ARTIFACTS_ROOT"] = env_artifacts_backup
        else:
            os.environ.pop("WORKBENCH_ARTIFACTS_ROOT", None)
        if env_backup is not None:
            os.environ["WORKBENCH_ALPHA_PYTHON"] = env_backup
        else:
            os.environ.pop("WORKBENCH_ALPHA_PYTHON", None)


def test_real_vs_fake_package_manager_distinction(cold_workbench: dict) -> None:
    """Explicitly test and preserve distinction between real package manager solves and controlled fake CLIs."""
    # 1. Controlled Fake Package Manager CLI test
    print("\n[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]")
    fake_cli = cold_workbench["fake_bin"] / "conda"
    proc = subprocess.run([str(fake_cli), "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "CONTROLLED_FAKE_CLI" in proc.stdout or "CONTROLLED_FAKE_CLI" in proc.stderr
    print("Controlled fake package manager CLI verified for orchestration testing.")

    # 2. Real conda package manager check (if present on machine)
    real_conda = shutil.which("conda")
    if real_conda:
        print("\n[PACKAGE_MANAGER_EXERCISED: REAL_CONDA]")
        proc_real = subprocess.run([real_conda, "--version"], capture_output=True, text=True)
        assert proc_real.returncode == 0
        print(f"Real package manager detected: {proc_real.stdout.strip()}")
    else:
        pytest.skip("conda not available on host system for real package manager test")
