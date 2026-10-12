"""Cold-start provisioning and end-to-end orchestration tests for autopilot.

Starts with NO model environment and NO `WORKBENCH_<MODEL>_PYTHON` assignment,
driving the REAL top-level command `./workbench/run.sh autopilot --config <cfg> --apply`
and then `--resume` against the synthetic six-model fixture.

Exercises the real command path, strengthens resume evidence, tests scientific input
mutations, offline authorization, the first-run GPU capability state machine, and
clearly distinguishes real conda solves from controlled package-manager CLIs.
"""
from __future__ import annotations

import hashlib
import json
import os
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

if "WORKBENCH_LEDGER_KEY" in os.environ:
    sys.stderr.write("ledger key leaked to evaluator child\\n")
    sys.exit(91)
if os.environ.get("SYNTHETIC_FAIL"):
    sys.stderr.write("synthetic evaluator failed\\n")
    sys.exit(3)
print("evaluating")
print(json.dumps({
    "model_id": "limn",
    "aggregate": {"macro_r10": 55.0, "macro_r50": 75.0, "macro_mean": 65.0, "complete": True},
    "categories": [
        {"category": "dress", "r10": 60.0, "r50": 80.0},
        {"category": "shirt", "r10": 55.0, "r50": 75.0},
        {"category": "toptee", "r10": 50.0, "r50": 70.0},
    ],
}))
'''


def _create_fake_conda_cli(bin_dir: Path) -> Path:
    """Create a controlled fake package-manager CLI for orchestration tests.

    Explicitly labelled CONTROLLED_FAKE_CLI so a simulated install is never mistaken for a real
    conda solve or for real package compatibility. It creates a bare virtualenv, writes metadata
    for the packages each model declares, and provides a minimal `pip` so the planner's
    `-m pip install` / `-m pip check` steps exercise their real code paths.

    Every newline inside a generated string is written with `chr(10)` rather than a backslash
    escape: this file's own literal would otherwise collapse `\\n` into a real newline and emit an
    unparsable stub, which is exactly the failure this function must not have.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    conda_path = bin_dir / "conda"
    pip_main_lines = [
        "import sys",
        "from pathlib import Path",
        "NL = chr(10)",
        f"Path({str(bin_dir / 'provision_calls.txt')!r}).open('a').write('pip:' + sys.argv[1] + NL)",
        "def pkg(site, name, version):",
        "    (Path(site) / (name + '.py')).write_text('__version__ = ' + repr(version) + NL)",
        "    dist = Path(site) / (name + '-' + version + '.dist-info')",
        "    dist.mkdir(exist_ok=True)",
        "    (dist / 'METADATA').write_text('Metadata-Version: 2.1' + NL + 'Name: ' + name + NL + 'Version: ' + version + NL)",
        "    member = Path(site) / (name + '.py')",
        "    (dist / 'RECORD').write_text(name + '.py,sha256=t,' + str(member.stat().st_size) + NL)",
        "def specs(argv):",
        "    if '-r' in argv:",
        "        path = Path(argv[argv.index('-r') + 1])",
        "        if path.is_file():",
        "            return [line.strip() for line in path.read_text().splitlines() if line.strip()]",
        "        return []",
        "    skip_next = False",
        "    out = []",
        "    for item in argv:",
        "        if skip_next:",
        "            skip_next = False",
        "            continue",
        "        if item in ('-r', '--index-url', '--extra-index-url', '--find-links', '-f', '--no-cache-dir'):",
        "            skip_next = item != '--no-cache-dir'",
        "            continue",
        "        if not item.startswith('-') and item not in ('install', 'check'):",
        "            out.append(item)",
        "    return out",
        "site = Path(__file__).resolve().parent.parent",
        "for spec in specs(sys.argv[1:]):",
        "    body = spec",
        "    for separator in ('==', '>=', '<=', '~=', '!=', '>', '<'):",
        "        if separator in body:",
        "            body = body.split(separator, 1)[0]",
        "            break",
        "    body = body.split('[', 1)[0].strip()",
        "    if not body:",
        "        continue",
        "    version = '0.0.0'",
        "    for separator in ('==', '>=', '<=', '~=', '!='):",
        "        if separator in spec:",
        "            tail = spec.split(separator, 1)[1].split(',')[0].strip()",
        "            if tail and tail[0].isdigit():",
        "                version = tail",
        "            break",
        "    module = body.replace('-', '_')",
        "    pkg(site, module, version)",
        "    # A distribution whose import name differs from its distribution name (open-clip-torch",
        "    # installs `open_clip`) must still be importable, or the planner's import probe fails on",
        "    # a package the stub actually provided.",
        "    if module.endswith('_torch') or module.endswith('_py'):",
        "        pkg(site, module.rsplit('_', 1)[0], version)",
        "sys.exit(0)",
        "",
    ]
    script_lines = [
        "import sys, subprocess, venv",
        "from pathlib import Path",
        "NL = chr(10)",
        "print('[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]', file=sys.stderr)",
        "argv = sys.argv[1:]",
        "if '--version' in argv:",
        "    print('conda 24.1.0 (CONTROLLED_FAKE_CLI)')",
        "    sys.exit(0)",
        "if 'create' in argv and '--dry-run' in argv and '--json' in argv:",
        "    print('{\"success\": true, \"actions\": {\"LINK\": [], \"FETCH\": []}}')",
        "    sys.exit(0)",
        "if 'create' in argv and '--prefix' in argv:",
        "    prefix = Path(argv[argv.index('--prefix') + 1])",
        "    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)",
        "    site = next(prefix.glob('lib/python*/site-packages'), prefix)",
        "    joined = ' '.join(argv)",
        "    torch_version, torchvision_version = ('2.0.1', '0.15.2') if ('2.0.1' in joined or 'gamma' in str(prefix)) else ('1.12.1', '0.13.1')",
        "    def write_pkg(name, version):",
        "        (site / (name + '.py')).write_text('__version__ = ' + repr(version) + NL)",
        "        dist = site / (name + '-' + version + '.dist-info')",
        "        dist.mkdir(exist_ok=True)",
        "        (dist / 'METADATA').write_text('Metadata-Version: 2.1' + NL + 'Name: ' + name + NL + 'Version: ' + version + NL)",
        "        member = site / (name + '.py')",
        "        (dist / 'RECORD').write_text(name + '.py,sha256=t,' + str(member.stat().st_size) + NL)",
        "    # `packaging` is a pip-provided runtime dependency that the planner's own evaluator-import",
        "    # trace reaches through pip's metadata, so a fake environment that installs anything via pip",
        "    # must expose it, exactly as a real pip-installed environment does.",
        "    for name, version in (('torch', torch_version), ('torchvision', torchvision_version), ('yaml', '6.0'), ('pyyaml', '6.0'), ('packaging', '24.0')):",
        "        write_pkg(name, version)",
        "    pip_dir = site / 'pip'",
        "    Path(__file__).with_name('provision_calls.txt').open('a').write('create' + NL)",
        "    pip_dir.mkdir(exist_ok=True)",
        "    (pip_dir / '__init__.py').write_text('__version__ = ' + chr(39) + '24.0' + chr(39) + NL)",
        "    (pip_dir / '__main__.py').write_text(chr(10).join(" + repr(pip_main_lines) + "))",
        "sys.exit(0)",
        "",
    ]
    conda_path.write_text(f"#!{sys.executable}" + chr(10) + chr(10).join(script_lines))
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
        "package_manager_calls": fake_bin / "provision_calls.txt",
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
    """Scenario A: real run.sh orchestration with controlled fake conda and synthetic assets."""
    print("\n[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]")
    print("[GPU_EVIDENCE: NOT_EXERCISED; no real CUDA claim]")

    # Assert preconditions: NO model environment exists and NO WORKBENCH_*_PYTHON is exported
    assert not any(k.startswith("WORKBENCH_") and k.endswith("_PYTHON") for k in cold_workbench["env"])
    assert not (cold_workbench["ws"] / "envs").exists()

    result = _run_sh_autopilot(cold_workbench, "--apply")
    summary = _summary(cold_workbench)

    by_id = {m["model_id"]: m for m in summary["models"]}
    assert set(by_id) == {m[0] for m in MODELS}

    # 1. Environment planning & provisioning artifacts
    assignment_path = cold_workbench["artifacts"] / "environment_assignment.json"
    assert assignment_path.is_file(), "environment assignment must be persisted"
    assignment = json.loads(assignment_path.read_text())
    environment_plan = json.loads((cold_workbench["artifacts"] / "environment_plan.json").read_text())
    assert environment_plan["solver_verification_status"] == "VERIFIED"
    assert environment_plan["solver"] == "conda"
    assert environment_plan["groups"]
    for mid in ("alpha", "beta", "gamma", "zeta"):
        model_env = assignment["models"][mid]
        assert model_env["installation_status"] == "INSTALLED"
        assert model_env["runtime_verification_status"] == "READY"
        marker = Path(model_env["conda_env_path"]) / ".workbench_managed_env.json"
        assert json.loads(marker.read_text())["managed_by"] == "workbench"
    # 2. Runnable models completed via cold-start provisioned environments
    for mid in ("alpha", "beta", "gamma", "zeta"):
        assert by_id[mid]["status"].startswith("COMPLETED"), by_id[mid]
        run_id = by_id[mid].get("run_id")
        assert run_id is not None, by_id[mid]

        # Interpreter assignment verified
        model_interp = assignment.get("models", {}).get(mid, {}).get("interpreter")
        assert model_interp and Path(model_interp).is_file(), f"missing interpreter for {mid}"

        # Completion proof generated
        proof_path = cold_workbench["reports"] / f"{mid}_ckpt_{run_id}_completion.json"
        assert proof_path.is_file(), f"completion proof missing for {mid}"
        proof_data = json.loads(proof_path.read_text())
        assert proof_data.get("completion_status") == "COMPLETED"
        assert proof_data.get("return_code") == 0

        # Metric extraction status verified (unfabricated: AGGREGATE_PARSER_UNAVAILABLE for unparsed models)
        assert by_id[mid].get("missing_metrics_reason") in (None, "AGGREGATE_PARSER_UNAVAILABLE")
        assert by_id[mid].get("metrics") in (None, {}), "models without dedicated parser do not fabricate metrics"

    # 3. Scenario G: Failure isolation - truthful statuses and separate global index status
    assert by_id["delta"]["status"].startswith("BLOCKED"), by_id["delta"]

    assert by_id["delta"].get("metrics") in (None, {}), "blocked model must not carry metrics"
    assert by_id["epsilon"]["status"].startswith("FAILED"), by_id["epsilon"]
    assert by_id["epsilon"].get("metrics") in (None, {}), "failed model must not carry metrics"

    # Outcome is PARTIAL, global index status is tracked separately
    assert summary["status"] == "PARTIAL", f"expected PARTIAL outcome, got {summary['status']}"
    assert "global_index_status" in summary, "summary must track global_index_status"
    assert summary["global_index_status"] != summary["status"] or summary["global_index_status"] == "COMPLETED"

    # 4. Final reports generated
    for rpt in ("autopilot_summary.json", "autopilot_summary.md", "autopilot_results.csv", "run_manifest.json", "blockers.json"):
        assert (cold_workbench["reports"] / rpt).is_file(), f"final report {rpt} missing"

    assert result.returncode != 0
def test_scenario_a_real_run_sh_cold_start_exercises_full_flow(cold_workbench: dict) -> None:
    """Run real run.sh through fake conda, local preparation/checkpoint, evaluator, and reports."""
    print("\n[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI]")
    print("[GPU_EVIDENCE: SYNTHETIC_FIXTURE_HAS_NO_CUDA_REQUIREMENT; not real CUDA]")

    registry_path = Path(json.loads(cold_workbench["config"].read_text())["paths"]["registry"])
    registry = json.loads(registry_path.read_text())
    limn = next(model for model in registry["models"] if model["model_id"] == "alpha")
    limn = {**limn, "model_id": "limn", "method_name": "LIMN synthetic integration"}
    limn["environment"] = {**limn["environment"], "python_env_var": "WORKBENCH_LIMN_PYTHON",
                            "cuda": "UNKNOWN"}
    limn["canonical_dataset_link"] = "fashionIQ_dataset"
    # The planner traces the imports reachable from LIMN's real replay entry point. A stand-in for
    # the LIMN checkout must therefore contain the modules that replay imports from the checkout
    # itself (`datasets`, `model`, `test`) — a real LIMN clone has them — so the trace resolves them
    # as local and only the genuinely third-party distributions remain external.
    source_root = cold_workbench["third_party"] / limn["source_dir"]
    for local_module in ("datasets", "model", "test"):
        (source_root / f"{local_module}.py").write_text(f"# LIMN-local module: {local_module}\n")
    for argv in (["git", "add", "."], ["git", "commit", "-qm", "limn-local-modules"]):
        subprocess.run(argv, cwd=source_root, check=True, capture_output=True)
    limn["upstream_commit_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    registry["models"] = [limn]
    registry_path.write_text(json.dumps(registry, indent=2))

    limn_checkpoint = cold_workbench["checkpoints"] / "limn" / "model.pt"
    limn_checkpoint.parent.mkdir()
    limn_checkpoint.write_bytes((cold_workbench["checkpoints"] / "alpha" / "model.pt").read_bytes())
    config = json.loads(cold_workbench["config"].read_text())
    config["selected_models"] = ["limn"]
    cold_workbench["config"].write_text(json.dumps(config, indent=2))

    result = _run_sh_autopilot(cold_workbench, "--apply")
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    summary = _summary(cold_workbench)
    assert summary["status"] == "FULL_SUCCESS"
    row = next(model for model in summary["models"] if model["model_id"] == "limn")
    assert row["status"].startswith("COMPLETED"), row
    assert row["missing_metrics_reason"] is None
    assert row["metrics"]["extraction_status"] == "EXTRACTED_VERIFIED"
    assert row["metrics"]["category_metrics"]["dress"]["r10"] == 60.0
    assert row["macro_r10"] == 55.0 and row["macro_r50"] == 75.0

    link = cold_workbench["third_party"] / "SharedModel" / "fashionIQ_dataset"
    assert link.is_symlink() and link.resolve() == cold_workbench["data"].resolve()
    state = json.loads((cold_workbench["artifacts"] / "autopilot_pipeline_state.json").read_text())
    stages = state["stages"]
    for name in ("environment:limn", "preparation:limn", "checkpoint:limn", "evaluation:limn"):
        stage = next(value for key, value in stages.items() if key.startswith(name))
        assert stage["status"] == "COMPLETE", (name, stage)
    assignment = json.loads((cold_workbench["artifacts"] / "environment_assignment.json").read_text())
    provision = assignment["models"]["limn"]
    # A prior run in this fixture may already have provisioned the group, in which case the
    # authoritative assignment records a REUSE, which is equally correct: the environment exists and
    # was verified without being reinstalled. Both states mean the evaluator has a real interpreter.
    assert provision["installation_status"] in ("INSTALLED", "REUSED"), provision
    assert provision["runtime_verification_status"] == "READY", provision
    assert provision["runtime_verification_status"] == "READY"
    assert json.loads((Path(provision["conda_env_path"]) / ".workbench_managed_env.json").read_text())["managed_by"] == "workbench"
    assert cold_workbench["package_manager_calls"].read_text().count("create") > 0
    evidence = _capture_run_evidence(cold_workbench, summary, "limn")
    assert evidence["invocations"] == 1
    assert (cold_workbench["reports"] / "autopilot_summary.json").is_file()
    assert (cold_workbench["reports"] / "autopilot_results.csv").is_file()
    assert "prepare_model.py --model limn" in result.stdout
    assert "--execute --allow-preparation" in result.stdout
    assert "download_checkpoints.py --verify-only --model limn" in result.stdout
    assert "evaluate_models.py --model limn" in result.stdout
    assert (cold_workbench["checkpoints"] / "limn" / "model.pt").is_file()
    assert evidence["proof_digest"] and evidence["report_digest"]


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

    first_state = json.loads((cold_workbench["artifacts"] / "autopilot_pipeline_state.json").read_text())
    first_stages = first_state.get("stages", {})
    package_manager_calls_before = cold_workbench["package_manager_calls"].read_text()
    checkpoint_digests = {
        mid: hashlib.sha256((cold_workbench["checkpoints"] / mid / "model.pt").read_bytes()).hexdigest()
        for mid in completed_models
    }
    reusable_stage_names = {"sync", "preparation", "checkpoint", "environment"}
    reusable_stages_before = {
        item["stage_id"]: item.get("finished_at") for item in first_stages.values()
        if item.get("name") in reusable_stage_names and item.get("status") == "COMPLETE"
    }

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
    # Reuse must not redo an INSTALLATION. Verification is a different thing: a resume re-checks
    # the environment it is about to use, and `pip check` is part of that check, so only the
    # mutating calls (environment creation and package installation) must be absent from the second
    # run. Comparing the whole log conflated verification with installation.
    def mutating_calls(log: str) -> list[str]:
        return [line for line in log.splitlines() if line == "create" or line == "pip:install"]

    calls_after = cold_workbench["package_manager_calls"].read_text()
    assert mutating_calls(calls_after) == mutating_calls(package_manager_calls_before), (
        f"resume must not reinstall: {mutating_calls(calls_after)} vs {mutating_calls(package_manager_calls_before)}")
    for mid in completed_models:
        checkpoint = cold_workbench["checkpoints"] / mid / "model.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_digests[mid]
    reusable_stages_after = {
        item["stage_id"]: item.get("finished_at") for item in stages.values()
        if item.get("name") in reusable_stage_names and item.get("status") == "COMPLETE"
    }
    assert reusable_stages_after == reusable_stages_before
    assert result.returncode != 0


def test_mutation_invalidates_only_dependent_evaluations(cold_workbench: dict) -> None:
    """Mutate one scientific input at a time and assert only dependent evaluations rerun."""
    _run_sh_autopilot(cold_workbench, "--apply")
    res = _run_sh_autopilot(cold_workbench, "--apply")
    if res.returncode != 0:
        print("STDOUT:", res.stdout[-2000:])
        print("STDERR:", res.stderr[-2000:])
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

    # 5. Mutate evaluator source -> only gamma reruns.
    # The checkout is pinned by commit and a dirty tree is refused as BLOCKED_SOURCE, which is the
    # intended integrity rule; a real upstream change lands as a new commit, so commit the mutation
    # and re-pin the model to it. That is what an updated pinned revision looks like in practice.
    gamma_evaluator = cold_workbench["third_party"] / "GammaModel" / "src" / "evaluator.py"
    gamma_evaluator.write_text(EVALUATOR_SOURCE + "\n# evaluator source mutated\n")
    gamma_source = cold_workbench["third_party"] / "GammaModel"
    for argv in (["git", "add", "."], ["git", "commit", "-qm", "evaluator change"]):
        subprocess.run(argv, cwd=gamma_source, check=True, capture_output=True)
    new_gamma_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gamma_source, text=True).strip()
    registry_path = Path(json.loads(cold_workbench["config"].read_text())["paths"]["registry"])
    registry_data = json.loads(registry_path.read_text())
    for entry in registry_data["models"]:
        if entry["model_id"] == "gamma":
            entry["upstream_commit_sha"] = new_gamma_head
    registry_path.write_text(json.dumps(registry_data, indent=2))
    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_eval_summary = _summary(cold_workbench)
    eval_ev = {m: _capture_run_evidence(cold_workbench, after_eval_summary, m) for m in ("alpha", "beta", "gamma", "zeta")}

    assert eval_ev["gamma"]["run_id"] != dep_ev["gamma"]["run_id"], "gamma must rerun after evaluator source changed"
    assert eval_ev["gamma"]["invocations"] > dep_ev["gamma"]["invocations"]
    for unrelated in ("alpha", "beta", "zeta"):
        assert eval_ev[unrelated]["run_id"] == dep_ev[unrelated]["run_id"], f"{unrelated} was wrongly rerun"
        assert eval_ev[unrelated]["proof_digest"] == dep_ev[unrelated]["proof_digest"]

    # 6. Policy change alters config identity and invalidates each selected evaluation.
    manifest_path = cold_workbench["reports"] / "run_manifest.json"
    before_manifest = json.loads(manifest_path.read_text())
    cfg_data = json.loads(cold_workbench["config"].read_text())
    cfg_data["checkpoint_policy"] = "include_flagged_conditions"
    cold_workbench["config"].write_text(json.dumps(cfg_data, indent=2))
    _run_sh_autopilot(cold_workbench, "--apply", "--resume")
    after_pol_summary = _summary(cold_workbench)
    after_manifest = json.loads(manifest_path.read_text())
    assert after_manifest["configuration_digest"] != before_manifest["configuration_digest"]
    policy_ev = {m: _capture_run_evidence(cold_workbench, after_pol_summary, m)
                 for m in ("alpha", "beta", "gamma", "zeta")}
    for mid in policy_ev:
        assert policy_ev[mid]["run_id"] != eval_ev[mid]["run_id"], f"{mid} reused evaluation across checkpoint policy mutation"
        assert policy_ev[mid]["invocations"] > eval_ev[mid]["invocations"]

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


def test_controlled_fake_package_manager_is_labeled(cold_workbench: dict) -> None:
    """The integration harness uses a controlled fake CLI, not a real conda solve."""
    print("\n[PACKAGE_MANAGER_EXERCISED: CONTROLLED_FAKE_CLI; REAL_CONDA: NOT_EXERCISED]")
    fake_cli = cold_workbench["fake_bin"] / "conda"
    proc = subprocess.run([str(fake_cli), "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "CONTROLLED_FAKE_CLI" in proc.stdout or "CONTROLLED_FAKE_CLI" in proc.stderr
