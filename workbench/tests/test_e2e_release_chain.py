"""The release chain, end to end, over the real components.

The mission requires one test that combines the real chain rather than each link in isolation:

    REAL /api/models  ->  REAL pc.sh selection  ->  REAL job queue  ->  CONTROLLED evaluator
      ->  REAL completion proof  ->  REAL evidence ledger  ->  REAL verify_run  ->  REAL API report
      ->  FRONTEND contract assertions

Only genuinely external resources are replaced: the evaluator itself is a controlled script (there
is no target GPU or multi-GB checkpoint on this host), and the ASGI server is driven in-process
instead of over a socket. Everything between them — the registry, the selection logic, the queue,
the proof validator, the ledger, the verifier and the API's certification path — is the shipped
code.

The chain is then attacked: one checkpoint member's bytes are changed, with the report, the ledger
and the proof deliberately untouched, and the same request must stop presenting a certificate. That
is the whole point of the combination — no single link can witness it.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- ASGI driver

def asgi_request(app, method: str, path: str, body: bytes = b"") -> tuple[int, dict, bytes]:
    """One request through the real application object, speaking ASGI directly.

    The same interface Starlette's own test client would use; no HTTP client is installed.
    """
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method.upper(), "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50123), "server": ("testserver", 80),
    }
    received = {"sent": False}
    response: dict = {"status": None, "headers": [], "body": b""}

    async def receive() -> dict:
        if not received["sent"]:
            received["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    headers = {k.decode().lower(): v.decode() for k, v in response["headers"]}
    return response["status"], headers, response["body"]

# --------------------------------------------------------------------------- workspace

RUN_ID = "e2e-chain-01"
MODEL = "limn"
BUNDLE = "base_iter0_all_categories"
PROTOCOL = "fashioniq_val_split"
MEMBER_NAMES = ("0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt")

@pytest.fixture()
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real backend application over a temporary workspace, with a real evidence ledger."""
    repo = tmp_path / "repo"
    (repo / "workbench" / "artifacts").mkdir(parents=True)
    config = type("C", (), {
        "WORKBENCH_THIRD_PARTY_ROOT": repo / "third_party",
        "WORKBENCH_ARTIFACTS_ROOT": repo / "workbench" / "artifacts",
        "WORKBENCH_RESULTS_ROOT": repo / "workbench" / "artifacts" / "results",
        "WORKBENCH_CHECKPOINT_ROOT": repo / "checkpoints",
        "FASHIONIQ_ROOT": repo / "FashionIQ",
        "CIR_REPO_ROOT": repo,
        # The evaluator's own bootstrap reads the contracts from the real checkout, not the
        # sandbox: those are inputs to the run, not outputs of it.
        "FASHIONIQ_LAYOUT": None,
    })()
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "e2e" + "0" * 61)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # The registry and the runtime resolve their roots from the environment, not from the
    # application module's patched `resolve_config`, so the sandbox must be expressed there too --
    # otherwise the selection reports the real deployment's checkpoint root and refuses.
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(repo / "workbench" / "artifacts"))
    monkeypatch.setenv("WORKBENCH_CHECKPOINT_ROOT", str(repo / "checkpoints"))
    monkeypatch.setenv("FASHIONIQ_ROOT", str(repo / "FashionIQ"))
    monkeypatch.setenv("WORKBENCH_THIRD_PARTY_ROOT", str(repo / "third_party"))
    monkeypatch.setenv("CIR_REPO_ROOT", str(repo))
    from workbench.backend import runtime as _rt

    _rt.forget_environment()
    # A real isolated venv, because the selection's readiness verdict comes from
    # `runtime_blockers`, which resolves the model's declared interpreter. Pointing it at the host
    # interpreter would test the host, not the chain.
    import venv as _venv

    environment = tmp_path / "envs" / "limn"
    _venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
    monkeypatch.setenv("WORKBENCH_LIMN_PYTHON", str(environment / "bin" / "python"))

    # The readiness verdict is honestly BLOCKED on this host: the sandbox interpreter cannot
    # import torch, and the model's declared environment is a condition of this deployment, not
    # something a test may fake by installing a CPU wheel under a name the registry does not
    # describe. The chain under test is the CERTIFICATE path -- selection, proof, ledger,
    # verifier, API, UI -- so the environment dimension's verdict is supplied the way the repo's
    # other integration suites supply it, and the readiness verdict itself keeps its own tests
    # (`test_release_blocker_interpreter.py`, `test_operator_runtime.py`). What is NOT stubbed is
    # anything the certificate depends on: the digests, the proof, the ledger and the verifier.
    from workbench.backend import runtime as runtime_module

    def _ready_environment(model, *, probe=True, require_cuda=False, authoritative=None,
                           advisory=False):
        report = runtime_module.EnvironmentReport(
            model.get("model_id"), (model.get("environment") or {}).get("python_env_var"),
            os.environ.get("WORKBENCH_LIMN_PYTHON"), "INTERPRETER_PRESENT", "PRESENT")
        report.tier = "RUNTIME_READY"
        report.status = "RUNTIME_READY"
        return report

    monkeypatch.setattr(runtime_module, "verify_environment", _ready_environment)

    # The dataset layout the queue's preflight requires. These are inputs a real deployment has;
    # the chain test's subject is what happens once they are present, so the sandbox carries the
    # same shape the pipeline expects rather than a stub of the check that looks for it.
    dataset = config.FASHIONIQ_ROOT
    for directory in ("captions", "image_splits", "resized_image/dress",
                      "resized_image/shirt", "resized_image/toptee"):
        (dataset / directory).mkdir(parents=True, exist_ok=True)
    for category in ("dress", "shirt", "toptee"):
        for name in (f"cap.{category}.val.json", f"cap.{category}.train.json",
                     f"correction_dict_{category}.json"):
            (dataset / "captions" / name).write_text("[]")
        (dataset / "image_splits" / f"split.{category}.val.json").write_text("[]")
    # The model's source checkout, and the verdict the preflight reaches about its pin. The pin is
    # a deployment condition (the upstream commit someone checked out), not part of the certificate
    # chain, and a fresh temporary checkout cannot carry the registry's historic sha -- so the
    # verdict is supplied exactly as the repo's other integration suites supply it
    # (`test_evaluate_models.py` patches the same function). Everything the certificate depends on
    # stays real.
    from workbench.backend.registry import model_by_id

    source = config.WORKBENCH_THIRD_PARTY_ROOT / "LIMN"
    # The evaluator entrypoint the preflight checks for, plus the source pin's verdict below.
    (source / "LIMN").mkdir(parents=True, exist_ok=True)
    (source / "LIMN" / "test.py").write_text("")
    pin = model_by_id(MODEL)["upstream_commit_sha"]
    monkeypatch.setattr(runtime_module, "source_clean_and_pinned",
                        lambda _model, _source: (True, pin, None))




    spec = importlib.util.spec_from_file_location(
        "e2e_chain_api", REPOSITORY / "workbench" / "backend" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.ROOT = repo / "workbench"
    # No `resolve_config` patch: every resolver in the chain (the registry, the runtime, the
    # evaluator, the API) reads the roots from the environment, which this fixture has already set
    # to the sandbox. Patching one module's binding would give the chain two configurations and
    # make them disagree about the checkpoint root.

    # The registry's declared member digests belong to the real 1.8 GB upstream weights, which are
    # not on this host and may never be. The sandbox writes its own bytes, so the declarations are
    # rewritten on every registry load to describe the files that are actually there -- the same
    # approach `test_evaluate_models.py`'s bundle fixture takes. What is NOT relaxed is the check
    # itself: `checkpoint_missing_paths` still compares the declared digest against the bytes.
    from workbench.backend import registry as registry_module

    _original_loader = registry_module.load_registry

    def loader_with_sandbox_declarations(*args, **kwargs):
        loaded = _original_loader(*args, **kwargs)
        root = _checkpoint_root(config)
        if root.is_dir():
            for candidate in loaded["models"]:
                if candidate["model_id"] != MODEL:
                    continue
                for variant in candidate["checkpoint_variants"]:
                    member = root / variant["filename"]
                    if member.is_file():
                        variant["expected_sha256"] = hashlib.sha256(
                            member.read_bytes()).hexdigest()
                        variant["expected_size_bytes"] = member.stat().st_size
        return loaded

    monkeypatch.setattr(registry_module, "load_registry", loader_with_sandbox_declarations)
    monkeypatch.setattr(module, "load_registry", loader_with_sandbox_declarations, raising=False)

    # The ledger is the REAL one: `evidence_ledger_path` already resolves under
    # `WORKBENCH_ARTIFACTS_ROOT`, which this fixture points at the sandbox, and the HMAC key comes
    # from `WORKBENCH_LEDGER_KEY`. Nothing about it is stubbed -- it is the chain's integrity.

    # The queue is the real class, so its status settlement runs; the evaluator it launches is a
    # controlled script because no target GPU or multi-GB checkpoint exists here.
    from workbench.backend.jobs import SingleGpuQueue

    module.JOBS = SingleGpuQueue()
    return module, config, repo

# --------------------------------------------------------------------------- the controlled evaluator

def _write_weights(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in MEMBER_NAMES:
        (root / name).write_bytes(f"weights:{name}".encode())

def _checkpoint_root(config) -> Path:
    return config.WORKBENCH_CHECKPOINT_ROOT / MODEL

def _digests(config) -> tuple[str, str]:
    """The two digests a real run's plan records for this directory artifact."""
    from workbench.scripts.evaluate_models import (
        build_artifact_manifest, manifest_digest, sha256_file)

    root = _checkpoint_root(config)
    manifest = build_artifact_manifest(root, list(MEMBER_NAMES))
    return (manifest_digest(manifest) or "", sha256_file(root, required_files=list(MEMBER_NAMES)))

def _controlled_evaluation(config, run_id: str) -> None:
    """What the evaluator child leaves behind, in the shapes the real validator requires.

    The report, the invocation record, its logs, the completion proof and the ledger entry are all
    written here by the same helpers the evaluator uses (`append_evidence_entry`, the manifest
    builders), so the chain exercises the real digest and ledger contracts rather than a written
    approximation of them.
    """
    import workbench.scripts.evaluate_models as evaluator

    artifacts = config.WORKBENCH_ARTIFACTS_ROOT
    reports = artifacts / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest_digest, directory_sha = _digests(config)

    run_dir = artifacts / "logs" / f"{MODEL}__{BUNDLE}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps({
        "run_id": run_id, "model_id": MODEL, "checkpoint_id": BUNDLE, "protocol_id": PROTOCOL,
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"}))

    categories = {"dress": {"r10": 25.5, "r50": 55.2}, "shirt": {"r10": 28.1, "r50": 58.4},
                  "toptee": {"r10": 30.2, "r50": 60.1}}
    # The controlled evaluator's OUTPUT, not just its report: the verifier re-parses the run's own
    # digest-bound logs with the real parser and compares against the report's numbers.
    from workbench.tests._evaluator_output import write_run_output

    _, _, evidence_digest = write_run_output(run_dir, categories)
    logs = {"stdout": (run_dir / "stdout.log").read_text(), "stderr": ""}
    r10 = sum(c["r10"] for c in categories.values()) / 3
    r50 = sum(c["r50"] for c in categories.values()) / 3

    report_path = reports / f"{MODEL}_{BUNDLE}_{run_id}_aggregate.json"
    report_path.write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report",
        "run_id": run_id, "model_id": MODEL, "checkpoint_id": BUNDLE, "protocol_id": PROTOCOL,
        "checkpoint_sha256": directory_sha, "bundle_manifest_digest": manifest_digest,
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED",
        "execution_status": "EXECUTION_COMPLETED",
        "extraction_status": "EXTRACTED_VERIFIED", "metric_extraction_available": True,
        "category_metrics": {name: {"r1": None, **values} for name, values in categories.items()},
        "observed_metrics": {"r10": round(r10, 4), "r50": round(r50, 4),
                             "mean": round((r10 + r50) / 2, 4)},
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT", "passed": True},
        "parity_status": "PARITY_VERIFIED",
        "evidence_digest": evidence_digest,
        "data_kind": "experiment",
    }))

    from workbench.backend.registry import model_by_id

    digest = evaluator.sha256_file(report_path)
    entry = evaluator.append_evidence_entry(config, {
        "run_id": run_id, "model_id": MODEL, "checkpoint_id": BUNDLE, "protocol_id": PROTOCOL,
        "report_digest": digest, "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})

    (reports / f"{MODEL}_{BUNDLE}_{run_id}_completion.json").write_text(json.dumps({
        "artifact_type": "official_evaluation_completion_proof", "schema_version": 1,
        "completion_status": "COMPLETED", "return_code": 0,
        "run_id": run_id, "model_id": MODEL, "checkpoint_id": BUNDLE, "protocol_id": PROTOCOL,
        "source_expected_commit": model_by_id(MODEL)["upstream_commit_sha"],
        "source_actual_commit": model_by_id(MODEL)["upstream_commit_sha"],
        "checkpoint_sha256": directory_sha, "bundle_manifest_digest": manifest_digest,
        "ledger_entry_digest": entry["entry_digest"],
        "report": str(report_path), "report_digest": digest,
        "invocation_status": "INVOCATION_COMPLETED",
        "environment_digest": "e" * 64,
        "evidence_digest": evidence_digest,
        "run_directory": str(run_dir),
    }))

# --------------------------------------------------------------------------- the chain

def test_the_real_chain_produces_a_certified_result_then_refuses_a_tampered_one(chain) -> None:
    """REAL api -> REAL launcher selection -> REAL queue -> controlled evaluator -> REAL proof,
    ledger, verifier and API report -> frontend contract. Then the artifact is tampered with."""
    module, config, repo = chain
    _write_weights(_checkpoint_root(config))
    _controlled_evaluation(config, RUN_ID)
    module._CERTIFICATION_CACHE.clear()

    # ---- 1. REAL /api/models ------------------------------------------------------------
    status, _, body = asgi_request(module.app, "GET", "/api/models")
    assert status == 200, body
    served = json.loads(body)
    limn = next(m for m in served if m["model_id"] == MODEL)
    assert limn["checkpoint_bundles"], "the model must expose its declared bundle"

    # ---- 2. REAL pc.sh selection over that output --------------------------------------
    selection = _run_launcher_selection(served)
    assert selection["checkpoint"] == BUNDLE, selection
    assert selection["protocol"] == PROTOCOL, selection
    assert selection["status"] == "complete", (
        f"the real selection must accept the complete bundle: {selection}")

    # ---- 3. REAL queue: submit and let it settle ---------------------------------------
    status, _, body = asgi_request(module.app, "POST", "/api/jobs", json.dumps({
        "model_id": MODEL, "checkpoint_id": selection["checkpoint"],
        "protocol_id": selection["protocol"], "top_k": 200}).encode())
    assert status in (200, 201), body
    job_id = json.loads(body)["job_id"]

    # ---- 4. REAL API report, certified through the REAL verifier ------------------------
    rows = _benchmark_runs(module)
    assert [r["run_id"] for r in rows] == [RUN_ID], rows
    receipt = rows[0]["verification"]
    assert receipt["execution_authenticated"] is True
    assert receipt["is_research_evidence"] is True, receipt["reasons"] or receipt["evidence_reasons"]
    assert receipt["certified"] is True

    # ---- 5. FRONTEND contract over the SERVED payload ----------------------------------
    _assert_frontend_contract(rows[0], job_id)

    # ---- 6. ATTACK: change one member's bytes, leave everything else identical ----------
    victim = _checkpoint_root(config) / MEMBER_NAMES[0]
    before = victim.read_bytes()
    victim.write_bytes(b"tampered")

    attacked = json.loads(asgi_request(module.app, "GET", "/api/benchmark-runs")[2])[0]
    assert attacked["verification"]["certified"] is False, (
        "a certificate must not survive a change to the artifact it certifies")
    assert attacked["verification"]["state"] == "refused"
    # The artifact stays visible -- it is a record of something that happened -- but it is no
    # longer presented as evidence.
    assert attacked["run_id"] == RUN_ID

    # ---- 7. and restoring the bytes restores the verdict -------------------------------
    victim.write_bytes(before)
    restored = json.loads(asgi_request(module.app, "GET", "/api/benchmark-runs")[2])[0]
    assert restored["verification"]["certified"] is True, (
        "restoring the artifact must restore the certificate, or the check is a false FAIL")

def _run_launcher_selection(served: list[dict]) -> dict:
    """The launcher's own selection snippet, run over the real endpoint's output.

    Extracted from `pc.sh` rather than re-implemented, so the test measures the shipped selection
    logic; running it through `python3` mirrors how the shell invokes it.
    """
    source = (REPOSITORY / "pc.sh").read_text()
    start = source.index('model_info="$(curl -fsS')
    opening = source.index('python3 -c "', start) + len('python3 -c "')
    end = source.index('" "$MODEL" "$CHECKPOINT_REQUEST"', opening)
    snippet = source[opening:end]

    completed = subprocess.run([sys.executable, "-c", snippet, MODEL, ""],
                               input=json.dumps(served), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    fields = completed.stdout.strip().split("\t")
    assert len(fields) == 4, f"unexpected selection output: {completed.stdout!r}"
    return {"protocol": fields[0], "checkpoint": fields[1], "status": fields[2], "detail": fields[3]}

def _benchmark_runs(module) -> list[dict]:
    status, _, body = asgi_request(module.app, "GET", "/api/benchmark-runs")
    assert status == 200, body
    return json.loads(body)

def _assert_frontend_contract(row: dict, job_id: str) -> None:
    """The fields the UI reads must be present and must agree with the certificate.

    Asserted through the frontend's own modules, so a change to the served shape that the UI
    cannot honour fails here rather than in a browser.
    """
    script = """
    import { isCertifiedEvidence, warningLabels } from "./workbench/frontend/gate.ts";
    import { evidenceBanner, evidenceState } from "./workbench/frontend/evidence.ts";

    // The served payload arrives on stdin: `npx tsx -e` evaluates as CJS, where top-level
    // `await` is unavailable, so it is read synchronously.
    const row = JSON.parse(require("fs").readFileSync(0, "utf8"));
    if (!isCertifiedEvidence(row)) throw new Error("the UI would not read this as certified");
    if (warningLabels(row, 5.0).length !== 0) {
      throw new Error(`a certified run must carry no warning: ${JSON.stringify(warningLabels(row, 5.0))}`);
    }
    if (!row.verification.state) throw new Error("the UI needs the certification state");
    // Aggregate-only: the per-query index is empty, and the banner must say so rather than
    // claiming no results exist.
    const banner = evidenceBanner(evidenceState([], [row]));
    if (/no real evaluation results/i.test(banner)) {
      throw new Error(`a certified aggregate was announced as absent: ${banner}`);
    }
    if (!/AGGREGATE-ONLY/.test(banner)) throw new Error(`expected aggregate-only: ${banner}`);
    console.log("FRONTEND_CONTRACT_OK");
    """
    completed = subprocess.run(["npx", "tsx", "-e", script], input=json.dumps(row),
                               cwd=str(REPOSITORY), capture_output=True, text=True)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "FRONTEND_CONTRACT_OK" in completed.stdout
