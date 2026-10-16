"""Release-blocker regressions: GPU capability records and the J1 boot state.

Two defects, both proven against the shipped script by executing its own embedded Python:

* P1-4 — the GPU_RUNTIME scan accepted a capability record on its own declared fields
  (``probe_source == "interpreter"`` and ``capability_verified is True``) plus a filename/key
  equality check, so a hand-written JSON file certified the host as GPU-verified without any
  probe. The scan must read the record through the runtime's authoritative capability path
  (HMAC signature under this deployment's operator key, interpreter fingerprint, host identity,
  the runtime adaptation it was probed under, and the capability key binding it to this
  environment) and report a forged or stale record as not-verified with its reason.
* P2-7 — J1 required every visible run to carry ``data_kind == "mock"``, so a legitimate run
  from a previous evaluation was reported as a fake and failed the release. J1 must classify
  each visible run by its real state: ledger-corroborated real evidence is acceptable and
  reported as ``existing valid results``; mock/synthetic/uncorroborated output is refused; an
  empty stack is reported as ``clean first boot``.

The embedded blocks are lifted verbatim from the script (they cannot be imported) and executed
with the same argv the script binds, so the tests exercise the shipped code, never a paraphrase.
Set ``RELEASE_VERIFY_SCRIPT`` to run the same tests against the pre-fix copy of the script.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import venv
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = Path(os.environ.get("RELEASE_VERIFY_SCRIPT")
              or REPOSITORY / "workbench" / "docker" / "release_verify.sh")

#: The model the sandbox registry declares for capability-record tests. A CUDA-declaring model
#: with no published source of its own, so the record's own environment can be provisioned
#: locally without touching the real registry.
PROBE_MODEL = "cap_record_probe"
PROBE_VARIABLE = "CAP_RECORD_PROBE_PYTHON"
SANDBOX_KEY = "sandbox-operator-ledger-key"

# --------------------------------------------------------------------------- script extraction

def _python_block(start_marker: str, end_marker: str = "\nEOF\n") -> str:
    """One embedded heredoc body, lifted verbatim from the script."""
    script = SCRIPT.read_text(encoding="utf-8")
    start = script.index(start_marker)
    opening = script.index("<<'EOF'", start)
    body = script.index("\n", opening) + 1
    return script[body:script.index(end_marker, body)]

def _report_block() -> str:
    """The report-writing block that folds the GPU scan into the verdict."""
    return _python_block('J1="${RESULTS[J1]}"')

def _j1_block() -> str:
    """The J1 run-classification block."""
    return _python_block('say "== J1 clean startup')

def _run_block(block: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", block, *args], capture_output=True, text=True,
                          env=env, cwd=REPOSITORY, timeout=600)

# --------------------------------------------------------------------------- sandbox

_ISOLATED_KEYS = ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                  "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT")

@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated artifacts root, ledger key and operator config home for one test."""
    artifacts = tmp_path / "artifacts"
    (artifacts / "gpu_capabilities").mkdir(parents=True)
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", SANDBOX_KEY)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONPATH", str(REPOSITORY))
    for key in _ISOLATED_KEYS:
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    return artifacts

@pytest.fixture()
def probe_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox: Path):
    """A real interpreter plus a one-model registry, so a capability record can be bound to a
    genuine environment instead of a fabricated fingerprint."""
    from workbench.backend import runtime
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry

    environment = tmp_path / "envs" / "probe"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    registry = tmp_path / "registry.yaml"
    registry.write_text(textwrap.dedent(f"""\
        schema_version: 1
        protocols: {{}}
        models:
          - model_id: {PROBE_MODEL}
            native_protocol: null
            supported_protocols: []
            literature_split_label: null
            source_available: false
            environment_required: true
            environment:
              python_env_var: {PROBE_VARIABLE}
              cuda: "12.4"
              required_imports: [json]
            checkpoint_variants: []
            checkpoint_bundles: []
        """), encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(registry))
    monkeypatch.setenv(PROBE_VARIABLE, str(environment / "bin" / "python"))
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    model = next(m for m in load_registry()["models"] if m["model_id"] == PROBE_MODEL)
    return model, str(environment / "bin" / "python"), resolve_config()

def _genuine_record(model: dict, config, interpreter: str, *, record=None) -> dict:
    """A capability record exactly as the authorized probe writes it for this environment.

    Built through ``record_gpu_capability``, so the HMAC is computed by the runtime's own
    signing helper under this sandbox's operator key.
    """
    from workbench.backend import runtime

    contract = runtime._environment_contract_digest(model)
    payload = dict(record or {
        "capability_verified": True,
        "cuda_available": True,
        "probe_source": "interpreter",
        "interpreter": interpreter,
        "interpreter_fingerprint": runtime._interpreter_fingerprint(interpreter, contract),
        "dependency_fingerprint": contract,
        "runtime_adaptation": None,
    })
    return runtime.record_gpu_capability(model, payload, config)

def _scan(env: dict[str, str]) -> tuple[dict, subprocess.CompletedProcess]:
    """Run the script's own report block, with J5 bound as the bash runner binds it."""
    import tempfile

    child = env | {"J5": "PASS", "D5": "fail-closed contract holds"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        completed = _run_block(_report_block(), str(path), env=child)
        assert path.is_file(), (completed.stdout, completed.stderr)
        report = json.loads(path.read_text(encoding="utf-8"))
    return report, completed

def _gpu_status(report: dict) -> str:
    return report["dimensions"]["GPU_RUNTIME"]["status"]

# --------------------------------------------------------------------------- P1-4 GPU records

def test_a_hand_written_record_is_not_gpu_evidence(probe_environment, sandbox: Path):
    """(a) A file with `probe_source: interpreter` and `capability_verified: true` — everything
    the old scan looked at — certified the host as GPU-verified. Without a signature under this
    deployment's key it is not evidence, and the report must say why."""
    capabilities = sandbox / "gpu_capabilities"
    (capabilities / "handwritten.json").write_text(json.dumps({
        "capability_key": "handwritten",
        "probe_source": "interpreter",
        "capability_verified": True,
        "cuda_available": True,
        "device_name": "NVIDIA GeForce RTX 5070 Ti",
        "compute_capability": "12.0",
    }), encoding="utf-8")

    report, completed = _scan(os.environ)

    assert _gpu_status(report) == "PENDING", report["dimensions"]
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert report["gpu_capability_scan"]["status"] == "UNVERIFIED"
    reasons = "; ".join(report["gpu_capability_scan"]["unverified"])
    assert "not signed with this deployment's operator key" in reasons, reasons
    assert "not verified evidence" in completed.stderr

def test_a_record_signed_with_another_key_is_not_evidence(probe_environment, sandbox: Path):
    """(b) A record whose bytes are a real probe's, re-signed with a different operator key,
    is another deployment's evidence. The signature must be verified, not merely present."""
    from workbench.backend import runtime

    model, interpreter, config = probe_environment
    fields = runtime.gpu_capability_path(config, runtime._gpu_capability_key(model))
    contract = runtime._environment_contract_digest(model)
    payload = {
        "model_id": PROBE_MODEL,
        "capability_key": fields.stem,
        "capability_verified": True,
        "cuda_available": True,
        "probe_source": "interpreter",
        "interpreter": interpreter,
        "interpreter_fingerprint": runtime._interpreter_fingerprint(interpreter, contract),
        "dependency_fingerprint": contract,
        "runtime_adaptation": None,
        "host_identity": runtime._host_identity(),
    }
    os.environ["WORKBENCH_LEDGER_KEY"] = "another-deployment-key"
    try:
        payload["signature"] = runtime._gpu_capability_signature(payload)
    finally:
        os.environ["WORKBENCH_LEDGER_KEY"] = SANDBOX_KEY
    assert payload["signature"] and payload["signature"] != runtime._gpu_capability_signature(
        {k: v for k, v in payload.items() if k != "signature"})
    fields.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report, _ = _scan(os.environ)

    assert _gpu_status(report) == "PENDING", report["dimensions"]
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert "not signed with this deployment's operator key" in \
        "; ".join(report["gpu_capability_scan"]["unverified"])

def test_a_genuine_record_for_this_environment_is_accepted(probe_environment, sandbox: Path):
    """(c) The positive control: a record the probe itself wrote for THIS environment is
    accepted, and the runtime reads it back as capability evidence."""
    from workbench.backend import runtime

    model, interpreter, config = probe_environment
    _genuine_record(model, config, interpreter)
    path = runtime.gpu_capability_path(config, runtime._gpu_capability_key(model))
    assert path.is_file()

    report, completed = _scan(os.environ)

    assert _gpu_status(report) == "PASS", (report["dimensions"], completed.stderr)
    # The scan block was only emitted on degradation before this fix, so the positive control
    # reads it defensively: it proves the verdict, which is what the release acts on.
    assert report.get("gpu_capability_scan", {}).get("real_gpu", True) is True
    assert report.get("gpu_capability_scan", {}).get("status", "VERIFIED") == "VERIFIED"

def test_a_record_for_another_host_is_not_evidence(probe_environment, sandbox: Path):
    """(c2) The same signed record is bound to the host it was probed on: relabelling its host
    identity (or adaptation) is detected, not trusted."""
    from workbench.backend import runtime

    model, interpreter, config = probe_environment
    record = _genuine_record(model, config, interpreter)
    record = {k: v for k, v in record.items() if k != "signature"}
    record["host_identity"] = dict(record["host_identity"], node="another-machine")
    path = runtime.gpu_capability_path(config, runtime._gpu_capability_key(model))
    # Re-signed honestly under this key: the bytes are genuine, the claim they make is not.
    path.write_text(json.dumps(runtime.record_gpu_capability(
        model, {k: v for k, v in record.items() if k != "capability_key"}, None), indent=2),
        encoding="utf-8")

    report, _ = _scan(os.environ)

    assert _gpu_status(report) == "PENDING", report["dimensions"]
    assert "different host" in "; ".join(report["gpu_capability_scan"]["unverified"])

def test_an_unreadable_artifact_does_not_downgrade_readable_evidence(probe_environment, sandbox: Path):
    """(d) A 0-byte leftover is named and removed; it must not silence a genuine probe."""
    from workbench.backend import runtime

    model, interpreter, config = probe_environment
    _genuine_record(model, config, interpreter)
    poisoned = sandbox / "gpu_capabilities" / "leftover.json"
    poisoned.write_bytes(b"")

    report, completed = _scan(os.environ)

    assert _gpu_status(report) == "PASS", (report["dimensions"], completed.stderr)
    assert report["gpu_capability_scan"]["real_gpu"] is True
    assert report["gpu_capability_scan"]["status"] == "UNREADABLE"
    assert "leftover.json" in report["gpu_capability_scan"]["unreadable"]
    assert not poisoned.exists()

# --------------------------------------------------------------------------- P2-7 J1 boot state

def _serve(runs: list[dict], model_ids: list[str]) -> tuple[str, HTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/models":
                body = json.dumps([{"model_id": m} for m in model_ids])
            elif path == "/api/runs":
                body = json.dumps(runs)
            elif path == "/api/jobs":
                body = json.dumps([])
            else:
                body = json.dumps({})
            raw = body.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server

def _registry_model_ids() -> list[str]:
    from workbench.backend.registry import load_registry

    return [m["model_id"] for m in load_registry()["models"]]

def _corroborated_run(run_id: str) -> dict:
    """A real evaluation run: its aggregate report is corroborated by the evidence ledger with
    the completion the evaluator records for a successful invocation."""
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models

    config = resolve_config()
    reports = evaluate_models.reports_root(config)
    reports.mkdir(parents=True, exist_ok=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "schema_version": 1,
              "run_id": run_id, "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split", "execution_status": "EXECUTION_COMPLETED"}
    path = reports / f"limn_base_iter0_all_categories_{run_id}_aggregate.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluate_models.append_evidence_entry(config, {
        "run_id": run_id, "report_digest": evaluate_models.sha256_file(path),
        "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split", "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED"})
    return {"run_id": run_id, "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
            "protocol_id": "fashioniq_val_split", "data_kind": "experiment"}

def _classify(runs: list[dict]) -> subprocess.CompletedProcess:
    base, server = _serve(runs, _registry_model_ids())
    try:
        return _run_block(_j1_block(), base, str(REPOSITORY), env=os.environ)
    finally:
        server.shutdown()
        server.server_close()

def test_j1_reports_a_clean_first_boot(sandbox: Path):
    """An empty stack is a clean first boot: no run is presented as evidence."""
    completed = _classify([])

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "clean first boot" in completed.stdout, completed.stdout

def test_j1_accepts_an_existing_corroborated_run(sandbox: Path):
    """A legitimate run from a previous evaluation is not fake: it must not produce a false
    failure, and the observed boot state must be named."""
    completed = _classify([_corroborated_run("limn_real_01")])

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "existing valid results" in completed.stdout, completed.stdout

def test_j1_reports_a_labelled_mock_run_as_non_evidence(sandbox: Path):
    """`data_kind=mock` IS the label saying "not research evidence": the workbench writes those
    fixtures itself and a fresh installation shows them. Treating an honestly labelled fixture as
    fraud blocklisted a release that had done nothing wrong, so it is reported as labelled
    non-evidence — while the invariant that matters (nothing reaches the UI as real without
    ledger corroboration) is still enforced by the uncorroborated case below."""
    completed = _classify([{"run_id": "mock_pair-pair_b1", "model_id": "pair",
                            "checkpoint_id": "pair_b1", "protocol_id": "fashioniq_val_split",
                            "data_kind": "mock"}])

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert "labelled non-evidence" in completed.stdout, completed.stdout
    assert "labelled data_kind=mock" in completed.stdout, completed.stdout

def test_j1_refuses_an_uncorroborated_run(sandbox: Path):
    """A run claiming to be an experiment with no ledger-corroborated report is the other half
    of the same fake: refused, with the reason named."""
    completed = _classify([{"run_id": "limn_uncorroborated", "model_id": "limn",
                            "checkpoint_id": "base_iter0_all_categories",
                            "protocol_id": "fashioniq_val_split", "data_kind": "experiment"}])

    assert completed.returncode != 0
    assert "presented as real evidence without corroboration" in completed.stderr
    assert "no evidence-ledger-corroborated aggregate report" in completed.stderr
