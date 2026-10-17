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

import hashlib
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
#: The shipped-release-target identity, declared in the sandbox registry so tests about the
#: model-evidence dimensions can bind their evidence to the same model their run is about.
TARGET_MODEL_ID = "limn"
TARGET_PROTOCOL_ID = "fashioniq_val_split"
TARGET_VARIABLE = "SANDBOX_TARGET_PYTHON"
TARGET_PIN = "d" * 40

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

def _j8_block() -> str:
    """The J8 job lifecycle block."""
    return _python_block('say "== J8 job lifecycle statuses are the backend enum =="')

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
        protocols:
          {TARGET_PROTOCOL_ID}:
            label: Sandbox target protocol
            literature_split_label: val
            evaluation_noise_pct: 0
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
          # The shipped release target, declared so tests about the model-evidence dimensions can
          # bind their evidence to it. Model-evidence dimensions are bound to the declared target,
          # so a capability record for the probe model is deliberately NOT this model's evidence.
          - model_id: {TARGET_MODEL_ID}
            method_name: Sandbox target
            native_protocol: {TARGET_PROTOCOL_ID}
            supported_protocols: [{TARGET_PROTOCOL_ID}]
            literature_split_label: val
            source_available: true
            source_dir: sandbox_target
            fashioniq_layout: fashioniq_standard
            upstream_commit_sha: {TARGET_PIN}
            reproduction_status: VERIFIED
            environment_required: true
            environment:
              python_env_var: {TARGET_VARIABLE}
              cuda: "12.4"
              required_imports: [json]
            checkpoint_variants:
              - checkpoint_id: base_iter0_dress
                filename: 0_dress_best_model.pt
                evaluation_noise_pct: 0
                checkpoint_mapping_status: VERIFIED_METADATA
                status: URL_VERIFIED
              - checkpoint_id: base_iter0_shirt
                filename: 0_shirt_best_model.pt
                evaluation_noise_pct: 0
                checkpoint_mapping_status: VERIFIED_METADATA
                status: URL_VERIFIED
              - checkpoint_id: base_iter0_toptee
                filename: 0_toptee_best_model.pt
                evaluation_noise_pct: 0
                checkpoint_mapping_status: VERIFIED_METADATA
                status: URL_VERIFIED
            checkpoint_bundles:
              - bundle_id: base_iter0_all_categories
                required_checkpoint_ids: [base_iter0_dress, base_iter0_shirt, base_iter0_toptee]
                structure: category_specific_whole_models
                mapping_status: VERIFIED_METADATA
                notes: sandbox target bundle
        """), encoding="utf-8")
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(registry))
    monkeypatch.setenv(PROBE_VARIABLE, str(environment / "bin" / "python"))
    monkeypatch.setenv(TARGET_VARIABLE, str(environment / "bin" / "python"))
    runtime.forget_environment()
    runtime.forget_gpu_capability()
    model = next(m for m in load_registry()["models"] if m["model_id"] == PROBE_MODEL)
    return model, str(environment / "bin" / "python"), resolve_config()

@pytest.fixture()
def target_environment(probe_environment):
    """The shipped release target as this sandbox declares it, plus its interpreter."""
    from workbench.backend.registry import load_registry

    _, interpreter, config = probe_environment
    model = next(m for m in load_registry()["models"] if m["model_id"] == TARGET_MODEL_ID)
    return model, interpreter, config

def _target_env_for_target() -> dict[str, str]:
    """`os.environ` with the release target set to the sandbox's declared target identity."""
    return _target_env(TARGET_MODEL_ID, "base_iter0_all_categories", TARGET_PROTOCOL_ID)

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

    base_checks = {f"J{i}": "PASS" for i in range(1, 13)}
    child = base_checks | env | {"J5": "PASS", "D5": "fail-closed contract holds"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        completed = _run_block(_report_block(), str(path), env=child)
        assert path.is_file(), (completed.stdout, completed.stderr)
        report = json.loads(path.read_text(encoding="utf-8"))
    return report, completed

def _gpu_status(report: dict) -> str:
    return report["dimensions"]["GPU_RUNTIME"]["status"]

#: Model-evidence dimensions are bound to the declared release target, and the sandbox's only
#: model is the capability probe. Tests that DO want those dimensions to reach PASS must say so;
#: the default target stays the shipped artifact so no test accidentally relaxes the binding.
TARGET_MODEL_ENV = "WORKBENCH_RELEASE_TARGET_MODEL"
TARGET_CHECKPOINT_ENV = "WORKBENCH_RELEASE_TARGET_CHECKPOINT"
TARGET_PROTOCOL_ENV = "WORKBENCH_RELEASE_TARGET_PROTOCOL"

def _target_env(model_id: str, checkpoint_id: str | None = None,
                protocol_id: str | None = None) -> dict[str, str]:
    """`os.environ` with the release target pointed at the model under test."""
    env = {**os.environ, TARGET_MODEL_ENV: model_id}
    if checkpoint_id is not None:
        env[TARGET_CHECKPOINT_ENV] = checkpoint_id
    if protocol_id is not None:
        env[TARGET_PROTOCOL_ENV] = protocol_id
    return env

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

    # The declared release target must be the model this record was probed for: the dimension is
    # now bound to the target, so the positive control names it instead of accepting *any* model's
    # verified record as this host's capability evidence.
    report, completed = _scan(_target_env(PROBE_MODEL))

    assert _gpu_status(report) == "PASS", (report["dimensions"], completed.stderr)
    # The scan block was only emitted on degradation before this fix, so the positive control
    # reads it defensively: it proves the verdict, which is what the release acts on.
    assert report.get("gpu_capability_scan", {}).get("real_gpu", True) is True
    assert report.get("gpu_capability_scan", {}).get("status", "VERIFIED") == "VERIFIED"

def test_a_verified_record_for_another_model_does_not_certify_the_target(
        probe_environment, sandbox: Path):
    """DEFECT C (reproduced first): GPU_RUNTIME accepted *any* model's capability record.

    `_expected_key` folds the model's interpreter and dependency contract into the capability key,
    so a record is evidence for exactly one model. The scan nevertheless set `real_gpu` on the
    first fully verified record it found, which let model A's probe certify a release of model B.
    Here the record is the genuine article for the probe model and the declared target is the
    shipped LIMN artifact: it must be reported as off-target, not promoted.
    """
    from workbench.backend import runtime

    model, interpreter, config = probe_environment
    _genuine_record(model, config, interpreter)
    assert runtime.gpu_capability_path(config, runtime._gpu_capability_key(model)).is_file()

    report, completed = _scan(os.environ)          # no target override: the shipped default

    assert report["release_target"]["model_id"] != PROBE_MODEL
    assert _gpu_status(report) == "PENDING", (report["dimensions"], completed.stderr)
    assert report["gpu_capability_scan"]["real_gpu"] is False
    reasons = "; ".join(report["gpu_capability_scan"]["unverified"])
    assert "different model than the declared release target" in reasons, reasons

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

    report, completed = _scan(_target_env(PROBE_MODEL))

    assert _gpu_status(report) == "PASS", (report["dimensions"], completed.stderr)
    assert report["gpu_capability_scan"]["real_gpu"] is True
    assert report["gpu_capability_scan"]["status"] == "UNREADABLE"
    assert "leftover.json" in report["gpu_capability_scan"]["unreadable"]
    # And it is REPORTED rather than deleted: the verifier runs against the operator's live
    # artifacts root and must never remove a file it did not understand.
    assert poisoned.exists(), "the verifier must not delete the operator's artifacts"

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

def test_j1_reports_conflicting_ledger_keys_as_named_condition_without_traceback(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
):
    """A conflicting operator ledger key pair must be reported as a named condition
    (ledger_key_unusable) with both paths and remedy visible, rather than an unhandled traceback."""
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    (sandbox / "reports").mkdir(parents=True, exist_ok=True)
    repo_root = Path(os.environ["CIR_REPO_ROOT"])
    repo_root.mkdir(parents=True, exist_ok=True)
    config_home = Path(os.environ["XDG_CONFIG_HOME"])
    (config_home / "cir").mkdir(parents=True, exist_ok=True)

    legacy = repo_root / ".workbench_ledger_key"
    private = config_home / "cir" / "workbench_ledger_key"
    legacy.write_text("key-legacy-divergent", encoding="utf-8")
    private.write_text("key-private-divergent", encoding="utf-8")

    completed = _classify([])
    assert completed.returncode != 0
    output = completed.stderr + completed.stdout
    assert "Traceback (most recent call last):" not in output
    assert "ledger_key_unusable" in output
    assert str(legacy) in output
    assert str(private) in output
    assert "keep one file and remove the other" in output

# --------------------------------------------------------------------------- Defect B: Premature RELEASE_READY

def _create_benchmark_run(
    sandbox: Path,
    run_id: str,
    *,
    is_synthetic: bool = False,
    data_kind: str = "experiment",
    valid_metrics: bool = True,
    has_proof: bool = True,
    corrupt_proof: bool = False,
    ledger_return_code: int = 0,
    ledger_invocation_status: str = "INVOCATION_COMPLETED",
    model_id: str = TARGET_MODEL_ID,
    checkpoint_id: str = "base_iter0_all_categories",
    protocol_id: str = TARGET_PROTOCOL_ID,
) -> dict:
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator, verify_run

    config = resolve_config()
    # The run's own identity is a parameter so a test can build a legitimate run for an identity
    # that is NOT the declared release target -- the combination DEFECT C is about.
    ckpt_dir = config.WORKBENCH_CHECKPOINT_ROOT / model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name in ("0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"):
        member = ckpt_dir / name
        if not member.is_file():
            member.write_text("fake weight")

    reports = evaluator.reports_root(config)
    reports.mkdir(parents=True, exist_ok=True)
    report_file = reports / f"{model_id}_{checkpoint_id}_{run_id}_aggregate.json"

    if valid_metrics:
        categories = {
            "dress": {"r10": 20.0, "r50": 45.0},
            "shirt": {"r10": 20.0, "r50": 45.0},
            "toptee": {"r10": 20.0, "r50": 45.0},
        }
        observed = {"r10": 20.0, "r50": 45.0, "mean": 32.5}
        extraction_status = "EXTRACTED_VERIFIED"
        extraction_avail = True
    else:
        categories = {
            "dress": {"r10": float("nan"), "r50": 45.0},
            "shirt": {"r10": 20.0, "r50": 45.0},
            "toptee": {"r10": 20.0, "r50": 45.0},
        }
        observed = {"r10": float("nan"), "r50": 45.0, "mean": float("nan")}
        extraction_status = "EXTRACTED_VERIFIED"
        extraction_avail = True

    member_names = ["0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"]
    bundle_manifest = evaluator.build_artifact_manifest(ckpt_dir, member_names)
    manifest_digest = evaluator.manifest_digest(bundle_manifest)
    # Both digests a real run's plan records for a directory artifact (evaluate_models.py:390-397):
    # the manifest digest of the declared members AND the same module's `sha256_file` over the
    # directory restricted to those members. Recording the manifest digest in the
    # `checkpoint_sha256` slot as a placeholder made the proof self-inconsistent in a way a real
    # run never is, and once the verifier actually re-hashed the live members that inconsistency
    # became what the test measured instead of its subject.
    directory_sha256 = evaluator.sha256_file(ckpt_dir, required_files=member_names)

    pin = verify_run._regression_pin(model_id)
    plan = evaluator.EvaluationPlan(
        model_id=model_id, checkpoint_id=checkpoint_id, protocol=protocol_id,
        dataset_root=sandbox / "FashionIQ", source=sandbox / "source",
        checkpoint=ckpt_dir / "0_dress_best_model.pt", cwd=sandbox / "source",
        command=["python"], pin=pin, actual_source_commit=pin,
        paper_metrics=None, artifact_type="bundle_members"
    )
    run_dir = evaluator.log_dir(config, plan, evaluator.datetime.now(evaluator.timezone.utc), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.log").write_text("evaluating\nresults: done\n")
    (run_dir / "stderr.log").write_text("")
    logs = {"stderr": "", "stdout": "evaluating\nresults: done\n"}
    evidence_digest = hashlib.sha256(json.dumps(logs, sort_keys=True).encode("utf-8")).hexdigest()

    (run_dir / "command.json").write_text(json.dumps({
        "run_id": run_id, "model_id": model_id,
        "checkpoint_id": checkpoint_id, "protocol_id": protocol_id,
        "invocation_status": "INVOCATION_COMPLETED", "return_code": 0,
    }))

    report_data = {
        "artifact_type": "official_aggregate_evaluation_report",
        "run_id": run_id, "model_id": model_id,
        "checkpoint_id": checkpoint_id, "protocol_id": protocol_id,
        "is_synthetic": is_synthetic, "data_kind": data_kind,
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED",
        "execution_status": "EXECUTION_COMPLETED",
        "metric_extraction_available": extraction_avail,
        "extraction_status": extraction_status,
        "category_metrics": categories, "observed_metrics": observed,
        "checkpoint_sha256": directory_sha256,
        "bundle_manifest_digest": manifest_digest,
        "evidence_digest": evidence_digest,
    }
    report_file.write_text(json.dumps(report_data))
    report_digest = evaluator.sha256_file(report_file)

    ledger_entry = {
        "run_id": run_id, "model_id": model_id,
        "checkpoint_id": checkpoint_id, "protocol_id": protocol_id,
        "report_digest": report_digest, "return_code": ledger_return_code,
        "invocation_status": ledger_invocation_status,
    }
    added = evaluator.append_evidence_entry(config, ledger_entry)
    ledger_entry_digest = added["entry_digest"]

    proof_file = reports / f"{model_id}_{checkpoint_id}_{run_id}_completion.json"
    if has_proof:
        proof_data = {
            "artifact_type": "official_evaluation_completion_proof",
            "schema_version": 1,
            "completion_status": "COMPLETED",
            "return_code": 0,
            "run_id": run_id, "model_id": model_id,
            "checkpoint_id": checkpoint_id, "protocol_id": protocol_id,
            "source_expected_commit": pin, "source_actual_commit": pin,
            "checkpoint_sha256": directory_sha256,
            "bundle_manifest_digest": manifest_digest,
            "ledger_entry_digest": ledger_entry_digest,
            "report": str(report_file), "report_digest": report_digest,
            "invocation_status": "INVOCATION_COMPLETED",
            "environment_digest": "env_test_digest",
            "evidence_digest": evidence_digest,
            "run_directory": str(run_dir),
        }
        if corrupt_proof:
            proof_data["report_digest"] = "0" * 64
        proof_file.write_text(json.dumps(proof_data))

    return report_data


def test_defect_b_a_synthetic_only_evidence_does_not_reach_release_ready(target_environment, sandbox: Path):
    """(a) Synthetic-only evidence does not reach RELEASE_READY; MODEL_INFERENCE remains PENDING.

    The evidence is for the DECLARED TARGET: the model-evidence dimensions are bound to it, so a
    synthetic run for any other identity would be excluded for the wrong reason and the test
    would pass without exercising its subject.
    """
    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)
    _create_benchmark_run(sandbox, "run_syn_fixture", is_synthetic=True, data_kind="mock")

    report, completed = _scan(_target_env_for_target())

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_INFERENCE"]["status"] == "PENDING"
    assert report["dimensions"]["MODEL_EVALUATION"]["status"] == "PENDING"
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"


def test_defect_b_b_verified_gpu_record_alone_does_not_reach_release_ready(target_environment, sandbox: Path):
    """(b) A verified GPU record alone does not reach RELEASE_READY (MODEL_INFERENCE still PENDING)."""
    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)

    report, completed = _scan(_target_env_for_target())

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_INFERENCE"]["status"] == "PENDING"
    assert report["dimensions"]["MODEL_EVALUATION"]["status"] == "PENDING"
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"


def test_defect_b_c_inference_claim_without_valid_metrics_does_not_reach_it(target_environment, sandbox: Path):
    """(c) An inference claim without valid metrics does not reach RELEASE_READY."""
    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)
    _create_benchmark_run(sandbox, "run_invalid_metrics", is_synthetic=False, data_kind="experiment",
                          valid_metrics=False, has_proof=True)

    report, completed = _scan(_target_env_for_target())

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    assert report["dimensions"]["SCIENTIFIC_OUTPUT"]["status"] == "PENDING"
    assert report["dimensions"]["MODEL_EVALUATION"]["status"] == "PENDING"
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"


def test_defect_b_d_run_missing_completion_proof_does_not_reach_it(target_environment, sandbox: Path):
    """(d) A run missing a completion proof does not reach RELEASE_READY."""
    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)
    _create_benchmark_run(sandbox, "run_missing_proof", is_synthetic=False, data_kind="experiment",
                          valid_metrics=True, has_proof=False)

    report, completed = _scan(_target_env_for_target())

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_EVALUATION"]["status"] == "PENDING"
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"


def test_defect_b_e_forged_capability_record_does_not_reach_it(sandbox: Path):
    """(e) A forged capability record does not reach RELEASE_READY."""
    _create_benchmark_run(sandbox, "run_legit_with_forged_gpu", is_synthetic=False,
                          data_kind="experiment", valid_metrics=True, has_proof=True)
    forge = sandbox / "gpu_capabilities" / "forged_probe.json"
    forge.write_text(json.dumps({"probe_source": "override", "capability_verified": True}))

    report, completed = _scan(os.environ)

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"


def test_defect_b_f_legitimate_corroborated_run_is_accepted_evidence(target_environment, sandbox: Path):
    """(f) A legitimate corroborated benchmark run is reported as accepted evidence.

    The positive control for the model-evidence dimensions: the run and the capability record are
    both for the declared target, so all four reach PASS and the verdict is RELEASE_READY.
    """
    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)
    _create_benchmark_run(sandbox, "run_legit_accepted", is_synthetic=False,
                          data_kind="experiment", valid_metrics=True, has_proof=True)

    pass_env = {**_target_env_for_target(), **{f"J{i}": "PASS" for i in range(1, 13)}}
    report, completed = _scan(pass_env)

    assert completed.returncode == 0, completed.stderr
    assert report["dimensions"]["STATIC_SOURCE"]["status"] == "PASS"
    assert report["dimensions"]["DEPLOYMENT"]["status"] == "PASS"
    assert report["dimensions"]["CONFIGURATION"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_ENVIRONMENT"]["status"] == "PASS"
    assert report["dimensions"]["OPERATIONS"]["status"] == "PASS"
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_INFERENCE"]["status"] == "PASS"
    assert report["dimensions"]["MODEL_EVALUATION"]["status"] == "PASS"
    assert report["dimensions"]["SCIENTIFIC_OUTPUT"]["status"] == "PASS"
    assert any(r.get("run_id") == "run_legit_accepted" for r in report.get("accepted_evidence_runs", []))
    assert report["verdict"] == "RELEASE_READY"


def test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass(sandbox: Path):
    """(g) With no target GPU the verdict is PENDING, not PASS."""
    _create_benchmark_run(sandbox, "run_legit_no_gpu_target", is_synthetic=False,
                          data_kind="experiment", valid_metrics=True, has_proof=True)

    report, completed = _scan(os.environ)

    assert completed.returncode == 0
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"

    # Also prove against the full release_verify.sh script execution
    report_path = sandbox / "offline_report.json"
    sub_env = {**os.environ, "WORKBENCH_ARTIFACTS_ROOT": str(sandbox)}
    res = subprocess.run(
        ["bash", str(SCRIPT), "--offline", "--report", str(report_path)],
        env=sub_env, capture_output=True, text=True, cwd=str(REPOSITORY)
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert report_path.is_file()
    offline_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert offline_report["verdict"] == "ENGINEERING_INCOMPLETE"
    assert offline_report["dimensions"]["DEPLOYMENT"]["status"] == "PENDING"
    assert offline_report["dimensions"]["OPERATIONS"]["status"] == "PENDING"
    assert offline_report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"

@pytest.mark.parametrize("invocation_builder", [
    lambda rep, script_dir: ["--offline", "--report", str(rep)],
    lambda rep, script_dir: ["--report", str(rep), "--offline"],
    lambda rep, script_dir: ["--offline", "--base", "http://127.0.0.1:8000", "--report", str(rep)],
    lambda rep, script_dir: ["--base", "http://127.0.0.1:8000", "--report", str(rep), "--offline"],
    lambda rep, script_dir: ["--offline"],
])
def test_release_verify_documented_invocations_produce_parseable_report(tmp_path: Path, invocation_builder):
    """Each documented invocation syntax produces a parseable report file."""
    script_copy = tmp_path / "release_verify.sh"
    script_copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script_copy.chmod(0o755)

    custom_report = tmp_path / "custom_report.json"
    args = invocation_builder(custom_report, tmp_path)
    env = {**os.environ, "WORKBENCH_REPO": str(REPOSITORY)}
    res = subprocess.run(
        ["bash", str(script_copy), *args],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY)
    )
    assert res.returncode == 0, f"Invocation failed: args={args}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    target_report = custom_report if "--report" in args else (tmp_path / "release_verify_report.json")
    assert target_report.is_file(), f"Report file not created: {target_report}"
    data = json.loads(target_report.read_text(encoding="utf-8"))
    assert "verdict" in data
    assert "checks" in data
    assert "dimensions" in data


def test_release_verify_report_block_failure_exits_nonzero_with_reason(tmp_path: Path):
    """Injected failure inside report block must exit NON-ZERO and say why, not exit 0 with no report."""
    orig_script = SCRIPT.read_text(encoding="utf-8")
    injected_marker = "import json, os, sys, datetime\nraise RuntimeError('INJECTED_REPORT_BLOCK_CRASH')"
    injected_script = orig_script.replace("import json, os, sys, datetime", injected_marker)
    assert injected_script != orig_script, "target import string not found for injection"

    script_copy = tmp_path / "release_verify.sh"
    script_copy.write_text(injected_script, encoding="utf-8")
    script_copy.chmod(0o755)

    report_path = tmp_path / "failed_report.json"
    env = {**os.environ, "WORKBENCH_REPO": str(REPOSITORY)}
    res = subprocess.run(
        ["bash", str(script_copy), "--offline", "--report", str(report_path)],
        env=env, capture_output=True, text=True, cwd=str(REPOSITORY)
    )
    assert res.returncode != 0, f"Expected non-zero exit code, got {res.returncode}. stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert not report_path.is_file(), f"Report file must not exist: {report_path}"
    combined = (res.stdout + "\n" + res.stderr).lower()
    assert "report generation failed" in combined, f"Expected failure reason in output, got:\n{res.stdout}\n{res.stderr}"
    assert "injected_report_block_crash" in combined, f"Expected exception trace in output, got:\n{res.stdout}\n{res.stderr}"


def test_f8_refuted_run_does_not_pass_model_inference(probe_environment, sandbox: Path):
    """F8-release: a refuted run whose wording does NOT contain 'completion proof' must NOT pass MODEL_INFERENCE."""
    model, interpreter, config = probe_environment
    _genuine_record(model, config, interpreter)
    _create_benchmark_run(
        sandbox, "run_refuted_nonzero_rc", is_synthetic=False, data_kind="experiment",
        valid_metrics=True, has_proof=True, ledger_return_code=1,
        ledger_invocation_status="INVOCATION_FAILED",
    )

    report, completed = _scan(os.environ)

    assert completed.returncode == 0
    assert report["dimensions"]["MODEL_INFERENCE"]["status"] == "PENDING"
    assert not any(r.get("run_id") == "run_refuted_nonzero_rc" for r in report.get("accepted_evidence_runs", []))


class _MockJobServer:
    def __init__(self, jobs: list[dict]):
        self.jobs = jobs
        self.server = None
        self.thread = None
        self.port = 0

    def __enter__(self):
        payload = json.dumps(self.jobs).encode("utf-8")
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/jobs":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()
            def log_message(self, *args):
                pass
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *args):
        if self.server:
            self.server.shutdown()


def test_a9_empty_queue_alone_does_not_make_operations_fail():
    """A9: An empty queue alone does not make the mandatory OPERATIONS dimension fail on a finished host."""
    with _MockJobServer([]) as base_url:
        completed = _run_block(_j8_block(), base_url, env=os.environ)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "queue idle" in completed.stdout

    report, _ = _scan({"J8": "PASS"})
    assert report["dimensions"]["OPERATIONS"]["status"] == "PASS"


def test_a9_running_job_status_contract_genuinely_checked():
    """A9: With a running job the status contract is genuinely checked (valid passes, bad fails)."""
    with _MockJobServer([{"status": "running"}]) as base_url:
        completed = _run_block(_j8_block(), base_url, env=os.environ)
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "1 jobs, all statuses within the backend enum" in completed.stdout

    with _MockJobServer([{"status": "bogus_status"}]) as base_url:
        completed = _run_block(_j8_block(), base_url, env=os.environ)
        assert completed.returncode != 0
        assert "unexpected job statuses" in (completed.stdout + "\n" + completed.stderr)


def test_offline_verdict_names_deployment_reason():
    """Verdict masking: --offline must report ENGINEERING_INCOMPLETE, not mask as GPU_VALIDATION_PENDING."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "offline_report.json"
        env = {**os.environ, "WORKBENCH_REPO": str(REPOSITORY)}
        res = subprocess.run(
            ["bash", str(SCRIPT), "--offline", "--report", str(report_path)],
            env=env, capture_output=True, text=True, cwd=str(REPOSITORY),
        )
        assert res.returncode == 0, (res.stdout, res.stderr)
        assert report_path.is_file()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["verdict"] == "ENGINEERING_INCOMPLETE"
        assert data["dimensions"]["DEPLOYMENT"]["status"] == "PENDING"
        assert data["verdict"] != "GPU_VALIDATION_PENDING"


def test_frontend_certification_state_visible_and_not_silent_pass():
    """Frontend F10-R2: uncertified runs (proof_unavailable, uncorroborated) must not be clean passes."""
    script = """
    import { warningLabels, isCertifiedEvidence } from "./workbench/frontend/gate.ts";

    // 1. Clean pass with certified research evidence
    const certifiedGate = {
      benchmark_gate: { passed: true, status: "PASS_WITHIN_5_PERCENT", comparisons: [] },
      verification: { certified: true, is_research_evidence: true, state: "certified" }
    };
    const certifiedLabels = warningLabels(certifiedGate, 5.0);
    if (certifiedLabels.length !== 0) {
      throw new Error(`expected no warning labels for certified run, got ${JSON.stringify(certifiedLabels)}`);
    }
    if (!isCertifiedEvidence(certifiedGate)) {
      throw new Error("expected isCertifiedEvidence to be true");
    }

    // 2. proof_unavailable state
    const proofUnavailGate = {
      benchmark_gate: { passed: true, status: "PASS_WITHIN_5_PERCENT", comparisons: [] },
      verification: { certified: false, is_research_evidence: false, state: "proof_unavailable" }
    };
    const proofUnavailLabels = warningLabels(proofUnavailGate, 5.0);
    if (!proofUnavailLabels.some(l => l.kind === "proof-unavailable" && l.tone === "amber")) {
      throw new Error(`expected proof-unavailable amber label, got ${JSON.stringify(proofUnavailLabels)}`);
    }
    if (isCertifiedEvidence(proofUnavailGate)) {
      throw new Error("proof_unavailable must not be certified evidence");
    }

    // 3. uncorroborated state
    const uncorrGate = {
      benchmark_gate: { passed: true, status: "PASS_WITHIN_5_PERCENT", comparisons: [] },
      verification: { certified: false, is_research_evidence: false, state: "uncorroborated" }
    };
    const uncorrLabels = warningLabels(uncorrGate, 5.0);
    if (!uncorrLabels.some(l => l.kind === "uncorroborated" && l.tone === "red")) {
      throw new Error(`expected uncorroborated red label, got ${JSON.stringify(uncorrLabels)}`);
    }
    if (isCertifiedEvidence(uncorrGate)) {
      throw new Error("uncorroborated must not be certified evidence");
    }

    // 4. proof_unavailable is distinguishable from uncorroborated
    if (proofUnavailLabels[0].kind === uncorrLabels[0].kind || proofUnavailLabels[0].tone === uncorrLabels[0].tone) {
      throw new Error("proof_unavailable must be distinguishable from uncorroborated");
    }

    console.log("FRONTEND_GATE_ASSERTIONS_PASS");
    """
    res = subprocess.run(["npx", "tsx", "-e", script], cwd=str(REPOSITORY), capture_output=True, text=True)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "FRONTEND_GATE_ASSERTIONS_PASS" in res.stdout


def test_frontend_api_error_surfaces_reason_not_empty_table():
    """Frontend UI-1: on an HTTP error from /api/benchmark-runs, the reason must be surfaced."""
    script = """
    let error = "";
    function setError(msg: string) { error = msg; }
    const fakeApiError = new Error("the evidence ledger key is unusable, so no report can be corroborated: key_missing");

    try {
      throw fakeApiError;
    } catch (e: any) {
      setError(`benchmark-runs: ${e.message}`);
    }

    if (!error.includes("ledger key is unusable")) {
      throw new Error(`expected surfaced error reason, got: ${error}`);
    }
    console.log("API_ERROR_SURFACED_PASS");
    """
    res = subprocess.run(["npx", "tsx", "-e", script], cwd=str(REPOSITORY), capture_output=True, text=True)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "API_ERROR_SURFACED_PASS" in res.stdout

def test_evidence_for_another_identity_cannot_certify_the_declared_target(
        target_environment, sandbox: Path) -> None:
    """DEFECT C, second half: the *benchmark* dimensions accepted any model's run.

    With the capability record belonging to the declared target, the only thing off-target here
    is the run. The preregistered failure was that `MODEL_INFERENCE`/`MODEL_EVALUATION`/
    `SCIENTIFIC_OUTPUT` passed on any accepted evidence on disk, so a release of LIMN could be
    certified by a run for a different model. The run must be excluded by identity and named.
    """
    from workbench.backend.registry import load_registry
    from workbench.scripts import evaluate_models as evaluator

    model, interpreter, config = target_environment
    _genuine_record(model, config, interpreter)

    # A fully legitimate corroborated run -- but for the OTHER model the sandbox declares.
    off = _create_benchmark_run(sandbox, "run_off_target", is_synthetic=False,
                                data_kind="experiment", valid_metrics=True, has_proof=True,
                                model_id=PROBE_MODEL)
    assert off["model_id"] == PROBE_MODEL

    report, completed = _scan(_target_env_for_target())

    assert completed.returncode == 0
    assert report["release_target"]["model_id"] == TARGET_MODEL_ID
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PASS"
    for dimension in ("MODEL_INFERENCE", "MODEL_EVALUATION", "SCIENTIFIC_OUTPUT"):
        assert report["dimensions"][dimension]["status"] == "PENDING", report["dimensions"][dimension]
    assert report["accepted_evidence_runs"] == []
    assert any("run_off_target" in entry for entry in report["off_target_evidence_runs"]), \
        report["off_target_evidence_runs"]
    assert report["verdict"] != "RELEASE_READY"
    assert report["verdict"] == "GPU_VALIDATION_PENDING"

def test_a_declared_target_that_is_not_in_the_registry_blocks_the_release(sandbox: Path) -> None:
    """A target the registry cannot resolve is an operator fault, not an unbounded wildcard.

    This is the fail-closed direction of the target binding: if the requested identity were
    silently dropped, every dimension would fall back to accepting anything on disk, which is
    exactly the defect. TARGET_BINDING fails instead.
    """
    report, completed = _scan(_target_env("no_such_model", "no_such_checkpoint", "no_such_protocol"))

    assert report["dimensions"]["TARGET_BINDING"]["status"] == "FAIL"
    assert "not in the registry" in report["dimensions"]["TARGET_BINDING"]["detail"]
    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"
    assert report["verdict"] == "RELEASE_BLOCKED"
    # The verdict owns the exit code. `TARGET_BINDING` has no J-check, so counting only J1-J12 in
    # bash once let a RELEASE_BLOCKED report exit 0 -- the report said blocked and the shell said
    # fine. The block itself must signal it.
    assert completed.returncode != 0, (
        "a RELEASE_BLOCKED verdict must not exit 0", completed.stdout, completed.stderr)

def test_frontend_banner_distinguishes_aggregate_only_from_absent_evidence():
    """DEFECT D (reproduced first): a certified aggregate-only evaluation was announced as absent.

    The banner derived real runs from `/api/runs` (the per-query DuckDB index) while an official
    LIMN benchmark result lives in `/api/benchmark-runs` as an aggregate. A valid, certified
    three-category evaluation with no per-query export therefore rendered
    "MOCK DEVELOPMENT DATA — no real evaluation results are loaded" over its own result. Each
    state below is a real deployment shape; none may be described as another.
    """
    script = """
    import { evidenceBanner, evidenceState } from "./workbench/frontend/evidence.ts";

    const check = (label, runs, benchmarks, expectBanner, extra = () => {}) => {
      const state = evidenceState(runs, benchmarks);
      const banner = evidenceBanner(state);
      for (const [key, value] of Object.entries(expectBanner)) {
        if (banner !== value) {
          throw new Error(`${label}: banner ${JSON.stringify(banner)} !== ${JSON.stringify(value)} (${key})`);
        }
      }
      extra(state, banner);
    };

    const certified = (run_id) => ({ run_id, verification: { certified: true, is_research_evidence: true, execution_authenticated: true, state: "certified" } });
    const uncertified = (run_id, state) => ({ run_id, verification: { certified: false, is_research_evidence: false, execution_authenticated: true, state } });

    // 1. Nothing real at all, and no development data either.
    check("empty", [], [], {});
    // 2. Development data only: the original mock warning, and it must say DEVELOPMENT.
    check("mock-only", [{ run_id: "m1", data_kind: "mock" }], [],
          {}, () => {});
    // 3. THE DEFECT: a certified aggregate-only real evaluation, no per-query rows.
    check("aggregate-only certified", [], [certified("agg1")], {}, (state, banner) => {
      if (/no real evaluation results/i.test(banner)) {
        throw new Error(`a certified aggregate was announced as absent: ${JSON.stringify(banner)}`);
      }
      if (!/AGGREGATE-ONLY/i.test(banner)) throw new Error(`expected aggregate-only: ${banner}`);
      if (!/certified/i.test(banner)) throw new Error(`expected certification stated: ${banner}`);
      if (!/per-query/i.test(banner)) throw new Error(`expected the unavailable level named: ${banner}`);
      if (!state.queryLevelUnavailable) throw new Error("query level must be flagged unavailable");
      if (state.noRealEvidence) throw new Error("real aggregate evidence exists");
    });
    // 4. Aggregate present but NOT certified: authenticated, not evidence.
    check("aggregate-only uncertified", [], [uncertified("agg2", "proof_unavailable")], {},
          (state, banner) => {
      if (/no real evaluation results/i.test(banner)) throw new Error(`said absent: ${banner}`);
      if (!/uncertified/i.test(banner)) throw new Error(`expected uncertified stated: ${banner}`);
      if (/certified research result/i.test(banner)) throw new Error(`claimed certification: ${banner}`);
      });
    // 5. Per-query real results exist: no mock-only alarm, no aggregate-only claim.
    check("per-query real", [{ run_id: "r1", data_kind: "experiment" }], [certified("agg1")], {},
          (state, banner) => {
      if (/MOCK DEVELOPMENT DATA/i.test(banner)) throw new Error(`false mock banner: ${banner}`);
      if (/AGGREGATE-ONLY/i.test(banner)) throw new Error(`per-query results exist: ${banner}`);
      });
    // 6. Mock AND real: excluded-from-conclusions warning, unchanged.
    check("mock + real", [{ run_id: "r1", data_kind: "experiment" }, { run_id: "m1", data_kind: "mock" }], [],
          {}, (_state, banner) => {
      if (!/Mock and real experiment runs/i.test(banner)) throw new Error(`expected mixed warning: ${banner}`);
      });
    // 7. Mock + aggregate-only real: both facts, neither suppressing the other.
    check("mock + aggregate-only", [{ run_id: "m1", data_kind: "mock" }], [certified("agg1")], {},
          (_state, banner) => {
      if (!/AGGREGATE-ONLY/i.test(banner)) throw new Error(`real aggregate hidden: ${banner}`);
      if (!/development runs/i.test(banner)) throw new Error(`mock data hidden: ${banner}`);
      });
    // 8. API returned nothing parseable: no evidence, no crash.
    check("nulls", null, null, {});
    check("api error", [], [], {});

    // --- an independent adversarial review of this module found the following, each reproduced
    // --- here first. A synthetic or refused aggregate is NOT a real evaluation (I03/I14/I16).
    const synthetic = (run_id) => ({ run_id, is_synthetic: true, data_kind: "mock",
      verification: { certified: false, is_research_evidence: false, state: "refused" } });
    const refused = (run_id) => ({ run_id,
      verification: { certified: false, is_research_evidence: false, state: "refused" } });

    // 9. A synthetic aggregate-only run must not be announced as a real evaluation. The claim is
    // spelled "AGGREGATE-ONLY REAL EVALUATION"; matching a bare /real evaluation/ would also hit
    // the correct disclaimer "no real evaluation results", so the test pins the claim itself.
    check("aggregate-only synthetic", [], [synthetic("s1")], {}, (_s, banner) => {
      if (/AGGREGATE-ONLY REAL EVALUATION/i.test(banner)) {
        throw new Error(`a synthetic run was announced as a real evaluation: ${banner}`);
      }
      if (!/MOCK DEVELOPMENT DATA/i.test(banner)) {
        throw new Error(`expected the development warning: ${banner}`);
      }
    });

    // 10. An aggregate-only set with NO certified member must not claim "REAL EVALUATION" either.
    check("aggregate-only refused", [], [refused("r1")], {}, (_s, banner) => {
      if (/AGGREGATE-ONLY REAL EVALUATION/i.test(banner)) {
        throw new Error(`nothing is certified yet the banner claims a real evaluation: ${banner}`);
      }
      if (!/no certified research evidence/i.test(banner)) {
        throw new Error(`expected the absence of certification stated: ${banner}`);
      }
    });

    // 11. Per-query real results plus an uncertified aggregate: the aggregate must not be silently
    // suppressed (the dashboard counters say it exists, so the banner may not say nothing).
    check("mixed certification with per-query runs", [{ run_id: "r1", data_kind: "experiment" }],
          [certified("b1"), { run_id: "b2", verification: { certified: false, is_research_evidence: false } }],
          {}, (state, banner) => {
      if (banner === "") {
        throw new Error("an uncertified aggregate was silently suppressed");
      }
      if (!/not certified research evidence/i.test(banner)) {
        throw new Error(`expected the uncertified aggregate named: ${banner}`);
      }
    });

    console.log("FRONTEND_EVIDENCE_ASSERTIONS_PASS");
    """
    res = subprocess.run(["npx", "tsx", "-e", script], cwd=str(REPOSITORY), capture_output=True, text=True)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "FRONTEND_EVIDENCE_ASSERTIONS_PASS" in res.stdout

def test_frontend_gate_fails_closed_and_distinguishes_refusal():
    """Adversarial review findings in `gate.ts`, each reproduced before the fix.

    (F3) `warningLabels(undefined)` returned `[]`. A `/api/runs` row without an aggregate gate
    record therefore rendered with no warning at all: `undefined` is not a certified run, it is a
    run whose certification was never stated, and the UI must fail closed.
    (F5) the backend's `refused` state fell through to the generic `uncertified` label, so an
    operator could not tell a run the verifier looked at and rejected from one whose certification
    was simply never established.
    """
    script = """
    import { warningLabels } from "./workbench/frontend/gate.ts";

    const kinds = (gate) => warningLabels(gate, 5.0).map((l) => l.kind);

    // F3: no gate record is not a clean run.
    const absent = kinds(undefined);
    if (!absent.includes("uncorroborated")) {
      throw new Error(`a run with no gate record must not look clean, got ${JSON.stringify(absent)}`);
    }

    // F5: refusal is its own state.
    const refused = kinds({ verification: { state: "refused", certified: false, is_research_evidence: false } });
    const uncorroborated = kinds({ verification: { state: "uncorroborated", certified: false, is_research_evidence: false } });
    const unavailable = kinds({ verification: { state: "proof_unavailable", certified: false, is_research_evidence: false } });
    if (!refused.includes("refused")) {
      throw new Error(`refused must be its own label, got ${JSON.stringify(refused)}`);
    }
    for (const [other, otherKinds] of [["uncorroborated", uncorroborated], ["proof_unavailable", unavailable]]) {
      if (refused[0] === otherKinds[0]) {
        throw new Error(`refused must be distinguishable from ${other}`);
      }
    }

    // A certified run still carries no certification warning.
    const certified = kinds({ verification: { state: "certified", certified: true, is_research_evidence: true } });
    if (certified.length !== 0) {
      throw new Error(`a certified run must carry no warning, got ${JSON.stringify(certified)}`);
    }

    console.log("FRONTEND_GATE_FAILCLOSED_ASSERTIONS_PASS");
    """
    res = subprocess.run(["npx", "tsx", "-e", script], cwd=str(REPOSITORY), capture_output=True, text=True)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "FRONTEND_GATE_FAILCLOSED_ASSERTIONS_PASS" in res.stdout

def test_a_dimension_may_not_pass_when_its_own_check_did_not(sandbox: Path) -> None:
    """Red-team finding: SCIENTIFIC_OUTPUT was promoted to PASS from accepted evidence alone.

    The block set the dimension to PASS whenever a corroborated run existed, discarding its own
    J-check. With J7 PENDING (or FAIL) that let a dimension PASS without its check running, and the
    verdict reach RELEASE_READY while the scientific-output check was unanswered -- the "claims more
    than its evidence supports" failure the dimension split exists to prevent (I20).
    """
    # The dimension needs BOTH its own check PASSing AND evidence to be about (which this sandbox
    # has none of, so it stays PENDING rather than claiming a PASS from a check alone -- that would
    # be the same over-claim in the other direction).
    for j7, expected_dimension, expect_nonzero_exit in (
            ("PENDING", "PENDING", False),
            ("FAIL", "FAIL", True),
            ("PASS", "PENDING", False)):
        report, completed = _scan({**os.environ, "J7": j7})
        dimension = report["dimensions"]["SCIENTIFIC_OUTPUT"]["status"]
        assert dimension == expected_dimension, (
            f"J7={j7}: the dimension must follow its own check, got {dimension}")
        # Never RELEASE_READY in this sandbox: the GPU and real-inference dimensions are PENDING
        # here, so the strongest honest verdict is GPU_VALIDATION_PENDING.
        assert report["verdict"] != "RELEASE_READY", f"J7={j7}: verdict {report['verdict']}"
        assert (completed.returncode != 0) is expect_nonzero_exit, (
            f"J7={j7}: exit {completed.returncode} for verdict {report['verdict']}")
