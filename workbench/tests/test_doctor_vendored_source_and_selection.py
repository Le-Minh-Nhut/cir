from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry
from workbench.scripts import doctor

#: The repository this test runs from; the vendored checkouts live inside it, which is the whole
#: point of the last test -- the defect needs an ENCLOSING repository to walk up to.
REPO = Path(__file__).resolve().parents[2]


def config(root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(root, root / "data", root / "FashionIQ", "127.0.0.1", 8000, 5173,
                           root / "checkpoints", root / "results", root / "third_party")


def test_vendored_source_state_uses_provenance_and_local_changes_only(tmp_path: Path) -> None:
    model = next(model for model in load_registry()["models"] if model["model_id"] == "limn")
    source = Path("workbench/third_party") / model["source_dir"]
    checkout = tmp_path / "third_party" / model["source_dir"]
    shutil.copytree(source, checkout, ignore=shutil.ignore_patterns(".git"))
    assert (checkout / "provenance.json").is_file()
    assert not any(checkout.rglob(".git"))

    state = doctor.source_state(model, tmp_path / "third_party")
    record = json.loads((checkout / "provenance.json").read_text())
    assert state["state"] == "clean"
    assert state["head"] == record["upstream_commit"] == model["upstream_commit_sha"]

    (tmp_path / "unrelated.py").write_text("outside vendored checkout")
    state = doctor.source_state(model, tmp_path / "third_party")
    assert state["state"] == "clean"
    assert not state["tracked_modifications"]

    target = next(path for path in checkout.rglob("*.py") if path.is_file())
    target.write_text(target.read_text() + "\n# unauthorized edit\n")
    state = doctor.source_state(model, tmp_path / "third_party")
    assert state["state"] == "dirty"
    assert target.relative_to(checkout).as_posix() in state["tracked_modifications"]


def test_doctor_cli_selection_checks_only_requested_variant(monkeypatch, tmp_path: Path, capsys) -> None:
    body = b"verified weights"
    digest = hashlib.sha256(body).hexdigest()
    variant = {"checkpoint_id": "noise_20", "filename": "noise_20.pth", "expected_sha256": digest,
               "status": "NO_CLEAN_CHECKPOINT", "checkpoint_mapping_status": "VERIFIED_METADATA",
               "checkpoint_training_noise_pct": 20}
    model = {"model_id": "habit", "source_dir": None, "checkpoint_variants": [
        variant, {**variant, "checkpoint_id": "noise_50", "filename": "noise_50.pth"},
        {**variant, "checkpoint_id": "noise_80", "filename": "noise_80.pth"}],
        "checkpoint_bundles": [], "required_runtime_assets": [], "environment": None,
        "command_status": "COMMAND_AUDITED"}
    settings = config(tmp_path)
    checkpoint = settings.WORKBENCH_CHECKPOINT_ROOT / "habit" / variant["filename"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(body)
    monkeypatch.setattr(doctor, "load_registry", lambda: {"models": [model], "protocols": {}})
    monkeypatch.setattr(doctor, "resolve_config", lambda: settings)
    monkeypatch.setattr(doctor, "workbench_checks", lambda *_: [])
    monkeypatch.setattr(doctor, "fashioniq_check", lambda *_: {"name": "fashioniq", "status": "OK"})
    monkeypatch.setattr(doctor, "source_state", lambda *_: {"path": None, "state": "clean", "head": "pinned",
                                                               "dirty": False, "pin": "pinned"})
    monkeypatch.setattr(doctor, "preflight_blockers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(doctor, "environment_status", lambda *_args, **_kwargs: {
        "status": "READY", "interpreter": "python", "tier": "cpu", "variable": None,
        "verified": True, "gpu_deferred": False, "reason": None})
    monkeypatch.setattr(doctor, "preparation_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "external_asset_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "required_runtime_assets", lambda *_: [])

    assert doctor.main(["--scope", "real", "--model", "habit", "--checkpoint", "noise_20",
                        "--json", "--dataset-root", str(settings.FASHIONIQ_ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)
    runtime = next(item for item in report["checks"] if item["name"] == "runtime:habit")
    assert runtime["status"] == "OK"
    assert [item["checkpoint_id"] for item in runtime["evidence"]["checkpoints"]] == ["noise_20"]


def test_doctor_checkpoint_facts_and_blocked_status(monkeypatch, tmp_path: Path, capsys) -> None:
    body = b"verified weights"
    digest = hashlib.sha256(body).hexdigest()
    variant = {"checkpoint_id": "noise_20", "filename": "noise_20.pth", "expected_sha256": digest,
               "status": "NO_CLEAN_CHECKPOINT", "checkpoint_mapping_status": "VERIFIED_METADATA",
               "checkpoint_training_noise_pct": 20}
    model = {"model_id": "habit", "source_dir": None, "checkpoint_variants": [variant],
             "checkpoint_bundles": [], "required_runtime_assets": [], "environment": None,
             "command_status": "COMMAND_AUDITED"}
    settings = config(tmp_path)
    checkpoint = settings.WORKBENCH_CHECKPOINT_ROOT / "habit" / variant["filename"]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(body)
    monkeypatch.setattr(doctor, "load_registry", lambda: {"models": [model], "protocols": {}})
    monkeypatch.setattr(doctor, "resolve_config", lambda: settings)
    monkeypatch.setattr(doctor, "workbench_checks", lambda *_: [])
    monkeypatch.setattr(doctor, "fashioniq_check", lambda *_: {"name": "fashioniq", "status": "OK"})
    monkeypatch.setattr(doctor, "source_state", lambda *_: {"path": None, "state": "clean", "head": "pinned",
                                                               "dirty": False, "pin": "pinned"})
    monkeypatch.setattr(doctor, "preflight_blockers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(doctor, "environment_status", lambda *_args, **_kwargs: {
        "status": "READY", "interpreter": "python", "tier": "cpu", "variable": None,
        "verified": True, "gpu_deferred": False, "reason": None})
    monkeypatch.setattr(doctor, "preparation_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "external_asset_checks", lambda *_: ([], []))
    monkeypatch.setattr(doctor, "required_runtime_assets", lambda *_: [])

    def run() -> dict:
        assert doctor.main(["--scope", "real", "--model", "habit", "--checkpoint", "noise_20",
                            "--json", "--dataset-root", str(settings.FASHIONIQ_ROOT)]) in (0, 2)
        report = json.loads(capsys.readouterr().out)
        return next(item for item in report["checks"] if item["name"] == "runtime:habit")

    assert run()["status"] == "OK"
    checkpoint.unlink()
    assert run()["status"] == "BLOCKED"
    checkpoint.write_bytes(b"wrong")
    assert run()["status"] == "BLOCKED"
    checkpoint.write_bytes(body)
    variant["checkpoint_mapping_status"] = "UNVERIFIED"
    assert run()["status"] == "BLOCKED"
    variant["checkpoint_mapping_status"] = "VERIFIED_METADATA"
    variant["status"] = "BLOCKED_SOURCE"
    assert run()["status"] == "BLOCKED"
    variant["status"] = "BLOCKED"
    assert run()["status"] == "BLOCKED"


def test_doctor_never_reports_the_containing_repository_commit_as_a_model_revision() -> None:
    """The exact reported symptom, on the real tree rather than a fixture.

    Third-party checkouts are vendored with no nested `.git`, so a `git rev-parse HEAD` run inside
    one walks upward and returns the CONTAINING CIR repository's commit. Doctor compared that with
    the model's upstream pin and rejected valid source (LIMN at `7d7bc9b` was reported at the CIR
    commit), and the dirtiness of the parent repository was attributed to the model. A fixture built
    from independent nested Git repositories cannot reproduce it: the walk-up needs an enclosing
    repository, which is what the real layout provides.
    """
    inside = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not inside a Git work tree")
    containing_commit = inside.stdout.strip()

    checked = 0
    for model in load_registry()["models"]:
        source_dir = model.get("source_dir")
        pin = model.get("upstream_commit_sha")
        if not source_dir or not pin:
            continue
        checkout = REPO / "workbench" / "third_party" / source_dir
        if not checkout.is_dir() or (checkout / ".git").exists():
            continue
        checked += 1
        state = doctor.source_state(model, REPO / "workbench" / "third_party")
        assert state["head"] == pin, (
            f"{model['model_id']}: doctor reports {state['head']}, the pin is {pin}")
        assert state["head"] != containing_commit, (
            f"{model['model_id']}: doctor reported the containing repository's commit as this "
            f"model's revision")
        assert state["state"] == "clean"
    assert checked, "no vendored checkout was exercised"
