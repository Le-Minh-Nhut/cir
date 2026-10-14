from __future__ import annotations

import hashlib
from pathlib import Path

import sys

from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import CSMCIRAdapter
from workbench.backend.index import result_files
from workbench.backend.main import resolve_image
from workbench.backend.registry import checkpoint_availability, checkpoint_path, load_registry


def test_backend_uses_current_configured_roots(monkeypatch, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    third_party = tmp_path / "third-party"
    results = tmp_path / "results"
    fashioniq = tmp_path / "FashionIQ"
    for name, path in {
        "WORKBENCH_CHECKPOINT_ROOT": checkpoints,
        "WORKBENCH_THIRD_PARTY_ROOT": third_party,
        "WORKBENCH_RESULTS_ROOT": results,
        "FASHIONIQ_ROOT": fashioniq,
    }.items():
        monkeypatch.setenv(name, str(path))

    model = next(item for item in load_registry()["models"] if item["model_id"] == "csmcir")
    checkpoint = model["checkpoint_variants"][0]
    checkpoint_path = checkpoints / model["model_id"] / checkpoint["filename"]
    checkpoint_path.parent.mkdir(parents=True)
    # The registry records this checkpoint's published sha256 and size, so a stub of arbitrary bytes
    # is refused by the check that exists to catch a substituted file. Give the stub content that
    # matches its own declaration, which keeps the fixture honest about what it is standing in for.
    payload = b"checkpoint"
    checkpoint_path.write_bytes(payload)
    if checkpoint.get("expected_sha256") or checkpoint.get("expected_size_bytes"):
        checkpoint["expected_sha256"] = hashlib.sha256(payload).hexdigest()
        checkpoint["expected_size_bytes"] = len(payload)
    source = third_party / model["source_dir"]
    source.mkdir(parents=True)
    (results / "nested").mkdir(parents=True)
    result = results / "nested" / "run.json"
    result.write_text("{}")
    image = fashioniq / "images" / "sample.jpeg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")

    availability = checkpoint_availability(model, checkpoint)
    # The adapter re-reads the registry for its model record, so hand it the fixture's view: the
    # declaration was rewritten above to describe the stub bytes, and a second load would not see it.
    command = CSMCIRAdapter({"models": [model]}).build_command(
        EvalRequest("csmcir", checkpoint["checkpoint_id"], "fashioniq_original_split", fashioniq, tmp_path / "output.json")
    )

    assert availability["checkpoint_path"] == str(checkpoint_path)
    assert availability["checkpoint_root"] == str(checkpoints)
    assert availability["source_root"] == str(third_party)
    assert availability["source_synced_locally"] is True
    assert command[0] == sys.executable and command[0] != "python"
    assert command[1:] == [str(source / "src" / "validate_blip_csmcir.py"), "--dataset", "fashionIQ",
                           "--blip-model-path", str(checkpoint_path)]
    assert result_files() == [result]
    assert resolve_image("dress", "sample") == image


def test_backend_explicit_roots_override_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORKBENCH_RESULTS_ROOT", str(tmp_path / "configured"))
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    result = explicit / "run.json"
    result.write_text("{}")
    model = next(item for item in load_registry()["models"] if item["model_id"] == "csmcir")
    checkpoint = model["checkpoint_variants"][0]
    assert checkpoint_path(model["model_id"], checkpoint, explicit) == explicit / model["model_id"] / checkpoint["filename"]

    assert result_files(explicit) == [result]
