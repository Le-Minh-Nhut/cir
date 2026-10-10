"""Release-blocker regression tests: source provenance and runtime artifacts.

SRC-01..SRC-10.

The central property: an artifact the workbench itself legitimately prepares
(the CSMCIR canonical dataset link, recorded auxiliary assets) must not
invalidate source pinning, while a real source change must never be hidden.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.registry import model_by_id, sha256_file
from workbench.backend.runtime import _source_dirty, source_clean_and_pinned, source_provenance


PIN = "0" * 40


@pytest.fixture
def csmcir_checkout(tmp_path: Path, monkeypatch):
    """A tiny pinned CSMCIR checkout plus the canonical FashionIQ root."""
    repo_root = tmp_path / "repo"
    third_party = repo_root / "third_party"
    source = third_party / "CSMCIR"
    canonical = repo_root / "data" / "FashionIQ"
    source.mkdir(parents=True)
    canonical.mkdir(parents=True)

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=source, check=True,
                              capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@t.invalid")
    git("config", "user.name", "T")
    (source / "src").mkdir()
    (source / "src" / "validate_blip_csmcir.py").write_text("# official evaluator\n")
    (source / "src" / "data_utils.py").write_text("# official helper\n")
    git("add", ".")
    git("commit", "-qm", "pin")
    head = git("rev-parse", "HEAD")

    monkeypatch.setenv("CIR_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("FASHIONIQ_ROOT", str(canonical))
    monkeypatch.setenv("WORKBENCH_THIRD_PARTY_ROOT", str(third_party))

    return {"repo_root": repo_root, "source": source, "canonical": canonical, "head": head}


def csmcir_model(head: str) -> dict:
    model = dict(model_by_id("csmcir"))
    model["upstream_commit_sha"] = head
    return model


def link(source: Path, target: Path) -> Path:
    path = source / "fashionIQ_dataset"
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target, target_is_directory=True)
    return path


# ------------------------------------------------------------------ SRC-01, SRC-05, SRC-06

def test_src01_clean_pinned_source_accepted(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    clean, head, reason = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is True, reason
    assert head == csmcir_checkout["head"]


def test_src05_unauthorized_untracked_python_rejected(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    (csmcir_checkout["source"] / "rogue.py").write_text("print('unauthorized')\n")

    assert _source_dirty(csmcir_checkout["source"]) is True
    clean, _, reason = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is False and "rogue.py" in reason


def test_src05b_gitignored_asset_is_not_a_declared_artifact(csmcir_checkout):
    source = csmcir_checkout["source"]
    (source / ".gitignore").write_text("*.bin\n__pycache__/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "ignore"], cwd=source, check=True, capture_output=True)
    csmcir_checkout["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()

    (source / "backbone.bin").write_bytes(b"locally placed weights")
    assert _source_dirty(source) is True, ".gitignore must not whitelist source assets"


def test_src06_tracked_evaluator_modification_rejected(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    source = csmcir_checkout["source"]
    evaluator = source / "src" / "validate_blip_csmcir.py"
    evaluator.write_text(evaluator.read_text() + "# uncommitted local edit\n")

    assert _source_dirty(source) is True
    clean, _, reason = source_clean_and_pinned(model, source)
    assert clean is False and "validate_blip_csmcir.py" in reason
    provenance = source_provenance(model, source)
    assert provenance["verified"] is False
    assert provenance["tracked_modifications"]


# ------------------------------------------------------------------ SRC-02 SRC-03 SRC-04

def test_src02_valid_dataset_symlink_does_not_dirty_source(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    link(csmcir_checkout["source"], csmcir_checkout["canonical"])

    assert _source_dirty(csmcir_checkout["source"]) is False, "approved runtime artifact"
    clean, _, reason = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is True, reason
    provenance = source_provenance(model, csmcir_checkout["source"])
    assert provenance["verified"] is True
    assert "fashionIQ_dataset" in provenance["approved_runtime_artifacts"]


def test_src03_wrong_dataset_symlink_rejected(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    wrong = csmcir_checkout["repo_root"] / "elsewhere"
    wrong.mkdir(parents=True)
    link(csmcir_checkout["source"], wrong)

    assert _source_dirty(csmcir_checkout["source"]) is True
    clean, _, reason = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is False and "fashionIQ_dataset" in reason
    assert source_provenance(model, csmcir_checkout["source"])["invalid_runtime_artifacts"]


def test_src04_dangling_dataset_symlink_rejected(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    link(csmcir_checkout["source"], csmcir_checkout["canonical"])
    csmcir_checkout["canonical"].rmdir()

    assert _source_dirty(csmcir_checkout["source"]) is True
    clean, _, _ = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is False


def test_src04b_deleted_symlink_is_absent_not_dirty(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    path = link(csmcir_checkout["source"], csmcir_checkout["canonical"])
    assert _source_dirty(csmcir_checkout["source"]) is False
    path.unlink()

    assert _source_dirty(csmcir_checkout["source"]) is False, "absence is not modification"
    # The link itself is still required for evaluation, which runtime_blockers reports.
    blockers = runtime.runtime_blockers(model, model["checkpoint_variants"][0],
                                        _config(csmcir_checkout), model["native_protocol"],
                                        csmcir_checkout["canonical"])
    assert any("canonical dataset link" in blocker for blocker in blockers)


def test_src04c_symlink_escape_rejected(csmcir_checkout):
    """A declared auxiliary asset may not be replaced by a symlink out of the checkout."""
    model = csmcir_model(csmcir_checkout["head"])
    source = csmcir_checkout["source"]
    outside = csmcir_checkout["repo_root"] / "outside.json"
    outside.write_text("{}")
    asset_dir = source / "COT_ours2" / "fashioniq"
    asset_dir.mkdir(parents=True)

    from workbench.backend.runtime import _approved_source_artifacts

    approved = _approved_source_artifacts(source)
    assert "COT_ours2/fashioniq/dress_cot_val.json" in approved

    for category in ("dress", "shirt", "toptee"):
        (asset_dir / f"{category}_cot_val.json").write_text("{}")
    assert _source_dirty(source) is True, "unverified declared assets are not trusted"

    (asset_dir / "dress_cot_val.json").unlink()
    (asset_dir / "dress_cot_val.json").symlink_to(outside)
    assert _source_dirty(source) is True
    assert source_provenance(model, source)["invalid_runtime_artifacts"]


# ------------------------------------------------------------------ SRC-07, SRC-08

def _record_auxiliary_digests(csmcir_checkout, paths: list[Path]) -> None:
    manifest = csmcir_checkout["repo_root"] / "workbench" / "artifacts" / "auxiliary" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([
        {"destination": str(path), "downloaded_sha256": sha256_file(path)}
        for path in paths
    ]))


def test_src07_declared_auxiliary_asset_accepted_only_with_provenance(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    source = csmcir_checkout["source"]
    asset_dir = source / "COT_ours2" / "fashioniq"
    asset_dir.mkdir(parents=True)
    assets = []
    for category in ("dress", "shirt", "toptee"):
        path = asset_dir / f"{category}_cot_val.json"
        path.write_text('{"captions": []}')
        assets.append(path)

    # Without any recorded acquisition digest the declared asset cannot be trusted.
    assert _source_dirty(source) is True

    _record_auxiliary_digests(csmcir_checkout, assets)
    assert _source_dirty(source) is False, "recorded declared assets are approved runtime artifacts"
    clean, _, reason = source_clean_and_pinned(model, source)
    assert clean is True, reason


def test_src08_modified_auxiliary_asset_rejected(csmcir_checkout):
    model = csmcir_model(csmcir_checkout["head"])
    source = csmcir_checkout["source"]
    asset_dir = source / "COT_ours2" / "fashioniq"
    asset_dir.mkdir(parents=True)
    assets = []
    for category in ("dress", "shirt", "toptee"):
        path = asset_dir / f"{category}_cot_val.json"
        path.write_text('{"captions": []}')
        assets.append(path)
    _record_auxiliary_digests(csmcir_checkout, assets)
    assert _source_dirty(source) is False

    (asset_dir / "dress_cot_val.json").write_text('{"captions": ["tampered"]}')
    assert _source_dirty(source) is True
    clean, _, reason = source_clean_and_pinned(model, source)
    assert clean is False
    assert "dress_cot_val.json" in reason


def test_src07b_expected_checksum_in_registry_is_enforced(csmcir_checkout, monkeypatch):
    source = csmcir_checkout["source"]
    asset_dir = source / "COT_ours2" / "fashioniq"
    asset_dir.mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (asset_dir / f"{category}_cot_val.json").write_text('{"captions": []}')

    import workbench.backend.registry as registry

    original = registry.auxiliary_assets_for_model

    def patched(model_id: str):
        assets = original(model_id)
        return [{**asset, "expected_sha256": "f" * 64} for asset in assets]

    monkeypatch.setattr(registry, "auxiliary_assets_for_model", patched)
    assert _source_dirty(source) is True, "a mismatching official checksum must be rejected"


# ---------------------------------------------------------------------------- SRC-09

def test_src09_source_pin_mismatch_rejected(csmcir_checkout):
    model = csmcir_model("1" * 40)
    clean, head, reason = source_clean_and_pinned(model, csmcir_checkout["source"])
    assert clean is False
    assert head == csmcir_checkout["head"]
    assert "pin mismatch" in reason


# ---------------------------------------------------------------------------- SRC-10

def test_src10_resume_does_not_reuse_dirty_source(csmcir_checkout):
    """A source change must invalidate a completed stage's fingerprint."""
    import workbench.scripts.pipeline as pipeline

    config = _config(csmcir_checkout)
    stage = pipeline.Stage(name="evaluation", commands=(["/bin/true"],), model_id="csmcir")
    before = pipeline.compute_stage_input_fingerprint(stage, config)
    assert pipeline.compute_stage_input_fingerprint(stage, config) == before

    source = csmcir_checkout["source"]
    (source / "src" / "data_utils.py").write_text("# local modification\n")
    after = pipeline.compute_stage_input_fingerprint(stage, config)
    assert after != before, "a dirty source must change the resume fingerprint"

    provenance = source_provenance(csmcir_model(csmcir_checkout["head"]), source)
    assert provenance["dirty"] is True and provenance["verified"] is False


def _config(csmcir_checkout) -> "object":
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config

    return replace(
        resolve_config(),
        CIR_REPO_ROOT=csmcir_checkout["repo_root"],
        FASHIONIQ_ROOT=csmcir_checkout["canonical"],
        WORKBENCH_THIRD_PARTY_ROOT=csmcir_checkout["repo_root"] / "third_party",
    )


# --------------------------------------------------------------- sync_upstreams agreement

def test_sync_upstreams_uses_the_same_classification(tmp_path: Path):
    """sync_upstreams, doctor, and the execution gate must not disagree on dirtiness."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sync_upstreams", Path("workbench/scripts/sync_upstreams.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo = tmp_path / "UPSTREAM"
    repo.mkdir()
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"]):
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    (repo / "train.py").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=repo, check=True, capture_output=True)

    # Bytecode alone must not report the checkout dirty.
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "train.cpython-313.pyc").write_bytes(b"cache")
    state, _, dirty = module.local_state(repo)
    assert state == "clean" and dirty is False, "bytecode must not dirty an upstream checkout"

    # A real source edit must.
    (repo / "train.py").write_text("modified")
    state, _, dirty = module.local_state(repo)
    assert state == "dirty" and dirty is True
