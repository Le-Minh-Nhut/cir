from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import BertTokenizer
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
from workbench.backend.operator_config import WorkbenchConfig, read_env_file, resolve_config
from workbench.backend.registry import (
    auxiliary_assets_for_model,
    cache_asset_blocker,
    hf_hub_cache_root,
    hub_cache_layout_problem,
    hub_cache_snapshot,
    validate_auxiliary_asset,
)
from workbench.scripts import download_auxiliary_assets as downloader
from workbench.scripts import pipeline


@pytest.fixture
def clean_tmp(tmp_path: Path):
    """Ensure tiny temp structures and clean up after test."""
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def create_tiny_bert_cache(repo_dir: Path, revision: str = "rev_valid", files: tuple[str, ...] = (
    "config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"
)) -> Path:
    snapshot_dir = repo_dir / "snapshots" / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision, encoding="utf-8")
    (snapshot_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    (snapshot_dir / "tokenizer_config.json").write_text('{"fixture": true}', encoding="utf-8")
    (snapshot_dir / "tokenizer.json").write_text('{"added_tokens": [], "fixture": true}', encoding="utf-8")
    (snapshot_dir / "config.json").write_text('{"model_type": "bert", "fixture": true}', encoding="utf-8")
    for name in files:
        f = snapshot_dir / name
        if not f.exists():
            content = '{"fixture": true}' if name.endswith(".json") else "dummy-payload"
            f.write_text(content, encoding="utf-8")
    return snapshot_dir


def set_test_hub_payload_size(monkeypatch: pytest.MonkeyPatch, snapshot: Path,
                              files: tuple[str, ...] | list[str]) -> None:
    """Override fixture's declared bert total only; basename CONFIG_JSON can collide across hubs.

    Keep override monkeypatched per test, and do not check two distinct config.json repos in one
    test: expected_asset_size derives the hook key from destination basename, not repo id.
    """
    payload_size = sum((snapshot / name).stat().st_size for name in files)
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", str(payload_size))


def test_r23_02_hf_home_resolution_and_offline_library_agreement(clean_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """R23-02: HF_HOME in workbench.env, relative path cwd-independence, and offline library agreement."""
    # 1. Test relative path resolution against repo root independent of process cwd
    from workbench.backend.operator_config import _DEFAULT_REPO_ROOT
    monkeypatch.chdir(clean_tmp)  # Change process cwd away from repo root
    monkeypatch.setenv("HF_HOME", "test_cache/relative_hf")
    resolved_root = hf_hub_cache_root()
    assert resolved_root == (_DEFAULT_REPO_ROOT / "test_cache/relative_hf").resolve()

    # 2. Test configured root through workbench.env
    env_file = clean_tmp / "workbench.env"
    custom_hf = clean_tmp / "custom_hf"
    env_file.write_text(f"HF_HOME={custom_hf}\n", encoding="utf-8")
    parsed_env = read_env_file(env_file)
    assert parsed_env.get("HF_HOME") == str(custom_hf)

    # Populate tiny cache in custom_hf / "hub" / "models--bert-base-uncased"
    repo_dir = custom_hf / "hub" / "models--bert-base-uncased"
    tme_asset = next(a for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    rev = tme_asset.get("source_revision") or "86b5e0934494bd15c9632b12f734a8a67f723594"
    snapshot = create_tiny_bert_cache(repo_dir, revision=rev)
    set_test_hub_payload_size(monkeypatch, snapshot, tme_asset["hub_files"])

    # Direct real BertTokenizer loads offline with explicit cache_dir
    tok = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=str(custom_hf / "hub"), local_files_only=True)
    assert tok.tokenize("hello world") == ["hello", "world"]

    # Exercise the auxiliary-registry shape too: the old cache_path_suffix guard returned None
    # here without inspecting this otherwise-valid cache.
    monkeypatch.setenv("HF_HOME", str(custom_hf))
    status = {"path": str(repo_dir)}
    wrong = repo_dir / "refs" / "main"
    wrong_snapshot = repo_dir / "snapshots" / "unpinned_revision"
    wrong_snapshot.mkdir()
    for name in tme_asset["hub_files"]:
        (wrong_snapshot / name).write_text("fixture", encoding="utf-8")
    wrong.write_text("unpinned_revision", encoding="utf-8")
    blocker = cache_asset_blocker(tme_asset, status)
    assert blocker is not None and "expected pinned revision" in blocker
    create_tiny_bert_cache(repo_dir, revision=rev)
    wrong.write_text(rev, encoding="utf-8")
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: None)
    assert cache_asset_blocker(tme_asset, status) is None


@pytest.mark.parametrize("cache_state", ["absent", "unreferenced", "unpinned"])
def test_auxiliary_hub_cache_shape_is_never_skipped(clean_tmp: Path, monkeypatch: pytest.MonkeyPatch,
                                                     cache_state: str):
    asset = next(a for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    repo_dir = clean_tmp / "hub" / "models--bert-base-uncased"
    monkeypatch.setenv("HF_HOME", str(clean_tmp))
    if cache_state != "absent":
        snapshot = create_tiny_bert_cache(repo_dir, revision="unpinned_rev")
        if cache_state == "unreferenced":
            (repo_dir / "refs" / "main").unlink()
        elif cache_state == "unpinned":
            (repo_dir / "refs" / "main").write_text("unpinned_rev", encoding="utf-8")
    blocker = cache_asset_blocker(asset, {"path": str(repo_dir)})
    assert blocker is not None, (cache_state, blocker)


def test_digestless_file_admission_keeps_cache_shapes_out(tmp_path: Path):
    from workbench.backend.registry import asset_admission_status

    path = tmp_path / "asset.bin"
    path.write_bytes(b"plausible payload")
    status = {"status": "EXISTS_UNVERIFIED_CHECKSUM", "path": str(path)}
    assert asset_admission_status({"location_type": "directory", "cache_path_suffix": "hub/models--x"}, status) == status["status"]
    assert asset_admission_status({"hub_repo_id": "owner/repo"}, status) == status["status"]
    directory = tmp_path / "cache"
    directory.mkdir()
    directory_status = {"status": "EXISTS_UNVERIFIED_CHECKSUM", "path": str(directory)}
    assert asset_admission_status({"location_type": "directory"}, directory_status) == directory_status["status"]


def test_r23_03_validator_ref_resolution_and_four_cases(clean_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """All four hub-cache consumers require the same active pinned payload."""
    from workbench.backend.registry import cache_asset_blocker, model_by_id, preparation_contract_for_model

    repo_dir = clean_tmp / "hub" / "models--bert-base-uncased"
    tme_asset = next(a for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    pinned_rev = tme_asset["source_revision"]
    contract_asset = next(a for a in preparation_contract_for_model(model_by_id("tme"))["external_assets"]
                          if a["name"] == "bert-base-uncased tokenizer")
    asset = next(a for a in downloader.assets({"tme"}, clean_tmp / "third_party", clean_tmp / "FashionIQ")
                 if a.hub_repo_id)
    monkeypatch.setenv("HF_HOME", str(clean_tmp))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: None)

    def decisions():
        status = {"path": str(repo_dir)}
        layout_problem = hub_cache_layout_problem("bert-base-uncased", root=clean_tmp)
        blocker = cache_asset_blocker(contract_asset, status)
        auxiliary_blocker = cache_asset_blocker(tme_asset, {"path": str(repo_dir)})
        validation = validate_auxiliary_asset(tme_asset, repo_dir)
        verification = downloader.verify(asset)
        return layout_problem, blocker, auxiliary_blocker, validation.ok, verification

    def assert_rejected():
        layout_problem, blocker, auxiliary_blocker, validation_ok, verified = decisions()
        assert layout_problem is not None
        assert blocker is not None
        assert auxiliary_blocker is not None
        assert not validation_ok
        assert not verified


    # No refs/main: an unrelated ref and a complete snapshot cannot resolve the repo-id lookup.
    create_tiny_bert_cache(repo_dir, revision="rev_other")
    (repo_dir / "refs" / "main").unlink()
    (repo_dir / "refs" / "other").write_text("rev_other", encoding="utf-8")
    assert_rejected()

    # Stale unrelated refs do not defeat the selected, complete pinned revision.
    snap = create_tiny_bert_cache(repo_dir, revision=pinned_rev)
    set_test_hub_payload_size(monkeypatch, snap, tme_asset["hub_files"])
    (repo_dir / "refs" / "main").write_text(pinned_rev, encoding="utf-8")
    (repo_dir / "refs" / "other").write_text("rev_other", encoding="utf-8")
    layout_problem, blocker, auxiliary_blocker, validation_ok, verified = decisions()
    assert layout_problem is None and blocker is None and auxiliary_blocker is None and validation_ok and verified

    # A main ref to a nonexistent snapshot is refused by every consumer.
    (repo_dir / "refs" / "main").write_text("missing_revision", encoding="utf-8")
    assert_rejected()

    # A real but unpinned snapshot is also refused by every consumer.
    (repo_dir / "refs" / "main").write_text("rev_other", encoding="utf-8")
    assert_rejected()

    # A complete cache at the pinned revision is admitted end-to-end.
    (repo_dir / "refs" / "main").write_text(pinned_rev, encoding="utf-8")
    layout_problem, blocker, auxiliary_blocker, validation_ok, verified = decisions()
    assert layout_problem is None and blocker is None and auxiliary_blocker is None and validation_ok and verified


def test_r23_04_payload_budget_vs_cache_footprint(clean_tmp: Path, monkeypatch: pytest.MonkeyPatch):
    """R23-04: 35-byte payload + 40-byte refs/main does not exceed 35-byte budget."""
    repo_dir = clean_tmp / "hub" / "models--bert-base-uncased"
    pinned_rev = next(a["source_revision"] for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    snapshot_dir = repo_dir / "snapshots" / pinned_rev
    snapshot_dir.mkdir(parents=True)
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text(pinned_rev, encoding="utf-8")  # 40 bytes

    # 5 files of 7 bytes each = 35 bytes
    payload_files = ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
    for name in payload_files:
        (snapshot_dir / name).write_bytes(b"1234567")
    (snapshot_dir / "config-alias.json").symlink_to(snapshot_dir / "config.json")
    dedup_files = (*payload_files, "config-alias.json")
    declared_files = payload_files


    # Add an unrelated old snapshot of 100 bytes to verify old revisions do not count against budget
    old_snapshot = repo_dir / "snapshots" / "old_rev"
    old_snapshot.mkdir(parents=True)
    (old_snapshot / "large_file.bin").write_bytes(b"x" * 100)

    # Verify helper computes only declared payload at the snapshot (35 bytes)
    payload = downloader.hub_cache_payload_bytes(snapshot_dir, dedup_files)
    assert payload == 35, f"Expected 35 bytes payload, got {payload}"

    # Total cache footprint includes refs and old snapshots (40 + 35 + 100 = 175)
    total_footprint = sum(f.stat().st_size for f in repo_dir.rglob("*") if f.is_file() and not f.is_symlink())
    assert total_footprint > 75

    # Mock snapshot_download to return our snapshot
    monkeypatch.setenv("HF_HOME", str(clean_tmp))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: str(snapshot_dir))

    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", "35")
    asset = downloader.Asset(
        model_id="tme",
        family="bert-base-uncased Qformer payload",
        filename="config.json",
        destination=repo_dir,
        required=True,
        automatic_download_available=True,
        source_url="https://huggingface.co/bert-base-uncased",
        source_revision=pinned_rev,
        expected_sha256=None,
        provenance_status="VERIFIED",
        expected_size_bytes=35,
        hub_repo_id="bert-base-uncased",
        hub_files=declared_files,
    )

    from dataclasses import replace
    oversized = replace(asset, expected_size_bytes=34)
    ok, msg = downloader.bootstrap_hub_asset(oversized)
    assert not ok and ("above the declared bound of 34 bytes" in msg or "registry declares 34 bytes" in msg)
    ok, msg = downloader.bootstrap_hub_asset(asset)
    assert ok, f"bootstrap_hub_asset should succeed with 35-byte payload bound, failed with: {msg}"


@pytest.mark.parametrize("declared_bytes,actual_bytes", [(34, 35), (35, 34)])
def test_cache_declared_total_mismatch_blocks_readiness_and_acquisition(
        clean_tmp: Path, monkeypatch: pytest.MonkeyPatch, declared_bytes: int, actual_bytes: int):
    from workbench.backend.registry import hub_cache_size_problem

    monkeypatch.setenv("HF_HOME", str(clean_tmp))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", str(declared_bytes))
    auxiliary = next(a for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    revision = auxiliary["source_revision"]
    repo = clean_tmp / "hub" / "models--bert-base-uncased"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    names = auxiliary["hub_files"]
    sizes = [7, 7, 7, 7, actual_bytes - 28]
    for name, size in zip(names, sizes):
        (snapshot / name).write_bytes((b"payload")[:size])

    problem = hub_cache_size_problem(auxiliary, repo)
    assert problem is not None and problem.startswith("PAYLOAD_SIZE_MISMATCH")
    assert f"declared payload has {actual_bytes} bytes" in problem
    assert f"registry declares {declared_bytes} bytes" in problem
    from workbench.backend.registry import external_asset_statuses, model_by_id, preparation_contract_for_model
    model = model_by_id("tme")
    contract_asset = next(a for a in preparation_contract_for_model(model)["external_assets"]
                          if a["name"] == "bert-base-uncased tokenizer")
    status = next(s for s in external_asset_statuses(model) if s["name"] == contract_asset["name"])
    blocker = cache_asset_blocker(contract_asset, status)
    assert blocker is not None and blocker.startswith("PAYLOAD_SIZE_MISMATCH")

    asset = next(item for item in downloader.assets({"tme"}, clean_tmp / "third_party", clean_tmp / "FashionIQ")
                 if item.hub_repo_id)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **_kwargs: str(snapshot))
    ok, message = downloader.bootstrap_hub_asset(asset)
    assert not ok
    assert message.startswith("PAYLOAD_SIZE_MISMATCH")
    assert f"declared payload has {actual_bytes} bytes" in message
    cfg = WorkbenchConfig(
        clean_tmp,
        clean_tmp / "data",
        clean_tmp / "FashionIQ",
        "127.0.0.1",
        8000,
        5173,
        clean_tmp / "checkpoints",
        clean_tmp / "results",
        clean_tmp / "third_party",
    )
    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(downloader, "resolve_config", lambda: cfg)

    # 1. Setup complete BERT cache
    hf_home = clean_tmp / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    repo_dir = hf_home / "hub" / "models--bert-base-uncased"
    tme_asset = next(a for a in auxiliary_assets_for_model("tme") if a.get("hub_repo_id"))
    rev = tme_asset.get("source_revision") or "86b5e0934494bd15c9632b12f734a8a67f723594"
    snapshot = create_tiny_bert_cache(repo_dir, revision=rev)
    set_test_hub_payload_size(monkeypatch, snapshot, tme_asset["hub_files"])

    # 2. Setup zero-byte TME weight
    w1 = cfg.WORKBENCH_THIRD_PARTY_ROOT / "TME" / "weight" / "blip2_pretrained.pth"
    w2 = cfg.WORKBENCH_THIRD_PARTY_ROOT / "TME" / "weight" / "eva_vit_g.pth"
    w1.parent.mkdir(parents=True)
    w1.touch()  # 0 bytes!
    w2.touch()  # 0 bytes!

    # Fast-path environment checks
    from workbench.backend import runtime
    monkeypatch.setattr(runtime, "verify_environment", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runtime, "_cached_gpu_capability", lambda *a, **kw: None)

    # Planner must identify work owed and require capabilities
    args = pipeline.parser_for().parse_args(["all", "--model", "tme"])
    stages = pipeline.all_stages(args, cfg)
    aux_stage = next(s for s in stages if s.name == "auxiliary-assets" and s.model_id == "tme")
    assert aux_stage.required_capabilities == frozenset({"allow_network", "allow_large_downloads"})

    # Runner without authorized capabilities refuses execution
    ret = pipeline.run_stages(
        [aux_stage],
        dry_run=False,
        apply=True,
        authorized_capabilities=set(),  # NOT authorized!
        config=cfg,
        mode="all",
    )
    assert ret == 1, "Pipeline must refuse stage without required capabilities"

    # Authorized execution performs repair
    admitted_bytes = b"non_zero_admitted_weight_bytes"
    monkeypatch.setattr(downloader, "download", lambda url, partial, limit: partial.write_bytes(admitted_bytes))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_BLIP2_PRETRAINED_PTH", str(len(admitted_bytes)))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_EVA_VIT_G_PTH", str(len(admitted_bytes)))

    ret_auth = downloader.main(["--model", "tme"])
    assert ret_auth == 0
    assert w1.stat().st_size == len(admitted_bytes)
    assert w2.stat().st_size == len(admitted_bytes)

    # Unchanged resume sees warm verified assets: capabilities now empty
    stages_resume = pipeline.all_stages(args, cfg)
    aux_stage_resume = next(s for s in stages_resume if s.name == "auxiliary-assets" and s.model_id == "tme")
    assert aux_stage_resume.required_capabilities == frozenset()
    assert aux_stage.commands == aux_stage_resume.commands
