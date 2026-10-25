"""Tests for PTHA+MTST source integrity, evaluator flags, and preflight blockers."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.ptha_mtst import evaluator_path, preflight
MTST_DIR = REPO / "workbench" / "third_party" / "MTST_PTHA"
DATA_UTILS = MTST_DIR / "MTST_PTHA" / "src" / "data_utils.py"
MODELS_INIT = MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "__init__.py"
EVALUATOR = MTST_DIR / "MTST_PTHA" / "src" / "validate_blip.py"
DISPATCHER = MTST_DIR / "MTST_PTHA" / "src" / "blip_validate.py"


def test_mtst_data_root_fallback() -> None:
    assert DATA_UTILS.is_file(), f"Missing {DATA_UTILS}"
    content = DATA_UTILS.read_text(encoding="utf-8")
    assert 'os.environ.get("MTST_DATA_ROOT", "path_to_data_dir")' in content, (
        "data_utils.py lacks MTST_DATA_ROOT fallback"
    )


def test_guarded_import_in_models_init() -> None:
    assert MODELS_INIT.is_file(), f"Missing {MODELS_INIT}"
    content = MODELS_INIT.read_text(encoding="utf-8")
    assert "Blip2Qformer4TGIRCESIMZYN = None" in content, (
        "models/__init__.py lacks guarded import for Blip2Qformer4TGIRCESIMZYN"
    )
    assert "@registry.register_model(\"blip2_tgir_ce_sim\")" in (
        MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "blip2_models" / "blip2_qformer_tgir_concatembedding_sim.py"
    ).read_text(encoding="utf-8")


def test_evaluator_flags() -> None:
    assert DISPATCHER.is_file(), f"Missing {DISPATCHER}"
    content = DISPATCHER.read_text(encoding="utf-8")
    for flag in ["--dataset", "--blip-model-name", "--backbone", "--model-path"]:
        assert f'"{flag}"' in content, f"Missing flag {flag} in {DISPATCHER}"


def test_validate_blip_cli_only_dispatches_cirr() -> None:
    content = EVALUATOR.read_text(encoding="utf-8")
    main = content[content.index("def main():"):]
    assert "if args.dataset == 'CIRR':" in main
    assert "args.dataset == 'fashionIQ'" not in main


def _dataset_root(root: Path) -> Path:
    for category in ("dress", "shirt", "toptee"):
        for relative in (
            f"captions/cap.{category}.val.json",
            f"image_splits/split.{category}.val.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


def test_preflight_uses_pinned_validate_blip_module() -> None:
    assert evaluator_path(MTST_DIR) == EVALUATOR
    assert EVALUATOR.is_file()
    assert DISPATCHER.is_file()
    blockers = preflight(MTST_DIR, REPO / "data" / "fashionIQ_dataset", REPO / "workbench" / "artifacts" / "checkpoints")
    assert not any("pinned evaluator missing" in blocker for blocker in blockers)


def test_cirr_checkpoint_requires_explicit_transfer_opt_in(tmp_path: Path) -> None:
    # The checkpoint root IS the member root: `_replay_command` passes `checkpoint_member_root` of the
    # resolved path, which for a single file is its own directory, and every wrapper then opens
    # `<root>/<registry filename>`. Nesting the model directory here encoded the doubled-id bug this
    # test is supposed to catch, so the fixture now matches the real convention.
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "tuned_mtst_cirr.pt").write_bytes(b"placeholder")
    dataset_root = _dataset_root(tmp_path / "fashioniq")

    normal = preflight(MTST_DIR, dataset_root, checkpoint_root)
    assert any("CIRR weights require explicit transfer opt-in" in blocker for blocker in normal)

    transfer = preflight(
        MTST_DIR,
        dataset_root,
        checkpoint_root,
        transfer_cirr_to_fashioniq=True,
    )
    assert not any("CIRR weights require explicit transfer opt-in" in blocker for blocker in transfer)
    assert not any("CIRR checkpoint missing" in blocker for blocker in transfer)


def test_preflight_blocks_on_missing_fashioniq_weights(tmp_path: Path) -> None:
    blockers = preflight(
        source_root=MTST_DIR,
        dataset_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    assert any("FashionIQ fine-tuned checkpoint missing" in blocker for blocker in blockers)
    assert any("tuned_mtst_cirr.pt" in blocker for blocker in blockers)
    # And the path it names must be `<root>/<filename>`, with no model directory appended twice: a
    # doubled root made a correctly placed checkpoint unreadable, which is the defect R19-02 records.
    assert not any("ptha_mtst/ptha_mtst/" in blocker for blocker in blockers), blockers


def test_the_nested_source_layout_resolves_to_the_directory_holding_src() -> None:
    """The vendored checkout keeps the repository's own top directory inside itself.

    `third_party/MTST_PTHA/` contains `MTST_PTHA/src/`, while `requirements.txt` and `provenance.json`
    sit at the outer level. Both call sites used to name `<checkout>/MTST_PTHA/src` -- wait, they named
    `<checkout>/src`, which does not exist -- so preflight reported the pinned evaluator missing for a
    checkout that contains it, and the import path would have failed on `data_utils`. ONE resolver now
    answers for both, and this test pins the layout it must handle: nested when the tree is nested,
    flat when a future vendoring flattens it.
    """
    from pathlib import Path

    from workbench.replay import ptha_mtst

    real = Path("workbench/third_party/MTST_PTHA")
    if not real.is_dir():
        return pytest.skip("vendored MTST_PTHA checkout is not present")
    tree = ptha_mtst.source_tree(real)
    assert (tree / "src" / "validate_blip.py").is_file(), (
        f"{tree} does not hold src/validate_blip.py; the resolver picked the wrong level")
    # `evaluator_path` must agree with the resolver, since they are read by different callers.
    assert ptha_mtst.evaluator_path(real) == tree / "src" / "validate_blip.py"


def test_a_transfer_run_needs_the_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CIRR checkpoint must never be evaluated on FashionIQ by default.

    The refused message names the reason (the author publishes no FashionIQ fine-tune), so an operator
    who has not read the prompt learns why rather than seeing an empty block.
    """
    from pathlib import Path

    from workbench.replay import ptha_mtst

    source = Path("workbench/third_party/MTST_PTHA")
    if not source.is_dir():
        return pytest.skip("vendored MTST_PTHA checkout is not present")
    dataset = tmp_path / "FashionIQ"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for relative in ptha_mtst.RAW_INPUTS:
        (dataset / relative).parent.mkdir(parents=True, exist_ok=True)
        (dataset / relative).write_text("[]")

    # Flat member root, matching `checkpoint_member_root` and every wrapper's `root / filename`:
    # a nested `ptha_mtst/` here made a correctly placed checkpoint read as missing.
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / ptha_mtst.PUBLISHED_CIRR_WEIGHT).write_bytes(b"weights")

    torch_home = tmp_path / "torch"
    eva = torch_home / "hub/checkpoints/eva_vit_g.pth"
    eva.parent.mkdir(parents=True)
    eva.write_bytes(b"eva")
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_EVA_VIT_G_PTH", "3")
    assert ptha_mtst.preflight(source, dataset, checkpoint_root,
                               transfer_cirr_to_fashioniq=True) == []
    refused = ptha_mtst.preflight(source, dataset, checkpoint_root,
                                  transfer_cirr_to_fashioniq=False)
    assert refused and "transfer opt-in" in refused[0], refused
