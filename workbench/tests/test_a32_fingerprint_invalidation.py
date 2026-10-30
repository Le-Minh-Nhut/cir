"""Two fingerprint blind spots found on the audited tip.

A) `_stage_input_fingerprint_evidence` recorded an unresolvable or uninspectable dataset
   symlink's target only in the returned evidence, never in the digest. Retargeting such a link
   left the stage digest byte-identical, so a resume would reuse a result computed from a
   different dataset.

B) `_interpreter_fingerprint` folded the site-packages *directory* mtime, which any interpreter
   import moves by materializing `__pycache__`. A legitimate `--resume` was then refused and the
   stage re-ran with a new run_id.

Both are behavioural: they recompute real fingerprints and fail if the property regresses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.operator_config import WorkbenchConfig
from workbench.scripts import pipeline

DATASET_LINK = "fashionIQ_dataset"


def _config(tmp: Path) -> WorkbenchConfig:
    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp / "repo", CIR_DATA_ROOT=tmp / "data",
        FASHIONIQ_ROOT=tmp / "data" / "FashionIQ",
        WORKBENCH_HOST="127.0.0.1", WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp / "checkpoints", WORKBENCH_RESULTS_ROOT=tmp / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp / "third_party")


def _dataset(root: Path, marker: bytes) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\n" + marker)
    return root


def _declared_link(tmp: Path, target: Path) -> tuple[Path, WorkbenchConfig]:
    config = _config(tmp)
    link = config.WORKBENCH_THIRD_PARTY_ROOT / "IlearnModel" / DATASET_LINK
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return link, config


def _digest(link: Path, config: WorkbenchConfig) -> str:
    # `model_id` is deliberately absent: this is the input-evidence path, not the execution
    # identity. The roots list still comes from the config, which is what classifies the link.
    stage = pipeline.Stage("evaluation", input_paths=(link,))
    return pipeline.stage_input_fingerprint_info(stage, config)["digest"]


@pytest.mark.parametrize("failure", ["resolve", "stat"])
def test_a_uninspectable_symlink_retarget_invalidates(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """An uninspectable link still NAMES a target, and that name decides the dataset."""
    inside = _dataset(tmp_path / "data" / "FashionIQ", b"inside")
    outside = _dataset(tmp_path / "OtherFashionIQ", b"outside")
    link, config = _declared_link(tmp_path, inside)

    real_resolve, real_stat = Path.resolve, Path.stat

    def resolve_with_failure(self: Path, *args: object, **kwargs: object) -> Path:
        if failure == "resolve" and self == link:
            raise OSError(121, "Remote I/O error")
        return real_resolve(self, *args, **kwargs)

    def stat_with_failure(self: Path, *args: object, **kwargs: object) -> object:
        if failure == "stat" and str(self) in {str(inside), str(outside)}:
            raise OSError(5, "Input/output error")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_with_failure)
    monkeypatch.setattr(Path, "stat", stat_with_failure)

    before = _digest(link, config)
    link.unlink()
    link.symlink_to(outside, target_is_directory=True)
    after = _digest(link, config)

    assert after != before, (
        f"a retarget of a link whose {failure} fails must invalidate the stage")


def test_a_external_symlink_retarget_invalidates(tmp_path: Path) -> None:
    """Contrast: the already-folding EXTERNAL arm keeps detecting a retarget."""
    left = _dataset(tmp_path / "left", b"left")
    right = _dataset(tmp_path / "right", b"right")
    link, config = _declared_link(tmp_path, left)

    before = _digest(link, config)
    link.unlink()
    link.symlink_to(right, target_is_directory=True)

    assert _digest(link, config) != before


def _fake_environment(prefix: Path) -> Path:
    """A minimal real environment: an interpreter symlink and one installed distribution."""
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "bin" / "python").symlink_to(sys.executable)
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site.mkdir(parents=True)
    _install_distribution(site, "torch", "2.0.1")
    return site


def _install_distribution(site: Path, name: str, version: str) -> None:
    module = site / f"{name}.py"
    module.write_text(f"__version__ = {version!r}\n")
    dist = site / f"{name}-{version}.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    (dist / "RECORD").write_text(f"{name}.py,sha256=deadbeef,{module.stat().st_size}\n")


def test_b_bytecode_side_effects_do_not_move_the_environment_identity(tmp_path: Path) -> None:
    """Running an environment writes `__pycache__`; that must not rewrite its identity."""
    prefix = tmp_path / "cir-abc"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)

    # Exactly what an import of an un-cached module does: a new `__pycache__` directory holding a
    # compiled module. Its parent directory really does move, so the guard below is meaningful.
    mtime_before_run = site.stat().st_mtime_ns
    (site / "__pycache__").mkdir()
    (site / "__pycache__" / "torch.cpython-000.pyc").write_bytes(b"\x00" * 16)

    assert site.stat().st_mtime_ns != mtime_before_run, (
        "the guard below is only meaningful if the write really moved the directory mtime")

    after = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                             authoritative=True)
    assert after == before, (
        "bytecode materialization must not change the environment identity, or a resume is "
        "refused after every import")


def test_b_a_startup_executing_stray_module_moves_the_fingerprint(tmp_path: Path) -> None:
    """A non-distribution module is still executable content and must move the identity.

    `sitecustomize.py` (and a `*.pth` file) is imported by the interpreter at startup, so a
    dropped-in copy changes evaluator semantics while appearing in no distribution's RECORD and in
    no `*.dist-info`. A fingerprint that only watched distributions would call it unchanged.
    """
    prefix = tmp_path / "cir-ghi"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)

    (site / "sitecustomize.py").write_text("import os\nos.environ['INJECTED'] = '1'\n")

    after = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                             authoritative=True)
    assert after != before, (
        "a startup-executing module with no distribution metadata must change the identity")

    # A `.pth` file is the other startup-executing shape, recorded by nothing.
    (site / "extra_paths.pth").write_text("/tmp/attacker\n")
    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != after


def test_b_a_real_install_still_moves_the_environment_identity(tmp_path: Path) -> None:
    """Positive control: removing the mtime fold must not blind the fingerprint to installs."""
    prefix = tmp_path / "cir-def"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)
    _install_distribution(site, "newdep", "1.2.3")

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != before
