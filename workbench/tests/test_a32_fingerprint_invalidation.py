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

import os
import subprocess
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


def _startup_env(prefix: Path) -> tuple[Path, str]:
    """A real environment carrying a `sitecustomize.py` that sets a variable at startup."""
    site = _fake_environment(prefix)
    (site / "sitecustomize.py").write_text("import os\nos.environ['A33_MODE'] = 'one'\n")
    return site, str(prefix / "bin" / "python")


def _startup_mode(site: Path, *, via_path: bool = False) -> str:
    """What the interpreter actually exports at startup, read by a real interpreter run.

    `via_path=False` exercises a `*.pth`: the directory is added to the path, which is what makes
    the interpreter process the `.pth` files in it. `via_path=True` exercises `sitecustomize.py`,
    which `site` imports only during a real startup, so the directory must already be on the path.
    """
    env = {key: value for key, value in os.environ.items() if key != "A33_MODE"}
    if via_path:
        env["PYTHONPATH"] = str(site)
        code = "import os; print(os.environ.get('A33_MODE'))"
    else:
        code = (f"import site; site.addsitedir(r'{site}'); "
                "import os; print(os.environ.get('A33_MODE'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_c_rewriting_a_pth_content_moves_the_fingerprint(tmp_path: Path) -> None:
    """Same `.pth` filename, rewritten CONTENT: startup behaviour changed, so identity must move.

    A name-only fold saw an identical entry name and an unchanged distribution set, so the
    authoritative fingerprint did not move while the interpreter's startup environment did.
    """
    prefix = tmp_path / "cir-pth"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")
    pth = site / "research_paths.pth"
    pth.write_text("import os; os.environ['A33_MODE'] = 'one'\n")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)

    pth.write_text("import os; os.environ['A33_MODE'] = 'two'\n")

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != before, (
        "a rewritten .pth changes what the interpreter executes at startup")

    # The behaviour the fingerprint is supposed to witness, demonstrated on a real interpreter.
    assert _startup_mode(site) == "two"


def test_c_rewriting_sitecustomize_content_moves_the_fingerprint(tmp_path: Path) -> None:
    """Same `sitecustomize.py` filename, rewritten CONTENT: identity must move."""
    prefix = tmp_path / "cir-sck"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")
    custom = site / "sitecustomize.py"
    custom.write_text("import os\nos.environ['A33_MODE'] = 'one'\n")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)
    assert _startup_mode(site, via_path=True) == "one"

    custom.write_text("import os\nos.environ['A33_MODE'] = 'two-two'\n")

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != before
    assert _startup_mode(site, via_path=True) == "two-two"


def test_c_rewriting_the_same_startup_bytes_does_not_move_the_fingerprint(tmp_path: Path) -> None:
    """Touch-without-change: identical startup bytes must leave the identity alone."""
    prefix = tmp_path / "cir-same"
    site, interpreter = _startup_env(prefix)
    pth = site / "research_paths.pth"
    pth.write_text("import os; os.environ['A33_MODE'] = 'one'\n")
    custom = site / "sitecustomize.py"

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)
    # Same filenames, same bytes: no semantic change, so no identity change.
    custom.write_text("import os\nos.environ['A33_MODE'] = 'one'\n")
    pth.write_text("import os; os.environ['A33_MODE'] = 'one'\n")

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) == before


def test_c_retargeting_a_symlinked_startup_file_moves_the_fingerprint(tmp_path: Path) -> None:
    """A symlinked `*.pth` is folded by the target it NAMES, so a retarget moves the identity.

    `_member_digest` rejects symlinks, so the content arm silently skipped them and left a
    retargeted `*.pth` invisible while the interpreter's startup behaviour changed.
    """
    prefix = tmp_path / "cir-link"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")
    target_a = tmp_path / "a.pth"
    target_b = tmp_path / "b.pth"
    target_a.write_text("import os; os.environ['A33_MODE'] = 'one'\n")
    target_b.write_text("import os; os.environ['A33_MODE'] = 'one'\n")   # identical bytes
    link = site / "research_paths.pth"
    link.symlink_to(target_a)

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)

    link.unlink()
    link.symlink_to(target_b)

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != before, (
        "a retargeted startup symlink must move the identity even with identical target bytes")

    from workbench.backend import registry as _registry

    # The guard above is only meaningful because the content arm genuinely rejects symlinks.
    with pytest.raises(ValueError):
        _registry.sha256_file(link)


def test_c_rewriting_a_startup_package_moves_the_fingerprint(tmp_path: Path) -> None:
    """The PACKAGE form (`sitecustomize/__init__.py`) executes at startup and must be folded."""
    prefix = tmp_path / "cir-pkg"
    site = _fake_environment(prefix)
    interpreter = str(prefix / "bin" / "python")
    package = site / "sitecustomize"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("import os\nos.environ['A33_MODE'] = 'one'\n")

    before = runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                              authoritative=True)

    init.write_text("import os\nos.environ['A33_MODE'] = 'two-two'\n")

    assert runtime._interpreter_fingerprint(interpreter, "contract", str(prefix),
                                            authoritative=True) != before

    # Real startup behaviour: `site` prefers the package over a same-named module.
    assert _startup_mode(package.parent, via_path=True) == "two-two"
