"""CSMCIR's PUBLISHED source cannot import, and the repair must be the declared one.

The registered entrypoint is `src/validate_blip_csmcir.py`, which imports `lavis.models`. In the
published revision that module runs a block of `from … import …` statements naming files the author
never committed, *before* the line that registers the class CSMCIR needs, so the import raises and the
model name never resolves.

This module used to claim the workbench therefore never patched the package. That is no longer the
contract, and this file was rewritten rather than deleted because the underlying defect is still real
and still worth pinning:

  * the DEFECT is a property of the published bytes, so it is asserted against the declared
    `base_sha256`, reconstructed from git rather than read from a working tree that may be repaired;
  * the REPAIR is the declared patch, so a checkout is accepted in exactly two states -- published
    base, or the recorded `derived_sha256` -- and any third state is refused;
  * the repaired tree is what makes the checkpoint's own class reachable, which is why the repair
    exists at all (CHECKPOINT_SELECTION_REPORT.md 3.23).

Reading the working tree and asserting it is pristine is what made this module fail on every host
that had prepared the model. The claim it should have made is about the published revision.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CSMCIR = REPO / "workbench" / "third_party" / "CSMCIR"
INIT = CSMCIR / "src" / "lavis" / "models" / "__init__.py"
MODEL_FILE = (CSMCIR / "src" / "lavis" / "models" / "blip2_models"
              / "blip2_qformer_cir_align_prompt_csmcir.py")
ENTRYPOINT = CSMCIR / "src" / "validate_blip_csmcir.py"

pytestmark = pytest.mark.skipif(
    not INIT.is_file(), reason=f"pinned CSMCIR checkout not present at {CSMCIR}")


def _unresolvable_imports() -> list[tuple[int, str]]:
    """Line numbers and module names of imports whose target file does not exist."""
    base = INIT.parent
    missing: list[tuple[int, str]] = []
    for number, line in enumerate(INIT.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"\s*from (lavis\.models\.[\w.]+) import (\w+)", line)
        if not match:
            continue
        module = match.group(1).replace("lavis.models.", "").replace(".", "/")
        if not (base / f"{module}.py").is_file():
            missing.append((number, match.group(1)))
    return missing


def test_the_entrypoint_really_imports_the_model_package() -> None:
    """The blocker is only meaningful because this import is on the registered path."""
    assert ENTRYPOINT.is_file(), ENTRYPOINT
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "from lavis.models import" in text, "entrypoint no longer imports lavis.models"
    assert re.search(r"^from data_utils_csmcir import .*FashionIQDataset", text, re.M), \
        "entrypoint no longer imports the FashionIQ dataset class"


def _published_init() -> str:
    """The `__init__.py` text of the PUBLISHED revision, whatever the working tree holds."""
    import subprocess

    contract = next(m for m in __import__("workbench.backend.registry", fromlist=["load_registry"])
                    .load_registry()["models"] if m["model_id"] == "csmcir")["approved_source_patches"][0]
    text = INIT.read_text(encoding="utf-8")
    import hashlib as _hashlib
    if _hashlib.sha256(text.encode()).hexdigest() == contract["base_sha256"]:
        return text
    scratch = Path(__import__("tempfile").mkdtemp()) / "CSMCIR"
    staged = scratch / contract["path"]
    staged.parent.mkdir(parents=True)
    staged.write_text(text)
    subprocess.run(["patch", "-p1", "--reverse", "--quiet", "--directory", str(scratch), "-i",
                    str(REPO / contract["patch_file"])], check=True, capture_output=True, text=True)
    return staged.read_text(encoding="utf-8")


def test_the_published_package_references_files_that_do_not_exist() -> None:
    """The defect is a property of the published bytes, not of the working tree."""
    published = _published_init()
    base = INIT.parent
    missing: list[tuple[int, str]] = []
    for number, line in enumerate(published.splitlines(), 1):
        match = re.match(r"\s*from (lavis\.models\.[\w.]+) import (\w+)", line)
        if not match:
            continue
        module = match.group(1).replace("lavis.models.", "").replace(".", "/")
        if not (base / f"{module}.py").is_file():
            missing.append((number, match.group(1)))
    assert len(missing) > 1, f"expected the known broken imports in the published revision, found {len(missing)}"

    registration_line = next(
        number for number, line in enumerate(published.splitlines(), 1)
        if "blip2_qformer_cir_align_prompt_csmcir" in line)
    assert max(number for number, _ in missing) < registration_line, (
        "in the published revision the broken imports must precede the registration, or the defect "
        "described here is not the one that existed")


def test_the_published_imports_are_not_guarded() -> None:
    """A try/except in the published file would make this survivable, and would change the defect."""
    published = _published_init().splitlines()
    registration = next(i for i, line in enumerate(published)
                        if "blip2_qformer_cir_align_prompt_csmcir" in line)
    before = published[:registration]
    assert not any(line.strip().startswith(("try:", "except")) for line in before), (
        "the published revision guards its imports; re-check whether the recorded blocker was real")


def test_the_checkout_is_published_or_the_declared_repair_and_nothing_else() -> None:
    """Exactly two states are acceptable; a third is drift the registry does not describe."""
    import hashlib as _hashlib

    from workbench.backend.registry import load_registry

    model = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    for declaration in model["approved_source_patches"]:
        target = CSMCIR / declaration["path"]
        digest = _hashlib.sha256(target.read_bytes()).hexdigest()
        assert digest in (declaration["base_sha256"], declaration["derived_sha256"]), (
            f"{declaration['path']} matches neither the published base nor the declared repair "
            f"({digest[:16]}...); an undeclared edit is not covered by any approval")


def test_the_needed_class_is_the_one_registered_under_the_expected_name() -> None:
    """The adapter asks LAVIS for this exact name, so the registration string matters."""
    assert MODEL_FILE.is_file(), MODEL_FILE
    text = MODEL_FILE.read_text(encoding="utf-8")
    assert '@registry.register_model("blip2_cir_align_prompt_csmcir")' in text, \
        "the model's registered name changed; the generated command would no longer resolve"
