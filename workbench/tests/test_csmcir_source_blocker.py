"""CSMCIR's pinned source cannot import, and the workbench must keep saying so.

The registered entrypoint is `src/validate_blip_csmcir.py`, which imports `lavis.models`. That module
executes a block of `from … import …` statements naming files the author never committed, *before* the
line that registers the class CSMCIR needs, so the import raises and the model name never resolves.
The workbench does not patch the model package (that would change which classes LAVIS registers), so
the honest state is a documented blocker. These tests lock that state in, so a future edit to the
pinned checkout or to the registry cannot silently turn CSMCIR into a model that looks runnable.

They read the pinned checkout directly. If the checkout is absent they skip: the claim is about that
source, and a missing checkout is not evidence either way.
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


def test_the_model_package_references_files_that_do_not_exist() -> None:
    """The defect: unguarded imports of absent modules, before the registration CSMCIR needs."""
    missing = _unresolvable_imports()
    assert len(missing) > 1, f"expected the known broken imports, found {len(missing)}"

    # Every one of them sits before the line that imports CSMCIR's own class, so the failure happens
    # first and the registration below it is never reached.
    registration_line = next(
        number for number, line in enumerate(INIT.read_text(encoding="utf-8").splitlines(), 1)
        if "blip2_qformer_cir_align_prompt_csmcir" in line)
    assert max(number for number, _ in missing) < registration_line, \
        "the broken imports no longer precede the registration, so the package may now load"


def test_the_missing_imports_are_not_guarded() -> None:
    """A try/except would make this survivable, and would change the conclusion."""
    source = INIT.read_text(encoding="utf-8").splitlines()[:INIT.read_text(encoding="utf-8")
                                                             .splitlines().index(
        next(line for line in INIT.read_text(encoding="utf-8").splitlines()
             if "blip2_qformer_cir_align_prompt_csmcir" in line))]
    assert not any(line.strip().startswith(("try:", "except")) for line in source), \
        "the model package now guards its imports; re-check whether CSMCIR can load"


def test_the_needed_class_is_the_one_registered_under_the_expected_name() -> None:
    """The adapter asks LAVIS for this exact name, so the registration string matters."""
    assert MODEL_FILE.is_file(), MODEL_FILE
    text = MODEL_FILE.read_text(encoding="utf-8")
    assert '@registry.register_model("blip2_cir_align_prompt_csmcir")' in text, \
        "the model's registered name changed; the generated command would no longer resolve"
