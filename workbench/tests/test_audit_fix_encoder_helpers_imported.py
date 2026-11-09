"""Regression: a composition adapter must never call a shared helper it does not bind.

Defect (found while testing F4): the parallel-decode change added throughput helper calls to several
adapters but only some got the imports. Measured on the tree:

* `model_encoder.py` called `limit_torch_threads()` (:330), `resolved_decode_workers()` (:331) and
  `prefetch_rows(...)` (:353) while importing none of the three;
* `model_ptha_mtst.py` called the same three (:320, :321, :345) with the same result.

Both raise `NameError` inside `_worker_main`, before any feature is produced. Each worker's own
`except Exception` converts that into `reply(False, error_type="NameError", ...)`, so the run failed
at model load with a message that looks like a model problem. ENCODER and PTHA+MTST are both models
with published numbers, so neither was runnable from this tree.

**Why no existing test caught it.** The adapter tests import the module and construct `Adapter()`.
Neither enters `_worker_main`, and a `NameError` in a function body is invisible until that line
executes — so a smoke import passes happily on a file that cannot encode a single image. The check
therefore has to be static.

**How this test works.** `symtable` compiles the module, and for every scope it reports symbols that
are *referenced as globals*. Subtracting what the module actually binds — imports, assignments,
definitions, namespace (class/def) symbols — and the builtins yields the names that would raise
`NameError` at call time. This is the interpreter's own analysis, not a string search: a grep for
`limit_torch_threads` passed on the broken file, because the *call* was present and only the *binding*
was missing.

The assertion has two layers, deliberately:

1. **Helper-layer** — no name from `workbench.composition.adapters`'s public surface may be
   unresolved. This is the class of bug being fixed, and it is derived from `adapters.py` so a future
   helper is covered without editing this file.
2. **Allowlist-layer** — the *complete* unresolved set must be a subset of a small allowlist, so a new
   unresolved name outside the helper set also fails here rather than waiting for a GPU run. The
   allowlist is one entry (`__file__`, provided by the interpreter), which keeps it honest.
"""
from __future__ import annotations

import builtins
import importlib
import inspect
import symtable
from pathlib import Path

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "workbench" / "registry" / "models.yaml"

#: Names the interpreter supplies or the module cannot bind itself. Keep this at zero-ish: every
#: entry is a name whose brokenness this test refuses to detect.
UNRESOLVED_ALLOWLIST: frozenset[str] = frozenset({"__file__"})


def _adapter_modules() -> list[str]:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return sorted(row["composition_adapter"] for row in registry["models"]
                  if row.get("composition_adapter"))


def _helper_names() -> frozenset[str]:
    """The public callables `workbench.composition.adapters` offers to adapters."""
    module = importlib.import_module("workbench.composition.adapters")
    return frozenset(name for name, value in vars(module).items()
                     if not name.startswith("_") and callable(value))


def unresolved_globals(source: str, filename: str) -> set[str]:
    """Names the module reads as globals but never binds — i.e. `NameError` at call time."""
    top = symtable.symtable(source, filename, "exec")

    referenced: set[str] = set()

    def collect(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                referenced.add(symbol.get_name())
        for child in table.get_children():
            collect(child)

    collect(top)

    bound: set[str] = set()
    for symbol in top.get_symbols():
        # `is_imported` covers `import x` / `from m import x`; `is_assigned` covers `x = ...`;
        # `is_namespace` covers `def x` / `class x`. Together these are every way a module-scope
        # name becomes resolvable for a function defined in the module.
        if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace():
            bound.add(symbol.get_name())

    return referenced - bound - set(dir(builtins))


@pytest.mark.parametrize("module_path", _adapter_modules())
def test_adapter_binds_every_helper_it_calls(module_path: str) -> None:
    """The class-level check: no adapter may call an adapter helper it never bound."""
    module = importlib.import_module(module_path)
    source = inspect.getsource(module)
    unresolved = unresolved_globals(source, f"{module_path}.py")

    missing_helpers = sorted(unresolved & _helper_names())
    assert not missing_helpers, (
        f"{module_path} calls {missing_helpers} but never imports or defines them; the call raises "
        f"NameError inside the worker, which the worker reports as a model-load failure")


@pytest.mark.parametrize("module_path", _adapter_modules())
def test_adapter_has_no_unexplained_unresolved_names(module_path: str) -> None:
    """Wider net: any unbound global is a latent `NameError`, not only the known helper set."""
    module = importlib.import_module(module_path)
    source = inspect.getsource(module)
    unresolved = unresolved_globals(source, f"{module_path}.py")

    unexpected = sorted(unresolved - UNRESOLVED_ALLOWLIST)
    assert not unexpected, (
        f"{module_path} references {unexpected} without binding them; add the import (or, if the "
        f"name is genuinely supplied at runtime, add it to UNRESOLVED_ALLOWLIST with a reason)")


def test_every_adapter_is_discovered() -> None:
    """The registry decides the population; and every declared module must actually resolve.

    This used to assert `len(_adapter_modules()) == 9`. Adding the Air-Know adapter (`20a406a`)
    made the real population 10, so the literal failed while nothing was wrong — and it would have
    to be bumped by hand for every future model, which is a guard that drifts rather than a guard
    that protects. What must hold instead:

      * the population is non-empty and duplicate-free,
      * each registry module imports,
      * each exports an `ADAPTER` class that subclasses the shared base,
      * every model row's `composition_adapter` appears in the discovered set, so a typo in a
        registry path cannot hide behind a count that happens to match.
    """
    from workbench.composition.adapters import CompositionAdapter

    modules = _adapter_modules()
    assert modules, "no composition adapter declared; this test would otherwise pass vacuously"
    assert len(modules) == len(set(modules)), (
        f"duplicate adapter module(s): "
        f"{sorted({m for m in modules if modules.count(m) > 1})}")

    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    declared_by_model = {row["model_id"]: row["composition_adapter"] for row in registry["models"]
                         if row.get("composition_adapter")}
    assert set(declared_by_model.values()) == set(modules), (
        "the discovered module list disagrees with the per-model declarations")

    for model_id, module_path in declared_by_model.items():
        module = importlib.import_module(module_path)
        adapter = getattr(module, "ADAPTER", None)
        assert isinstance(adapter, type) and issubclass(adapter, CompositionAdapter), (
            f"{model_id}: {module_path} does not export an ADAPTER subclassing CompositionAdapter")
        assert getattr(adapter, "model_id", None) == model_id, (
            f"{model_id}: {module_path}.ADAPTER declares model_id "
            f"{getattr(adapter, 'model_id', None)!r}; the registry and the adapter disagree")


def test_helper_set_is_not_empty_and_named() -> None:
    """Guard the fixture: the helper set must still contain the names this defect involved."""
    helpers = _helper_names()
    for name in ("limit_torch_threads", "prefetch_rows", "resolved_decode_workers",
                 "resolved_batch_size", "worker_environment"):
        assert name in helpers, f"adapters.py no longer exposes {name!r}"


def test_the_detector_fires_on_the_original_defect() -> None:
    """Self-check: strip the imports from a real adapter source and the detector must complain.

    Without this, a detector that silently returned nothing would pass both checks above.
    """
    from workbench.composition import model_ptha_mtst

    source = inspect.getsource(model_ptha_mtst)
    broken = source.replace(
        "    limit_torch_threads, prefetch_rows, resolved_decode_workers,\n", "", 1)
    assert broken != source, "fixture drifted: the import line to remove is gone"

    unresolved = unresolved_globals(broken, "model_ptha_mtst.py")
    assert {"limit_torch_threads", "prefetch_rows", "resolved_decode_workers"} <= unresolved, (
        f"detector missed the original defect; it found only {sorted(unresolved)}")


@pytest.mark.parametrize("module_path", [
    "workbench.composition.model_encoder",
    "workbench.composition.model_ptha_mtst",
])
def test_the_two_fixed_adapters_bind_the_worker_helpers(module_path: str) -> None:
    """Pin the specific files that were broken, bound to the shared implementations."""
    from workbench.composition import adapters

    module = importlib.import_module(module_path)
    for name in ("limit_torch_threads", "prefetch_rows", "resolved_decode_workers"):
        assert hasattr(module, name), f"{module_path} does not bind {name}"
        assert getattr(module, name) is getattr(adapters, name), (
            f"{module_path}.{name} is not the shared implementation")
