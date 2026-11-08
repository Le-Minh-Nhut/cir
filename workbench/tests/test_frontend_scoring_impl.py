"""Runs scored with different contractions must not be ranked as if they were comparable.

`einsum`, `einsum(..., optimize=True)` and `matmul` accumulate in different orders, so a rank from
one implementation is not comparable to a rank from another at query level — the equivalence audit
measured 207/18,009 rank changes from that alone. The table shows all of them side by side, so it has
to say when the visible rows mix implementations, and stay silent when they do not: a notice that
always fires is noise, and one that never fires asserts a comparability the arithmetic lacks.

Pure function, no DOM: `scoringImplsPresent` is what the component branches on.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("catalog.ts", Path("/tmp/cir_frontend_catalog_scoring"))
SRC = json.dumps(str(MODULE))


def _present(rows: list[dict]) -> list[str]:
    script = (
        f"import {{ scoringImplsPresent }} from {SRC};\n"
        f"console.log('__R__' + JSON.stringify(scoringImplsPresent({json.dumps(rows)})));"
    )
    return support.run_json(script)


def _run(model: str, impl: str | None) -> dict:
    run = {"run_id": f"{model}-1", "model_id": model, "checkpoint_id": "ckpt",
           "protocol_id": "circus_fashioniq_release_v1"}
    if impl is not None:
        run["scoring_impl"] = {"impl": impl}
    return run


def test_a_mixed_cohort_names_every_implementation() -> None:
    """The live cohort's real shape: three implementations in one table."""
    rows = [
        _run("habit", "einsum"), _run("intent", "einsum"), _run("hint", "einsum"),
        _run("conesep", "einsum-optimized"), _run("ptha_mtst", "einsum-optimized"),
        _run("limn", "matmul"), _run("encoder", "matmul"),
    ]
    present = _present(rows)
    assert present == ["einsum", "einsum-optimized", "matmul"], present
    # More than one distinct value is exactly the condition the component renders a notice on.
    assert len(present) > 1


def test_a_uniform_cohort_produces_no_notice() -> None:
    """All four naive-einsum models on their own: one implementation, so nothing to warn about."""
    present = _present([_run("habit", "einsum"), _run("intent", "einsum"), _run("tme", "einsum")])
    assert present == ["einsum"], present
    assert len(present) == 1


def test_a_missing_derivation_is_its_own_value_not_folded_into_a_real_one() -> None:
    """An absent backend derivation is `unknown`, never silently treated as `einsum`."""
    present = _present([_run("habit", "einsum"), _run("mystery", None)])
    assert present == ["einsum", "unknown"], present
    assert len(present) > 1


def test_an_empty_table_has_nothing_to_announce() -> None:
    assert _present([]) == []
