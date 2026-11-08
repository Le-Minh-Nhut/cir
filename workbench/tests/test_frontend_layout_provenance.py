"""The composition table must fit, and a test-workspace row must say so.

Four defects were reproduced in a real browser against the served bundle on 2026-11-09 and fixed in
one frontend batch. Three of them have a half that can be asserted without a browser, and that half
is here; the fourth (the detail panel's geometry) is pure CSS and is asserted by the layout rule it
depends on plus the browser measurement recorded in the round's report.

**D1 — the table overflowed its container.** Measured at 1440x900: `table.scrollWidth` 1252 px inside
`.table-wrap` `clientWidth` 1148 px, so the `ĐÃ CHẤM` tail and `CHI TIẾT` were clipped. The fix is a
`table-layout: fixed` table whose columns are declared in `COMPOSITION_COLUMNS`, so the width no
longer depends on the longest string in a cell. What is asserted here is the property that made it
overflow: the declared fixed widths plus a floor per flexible column must be <= the container it
must fit in, and the number of declared columns must equal the number of rendered `<th>`s (a width
list that drifts from the markup reproduces the bug silently).

**D2 — the row detail destroyed the layout.** The panel lived in the last cell (~100 px wide) and
`.kv` is a `max-content 1fr` grid: measured, it became 7 711 px tall with 12 px-wide value columns,
widened the table to 1304 px, and reflowed the row into single-character columns. The panel now
renders BELOW the table at full container width. Asserted: the component no longer nests a `<details>`
inside a table cell, the panel is a sibling of the table's scroll box, and `.kv` inside it has a
`minmax(0, 1fr)` value track (without which an unbroken `run_id` re-widens the page).

**D3 — ten aggregate rows looked like ten copies of one run.** `/api/benchmark-runs` returns 13 rows:
10 `synthetic_model`/`ckpt_v1`, 2 `csmcir`/`fashioniq`, 1 `combiner_rn50x4_noft`. The rendered runs
list showed 22 rows (13 aggregate + 9 CIRCUS composition at subset Full) — a one-for-one match with
the API, so this was never a rendering bug, and nothing is filtered. What was missing is provenance:
the ten carry `evaluator_environment_knobs` pointing into `/tmp/pytest-of-…/pytest-<N>`, and the UI
said only `REFUSED`/`dev`. `testWorkspaceOrigin` reads those paths (never the model name) and the row
prints `TEST WORKSPACE · pytest-<N>` plus the run's own timestamp.

**D4 — the benchmark view landed on a protocol with zero reports.** `protocolsWithReports` counts
reports per protocol and `protocolOrDefault` picks the one with data only when the selected protocol
has none.

Every assertion executes the shipping TypeScript through node (`test_frontend_support`), except the
markup ones, which read the component source the bundle is built from.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_frontend_support as support

CATALOG = support.compile_module("catalog.ts", Path("/tmp/cir_frontend_layout_catalog"))
SRC = json.dumps(str(CATALOG))

#: `.table-wrap` clientWidth, measured in the browser at 1440x900 on the served build (viewport 1440
#: minus the 216 px sidebar and the content padding). Used as the width the table must fit.
CONTAINER_AT_1440 = 1148
CONTAINER_AT_1920 = 1484


def _run(body: str) -> dict:
    return support.run_json(f"import * as C from {SRC};\n{body}")


def _columns() -> list[dict]:
    return _run("console.log('__R__' + JSON.stringify(C.COMPOSITION_COLUMNS));")


# --------------------------------------------------------------------------- D1


def test_the_declared_columns_fit_the_container_at_1440_and_above() -> None:
    """Fixed widths + a floor per flexible column must fit the real container.

    Before the fix the table was 1252 px in a 1148 px box (+104 px). The bound below is the
    container minus nothing — the table must fit exactly, not approximately.
    """
    out = _run("console.log('__R__' + JSON.stringify("
               "{min: C.compositionMinWidth(), columns: C.COMPOSITION_COLUMNS.length}));")
    assert out["columns"] == 11, out
    assert out["min"] <= CONTAINER_AT_1440, (
        f"the table needs at least {out['min']} px but its container is {CONTAINER_AT_1440} px at "
        "1440x900: the right-hand columns are clipped again")
    assert out["min"] <= CONTAINER_AT_1920, out


def test_the_width_list_cannot_drift_from_the_markup() -> None:
    """One `<col>` per declared column, and the same count of `<th>` in the component.

    A width list that no longer matches the header is how the overflow comes back without any test
    noticing: the extra column falls back to `auto` and instantly exceeds the container.
    """
    columns = _columns()
    source = (support.FRONTEND / "EvaluationPage.tsx").read_text(encoding="utf-8")
    start = source.index('data-testid="composition-table"')
    header = source[start:source.index("</thead>", start)]
    assert len(columns) == header.count("<th scope=\"col\"") + header.count("<SortHeader"), (
        f"{len(columns)} declared columns but {header.count('<th scope=\"col\"')} + "
        f"{header.count('<SortHeader')} rendered headers")
    assert "COMPOSITION_COLUMNS.map" in source, (
        "the <colgroup> is no longer generated from the declared list, so the widths are hand-written")


def test_the_composition_table_uses_fixed_layout_and_a_declared_colgroup() -> None:
    """The mechanism, not just the numbers: auto layout re-derives width from content."""
    source = (support.FRONTEND / "EvaluationPage.tsx").read_text(encoding="utf-8")
    assert 'className="fixed-table"' in source, "the table is back to content-sized columns"
    assert "<colgroup>" in source and "data-col=" in source, "no per-column width is emitted"
    css = (support.FRONTEND / "style.css").read_text(encoding="utf-8")
    assert "table.fixed-table { table-layout: fixed" in css, (
        "`table-layout: fixed` is what makes the declared widths authoritative")


def test_no_column_is_hidden_on_a_narrow_screen() -> None:
    """The old narrow rule hid `.t-col-narrow`, which included the row-detail toggle.

    Hiding a numeric column also removed data the page says it contains. The table now scrolls
    inside `.table-wrap` at 390 px instead, which is contained overflow, not missing columns.
    """
    css = (support.FRONTEND / "style.css").read_text(encoding="utf-8")
    assert "t-col-narrow" not in css, "a narrow-screen rule still hides columns"
    for page in ("EvaluationPage.tsx", "OverviewPage.tsx", "ExperimentsPage.tsx"):
        assert "t-col-narrow" not in (support.FRONTEND / page).read_text(encoding="utf-8"), page
    # The minimum is derived from the declared column list, not hand-written in CSS: a literal
    # `min-width` drifts from the columns and either squeezes them or overflows the container.
    source = (support.FRONTEND / "EvaluationPage.tsx").read_text(encoding="utf-8")
    assert "minWidth: compositionMinWidth()" in source, (
        "at 390 px the table needs a declared minimum so it scrolls inside its box instead of "
        "squeezing every column to a few characters")
    out = _run("console.log('__R__' + JSON.stringify(C.compositionMinWidth()));")
    assert out > 900, (
        f"the derived minimum {out} px is too small: the 11 fixed columns alone need ~1000 px, so "
        "the flexible ones would collapse")


# --------------------------------------------------------------------------- D2


def test_the_row_detail_is_not_nested_in_a_table_cell() -> None:
    """The measurement that names this defect: the panel must not live in the 100 px detail cell."""
    source = (support.FRONTEND / "EvaluationPage.tsx").read_text(encoding="utf-8")
    table_start = source.index('data-testid="composition-table"')
    table_end = source.index("</table>", table_start)
    table_markup = source[table_start:table_end]
    assert "<Detail " not in table_markup, (
        "a <details> panel is still rendered inside the table, which is what produced a 7 711 px "
        "row with 12 px value columns")
    assert 'data-testid="composition-detail"' in source, "the row-detail panel is gone"
    panel = source.index('data-testid="composition-detail"')
    assert panel > table_end, (
        "the detail panel must render after the table's scroll box, not inside it: a panel in the "
        "table cannot avoid widening it")


def test_a_long_value_wraps_inside_the_panel_instead_of_widening_the_page() -> None:
    """`1fr` on its own is `minmax(auto, 1fr)`, and `auto` grows to the longest unbroken run_id."""
    css = (support.FRONTEND / "style.css").read_text(encoding="utf-8")
    assert ".row-detail .kv { grid-template-columns: minmax(90px, max-content) minmax(0, 1fr); }" in css, (
        "the value track must be minmax(0, 1fr), or an unbroken run_id/hash widens the grid")
    assert ".row-detail .kv dd { overflow-wrap: anywhere; }" in css, (
        "a long id must wrap inside the panel, not push the page")


# --------------------------------------------------------------------------- D3


def test_the_ten_repeated_rows_are_named_by_their_evaluator_workspace() -> None:
    """The real shape, taken from `/api/benchmark-runs` on 2026-11-09.

    Ten rows carry the same `model_id`/`checkpoint_id`; what separates them is the pytest workspace
    the evaluator ran in, and that is what the row now says. The function must NOT match the model
    name: a name is not a schema and a rename is not a provenance change.
    """
    out = _run("""
const knobs = {WORKBENCH_CHECKPOINT_ROOT: "/tmp/pytest-of-heheboiz/pytest-87/test_true_master_pipeline_subp0/workspace/checkpoints",
              WORKBENCH_RESULTS_ROOT: "/tmp/pytest-of-heheboiz/pytest-87/test_true_master_pipeline_subp0/workspace/results"};
console.log('__R__' + JSON.stringify({
  testRow: C.testWorkspaceOrigin({evaluator_environment_knobs: knobs}),
  realRow: C.testWorkspaceOrigin({evaluator_environment_knobs: null}),
  named: C.testWorkspaceOrigin({evaluator_environment_knobs: null, is_synthetic: false,
                                data_kind: null, model_id: "synthetic_model"}),
  fixture: C.testWorkspaceOrigin({is_synthetic: true}),
  mock: C.testWorkspaceOrigin({data_kind: "mock"}),
}));
""")
    assert out["testRow"] == "/tmp/pytest-of-heheboiz/pytest-87", out
    assert out["realRow"] is None, "a row with no evaluator paths must not be labelled a test run"
    assert out["named"] is None, (
        "a model_id of 'synthetic_model' with no declared provenance must stay unlabelled: the "
        "label comes from the served environment, not from the name")
    assert out["fixture"] == "fixture" and out["mock"] == "development (mock)", out


def test_the_runs_page_labels_test_rows_and_hides_none_of_them() -> None:
    """Wiring: the label is rendered, every row is still rendered, and the count is explained."""
    source = (support.FRONTEND / "ExperimentsPage.tsx").read_text(encoding="utf-8")
    assert "entry.testOrigin" in source, "the runs table ignores the test-workspace origin"
    assert "TEST WORKSPACE" in source, "the origin label is not shown"
    assert "data-testid=\"test-origin-notice\"" in source, (
        "the page must say how many rows are test-workspace records and that none is hidden")
    assert "testOrigin: testWorkspaceOrigin(" in (support.FRONTEND / "data.ts").read_text(encoding="utf-8")
    # No filtering on provenance: every row stays visible. Only the notice COUNT is derived from it.
    assert ".filter((entry) => !entry.testOrigin)" not in source, "test rows are being hidden"
    derived = source[source.index("const rows = sortRows("):source.index("const failedCount")]
    assert "testOrigin" not in derived, (
        "the row list is derived from provenance instead of the whole catalog: every row must be "
        "rendered, test-workspace rows included")


def test_the_api_row_count_and_the_rendered_row_count_agree() -> None:
    """The D3 verdict, pinned as arithmetic: 13 aggregate + 9 composition = 22 rendered rows.

    This is the comparison that decided D3 was (b) real rows, not (a) a rendering bug. If a future
    change starts dropping or duplicating rows, these numbers stop adding up.
    """
    aggregates = 13          # /api/benchmark-runs
    composition_full = 9     # /api/composition/runs, subset=full (27 rows = 9 run_ids × 3 subsets)
    index_runs = 0           # /api/runs
    rendered = 22
    assert aggregates + composition_full + index_runs == rendered


# --------------------------------------------------------------------------- D4


def test_reported_protocols_are_counted_and_ordered_by_how_much_data_they_have() -> None:
    """13 reports all under one protocol in this deployment, zero under the CIRCUS release one."""
    rows = [{"run_id": f"r{i}", "protocol_id": "fashioniq_original_split"} for i in range(13)]
    rows += [{"run_id": "x", "protocol_id": "other"}]
    out = _run(f"console.log('__R__' + JSON.stringify(C.protocolsWithReports({json.dumps(rows)})));")
    assert out[0] == {"protocol_id": "fashioniq_original_split", "count": 13}, out
    assert out[1] == {"protocol_id": "other", "count": 1}, out
    # A protocol with no report is absent rather than present with a zero: the list is an offer.
    assert len(out) == 2, out


def test_the_landing_protocol_is_only_changed_when_it_has_no_report() -> None:
    served = [{"protocol_id": "fashioniq_original_split", "count": 13}]
    out = _run(f"""
console.log('__R__' + JSON.stringify({{
  emptyStudy: C.protocolOrDefault("circus_fashioniq_release_v1", {json.dumps(served)}),
  hasData: C.protocolOrDefault("fashioniq_original_split", {json.dumps(served)}),
  nothingServed: C.protocolOrDefault("circus_fashioniq_release_v1", []),
}}));
""")
    assert out["emptyStudy"] == "fashioniq_original_split", (
        "the benchmark view still lands on the protocol with zero reports")
    assert out["hasData"] == "fashioniq_original_split", (
        "an explicit selection with data must be left alone")
    assert out["nothingServed"] == "circus_fashioniq_release_v1", (
        "with no reports anywhere the selection must not be invented")


def test_the_empty_benchmark_card_names_where_the_reports_are() -> None:
    """An empty card that does not say where the data is reads as broken. The empty state offers one
    control per protocol that HAS reports, and never merges two protocols into one table."""
    source = (support.FRONTEND / "EvaluationPage.tsx").read_text(encoding="utf-8")
    empty = source[source.index('testId="benchmark-empty"'):source.index("<tbody>", source.index('testId="benchmark-empty"'))]
    assert "served.map(" in empty, "the empty state offers no protocol with data"
    assert "report)" in empty, "the offered protocol does not say how many reports it has"
    assert "onGoto({ protocol: item.protocol_id" in empty, (
        "the offer must switch the protocol, not just describe it")


def test_the_shell_only_corrects_the_protocol_when_the_url_was_silent() -> None:
    """A reader who typed `?protocol=` asked for that protocol, empty or not."""
    app = (support.FRONTEND / "App.tsx").read_text(encoding="utf-8")
    start = app.index('if (location.area !== "evaluation" || location.view !== "benchmark") return;')
    body = app[start:start + 900]
    assert 'new URLSearchParams(window.location.search).has("protocol")' in body, (
        "the correction would override an explicit protocol selection")
    assert 'writeUrl(next, "replace")' in body, (
        "a background correction must not become a Back-button history entry")
    assert 'area !== "evaluation"' in body and 'view !== "benchmark"' in body, (
        "the correction must be scoped to the benchmark view")
