"""The URL must mean exactly one context, and an invalid URL must say so.

A workbench whose state lives only in React memory cannot be refreshed, linked, or navigated with
Back/Forward, and a silently coerced value is worse than a rejection: it shows one subset's numbers
under another subset's label. These tests execute `workbench/frontend/routes.ts`.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("routes.ts", Path("/tmp/cir_frontend_routes"))
SRC = json.dumps(str(MODULE))
PROTOCOLS = ["circus_fashioniq_release_v1", "fashioniq_original_split", "fashioniq_val_split",
             "fashioniq_full_gallery_ref_excluded"]


def _parse(search: str) -> dict:
    return support.run_json(f"""
import * as R from {SRC};
const out = R.parseLocation({json.dumps(search)}, {json.dumps(PROTOCOLS)});
console.log('__R__' + JSON.stringify({{location: out.location, problems: out.problems,
                             roundTrip: R.toSearch(out.location)}}));
""")


def test_a_new_context_url_round_trips_exactly() -> None:
    out = _parse("?area=failures&view=disagreement&protocol=circus_fashioniq_release_v1"
                 "&subset=sf&run=abc&runs=r1,r2&query=7:1&condition=text_only_black_image"
                 "&outcome=unresolved&category=dress&order=best&offset=40&q=7%3A2&k=20&preset=80")
    location = out["location"]
    assert location["area"] == "failures"
    assert location["view"] == "disagreement"
    assert location["subset"] == "sf"
    assert location["runs"] == ["r1", "r2"]
    assert location["condition"] == "text_only_black_image"
    assert location["offset"] == 40 and location["k"] == 20 and location["preset"] == "80"
    assert out["problems"] == []
    # The canonical form of the same state must parse back to the same state.
    again = _parse(out["roundTrip"])
    assert again["location"] == location, again


LEGACY_TABS = ["Dashboard", "Models", "Evaluation Runs", "Benchmark", "Composition",
               "Sample Explorer", "Compare Models", "Common Failures", "Disagreement",
               "Failure Analytics", "Annotations", "Hypotheses"]

#: Each tab must land on ITS OWN view, not the area's first: `Benchmark` landing on the composition
#: table and `Disagreement` on the consensus table were silent misroutes found by reading the
#: running app rather than the source.
LEGACY_EXPECTED_VIEW = {
    "Dashboard": "", "Models": "models", "Evaluation Runs": "runs", "Benchmark": "benchmark",
    "Composition": "composition", "Sample Explorer": "explorer", "Compare Models": "compare",
    "Common Failures": "consensus", "Disagreement": "disagreement",
    "Failure Analytics": "analytics", "Annotations": "annotations", "Hypotheses": "cohorts",
}


def test_every_one_of_the_twelve_legacy_tabs_lands_somewhere_correct() -> None:
    out = support.run_json(f"""
import * as R from {SRC};
const legacy = {json.dumps(LEGACY_TABS)};
const rows = legacy.map((page) => {{
  const parsed = R.parseLocation(`?page=${{encodeURIComponent(page)}}`, {json.dumps(PROTOCOLS)});
  const views = R.areaDef(parsed.location.area).views.map((v) => v.id);
  return {{page, area: parsed.location.area, view: parsed.location.view,
           viewIsValid: views.length === 0 || views.includes(parsed.location.view),
           problems: parsed.problems}};
}});
console.log('__R__' + JSON.stringify({{rows, covered: {json.dumps(LEGACY_TABS)}.every((page) => R.LEGACY_PAGES.includes(page))}}));
""")
    assert out["covered"], "a legacy tab name is not in the route module's map"
    assert len(out["rows"]) == 12, out
    expected = LEGACY_EXPECTED_VIEW
    for row in out["rows"]:
        assert row["problems"] == [], f"{row['page']} produced a complaint: {row}"
        assert row["viewIsValid"], f"{row['page']} mapped to an unknown view: {row}"
        assert row["view"] == expected[row["page"]], (
            f"{row['page']} landed on view {row['view']!r}, expected "
            f"{expected[row['page']]!r} — a bookmarked tab must open its own view")


def test_the_old_composition_link_keeps_working_and_names_the_release_protocol() -> None:
    """Old bookmarks carry `?page=Composition&comp_subset=sf&comp_run=…` and no protocol.

    They rendered CIRCUS release content while the protocol control said "FashionIQ Original". The
    route must resolve to the evaluation area AND the release protocol, so the label follows data.
    """
    out = _parse("?page=Composition&comp_subset=sf&comp_run=abc&comp_cond=multimodal&comp_offset=20")
    location = out["location"]
    assert location["area"] == "evaluation" and location["view"] == "composition", location
    assert location["protocol"] == "circus_fashioniq_release_v1", location
    assert location["subset"] == "sf" and location["run"] == "abc", location
    assert location["offset"] == 20, location


def test_an_invalid_value_is_reported_not_coerced() -> None:
    out = _parse("?area=nowhere&subset=bogus&condition=Not A Condition&order=sideways&offset=-4"
                 "&k=999&protocol=made_up&outcome=invented&category=hats&preset=maybe&view=ghost")
    problems = {problem["param"]: problem for problem in out["problems"]}
    assert set(problems) == {"area", "subset", "condition", "order", "offset", "k",
                             "protocol", "outcome", "category", "preset"}, problems
    # A rejected value must not have been applied.
    location = out["location"]
    assert location["subset"] == "sf", "an invalid subset was coerced into a real one"
    assert location["protocol"] == "circus_fashioniq_release_v1", location
    assert location["offset"] == 0 and location["k"] == 10, location
    assert location["category"] == "" and location["outcome"] == "", location
    for problem in problems.values():
        assert problem["message"], "a rejected value must explain itself"


def test_an_unknown_page_is_reported_rather_than_silently_replaced() -> None:
    out = _parse("?page=No+Such+Tab")
    assert [problem["param"] for problem in out["problems"]] == ["page"]
    assert out["location"]["area"] == "overview"


def test_the_defaults_are_a_real_shareable_context() -> None:
    out = support.run_json(f"""
import * as R from {SRC};
const fresh = R.defaultLocation();
const parsed = R.parseLocation("", {json.dumps(PROTOCOLS)});
console.log('__R__' + JSON.stringify({{fresh, parsed: parsed.location, problems: parsed.problems,
                            search: R.toSearch(fresh)}}));
""")
    assert out["problems"] == [], out
    assert out["fresh"] == out["parsed"], "the empty URL must parse to the documented default"
    # Fixed protocol: the release protocol is what this deployment actually serves query data for.
    assert out["fresh"]["protocol"] == "circus_fashioniq_release_v1", out["fresh"]
    assert "area=overview" in out["search"], out


def test_the_shell_reads_and_writes_the_url() -> None:
    """The pure route module is only useful if the app renders from it and pushes history."""
    app = (support.FRONTEND / "App.tsx").read_text(encoding="utf-8")
    assert 'from "./routes"' in app, "App.tsx does not import the route module"
    assert "parseLocation(" in app and "toSearch(" in app, "the URL is not the source of truth"
    assert "popstate" in app, "Back/Forward cannot restore context without a popstate listener"
    assert "pushState" in app, "navigating must push history, or Back skips whole areas"


def test_the_native_scope_has_a_view_that_lists_it() -> None:
    """W-02: a native-only row must open a table that actually contains it.

    The link used to be `?view=benchmark&protocol=…`, and the benchmark table lists `aggregate`
    reports only — so a native macro row opened an empty page under a protocol label. The view has
    to exist in the area's own view list, or `parseLocation` rejects the URL that links to it.
    """
    out = _parse("?area=evaluation&view=native&protocol=fashioniq_full_gallery_ref_excluded")
    assert out["problems"] == [], out["problems"]
    assert out["location"]["area"] == "evaluation"
    assert out["location"]["view"] == "native", out["location"]
    # And it round-trips, so a copied link to a native row stays on that view.
    assert "view=native" in out["roundTrip"], out["roundTrip"]


def test_a_condition_id_the_protocol_triple_does_not_name_is_admitted() -> None:
    """A condition id is a backend identifier, not a closed three-item enum.

    `clvc_image_encoder_only` is a real condition of a real run. The parser used to reject anything
    outside the protocol constant, so a link to the condition a variant run actually scored carried
    a `condition` problem and the value was dropped -- the operator could not select the one
    condition that run measured. Whether a condition EXISTS is a fact about the selected run, which
    only a payload can answer, so the parser admits the id and the page degrades an unknown one to
    the run's first condition.
    """
    out = _parse("?area=samples&view=explorer&condition=clvc_image_encoder_only")
    assert out["problems"] == [], out["problems"]
    assert out["location"]["condition"] == "clvc_image_encoder_only", out["location"]

def test_the_protocol_triple_still_parses_exactly_as_before() -> None:
    """Byte-identical for the runs whose conditions ARE the protocol triple."""
    for condition in ("multimodal", "text_only_black_image", "image_only_empty_text"):
        out = _parse(f"?area=samples&view=explorer&condition={condition}")
        assert out["problems"] == [], (condition, out["problems"])
        assert out["location"]["condition"] == condition, out["location"]
    # No condition in the URL keeps the documented default.
    out = _parse("?area=samples&view=explorer")
    assert out["location"]["condition"] == "multimodal", out["location"]

def test_the_run_driven_option_list_prefers_the_payload_and_falls_back_to_the_triple() -> None:
    """One function decides the options, so the select and the request cannot disagree.

    * a payload naming an ablation offers THAT ablation, and not the protocol's third condition;
    * a payload naming none (older artifact, catalog row with no run) offers the protocol triple
      exactly as before;
    * an unknown id is never given another condition's label.
    """
    serveds = json.dumps([
        [{"id": "multimodal", "label": "Image + text"},
         {"id": "text_only_black_image", "label": "Text only"},
         {"id": "clvc_image_encoder_only", "label": "Image only (image encoder)"}],
        None,
        [],
        [{"id": "mystery_condition", "label": None}],
    ])
    out = support.run_json(f"""
import * as R from {SRC};
const serveds = {serveds};
const out = serveds.map((served) => R.conditionsForRun(served));
const legacy = R.conditionsForRun([
  {{ id: "multimodal" }}, {{ id: "text_only_black_image" }}, {{ id: "image_only_empty_text" }}]);
console.log('__R__' + JSON.stringify({{
  variant: out[0], none: out[1], empty: out[2], unknown: out[3], legacy,
  degrade: [R.resolveCondition("image_only_empty_text", out[0]),
            R.resolveCondition("clvc_image_encoder_only", out[0]),
            R.resolveCondition("nope", out[0])],
  protocol: R.conditionsForRun(null).map((item) => item.id),
}}));
""")
    variant = [item["id"] for item in out["variant"]]
    assert variant == ["multimodal", "text_only_black_image", "clvc_image_encoder_only"], out
    assert "image_only_empty_text" not in variant
    assert out["variant"][2]["label"] == "Image only (image encoder)", out["variant"]

    # Absent payload -> the protocol triple, same ids and same labels as the constant.
    assert out["protocol"] == ["multimodal", "text_only_black_image", "image_only_empty_text"], out
    assert [item["id"] for item in out["none"]] == out["protocol"]
    assert [item["id"] for item in out["empty"]] == out["protocol"]
    assert out["none"][2]["label"] == "Image-only (text rỗng)", out["none"]
    assert out["legacy"] == out["none"], "a label-less payload must render the constant's labels"

    # An unknown id keeps its own name; it is never relabelled as another condition.
    assert out["unknown"][0]["id"] == "mystery_condition"
    assert out["unknown"][0]["label"] == "mystery_condition", out["unknown"]

    # A condition the run lacks degrades to the run's first, and a served one is kept.
    assert out["degrade"] == ["multimodal", "clvc_image_encoder_only", "multimodal"], out["degrade"]
