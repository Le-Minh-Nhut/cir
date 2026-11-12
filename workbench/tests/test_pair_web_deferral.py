"""PAIR is DEFERRED from the website, not deleted from the research registry.

The decision: while the official checkpoint mapping is unresolved (`reproduction_status:
BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`, `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`), PAIR stays a registry
entry with its source history, checkpoints, offline CLI and artifacts, and its exploratory numbers
stay readable for offline audit. Only the WEB publication surfaces drop it.

Every assertion below is one of the six things that decision can get wrong:

* the flag does not exist, so nothing is deferred                          (`test_registry_declares_...`)
* the model list still serves it, or serves a count that includes it        (`test_model_list_...`)
* the native catalogue still lists its rows while its own `counts` still count them
  (`test_native_listing_...`, `test_native_counts_...`)
* a new PAIR job is answered with `404 model_not_found`, which is a lie      (`test_queue_refuses_...`)
* the deferral quietly deleted the record instead of unpublishing it        (`test_measurement_...`)
* the UI hint says nothing, so a `?model=pair` deep link reads as "no data" (`test_frontend_notice_...`)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "workbench" / "artifacts"

PAIR_EXPLORATORY = ["61ccfc4d63fb69eb", "27a9b77d146b9b97", "f4655ef992695f90", "8cb65b73274a86f8"]
PAIR_BLOCKER = "a804c2973c98c6de"
#: The roll-up rows whose keys name a variant (`B1/H_normal`), not the model. They were never in the
#: public native table (no `protocol_id`), and deferring PAIR must not be the thing that changes that.
PAIR_ROLLUP_IDS = ["dd42f32772c4e5cf", "bc0db3353d0706ff", "573ac9482e55918c", "c10e512c3a92a4c0",
                   "5913c767a8e7d4a2", "b771dca51e180479", "606fa3cc698a956a", "d780cbab298c45b9"]

# --------------------------------------------------------------------------- the flag


def test_registry_declares_pair_deferred_and_keeps_every_research_fact() -> None:
    """The deferral is a publication flag. Nothing about the research record may move."""
    from workbench.backend.registry import load_registry, web_deferred_models

    registry = load_registry()
    deferred = web_deferred_models(registry)
    assert "pair" in deferred, "the registry does not declare PAIR deferred from the web"
    assert deferred["pair"].strip(), "a deferral without a reason is an unexplained absence"

    pair = next(model for model in registry["models"] if model["model_id"] == "pair")
    assert pair["reproduction_status"] == "BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED"
    assert pair["command_status"] == "COMMAND_BLOCKED_SOURCE_ENTRYPOINT"
    assert pair["strict_clean_checkpoint_available"] is False
    assert {variant["checkpoint_id"] for variant in pair["checkpoint_variants"]} == {"pair_b1", "pair_b2"}
    assert pair["upstream_commit_sha"] and pair["source_dir"] == "PAIR"
    # The registry still holds every entry; only the publication list is narrower.
    assert len(registry["models"]) == 17
    assert len(registry["models"]) - len(deferred) == 16


def test_every_other_registered_model_stays_published() -> None:
    from workbench.backend.registry import load_registry, web_deferred_models

    deferred = web_deferred_models(load_registry())
    assert set(deferred) == {"pair"}, f"unexpected deferral(s): {sorted(deferred)}"


def test_publication_status_vocabulary_is_closed() -> None:
    """A typo in the flag would silently publish the model again (`DEFERED` is not `DEFERRED`)."""
    from workbench.backend.registry import load_registry

    registry = load_registry()
    for model in registry["models"]:
        block = model.get("web_publication")
        if block is None:
            continue
        assert block["status"] in {"PUBLISHED", "DEFERRED"}, model["model_id"]
        if block["status"] == "DEFERRED":
            assert block.get("reason", "").strip(), model["model_id"]


# --------------------------------------------------------------------------- the model list


@pytest.fixture()
def served_models(monkeypatch: pytest.MonkeyPatch):
    """`/api/models` through the REAL handler, with the cheap parts of readiness stubbed.

    The route's own filter is what is under test; hashing every checkpoint (its cold start) is
    not. `load_registry` is the real document, because the deferral lives in it.
    """
    from workbench.backend import main

    monkeypatch.setattr(main, "preflight_blockers", lambda model, config: [])
    monkeypatch.setattr(main, "_models_state_stamp", lambda registry, config: ("stub",))
    original = main.resolve_config

    class _Config:
        WORKBENCH_CHECKPOINT_ROOT = REPO / "workbench" / "artifacts" / "checkpoints"
        WORKBENCH_THIRD_PARTY_ROOT = REPO / "workbench" / "third_party"
        WORKBENCH_ARTIFACTS_ROOT = ARTIFACTS
        FASHIONIQ_ROOT = REPO / "workbench" / "artifacts" / "fashioniq"

    monkeypatch.setattr(main, "resolve_config", lambda: _Config())
    monkeypatch.setattr(main, "checkpoint_availability",
                        lambda model, item, model_blockers=None: {
                            "checkpoint_id": item["checkpoint_id"], "runnable": False,
                            "checkpoint_downloaded": False, "adapter_command_ready": False,
                            "blocking_reasons": []})
    main._MODELS_CACHE = None
    main._MODELS_CACHE_KEY = None
    yield main
    main._MODELS_CACHE = None
    main._MODELS_CACHE_KEY = None
    monkeypatch.setattr(main, "resolve_config", original)


def test_model_list_does_not_serve_the_deferred_model(served_models) -> None:
    from workbench.backend.registry import load_registry, web_deferred_models

    served = served_models.models()
    ids = [model["model_id"] for model in served]
    assert "pair" not in ids, "PAIR is still on the model inventory surface"
    assert len(ids) == 16 and len(set(ids)) == 16, ids
    registry = load_registry()
    assert len(ids) == len(registry["models"]) - len(web_deferred_models(registry))
    # Nothing else was withheld with it.
    assert set(ids) == {model["model_id"] for model in registry["models"]} - {"pair"}


# --------------------------------------------------------------------------- the native catalogue


def test_native_listing_omits_the_deferred_models_rows() -> None:
    from workbench.backend.native_measurements import list_native_measurements

    payload = list_native_measurements(ARTIFACTS, exclude_model_ids=frozenset({"pair"}))
    listed = {row["measurement_id"] for row in payload["measurements"]}
    assert not listed & set(PAIR_EXPLORATORY + [PAIR_BLOCKER]), (
        "a PAIR row is still listed by the native table")
    assert len(payload["measurements"]) == 51, len(payload["measurements"])


def test_native_counts_are_recomputed_from_the_visible_rows() -> None:
    """A count computed before the filter is the defect this decision must not reintroduce."""
    from workbench.backend.native_measurements import list_native_measurements

    full = list_native_measurements(ARTIFACTS)
    listed = list_native_measurements(ARTIFACTS, exclude_model_ids=frozenset({"pair"}))
    visible = listed["measurements"]

    assert full["counts"]["blocks_with_metrics"] == listed["counts"]["blocks_with_metrics"] + 4
    assert full["counts"]["signposted_blockers"] == listed["counts"]["signposted_blockers"] + 1
    assert listed["counts"]["blocks_with_metrics"] == sum(1 for row in visible if row.get("metrics"))
    assert listed["counts"]["signposted_blockers"] == sum(
        1 for row in visible if not row.get("metrics") and row.get("blocker"))
    assert listed["counts"]["models_with_metrics"] == len(
        {row["model_id"] for row in visible
         if row.get("metrics") and row.get("counted", True) and row.get("artifact_backed")})
    # No PAIR number is smuggled into another model's count.
    assert "pair" not in {row.get("model_id") for row in visible}


def test_pair_rollup_rows_do_not_enter_the_public_table() -> None:
    """The roll-ups are keyed `B1/H_*`, so a filter on `model_id` cannot hide them by accident.

    They stay served-but-invisible for the reason they always were: no `protocol_id`, so the table
    (`row.metrics && row.protocol_id === protocol`) never lists them. If the deferral had been
    implemented by dropping PAIR's files instead, these would have changed too.
    """
    from workbench.backend.native_measurements import list_native_measurements

    payload = list_native_measurements(ARTIFACTS)
    by_id = {row["measurement_id"]: row for row in payload["measurements"]}
    for measurement_id in PAIR_ROLLUP_IDS:
        row = by_id[measurement_id]
        assert row["rollup"] is True and row["counted"] is False
        assert row["protocol_id"] is None
        assert not (row["metrics"] and row["protocol_id"]), "a roll-up became a public table row"


def test_publication_counts_agree_with_the_served_list(served_models) -> None:
    """The invariant a client is allowed to rely on — both sides derived, no literal.

    A dashboard that prints a count computed before a filter is the defect this decision was made
    to avoid, and a client that derives the registry size as `len(models) + len(deferred)` is the
    same defect deferred by one line. So: `published_count == len(/api/models)`, and
    `registry_count == published_count + deferred_count`, both read from the endpoints.
    """
    from workbench.backend.registry import load_registry

    publication = served_models.models_publication()
    served = served_models.models()
    registry = load_registry()

    published_ids = {model["model_id"] for model in served}
    assert publication["published_count"] == len(served), publication
    assert set(publication["published_model_ids"]) == published_ids
    assert publication["registry_count"] == len(registry["models"])
    assert publication["registry_count"] == publication["published_count"] + publication["deferred_count"]
    # And the registry count is the whole document, not the sum the client would have computed.
    assert publication["registry_count"] == len(
        {model["model_id"] for model in registry["models"]})
    assert publication["deferred"] == {"pair": publication["deferred"]["pair"]}


def test_deferred_count_is_derived_from_the_flag_not_from_a_constant() -> None:
    """Adding a deferral to the registry must move every number with no code change."""
    import copy

    from workbench.backend.registry import load_registry, web_deferred_models

    registry = copy.deepcopy(load_registry())
    assert len(web_deferred_models(registry)) == 1
    second = next(model for model in registry["models"]
                  if model["model_id"] != "pair" and "web_publication" not in model)
    second["web_publication"] = {"status": "DEFERRED", "reason": "test-only second deferral"}
    deferred = web_deferred_models(registry)
    assert set(deferred) == {"pair", second["model_id"]}, deferred
    assert len(registry["models"]) - len(deferred) == 15
    # The served document itself still holds every entry: deferring is not deleting.
    assert len(registry["models"]) == 17


# --------------------------------------------------------------------------- the queue refusal


def test_queue_refuses_a_pair_job_by_name_not_with_a_404() -> None:
    from fastapi import HTTPException

    from workbench.backend import main

    with pytest.raises(HTTPException) as raised:
        main.queue_job(main.JobInput(model_id="pair", checkpoint_id="pair_b1",
                                     protocol_id="fashioniq_val_split", top_k=200))
    error = raised.value
    assert error.status_code == 409, error.detail
    detail = error.detail
    assert detail["error"] == "model_deferred_from_web", detail
    assert "defer" in detail["message"].lower(), detail
    assert "pair" in json.dumps(detail).lower()
    assert detail["details"]["model_id"] == "pair"
    assert detail["details"]["offline"]["command"], (
        "a refusal that names no offline path is a dead end")
    # The offered commands must be PASTEABLE: `--model` and `--list` are mutually exclusive in the
    # real CLI, so a refusal that glued them together named a command that fails when run.
    offline = detail["details"]["offline"]
    assert "--list" not in offline["command"], offline
    assert offline["list_command"].endswith("--list"), offline


def test_a_published_model_is_not_refused_by_the_deferral_guard() -> None:
    """The guard is keyed by the registry flag, not by "the first model", and not by a literal."""
    from fastapi import HTTPException

    from workbench.backend import main

    with pytest.raises(HTTPException) as raised:
        main.queue_job(main.JobInput(model_id="limn", checkpoint_id="limn_bundle",
                                     protocol_id="circus_fashioniq_release_v1", top_k=200))
    assert raised.value.detail["error"] != "model_deferred_from_web", raised.value.detail


# --------------------------------------------------------------------------- still retrievable


def test_measurement_is_still_retrievable_by_its_own_id(served_models) -> None:
    """The audit path: unlisted, not unpublished. Each row keeps its own provenance."""
    main = served_models
    for measurement_id in PAIR_EXPLORATORY:
        row = main.native_measurement(measurement_id)
        assert row["measurement_id"] == measurement_id
        assert row["mapping_status"] == "EXPLORATORY_PUBLIC_MAPPING", row
        assert row["model_id"] == "pair"
    blocker = main.native_measurement(PAIR_BLOCKER)
    assert blocker["blocker"] == "BLOCKED_SOURCE_CLASSES_UNPUBLISHED", blocker
    assert blocker["metrics"] is None


def test_an_unknown_measurement_id_is_a_named_404(served_models) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        served_models.native_measurement("0000000000000000")
    assert raised.value.status_code == 404
    assert raised.value.detail["error"] == "native_measurement_not_found"


def test_the_offline_evidence_file_still_backs_the_exploratory_rows() -> None:
    """Unpublishing a row must not remove the artifact it was read from."""
    from workbench.backend.native_measurements import list_native_measurements

    rows = {row["measurement_id"]: row for row in list_native_measurements(ARTIFACTS)["measurements"]}
    for measurement_id in PAIR_EXPLORATORY:
        source = ARTIFACTS / rows[measurement_id]["source"]["path"]
        assert source.is_file(), source
        document = json.loads(source.read_text())
        assert document["model"] == "pair" and document["macro"]["macro_r10"] == pytest.approx(
            rows[measurement_id]["metrics"]["r10"])


# --------------------------------------------------------------------------- the frontend hint


def test_frontend_renders_an_explicit_notice_for_a_deferred_deep_link() -> None:
    """`?view=native&model=pair` used to highlight rows; now it must say why they are gone."""
    import test_frontend_support as support

    module = support.compile_module("scope.ts", Path("/tmp/cir_frontend_pair_deferral"))
    out = support.run_json(f"""
import * as S from {json.dumps(str(module))};
const deferred = {{ pair: "checkpoint mapping chưa xác minh" }};
const pair = S.webDeferralNotice("pair", deferred);
const limn = S.webDeferralNotice("limn", deferred);
const none = S.webDeferralNotice(null, deferred);
console.log('__R__' + JSON.stringify({{pair, limn, none}}));
""")
    assert out["pair"], "no notice for a deferred model deep link"
    assert "checkpoint mapping chưa xác minh" in out["pair"], out["pair"]
    assert "id" in out["pair"], "the notice must say the record is still retrievable by id"
    assert out["limn"] is None and out["none"] is None
