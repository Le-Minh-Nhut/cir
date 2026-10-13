"""Checkpoint selection must resolve from registry facts and never guess."""
from __future__ import annotations

import pytest
import yaml

from workbench.backend.checkpoint_selection import (INCLUDE_FLAGGED_CONDITIONS, STRICT_CLEAN,
                                                    select_checkpoints)
from workbench.backend.registry import load_registry


@pytest.fixture(scope="module")
def models():
    return {model["model_id"]: model for model in load_registry()["models"]}


def variant(checkpoint_id: str, *, noise=None, mapping="VERIFIED_METADATA", url="https://example/x.pt"):
    return {"checkpoint_id": checkpoint_id, "filename": f"{checkpoint_id}.pt",
            "download_url": url, "checkpoint_training_noise_pct": noise,
            "checkpoint_mapping_status": mapping}


def synthetic_model(model_id="synthetic", variants=(), bundles=()):
    return {"model_id": model_id, "checkpoint_variants": list(variants),
            "checkpoint_bundles": list(bundles)}


def test_a_recording_noise_condition_is_never_called_clean() -> None:
    """The registry records 50%: a clean policy must refuse rather than assume."""
    subject = synthetic_model(variants=[variant("noisy", noise=50)])
    selection = select_checkpoints(subject, policy=STRICT_CLEAN)
    assert selection.status == "BLOCKED"
    assert selection.reason_code == "BLOCKED_CHECKPOINT"
    assert "50%" in selection.reason
    assert selection.strict_clean_checkpoint_available is False


def test_an_unrecorded_condition_is_not_assumed_clean() -> None:
    subject = synthetic_model(variants=[variant("unknown_condition", noise=None)])
    selection = select_checkpoints(subject, policy=STRICT_CLEAN)
    assert selection.status == "BLOCKED"
    assert "records a training condition" in selection.reason


def test_a_single_clean_recorded_variant_is_selected() -> None:
    selection = select_checkpoints(synthetic_model(variants=[variant("clean", noise=0)]), policy=STRICT_CLEAN)
    assert selection.status == "SELECTED"
    assert selection.checkpoint_ids == ("clean",)
    assert selection.strict_clean_checkpoint_available is True


def test_two_clean_variants_without_a_bundle_are_ambiguous() -> None:
    subject = synthetic_model(variants=[variant("a", noise=0), variant("b", noise=0)])
    selection = select_checkpoints(subject, policy=STRICT_CLEAN)
    assert selection.status == "BLOCKED"
    assert selection.reason_code == "BLOCKED_CHECKPOINT_SELECTION"
    assert "a" in selection.reason and "b" in selection.reason


def test_a_clean_bundle_is_selected_as_one_experiment() -> None:
    subject = synthetic_model(variants=[variant("dress", noise=0), variant("shirt", noise=0), variant("toptee", noise=0)],
                    bundles=[{"bundle_id": "all_categories", "mapping_status": "VERIFIED_METADATA",
                              "required_checkpoint_ids": ["dress", "shirt", "toptee"],
                              "structure": "per_category", "notes": "categories evaluated together"}])
    selection = select_checkpoints(subject, policy=STRICT_CLEAN)
    assert selection.status == "SELECTED"
    assert selection.bundle_id == "all_categories"
    assert selection.checkpoint_ids == ("dress", "shirt", "toptee")


def test_an_unverified_mapping_is_named_as_the_blocker() -> None:
    subject = synthetic_model(variants=[variant("pair_b1", noise=None, mapping="UNVERIFIED"),
                              variant("pair_b2", noise=None, mapping="UNVERIFIED")])
    strict = select_checkpoints(subject, policy=STRICT_CLEAN)
    flagged = select_checkpoints(subject, policy=INCLUDE_FLAGGED_CONDITIONS)
    assert strict.reason_code == "BLOCKED_CHECKPOINT"
    assert flagged.reason_code == "BLOCKED_CHECKPOINT"
    assert "mapping unverified" in flagged.reason


def test_an_explicit_request_wins_but_records_the_override() -> None:
    """A human-resolved mapping is honoured without being relabelled clean."""
    subject = synthetic_model(variants=[variant("noisy", noise=50)])
    selection = select_checkpoints(subject, policy=STRICT_CLEAN, requested=["noisy"])
    assert selection.status == "SELECTED"
    assert selection.checkpoint_ids == ("noisy",)
    assert selection.policy_overridden is True
    assert selection.strict_clean_checkpoint_available is False


def test_an_unknown_or_unobtainable_request_is_refused() -> None:
    subject = synthetic_model(variants=[variant("real", noise=0)])
    unknown = select_checkpoints(subject, policy=STRICT_CLEAN, requested=["missing"])
    assert unknown.status == "BLOCKED"
    assert unknown.reason_code == "BLOCKED_CHECKPOINT_SELECTION"
    assert "not registered" in unknown.reason

    unavailable = synthetic_model(variants=[variant("no_source", noise=0, url=None)])
    selection = select_checkpoints(unavailable, policy=STRICT_CLEAN, requested=["no_source"])
    assert selection.status == "BLOCKED"
    assert "obtainable source" in selection.reason


def test_a_model_with_no_variants_is_blocked() -> None:
    selection = select_checkpoints(synthetic_model(), policy=STRICT_CLEAN)
    assert selection.status == "BLOCKED"
    assert "no checkpoint variant is declared" in selection.reason


def test_an_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown checkpoint policy"):
        select_checkpoints(synthetic_model(variants=[variant("clean", noise=0)]), policy="whatever")


# --- the real registry, so a registry edit that breaks the policy is caught here -------------

def test_the_real_registry_is_reconciled_under_both_policies(models) -> None:
    for policy in (STRICT_CLEAN, INCLUDE_FLAGGED_CONDITIONS):
        for model_id, subject in models.items():
            selection = select_checkpoints(subject, policy=policy)
            assert selection.status in {"SELECTED", "BLOCKED"}
            if selection.blocked:
                assert selection.reason, model_id
            else:
                assert selection.checkpoint_ids, model_id


def test_noise_trained_checkpoints_are_never_selected_as_clean(models) -> None:
    """Air-Know, ConeSep, HABIT and INTENT publish only noise-trained weights."""
    for model_id in ("airknow", "conesep", "habit", "intent"):
        selection = select_checkpoints(models[model_id], policy=STRICT_CLEAN)
        assert selection.blocked, model_id
        assert selection.strict_clean_checkpoint_available is False


def test_the_val_split_family_records_no_training_condition(models) -> None:
    """LIMN's three category checkpoints are one verified bundle, but unrecorded conditions."""
    selection = select_checkpoints(models["limn"], policy=INCLUDE_FLAGGED_CONDITIONS)
    assert selection.checkpoint_ids == ("base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee")
    assert selection.bundle_id == "base_iter0_all_categories"

def test_a_verified_mapping_keeps_its_source_state_separate(models) -> None:
    """A verified mapping and a verified URL are different facts and must not be transposed.

    ENCODER's FashionIQ filename and mapping are established by its official README, but its direct
    file URL is unresolved. That pairing (verified mapping, unresolved URL) must stay legal, because
    the operator can still obtain the file from the author's Drive folder and the registry says so.
    """
    variant = models["encoder"]["checkpoint_variants"][0]
    assert variant["checkpoint_mapping_status"] == "VERIFIED_METADATA"
    assert variant["status"] == "URL_UNRESOLVED"
    assert variant["download_url"] is None

def test_an_unverified_mapping_is_refused_even_when_the_bytes_are_in_hand(models) -> None:
    """Provenance is not obtainability: an unverified mapping must not be selected as clean.

    PAIR ships two checkpoints and the upstream repository does not establish which is the FashionIQ
    benchmark variant. Having downloaded them changes nothing, so selection must still refuse them
    rather than evaluate one as if the mapping were known.
    """
    selection = select_checkpoints(models["pair"], policy=STRICT_CLEAN)
    assert selection.blocked
    assert selection.strict_clean_checkpoint_available is False
    assert all("UNVERIFIED" in option for option in selection.evidence["available"])

def test_a_transposed_mapping_and_source_state_is_rejected(tmp_path) -> None:
    """The registry validator must catch the two fields swapped.

    This is the defect that made ENCODER unresolvable: ``checkpoint_mapping_status`` accepted
    ``URL_UNRESOLVED`` as a legal value, so an entry with the fields transposed loaded cleanly and
    the model was silently blocked. Each field now has its own vocabulary, and the two must agree on
    whether the mapping is verified.
    """
    from workbench.backend.registry import load_registry

    good = load_registry()["models"][0]["checkpoint_variants"][0]

    def registry_with(**variant_overrides):
        variant = dict(good)
        variant.update(variant_overrides)
        return {"schema_version": 1, "protocols": {},
                "models": [{"model_id": "transposed", "native_protocol": None, "supported_protocols": [],
                            "source_available": False, "checkpoint_variants": [variant]}]}

    # (a) The exact ENCODER transposition: the source value in the mapping field is no longer a
    # legal mapping state, so it cannot be spelled this way at all.
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(registry_with(checkpoint_mapping_status="URL_UNRESOLVED",
                                                 status="VERIFIED_METADATA")), encoding="utf-8")
    with pytest.raises(ValueError):
        load_registry(path)

    # (b) Both values are individually legal but describe contradictory facts: the source is
    # unverified while the mapping claims to be verified. Only the coherence rule can catch this.
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(registry_with(status="CHECKPOINT_MAPPING_UNVERIFIED",
                                                 checkpoint_mapping_status="VERIFIED_METADATA")),
                    encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        load_registry(path)
    assert "disagree" in str(failure.value)
