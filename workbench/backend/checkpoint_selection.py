"""Source-backed checkpoint selection.

Autopilot must not decide *which* checkpoint a model is evaluated with by guesswork. Several
registry entries declare more than one variant, some declare variants trained under injected
noise, and some declare variants whose benchmark mapping the upstream authors never established.
Selecting the wrong one silently changes the experiment, so this module resolves a selection from
recorded facts only and refuses to guess when the facts do not determine an answer.

Two policies exist, and neither invents anything:

``strict_clean``
    Only a variant whose recorded training-noise condition is exactly 0 is selectable. A variant
    whose condition is unrecorded is *not* assumed to be clean — the project's own rule is that a
    checkpoint is never described as clean without upstream evidence.

``include_flagged_conditions``
    An explicit opt-in for the exploratory secondary cohort. Variants with an unknown or non-zero
    recorded condition become selectable, but each one carries the condition it was recorded with,
    and the model is marked as not eligible for a strict clean comparison. Nothing is relabelled:
    a 20%-noise checkpoint stays a 20%-noise checkpoint.

An explicit per-model request always wins over the policy, because the operator may have resolved a
mapping by hand. It is still validated against the registry, and an unresolvable request is refused
rather than quietly falling back to something else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STRICT_CLEAN = "strict_clean"
INCLUDE_FLAGGED_CONDITIONS = "include_flagged_conditions"
POLICIES = (STRICT_CLEAN, INCLUDE_FLAGGED_CONDITIONS)

#: A variant is usable only when its provenance is established and it is actually obtainable.
_VERIFIED_MAPPING = "VERIFIED_METADATA"


@dataclass(frozen=True)
class CheckpointSelection:
    """The outcome of resolving one model's checkpoint selection."""

    model_id: str
    policy: str
    status: str
    reason_code: str | None = None
    reason: str | None = None
    checkpoint_ids: tuple[str, ...] = ()
    bundle_id: str | None = None
    strict_clean_checkpoint_available: bool = False
    conditions: dict[str, int | None] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    #: True when the selection came from an explicit operator request that the active policy would
    #: have refused. Reports badge it: the checkpoint is source-backed, but its training condition
    #: does not satisfy the current policy and must not be presented as if it did.
    policy_overridden: bool = False

    @property
    def blocked(self) -> bool:
        return self.status == "BLOCKED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "policy": self.policy, "status": self.status,
            "reason_code": self.reason_code, "reason": self.reason,
            "checkpoint_ids": list(self.checkpoint_ids), "bundle_id": self.bundle_id,
            "strict_clean_checkpoint_available": self.strict_clean_checkpoint_available,
            "conditions": dict(self.conditions), "evidence": dict(self.evidence),
            "policy_overridden": self.policy_overridden,
        }


def _condition(value: Any) -> int | None:
    """The recorded training-noise percentage, or None when the registry does not record one."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _obtainable(variant: dict[str, Any], checkpoint_root: Path | None, model_id: str) -> bool:
    """True when the variant is present locally or has an author-linked download URL."""
    if variant.get("download_url"):
        return True
    return _present(variant, checkpoint_root, model_id)


def _present(variant: dict[str, Any], checkpoint_root: Path | None, model_id: str) -> bool:
    """True when the variant's bytes are already on disk and it needs no acquisition."""
    if checkpoint_root is None:
        return False
    from workbench.backend.registry import checkpoint_path

    return checkpoint_path(model_id, variant, checkpoint_root).exists()


def _describe(variant: dict[str, Any]) -> str:
    condition = _condition(variant.get("checkpoint_training_noise_pct"))
    noise = "training condition unrecorded" if condition is None else f"trained with {condition}% noise"
    return f"{variant.get('checkpoint_id')} ({variant.get('checkpoint_mapping_status')}, {noise})"


def _options(model: dict[str, Any]) -> list[str]:
    return [_describe(variant) for variant in model.get("checkpoint_variants") or []]


def select_checkpoints(model: dict[str, Any], *, policy: str = STRICT_CLEAN,
                       requested: list[str] | None = None,
                       checkpoint_root: Path | None = None) -> CheckpointSelection:
    """Resolve a model's checkpoint selection from registry facts alone."""
    model_id = str(model.get("model_id"))
    if policy not in POLICIES:
        raise ValueError(f"unknown checkpoint policy: {policy!r}")
    variants = {variant["checkpoint_id"]: variant for variant in model.get("checkpoint_variants") or []}
    conditions = {key: _condition(variant.get("checkpoint_training_noise_pct"))
                  for key, variant in variants.items()}
    options = _options(model)

    if not variants:
        return CheckpointSelection(model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
                                   "no checkpoint variant is declared in the registry",
                                   conditions=conditions, evidence={"available": options})

    strict_clean = [key for key, variant in variants.items()
                    if _condition(variant.get("checkpoint_training_noise_pct")) == 0
                    and variant.get("checkpoint_mapping_status") == _VERIFIED_MAPPING
                    and _obtainable(variant, checkpoint_root, model_id)]
    strict_clean_available = bool(strict_clean)

    def refused(reason: str) -> CheckpointSelection:
        return CheckpointSelection(model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT_SELECTION", reason,
                                   strict_clean_checkpoint_available=strict_clean_available,
                                   conditions=conditions, evidence={"available": options})

    # An explicit request is the operator's answer; validate it rather than second-guessing it.
    if requested:
        unknown = [key for key in requested if key not in variants]
        if unknown:
            return refused(f"requested checkpoint id(s) are not registered: {', '.join(sorted(unknown))}. "
                           f"Source-backed options: {'; '.join(options)}")
        unusable = [key for key in requested
                    if variants[key].get("checkpoint_mapping_status") != _VERIFIED_MAPPING
                    or not _obtainable(variants[key], checkpoint_root, model_id)]
        if unusable:
            return refused(f"requested checkpoint id(s) lack verified provenance or an obtainable "
                           f"source: {', '.join(sorted(unusable))}")
        # An explicit request is the operator's resolved answer, not an ambiguity: the registry
        # cannot second-guess which variant a human verified against the upstream repository. What
        # it can do is refuse to *present* the result as clean when the requested checkpoint's
        # recorded condition does not satisfy the active policy.
        overridden = policy == STRICT_CLEAN and any(conditions[key] != 0 for key in requested)
        return CheckpointSelection(model_id, policy, "SELECTED", checkpoint_ids=tuple(requested),
                                   strict_clean_checkpoint_available=strict_clean_available,
                                   conditions=conditions, evidence={"available": options},
                                   policy_overridden=overridden)

    bundles = [bundle for bundle in model.get("checkpoint_bundles") or []
               if bundle.get("mapping_status") == _VERIFIED_MAPPING
               and set(bundle.get("required_checkpoint_ids") or []).issubset(variants)]

    if strict_clean:
        # A bundle whose every member is clean is one experiment, not an ambiguity: the upstream
        # authors evaluate the categories together.
        for bundle in bundles:
            required = list(bundle["required_checkpoint_ids"])
            if set(required).issubset(strict_clean):
                return CheckpointSelection(
                    model_id, policy, "SELECTED", checkpoint_ids=tuple(required),
                    bundle_id=bundle["bundle_id"], strict_clean_checkpoint_available=True,
                    conditions=conditions, evidence={"available": options})
        if len(strict_clean) == 1:
            return CheckpointSelection(model_id, policy, "SELECTED", checkpoint_ids=(strict_clean[0],),
                                       strict_clean_checkpoint_available=True, conditions=conditions,
                                       evidence={"available": options})
        # A model may declare which of its clean variants this workbench's workflow is FOR. TME is
        # the case that needs it: it publishes a FashionIQ-trained 0% weight and a CIRR-trained 0%
        # weight, both clean, and without a declared preference the FashionIQ workflow is blocked as
        # ambiguous. The declaration is not a guess among siblings -- it is a registry fact that the
        # FashionIQ weight (its `checkpoint_training_dataset` matches the benchmark's dataset) is the
        # one a FashionIQ run means, and CIRR weights must never silently become the FashionIQ default.
        preference = model.get("checkpoint_selection_preference")
        if preference is not None:
            if preference not in variants:
                return refused(f"declared checkpoint_selection_preference {preference!r} is not a "
                               f"registered variant. Source-backed options: {'; '.join(options)}")
            if preference not in strict_clean:
                return refused(f"declared checkpoint_selection_preference {preference!r} is not "
                               f"clean-trained and source-backed; options: {'; '.join(options)}")
            return CheckpointSelection(model_id, policy, "SELECTED", checkpoint_ids=(preference,),
                                       strict_clean_checkpoint_available=True, conditions=conditions,
                                       evidence={"available": options})
        return refused("more than one clean-trained variant is registered and none is declared a "
                       f"bundle: {', '.join(sorted(strict_clean))}. Source-backed options: "
                       f"{'; '.join(options)}")

    if policy == INCLUDE_FLAGGED_CONDITIONS:
        flagged = [key for key in variants
                   if variants[key].get("checkpoint_mapping_status") == _VERIFIED_MAPPING
                   and _obtainable(variants[key], checkpoint_root, model_id)]
        for bundle in bundles:
            required = list(bundle["required_checkpoint_ids"])
            if set(required).issubset(flagged):
                return CheckpointSelection(
                    model_id, policy, "SELECTED", checkpoint_ids=tuple(required),
                    bundle_id=bundle["bundle_id"], strict_clean_checkpoint_available=strict_clean_available,
                    conditions=conditions, evidence={"available": options})
        if len(flagged) == 1:
            return CheckpointSelection(model_id, policy, "SELECTED", checkpoint_ids=(flagged[0],),
                                       strict_clean_checkpoint_available=strict_clean_available,
                                       conditions=conditions, evidence={"available": options})
        if len(flagged) > 1:
            return refused("more than one source-backed variant is registered and none is declared a "
                           f"bundle: {', '.join(sorted(flagged))}. Source-backed options: "
                           f"{'; '.join(options)}")
        unverified = [key for key in variants
                      if variants[key].get("checkpoint_mapping_status") != _VERIFIED_MAPPING]
        if unverified:
            return CheckpointSelection(
                model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
                f"checkpoint mapping unverified for {', '.join(sorted(unverified))}; the upstream "
                "repository does not establish which variant is the benchmark checkpoint",
                strict_clean_checkpoint_available=strict_clean_available, conditions=conditions,
                evidence={"available": options})
        return CheckpointSelection(
            model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
            f"no obtainable checkpoint variant: {'; '.join(options)}",
            strict_clean_checkpoint_available=strict_clean_available, conditions=conditions,
            evidence={"available": options})

    # strict_clean with nothing selectable: say exactly why, so the operator can either opt into
    # the flagged cohort deliberately or wait for provenance, rather than guessing.
    reclaimed = [key for key, variant in variants.items()
                 if _condition(variant.get("checkpoint_training_noise_pct")) == 0]
    unverified_mapping = [key for key in reclaimed
                          if variants[key].get("checkpoint_mapping_status") != _VERIFIED_MAPPING]
    if unverified_mapping:
        return CheckpointSelection(
            model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
            "checkpoint mapping unverified for "
            f"{', '.join(sorted(unverified_mapping))}; the upstream repository does not establish "
            "which variant is the benchmark checkpoint",
            strict_clean_checkpoint_available=False, conditions=conditions,
            evidence={"available": options})
    unobtainable = [key for key in reclaimed
                    if variants[key].get("checkpoint_mapping_status") == _VERIFIED_MAPPING
                    and not _obtainable(variants[key], checkpoint_root, model_id)]
    if unobtainable:
        return CheckpointSelection(
            model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
            f"no obtainable checkpoint variant: {', '.join(sorted(unobtainable))} is registered "
            "without an author-linked download URL and is not present locally",
            strict_clean_checkpoint_available=False, conditions=conditions,
            evidence={"available": options})
    recorded = {key: conditions[key] for key in variants if conditions[key] is not None}
    if recorded:
        return CheckpointSelection(
            model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
            "no checkpoint is recorded as clean-trained under policy "
            f"{policy}: {', '.join(f'{key}={value}%' for key, value in sorted(recorded.items()))}. "
            f"Policy {INCLUDE_FLAGGED_CONDITIONS} runs these as an exploratory secondary cohort.",
            strict_clean_checkpoint_available=False, conditions=conditions,
            evidence={"available": options})
    return CheckpointSelection(
        model_id, policy, "BLOCKED", "BLOCKED_CHECKPOINT",
        "no checkpoint variant records a training condition, so none can be called clean under "
        f"policy {policy}. Policy {INCLUDE_FLAGGED_CONDITIONS} runs these with the condition "
        "recorded as unknown.", strict_clean_checkpoint_available=False, conditions=conditions,
        evidence={"available": options})


def select_for_models(models: list[dict[str, Any]], *, policy: str = STRICT_CLEAN,
                      requested: dict[str, list[str]] | None = None,
                      checkpoint_root: Path | None = None) -> dict[str, CheckpointSelection]:
    """Resolve a selection for each model, keeping every model's own outcome independent."""
    requested = requested or {}
    return {str(model["model_id"]): select_checkpoints(
        model, policy=policy, requested=requested.get(str(model["model_id"])),
        checkpoint_root=checkpoint_root) for model in models}
