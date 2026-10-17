from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The ONE owner of the protocol -> literature-label map. Every other module imports this; a second
#: copy drifted silently (`replay/publication.py` had `fashioniq_full_gallery_ref_excluded` as "val"
#: while this file had "original") because each consumer read its own. The schema owns it because it
#: VALIDATES the pairing below, so a disagreement here is not a matter of opinion.
#:
#: These labels are the LITERATURE's words for the split a paper reports. They are not the protocol:
#: two different protocols share the label "original" because both draw on the full validation
#: split, and they differ in whether the reference stays eligible. Executable semantics come from
#: ``protocol_id`` alone, never from this label.
PROTOCOL_LITERATURE_LABELS = {
    "fashioniq_original_split": "original",
    "fashioniq_full_gallery_ref_excluded": "original",
    "fashioniq_val_split": "val",
}
PROTOCOL_IDS = frozenset(PROTOCOL_LITERATURE_LABELS)

#: The declared origin of a run's per-query results. ``experiment`` is a real evaluation;
#: ``mock`` is a deterministic development fixture (``tests/mock_data.py``); ``synthetic`` is a
#: registry-declared synthetic source (``models.yaml`` ``synthetic_source``), which the pipeline
#: records as its own kind (``evaluate_models.py:185``). All three are produced by the system, so
#: all three must be representable; being representable says nothing about being evidence. The
#: paper gate refuses anything but ``experiment`` (``paper_gate.py:311-318``) and the verifier
#: refuses it too (``verify_run.py:222-226``).
DATA_KINDS = ("experiment", "mock", "synthetic")

#: The FashionIQ categories a complete benchmark covers. Named here so the admission policy, the
#: observer and the docs cannot disagree about what "complete" means.
FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")

#: The protocols whose reference policy removes the reference as a candidate. Named once so the
#: eligible-bound rule and the protocol definitions cannot drift apart.
REFERENCE_EXCLUDED_PROTOCOLS = frozenset({
    "fashioniq_full_gallery_ref_excluded",
    "fashioniq_val_split",
})


class ReportedMetrics(BaseModel):
    r10: float | None = None
    r50: float | None = None
    mean: float | None = None


class TopResult(BaseModel):
    rank: Annotated[int, Field(ge=1)]
    image_id: Annotated[str, Field(min_length=1)]
    score: float


class QueryResult(BaseModel):
    query_id: Annotated[str, Field(min_length=1)]
    category: Literal["dress", "shirt", "toptee"]
    annotation_index: Annotated[int, Field(ge=0)]
    reference_id: Annotated[str, Field(min_length=1)]
    target_id: Annotated[str, Field(min_length=1)]
    raw_captions: Annotated[list[str], Field(min_length=2, max_length=2)]
    model_input_text: str
    target_rank: Annotated[int, Field(ge=1)]
    top_results: list[TopResult]

    @model_validator(mode="after")
    def validate_query(self) -> "QueryResult":
        if [item.rank for item in self.top_results] != list(range(1, len(self.top_results) + 1)):
            raise ValueError("top_results ranks must be contiguous from 1")
        if len({item.image_id for item in self.top_results}) != len(self.top_results):
            raise ValueError("top_results image IDs must be unique")
        expected_query_id = f"{self.category}:{self.annotation_index}:{self.reference_id}:{self.target_id}"
        if self.query_id != expected_query_id:
            raise ValueError("query_id must use canonical category:index:reference:target identity")
        return self

class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[str, Field(min_length=1)]
    dataset: Literal["FashionIQ"]
    split: Literal["val"]
    protocol_id: str
    literature_split_label: Literal["original", "val"]
    evaluation_noise_pct: Literal[0]
    model_id: str
    method_name: str
    checkpoint_id: Annotated[str, Field(min_length=1)]
    checkpoint_training_noise_pct: int | None = Field(default=None, ge=0, le=100)
    top_k_saved: Annotated[int, Field(ge=1)]
    gallery_size: Annotated[int, Field(ge=1)]
    #: Per-category raw gallery sizes. FashionIQ categories have genuinely different gallery
    #: sizes, and a single scalar cannot express a category-specific bound — a dress toptee-sized
    #: rank would pass a global check it should fail. ``gallery_size`` remains the declared upper
    #: bound (the largest category) so schema-v2 artifacts written before this field stay valid and
    #: keep their meaning; when present this mapping is authoritative per category.
    category_gallery_sizes: dict[str, Annotated[int, Field(ge=1)]] | None = None
    #: Per-category eligible candidate count after the protocol's reference policy is applied.
    #: Under an excluded protocol this is ``size - 1``; under Protocol A the reference stays
    #: eligible and it equals ``size``. Present so the two are never conflated.
    category_eligible_counts: dict[str, Annotated[int, Field(ge=1)]] | None = None
    upstream_repo: str | None = None
    upstream_commit: str | None = None
    checkpoint_path: str | None = None
    checkpoint_source: str | None = None
    checkpoint_sha256: str | None = None
    instrumentation_patch_sha256: str | None = None
    command_digest: str | None = None
    environment_digest: str | None = None
    #: The declared runtime adaptation the run executed under (e.g. ``sm_120``), from the
    #: registry's ``adapted_runtime`` block. ``None`` means the historical environment ran
    #: unchanged. Present so an adapted run is never exported as if it were an exact
    #: reproduction; the variant labelling elsewhere keys off this value.
    runtime_adaptation: str | None = None
    #: The declared checkpoint BUNDLE this run evaluated, as a digest over every member's
    #: ``filename`` + content digest (``evaluate_models.manifest_digest``). A single scalar
    #: ``checkpoint_sha256`` cannot identify LIMN's three independently trained category models: it
    #: would be the dress weights while shirt and toptee went unrecorded, so a swapped or missing
    #: category member would not change any identity. The digest changes if ANY member changes.
    bundle_manifest_digest: str | None = None
    #: The bundle members actually evaluated, keyed by name: ``{name: {filename, sha256, ...}}``.
    #: Recorded so a reader can see WHICH member was which category, not merely that a bundle existed.
    checkpoint_bundle: dict[str, dict] | None = None
    #: The category-checkpoint mapping for a run whose categories use distinct weights, e.g.
    #: ``{"dress": "base_iter0_dress", ...}``. Present so a per-category substitution is visible.
    category_checkpoint_ids: dict[str, str] | None = None
    #: The per-query export's own integrity: the digest of the canonical artifact and whether the
    #: observation reproduced the native ranking. Absent on aggregate-only runs.
    per_query_artifact_digest: str | None = None
    instrumentation_parity: str | None = None
    timestamp: datetime
    data_kind: Literal["experiment", "mock", "synthetic"] = "experiment"
    reported_paper_metrics: ReportedMetrics = Field(default_factory=ReportedMetrics)
    reproduced_metrics: ReportedMetrics = Field(default_factory=ReportedMetrics)

    @model_validator(mode="after")
    def validate_metadata(self) -> "RunMetadata":
        expected_label = PROTOCOL_LITERATURE_LABELS.get(self.protocol_id)
        if expected_label is None:
            raise ValueError(f"unsupported protocol_id: {self.protocol_id}")
        if self.literature_split_label != expected_label:
            raise ValueError("literature_split_label does not match protocol_id")
        return self

class ResultRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    run: RunMetadata
    queries: list[QueryResult]

    @model_validator(mode="after")
    def validate_queries(self) -> "ResultRun":
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("query_id values must be unique per run")
        sizes = self.run.category_gallery_sizes
        if sizes is not None:
            unknown = sorted({query.category for query in self.queries} - set(sizes))
            if unknown:
                raise ValueError(f"category_gallery_sizes is missing categories present in queries: {unknown}")
            # The scalar must remain the declared upper bound, or the two disagree about what the
            # run's gallery actually was.
            if max(sizes.values()) != self.run.gallery_size:
                raise ValueError("gallery_size must equal the largest category_gallery_sizes entry")
        for query in self.queries:
            # A category-specific bound when one is declared, otherwise the historical scalar. A
            # dress rank of 300 is not excused by a larger toptee gallery.
            category_size = sizes[query.category] if sizes is not None else self.run.gallery_size
            # The saved depth is bounded by the ELIGIBLE candidate count, not the raw gallery: under
            # an excluded protocol the masked reference is not a retrievable candidate, so an
            # artifact with `gallery_size <= top_k_saved` correctly stores one fewer result than the
            # raw size implies. Using the raw size here rejected exactly that artifact (P2).
            declared_eligible = (self.run.category_eligible_counts or {}).get(query.category)
            if declared_eligible is not None:
                depth_bound = min(category_size, declared_eligible)
            elif self.run.protocol_id in REFERENCE_EXCLUDED_PROTOCOLS:
                depth_bound = category_size - 1
            else:
                depth_bound = category_size
            effective_depth = min(self.run.top_k_saved, max(depth_bound, 0))
            if len(query.top_results) != effective_depth:
                raise ValueError(
                    f"query top_results must match the effective saved retrieval depth "
                    f"({effective_depth}); stored {len(query.top_results)}")
            # The ELIGIBLE bound when the protocol excludes the reference: under an excluded
            # protocol the reference still occupies a rank slot in the native matrix, so the true
            # candidate count is size-1 and the last slot is NOT retrievable. Checking only the raw
            # gallery size accepted a phantom rank equal to the masked reference's position (F-15).
            eligible = declared_eligible
            if eligible is not None:
                bound = min(category_size, eligible)
            elif self.run.protocol_id in REFERENCE_EXCLUDED_PROTOCOLS:
                # An excluded protocol has no retrievable slot for the reference, and the reference is
                # always one candidate, so the eligible count is AT MOST size-1. Legacy artifacts that
                # predate the field are bounded by that conservative value rather than by the raw
                # gallery size, which would have admitted a rank equal to the masked slot (found in
                # review). The field is still what a new artifact should record, and it is required
                # by `admission` for an experiment.
                bound = category_size - 1
            else:
                bound = category_size
            if query.target_rank > bound:
                raise ValueError(
                    f"target_rank exceeds the eligible candidate count for {query.category} ({bound})")
            if query.target_rank <= effective_depth and query.top_results[query.target_rank - 1].image_id != query.target_id:
                raise ValueError("target_rank does not match saved target retrieval")
            if query.target_rank > effective_depth and any(item.image_id == query.target_id for item in query.top_results):
                raise ValueError("saved target retrieval conflicts with target_rank")
        return self


#: The admission states a per-query artifact can be in, in decreasing trust. Derived by
#: :func:`admission`, never by a consumer reading `data_kind`: `data_kind` is a DECLARATION the
#: producer writes about itself, and a schema-valid JSON file must not certify its own authenticity.
PER_QUERY_UNVERIFIED = "PER_QUERY_UNVERIFIED"
PER_QUERY_PARITY_PENDING = "PER_QUERY_PARITY_PENDING"
PER_QUERY_PARITY_FAILED = "PER_QUERY_PARITY_FAILED"
PER_QUERY_VERIFIED = "PER_QUERY_VERIFIED"
SYNTHETIC_OR_MOCK = "SYNTHETIC_OR_MOCK"

#: The states `instrumentation_parity` may hold. Named as a closed set because the promotion filter
#: below must distinguish "this artifact is only waiting to be checked" from a defect -- and a free
#: string let an artifact choose its own reason text and slip through a substring test.
INSTRUMENTATION_PARITY_STATES = frozenset({
    "INSTRUMENTATION_PARITY_VERIFIED",
    "INSTRUMENTATION_PARITY_MISMATCH",
})

#: The exact reason that means "no defect; a verifier has simply not spoken yet". Compared by
#: EQUALITY, never as a substring: an artifact controls its own field values, so any substring test
#: over its text can be satisfied by the artifact itself.
AWAITING_CORROBORATION = ("awaiting corroboration by the authoritative verifier "
                          "(evidence ledger + completion proof)")

#: What an experiment artifact must RECORD before any layer may call it verified evidence. Named
#: here once so the exporter, the validator, the index and the API cannot drift apart (F-13).
EXPERIMENT_REQUIRED_PROVENANCE = ("command_digest", "environment_digest", "bundle_manifest_digest")

def admission(result: "ResultRun") -> tuple[str, list[str]]:
    """The ONE admission policy: (state, reasons). Every entry point calls this.

    The distinction this encodes is the whole point of the F-07 finding. A run's ``data_kind`` is
    what its producer DECLARED. Whether it is *verified evidence* is a separate question answered by
    the authoritative verifier (``scripts/verify_run.py``), which checks the evidence ledger, the
    completion proof and metric re-extraction -- none of which is visible in the artifact itself.

    This function therefore answers only what the artifact's OWN contents can support, and it is
    deliberately conservative: a real experiment whose own recorded provenance is incomplete is
    ``PER_QUERY_UNVERIFIED``, because the pieces its verification compares are absent. It never
    returns ``PER_QUERY_VERIFIED``: that state requires the verifier's corroboration, so the caller
    that HAS a verification result is the only one allowed to promote to it (see ``promote``).
    """
    run = result.run
    if run.data_kind == "mock":
        # A mock is a DECLARED development fixture: structurally valid, used to build the UI, and
        # never evidence. It is the only kind admitted without further question.
        return SYNTHETIC_OR_MOCK, [f"data_kind is {run.data_kind!r}: development data, never evidence"]
    if run.data_kind != "experiment":
        # `synthetic` is a registry-declared pipeline self-test whose evaluator prints its own
        # aggregate. It must be REPRESENTABLE (the pipeline produces it) but it is not exempt: the
        # verifier refuses it on its own declaration, and a strict validator must refuse it too
        # rather than treating every non-experiment kind as a development fixture.
        return PER_QUERY_UNVERIFIED, [
            f"data_kind is {run.data_kind!r}: a declared self-test, not evidence about reproduction"]

    reasons: list[str] = []
    # A full FashionIQ benchmark covers all three categories. A one-category diagnostic is a
    # legitimate thing to run, but it must never be admitted as the complete benchmark (the
    # gateway `PC_PILOT_HANDOFF` conditions name this explicitly). Checked on the admission policy
    # rather than in the schema so a development fixture may still be partial.
    present = {query.category for query in result.queries}
    if present != set(FASHIONIQ_CATEGORIES):
        missing_categories = sorted(set(FASHIONIQ_CATEGORIES) - present)
        reasons.append(
            f"an experiment must cover all three FashionIQ categories; missing: {missing_categories}")
    # LIMN's three category models are independently trained: one checkpoint must never serve two
    # categories, and every category present must be mapped, or a substituted weight would be
    # invisible in the recorded provenance (F-12).
    if run.category_checkpoint_ids is not None:
        values = list(run.category_checkpoint_ids.values())
        if len(set(values)) != len(values):
            reasons.append("category_checkpoint_ids reuses one checkpoint for more than one "
                           "category; the categories are independently trained models")
        unmapped = sorted({query.category for query in result.queries}
                          - set(run.category_checkpoint_ids))
        if unmapped:
            reasons.append(f"category_checkpoint_ids is missing categories present in queries: {unmapped}")
    missing = [field for field in EXPERIMENT_REQUIRED_PROVENANCE if not getattr(run, field)]
    if not run.checkpoint_sha256 and not run.bundle_manifest_digest:
        missing.append("checkpoint_sha256 or bundle_manifest_digest")
    if missing:
        reasons.append("the artifact does not record the provenance its verification needs: "
                       + ", ".join(missing))
    if run.instrumentation_parity == "INSTRUMENTATION_PARITY_MISMATCH":
        reasons.append("instrumentation parity failed: the observed ranking does not reproduce the "
                       "native hit counts")
        return PER_QUERY_PARITY_FAILED, reasons
    if run.instrumentation_parity is None:
        reasons.append("instrumentation parity has not been recorded for this artifact")
        return PER_QUERY_PARITY_PENDING, reasons
    if run.instrumentation_parity not in INSTRUMENTATION_PARITY_STATES:
        # An unrecognized state is a DEFECT, not a passthrough: the closed set above is what the
        # promotion filter relies on, and a free string here was exploitable.
        reasons.append("unrecognized instrumentation parity state "
                       f"{run.instrumentation_parity!r}; expected one of "
                       f"{sorted(INSTRUMENTATION_PARITY_STATES)}")
        return PER_QUERY_UNVERIFIED, reasons
    if reasons:
        # Provenance incomplete: the artifact cannot be promoted even by a verifier, because the
        # digests that tie it to an invocation are absent.
        return PER_QUERY_UNVERIFIED, reasons
    # Everything the artifact can prove about itself is present. That is still not verification:
    # the verifier must corroborate it against the ledger and the completion proof.
    return PER_QUERY_UNVERIFIED, [AWAITING_CORROBORATION]

def promote(result: "ResultRun", state: str, verifier_verdict: dict | None,
            artifact_digest: str | None = None) -> tuple[str, list[str]]:
    """Apply the authoritative verifier's verdict to an admission state.

    Only the verifier can move a run to ``PER_QUERY_VERIFIED``, and only for the artifact it actually
    spoke about. A positive verdict about a run_id is NOT enough on its own: an attacker who knows a
    verified run's id could write a per-query artifact claiming it, and a run_id-only check would
    promote fabricated rankings. The verdict must therefore name THIS artifact -- by digest, model and
    checkpoint -- or the promotion is refused.

    ``verifier_verdict`` is the receipt the API already serves: it carries the aggregate report's
    identity and, when the run exported per-query results, the digest of that artifact as recorded in
    the HMAC-ledgered report.
    """
    if state != PER_QUERY_UNVERIFIED:
        return state, []
    # A receipt about the bytes is not enough if the artifact is WRONG about itself: a partial
    # category set, a reused checkpoint, a missing bundle digest. The verifier cannot see those -- it
    # judges the aggregate report -- so they are re-checked here. Without this, a dress-only
    # experiment with a matching digest became PER_QUERY_VERIFIED (found in review).
    blocking = [reason for reason in admission(result)[1] if reason != AWAITING_CORROBORATION]
    if blocking:
        return state, blocking
    if not verifier_verdict:
        return state, []
    reasons = list(verifier_verdict.get("evidence_reasons") or [])
    if not all(verifier_verdict.get(key) is True for key in ("certified", "is_research_evidence")):
        return state, reasons

    # The verdict must identify THIS artifact. `per_query_artifact_digest` is computed by the parent
    # evaluator over the bytes the observer wrote and carried in the report the ledger signs, so a
    # match means the corroborated report pointed at exactly these bytes.
    recorded = verifier_verdict.get("per_query_artifact_digest")
    # The digest MUST be supplied by the reader from the bytes on disk. Falling back to a field the
    # artifact writes about itself would let a forged file nominate its own digest and pass -- the
    # exact self-certification this whole mechanism exists to prevent (found in review: the fallback
    # was added to make a test pass, and it reopened the hole).
    claimed = artifact_digest
    if not recorded or not claimed:
        return state, ["the verifier's receipt does not bind a per-query artifact digest, so this "
                       "artifact cannot be attributed to the verified run"]
    if recorded != claimed:
        return state, [f"the artifact digest {claimed[:16]}... is not the digest the verified run "
                       f"recorded ({recorded[:16]}...); this artifact is not the one that was checked"]
    # ...and both must agree on WHICH run they describe.
    for field in ("model_id", "checkpoint_id"):
        expected = verifier_verdict.get(field)
        if expected is not None and getattr(result.run, field) != expected:
            return state, [f"the verifier's receipt describes {field}={expected!r}, not "
                           f"{getattr(result.run, field)!r}"]
    return PER_QUERY_VERIFIED, reasons

class Annotation(BaseModel):
    query_id: str
    protocol_id: str
    annotation_scope: Literal["dataset_query"] = "dataset_query"
    labels: list[str]
    note: str = ""
    updated_at: datetime


class SavedCohort(BaseModel):
    cohort_id: str
    protocol_id: str
    query_ids: list[str]
    run_ids: list[str]
    definition: dict
    notes: str = ""
    created_at: datetime
