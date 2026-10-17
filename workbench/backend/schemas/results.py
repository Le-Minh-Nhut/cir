from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
            effective_depth = min(self.run.top_k_saved, category_size)
            if len(query.top_results) != effective_depth:
                raise ValueError("query top_results must match effective saved retrieval depth")
            if query.target_rank > category_size:
                raise ValueError("target_rank exceeds gallery_size")
            if query.target_rank <= effective_depth and query.top_results[query.target_rank - 1].image_id != query.target_id:
                raise ValueError("target_rank does not match saved target retrieval")
            if query.target_rank > effective_depth and any(item.image_id == query.target_id for item in query.top_results):
                raise ValueError("saved target retrieval conflicts with target_rank")
        return self


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
