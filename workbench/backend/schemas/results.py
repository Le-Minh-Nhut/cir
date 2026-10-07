from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_LITERATURE_LABELS = {
    "fashioniq_original_split": "original",
    "fashioniq_val_split": "val",
}
PROTOCOL_IDS = frozenset(PROTOCOL_LITERATURE_LABELS)


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
    upstream_repo: str | None = None
    upstream_commit: str | None = None
    checkpoint_path: str | None = None
    checkpoint_source: str | None = None
    checkpoint_sha256: str | None = None
    instrumentation_patch_sha256: str | None = None
    command_digest: str | None = None
    environment_digest: str | None = None
    timestamp: datetime
    data_kind: Literal["experiment", "mock"] = "experiment"
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
        effective_depth = min(self.run.top_k_saved, self.run.gallery_size)
        for query in self.queries:
            if len(query.top_results) != effective_depth:
                raise ValueError("query top_results must match effective saved retrieval depth")
            if query.target_rank > self.run.gallery_size:
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
