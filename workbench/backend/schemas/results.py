from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_IDS = frozenset({"fashioniq_original_split", "fashioniq_val_split"})


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
    def validate_ranks(self) -> "QueryResult":
        ranks = [result.rank for result in self.top_results]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("top_results ranks must be contiguous from 1")
        if len({result.image_id for result in self.top_results}) != len(self.top_results):
            raise ValueError("top_results image IDs must be unique")
        return self


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[str, Field(min_length=1)]
    dataset: Literal["FashionIQ"]
    split: Literal["val"]
    protocol_id: str
    evaluation_noise_pct: Literal[0]
    model_id: str
    method_name: str
    upstream_repo: str | None = None
    upstream_commit: str | None = None
    checkpoint_path: str | None = None
    checkpoint_source: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_training_noise_pct: int | None = Field(default=None, ge=0, le=100)
    instrumentation_patch_sha256: str | None = None
    timestamp: datetime
    data_kind: Literal["experiment", "mock"] = "experiment"
    reported_paper_metrics: ReportedMetrics = Field(default_factory=ReportedMetrics)
    reproduced_metrics: ReportedMetrics = Field(default_factory=ReportedMetrics)

    @model_validator(mode="after")
    def validate_protocol(self) -> "RunMetadata":
        if self.protocol_id not in PROTOCOL_IDS:
            raise ValueError(f"unsupported protocol_id: {self.protocol_id}")
        return self


class ResultRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    run: RunMetadata
    queries: list[QueryResult]

    @model_validator(mode="after")
    def validate_query_ids(self) -> "ResultRun":
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("query_id values must be unique per run")
        return self


class Annotation(BaseModel):
    query_id: str
    labels: list[str]
    note: str = ""
    updated_at: datetime


class SavedCohort(BaseModel):
    cohort_id: str
    protocol_id: str
    query_ids: list[str]
    model_ids: list[str]
    run_ids: list[str]
    definition: dict
    notes: str = ""
    created_at: datetime
