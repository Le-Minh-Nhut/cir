"""Shared benchmark split loaders, PyTorch DataLoaders, and evaluators for EDIR and PinPoint.

Designed to work across ALL CIR model environments (Python 3.8–3.12) with zero
dependency on duckdb, pyarrow, or pandas at runtime.
"""

from workbench.benchmarks.datasets import (
    CIRCorpusDataset,
    CIRQueryEvalDataset,
    CIRTrainingDataset,
    build_dataloaders,
    collate_corpus,
    collate_eval_queries,
    collate_train_batch,
    seed_worker,
)
from workbench.benchmarks.evaluators import (
    evaluate_edir,
    evaluate_pinpoint,
    rank_embeddings,
)
from workbench.benchmarks.schema import (
    CIRCorpusRecord,
    CIRQueryRecord,
    SplitBundle,
    load_split_bundle,
    load_split_ids,
    verify_split_integrity,
)

__all__ = [
    "CIRCorpusDataset",
    "CIRCorpusRecord",
    "CIRQueryEvalDataset",
    "CIRQueryRecord",
    "CIRTrainingDataset",
    "SplitBundle",
    "build_dataloaders",
    "collate_corpus",
    "collate_eval_queries",
    "collate_train_batch",
    "evaluate_edir",
    "evaluate_pinpoint",
    "load_split_bundle",
    "load_split_ids",
    "rank_embeddings",
    "seed_worker",
    "verify_split_integrity",
]
