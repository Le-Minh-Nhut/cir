"""Committed paper benchmark references, loaded read-only by the runtime gate.

The numbers a run is judged against come from files committed to the repository, produced by an
offline extraction-and-verification stage. Nothing here calls a language model: the runtime loads
structured records and compares numbers.

Two properties matter more than convenience:

- **Immutable per run.** A reference is selected once and its values, plus the digest of the registry
  file it came from, are recorded with the run. Correcting a number later creates a new registry
  version and leaves older runs auditable rather than retroactively re-judged.
- **Refuse to guess.** An `UNVERIFIED` reference, a metric the paper never printed, or a checkpoint
  condition that does not match the paper's row all refuse the comparison instead of producing a
  number that looks like agreement.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Registry layout: one YAML file holding every reference record.
BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "registry" / "paper_benchmarks"
REFERENCES_FILE = BENCHMARK_ROOT / "papers.yaml"

#: Provenance markers. A value the paper prints is `PAPER_REPORTED`; a value the workbench computes
#: from printed ones is `WORKBENCH_DERIVED` and carries its formula. Conflating them would attribute
#: a number to an author who never wrote it.
PAPER_REPORTED = "PAPER_REPORTED"
WORKBENCH_DERIVED = "WORKBENCH_DERIVED"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

#: Reference states. Only `VERIFIED` may be compared against.
REFERENCE_VERIFIED = "VERIFIED"
REFERENCE_CONFLICT = "CONFLICT"
REFERENCE_UNVERIFIED = "UNVERIFIED"

METRIC_KEYS = ("r10", "r50", "mean")


@dataclass(frozen=True)
class BenchmarkReference:
    reference_id: str
    model_id: str
    paper_id: str
    paper_title: str
    payload: dict[str, Any]
    source_digest: str
    registry_version: str

    @property
    def reported_metrics(self) -> dict[str, float | None]:
        return dict(self.payload.get("reported_metrics") or {})

    @property
    def reference_status(self) -> str:
        return str(self.payload.get("reference_status") or REFERENCE_UNVERIFIED)

    def metric_provenance(self, key: str) -> str:
        """How the value for this metric came to exist, or `UNAVAILABLE`."""
        if self.reported_metrics.get(key) is not None:
            return PAPER_REPORTED
        for derived in self.payload.get("derived_metrics") or []:
            if derived.get("key") == key and derived.get("value") is not None:
                return WORKBENCH_DERIVED
        return PROVENANCE_UNAVAILABLE

    def value_for(self, key: str) -> float | None:
        """The value to compare against, whichever way the paper expressed it."""
        reported = self.reported_metrics.get(key)
        if reported is not None:
            return float(reported)
        for derived in self.payload.get("derived_metrics") or []:
            if derived.get("key") == key and derived.get("value") is not None:
                return float(derived["value"])
        return None

    def provenance_for_all(self) -> dict[str, str]:
        return {key: self.metric_provenance(key) for key in METRIC_KEYS}

    def as_evidence(self) -> dict[str, Any]:
        """What a run record keeps so its verdict can be re-checked years later."""
        return {
            "reference_id": self.reference_id,
            "model_id": self.model_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "registry_version": self.registry_version,
            "registry_file": str(REFERENCES_FILE),
            "registry_digest": self.source_digest,
            "reference_status": self.reference_status,
            # The reference's own record of how confidently its paper row maps to the
            # checkpoint asset: the gate must not certify identity more strongly than the
            # reference itself claims.
            "checkpoint_mapping_status": self.payload.get("checkpoint_mapping_status"),
            "reported_metrics": self.reported_metrics,
            "derived_metrics": self.payload.get("derived_metrics") or [],
            "metric_provenance": self.provenance_for_all(),
            "category_metrics": self.payload.get("category_metrics") or {},
            "checkpoint_training_noise_pct": self.payload.get("checkpoint_training_noise_pct"),
            "gallery_protocol": self.payload.get("gallery_protocol"),
        }


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """The committed benchmark registry, read-only. Raises on a structural problem."""
    target = path or REFERENCES_FILE
    if not target.is_file():
        raise FileNotFoundError(f"paper benchmark registry missing: {target}")
    import yaml

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("references"), list):
        raise ValueError(f"unsupported paper benchmark registry: {target}")
    return data


def registry_version(path: Path | None = None) -> str:
    return str(load_registry(path).get("registry_version") or "unversioned")


def references_for_model(model_id: str, path: Path | None = None) -> list[BenchmarkReference]:
    """Every committed reference for one model, with the registry digest attached."""
    target = path or REFERENCES_FILE
    data = load_registry(target)
    digest = _file_digest(target)
    version = str(data.get("registry_version") or "unversioned")
    out = []
    for record in data["references"]:
        if record.get("model_id") != model_id:
            continue
        out.append(BenchmarkReference(
            reference_id=str(record["reference_id"]),
            model_id=model_id,
            paper_id=str(record.get("paper_id") or ""),
            paper_title=str(record.get("paper_title") or ""),
            payload=record,
            source_digest=digest,
            registry_version=version,
        ))
    return out


def select_reference(
    model_id: str,
    *,
    checkpoint_training_noise_pct: int | None = None,
    protocol_id: str | None = None,
    path: Path | None = None,
) -> tuple[BenchmarkReference | None, str]:
    """Pick the reference describing the experiment about to be run.

    Selection is by *condition*, not by metric: a paper's 0%-noise row cannot be the reference for a
    50%-noise checkpoint no matter how close the numbers happen to be. A `None` result means no
    committed reference describes this run, which the caller reports as an unverified reference
    rather than as a failure of the model.
    """
    candidates = references_for_model(model_id, path)
    if not candidates:
        return None, f"no committed paper reference for {model_id}"

    verified = [c for c in candidates if c.reference_status == REFERENCE_VERIFIED]
    if not verified:
        statuses = sorted({c.reference_status for c in candidates})
        return None, f"{model_id} has references but none is verified: {statuses}"

    def matches(candidate: BenchmarkReference) -> bool:
        record = candidate.payload
        if checkpoint_training_noise_pct is not None:
            recorded = record.get("checkpoint_training_noise_pct")
            if recorded is None or int(recorded) != int(checkpoint_training_noise_pct):
                return False
        if protocol_id is not None:
            recorded = record.get("gallery_protocol")
            if recorded is not None and recorded != protocol_id:
                return False
        return True

    exact = [c for c in verified if matches(c)]
    if len(exact) == 1:
        return exact[0], "exact condition match"
    if len(exact) > 1:
        return None, f"more than one verified reference matches this run: {sorted(c.reference_id for c in exact)}"
    condition = []
    if checkpoint_training_noise_pct is not None:
        condition.append(f"training noise {checkpoint_training_noise_pct}%")
    if protocol_id is not None:
        condition.append(f"protocol {protocol_id}")
    return None, (f"no verified reference for {model_id} under {', '.join(condition) or 'the given conditions'}; "
                  f"committed rows are {sorted(c.reference_id for c in verified)}")
