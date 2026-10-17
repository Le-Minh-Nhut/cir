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

#: Registry location for the model-level facts the benchmark registry cannot express, because a
#: paper row describes an experiment and this describes a *model*: whether the authors published a
#: comparable number for it at all.
MODEL_REGISTRY_FILE = BENCHMARK_ROOT.parent / "models.yaml"

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

#: Why a run has no reference to be compared against. These are *outcomes about the registry*, not
#: failures of the model: a model whose paper never published the configuration, and an experiment
#: the registry documents twice, both describe a real evaluation that cannot be judged -- for a
#: named reason the operator can act on, and never as though the evidence were forged.
SELECTION_MATCHED = "MATCHED"
#: Every committed row for this model is `UNVERIFIED`: its author published no number for the
#: experiment the run performed, so there is nothing to compare against -- permanently, unless the
#: authors publish or the registry is corrected. Checked *before* condition matching, because a
#: registry with no publishable row cannot publish one for any condition.
SELECTION_NO_PUBLISHED_COMPARISON = "NO_PUBLISHED_COMPARISON"
#: No committed row describes this run's declared condition. Actionable: the registry may be
#: missing a row the paper does print, or the run may not be the experiment it claims to be.
SELECTION_NO_MATCHING_CONDITION = "NO_MATCHING_CONDITION"
#: More than one verified row describes this run's declared condition. The registry does not say
#: which of them is *this* experiment, and picking one would assert an identity nobody established,
#: so the refusal names the ambiguity itself.
SELECTION_AMBIGUOUS = "AMBIGUOUS"
#: No committed row for this model at all.
SELECTION_UNREGISTERED = "UNREGISTERED"

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


@dataclass(frozen=True)
class ReferenceSelection:
    """Why a run has (or has not) a committed reference, as a value rather than prose.

    `reason` is for a human reader and always names the model; `outcome` is what a caller branches
    on, so no consumer has to parse an English sentence to tell "this model has no publishable
    comparison" apart from "the registry has no row for this condition". `references_for_model` +
    `value_for` remain the only way to ask whether any row publishes a metric at all, which is the
    question a caller must answer before writing a sentence about what the report carries.
    """

    reference: BenchmarkReference | None
    outcome: str
    reason: str
    candidates: tuple[BenchmarkReference, ...] = ()
    #: The verified rows that matched the run's declared condition -- one when the run has a
    #: reference, several when the registry is ambiguous, none otherwise. A caller that must say
    #: something about "the rows for this run" reads this, never `candidates`.
    matched: tuple[BenchmarkReference, ...] = ()
    #: `unbenchmarked(model_id, path)`: the authors published no comparable number for this model,
    #: so "no publishable comparison" describes the model and cannot be fixed by this run.
    author_published_no_reference: bool = False


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


def _condition_prose(checkpoint_training_noise_pct: int | None, protocol_id: str | None) -> list[str]:
    described = []
    if checkpoint_training_noise_pct is not None:
        described.append(f"training noise {checkpoint_training_noise_pct}%")
    if protocol_id is not None:
        described.append(f"protocol {protocol_id}")
    return described or ["no declared condition"]


def unbenchmarked(model_id: str, path: Path | None = None) -> bool:
    """Whether this model's authors published no comparable FashionIQ number at all.

    Registry-declared, on the model: `reported_r10`/`reported_r50`/`reported_mean` all absent means
    the model is a reference *variant* -- an author release with no results row in its paper. It is
    the fact that makes `NO_PUBLISHED_COMPARISON` permanent for this model rather than a gap in the
    committed table, and it is the only reason a legitimate run of such a model may be described as
    having nothing to compare instead of as a failed verification.

    Read from the model registry rather than from prose or the model id, so declaring a published
    number (or none) is the entire mechanism. Raises rather than guessing: a model absent from the
    registry is unknown, not unbenchmarked.
    """
    import yaml

    from workbench.backend.registry import model_by_id

    if path is not None:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        model = next((m for m in document.get("models") or [] if m.get("model_id") == model_id),
                     None)
        if model is None:
            raise KeyError(model_id)
    else:
        model = model_by_id(model_id)
    return all(model.get(key) is None
               for key in ("reported_r10", "reported_r50", "reported_mean"))


def select_reference_outcome(
    model_id: str,
    *,
    checkpoint_training_noise_pct: int | None = None,
    protocol_id: str | None = None,
    path: Path | None = None,
) -> ReferenceSelection:
    """Pick the reference describing the experiment about to be run, and say why.

    Selection is by *condition*, not by metric: a paper's 0%-noise row cannot be the reference for a
    50%-noise checkpoint no matter how close the numbers happen to be. In descending order of
    specificity:

    1. no committed row for this model;
    2. committed rows exist, none verified -- the model has no publishable comparison at all, which
       is a permanent property of the paper, not a defect of the run;
    3. exactly one verified row matches the run's declared condition -- the reference;
    4. several verified rows match the same condition -- the registry documents the experiment
       twice without saying which row is this one, so the refusal names the ambiguity and the
       competing rows, and picks none;
    5. none matches the declared condition -- the registry has no row for this experiment.
    """
    candidates = tuple(references_for_model(model_id, path))
    if not candidates:
        # Asked only once there are rows to interpret: a model in neither registry is unknown, and
        # "the authors published nothing" must not be asserted about an id nothing declares.
        return ReferenceSelection(None, SELECTION_UNREGISTERED,
                                  f"{model_id}: no committed paper reference")

    unbenchmarked_model = unbenchmarked(model_id, path)
    verified = tuple(c for c in candidates if c.reference_status == REFERENCE_VERIFIED)
    if not verified:
        statuses = sorted({c.reference_status for c in candidates})
        return ReferenceSelection(
            None, SELECTION_NO_PUBLISHED_COMPARISON,
            f"{model_id}: no publishable comparison -- every committed row for it is "
            f"{statuses}, so no published value describes the experiment this run performed: "
            f"there is nothing to compare a reproduction against, and no value would be reported "
            f"as if there were. Rows: {sorted(c.reference_id for c in candidates)}",
            candidates, author_published_no_reference=unbenchmarked_model)

    def matches(candidate: BenchmarkReference) -> bool:
        record = candidate.payload
        if checkpoint_training_noise_pct is not None:
            recorded = record.get("checkpoint_training_noise_pct")
            if recorded is None or int(recorded) != int(checkpoint_training_noise_pct):
                return False
        if protocol_id is not None:
            recorded = record.get("gallery_protocol")
            if recorded is None:
                # A row whose protocol nobody established does NOT match every protocol. Treating
                # an absent value as a wildcard let a paper that states no gallery construction
                # bind to any run -- an identity asserted by default rather than by evidence. The
                # row is still reported; it simply cannot be chosen as the reference.
                return False
            if recorded != protocol_id:
                return False
        return True

    exact = tuple(c for c in verified if matches(c))
    if len(exact) == 1:
        return ReferenceSelection(exact[0], SELECTION_MATCHED, "exact condition match",
                                  candidates, exact,
                                  author_published_no_reference=unbenchmarked_model)
    if len(exact) > 1:
        ids = sorted(c.reference_id for c in exact)
        return ReferenceSelection(
            None, SELECTION_AMBIGUOUS,
            f"{model_id}: the registry documents this exact experiment more than once -- "
            f"{len(exact)} verified rows match the run's condition ({', '.join(_condition_prose(checkpoint_training_noise_pct, protocol_id))}) "
            f"and disagree on which one this run reproduces: {ids}. They differ in conditions "
            f"the selector does not match on (their method_variant), so no row can be chosen "
            f"without asserting an identity nobody established -- give the rows a selectable "
            f"condition that separates them, or declare which row is this experiment, before "
            f"this run can be judged",
            candidates, exact, author_published_no_reference=unbenchmarked_model)
    unresolved = sorted(row.reference_id for row in unresolved_protocol_rows(verified))
    if unresolved:
        # A row that publishes metrics but declares no protocol must not be reported as a row
        # that does not exist. The refusal has to say what is actually missing: the paper's
        # gallery construction, which nobody has established.
        return ReferenceSelection(
            None, SELECTION_NO_MATCHING_CONDITION,
            (f"{model_id}: no verified reference for it under "
             f"{', '.join(_condition_prose(checkpoint_training_noise_pct, protocol_id))}; the rows "
             f"that do exist for it ({', '.join(unresolved)}) publish metrics but declare no "
             f"protocol, and this paper states no gallery construction or reference policy, so no "
             f"protocol can be matched by evidence and none is matched by default. This stays "
             f"refused until the registry is resolved from pinned source or the paper's own text, "
             f"not once a run is produced"),
            candidates, exact, author_published_no_reference=unbenchmarked_model)
    return ReferenceSelection(
        None, SELECTION_NO_MATCHING_CONDITION,
        (f"{model_id}: no verified reference for it under "
         f"{', '.join(_condition_prose(checkpoint_training_noise_pct, protocol_id))}; "
         f"committed rows are "
         f"{sorted(c.reference_id for c in verified)}"),
        candidates, exact, author_published_no_reference=unbenchmarked_model)

def unresolved_protocol_rows(references: Iterable[BenchmarkReference]) -> list[BenchmarkReference]:
    """Rows that publish metrics but declare no gallery protocol.

    A paper that states no gallery construction or reference policy leaves its rows unusable as a
    comparison target (see ``matches``), but those rows must still be visible: reporting them as
    absent would send the operator looking for a table row that exists.
    """
    return sorted((row for row in references if row.payload.get("gallery_protocol") is None),
                  key=lambda row: row.reference_id)


def select_reference(
    model_id: str,
    *,
    checkpoint_training_noise_pct: int | None = None,
    protocol_id: str | None = None,
    path: Path | None = None,
) -> tuple[BenchmarkReference | None, str]:
    """`select_reference_outcome` as the `(reference, reason)` pair callers already read.

    The prose is the outcome's own reason, so a reader of a report, a readiness row or a gate
    verdict cannot be told something different from what the caller branched on.
    """
    selection = select_reference_outcome(
        model_id, checkpoint_training_noise_pct=checkpoint_training_noise_pct,
        protocol_id=protocol_id, path=path)
    return selection.reference, selection.reason
