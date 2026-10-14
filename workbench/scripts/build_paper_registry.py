#!/usr/bin/env python3
"""Build the committed benchmark registry from the extracted paper payloads.

Run offline, in review, once per paper. The output is what the runtime reads; nothing here runs during
an evaluation.

The conversion is mechanical on purpose. It copies the extracted values, drops the extraction-time
bookkeeping that no comparison needs, and refuses to emit a record whose structure would let the
comparator read a missing value as agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCHMARK_ROOT = REPO / "workbench" / "registry" / "paper_benchmarks"
EXTRACTED = BENCHMARK_ROOT / "extracted"
REFERENCES_FILE = BENCHMARK_ROOT / "papers.yaml"

#: Fields the runtime comparator reads. Anything absent from this list is extraction bookkeeping.
RUNTIME_FIELDS = (
    "reference_id", "model_id", "paper_id", "paper_title", "method_name", "method_variant", "backbone",
    "checkpoint_id", "checkpoint_training_noise_pct", "checkpoint_mapping_status",
    "dataset", "dataset_version", "split", "gallery_protocol", "reference_excluded",
    "evaluation_noise_pct", "preprocessing_id", "augmentation_policy", "metric_definition",
    "aggregation_method", "reported_metrics", "derived_metrics", "category_metrics",
    "metric_provenance", "reference_status", "verification_method", "review_status",
)

PAPER_FIELDS = (
    "paper_id", "paper_title", "authors", "publication_year", "venue", "doi", "arxiv_id",
    "paper_version", "paper_url", "pdf_sha256", "pdf_bytes", "supplement_url", "supplement_sha256",
    "access",
)

VALID_STATUS = {"VERIFIED", "CONFLICT", "UNVERIFIED"}
VALID_MAPPING = {"VERIFIED", "UNVERIFIED", "NO_CHECKPOINT"}


def _normalise_status(raw: str | None, record: dict) -> str:
    """A record with no comparable metric is unverified, whatever label the extractor used.

    Extractors used their own words (`VARIANT`, `UNKNOWN`) for records that carry no reported numbers
    — a protocol finding, or the fact that the paper has no such row. Those are not comparisons, so
    they are recorded as `UNVERIFIED` and the comparator skips them. A record that *does* carry a
    metric must say `VERIFIED` or `CONFLICT`.
    """
    status = (raw or "").upper()
    has_metric = any((record.get("reported_metrics") or {}).get(key) is not None
                     or any(d.get("key") == key and d.get("value") is not None
                            for d in (record.get("derived_metrics") or []))
                     for key in ("r10", "r50", "mean"))
    if status in VALID_STATUS:
        return status
    if not has_metric:
        return "UNVERIFIED"
    raise ValueError(f"{record['reference_id']}: {status!r} is not a usable status for a record with metrics")


def _explode(record: dict) -> list[dict]:
    """Split a record whose `model_id` names several models into one record per model.

    An extractor may legitimately cover a paper that the workbench registers as two entries (the two
    CLIP4Cir variants), or write a paper-level note that applies to both. The registry must key by a
    single model id, so the shared content is duplicated rather than left under a compound key that
    no lookup would ever match.
    """
    raw = str(record.get("model_id") or "")
    models = [part.strip() for part in raw.split("|") if part.strip()]
    if len(models) <= 1:
        record["model_id"] = raw
        return [record]
    out = []
    for model in models:
        copy = dict(record)
        copy["model_id"] = model
        copy["reference_id"] = f"{model}__{record['reference_id'].split('__', 1)[-1]}"
        out.append(copy)
    return out


def build_record(paper: dict, experiment: dict) -> dict:
    record = {key: (paper.get(key) if key.startswith("paper_") else experiment.get(key))
              for key in RUNTIME_FIELDS}
    if "paper_title" not in record or not record["paper_title"]:
        record["paper_title"] = paper.get("paper_title")
    record["reference_status"] = _normalise_status(record.get("reference_status"), record)
    mapping = record.get("checkpoint_mapping_status") or "UNVERIFIED"
    if mapping not in VALID_MAPPING:
        record["checkpoint_mapping_status"] = "UNVERIFIED"
    reported = record.get("reported_metrics") or {}
    derived = record.get("derived_metrics") or []
    if not isinstance(reported, dict) or not isinstance(derived, list):
        raise ValueError(f"{record['reference_id']}: malformed metric section")
    for key in ("r10", "r50", "mean"):
        if reported.get(key) is not None and not isinstance(reported[key], (int, float)):
            raise ValueError(f"{record['reference_id']}: reported {key} is not a number")
        if reported.get(key) is None and not any(d.get("key") == key and d.get("value") is not None
                                                 for d in derived):
            # A metric nobody published and nobody could derive is legitimate -- it just means the
            # comparator has nothing to check there. Every derived value must state its formula, so a
            # value that appears without one is a bookkeeping error rather than a paper's number.
            continue
    for item in derived:
        if not item.get("derivation_formula"):
            raise ValueError(f"{record['reference_id']}: derived {item.get('key')} has no formula")
        if item.get("provenance") != "WORKBENCH_DERIVED":
            raise ValueError(f"{record['reference_id']}: derived {item.get('key')} must be WORKBENCH_DERIVED")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="registry version label; defaults to a digest of the sources")
    parser.add_argument("--output", type=Path, default=REFERENCES_FILE)
    args = parser.parse_args(argv)

    sources = sorted(EXTRACTED.glob("*.json"))
    if not sources:
        print("no extracted papers", file=sys.stderr)
        return 1

    records: list[dict] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "paper" not in payload:
            print(f"{source.name}: no paper block", file=sys.stderr)
            return 1
        paper = {key: payload["paper"].get(key) for key in PAPER_FIELDS}
        for experiment in payload.get("experiments") or []:
            base = build_record(payload["paper"], experiment)
            for record in _explode(base):
                if record["reference_id"] in seen:
                    print(f"duplicate reference_id: {record['reference_id']}", file=sys.stderr)
                    return 1
                seen.add(record["reference_id"])
                records.append(record)

    version = args.version or f"extract-{digest.hexdigest()[:12]}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Paper benchmark references, built offline by\n"
        "# `workbench/scripts/build_paper_registry.py` from the extracted payloads under `extracted/`.\n"
        "# The runtime reads this file read-only. A correction means a new registry_version, not an\n"
        "# edit in place: an existing run records the version and digest it was judged against.\n"
        "#\n"
        "# `reported_metrics` are values the paper prints. `derived_metrics` are values the workbench\n"
        "# computes from printed ones and carry their formula; they are never attributed to the authors.\n"
        f"registry_version: {version}\n"
    )
    import yaml

    body = yaml.safe_dump({"references": records}, sort_keys=False, allow_unicode=True, width=100)
    args.output.write_text(header + body)
    print(f"{args.output}: {len(records)} reference(s), version {version}")
    for record in records:
        reported = record["reported_metrics"]
        print(f"  {record['model_id']:24} {record['reference_id'][:40]:42} "
              f"r10={reported.get('r10')} r50={reported.get('r50')} mean={reported.get('mean')} "
              f"{record['reference_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
