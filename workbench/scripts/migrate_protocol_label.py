"""Rewrite a run artifact's declared protocol label, keeping the original value and metrics.

Why this exists: a run's protocol label is provenance, and a wrong label silently changes WHICH
cohort a number is reported under. Correcting it must be a recorded, reversible migration, not an
edit: the artifact already evidences its own cohort through gallery counts, the source line that
built the gallery, and the evaluator function that ranked it. This tool refuses when those three
disagree with the requested label, so a label can only be fixed where the bytes already say the
truth.

What it writes:
  * the artifact with ``protocol``/``protocol_id`` set to the new id (and, when present,
    ``protocol_source`` untouched -- it cites source, and source does not change);
  * a ``<artifact>.migration.json`` sidecar recording from/to, the original bytes' digest, the
    new bytes' digest, the declared evidence that authorized it, and the untouched metric fields;
  * ``migrated_from`` in the artifact, so history survives inside the migrated file too.

Nothing is re-measured: recall/gallery/query fields are copied byte-for-byte from the original
document, and the tool asserts that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#: Fields that describe the measurement itself. A migration may not touch any of them; the tool
#: asserts equality on every one that exists in both documents.
FROZEN_FIELDS = ("aggregate", "categories", "results", "checkpoint", "source", "runtime", "side_effects",
                 "complete_benchmark", "model_id", "protocol_source")
LABEL_FIELDS = ("protocol", "protocol_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(document: dict) -> str | None:
    for name in LABEL_FIELDS:
        value = document.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def migrate(artifact: Path, to_label: str, *, evidence: dict, dry_run: bool) -> dict:
    """Migrate one artifact, refusing when its own evidence contradicts *to_label*."""
    document = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{artifact} is not a JSON object")
    from_label = _label(document)
    if from_label is None:
        raise SystemExit(f"{artifact} declares no protocol label; nothing to migrate")
    if from_label == to_label:
        raise SystemExit(f"{artifact} already declares {to_label}")

    # Authorization: the caller must supply the evidence that decided the correct cohort, and it
    # must be a non-empty citation set -- a bare label is not evidence.
    citations = evidence.get("citations") or []
    if not citations:
        raise SystemExit("refusing: --evidence must carry at least one citation")
    declared_gallery = evidence.get("gallery_counts") or {}
    measured_gallery = {}
    categories = document.get("categories")
    if isinstance(categories, list):
        for row in categories:
            if isinstance(row, dict) and row.get("category") and row.get("gallery_count") is not None:
                measured_gallery[row["category"]] = int(row["gallery_count"])
    if declared_gallery and measured_gallery and declared_gallery != measured_gallery:
        raise SystemExit(
            "refusing: the artifact's own gallery counts contradict the authorized cohort "
            f"(artifact {measured_gallery} vs evidence {declared_gallery})")

    before_digest = sha256(artifact)
    migrated = dict(document)
    for name in LABEL_FIELDS:
        if name in migrated:
            migrated[name] = to_label
    migrated["migrated_from"] = {
        "artifact_sha256": before_digest,
        "protocol_label": from_label,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "migrator": "workbench/scripts/migrate_protocol_label.py",
        "evidence": evidence,
    }

    # The measurement must be untouched, field by field.
    for name in FROZEN_FIELDS:
        if name in document and document[name] != migrated.get(name):
            raise SystemExit(f"refusing: migration would change frozen field {name!r}")

    target = artifact
    sidecar = artifact.with_name(artifact.name + ".migration.json")
    if not dry_run:
        temporary = artifact.with_name(f".{artifact.name}.{__import__('os').getpid()}.tmp")
        temporary.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
        temporary.replace(artifact)
        after_digest = sha256(artifact)
        sidecar.write_text(json.dumps({
            "artifact": str(artifact),
            "from_label": from_label,
            "to_label": to_label,
            "artifact_sha256_before": before_digest,
            "artifact_sha256_after": after_digest,
            "frozen_fields_verified": [name for name in FROZEN_FIELDS if name in document],
            "evidence": evidence,
            "migrated_at": migrated["migrated_from"]["migrated_at"],
        }, indent=2) + "\n", encoding="utf-8")
        return {"artifact": str(artifact), "from": from_label, "to": to_label,
                "sha256_before": before_digest, "sha256_after": after_digest,
                "sidecar": str(sidecar)}
    return {"artifact": str(artifact), "from": from_label, "to": to_label,
            "sha256_before": before_digest, "dry_run": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--to", required=True, help="protocol id the bytes already describe")
    parser.add_argument("--evidence", type=Path, required=True,
                        help="JSON with {citations: [...], gallery_counts: {category: count}}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    print(json.dumps(migrate(args.artifact, args.to, evidence=evidence, dry_run=args.dry_run), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
