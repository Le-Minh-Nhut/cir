"""Native (non-queue) FashionIQ measurements, served explicitly as UNVERIFIED.

The measurement files this repository actually holds do NOT share one shape. Enumerated from the tree
(not from a guessed schema), a macro triple appears at five different places and two of the files are
roll-ups of other files:

    aggregate.macro_r10        conesep, encoder, habit, hint, intent, limn, tme, airknow, clvc, dcnet,
                              sprc, tgcir            (measurements/ and cohort/measurements/)
    macro.macro_r10            pair_b1_H_normal, pair_b1_H_rev, pair_b2_H_normal, pair_b2_H_rev
    macro_metrics.{r10,r50}    ptha_transfer_run1     (macro 37.8978)
    macro_r10 (top level)      csmcir/measurement.json (57.0507)
    <variant>.macro_r10        clip4cir_parsed.json    (clip4cir_noft 35.1407, clip4cir_fullft 43.8182)

A reader that only accepts `aggregate.*` under two globs reports a subset of the measurements and calls
it the census — the mistake this module now avoids by WALKING each document and recording the path of
the block it found. Two shapes are deliberately not read as per-model measurements:

* `all_measured_summary.json`, `selection_decisions.json`, `clip4cir_parsed.json` and
  `pair_compat/measurements/summary.json` are roll-ups: every block in them is another file's number.
  They are still served (a model whose ONLY number lives in a roll-up, like
  `combiner_rn50x4_noft`, would otherwise vanish) but flagged `rollup: true` and excluded from the
  per-model measurement count.
* `macro_fused_r10` in `dcnet_patched_cli.json` is a *fused-mode* variant of the same run whose
  `aggregate` block already appears in `dcnet_run1.json`; counting it would present one model twice.

Nothing here is certified. These carry no completion proof, no invocation outcome, no evidence-ledger
entry and no verifier receipt, so every row is `verified: false` with a named state, and the endpoint
is separate from `/api/benchmark-runs` (which `/api/evidence/summary` counts).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

#: Served metric names, and the block keys that carry them at each depth.
METRIC_ALIASES = {
    "r10": ("macro_r10", "r10", "macro_metrics_r10"),
    "r50": ("macro_r50", "r50", "macro_metrics_r50"),
    "mean": ("macro_mean", "mean", "macro_metrics_mean"),
}
#: Keys that mark a mapping as a macro block rather than model metadata.
MACRO_MARKERS = ("macro_r10", "macro_r50", "macro_mean", "macro_metrics")
#: Keys under which a block holds bare `r10`/`r50` instead of `macro_r10`/`macro_r50`
#: (`ptha_transfer_run1.json` is `macro_metrics.{r10,r50}`).
MACRO_BLOCK_KEYS = ("aggregate", "macro", "macro_metrics", "metrics")
#: A block reached under one of these keys is a REFERENCE, not a measurement: `dcnet_run1.paper` and
#: `pair_b1_H_normal.paper_reference` are the paper's numbers, and counting them would report the
#: publication as this deployment's result.
EXCLUDED_TOKENS = ("paper", "reported", "reference", "context_only")
#: Directory names inside the selection tree that hold installed code, not measurements.
SKIP_PARTS = ("/downloads/", "/environments/", "/conda-meta/", "/site-packages/")
#: A file whose stem says it is a summary of other files. Detection below is structural; this list
#: only names the four that already exist, so a reader can see what was classified and why.
KNOWN_ROLLUPS = ("all_measured_summary.json", "selection_decisions.json",
                 "clip4cir_parsed.json", "pair_compat/measurements/summary.json")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numbers(block: dict[str, Any]) -> dict[str, float]:
    served = {}
    for name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            value = block.get(alias)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                served[name] = float(value)
                break
    return served


def _blocks(node: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, float]]]:
    """Every macro block in a document, with the key path that reached it.

    Walks mappings and lists alike, because `cohort_summary.json` is a list of per-model objects and
    `aggregate` may itself sit inside a list member.
    """
    if isinstance(node, dict):
        key = trail[-1].lower() if trail else ""
        excluded = any(token in key for token in EXCLUDED_TOKENS)
        numbers = _numbers(node)
        # Bare `r10`/`r50` only count as a measurement when the block was reached under a
        # macro-shaped key. Without that gate `all_measured_summary.json`'s per-category
        # `dress`/`shirt`/`toptee` blocks each read as a macro measurement.
        macro_shaped = not trail or key in MACRO_BLOCK_KEYS or any(m in node for m in MACRO_MARKERS)
        if numbers and not excluded and macro_shaped:
            yield trail, numbers
            # A macro block can also contain per-category detail; do not descend into it, or the
            # same macro would be counted again from a nested copy.
            return
        for key_name, value in node.items():
            yield from _blocks(value, trail + (str(key_name),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _blocks(value, trail + (str(index),))


def _first_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _owner_key(trail: tuple[str, ...]) -> str | None:
    """The nearest enclosing key that is not metric vocabulary: `macro`, `metrics`, `aggregate`…"""
    ignored = {"aggregate", "macro", "macro_metrics", "metrics", "per_category", "per_cat", "results"}
    for part in reversed(trail):
        if part.isdigit() or part.lower() in ignored:
            continue
        return part
    return None


def _categories(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("categories")
    names: list[str] = []
    if isinstance(raw, dict):
        names = [str(key) for key in raw]
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("category"), str):
                names.append(item["category"])
            elif isinstance(item, str):
                names.append(item)
    elif isinstance(payload.get("results"), list):
        names = [str(item.get("category")) for item in payload["results"]
                 if isinstance(item, dict) and item.get("category")]
    return sorted(set(names))


def _rows_for_file(path: Path, root: Path) -> list[dict[str, Any]]:
    """Every measurement block in one file, each with its own provenance and block path."""
    relative = str(path.relative_to(root))
    base = {
        "source": {
            "path": relative,
            "sha256": None,
            "bytes": path.stat().st_size,
            "artifact_type": "native_measurement",
        },
        # Never true here, and stated as data rather than left to a consumer to remember.
        "verified": False,
        "verification": {
            "state": "UNVERIFIED_NATIVE_MEASUREMENT",
            "certified": False,
            "is_research_evidence": False,
            "reason": ("measured outside the job queue, so there is no completion proof, invocation "
                       "outcome, evidence-ledger entry or verifier receipt for it"),
        },
        "unreadable": None,
        "model_id": None,
        "checkpoint_id": None,
        "protocol_id": None,
        "metrics": None,
        "metric_path": None,
        "complete": None,
        "categories": [],
        "recorded_at": None,
        "label": None,
        "blocker": None,
        "rollup": False,
        "rollup_entry": None,
    }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{**base, "measurement_id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                 "unreadable": f"{type(error).__name__}: {error}"}]
    if not isinstance(payload, dict):
        # A JSON array of some other structure (attempts.json) is not a measurement document.
        return []

    blocks = [(trail, numbers) for trail, numbers in _blocks(payload)]
    if not blocks:
        # Not a measurement document. A model/protocol naming an explicit blocker IS worth serving:
        # "why is this model absent" then has an answer sourced from an artifact.
        model = _first_text(payload, "model_id", "model")
        blocker = _first_text(payload, "conclusion", "verdict", "status", "error")
        if not (model and blocker):
            return []
        return [{**base,
                 "measurement_id": hashlib.sha256(f"{relative}#blocker".encode()).hexdigest()[:16],
                 "source": {**base["source"], "sha256": _digest(path)},
                 "model_id": model,
                 "checkpoint_id": _first_text(payload, "checkpoint_id", "variant", "checkpoint_name"),
                 "protocol_id": _first_text(payload, "protocol_id", "protocol"),
                 "blocker": blocker,
                 "categories": _categories(payload)}]

    digest = _digest(path)
    model_hint = _first_text(payload, "model_id", "model", "method_id")
    checkpoint_hint = _first_text(payload, "checkpoint_id", "variant", "checkpoint_name", "label")
    protocol_hint = _first_text(payload, "protocol_id", "protocol")
    # Two or more blocks with DIFFERENT owners in one file is a roll-up by construction: a single
    # measurement cannot be about two models.
    owners = {_owner_key(trail) for trail, _numbers in blocks}
    rollup = len(blocks) > 1 and len(owners - {None}) > 1

    rows: list[dict[str, Any]] = []
    for trail, numbers in blocks:
        owner = _owner_key(trail)
        # A roll-up entry names the MODEL in its key (`clip4cir_noft`, `B1/H_normal`,
        # `combiner_rn50x4_noft`); a direct file names it in `model_id` or in its own stem.
        model = owner if rollup else (model_hint or owner)
        rows.append({
            **base,
            "measurement_id": hashlib.sha256(f"{relative}#{'.'.join(trail)}".encode()).hexdigest()[:16],
            "source": {**base["source"], "sha256": digest},
            "model_id": model,
            "checkpoint_id": (None if rollup else checkpoint_hint) or (
                None if owner and owner == model_hint else owner),
            "protocol_id": protocol_hint,
            "metrics": numbers,
            "metric_path": ".".join(trail) or "(top level)",
            "complete": payload.get("complete_benchmark", payload.get("complete")),
            "categories": _categories(payload),
            "recorded_at": _first_text(payload, "recorded_at", "timestamp"),
            "label": _first_text(payload, "label"),
            "rollup": rollup,
            "rollup_entry": owner if rollup else None,
        })
    return rows


def _model_from_stem(path: Path) -> str | None:
    """`pair_b1_H_normal.json` -> `pair`: the file states no `model_id`, and its stem does."""
    stem = path.stem
    if "_" not in stem:
        return None
    head = stem.split("_")[0]
    return head or None


def list_native_measurements(artifacts_root: Path) -> dict[str, Any]:
    """Every native measurement in the selection tree, with the path of the block that states it."""
    selection = artifacts_root / "checkpoint_selection"
    paths: list[Path] = []
    if selection.is_dir():
        for path in sorted(selection.rglob("*.json")):
            text = str(path)
            if any(part in text for part in SKIP_PARTS):
                continue
            paths.append(path)

    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _rows_for_file(path, artifacts_root):
            if row["model_id"] is None and row["metrics"]:
                row["model_id"] = _model_from_stem(path)
            rows.append(row)

    rows.sort(key=lambda item: (str(item.get("model_id")), str(item.get("protocol_id")),
                                str(item.get("source", {}).get("path")),
                                str(item.get("metric_path"))))
    measured = [row for row in rows if row.get("metrics")]
    per_model = [row for row in measured if not row.get("rollup")]
    rollups = [row for row in measured if row.get("rollup")]
    return {
        "measurements": rows,
        "counts": {
            "files_scanned": len(paths),
            "files": len({row["source"]["path"] for row in rows}),
            "blocks_with_metrics": len(measured),
            "with_metrics": len(per_model),
            "rollup_rows": len(rollups),
            "unreadable": len([row for row in rows if row.get("unreadable")]),
            "signposted_blockers": len([row for row in rows
                                        if not row.get("metrics") and row.get("blocker")]),
            # Two different questions, both answered: models this catalog states a number for
            # directly, and models that only appear inside a roll-up (still served, still unverified).
            "models_with_metrics": len({row.get("model_id") for row in per_model
                                        if row.get("model_id")}),
            "models_rollup_only": len({row.get("model_id") for row in rollups
                                       if row.get("model_id")}
                                      - {row.get("model_id") for row in per_model
                                         if row.get("model_id")}),
            "verified": 0,
        },
        "rollup_sources": sorted({row["source"]["path"] for row in rollups}),
        "verified": False,
        "note": ("Native measurements are served here as UNVERIFIED. They carry no completion proof, "
                 "evidence-ledger entry or verifier receipt, are not counted by /api/evidence/summary, "
                 "and must not be read as a leaderboard position. `rollup: true` rows repeat a number "
                 "another file states and are excluded from the per-model count."),
    }


def native_measurement_protocols(payload: dict[str, Any]) -> list[str]:
    """Protocols this catalog holds a measurement for, most measurements first."""
    counts: dict[str, int] = {}
    for row in payload.get("measurements", []):
        protocol = row.get("protocol_id")
        if protocol and row.get("metrics"):
            counts[protocol] = counts.get(protocol, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
