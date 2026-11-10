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
import math
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


#: Keys a document uses to DECLARE the unit of its macro numbers. A declared unit always wins: a
#: value of `0.2793` is only a fraction because nothing in that file says otherwise.
UNIT_KEYS = ("metric_unit", "units", "unit", "recall_unit")
UNIT_FRACTION = ("fraction", "ratio", "proportion", "0-1", "share")
UNIT_PERCENT = ("percent", "percentage", "%", "0-100")
#: A block key that ENDS in one of these declares the unit of the number it carries, with no
#: document-level `unit` needed: `dcnet.json` states `aggregate.macro_r10_percent` beside the raw
#: `macro_r10`, so the scale is written down inside the block and nothing is left to infer.
UNIT_SUFFIX_DECLARATIONS = (("_percent", "percent"), ("_pct", "percent"), ("_fraction", "fraction"))
#: Suffix precedence for reading a report block: an explicit percent declaration first, the bare key
#: second, an explicit fraction declaration last. `dcnet.json` carries BOTH `macro_r10: 0.2793` and
#: `macro_r10_percent: 27.9297`; the declared percent is the number the file means.
UNIT_SUFFIX_ORDER = ("_percent", "_pct", "", "_fraction")
#: `unit_source` values that mean the FILE stated the scale, as opposed to magnitude inference.
DECLARED_UNIT_SOURCES = ("declared", "sibling_key")


def _declared_unit(payload: dict[str, Any]) -> str | None:
    """`"fraction"` / `"percent"` when the DOCUMENT says so, else None.

    Read from the document root only: a nested block's own `unit` would describe a different number
    than the macro block this row serves.
    """
    for key in UNIT_KEYS:
        value = payload.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip().lower()
        if any(token in text for token in UNIT_FRACTION):
            return "fraction"
        if any(token in text for token in UNIT_PERCENT):
            return "percent"
    return None


def _suffix_declared_unit(block: dict[str, Any]) -> str | None:
    """The unit a SUFFIXED SIBLING key declares inside this very block, e.g. `macro_r10_percent`.

    This is a DECLARATION, not an inference: the file names the scale of its own number, so it
    outranks magnitude inference. Without it a percent-valued block whose three categories all
    happen to score under 1 % would be read as a fraction and rendered 100x too small.
    """
    for key in block:
        lowered = str(key).lower()
        for suffix, unit in UNIT_SUFFIX_DECLARATIONS:
            if lowered.endswith(suffix):
                return unit
    return None


def _unit_of(numbers: dict[str, float], declared: str | None) -> tuple[str | None, str]:
    """The unit this block's numbers are in, and how that was decided.

    Never inferred from a single value: one category at 0.2793 beside a macro at 27.93 is a
    mixed-scale document, and calling that "fraction" would divide the percent by 100 again.
    Magnitude inference is the LAST resort — a caller holding a declaration never reaches it.
    """
    if declared:
        return declared, "declared"
    values = [value for value in numbers.values() if isinstance(value, (int, float))]
    if values and all(0.0 <= value <= 1.0 for value in values):
        return "fraction", "inferred_from_magnitude"
    if values and all(value >= 0.0 for value in values):
        return "percent", "inferred_from_magnitude"
    return None, "undeclared"


def _blocks(node: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, float], str | None]]:
    """Every macro block in a document, with the key path that reached it and its DECLARED unit.

    Walks mappings and lists alike, because `cohort_summary.json` is a list of per-model objects and
    `aggregate` may itself sit inside a list member.
    """
    if isinstance(node, dict):
        key = trail[-1].lower() if trail else ""
        # The whole PATH decides, not the leaf key. `pair_b1_H_normal.paper_reference.aggregate`
        # reaches `aggregate` — a macro-block key — from inside a reference subtree, so testing only
        # the last element counted the publication's macro_r10 as this deployment's measurement.
        excluded = any(token in part.lower() for part in trail for token in EXCLUDED_TOKENS)
        numbers = _numbers(node)
        # Bare `r10`/`r50` only count as a measurement when the block was reached under a
        # macro-shaped key. Without that gate `all_measured_summary.json`'s per-category
        # `dress`/`shirt`/`toptee` blocks each read as a macro measurement.
        macro_shaped = not trail or key in MACRO_BLOCK_KEYS or any(m in node for m in MACRO_MARKERS)
        if numbers and not excluded and macro_shaped:
            yield trail, numbers, _suffix_declared_unit(node)
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
        # Present on every row, including blockers: `None` means "the artifact does not say", which
        # is the honest answer and the one a consumer must render as a warning.
        "unit": None,
        "unit_source": None,
        "mapping_status": None,
        "counted": True,
        "duplicate_of": None,
        "artifact_backed": False,
        "source_kind": None,
        "backed_by": None,
        "raw_metrics": None,
        # The pre-replacement scale, kept beside `raw_metrics` for the same reason: a replaced value
        # must be readable on the scale its own file used.
        "raw_unit": None,
        "raw_unit_source": None,
    }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{**base, "measurement_id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                 "unreadable": f"{type(error).__name__}: {error}"}]
    if not isinstance(payload, dict):
        # A JSON array of some other structure (attempts.json) is not a measurement document.
        return []

    blocks = [(trail, numbers, declared_in_block)
              for trail, numbers, declared_in_block in _blocks(payload)]
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
    declared_unit = _declared_unit(payload)
    # `mapping_status` is what makes an exploratory run readable as exploratory. `pair_b1_H_*.json`
    # carries `EXPLORATORY_PUBLIC_MAPPING` and the row used to drop it, so a reader saw a number with
    # no statement that the class mapping is unverified.
    mapping_status = _first_text(payload, "mapping_status", "provenance_label")
    # Two or more blocks with DIFFERENT owners in one file is a roll-up by construction: a single
    # measurement cannot be about two models. A known roll-up file is one even when it holds a single
    # block: `pair_compat/measurements/summary.json` repeats numbers its siblings state.
    owners = {_owner_key(trail) for trail, _numbers, _unit in blocks}
    known_rollup = any(relative.endswith(name) for name in KNOWN_ROLLUPS)
    rollup = known_rollup or (len(blocks) > 1 and len(owners - {None}) > 1)

    rows: list[dict[str, Any]] = []
    for trail, numbers, declared_in_block in blocks:
        owner = _owner_key(trail)
        # A roll-up entry names the MODEL in its key (`clip4cir_noft`, `B1/H_normal`,
        # `combiner_rn50x4_noft`); a direct file names it in `model_id` or in its own stem.
        model = owner if rollup else (model_hint or owner)
        # A document-level declaration outranks the block's own suffixed sibling key, which in turn
        # outranks magnitude inference. The two declaration kinds are labelled apart, because
        # "the document says percent" and "this block names its own scale" are different evidence.
        if declared_unit:
            unit, unit_source = declared_unit, "declared"
        elif declared_in_block:
            unit, unit_source = declared_in_block, "sibling_key"
        else:
            unit, unit_source = _unit_of(numbers, None)
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
            # The scale the numbers are on, stated rather than assumed: a consumer that prints `r10`
            # as a percent must not print a fraction as 0.28 %.
            "unit": unit,
            "unit_source": unit_source,
            "mapping_status": mapping_status,
            "rollup": rollup,
            "rollup_entry": owner if rollup else None,
            # A roll-up STATES a number; it does not BACK one. `None` because the backing pass
            # decides, and a default of True would be a claim made before the evidence was read.
            "counted": True,
            "duplicate_of": None,
            "artifact_backed": not rollup,
            "source_kind": "rollup" if rollup else "artifact",
            "backed_by": None,
            # The number the file wrote, kept verbatim so a replaced value can be audited against
            # what the roll-up claimed. Nothing served is ever silently overwritten.
            "raw_metrics": None,
            "raw_unit": None,
            "raw_unit_source": None,
        })
    return rows


def _model_from_stem(path: Path) -> str | None:
    """`pair_b1_H_normal.json` -> `pair`: the file states no `model_id`, and its stem does."""
    stem = path.stem
    if "_" not in stem:
        return None
    head = stem.split("_")[0]
    return head or None


#: `clip4cir_fullft` and `clip4cir_rn50x4_fullft` are ONE configuration of ONE upstream: the roll-up
#: writes the variant name, the reproducing artifact writes the model id. Serving both presented one
#: number twice under two names, which reads as two configurations that agree.
ALIASES = {
    "clip4cir_fullft": "clip4cir_rn50x4_fullft",
    "clip4cir_noft": "clip4cir_rn50x4_noft",
    "clvc": "clvc_net",
    "sprc_vitl": "sprc",
}


def _canonical_model(model: str | None) -> str | None:
    return ALIASES.get(model, model) if model else model


def _provenance_unit(payload: dict[str, Any]) -> str | None:
    """A unit the document states UNANIMOUSLY in its per-category provenance.

    `clip4cir_fullft.json` writes `categories[*].provenance.units = "percent, as printed by upstream
    (recall_at10 * 100)"` — a declaration of the macro's scale that sits beside the numbers instead of
    in the aggregate block. It counts only when every category agrees: a document whose categories
    disagree states no single scale for the macro.
    """
    texts = {(category.get("provenance") or {}).get("units") for category in payload.get("categories") or []}
    texts = {text.strip().lower() for text in texts if isinstance(text, str) and text.strip()}
    if len(texts) != 1:
        return None
    text = texts.pop()
    if any(token in text for token in UNIT_FRACTION):
        return "fraction"
    if any(token in text for token in UNIT_PERCENT):
        return "percent"
    return None

def _report_metrics(metrics: dict[str, Any],
                    document_unit: str | None = None) -> tuple[dict[str, float], str | None, str]:
    """The served numbers of ONE report block, normalised to the scale the block declares.

    W-01: `dcnet.json` states its aggregate twice — `macro_r10: 0.2793` beside
    `macro_r10_percent: 27.9297`. Reading the bare key served a FRACTION while the row's `unit` came
    from the *source* roll-up of a different run, so DCNet rendered as 0.28 % against every other
    model's ~50 %. A suffixed sibling is the report naming its own scale, so it wins and the metric
    keeps the value the declaration covers.

    Returns `(values, unit, unit_source)`. Precedence, highest first: a suffixed sibling key in the
    block (`sibling_key`), the document's own unanimous provenance declaration (`declared`), then
    magnitude inference — which `unit_inferred` names so it is never mistaken for a declaration.
    """
    values: dict[str, float] = {}
    suffix_unit: str | None = None
    for name, aliases in METRIC_ALIASES.items():
        for suffix in UNIT_SUFFIX_ORDER:
            hit = next((metrics[f"{alias}{suffix}"] for alias in aliases
                        if isinstance(metrics.get(f"{alias}{suffix}"), (int, float))
                        and not isinstance(metrics.get(f"{alias}{suffix}"), bool)), None)
            if hit is None:
                continue
            values[name] = float(hit)
            if suffix and suffix_unit is None:
                suffix_unit = dict(UNIT_SUFFIX_DECLARATIONS).get(suffix)
            break
    if suffix_unit:
        return values, suffix_unit, "sibling_key"
    if document_unit:
        return values, document_unit, "declared"
    # Nothing declared: magnitude inference is the last resort, and `unit_inferred` names it.
    unit, unit_source = _unit_of(values, None)
    return values, unit, unit_source


def _repo_root_for(artifacts_root: Path) -> Path:
    """The checkout holding both `workbench/artifacts/` and `workbench/docs/reviews/`.

    Found by walking up for the first ancestor that has `docs/reviews`, so a temp artifacts root in a
    test simply finds nothing rather than reaching outside itself.
    """
    for candidate in (artifacts_root, *artifacts_root.parents):
        if (candidate / "docs" / "reviews").is_dir():
            return candidate
        if (candidate / "workbench" / "docs" / "reviews").is_dir():
            return candidate / "workbench"
    return artifacts_root


def _backing_reports(artifacts_root: Path) -> list[dict[str, Any]]:
    """Artifacts that can BACK a served number.

    Two structural sources, never a roll-up:

    * `reports/*_aggregate.json` — a run whose aggregate report states its own observed metrics;
    * `docs/reviews/*/raw/native/*.json` with a `replay_type` and an `aggregate` — a
      SOURCE_NATIVE_ENTRYPOINT replay, which also names per-category query and gallery counts.

    A roll-up under `checkpoint_selection/` can back neither: it is exactly what must not be able to
    vouch for itself.
    """
    reports: list[dict[str, Any]] = []

    reports_dir = artifacts_root / "reports"
    if reports_dir.is_dir():
        for path in sorted(reports_dir.glob("*_aggregate.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("model_id"):
                continue
            values, unit, unit_source = _report_metrics(payload.get("observed_metrics") or {})
            if not values:
                continue
            reports.append({
                "model_id": payload["model_id"],
                "canonical_model": _canonical_model(payload["model_id"]),
                "protocol_id": payload.get("protocol_id"),
                "run_id": payload.get("run_id"),
                "metrics": values,
                "unit": unit,
                "unit_source": unit_source,
                "path": str(path.relative_to(artifacts_root)),
                "sha256": _digest(path),
                "source_kind": "aggregate_report",
            })

    for path in sorted(_repo_root_for(artifacts_root).glob("docs/reviews/*/raw/native/*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("replay_type"):
            continue
        aggregate = payload.get("aggregate")
        if not isinstance(aggregate, dict) or not payload.get("model_id"):
            continue
        values, unit, unit_source = _report_metrics(aggregate, _provenance_unit(payload))
        if not values:
            continue
        reports.append({
            "model_id": payload["model_id"],
            "canonical_model": _canonical_model(payload["model_id"]),
            "protocol_id": payload.get("protocol"),
            "run_id": payload.get("replay_type"),
            "metrics": values,
            "unit": unit,
            "unit_source": unit_source,
            "path": str(path),
            "sha256": _digest(path),
            "source_kind": "native_replay",
        })
    return reports


def _backed_by_artifact(rows: list[dict[str, Any]],
                        artifacts_root: Path) -> list[dict[str, Any]]:
    """Replace an unbacked number with the artifact's own, then mark repeated identities.

    The rule the payload now enforces: **a served measurement value is one an artifact backs.** A
    roll-up (`clip4cir_parsed.json`, `selection_decisions.json`) states 43.8182 for CLIP4Cir FullFT
    while the replay artifact that produced a number states 43.8017 — two different runs, and no file
    under `checkpoint_selection/` corresponds to the roll-up's dress/toptee cells. Serving the roll-up
    value presented an unreproducible number as a measurement, twice (once under `clip4cir_fullft`,
    once under `clip4cir_rn50x4_fullft`).

    Order matters: canonicalise the model first (so the alias pair is one model), then back, then
    mark repeats.
    """
    reports = _backing_reports(artifacts_root)
    by_model: dict[tuple[str, str | None], dict[str, Any]] = {}
    by_model_any_protocol: dict[str, dict[str, Any]] = {}
    for report in reports:
        by_model.setdefault((report["canonical_model"], report["protocol_id"]), report)
        by_model_any_protocol.setdefault(report["canonical_model"], report)

    for row in rows:
        model = _canonical_model(row.get("model_id"))
        row["model_id"] = model
        if not row.get("metrics"):
            continue
        report = (by_model.get((model, row.get("protocol_id")))
                  # A roll-up often omits the protocol; the only report for that model still backs it.
                  or by_model_any_protocol.get(model or ""))
        if report is None:
            if row.get("artifact_backed"):
                # A direct artifact is its own backing.
                row["backed_by"] = row["source"]["path"]
            else:
                row["artifact_backed"] = False
                row["source_kind"] = f"rollup:{row['source']['path'].split('/')[-1]}"
                # W-01: a number no artifact backs is DISPLAYED, labelled, and never counted. It used
                # to keep `counted: true`, so 18 unbacked roll-up claims fed the per-model tally.
                row["counted"] = False
            continue
        # The number is replaced, so its SCALE must be re-derived from the report that replaced it.
        # Keeping the pre-replacement `unit` printed DCNet's 0.2793 fraction under a `percent` the
        # module had inferred from a different file.
        row["raw_metrics"] = dict(row["metrics"])
        row["raw_unit"] = row.get("unit")
        row["raw_unit_source"] = row.get("unit_source")
        row["metrics"] = dict(report["metrics"])
        row["unit"] = report["unit"]
        row["unit_source"] = report["unit_source"]
        row["artifact_backed"] = True
        row["source_kind"] = "artifact"
        row["backed_by"] = report["path"]
        row["backed_by_sha256"] = report["sha256"]
        row["backed_by_run_id"] = report["run_id"]
        row["backed_by_kind"] = report["source_kind"]
        row["protocol_id"] = row.get("protocol_id") or report["protocol_id"]

    return _mark_duplicates(rows)


def _mark_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark repeated identities; NEVER drop a row.

    A roll-up repeating an artifact's number is not a second measurement, and the alias pair is ONE
    configuration of one upstream. `counted` is False on the repeat, but the row stays in
    `measurements`: a reader must be able to see that two files state one number, and a silently
    deleted row cannot be audited at all.

    Identity is a TOLERANCE, not `round(x, 6)`: `airknow` states 52.5466 in the roll-up and
    52.54661498980954 in the run file — one experiment written at two precisions — while
    `airknow_n05` (52.5466) and `airknow_n08` (50.6417) are two DIFFERENT runs and stay separate.
    Rounding at 6 decimals split the first pair and would merge nothing at all, so the precision
    repeats were counted twice.

    The artifact-backed row wins; between two rows of the same class the first in path order does,
    so the outcome is deterministic.
    """
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        metrics = row.get("metrics")
        if not metrics:
            continue
        key = (row.get("model_id"), row.get("protocol_id"),
               tuple(sorted((name, round(value, 4)) for name, value in metrics.items())))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            row["duplicate_of"] = None
            continue
        if row.get("artifact_backed") and not existing.get("artifact_backed"):
            row["duplicate_of"] = None
            existing["duplicate_of"] = row["source"]["path"]
            existing["counted"] = False
            by_key[key] = row
        else:
            row["duplicate_of"] = existing["source"]["path"]
            row["counted"] = False
    return rows


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

    rows = _backed_by_artifact(rows, artifacts_root)
    rows.sort(key=lambda item: (str(item.get("model_id")), str(item.get("protocol_id")),
                                str(item.get("source", {}).get("path")),
                                str(item.get("metric_path"))))
    measured = [row for row in rows if row.get("metrics")]
    # A served number is one an artifact backs; a repeat of one identity does not count twice, and a
    # number no artifact backs is never counted at all (it may still be displayed, labelled).
    counted_rows = [row for row in measured if row.get("counted", True)]
    per_model = [row for row in counted_rows if row.get("artifact_backed")]
    rollups = [row for row in measured if not row.get("artifact_backed")]
    return {
        "measurements": rows,
        "counts": {
            "files_scanned": len(paths),
            "files": len({row["source"]["path"] for row in rows}),
            # Every block the tree states (audit total), then the DISTINCT artifact-backed
            # measurements behind them, then the unbacked claims and the repeats.
            "blocks_with_metrics": len(measured),
            "with_metrics": len(per_model),
            "rollup_rows": len(rollups),
            "duplicate_rows": len([row for row in measured if not row.get("counted", True)]),
            "unbacked_rows": len(rollups),
            # MUST be 0: an unbacked number is displayed and labelled, never counted.
            "unbacked_counted": len([row for row in measured
                                     if not row.get("artifact_backed") and row.get("counted", True)]),
            # W-01: rows whose scale the artifact never DECLARES. Named as a count and as a list,
            # because a count alone cannot be audited.
            "unit_declared": len([row for row in measured
                                  if row.get("unit_source") in DECLARED_UNIT_SOURCES]),
            "unit_sibling_key": len([row for row in measured
                                     if row.get("unit_source") == "sibling_key"]),
            "unit_inferred": len([row for row in measured
                                  if row.get("unit_source") == "inferred_from_magnitude"]),
            "unit_undeclared": len([row for row in measured if not row.get("unit")]),
            "unit_not_declared": len([row for row in measured
                                      if row.get("unit_source") not in DECLARED_UNIT_SOURCES]),
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
        "undeclared_unit_rows": [
            {"measurement_id": row["measurement_id"], "model_id": row.get("model_id"),
             "metric_path": row.get("metric_path"), "metrics": row.get("metrics"),
             "unit": row.get("unit"), "unit_source": row.get("unit_source"),
             "source": row["source"]["path"]}
            for row in measured if row.get("unit_source") not in DECLARED_UNIT_SOURCES
        ],
        # Every file that STATES a number no artifact backs, whether or not it is the counted one.
        "rollup_sources": sorted({row["source"]["path"] for row in measured
                                  if not row.get("artifact_backed")}),
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
