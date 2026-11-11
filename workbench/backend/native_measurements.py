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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def metric_values(block: dict[str, Any]) -> tuple[dict[str, float], str | None, str, dict[str, str]]:
    """`(values, unit, unit_source, per_metric_units)` for ONE block, by ONE rule for every caller.

    R07: the value served and the scale it is labelled with MUST come from the same key. Per metric,
    the suffixes are tried in `UNIT_SUFFIX_ORDER` and the first key that is a real number wins, so
    `macro_r10: 0.2793` beside `macro_r10_percent: 27.93` serves 27.93 as `percent` — never 0.2793
    labelled `percent`.

    R07b: a suffix only declares anything when it is appended to THAT metric's own alias. A block
    carrying `dead_parameters_pct` says nothing about the recall scale, and is not read as if it did.

    `per_metric_units` names the suffix each metric's value came from (`_percent`, `_pct`, `` for the
    bare key, `_fraction`), so a block that mixes scales is visible instead of collapsed to one guess.
    `unit_source` is `sibling_key` when a suffix stated the scale and `undeclared` otherwise; callers
    add their own `declared` (document-level) and magnitude fallbacks.
    """
    values: dict[str, float] = {}
    per_metric_units: dict[str, str] = {}
    declared: str | None = None
    for name, aliases in METRIC_ALIASES.items():
        chosen: str | None = None
        for suffix in UNIT_SUFFIX_ORDER:
            hit = next((block[f"{alias}{suffix}"] for alias in aliases
                        if _is_number(block.get(f"{alias}{suffix}"))), None)
            if hit is None:
                continue
            values[name] = float(hit)
            chosen = suffix
            break
        if chosen is None:
            continue
        per_metric_units[name] = chosen
        # The block's scale is stated per METRIC, so a `_fraction` key of the same metric still
        # declares it when the bare key happens to win the value. A key of ANOTHER metric never does,
        # and a declaration that would describe a different number than the one served is refused.
        scale = chosen if chosen else _agreed_suffix(block, aliases, values[name])
        if scale and declared is None:
            declared = dict(UNIT_SUFFIX_DECLARATIONS)[scale]
    if declared:
        return values, declared, "sibling_key", per_metric_units
    return values, None, "undeclared", per_metric_units

def _agreed_suffix(block: dict[str, Any], aliases: tuple[str, ...], bare: float) -> str | None:
    """The suffix THIS metric's aliases carry that states the same number on another scale.

    `{"macro_r10": 0.6, "macro_r10_fraction": 0.6}` states one recall twice and names its scale; that
    declaration is used (`fraction`). `{"macro_r10": 12.0, "macro_r10_fraction": 0.12}` also carries
    both, but only the bare key is used here, and 0.12 does not describe 12.0 — so it declares
    nothing and the scale stays undeclared rather than being asserted from a different number.
    """
    for suffix in UNIT_SUFFIX_ORDER:
        if not suffix:
            continue
        if not any(_is_number(block.get(f"{alias}{suffix}")) for alias in aliases):
            continue
        value = next(float(block[f"{alias}{suffix}"]) for alias in aliases
                     if _is_number(block.get(f"{alias}{suffix}")))
        expected = bare * 100.0 if suffix in ("_percent", "_pct") else bare
        if math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-9):
            return suffix
    return None

def served_metrics(block: dict[str, Any], document_unit: str | None = None
                   ) -> tuple[dict[str, float], str | None, str, dict[str, str]]:
    """`metric_values` plus the two fallbacks, in the one order both paths use.

    A suffixed sibling of the metric's own key states the scale of the number it sits beside, so it
    outranks a document-level `unit` (`sibling_key`), which outranks magnitude inference
    (`inferred_from_magnitude`). Nothing here guesses a scale a file wrote down.
    """
    values, unit, unit_source, per_metric_units = metric_values(block)
    if unit:
        return values, unit, unit_source, per_metric_units
    if document_unit:
        return values, document_unit, "declared", per_metric_units
    unit, unit_source = _unit_of(values, None)
    return values, unit, unit_source, per_metric_units


def _blocks(node: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict[str, float], str | None, str, dict[str, str]]]:
    """Every macro block in a document, with the key path that reached it and its parsed scale.

    Walks mappings and lists alike, because `cohort_summary.json` is a list of per-model objects and
    `aggregate` may itself sit inside a list member.

    The scale comes from the SAME per-metric parser the report path uses (`metric_values`), so a row
    can never pair one key's number with another key's unit.
    """
    if isinstance(node, dict):
        key = trail[-1].lower() if trail else ""
        # The whole PATH decides, not the leaf key. `pair_b1_H_normal.paper_reference.aggregate`
        # reaches `aggregate` — a macro-block key — from inside a reference subtree, so testing only
        # the last element counted the publication's macro_r10 as this deployment's measurement.
        excluded = any(token in part.lower() for part in trail for token in EXCLUDED_TOKENS)
        numbers, unit, unit_source, per_metric_units = metric_values(node)
        # Bare `r10`/`r50` only count as a measurement when the block was reached under a
        # macro-shaped key. Without that gate `all_measured_summary.json`'s per-category
        # `dress`/`shirt`/`toptee` blocks each read as a macro measurement.
        macro_shaped = not trail or key in MACRO_BLOCK_KEYS or any(m in node for m in MACRO_MARKERS)
        if numbers and not excluded and macro_shaped:
            yield trail, numbers, unit, unit_source, per_metric_units
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


def _checkpoint_fingerprint(payload: dict[str, Any]) -> tuple[str, ...] | None:
    """The checkpoint identities a document states UNANIMOUSLY, or None when it states none.

    `hint_fashioniq.json` and `hint_doc_command.json` are ONE run written twice: same
    `checkpoint_sha256` (`bf9c72d4…`), same `checkpoint_id` (`fashioniq`), same numbers, different
    timestamps. Reading the weights digest the file itself records is what makes that a statement of
    ancestry rather than a guess from equal scores — two genuinely different checkpoints produce
    different digests, and two checkpoints of one model can easily produce identical recall.
    """
    from_payload = payload.get("checkpoint_sha256")
    if isinstance(from_payload, str) and from_payload.strip():
        return (from_payload.strip(),)
    members = payload.get("categories")
    if not isinstance(members, list):
        members = payload.get("results")
    if not isinstance(members, list):
        return None
    digests = {(member.get("provenance") or {}).get("checkpoint_sha256")
               for member in members if isinstance(member, dict)}
    digests = {digest.strip() for digest in digests if isinstance(digest, str) and digest.strip()}
    return tuple(sorted(digests)) if digests else None

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
        # R01: the identity the row wore BEFORE a report replaced its number, so the original label
        # is never lost. `None` (not `True`) until the backing pass reads the evidence.
        "raw_model_id": None,
        "raw_checkpoint_id": None,
        "raw_protocol_id": None,
        "raw_source_path": None,
        # Which identity rule joined the row to its report (`exact` / `single_candidate_protocol_unknown`)
        # and, when nothing was substituted, the reports that made the join ambiguous.
        "identity_join": None,
        "identity_ambiguous": None,
        # R08: equal numbers with no shared ancestry — informational, never a demotion.
        "same_metrics_as": None,
        # Which suffix each metric's value came from (R07: mixed scales stay visible).
        "per_metric_units": None,
        "checkpoint_fingerprints": None,
    }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [{**base, "measurement_id": hashlib.sha256(relative.encode()).hexdigest()[:16],
                 "unreadable": f"{type(error).__name__}: {error}"}]
    if not isinstance(payload, dict):
        # A JSON array of some other structure (attempts.json) is not a measurement document.
        return []

    blocks = list(_blocks(payload))
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
    owners = {_owner_key(trail) for trail, *_rest in blocks}
    known_rollup = any(relative.endswith(name) for name in KNOWN_ROLLUPS)
    rollup = known_rollup or (len(blocks) > 1 and len(owners - {None}) > 1)

    rows: list[dict[str, Any]] = []
    for trail, numbers, block_unit, block_unit_source, per_metric_units in blocks:
        owner = _owner_key(trail)
        # A roll-up entry names the MODEL in its key (`clip4cir_noft`, `B1/H_normal`,
        # `combiner_rn50x4_noft`); a direct file names it in `model_id` or in its own stem.
        model = owner if rollup else (model_hint or owner)
        # R07: the block's scale came from the SAME per-metric parser the report path uses, so the
        # value and its unit describe one key. A document-level declaration still outranks it —
        # "the file says percent" and "this block names its own scale" are different evidence —
        # and magnitude inference is the last resort, never a substitute for either.
        if declared_unit:
            unit, unit_source = declared_unit, "declared"
        elif block_unit:
            unit, unit_source = block_unit, block_unit_source
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
            # Which suffix each metric's value came from, so a mixed-scale block is visible rather
            # than collapsed into the one label above.
            "per_metric_units": per_metric_units,
            # The weights the document says it measured (see `_checkpoint_fingerprint`).
            "checkpoint_fingerprints": _checkpoint_fingerprint(payload),
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

def _report_metrics(metrics: dict[str, Any], document_unit: str | None = None
                    ) -> tuple[dict[str, float], str | None, str, dict[str, str]]:
    """The served numbers of ONE report block — `served_metrics` under its historical name.

    W-01: `dcnet.json` states its aggregate twice — `macro_r10: 0.2793` beside
    `macro_r10_percent: 27.9297`. Reading the bare key served a FRACTION while the row's `unit` came
    from the *source* roll-up of a different run, so DCNet rendered as 0.28 % against every other
    model's ~50 %.

    Delegates to the ONE parser the row path also uses, so a report and a row can never disagree
    about which key a value came from. Precedence, highest first: a suffixed sibling key of the
    metric's own alias (`sibling_key`), the document's own unanimous provenance declaration
    (`declared`), then magnitude inference (`inferred_from_magnitude`).
    """
    return served_metrics(metrics, document_unit)

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
            values, unit, unit_source, per_metric_units = _report_metrics(
                payload.get("observed_metrics") or {})
            if not values:
                continue
            reports.append({
                "model_id": payload["model_id"],
                "canonical_model": _canonical_model(payload["model_id"]),
                "checkpoint_id": payload.get("checkpoint_id"),
                "protocol_id": payload.get("protocol_id"),
                "run_id": payload.get("run_id"),
                "metrics": values,
                "unit": unit,
                "unit_source": unit_source,
                "per_metric_units": per_metric_units,
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
        values, unit, unit_source, per_metric_units = _report_metrics(
            aggregate, _provenance_unit(payload))
        if not values:
            continue
        reports.append({
            "model_id": payload["model_id"],
            "canonical_model": _canonical_model(payload["model_id"]),
            # A replay states no checkpoint of its own, so this is `None` and the join can only
            # check the fields BOTH sides state (see `IDENTITY_JOIN_FIELDS`).
            "checkpoint_id": payload.get("checkpoint_id"),
            "protocol_id": payload.get("protocol"),
            "run_id": payload.get("replay_type"),
            "metrics": values,
            "unit": unit,
            "unit_source": unit_source,
            "per_metric_units": per_metric_units,
            "path": str(path),
            "sha256": _digest(path),
            "source_kind": "native_replay",
        })
    return reports


#: Fields a row and a backing report must agree on before the report may replace the row's number.
#:
#: Read from BOTH shapes, not assumed:
#:   model_id      every row states one (`model_id`/`model`/`method_id`/roll-up entry key) and every
#:                 report does (`reports/*_aggregate.json`, `raw/native/*.json`).
#:   protocol_id   rows state `protocol_id` or `protocol`; aggregate reports state `protocol_id`,
#:                 native replays state `protocol`.
#:   checkpoint_id rows state `checkpoint_id`/`variant`/`checkpoint_name`/`label`; aggregate reports
#:                 state `checkpoint_id`; native replays state NOTHING — so this field is compared
#:                 only when the report states it (a field a report leaves blank cannot contradict).
#: Could NOT be included, because one side never carries it: `dataset`/`dataset_id` (neither side),
#: `benchmark`/`benchmark_reference` (reports only, and null on every report under `reports/`), and
#: `run_id` (reports only). Adding any of them would refuse every join rather than tighten one.
IDENTITY_JOIN_FIELDS = ("model_id", "protocol_id", "checkpoint_id")

def _row_checkpoint(row: dict[str, Any]) -> str | None:
    """The row's checkpoint LABEL, or None when what it states is not a checkpoint statement.

    A roll-up entry key that merely repeats the model name (`selection_decisions.json`'s
    `combiner_rn50x4_noft`) is the roll-up naming its model, not a checkpoint of it, so it cannot
    constrain which report may back the row.
    """
    checkpoint = row.get("checkpoint_id")
    if checkpoint is None or checkpoint == row.get("model_id"):
        return None
    return checkpoint

def _checkpoint_agrees(row: dict[str, Any], report: dict[str, Any]) -> bool:
    """Whether the row's checkpoint statement is absent or identical to the report's."""
    row_checkpoint = _row_checkpoint(row)
    report_checkpoint = report.get("checkpoint_id")
    return not (row_checkpoint and report_checkpoint and row_checkpoint != report_checkpoint)

def _identity_matches(row: dict[str, Any], report: dict[str, Any]) -> bool:
    """Whether `report` states the same measurement identity as `row` on every field both state."""
    row_protocol, report_protocol = row.get("protocol_id"), report.get("protocol_id")
    if row_protocol and report_protocol and row_protocol != report_protocol:
        return False
    return _checkpoint_agrees(row, report)

def _backed_by_artifact(rows: list[dict[str, Any]],
                        artifacts_root: Path) -> list[dict[str, Any]]:
    """Replace an unbacked number with the artifact's own, then mark repeated identities.

    The rule the payload enforces: **a served measurement value is one an artifact backs.** A roll-up
    (`clip4cir_parsed.json`, `selection_decisions.json`) states 43.8182 for CLIP4Cir FullFT while the
    replay artifact that produced a number states 43.8017 — two different runs, and no file under
    `checkpoint_selection/` corresponds to the roll-up's dress/toptee cells.

    R01: a report may back a row only when `IDENTITY_JOIN_FIELDS` agree — the row's label and the
    number served then describe ONE measurement. The previous `by_model_any_protocol` fallback let
    `checkpoint_A`/`protocol_A` be served 88.0 from an unrelated `checkpoint_B`/`protocol_B` report
    while keeping its own label, i.e. an artifact silently relabelling itself. That fallback is GONE.
    Policy when identities differ: REFUSE the replacement (never overwrite the row's fields) — the
    row keeps its own number, `artifact_backed` and `counted` stay exactly as they were, and the
    report is not served under the row's label.

    Two reports matching one identity are not one measurement's evidence either: the row keeps its
    own number and states `identity_ambiguous: [paths]`, so a reader sees why nothing was substituted.

    The single permitted relaxation: the row states NO protocol and the model has exactly ONE
    candidate report, so the protocol is taken from that report and the join is recorded as
    `identity_join: "single_candidate_protocol_unknown"`. The checkpoint identity must STILL agree
    (`_identity_matches`) — a row that names a checkpoint is a specific claim, not a wildcard, so
    `checkpoint_A` with a single `checkpoint_B` report is REFUSED and listed in
    `identity_ambiguous` like every other mismatch.

    Nothing is substituted unless `_identity_matches` held for it; that is asserted below, so no
    future edit can reopen the bypass this docstring describes.
    """
    reports = _backing_reports(artifacts_root)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_model.setdefault(report["canonical_model"] or "", []).append(report)

    for row in rows:
        # The label as the file wrote it, kept for `raw_model_id` below: canonicalising the alias is
        # a separate, deliberate step and must not be confused with losing the original name.
        original_model = row.get("model_id")
        model = _canonical_model(original_model)
        row["model_id"] = model
        if not row.get("metrics"):
            continue
        candidates = by_model.get(model or "", [])
        matches = [report for report in candidates if _identity_matches(row, report)]
        identity_join: str | None = None
        if len(matches) == 1:
            report: dict[str, Any] | None = matches[0]
            # The join is a RELAXATION, and says so, exactly when the row stated no protocol and the
            # single candidate supplied one. The checkpoint still had to agree (`_identity_matches`),
            # so a row naming `checkpoint_A` can never be backed by a `checkpoint_B` report.
            identity_join = ("single_candidate_protocol_unknown"
                             if row.get("protocol_id") is None and len(candidates) == 1
                             and report.get("protocol_id")
                             else "exact")
        elif len(matches) > 1:
            report = None
            row["identity_ambiguous"] = sorted(item["path"] for item in matches)
        else:
            # No report states this identity. When candidates exist, name them: a reader must be
            # able to see WHY its number was not substituted (R01: refuse, never relabel).
            report = None
            if candidates:
                row["identity_ambiguous"] = sorted(item["path"] for item in candidates)
        # The invariant, enforced rather than documented: nothing may be substituted unless the
        # identity really matched, so no future edit can reopen the R01 bypass.
        if report is not None and identity_join != "exact" and not _identity_matches(row, report):
            raise AssertionError(
                f"R01 bypass: {row['source']['path']} ({row.get('checkpoint_id')!r}, "
                f"{row.get('protocol_id')!r}) would be replaced by {report['path']} "
                f"({report.get('checkpoint_id')!r}, {report.get('protocol_id')!r})")
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
        # The number is replaced, so its SCALE must be re-derived from the report that replaced it,
        # and the row's OWN identity is kept beside it: the original label must never be lost.
        row["raw_metrics"] = dict(row["metrics"])
        row["raw_unit"] = row.get("unit")
        row["raw_unit_source"] = row.get("unit_source")
        row["raw_model_id"] = original_model
        row["raw_checkpoint_id"] = row.get("checkpoint_id")
        row["raw_protocol_id"] = row.get("protocol_id")
        row["raw_source_path"] = row["source"]["path"]
        row["metrics"] = dict(report["metrics"])
        row["unit"] = report["unit"]
        row["unit_source"] = report["unit_source"]
        row["per_metric_units"] = report.get("per_metric_units")
        row["artifact_backed"] = True
        row["source_kind"] = "artifact"
        row["backed_by"] = report["path"]
        row["backed_by_sha256"] = report["sha256"]
        row["backed_by_run_id"] = report["run_id"]
        row["backed_by_kind"] = report["source_kind"]
        row["identity_join"] = identity_join
        row["protocol_id"] = row.get("protocol_id") or report["protocol_id"]

    return _mark_duplicates(rows)

def _mark_duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark repeated measurements by IDENTITY; never by a rounded score; NEVER drop a row.

    R08: equal numbers are not identity. Two checkpoints of one model that happen to score the same
    are two measurements, so they both stay `counted: true` and merely record
    `same_metrics_as: <path>` — the earlier version keyed on `round(value, 4)` and silently demoted
    the second checkpoint to a duplicate of the first.

    `duplicate_of` is written in exactly three ancestry situations:

    1. Two rows are backed by the SAME artifact (`backed_by` equal) for the same model and protocol:
       two files stating one run's number (the roll-up pair, `clip4cir_parsed.json` and
       `selection_decisions.json`). The non-roll-up row, else the first in path order, counts.
    2. Two exports state the SAME `checkpoint_sha256` for the same model+protocol and one number:
       ONE run written twice under two filenames (`hint_fashioniq.json` / `hint_doc_command.json`,
       `tme_fashioniq_0.json` / `tme_normal_path_full.json`). The weights digest is the file stating
       its own ancestry; equal scores alone are not, and two different checkpoints stay two rows.
    3. A roll-up row restates an in-tree artifact's number: the relation is established from the
       source KIND (`rollup:*` vs `artifact`), and value equality only picks WHICH artifact it
       restates.

    A row that is merely numerically equal to another is not a duplicate and is not demoted.

    The stated identity — same source `sha256` AND same `metric_path`, i.e. one block stated twice —
    is enforced by construction rather than by a rule here: `_blocks` yields each block once and the
    walker visits each path once, so `(sha256, metric_path)` is already unique across rows. It is
    named in this docstring because it is the identity the rules below are built on, not a check that
    was skipped.
    """
    fingerprint = {}
    for row in rows:
        metrics = row.get("metrics")
        if not metrics:
            continue
        fingerprint[row["measurement_id"]] = tuple(
            sorted((name, round(value, 6)) for name, value in metrics.items()))

    # 1. A roll-up and a row sharing one backing artifact state one run twice. Two plain exports that
    #    merely agree are NOT demoted: nothing says one of them is the other's source.
    by_backing: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("metrics") or not row.get("backed_by") or row["backed_by"] == row["source"]["path"]:
            continue
        key = (row["backed_by"], row.get("model_id"), row.get("protocol_id"),
               fingerprint[row["measurement_id"]])
        by_backing.setdefault(key, []).append(row)
    for group in by_backing.values():
        if len(group) < 2 or not any(item.get("rollup") for item in group):
            continue
        keeper = min(group, key=lambda item: (item.get("rollup", False),
                                              str(item["source"]["path"])))
        for row in group:
            if row is keeper:
                continue
            row["duplicate_of"] = keeper["source"]["path"]
            row["counted"] = False

    # 2. Two exports of ONE checkpoint stating one number (same weights digest). This is ancestry the
    #    file states, not a guess from equal scores: two DIFFERENT checkpoints of one model can score
    #    identically, and those stay two measurements (`same_metrics_as` only).
    by_checkpoint: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        fingerprints = row.get("checkpoint_fingerprints")
        if not row.get("metrics") or row.get("duplicate_of") or not fingerprints:
            continue
        key = (row.get("model_id"), row.get("protocol_id"), fingerprints,
               fingerprint[row["measurement_id"]])
        by_checkpoint.setdefault(key, []).append(row)
    for group in by_checkpoint.values():
        if len(group) < 2:
            continue
        keeper = max(group, key=lambda item: (str(item.get("recorded_at") or ""),
                                              str(item["source"]["path"])))
        keeper["checkpoint_ancestry_count"] = len(group)
        for row in group:
            if row is keeper:
                continue
            row["duplicate_of"] = keeper["source"]["path"]
            row["counted"] = False

    # 3. A roll-up restating an artifact's number. A roll-up usually states NO protocol, so an
    #    absent protocol on either side does not break the ancestry; two DIFFERENT protocols do.
    artifacts = [row for row in rows
                 if row.get("metrics") and not row.get("rollup")
                 and not row.get("duplicate_of")]
    for row in rows:
        if not row.get("metrics") or not row.get("rollup") or row.get("duplicate_of"):
            continue
        for artifact in artifacts:
            protocols_agree = (artifact.get("protocol_id") is None
                               or row.get("protocol_id") is None
                               or artifact.get("protocol_id") == row.get("protocol_id"))
            if (artifact.get("model_id") == row.get("model_id")
                    and protocols_agree
                    and fingerprint[artifact["measurement_id"]]
                    == fingerprint[row["measurement_id"]]):
                row["duplicate_of"] = artifact["source"]["path"]
                row["counted"] = False
                break

    # 4. Equal numbers with no ancestry: informational only, NEVER a demotion.
    for row in rows:
        if not row.get("metrics") or row.get("duplicate_of"):
            continue
        peers = [other for other in rows
                 if other is not row and other.get("metrics")
                 and not other.get("duplicate_of")
                 and other["measurement_id"] != row["measurement_id"]
                 and fingerprint[other["measurement_id"]] == fingerprint[row["measurement_id"]]
                 and other.get("model_id") == row.get("model_id")]
        if peers:
            row["same_metrics_as"] = sorted(peer["source"]["path"] for peer in peers)[0]
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
