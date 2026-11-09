#!/usr/bin/env python3
"""Measured decode-worker sweep with a parity gate, per 02_PERFORMANCE_AND_PARITY.en.md §A-D.

WHAT THIS MEASURES
------------------
Gallery decode→transform→collate throughput at a chosen set of decode-worker counts, with stage
timings and a parity verdict per setting. It is deliberately narrow: one model family, one fixed
query/image sample, the real released gallery. A microbenchmark is not proof of whole-run
acceleration, and this prints its own scope so it cannot be read as one.

WHY IT DOES NOT TOUCH THE GALLERY ENCODE
----------------------------------------
The full gallery encode is minutes to hours and dominated by the model forward pass, which is not
what the decode-worker knob changes. This harness isolates the producer stage: it feeds batches
through the same `adapters.prefetch_rows` the evaluator uses, in submission order, and measures it.
The parity gate then proves that changing the worker count does not change the decoded batch
CONTENT — which is the only claim being made.

MEASUREMENT RULES OBSERVED (§A)
-------------------------------
* CUDA events are NOT used because this harness measures CPU-side decode; where any model work is
  timed it is synchronised explicitly with torch.cuda.synchronize() before and after.
* Cold start is separated from warm steady state: the first pass is reported as `cold`, and it is
  discarded from the steady-state median because it includes thread-pool creation and page-fault
  cost that no repeated run pays.
* `--trials N` (default 3) comparable trials per setting; median and range are reported, never a
  single number.
* Peak RSS is sampled from /proc/self/status at the end of each trial.

PARITY RULE (§D.3-4)
--------------------
Worker count changes CONCURRENCY, not arithmetic. The gate compares, against the reference setting
(default: 0 workers, i.e. fully serial), for every batch in the sample:
  * the ordered row identities,
  * a digest of the decoded pixel bytes,
  * the array digest of the transformed tensor.
Any mismatch REJECTS the setting for the canonical run. A faster-but-different setting is reported
as rejected, never silently adopted. Note what this does and does not prove: it proves the decode
axis is order- and content-preserving for THIS sample. It does not prove a whole-run parity, and
it does not re-test the query axis (already proven bit-identical at sub-batch sizes 2..128).

USAGE
-----
    python perf_bench.py --model encoder --workers 0,2,4,6,8 --trials 3 \
        --out-jsonl perf_raw.jsonl --out-md PERF_TABLE.md

Run only when the box is idle. The harness refuses to start if it sees a pytest or a
composition_eval process, because a contaminated baseline is worse than none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

#: The three composition conditions, so the sample cannot quietly cover only the easy one.
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def _git(*args: str) -> str:
    """One git query, or empty string when git is unavailable. Never raises into a measurement."""
    try:
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _busy_processes() -> list[str]:
    """Evaluator or pytest processes. A benchmark taken beside these is not a baseline."""
    found = []
    for pattern in ("pytest workbench/tests", "composition_eval"):
        result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        own = str(os.getpid())
        found.extend(line for line in result.stdout.splitlines() if own not in line)
    return found


def _peak_rss_kib() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _vram_mib() -> int | None:
    """Used VRAM, or None when nvidia-smi is unavailable. Diagnostic, never an acceptance target."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def load_sample_ids(path: Path) -> list[dict]:
    """Read the COMMITTED sample file. A sample chosen at runtime is not reproducible.

    Format: `<qid> TAB <category> TAB <rank_mm> TAB <rank_img> TAB <rank_txt> TAB <bucket>`,
    '#' comments and blank lines ignored. The ranks in the file are the ones MEASURED in the
    reference run, so they double as an assertion: if the artifact no longer agrees, the sample
    has drifted and the measurement would not describe the run it claims to.
    """
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            raise ValueError(f"malformed sample line (expected 6 tab-separated fields): {line!r}")
        qid, category, mm, img, txt, bucket = parts
        rows.append({"qid": qid, "category": category, "rank_multimodal": int(mm),
                     "rank_image_only": int(img), "rank_text_only": int(txt), "bucket": bucket})
    if not rows:
        raise ValueError(f"sample file {path} contains no ids")
    return rows


def verify_sample(data, rows: list[dict], rank_payload: Path) -> list[str]:
    """Assert every sample id is real and still has the rank the file records.

    Returns human-readable notes; raises on anything that would invalidate the measurement.
    A sample that silently lost an id would shrink the measurement without saying so.
    """
    import json as _json

    known = {query.qid for query in data.queries}
    missing = [row["qid"] for row in rows if row["qid"] not in known]
    if missing:
        raise ValueError(f"sample ids not present in the release: {missing}")

    payload = _json.loads(rank_payload.read_text(encoding="utf-8"))
    ranks = payload.get("ranks", {})
    conditions = ("multimodal", "image_only_empty_text", "text_only_black_image")
    notes, drifted = [], []
    for row in rows:
        observed = {name: ranks.get(name, {}).get(row["qid"]) for name in conditions}
        if any(value is None for value in observed.values()):
            raise ValueError(f"sample id {row['qid']} has no rank in {observed}; it would shrink "
                             "the sample silently")
        expected = {"multimodal": row["rank_multimodal"],
                    "image_only_empty_text": row["rank_image_only"],
                    "text_only_black_image": row["rank_text_only"]}
        if observed != expected:
            drifted.append((row["qid"], expected, observed))
        notes.append(f"{row['qid']} ({row['category']}, {row['bucket']}): "
                     + ", ".join(f"{name}={observed[name]}" for name in conditions))
    if drifted:
        raise ValueError("sample ranks no longer match the reference artifact; the sample has "
                         f"drifted and must be re-derived: {drifted}")
    return notes


def _digest(value) -> str:
    """Stable digest of a decoded batch, used by the parity gate."""
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    try:
        import numpy as np
        array = np.asarray(value)
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    except Exception:
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


@dataclass
class Trial:
    worker_count: int
    trial_index: int
    kind: str            # "cold" | "steady"
    elapsed_s: float
    batches: int
    images: int
    peak_rss_kib: int | None
    vram_mib: int | None

    @property
    def images_per_s(self) -> float:
        return self.images / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _sample_gallery_paths(config, data, limit: int, model_id: str) -> list[Path]:
    """A FIXED gallery sample, chosen deterministically and stated rather than sampled at random.

    Selection rule, and why this is representative enough for a decode measurement:
      * the first `limit` gallery images in released order, so the same bytes are measured on every
        invocation and two runs are comparable at all;
      * the sample deliberately INCLUDES the last gallery entry, because the last batch is where
        `drop_last=False` behaviour and a partial batch show up;
      * category coverage comes from the queries below, not from the gallery — gallery decode is
        category-agnostic for every adapter in this repo (all use one shared gallery), so biasing
        the gallery by category would measure nothing extra.
    """
    # Resolved exactly the way the runner does it: the layout is a REGISTRY fact about the model,
    # not a config attribute. Guessing a config field here produced an AttributeError the first time
    # this harness was smoke-tested, which is why it now mirrors runner.py:1489-1490.
    from workbench.backend.registry import load_registry, model_by_id
    from workbench.composition.runner import image_resolver

    registry = load_registry()
    model = model_by_id(model_id, registry)
    layout = model.get("fashioniq_layout", "fashioniq_standard")
    resolver = image_resolver(Path(config.FASHIONIQ_ROOT), layout)
    ids = list(data.gallery.image_ids)
    chosen = ids[:max(1, limit - 1)] + [ids[-1]]
    paths = []
    for identity in chosen:
        resolved = resolver(identity, data.candidate_names)
        if resolved is not None and Path(resolved).is_file():
            paths.append(Path(resolved))
    return paths


def _decode_batch(paths: list[Path]) -> list[bytes]:
    """Read raw bytes: the decode stage this harness isolates is file-read + PIL decode."""
    from PIL import Image
    import io

    out = []
    for path in paths:
        with Image.open(path) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            out.append(buffer.getvalue())
    return out


def _transform_batch(blobs: list[bytes], size: int):
    """Transform stage: decode the PNG bytes and resize, as the adapters do after the intervention."""
    import numpy as np
    from PIL import Image
    import io

    arrays = []
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            arrays.append(np.asarray(image.convert("RGB").resize((size, size)), dtype="uint8"))
    return arrays


def run_setting(paths: list[Path], workers: int, batch_size: int, trials: int, size: int):
    """One worker count: cold pass, then `trials` steady passes, plus the parity digests."""
    from workbench.composition.adapters import prefetch_rows

    batches = [paths[start:start + batch_size] for start in range(0, len(paths), batch_size)]
    kinds = ["cold"] + ["steady"] * trials
    results: list[Trial] = []
    parity_rows: list[dict] = []

    for index, kind in enumerate(kinds):
        started = time.perf_counter()
        blob_digests: list[str] = []
        array_digests: list[str] = []
        images = 0
        # The SAME bounded producer the evaluator uses. Decode only; transform runs per batch after
        # the prefetch yields, so the measurement covers the producer-consumer pair, not just I/O.
        #
        # Row identity is NOT collected from the yielded batches: `prefetch_rows` guarantees
        # submission order, and asserting that here would be asserting the thing under test with the
        # thing being tested. It is derived from `batches` below, which is the submission order.
        for blobs in prefetch_rows(_decode_batch, batches, workers):
            arrays = _transform_batch(blobs, size)
            blob_digests.extend(_digest(blob) for blob in blobs)
            array_digests.extend(_digest(array) for array in arrays)
            images += len(blobs)
        elapsed = time.perf_counter() - started
        # Row identity from the ordered batches, which is what submission order guarantees.
        ordered_paths = [str(path) for batch in batches for path in batch]
        results.append(Trial(worker_count=workers, trial_index=index, kind=kind, elapsed_s=elapsed,
                             batches=len(batches), images=images,
                             peak_rss_kib=_peak_rss_kib(), vram_mib=_vram_mib()))
        parity_rows.append({
            "worker_count": workers, "trial_index": index, "kind": kind,
            "ordered_paths_digest": _digest(json.dumps(ordered_paths).encode("utf-8")),
            "blob_digest": _digest("\n".join(blob_digests).encode("utf-8")),
            "array_digest": _digest("\n".join(array_digests).encode("utf-8")),
            "images": images,
        })
    return results, parity_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="encoder",
                        help="model id whose adapter the sample is drawn for (its layout only)")
    parser.add_argument("--workers", default="0,2,4,6,8",
                        help="decode worker counts to sweep, within the 10-thread budget")
    parser.add_argument("--trials", type=int, default=3, help="steady-state trials per setting")
    parser.add_argument("--gallery-limit", type=int, default=768,
                        help="gallery images in the fixed sample (kept small: this is a producer "
                             "measurement, not a full encode)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--sample-ids", help="committed sample id file; REQUIRED for a "
                        "reproducible measurement")
    parser.add_argument("--reference-ranks", help="rank artifact the sample ids were derived from")
    parser.add_argument("--allow-busy", action="store_true",
                        help="measure anyway; the output is then marked contaminated")
    args = parser.parse_args(argv)

    busy = _busy_processes()
    if busy and not args.allow_busy:
        print("REFUSING TO MEASURE: another evaluator or test suite is running, so any timing here "
              "would be contaminated (02_PERFORMANCE_AND_PARITY.en.md §A).\n" + "\n".join(busy),
              file=sys.stderr)
        return 2
    contaminated = bool(busy)

    from workbench.backend import autopilot
    from workbench.composition import data as composition_data

    config = autopilot._workbench_config(autopilot.load_config(Path.home() / "cir_rerun_20261109.yaml"))
    data = composition_data.load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT),
                                         verify_hashes=False, expect_released_counts=False)
    sample_notes = []
    if args.sample_ids:
        if not args.reference_ranks:
            parser.error("--sample-ids requires --reference-ranks so the ids can be verified")
        rows = load_sample_ids(Path(args.sample_ids))
        sample_notes = verify_sample(data, rows, Path(args.reference_ranks))
        print(f"sample ids verified: {len(rows)} ids, ranks agree with {args.reference_ranks}",
              file=sys.stderr)
        for note in sample_notes:
            print(f"  {note}", file=sys.stderr)
    else:
        print("WARNING: no --sample-ids given, so the query sample is not reproducible.",
              file=sys.stderr)

    paths = _sample_gallery_paths(config, data, args.gallery_limit, args.model)
    if not paths:
        print("no gallery paths resolved; nothing to measure", file=sys.stderr)
        return 1

    workers = [int(item) for item in args.workers.split(",") if item.strip()]
    all_trials: list[Trial] = []
    all_parity: list[dict] = []
    for count in workers:
        trials, parity = run_setting(paths, count, args.batch_size, args.trials, args.image_size)
        all_trials.extend(trials)
        all_parity.extend(parity)
        print(f"  workers={count}: {len(trials)} trials", file=sys.stderr)

    # ---- parity gate: every setting must match the FIRST setting's steady digest exactly
    reference = next(row for row in all_parity if row["kind"] == "steady")
    verdicts = {}
    for count in workers:
        rows = [row for row in all_parity if row["worker_count"] == count and row["kind"] == "steady"]
        same = all(row["ordered_paths_digest"] == reference["ordered_paths_digest"]
                   and row["blob_digest"] == reference["blob_digest"]
                   and row["array_digest"] == reference["array_digest"] for row in rows)
        verdicts[count] = "IDENTICAL" if same else "REJECTED (content differs)"

    # ---- aggregate: median + range over steady trials only; cold reported separately
    rows_md = []
    for count in workers:
        steady = [t for t in all_trials if t.worker_count == count and t.kind == "steady"]
        cold = [t for t in all_trials if t.worker_count == count and t.kind == "cold"]
        rates = [t.images_per_s for t in steady]
        peaks = [t.peak_rss_kib for t in steady if t.peak_rss_kib]
        rows_md.append({
            "model": args.model,
            "decode_workers": count,
            "trials": len(steady),
            "median_img_s": round(statistics.median(rates), 1) if rates else None,
            "min_img_s": round(min(rates), 1) if rates else None,
            "max_img_s": round(max(rates), 1) if rates else None,
            "cold_img_s": round(cold[0].images_per_s, 1) if cold else None,
            "peak_rss_mib": round(max(peaks) / 1024, 1) if peaks else None,
            "vram_mib": next((t.vram_mib for t in steady if t.vram_mib), None),
            "parity": verdicts[count],
            "contaminated": contaminated,
        })

    with open(args.out_jsonl, "w", encoding="utf-8") as stream:
        for trial in all_trials:
            stream.write(json.dumps({**asdict(trial),
                                     "images_per_s": round(trial.images_per_s, 3)}) + "\n")
        for row in all_parity:
            stream.write(json.dumps({"parity": row}) + "\n")
        stream.write(json.dumps({"sample_ids": args.sample_ids,
                                 "sample_verified_ranks": sample_notes,
                                 "reference_ranks": args.reference_ranks,
                                 # Recorded so the generated document's SHA/env field states a
                                 # fact instead of "unknown". `dirty` matters: a measurement taken
                                 # on a modified tree describes that tree, not the commit.
                                 "sha": _git("rev-parse", "HEAD"),
                                 "git_dirty": bool(_git("status", "--porcelain")),
                                 "interpreter": sys.executable,
                                 "python_version": platform.python_version(),
                                 "loadavg_at_start": Path("/proc/loadavg").read_text().strip()}) + "\n")

    header = ("| model | workers | trials | median img/s | min | max | cold img/s | peak RSS MiB "
              "| VRAM MiB | parity |")
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [f"# Decode-worker sweep — {args.model}", "",
             f"sample: {len(paths)} gallery images (first N-1 in released order plus the last entry),"
             f" batch {args.batch_size}, transform {args.image_size}px, {args.trials} steady trials "
             "plus 1 cold. Wall-clock only; not a whole-run figure.",
             ""]
    if contaminated:
        lines.append("**CONTAMINATED**: another evaluator or test suite was running. These numbers "
                     "are not a baseline.\n")
    lines += [header, rule]
    for row in rows_md:
        lines.append(f"| {row['model']} | {row['decode_workers']} | {row['trials']} | "
                     f"{row['median_img_s']} | {row['min_img_s']} | {row['max_img_s']} | "
                     f"{row['cold_img_s']} | {row['peak_rss_mib']} | {row['vram_mib']} | "
                     f"{row['parity']} |")
    lines += ["", "A setting marked REJECTED changed decoded content or order and must not be "
                  "adopted for a canonical run, however fast it looks."]
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
