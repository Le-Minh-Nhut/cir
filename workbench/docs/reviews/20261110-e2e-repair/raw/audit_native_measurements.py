"""Census of checkpoint_selection measurement files that carry numeric macro metrics.

Why this script exists: the metrics are nested under `aggregate` (and, for a few files, only in
`per_category`/`categories`), NOT at the top level. A scan that reads `d["macro_r10"]` finds ZERO
files and would "disprove" every native run in the tree. That is the same class of trap as the
`?q=` FALSE-NULL problem: the selector is wrong, not the data.

Run:
    workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/raw/audit_native_measurements.py

Output: raw/native_measurement_census.txt
"""
import glob
import json
import os

os.chdir("/home/heheboiz/data/cir")

PATS = ["workbench/artifacts/checkpoint_selection/**/measurements/*.json",
        "workbench/artifacts/checkpoint_selection/**/cohort/measurements/*.json"]

ACTIVE = ["csmcir", "airknow", "conesep", "ptha_mtst", "habit", "intent", "hint", "encoder", "pair",
          "clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc",
          "limn", "tme"]


def num(value):
    return value if isinstance(value, (int, float)) else None


def main() -> int:
    files = sorted({f for p in PATS for f in glob.glob(p, recursive=True)})
    rows = []
    for path in files:
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        agg = d.get("aggregate") if isinstance(d.get("aggregate"), dict) else {}
        r10 = num(agg.get("macro_r10"))
        r50 = num(agg.get("macro_r50"))
        if r10 is None and r50 is None:
            continue
        # native_evaluator is per-category, never at the top level.
        native_evaluator = None
        cats = d.get("categories")
        items = cats.values() if isinstance(cats, dict) else (cats or [])
        for c in items:
            if isinstance(c, dict):
                prov = c.get("provenance") or {}
                if prov.get("native_evaluator"):
                    native_evaluator = prov["native_evaluator"]
                    break
        rows.append(dict(file=os.path.basename(path), model_id=d.get("model_id"),
                         protocol=d.get("protocol"), replay_type=d.get("replay_type"),
                         r10=r10, r50=r50,
                         complete=d.get("complete") if d.get("complete") is not None
                         else agg.get("complete"),
                         native_evaluator=native_evaluator))

    lines = ["# Census: measurement files carrying numeric macro metrics",
             "# metrics are nested under `aggregate`; a top-level scan finds 0 files",
             f"# matched files: {len(files)} | with numeric macro: {len(rows)}", ""]
    for r in sorted(rows, key=lambda x: (x["model_id"] or "", x["file"])):
        lines.append(f"{r['file']:42s} mid={r['model_id']!s:12s} proto={r['protocol']!s:34s} "
                     f"replay={r['replay_type']!s:34s} r10={r['r10']!s:22s} r50={r['r50']!s:22s} "
                     f"complete={r['complete']!s:6s} native_evaluator={r['native_evaluator']!r}")
    with_macro = sorted({r["model_id"] for r in rows if r["model_id"]})
    lines += ["", f"Q1 distinct model_id with numeric macro_r10 ({len(with_macro)}): {with_macro}",
              f"Q2 active configs with NO such file: {[m for m in ACTIVE if m not in with_macro]}"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
