"""M-02 fixture probe: does the LIMN resolver refuse when one annotation source is absent/corrupt?

Four synthetic FashionIQ roots, each built from the 7:3038 / 7:3790 shape (same reference stem and
caption pair filed under shirt and under toptee). Nothing is patched: the real
``resolve_query_category`` and ``caption_scope_index`` are called.

Usage: python3 scripts/m02/probe_resolver.py > raw/m02/m02_resolver.txt
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[2]          # workbench/docs/reviews/<audit>
REPO = HERE.parents[3]                              # repository root
sys.path.insert(0, str(REPO))

from workbench.composition.model_limn import (  # noqa: E402
    Adapter,
    caption_scope_index,
    normalize_query_text,
    resolve_query_category,
)

SCOPES = Adapter.category_scopes
STEM = "B000AM"
NAME = f"mbeir_images/fashioniq_images/{STEM}.jpg"
CAPTIONS = ["is darker and has a different graphic", "is darker"]
QUERY_TXT = "Is darker and has a different graphic and is darker"
NAMES = {"7:3038": NAME}
QUERY = SimpleNamespace(qid="7:3038", query_txt=QUERY_TXT, reference_id="7:3038", category=None)


def write_captions(root: Path, category: str, rows: list[dict]) -> None:
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "captions" / f"cap.{category}.val.json").write_text(json.dumps(rows), encoding="utf-8")


def row() -> dict:
    return {"candidate": STEM, "captions": list(CAPTIONS)}


def probe(label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"m02-{label}-") as tmp:
        root = Path(tmp)
        write_captions(root, "shirt", [row()])
        toptee = root / "captions" / "cap.toptee.val.json"
        if label == "a_both_present":
            toptee.write_text(json.dumps([row()]), encoding="utf-8")
        elif label == "b_toptee_absent":
            pass
        elif label == "c_toptee_corrupt":
            toptee.write_text('{"rows": [', encoding="utf-8")
        elif label == "d_shirt_absent":
            toptee.write_text(json.dumps([row()]), encoding="utf-8")
            (root / "captions" / "cap.shirt.val.json").unlink()
        else:
            raise ValueError(label)
        index = caption_scope_index(root)
        key = (STEM, normalize_query_text(QUERY_TXT))
        return {
            "fixture": label,
            "files": sorted(path.name for path in (root / "captions").iterdir()),
            "toptee_bytes": (toptee.read_text(encoding="utf-8") if toptee.is_file() else None),
            "index_entry": (sorted(index[key]) if key in index else None),
            "resolved": resolve_query_category(SCOPES, index, QUERY, NAMES),
        }


def main() -> int:
    out = {
        "module": str(Path("workbench/composition/model_limn.py")),
        "scopes": list(SCOPES),
        "query": {"qid": QUERY.qid, "query_txt": QUERY_TXT, "reference_stem": STEM,
                  "loader_category": None, "normalized_key": normalize_query_text(QUERY_TXT)},
        "fixture_captions": {"shirt": CAPTIONS, "toptee": CAPTIONS},
        "probes": [probe(label) for label in
                   ("a_both_present", "b_toptee_absent", "c_toptee_corrupt", "d_shirt_absent")],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
