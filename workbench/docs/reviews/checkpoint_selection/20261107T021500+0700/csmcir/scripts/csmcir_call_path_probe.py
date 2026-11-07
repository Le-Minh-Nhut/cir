"""Prove which caption files the CSMCIR FashionIQ evaluation path actually opens.

The report and the audit disagreed: the report called `COT_ours2/bert_captions/` and
`captions/keywords_in_mods_*.json` blockers, the audit called their only reader a dead class. Grep
shows `FashionIQDataset_imageedit` defined once and instantiated nowhere, but a text search is not a
call path. This script patches `builtins.open` and records every path opened while the real
constructors named by `blip_validate_fashioniq` run, so the answer comes from execution.

It needs the pinned checkout on `sys.path` and nothing else: no GPU, no checkpoint, no model. That is
deliberate -- the question is which files the dataset layer reads, and loading a 2.7 GB BLIP-2 only
makes the answer slower to obtain.

This file ships inside the evidence bundle, where `parents[1]` is not a repository root: resolving the
checkout relative to `__file__` pointed at
`.../csmcir/third_party/CSMCIR`, which does not exist. The root is therefore located by walking up
until `workbench/third_party/CSMCIR/src` is found, and the checkout can be overridden with
`--checkout` for a derived tree.
"""
import builtins
import json
import sys
import traceback
from pathlib import Path

def _find_checkout(override: str | None) -> Path:
    if override:
        candidate = Path(override).resolve()
        if not (candidate / "src/data_utils_csmcir.py").is_file():
            raise SystemExit(f"--checkout {candidate} does not contain src/data_utils_csmcir.py")
        return candidate
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "workbench/third_party/CSMCIR"
        if (candidate / "src/data_utils_csmcir.py").is_file():
            return candidate
    raise SystemExit("could not locate workbench/third_party/CSMCIR above this file; pass --checkout")


CHECKOUT = _find_checkout(None)
sys.path.insert(0, str(CHECKOUT / "src"))

opened: list[tuple[str, str]] = []
_real_open = builtins.open


def _record(file, *args, **kwargs):
    stack = traceback.extract_stack()[-4:-1]
    origin = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(stack))
    opened.append((str(file), origin))
    return _real_open(file, *args, **kwargs)


def main():
    from data_utils_csmcir import FashionIQDataset  # noqa: E402

    # The evaluator builds one relative and one classic dataset per category
    # (`validate_blip_csmcir.py:596,598`); 'classic' is what the gallery feature extractor consumes.
    builtins.open = _record
    try:
        for category in ("dress", "shirt", "toptee"):
            for mode in ("relative", "classic"):
                FashionIQDataset("val", [category], mode, lambda im: im)
    finally:
        builtins.open = _real_open

    # The gallery caption features are extracted by `extract_index_blip_caption_features`, which
    # opens `COT_ours2/fashioniq/{cat}_cot_val.json` after building a DataLoader (construction reads
    # no images). Run it with an inert model: every caption file is opened before the first forward,
    # and the AttributeError it then raises is the expected stopping point, recorded as such.
    from utils_csmcir import extract_index_blip_caption_features  # noqa: E402

    gallery_notes = []
    builtins.open = _record
    try:
        for category in ("dress", "shirt", "toptee"):
            dataset = FashionIQDataset("val", [category], "classic", lambda im: im)
            try:
                extract_index_blip_caption_features(dataset, object(), save_memory=False)
            except Exception as error:  # noqa: BLE001 - the stop point is the evidence
                gallery_notes.append(f"{category}: stopped at {type(error).__name__}")
    finally:
        builtins.open = _real_open

    reads = sorted({p for p, _ in opened})
    print(json.dumps({
        "distinct_files_opened": len(reads),
        "files": reads,
        "imageedit_instantiated": any("imageedit" in o for _, o in opened),
        "reads_bert_captions": [p for p in reads if "bert_captions" in p],
        "reads_keywords_in_mods": [p for p in reads if "keywords_in_mods" in p],
        "reads_cot_ours2": [p for p in reads if "COT_ours2" in p],
        "gallery_stop_points": gallery_notes,
    }, indent=1))


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--checkout", default=None,
                         help="CSMCIR checkout root (default: search upwards for workbench/third_party/CSMCIR)")
    _args = _parser.parse_args()
    CHECKOUT = _find_checkout(_args.checkout)
    sys.path.insert(0, str(CHECKOUT / "src"))
    main()
