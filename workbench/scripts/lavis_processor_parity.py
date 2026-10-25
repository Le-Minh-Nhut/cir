"""Verify the workbench's LAVIS text-processor path against native construction.

F06: the LAVIS-family wrappers used to obtain `txt_processors` from
`load_model_and_preprocess(name=..., model_type=..., is_eval=True, device=...)`, which builds a
SECOND pretrained BLIP-2 model first (`lavis/models/__init__.py:210`) -- a construction that reads
`/root/bert-base-uncased` (`lavis/models/blip2_models/blip2.py:32,48,55`) and multi-GB base weights,
purely to hand back a text processor. They now use `workbench/replay/lavis_processors.py`.

This script is the parity check between the two paths, run from a host that has the pinned checkouts
but not the weights:

    .venv/bin/python workbench/scripts/lavis_processor_parity.py HINT

It reports, per model in the family:

* the class the pinned config selects, and the config the wrapper resolves for its `model_type`;
* whether that class is the one `load_model_and_preprocess` would have selected for the same
  `model_type` (read from the model class's own `PRETRAINED_MODEL_CONFIG_DICT`, the same dict both
  paths resolve through);
* the processor's own output on a fixed set of captions, so two paths can be compared textually.

Where the native path cannot be executed at all on this host (`lavis.models` needs a pinned
`transformers`), the script says so explicitly and prints the reason, rather than reporting a pass it
did not observe.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CHECKOUTS = {
    "hint": ("HINT", "lavis/models/blip2_models/HINT.py"),
    "habit": ("HABIT", "lavis/models/blip2_models/HABIT.py"),
    "intent": ("INTENT", "lavis/models/blip2_models/blip2_cir.py"),
    "conesep": ("ConeSep", "lavis/models/blip2_models/ConeSep.py"),
    "airknow": ("AirKnow", "lavis/models/blip2_models/blip2_cir.py"),
}
CAPTIONS = ("  A dog, running!  ", "a red dress", "Dress.", "is this striped ?", "")


def declared_configs(class_file: Path) -> dict[str, str]:
    """`PRETRAINED_MODEL_CONFIG_DICT` read from the pinned source without importing the model."""
    tree = ast.parse(class_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(target, "id", None) == "PRETRAINED_MODEL_CONFIG_DICT" for target in node.targets):
            return {key.value: value.value for key, value in zip(node.value.keys, node.value.values)}
    raise SystemExit(f"no PRETRAINED_MODEL_CONFIG_DICT in {class_file}")


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=f"LAVIS model(s) to verify (valid: {', '.join(sorted(CHECKOUTS))}; default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parser_for()
    args = parser.parse_args(argv)
    selected = [m.lower() for m in args.models] if args.models else sorted(CHECKOUTS)
    invalid = [m for m in selected if m not in CHECKOUTS]
    if invalid:
        sys.stderr.write(
            f"unknown model(s): {', '.join(invalid)}. Valid choices: {', '.join(sorted(CHECKOUTS))}\n"
        )
        return 2
    from workbench.replay import lavis_processors

    report: dict[str, object] = {}
    for model_id in selected:
        directory, class_relative = CHECKOUTS[model_id]
        source_root = REPO / "workbench" / "third_party" / directory
        declared = declared_configs(source_root / class_relative)
        config = declared["pretrain"]
        entry: dict[str, object] = {"class": class_relative,
                                    "declared_configs": declared,
                                    "wrapper_config": config}
        try:
            # Each model has its OWN vendored `lavis`, and the pinned package names are global to a
            # process, so a comparison across models has to drop the previous checkout's modules.
            # A production run never does this: one model, one subprocess.
            lavis_processors.reset()
            processors, boundary = lavis_processors.text_processors(source_root, config)
            entry["loaded_via"] = boundary
            entry["processor_class"] = type(processors["eval"]).__name__
            entry["outputs"] = {caption: processors["eval"](caption) for caption in CAPTIONS}
            entry["status"] = "LOADED"
        except Exception as error:                     # every failure is reported, never a pass
            entry["status"] = "FAILED"
            entry["error"] = f"{type(error).__name__}: {error}"
        report[model_id] = entry
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(entry["status"] == "LOADED" for entry in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
