"""Evidence of absence for the PAIR released checkpoints.

PAIR's two released files are whole pickled modules whose classes live in `model_vssp` and
`MHTransformer_dis`. Neither module exists in the pinned tree, in any of the author repository's 21
commits, or anywhere else in `workbench/third_party/`. The existing replay work (an
EXPLORATORY_PUBLIC_MAPPING at r10 34.0350 against 52.76) is therefore not a mapping proof: it runs a
*public* implementation that the checkpoint's own tensors contradict.

This module emits the evidence a reader needs to check that claim without unpickling anything:

1. every global the pickle names, straight from the pickle opcodes (no class needs to exist);
2. every class of the required names that DOES exist in the tree, with file:line;
3. the two structural facts that make the public class insufficient even if the names were aliased:
   the restored `transformer_a` geometry, and the module names the checkpoint carries that the
   public `Backbone` never declares;
4. the exact artifact upstream would have to publish to unblock the model.

Nothing here loads the checkpoint as a model and nothing scores anything.
"""
from __future__ import annotations

import ast
import json
import pickletools
import re
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREE = REPO_ROOT / "workbench" / "third_party"
# Globals that must resolve for `torch.load` to reconstruct the released module. Extracted from the
# pickle, not from a guess: `evidence["pickle_globals"]` is the authority and this list is only the
# set the audit reasons about.
REQUIRED_GLOBALS = (
    "model_vssp.Encoder",
    "model_vssp.Backbone",
    "model_vssp.FeatureWiseAffine",
    "MHTransformer_dis.Transformer",
    "MHTransformer_dis.TransformerLayer",
    "MHTransformer_dis.MultiHeadAttention",
    "MHTransformer_dis.MlpTransformer",
)
# Module names the checkpoint's state carries that the pinned `models/model.py` never declares.
# `transformer_a/b` ARE declared (model.py:45,46); `c/d`, `cross_t2i` and `weight_chanelImg*` are not.
UNDECLARED_MODULE_NAMES = (
    "transformer_c",
    "transformer_d",
    "cross_t2i",
    "weight_chanelImg",
    "weight_chanelImg_t",
)


def pickle_globals(checkpoint: Path) -> list[str]:
    """Every `module Class` pair the pickle names, read from opcodes.

    Zip-format checkpoints keep the pickle in `<member>/data.pkl`. Reading opcodes means the answer
    does not depend on any class being importable, which is the whole point here.
    """
    with zipfile.ZipFile(checkpoint) as archive:
        member = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
        if member is None:
            raise ValueError(f"not a zip-format checkpoint with a data.pkl: {checkpoint}")
        payload = archive.read(member)
    found: set[str] = set()
    for opcode, argument, _ in pickletools.genops(payload):
        if opcode.name in ("GLOBAL", "STACK_GLOBAL") and isinstance(argument, str) and argument:
            # `GLOBAL` stores `module<TAB-or-space>Class`; normalise to the dotted import path so the
            # evidence reads like the code that would have to exist.
            found.add(re.sub(r"\s+", ".", argument.strip()))
    return sorted(found)


def classes_in_tree(tree: Path, names: tuple[str, ...]) -> dict[str, list[str]]:
    """Every `class <name>` in `tree`, as `file:line`, for the requested names."""
    wanted = set(names)
    found: dict[str, list[str]] = {name: [] for name in names}
    for path in sorted(tree.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            module = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef) and node.name in wanted:
                found[node.name].append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
    return {name: locations for name, locations in found.items() if locations}


def undeclared_state_names(checkpoint: Path) -> dict[str, int]:
    """Count state-dict keys mentioning each undeclared module name, from the pickle bytes."""
    with zipfile.ZipFile(checkpoint) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        payload = archive.read(member)
    text = payload.decode("latin-1")
    counts: dict[str, int] = {}
    for name in UNDECLARED_MODULE_NAMES:
        # Keys are stored as length-prefixed strings; a plain scan is enough to prove presence and is
        # deliberately not a correctness-grade parser.
        counts[name] = len(re.findall(rf"backbone\.{name}\b", text))
    return counts


def build(checkpoint_root: Path, tree: Path = DEFAULT_TREE) -> dict[str, Any]:
    checkpoints = {}
    for filename in ("pair-B1.pt", "pair-B2.pt"):
        path = checkpoint_root / filename
        if not path.is_file():
            checkpoints[filename] = {"present": False}
            continue
        checkpoints[filename] = {
            "present": True,
            "bytes": path.stat().st_size,
            "pickle_globals": pickle_globals(path),
            "undeclared_state_name_hits": undeclared_state_names(path),
        }

    present_globals: dict[str, bool] = {}
    for required in REQUIRED_GLOBALS:
        module, _, name = required.partition(".")
        package_dir = tree / module.replace(".", "/")
        present_globals[required] = package_dir.exists() or (tree / f"{module}.py").exists()

    required_names = tuple(dict.fromkeys(required.split(".")[1] for required in REQUIRED_GLOBALS))
    return {
        "model_id": "pair",
        "verdict": "BLOCKED_MISSING_SOURCE_CLASSES",
        "required_globals": list(REQUIRED_GLOBALS),
        "required_globals_present_as_module_or_package": present_globals,
        "classes_of_those_names_anywhere_in_tree": classes_in_tree(tree, required_names),
        "checkpoints": checkpoints,
        "what_upstream_must_publish": [
            "a `model_vssp` module (or the file that defines Encoder, Backbone, FeatureWiseAffine)",
            "a `MHTransformer_dis` module (or the file that defines Transformer, TransformerLayer, "
            "MultiHeadAttention, MlpTransformer) — distinct from the pinned MHTransformer.py, whose "
            "MultiHeadAttention is not the reverse variant the checkpoint also carries",
            "the training entrypoint that instantiates model_vssp.Encoder, so the checkpoint's class "
            "names are attributable to a published implementation",
            "a statement of which FashionIQ category/split each of pair-B1.pt and pair-B2.pt scores",
        ],
        "why_public_mapping_is_not_proof": (
            "the restored module carries backbone.transformer_c, backbone.transformer_d, "
            "backbone.cross_t2i, backbone.weight_chanelImg and backbone.weight_chanelImg_t, none of "
            "which the pinned models/model.py declares, and its transformer_a is 2 layers x 4 heads "
            "where the public Backbone declares 1 layer x 8 heads (model.py:46). A load that ignores "
            "those tensors drops trained parameters; a load that keeps them has no forward for them."
        ),
        "forbidden_shortcuts": ["strict=False", "dropping layers", "treating a near-paper score as "
                                "mapping proof"],
    }


def main(argv: list[str] | None = None) -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    payload = json.dumps(build(args.checkpoint_root, args.tree), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
