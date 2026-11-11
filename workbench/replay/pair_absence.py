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

Presence is decided the way an import would be decided, not by a path guess. `workbench.replay.pair`
puts four directories on `sys.path` (the pinned source root and its `data/`, `models/` and `utils/`
children), so a bare `import model_vssp` resolves against `PAIR/models/`, not against `PAIR/`. This
module reads those roots out of `pair.py` and resolves each required module along them with
`importlib.util.find_spec`, then checks each required class name against the resolved file's AST. A
module that exists but lacks a class is a different verdict from a module that is absent, because the
first is repairable upstream and the second is a release gap.
"""
from __future__ import annotations

import ast
import importlib.util
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


def replay_import_roots(source_root: Path, pair_source: Path | None = None) -> dict[str, Any]:
    """The directories `workbench.replay.pair` puts on `sys.path`, read from that module.

    `_load_replay_modules` inserts the pinned source root and its `data/`, `models/` and `utils/`
    children. Those entries are the whole search space a bare `import model_vssp` sees, so they are
    extracted from the file rather than assumed: the audit's original path guess (`tree / module`)
    was wrong precisely because `PAIR/models/` is the root that matters.
    """
    pair_source = pair_source or Path(__file__).with_name("pair.py")
    relative: list[str] = []
    error: str | None = None
    try:
        module = ast.parse(pair_source.read_text(encoding="utf-8"))
        function = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_replay_modules"
        )
        loop = next(node for node in ast.walk(function) if isinstance(node, ast.For))
        if not isinstance(loop.iter, (ast.Tuple, ast.List)):
            raise ValueError("sys.path entries are not a literal tuple")
        for element in loop.iter.elts:
            if not (isinstance(element, ast.Call) and element.args):
                raise ValueError("unexpected sys.path entry form")
            argument = element.args[0]
            if isinstance(argument, ast.Name):
                relative.append("")
            elif (
                isinstance(argument, ast.BinOp)
                and isinstance(argument.op, ast.Div)
                and isinstance(argument.left, ast.Name)
                and isinstance(argument.right, ast.Constant)
                and isinstance(argument.right.value, str)
            ):
                relative.append(argument.right.value)
            else:
                raise ValueError(f"unrecognised sys.path entry: {ast.dump(argument)[:80]}")
    except (OSError, SyntaxError, StopIteration, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "pair_source": pair_source.relative_to(REPO_ROOT).as_posix()
        if pair_source.is_relative_to(REPO_ROOT) else str(pair_source),
        "relative_entries": relative,
        "roots": [str(source_root / entry) if entry else str(source_root) for entry in relative],
        "error": error,
    }


def _module_spec(module: str, roots: list[str]):
    """Resolve `module` against exactly `roots`, the way `import` would with those sys.path entries.

    `PathFinder` is the same finder the import system consults; it searches only the paths given and
    does not execute the module, so this stays an evidence tool rather than a loader.
    """
    from importlib.machinery import PathFinder

    return PathFinder.find_spec(module, roots)


def _defined_names(path: Path) -> tuple[set[str], str | None]:
    """Top-level names a module would expose: class definitions and simple assignments."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError) as exc:
        return set(), f"{type(exc).__name__}: {exc}"
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names, None


def source_class_evidence(source_root: Path, pair_source: Path | None = None) -> dict[str, Any]:
    """Resolve every required global the way the replay imports it, then check the class exists.

    Two independent facts per global: the module's resolved file (or `not_found`), and whether that
    file defines the class. A module that imports but lacks the class is repairable; a module that is
    absent is a release gap. Both are reported, and the verdict is derived from them.
    """
    roots_info = replay_import_roots(source_root, pair_source)
    roots = roots_info["roots"]

    per_global: dict[str, Any] = {}
    for required in REQUIRED_GLOBALS:
        module, _, name = required.partition(".")
        candidates = [
            str(Path(root) / f"{module}.py") for root in roots
        ] + [str(Path(root) / module / "__init__.py") for root in roots]
        spec = _module_spec(module, roots) if roots else None
        origin = getattr(spec, "origin", None) if spec is not None else None
        resolved = bool(origin) and Path(origin).is_file()
        defined: set[str] = set()
        parse_error: str | None = None
        if resolved:
            defined, parse_error = _defined_names(Path(origin))
        per_global[required] = {
            "module": module,
            "class": name,
            "module_file": origin if resolved else "not_found",
            "module_resolved": resolved,
            "candidate_paths": candidates,
            "class_defined_in_module": (name in defined) if resolved else False,
            "parse_error": parse_error,
        }

    missing_modules = sorted({
        entry["module"] for entry in per_global.values() if not entry["module_resolved"]
    })
    missing_classes = sorted(
        f"{entry['module']}.{entry['class']}"
        for entry in per_global.values()
        if entry["module_resolved"] and not entry["class_defined_in_module"]
    )

    if missing_modules:
        # An absent module cannot be imported at all, so nothing downstream is reachable.
        verdict = "BLOCKED_MISSING_SOURCE_CLASSES"
        reason = "module(s) not importable from the replay's own sys.path roots: " + ", ".join(missing_modules)
    elif missing_classes:
        verdict = "BLOCKED_INCOMPLETE_SOURCE"
        reason = "module(s) import, but these class(es) are not defined there: " + ", ".join(missing_classes)
    else:
        verdict = "SOURCE_CLASSES_PRESENT"
        reason = "all required classes are defined in modules the replay can import"

    return {
        "verdict": verdict,
        "reason": reason,
        "import_roots_searched": roots,
        "import_roots_source": roots_info,
        "required_globals": per_global,
        "missing_modules": missing_modules,
        "missing_classes": missing_classes,
        # What this module does NOT check, so SOURCE_CLASSES_PRESENT cannot be read as "PAIR works".
        "still_unproven_when_present": [
            "that the released checkpoints' pickle restores against these classes (no unpickling here)",
            "checkpoint state-dict key/shape agreement with these definitions",
            "that the classes' forward passes accept the checkpoint's tensors",
            "which FashionIQ category/split pair-B1.pt and pair-B2.pt each score",
            "any ranking or metric: this module never scores",
        ],
    }



def undeclared_state_names(checkpoint: Path) -> dict[str, int]:
    """Count state-dict keys naming each undeclared submodule, read from the pickle opcodes.

    The names are stored as standalone BINUNICODE keys (`transformer_c`), not as the dotted path
    `backbone.transformer_c`: the parent `backbone` key appears once, and each submodule key once
    beneath it. A regex over the decoded bytes for `backbone.<name>` therefore always returns 0 even
    when the submodule is present, which is what this function used to do. Reading the opcode string
    arguments is both exact and independent of pickling layout.
    """
    with zipfile.ZipFile(checkpoint) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        payload = archive.read(member)
    keys = {argument for opcode, argument, _ in pickletools.genops(payload)
            if isinstance(argument, str)}
    return {name: (1 if name in keys else 0) for name in UNDECLARED_MODULE_NAMES}


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

    source = source_class_evidence(tree / "PAIR")
    required_names = tuple(dict.fromkeys(required.split(".")[1] for required in REQUIRED_GLOBALS))
    return {
        "model_id": "pair",
        "verdict": source["verdict"],
        "verdict_reason": source["reason"],
        "source_classes": source,
        "required_globals": list(REQUIRED_GLOBALS),
        # Kept for continuity, but now labelled for what it is: a check against the tree ROOT, which
        # is not the directory the replay imports from. `source_classes` is the authoritative answer.
        "required_globals_present_as_module_or_package": present_globals,
        "required_globals_present_as_module_or_package_note": (
            "searched the tree root only; `workbench.replay.pair` also puts PAIR/models/, PAIR/data/ "
            "and PAIR/utils/ on sys.path, so a false here does not mean the module is unimportable. "
            "Use source_classes.required_globals for the resolved answer."
        ),
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
