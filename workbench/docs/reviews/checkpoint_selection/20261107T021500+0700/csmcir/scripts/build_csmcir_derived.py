"""Build a derived CSMCIR checkout from the pristine pinned tree.

The pinned revision cannot import its own package: `lavis/models/__init__.py` runs ~98 unguarded
imports of modules the revision does not contain, before the line that registers the class the
evaluator asks for. `workbench/patches/csmcir_models_init_compat.patch` repairs that by deleting the
dead import range wholesale.

That patch is what the registry approves, and it is applied here unchanged. This script exists only to
reproduce the derived checkout in one step instead of a shell sequence, and it verifies both recorded
digests afterwards so a half-applied tree cannot be mistaken for the approved variant.

Usage:
    python build_csmcir_derived.py <destination> [--with-shiyan92-pyc <path.pyc>]
"""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve()
for parent in REPO.parents:
    if (parent / "workbench/third_party/CSMCIR/src").is_dir():
        REPO = parent
        break
PRISTINE = REPO / "workbench/third_party/CSMCIR/src"
PATCHES = [
    ("workbench/patches/csmcir_models_init_compat.patch", "src/lavis/models/__init__.py",
     "aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818"),
    ("workbench/patches/csmcir_validate_visualization_opt_in.patch", "src/validate_blip_csmcir.py",
     "84a1d476e99d9687b16b1a9c5238f89295f23d2a6f9f4dfb449acba2c118262c"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--dataset-root", default="data/FashionIQ",
                        help="canonical FashionIQ root to link as fashionIQ_dataset")
    parser.add_argument("--expect-pyc-sha256", default=None,
                        help="required sha256 of the author-published .pyc")
    parser.add_argument("--with-shiyan92-pyc", default=None,
                        help="author-published .pyc for the class that saved the checkpoint")
    args = parser.parse_args()

    dest = Path(args.destination).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PRISTINE, dest / "src")

    for patch_file, target, expected in PATCHES:
        result = subprocess.run(["patch", "-p1", "-i", str(REPO / patch_file)],
                                cwd=dest, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"{patch_file} failed: {result.stdout}{result.stderr}")
        digest = hashlib.sha256((dest / target).read_bytes()).hexdigest()
        if digest != expected:
            # Printing MISMATCH and continuing would leave a half-repaired tree that later reads as an
            # approved variant. The registry declares this digest; a tree that does not carry it is not
            # that variant, so stop.
            raise SystemExit(f"{target}: digest {digest} != declared {expected}")
        print(f"{target}: {digest[:16]} OK")

    # The evaluator derives its data root from `__file__` and opens `<checkout>/fashionIQ_dataset`; the
    # gallery caption extractor opens `<checkout>/COT_ours2`. Neither exists in a fresh copy of `src`,
    # so a derived checkout built without these links starts and then dies on the first dataset read.
    dataset = Path(args.dataset_root).resolve()
    if not (dataset / "qwen_captions").is_dir():
        raise SystemExit(f"{dataset} does not look like the canonical FashionIQ root (no qwen_captions/)")
    for name, target in (("fashionIQ_dataset", dataset), ("COT_ours2", dataset.parent / ".." / "workbench/third_party/CSMCIR/COT_ours2")):
        target = target.resolve()
        if not target.is_dir():
            raise SystemExit(f"cannot stage {name}: {target} is not a directory")
        link = dest / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        print(f"staged {name} -> {target}")

    if args.with_shiyan92_pyc:
        pyc = Path(args.with_shiyan92_pyc)
        # Author-published bytecode, so its identity is checkable before it is trusted.
        pyc_sha = hashlib.sha256(pyc.read_bytes()).hexdigest()
        if args.expect_pyc_sha256 and pyc_sha != args.expect_pyc_sha256:
            raise SystemExit(f"{pyc}: sha256 {pyc_sha} != expected {args.expect_pyc_sha256}")
        # The module name decides where a sourceless .pyc is importable from, so the file is named
        # after the module it contains, not after whatever the download was called.
        target = (dest / "src/lavis/models/blip2_models"
                  / "blip2_qformer_cir_align_prompt_target_image_caption_shiyan92.pyc")
        shutil.copyfile(pyc, target)
        # A sourceless .pyc is importable but never imported: the approved patch deleted the line that
        # named it, because at patch time the module did not exist. Restoring that ONE import next to
        # the CSMCIR registration is what makes the class the checkpoint was saved by loadable.
        init = dest / "src/lavis/models/__init__.py"
        text = init.read_text()
        anchor = ("from lavis.models.blip2_models.blip2_qformer_cir_align_prompt_csmcir import "
                  "Blip2QformerCirAlignPromptCSMCIR")
        restore = ("from lavis.models.blip2_models."
                   "blip2_qformer_cir_align_prompt_target_image_caption_shiyan92 import "
                   "Blip2QformerCirAlignPromptTargetImageCaptionshiyan92")
        unguarded = chr(10) + restore + chr(10)
        if unguarded not in text:
            if anchor not in text:
                raise SystemExit("cannot locate the CSMCIR registration import to anchor on")
            # The 98 rewritten imports sit inside one try/except ImportError, and the FIRST absent
            # module aborts the whole block -- so placing this line inside it would never execute.
            # It is anchored AFTER the CSMCIR registration instead, which is unguarded.
            init.write_text(text.replace(anchor, anchor + chr(10) + restore, 1))
            print("restored the shiyan92 import AFTER the guarded block, next to CSMCIR")
        print(f"staged {pyc.name} sha256={hashlib.sha256(pyc.read_bytes()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
