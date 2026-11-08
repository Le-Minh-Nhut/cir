"""Structural fingerprint of each native checkpoint, asserted from the bytes.

Why this exists: `checkpoint_mapping_status: UNVERIFIED` on the native cohort was a claim that
nobody had established which architecture produced the staged file. That question is answerable
from the file itself - a `torch.save` archive carries the *pickle globals* that name the classes
that built it - and this file is the runnable form of that answer.

Deliberately cheap and dependency-free: it reads only the archive's `data.pkl` member name and its
byte strings, never a storage shard, never torch, never CUDA. A 2.7 GB checkpoint costs a few KB of
I/O. The argument for `assert <literal> in data` rather than a schema: the assertion IS the finding,
and the finding is exactly "these bytes name this class".

Set `WORKBENCH_CKPT_MAPPING_ROOT` to run against another staging session.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

SESSION = Path("workbench/artifacts/checkpoint_selection/20261106T002617+0700")
ROOT = Path(os.environ.get("WORKBENCH_CKPT_MAPPING_ROOT") or SESSION)
CHECKPOINTS = ROOT / "checkpoints"

pytestmark = pytest.mark.skipif(not CHECKPOINTS.is_dir(),
                                reason=f"checkpoint staging session absent: {CHECKPOINTS}")


def pickle_stream(path: Path) -> tuple[bytes, str]:
    """The `data.pkl` body and the archive member that holds it."""
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        return archive.read(member), member


@pytest.mark.parametrize("relative, expected_module, expected_class", [
    # CLVC-Net saves WHOLE models (`train.py:274-275`), so the module/class pair is the one the
    # pinned source defines and the evaluator rebuilds before loading.
    ("clvc/dress_local_model.pt", "model", "compose_local"),
    ("clvc/dress_global_model.pt", "model", "compose_global"),
    ("clvc/shirt_local_model.pt", "model", "compose_local"),
    ("clvc/shirt_global_model.pt", "model", "compose_global"),
    ("clvc/toptee_local_model.pt", "model", "compose_local"),
    ("clvc/toptee_global_model.pt", "model", "compose_global"),
    # TG-CIR saves a whole model too (`train.py:271`), and `model.py:147` names the class.
    ("../downloads/tgcir/x/checkpoints/Fashioniq.pt", "model", "VCG_CIR"),
])
def test_whole_model_pickles_name_the_pinned_class(relative, expected_module, expected_class) -> None:
    stream, member = pickle_stream((ROOT / "checkpoints" / relative).resolve())
    assert member.endswith("data.pkl")
    # A SHORT_BINUNICODE pickle opcode is 0x8c followed by a length byte, so the module and the
    # class appear as adjacent length-prefixed strings. Both must be present for the identity to
    # be the one asserted, which is why the module is checked as well as the class name.
    assert expected_module.encode() in stream
    assert expected_class.encode() in stream


def test_pair_pickles_name_modules_the_pinned_repo_does_not_ship() -> None:
    """The contradiction, stated as an assertion.

    `workbench/third_party/PAIR` ships `model.py`, `MHTransformer.py`, `MHTransformer_rev.py`. The
    released pickles additionally name `model_vssp` and `MHTransformer_dis`. A rebuild that binds
    those names to the public classes loads NON-strictly (`74` unexpected keys, `70` shape
    mismatches, `5.20 %` of the parameters unreachable from the pinned forward), so the pinned
    class is not the class that produced these weights.
    """
    stream, _ = pickle_stream(CHECKPOINTS / "pair" / "pair-B1.pt")
    for absent in (b"model_vssp", b"MHTransformer_dis"):
        assert absent in stream
    pinned = Path("workbench/third_party/PAIR/models")
    assert (pinned / "model.py").is_file()
    assert not (pinned / "model_vssp.py").exists()
    assert not (pinned / "MHTransformer_dis.py").exists()


def test_clip4cir_states_are_state_dicts_not_modules() -> None:
    """`validate.py:306` reads `state_dict["Combiner"]`; `:285` reads `saved_state_dict["CLIP"]`."""
    combiner, _ = pickle_stream(CHECKPOINTS / "clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt")
    assert b"Combiner" in combiner and b"epoch" in combiner
    # A bare state dict names no MODEL CLASS - only torch's tensor rebuilders and the parameter
    # names. `collections.OrderedDict` is the only global, so nothing here says which class built
    # these tensors; that claim comes from the shapes and the strict load instead.
    assert b"collections\nOrderedDict" in combiner
    assert b"torch.nn" not in combiner
    clip_state, _ = pickle_stream(CHECKPOINTS / "clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt")
    assert b"CLIP" in clip_state


def test_sprc_vitl_is_a_state_dict_keyed_by_class_name() -> None:
    """`blip_validate.py:49` indexes the checkpoint by `blip_model.__class__.__name__`."""
    stream, _ = pickle_stream(CHECKPOINTS / "sprc/sprc_fiq_vitl.pt")
    assert b"Blip2QformerCirAlignPrompt" in stream
    assert b"epoch" in stream


def test_every_confirmed_artifact_is_staged_at_the_declared_size() -> None:
    """Sizes are the cheap half of provenance; the digests are recorded in the review document."""
    expected = {
        "clvc/dress_local_model.pt": 217168235,
        "clvc/dress_global_model.pt": 176437419,
        "clvc/shirt_local_model.pt": 218603243,
        "clvc/shirt_global_model.pt": 177872427,
        "clvc/toptee_local_model.pt": 218308843,
        "clvc/toptee_global_model.pt": 177578027,
        "clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt": 236017921,
        "clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt": 236017921,
        "clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt": 713874695,
        "dcnet/trained_model.pth": 2877310172,
        # The staged run directory's config is the edited copy (data_dir, num_workers, save_dir);
        # `downloads/dcnet/config.json` is the author's 1982-byte original.
        "dcnet/config.json": 1734,
        "sprc/sprc_fiq_vitl.pt": 1459756541,
    }
    for relative, size in expected.items():
        path = CHECKPOINTS / relative
        assert path.is_file(), relative
        assert path.stat().st_size == size, relative
