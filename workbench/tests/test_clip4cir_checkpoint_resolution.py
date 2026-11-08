"""The CLIP4Cir adapters must name the files the registry declares -- nothing else.

Why this exists: `CLIP4CirFullftAdapter.command` used to spell its two members
`combiner_state.pt` / `clip_state.pt`. No file on disk has either name; the release ships
`fiq_comb_RN50x4_fullft.pt` / `fiq_clip_RN50x4_fullft.pt`, which is what `models.yaml`
declares in `required_files`/`bundle_members`. The mismatch made a complete, hash-verified
pair read as two non-existent paths, and because the failure surfaced as a missing argument
to `validate.py`, it was easy to blame on the Python environment instead.

The test resolves through `registry.checkpoint_path()` (never a naive `<root>/<model_id>/<name>`
join: 17 of 29 variants relocate via `checkpoint_base_relative` and 10 resolve only through the
fallback) and then asks the adapter for the command it would build. Both answers must agree with
the registry declaration.

No GPU, no model forward. The structural check reads only the torch archive's `data.pkl`
member, not any storage shard, so a 700 MB checkpoint costs a few KB of I/O.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from workbench.backend import runtime
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import CLIP4CirFullftAdapter, CombinerNoftAdapter
from workbench.backend.registry import (
    checkpoint_by_id,
    checkpoint_path,
    model_by_id,
)

FULLFT = "clip4cir_rn50x4_fullft"
NOFT = "combiner_rn50x4_noft"
COMBINER = "fiq_comb_RN50x4_fullft.pt"
CLIP = "fiq_clip_RN50x4_fullft.pt"


@pytest.fixture(autouse=True)
def _no_interpreter_lookup(monkeypatch):
    """The command is what is under test, not whether this host has a CLIP4Cir env.

    `python_executable` deliberately refuses an unverified interpreter; stubbing it keeps the
    assertion on the paths and never claims the environment is ready.
    """
    monkeypatch.setattr(runtime, "python_executable", lambda _model: "/usr/bin/python3")


def _variant(model_id: str) -> dict:
    model = model_by_id(model_id)
    assert len(model["checkpoint_variants"]) == 1, (
        f"{model_id} is expected to declare exactly one variant; the resolution asserted here "
        "is that variant's")
    return model["checkpoint_variants"][0]


def _request(model_id: str, checkpoint_id: str, tmp_path: Path) -> EvalRequest:
    return EvalRequest(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        protocol_id=model_by_id(model_id)["supported_protocols"][0],
        dataset_root=tmp_path / "fashioniq",
        output_path=tmp_path / "out.json",
    )


def _stage(tmp_path: Path, model_id: str, names: list[str]) -> Path:
    """A root that satisfies `checkpoint_path`'s conventional branch, holding the declared names."""
    root = tmp_path / "checkpoints"
    directory = root / model_id
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"staged-for-resolution-test")
    return root


def _flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_fullft_resolves_to_the_registry_declared_members(tmp_path: Path) -> None:
    variant = _variant(FULLFT)
    assert variant["artifact_type"] in {"bundle_directory", "bundle_placeholder"}
    assert variant["required_files"] == [COMBINER, CLIP]
    assert [member["filename"] for member in variant["bundle_members"]] == [COMBINER, CLIP]
    assert {member["filename"]: member["role"] for member in variant["bundle_members"]} == {
        COMBINER: "combiner", CLIP: "clip"}

    root = _stage(tmp_path, FULLFT, variant["required_files"])
    resolved = checkpoint_path(FULLFT, variant, root)

    assert resolved == root / FULLFT, (
        "a bundle_directory resolves to the directory holding its members")
    assert resolved.is_dir()
    assert sorted(path.name for path in resolved.iterdir()) == sorted(variant["required_files"])

    adapter = CLIP4CirFullftAdapter()
    command = adapter.command(
        source=tmp_path / "CLIP4Cir", checkpoint=resolved,
        request=_request(FULLFT, variant["checkpoint_id"], tmp_path))

    combiner = Path(_flag_value(command, "--combiner-path"))
    clip = Path(_flag_value(command, "--clip-model-path"))
    assert combiner == resolved / COMBINER, (
        "--combiner-path must name the declared Combiner member, not an adapter-invented "
        f"spelling; got {str(combiner)!r}")
    assert clip == resolved / CLIP, (
        f"--clip-model-path must name the declared CLIP member; got {str(clip)!r}")
    assert combiner.is_file() and clip.is_file()


def test_noft_resolves_to_the_registry_declared_file(tmp_path: Path) -> None:
    variant = _variant(NOFT)
    assert variant["artifact_type"] == "file"
    declared = variant["filename"]
    assert declared == "fiq_comb_RN50x4_noft.pt"

    root = _stage(tmp_path, NOFT, [declared])
    resolved = checkpoint_path(NOFT, variant, root)
    assert resolved == root / NOFT / declared
    assert resolved.is_file()

    adapter = CombinerNoftAdapter()
    command = adapter.command(
        source=tmp_path / "CLIP4Cir", checkpoint=resolved,
        request=_request(NOFT, variant["checkpoint_id"], tmp_path))

    assert Path(_flag_value(command, "--combiner-path")) == resolved, (
        "the noft evaluator takes the one declared file, resolved by the registry")


def test_adapter_does_not_spell_members_itself() -> None:
    """Guard the regression at its source: the adapter must not hard-code a member name.

    Comments are stripped first: the fix's own explanation names the old literals, and a guard
    that a comment can fail would push the next author to delete the reason.
    """
    import inspect
    import io
    import tokenize

    source = inspect.getsource(CLIP4CirFullftAdapter.command)
    code = "".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT)
    assert "combiner_state.pt" not in code
    assert "clip_state.pt" not in code
    assert "_declared_member_paths" in code, (
        "the adapter must resolve its two member paths from the registry declaration, not "
        "spell filenames itself")
    # The resolver is where the declaration is read; assert that too, so moving the lookup into a
    # helper that guesses cannot satisfy the guard.
    resolver = inspect.getsource(
        __import__("workbench.backend.adapters.models", fromlist=["_declared_member_paths"])
        ._declared_member_paths)
    assert "bundle_members" in resolver and "required_files" in resolver


# --------------------------------------------------------------------------- real staging session
#: The session whose `cohort/downloads` root the native rows were staged against. The
#: `checkpoint_base_relative` declarations are relative to THAT root, so resolving from any
#: other root is a different question (and the wrong one for this staging session).
SESSION = Path("workbench/artifacts/checkpoint_selection/20261106T002617+0700")
STAGING_ROOT = SESSION / "cohort" / "downloads"
_HAS_STAGING = (STAGING_ROOT / ".." / ".." / "checkpoints" / "clip4cir" /
                "FashionIQ" / "RN50x4_fullft").resolve().is_dir()


def _real_path(model_id: str) -> Path:
    return checkpoint_path(model_id, _variant(model_id), STAGING_ROOT)


@pytest.mark.skipif(not _HAS_STAGING, reason="checkpoint staging session absent")
def test_staged_fullft_pair_matches_the_declaration_and_the_adapter() -> None:
    variant = _variant(FULLFT)
    resolved = _real_path(FULLFT)
    assert resolved.is_dir()
    for name in variant["required_files"]:
        assert (resolved / name).is_file(), f"declared member absent on disk: {resolved / name}"

    adapter = CLIP4CirFullftAdapter()
    command = adapter.command(
        source=Path("workbench/third_party/CLIP4Cir"), checkpoint=resolved,
        request=_request(FULLFT, variant["checkpoint_id"], Path("/tmp")))
    assert Path(_flag_value(command, "--combiner-path")) == resolved / COMBINER
    assert Path(_flag_value(command, "--clip-model-path")) == resolved / CLIP


@pytest.mark.skipif(not _HAS_STAGING, reason="checkpoint staging session absent")
def test_staged_members_carry_the_declared_roles_structurally() -> None:
    """Cheap structural identity: the archive's pickle names the role the registry assigns.

    The Combiner member must name `Combiner` and not `CLIP`; the CLIP member the reverse. This
    reads `data.pkl` only -- never a storage shard, never torch, never CUDA -- so it cannot be
    mistaken for a forward pass and costs a few KB per file.
    """
    variant = _variant(FULLFT)
    roles = {member["filename"]: member.get("role") for member in variant["bundle_members"]}
    assert roles == {COMBINER: "combiner", CLIP: "clip"}

    resolved = _real_path(FULLFT)
    with zipfile.ZipFile(resolved / COMBINER) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        combiner_stream = archive.read(member)
    with zipfile.ZipFile(resolved / CLIP) as archive:
        member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        clip_stream = archive.read(member)

    assert b"Combiner" in combiner_stream
    assert b"CLIP" not in combiner_stream
    assert b"CLIP" in clip_stream
    assert b"Combiner" not in clip_stream
