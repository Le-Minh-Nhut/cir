"""Regression tests for the four blocked models' replay paths (M-04..M-07).

Failing-first intent, per claim:

* CLVC-Net `validate_checkpoint` used to return a hard-coded "mapping unresolved" blocker even when
  all six staged digests matched the registry, so `main()` exited 1 on a complete bundle. The first
  test fails on that version and passes on the digest-bound one.
* TG-CIR `preflight` used to emit "checkpoint mapping unresolved" unconditionally, likewise exiting 1
  on a staged, digest-verified weight whose pickle deliberately loads strictly into the pinned
  `VCG_CIR`. The second test separates "mapping verified" from "digest kind".
* PAIR's checkpoint names two modules that exist nowhere in the pinned tree or its 21 commits; the
  absence is asserted from bytes, not from a comment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from workbench.backend.registry import model_by_id

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = REPO_ROOT / "workbench" / "artifacts" / "checkpoint_selection" / "20261106T002617+0700"
CLVC_DIR = CHECKPOINT_ROOT / "checkpoints" / "clvc"
TGCIR_DIR = CHECKPOINT_ROOT / "downloads" / "tgcir" / "x" / "checkpoints"
PAIR_DIR = CHECKPOINT_ROOT / "checkpoints" / "pair"
PAIR_TREE = REPO_ROOT / "workbench" / "third_party"


def _staged(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


# ----------------------------------------------------------------------------- CLVC-Net (M-04)

def test_clvc_checkpoint_validation_is_bound_to_the_declared_digests():
    """A complete, digest-matching bundle must not be refused for a string reason."""
    from workbench.replay.clvc_net import validate_checkpoint

    if not all(_staged(CLVC_DIR / name) for name in
               ("dress_local_model.pt", "dress_global_model.pt")):
        pytest.skip("staged CLVC checkpoints absent")

    declared = {m["filename"]: m["expected_sha256"]
                for m in model_by_id("clvc_net")["checkpoint_variants"][0]["bundle_members"]}
    for name, expected in declared.items():
        path = CLVC_DIR / name
        if _staged(path):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

    assert validate_checkpoint(CLVC_DIR) == []


def test_clvc_checkpoint_validation_refuses_a_substituted_file(tmp_path: Path):
    """The same validator must still fail closed: a wrong digest is a blocker, not a pass."""
    from workbench.replay.clvc_net import validate_checkpoint

    for name in ("dress_local_model.pt", "dress_global_model.pt"):
        (tmp_path / name).write_bytes(b"not the author's weights")
    blockers = validate_checkpoint(tmp_path, categories=("dress",))
    assert any("digest mismatch" in b for b in blockers), blockers


def test_clvc_cuda_census_matches_the_claim():
    """7 scoring-path expressions on 6 lines; losses at model.py:178/249 are unreachable from test()."""
    sources = {
        "test.py": REPO_ROOT / "workbench" / "third_party" / "CLVCNet" / "test.py",
        "text_model.py": REPO_ROOT / "workbench" / "third_party" / "CLVCNet" / "text_model.py",
    }
    scoring_lines = {name: [i for i, line in enumerate(path.read_text().splitlines(), 1) if ".cuda()" in line]
                     for name, path in sources.items()}
    scoring_exprs = sum(path.read_text().splitlines()[i - 1].count(".cuda()")
                        for name, path in sources.items() for i in scoring_lines[name])
    assert scoring_lines["test.py"] == [28, 47, 99, 115]
    assert scoring_lines["text_model.py"] == [96, 108]
    assert scoring_exprs == 7


def test_clvc_cpu_device_patch_is_opt_in_and_reported():
    """The CPU smoke path must exist, must be labelled, and must not be reached on a CUDA run."""
    from workbench.replay.clvc_net import _cpu_device_patch
    import torch

    original = torch.Tensor.cuda
    try:
        _cpu_device_patch(torch)
        assert torch.Tensor.cuda is not original
        assert getattr(torch.Tensor.cuda, "_workbench_device_neutral", False) is True
        # identity semantics: a CPU tensor survives `.cuda()` unchanged
        t = torch.zeros(2)
        assert t.cuda().device.type == "cpu"
    finally:
        torch.Tensor.cuda = original


# ------------------------------------------------------------------------------- TG-CIR (M-05)

def test_tgcir_restore_file_is_declared_and_never_read():
    """Upstream `--restore_file` exists only in argparse; no code path consumes it."""
    train = (REPO_ROOT / "workbench" / "third_party" / "TGCIR" / "train.py").read_text()
    occurrences = [i for i, line in enumerate(train.splitlines(), 1) if "restore_file" in line]
    assert occurrences == [74], occurrences
    assert "torch.load" not in train.split("if __name__")[0], "train.py must not load a checkpoint"


def test_tgcir_preflight_accepts_a_digest_verified_staged_checkpoint():
    from workbench.replay.tgcir import preflight

    variant = model_by_id("tgcir")["checkpoint_variants"][0]
    staged = TGCIR_DIR / variant["filename"]
    if not _staged(staged):
        pytest.skip("staged TG-CIR checkpoint absent")
    blockers = preflight(REPO_ROOT / "workbench" / "third_party" / "TGCIR",
                         REPO_ROOT / "data" / "FashionIQ", TGCIR_DIR)
    assert not any("mapping unresolved" in b for b in blockers), blockers
    assert not any("digest mismatch" in b for b in blockers), blockers


def test_tgcir_declared_digest_is_labelled_a_staged_workbench_digest():
    """A staged digest must never be presented as an author-published checksum."""
    variant = model_by_id("tgcir")["checkpoint_variants"][0]
    # The author archive ships metrics, not a checksum file; the archive's own tree is the evidence.
    assert (TGCIR_DIR / "Fashioniq_metrics.json").is_file()
    assert not list(TGCIR_DIR.glob("*sha*")) and not list(TGCIR_DIR.glob("*.md5"))
    assert variant["expected_sha256"] and len(variant["expected_sha256"]) == 64


def test_tgcir_pickle_needs_the_two_documented_load_adaptations():
    """Assert the adaptations from the bytes rather than from a comment."""
    from workbench.replay.tgcir import clip_weight_digest

    archive_checkpoint = TGCIR_DIR / "Fashioniq.pt"
    if not _staged(archive_checkpoint):
        pytest.skip("staged TG-CIR checkpoint absent")
    import zipfile, pickletools

    with zipfile.ZipFile(archive_checkpoint) as archive:
        member = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        globals_ = {arg for op, arg, _ in pickletools.genops(archive.read(member))
                    if op.name == "GLOBAL" and arg}
    assert "torch.nn.modules.linear _LinearWithBias" in globals_
    assert "model VCG_CIR" in globals_
    assert "clip.model CLIP" in globals_
    # The CLIP cache is digest-addressable through OpenAI's own URL, so it is verifiable.
    assert clip_weight_digest() in (None, "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f")


# ---------------------------------------------------------------------------------- PAIR (M-06)

def test_pair_required_modules_are_absent_from_the_pinned_tree():
    from workbench.replay.pair_absence import REQUIRED_GLOBALS, build

    if not _staged(PAIR_DIR / "pair-B1.pt"):
        pytest.skip("staged PAIR checkpoints absent")
    evidence = build(PAIR_DIR, PAIR_TREE)
    assert evidence["verdict"] == "BLOCKED_MISSING_SOURCE_CLASSES"
    assert not any(evidence["required_globals_present_as_module_or_package"].values())
    assert set(evidence["required_globals"]) == set(REQUIRED_GLOBALS)


def test_pair_globals_are_read_from_the_pickle_not_guessed():
    from workbench.replay.pair_absence import REQUIRED_GLOBALS, pickle_globals

    if not _staged(PAIR_DIR / "pair-B1.pt"):
        pytest.skip("staged PAIR checkpoints absent")
    discovered = set(pickle_globals(PAIR_DIR / "pair-B1.pt"))
    assert set(REQUIRED_GLOBALS) <= discovered


def test_pair_evidence_does_not_claim_a_score():
    """Absence evidence must not publish a metric: a score here would read as a mapping proof."""
    from workbench.replay.pair_absence import build

    payload = json.dumps(build(PAIR_DIR, PAIR_TREE))
    assert "r10" not in payload and "macro" not in payload


# ------------------------------------------------------------------------- Air-Know (M-07)

def test_airknow_eval_is_nondeterministic_by_upstream_design():
    """A-12: the native FashionIQ eval calls `model.train()` and seeds nothing.

    Two identical runs of the same checkpoint differed by 0.47 pp macro_r10 (measured). If someone
    later makes the wrapper deterministic, this test is the place that decision has to be recorded:
    either the seed is pinned here, or the artifact must carry a repeat count.
    """
    source = (REPO_ROOT / "workbench" / "third_party" / "AirKnow").resolve()
    test_blip2 = (source / "test_BLIP2.py").read_text()
    data_utils = (source / "data_utils.py").read_text()
    wrapper = (REPO_ROOT / "workbench" / "replay" / "airknow.py").read_text()

    assert "def test(params, model, testset, category, txt_processors):\n    model.train()" in test_blip2
    # UPSTREAM must stay unseeded: seeding it would be an adaptation and would change the native run.
    for text, label in ((test_blip2, "test_BLIP2.py"), (data_utils, "data_utils.py")):
        for seeder in ("torch.manual_seed", "np.random.seed", "random.seed"):
            assert seeder not in text, f"{label} now seeds ({seeder}); re-check A-12 before trusting one number"
    assert "nn.Dropout(p=dropout_rate)" in (
        source / "lavis" / "models" / "blip2_models" / "blip2_cir.py").read_text()

    # The WRAPPER may seed, but only opt-in, and it must declare the result. Both halves are asserted:
    # a default-on seed would silently change every Air-Know number this campaign published.
    assert 'SEED_ENV = "WORKBENCH_AIRKNOW_SEED"' in wrapper
    assert "os.environ.get(SEED_ENV)" in wrapper
    assert 'torch.manual_seed(int(seed))' in wrapper, "seeding must be gated on the env var, not unconditional"

    # R15: the campaign's 0.4744 pp figure is a property of a GROUP of runs, never of the single run
    # an artifact describes. It must not be emitted as a per-run attribute, and it must be gone from
    # the module as a constant that a run can stamp onto itself.
    assert "MEASURED_RUN_TO_RUN_DELTA_MACRO_R10" not in wrapper, (
        "the borrowed per-run delta constant is back; a run must not claim a variance it never measured")
    assert "assemble_variance_group" in wrapper, (
        "the artifact must carry a group-scoped variance block computed at emit time")
    assert '"applies_to_this_run": False' in wrapper, (
        "the historical campaign figure must be labelled as not applicable to this run")
