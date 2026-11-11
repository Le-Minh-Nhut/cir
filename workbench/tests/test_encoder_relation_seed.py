"""ENCODER's one unseeded draw: `BindingDecoder.learnable_relation` must be pinned, in the right order.

Upstream (`workbench/third_party/ENCODER/model_try2.py:93`) draws `nn.Parameter(torch.randn(N_p,
dim)).cuda()`. `.cuda()` on a Parameter returns a plain tensor, so the attribute is never registered:
it is absent from `state_dict()` (415 entries) and from the released checkpoint (417 keys, two of
which the model never reads). Nothing seeds it, and it enters BOTH representations -- the query via
`extract_retrieval_compose` and the gallery via `extract_retrieval_target`.

Measured on HEAD before this test existed (`raw/m01/m01_unseeded_construct.py`, two clean processes):
unseeded constructions differ by max |Δ| 4.11787748336792 and 4.744387149810791 across runs, and
4.923079013824463 in the earlier probe; the same seed reproduces the tensor exactly (max |Δ| 0.0,
digest 20db0c27…ce92 in both processes). Two published-run artifacts (`5dee7425…`, `6a528e877…`)
differ in 2,819 of 6,003 multimodal ranks for that reason.

These checks are static on purpose: the vendor needs CUDA and `open_clip`, so the ordering can only
be asserted without running it. The measured digests live in the review's raw evidence.
"""
from __future__ import annotations

import ast
from pathlib import Path

from workbench.composition import model_encoder

SOURCE = Path(model_encoder.__file__).read_text(encoding="utf-8")


def _worker_main() -> ast.FunctionDef:
    tree = ast.parse(SOURCE)
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "_worker_main")


def _index_of_call(function: ast.FunctionDef, attribute: str) -> int:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == attribute:
            return node.lineno
    raise AssertionError(f"no {attribute}(...) call in the worker")


def test_the_draw_is_seeded_before_the_model_is_constructed() -> None:
    """Seeding after construction would pin nothing: the draw happens in `Encoder.__init__`."""
    worker = _worker_main()
    seed_line = _index_of_call(worker, "manual_seed")
    construction = [node.lineno for node in ast.walk(worker)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Encoder"]
    assert construction, "the worker must construct the vendor Encoder"
    assert seed_line < min(construction), (
        f"manual_seed is at line {seed_line}, after Encoder(...) at {min(construction)}; the relation "
        "is drawn inside the constructor, so that order pins nothing")


def test_the_seed_is_carried_by_the_job_not_a_module_default() -> None:
    """A module-level default would be inherited silently by any other caller of the worker."""
    assert isinstance(model_encoder.RELATION_SEED, int)
    assert '"relation_seed": RELATION_SEED' in SOURCE
    assert '"relation_seed"' in SOURCE.split("def _worker_main")[1] or "job[\"relation_seed\"]" in SOURCE


def test_the_drawn_tensor_is_persisted_with_its_digest() -> None:
    """`state_dict()` cannot carry it, so the bytes plus a digest are the only provenance."""
    worker = SOURCE.split("def _worker_main", 1)[1]
    assert "sha256" in worker and "relation.npy" in worker or "relation-seed" in worker
    assert '"in_state_dict": False' in worker
    for field in ("shape", "dtype", "seed", "sha256", "seed_placement"):
        assert f'"{field}"' in worker, f"the persisted record must state {field}"


def test_a_different_draw_is_visible_in_the_reply_not_silent() -> None:
    """The worker reports the digest it drew, so a run can be checked against its own record."""
    assert "relation_digest=relation_digest" in SOURCE
    assert model_encoder.RELATION_EXPECTED_DIGEST and len(model_encoder.RELATION_EXPECTED_DIGEST) == 64

def test_the_handshake_phase_does_no_image_work() -> None:
    """R03: the phase that feeds the run identity constructs the model and answers, nothing more.

    A handshake that encoded anything would make the identity depend on an encode call -- the exact
    ordering the defect had -- and would also run a forward pass before the run directory exists.
    """
    source = SOURCE.split("def run_one", 1)[1]
    branch = source.split('current.get("phase") == HANDSHAKE_PHASE', 1)[1].split("return", 1)[0]
    assert "relation_digest=relation_digest" in branch
    for forbidden in ("Image.open", "extract_retrieval", "prefetch_rows", "np.save"):
        assert forbidden not in branch, f"the handshake branch must not call {forbidden}"
