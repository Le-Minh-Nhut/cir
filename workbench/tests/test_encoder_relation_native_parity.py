"""R03 second half: the NATIVE replay must pin the same relation the composition worker pins.

`workbench/replay/encoder.py::evaluate_categories` used to construct `model_try2.Encoder(...)` with no
seed in front of it, so `learnable_relation` (drawn by `torch.randn` inside `__init__`, absent from
every checkpoint) came from an unpinned global RNG. The CIRCUS pin did not cover that path, so the
historical native numbers could not be reproduced from the code that produced them.

Three claims, all checked against the real module:

1. the seed is applied IMMEDIATELY before `Encoder(...)` (order, measured at run time, not by reading
   source);
2. it is the SAME constant CIRCUS uses -- imported, so the two cannot drift;
3. the drawn tensor and its digest are persisted with the run and returned in the metadata dict.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.composition import model_encoder
from workbench.replay import encoder as replay_encoder


def _evaluate(monkeypatch, tmp_path: Path, *, observation: Path | None):
    """Run the REAL `evaluate_categories` against a stub vendor tree."""
    import torch

    order: list[str] = []

    class Model:
        def __init__(self, **_kwargs):
            self.backbone = SimpleNamespace(binding_decoder=SimpleNamespace(
                learnable_relation=torch.arange(8, dtype=torch.float32).reshape(2, 4)))

        def to(self, _device):
            return self

        def eval(self):
            return None

        def load_state_dict(self, _state, strict=False):
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def manual_seed(seed):
        order.append(f"seed:{seed}")
        return None

    class Dataset:
        def FashionIQ(self, **_kwargs):
            return object()

    for name in ("captions", "image_splits", "resized_image"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    fake_open_clip = SimpleNamespace(create_model_and_transforms=lambda *_a, **_k: (None, None, None))
    fake_model = SimpleNamespace(Encoder=Model)
    monkeypatch.setattr(replay_encoder, "_load_replay_modules",
                        lambda _root: (torch, fake_open_clip, Dataset(), SimpleNamespace(), fake_model))
    monkeypatch.setattr(torch, "manual_seed", manual_seed)
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"weight": 1})
    monkeypatch.setattr(replay_encoder, "_assert_checkpoint_loaded",
                        lambda *_a, **_k: order.append("checkpoint"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if observation is None:
        monkeypatch.delenv("WORKBENCH_OBSERVATION_OUT", raising=False)
    else:
        monkeypatch.setenv("WORKBENCH_OBSERVATION_OUT", str(observation))

    results, aggregate = replay_encoder.evaluate_categories(
        tmp_path, tmp_path, tmp_path, categories=(), device="cpu")
    return order, results, aggregate


def test_the_native_construction_is_pinned_immediately_before_encoder(
        monkeypatch, tmp_path: Path) -> None:
    """Seeding after construction pins nothing: the draw happens inside `Encoder.__init__`."""
    order, _results, aggregate = _evaluate(monkeypatch, tmp_path, observation=None)

    assert order[0] == f"seed:{replay_encoder.RELATION_SEED}", order
    assert aggregate["relation_state"]["status"] == "recorded"


def test_the_native_seed_is_the_composition_constant(monkeypatch, tmp_path: Path) -> None:
    """Imported, not copied: a second literal is how the two paths drift apart."""
    _evaluate(monkeypatch, tmp_path, observation=None)
    assert replay_encoder.RELATION_SEED is model_encoder.RELATION_SEED
    assert replay_encoder.RELATION_EXPECTED_DIGEST == model_encoder.RELATION_EXPECTED_DIGEST


def test_the_drawn_tensor_and_digest_are_persisted_with_the_run(monkeypatch, tmp_path: Path) -> None:
    observation = tmp_path / "run" / "captured-features.npz"
    observation.parent.mkdir(parents=True)
    _order, results, aggregate = _evaluate(monkeypatch, tmp_path, observation=observation)

    state = aggregate["relation_state"]
    expected = hashlib.sha256(
        __import__("numpy").arange(8, dtype="float32").tobytes()).hexdigest()
    assert state["sha256"] == expected
    assert state["matches_pinned_digest"] is False, "8 bytes are not the pinned [2, 512] draw"
    stored = Path(state["path"])
    assert stored.is_file() and stored.name == f"relation-seed{replay_encoder.RELATION_SEED}.npy"
    record = json.loads(Path(str(stored) + ".json").read_text(encoding="utf-8"))
    assert record["sha256"] == expected and record["in_state_dict"] is False
    # The docstring's claim, carried by the artifact itself: a declared policy, not a restoration.
    assert "not a restoration" in record["policy"]
    for result in results:
        assert result.provenance["relation_state"]["sha256"] == expected


def test_the_pin_is_declared_as_policy_not_restoration() -> None:
    """The released checkpoint carries no relation: the docstring must say exactly that."""
    source = Path(replay_encoder.__file__).read_text(encoding="utf-8")
    assert "POLICY, not restoration" in source.split("def evaluate_categories", 1)[1]
    assert "does not contain " in source
    assert "`learnable_relation`, so the drawn tensor is reproducible under this seed" in source


def test_the_seed_call_precedes_the_constructor_in_the_ast() -> None:
    """Static companion to the runtime order check: a refactor cannot quietly swap them."""
    tree = ast.parse(Path(replay_encoder.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_categories")
    seed = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"]
    construction = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "Encoder"]
    assert seed and construction
    assert min(seed) < min(construction), (seed, construction)


@pytest.mark.parametrize("name", ["RELATION_SEED", "RELATION_EXPECTED_DIGEST"])
def test_the_constants_are_imported_not_redeclared(name: str) -> None:
    tree = ast.parse(Path(replay_encoder.__file__).read_text(encoding="utf-8"))
    assigned = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets)]
    assert not assigned, f"{name} must be imported from model_encoder, not redeclared"
