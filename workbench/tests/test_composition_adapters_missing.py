"""Regression tests for the composition adapters added for previously adapter-less models.

WHAT THIS FILE CAN AND CANNOT PROVE
-----------------------------------
These adapters are registered and their scoring contract is exercised on synthetic tensors, but
**none has been run against real model output**: this lane has no GPU and the input/forward parity
gates (G1/G2) are NOT_RUN for every model. So the tests here assert the properties that are
checkable without a forward pass and deliberately do NOT assert that any adapter "works":

* registration is real (the registry row and the imported class agree);
* the class is the contract the runner dispatches through;
* ``status()`` tells the truth about readiness and, where it reports not-ready, names a reason;
* scoring reproduces the upstream expression it claims to reproduce.

THE CONDITION-INPUT QUESTION, AND WHY THE ASSERTION IS NOT AT THE ADAPTER
-----------------------------------------------------------------------
``runner._stage_query_path`` calls ``inputs.adapt_query_inputs(image, text, condition)`` and hands
the adapter an already-adapted ``(path, text)`` pair. By the time an adapter is called the three
conditions are three ordinary inputs; the adapter has no way to know which condition produced
them, and no business changing them. Asserting "the three conditions are pairwise different"
against an adapter would therefore test nothing — the adapter never sees a condition name.

The regression that caught LIMN belongs one layer earlier, at the function that actually performs
the intervention, so it is asserted against ``adapt_query_inputs`` directly below. The
adapter-side complement is the *second* assertion: an adapter must not re-derive or rewrite the
texts it is given, because that is how a second collapse could be introduced downstream of the
first one. Both are in this file so the pair cannot drift apart.
"""
from __future__ import annotations

from importlib import import_module

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from workbench.composition.adapters import CompositionAdapter, adapter_class
from workbench.composition.inputs import CONDITIONS, adapt_query_inputs
from workbench.backend.registry import checkpoint_path, load_registry

#: Adapters added for models that previously had none. ``airknow`` is included because it was
#: added in the same programme and has the same "not verified on GPU" status. The last three
#: (``tgcir``, ``clvc_net``, ``dcnet``) carry a blocker no adapter can remove, so what is asserted
#: for them is a registered, machine-readable REFUSAL plus the scoring expression the pinned source
#: does define (or, for ``dcnet``, an explicit refusal to guess one).
NEW_ADAPTERS = {
    "airknow": "workbench.composition.model_airknow",
    "pair": "workbench.composition.model_pair",
    "combiner_rn50x4_noft": "workbench.composition.model_combiner_rn50x4_noft",
    "clip4cir_rn50x4_fullft": "workbench.composition.model_clip4cir_rn50x4_fullft",
    "tgcir": "workbench.composition.model_tgcir",
    "clvc_net": "workbench.composition.model_clvc_net",
    "dcnet": "workbench.composition.model_dcnet",
}

#: Feature width each adapter's ``score()`` consumes, taken from its own declaration. ``tgcir`` is
#: 512 (``model.py:48-49``), ``clvc_net`` 2048 (two concatenated 1024-d halves,
#: ``test.py:30,48,100,116``), ``dcnet`` 1024 (``experts.ce_shared_dim`` in the staged config).
DIMENSIONS = {"airknow": 256, "pair": 1024,
              "combiner_rn50x4_noft": 640, "clip4cir_rn50x4_fullft": 640,
              "tgcir": 512, "clvc_net": 2048, "dcnet": 1024}

def _model_row(model_id: str) -> dict:
    return next(row for row in load_registry()["models"] if row["model_id"] == model_id)

@pytest.mark.parametrize("model_id", sorted(NEW_ADAPTERS))
def test_adapter_is_registered_in_the_registry(model_id: str) -> None:
    """The registry row must name the module, and the module must be importable under that name."""
    row = _model_row(model_id)
    declared = row.get("composition_adapter")
    assert declared == NEW_ADAPTERS[model_id], (
        f"{model_id} declares composition_adapter={declared!r}, expected {NEW_ADAPTERS[model_id]!r}")
    # A registry entry pointing at a module that does not import is worse than no entry: the
    # dispatcher raises at run time instead of reporting an honest blocker.
    module = import_module(declared)
    assert module.__name__ == declared

@pytest.mark.parametrize("model_id", sorted(NEW_ADAPTERS))
def test_module_exports_the_adapter_contract(model_id: str) -> None:
    """``ADAPTER`` must be the class, subclass the base, and agree with the registry id."""
    module = import_module(NEW_ADAPTERS[model_id])
    assert module.ADAPTER is module.Adapter
    assert isinstance(module.Adapter, type)
    assert issubclass(module.Adapter, CompositionAdapter)
    assert module.Adapter.model_id == _model_row(model_id)["model_id"]
    # The dispatcher's own resolver must accept it -- that is what the runner calls.
    assert adapter_class(model_id) is module.Adapter

@pytest.mark.parametrize("model_id", sorted(NEW_ADAPTERS))
def test_status_is_truthful_and_reports_every_mode(model_id: str) -> None:
    """``status()`` must answer without a GPU, and a not-ready verdict must name a reason."""
    status = adapter_class(model_id).status()
    assert isinstance(status.ready, bool)
    if status.ready:
        assert status.reason is None
    else:
        # Not-ready is the expected state for every adapter here; an unexplained False would be
        # indistinguishable from a bug, so the reason is part of the contract.
        assert status.reason and status.reason.strip()
    assert set(status.per_mode) == set(CONDITIONS)
    assert status.checkpoint_ids, f"{model_id} reports no checkpoint ids"

#: Models whose readiness is now MEASURED, not hoped for, so the snapshot below no longer covers
#: them. Evidence, both `raw/verify/89-adapted-runtime-and-gpu-capability.md`:
#:
#: * `clvc_net` — tier `GPU_VERIFICATION_PENDING` from the registry's `adapted_runtime`, promoted to
#:   `GPU_RUNTIME_READY` by the authorized probe (`compute_capability` 12.0, RTX 5070 Ti), so
#:   `require_launch_ready` returns the interpreter; parity measured in
#:   `raw/verify/87-clvc-composition.md` §5.
#: * `tgcir` — the same tier progression on `tgcir39-la`; parity measured in
#:   `raw/verify/88-tgcir-composition.md` §3.
#:
#: ``status.ready`` for these two is a statement about the OPERATOR's provisioned state (interpreter
#: in `workbench.env`, checkpoint staged under `WORKBENCH_CHECKPOINT_ROOT`), which is why the test
#: below reads that file explicitly: the suite-wide fixture hides operator state, and a readiness
#: claim that needed it must say so rather than assert the opposite.
READY_WITH_EVIDENCE = {"clvc_net", "tgcir"}

@pytest.mark.parametrize("model_id", sorted(set(NEW_ADAPTERS) - READY_WITH_EVIDENCE))
def test_status_refuses_rather_than_claiming_readiness(model_id: str) -> None:
    """Pin the CURRENT honest state: these five are not ready, for a named reason.

    This assertion is intentionally a snapshot of an unverified state, not a quality target. When
    a variant genuinely becomes runnable (interpreter provisioned, mapping verified, GPU forward
    parity measured) this test must be updated *with that evidence*, never deleted to make CI green.
    ``clvc_net`` and ``tgcir`` left this set that way — see ``READY_WITH_EVIDENCE``.
    """
    status = adapter_class(model_id).status()
    assert status.ready is False, (
        f"{model_id} now reports ready=True; if that is real, update this test and cite the "
        "GPU/parity evidence that justifies it")

@pytest.mark.parametrize("model_id", sorted(READY_WITH_EVIDENCE))
def test_status_reports_the_measured_readiness(model_id: str, monkeypatch) -> None:
    """The two models measured ready are asserted ready, against the operator's own provisioning.

    Skipped, never weakened, when that provisioning is absent (a fresh clone has no staged
    checkpoint and no `workbench.env`): the claim being pinned is about provisioned operator state,
    and there is nothing honest to assert about it on a machine that has none. The evidence and the
    measured tier are named in ``READY_WITH_EVIDENCE``.
    """
    from workbench.backend import model_settings, operator_config

    operator_file = Path(__file__).resolve().parents[1] / "config" / "workbench.env"
    if not operator_file.is_file():
        pytest.skip("operator workbench.env absent")
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", operator_file)
    model_settings.file_settings.cache_clear()
    config = operator_config.resolve_config()
    model = _model_row(model_id)
    checkpoint = checkpoint_path(model_id, model["checkpoint_variants"][0],
                                 config.WORKBENCH_CHECKPOINT_ROOT)
    if not checkpoint.exists():
        pytest.skip(f"checkpoint not staged: {checkpoint}")

    status = adapter_class(model_id).status(config)
    assert status.ready is True, (
        f"{model_id} is measured ready (raw/verify/89-adapted-runtime-and-gpu-capability.md) but "
        f"status() refuses: {status.reason}")

@pytest.mark.parametrize("model_id", sorted(NEW_ADAPTERS))
def test_score_reproduces_the_declared_expression(model_id: str) -> None:
    """Scoring is checked against an INDEPENDENT reimplementation, not against itself.

    ``airknow`` maxes over 32 target tokens (``test_BLIP2.py:142-144``); ``pair`` and the two
    CLIP4Cir variants are plain cosine (``PAIR/test.py:57-60``; ``CLIP4Cir/src/validate.py:41,44``).
    ``tgcir`` (``TGCIR/test.py:53-61``) and ``clvc_net`` (``CLVCNet/test.py:122-129``) are the same
    plain cosine over their own widths; ``dcnet`` must REFUSE because its reported ranking fuses a
    per-query correction branch (``DCNet/trainer/trainer_joint.py:207-240``).
    """
    adapter = adapter_class(model_id).__new__(adapter_class(model_id))
    width = DIMENSIONS[model_id]
    rng = np.random.default_rng(0)
    queries = rng.random((4, width), dtype=np.float32)
    if model_id == "dcnet":
        # DCNet's reported ranking is not a score over two fixed matrices
        # (``trainer_joint.py:207-218,239-240``), so the adapter must REFUSE by name instead of
        # returning a cosine and letting it be labelled DCNet.
        from workbench.composition.model_dcnet import ScoringNotReproducible

        with pytest.raises(ScoringNotReproducible, match="correction"):
            adapter.score(queries, rng.random((6, width), dtype=np.float32))
    elif model_id == "airknow":
        tokens = rng.random((6, 32, width), dtype=np.float32)
        scores = adapter.score(queries, tokens.reshape(6, 32 * width))
        expected = np.array([
            [max(float(queries[i] @ tokens[m][g]) for g in range(32)) for m in range(6)]
            for i in range(4)], dtype=np.float32)
        np.testing.assert_allclose(scores, expected, atol=1e-5)
        # The flat and token-major forms must be the same input.
        np.testing.assert_allclose(scores, adapter.score(queries, tokens), atol=1e-6)
    else:
        targets = rng.random((6, width), dtype=np.float32)
        scores = adapter.score(queries, targets)
        q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        t = targets / np.linalg.norm(targets, axis=1, keepdims=True)
        np.testing.assert_allclose(scores, q @ t.T, atol=1e-6)
        # A cosine matrix is bounded by 1 and has no token axis to max over.
        assert scores.shape == (4, 6)
        assert float(scores.max()) <= 1.0 + 1e-5

def test_condition_intervention_is_pairwise_different_at_the_inputs_layer() -> None:
    """The LIMN regression, asserted where the intervention actually happens.

    Three conditions must produce three distinct (image, text) pairs. If any two coincide, a
    model is being asked the same question three times and the "composition gap" is an artefact
    of the harness, not a property of the model. The adapter never sees a condition name, so this
    cannot be checked there.
    """
    reference = Image.new("RGB", (7, 5), (17, 29, 41))
    edit = "make the sleeves shorter"
    pairs = {condition: adapt_query_inputs(reference, edit, condition) for condition in CONDITIONS}

    # Text axis: multimodal and text-only share the edit; image-only must be empty.
    assert pairs["multimodal"][1] == edit
    assert pairs["text_only_black_image"][1] == edit
    assert pairs["image_only_empty_text"][1] == ""

    # Image axis: multimodal and image-only share the reference; text-only must be blacked out.
    assert pairs["multimodal"][0].getbbox() is not None
    assert pairs["image_only_empty_text"][0].getbbox() is not None
    assert pairs["text_only_black_image"][0].getbbox() is None
    assert pairs["text_only_black_image"][0].size == reference.size

    # The actual pairwise claim: no two conditions produce the same (pixels, text) pair.
    def signature(image: Image.Image, text: str) -> tuple:
        return (image.size, image.tobytes(), text)

    signatures = {condition: signature(*pair) for condition, pair in pairs.items()}
    assert len(set(signatures.values())) == len(CONDITIONS), (
        f"two composition conditions collapsed to the same input: {signatures}")

@pytest.mark.parametrize("model_id", sorted(NEW_ADAPTERS))
def test_adapter_does_not_rewrite_the_texts_it_is_given(model_id: str, monkeypatch, tmp_path) -> None:
    """The adapter-side complement: no second condition collapse downstream of the first.

    The adapter receives already-adapted inputs and must transport them verbatim. A rewrite here
    (e.g. substituting a default caption when the text is empty) would silently turn
    ``image_only_empty_text`` back into a text query, undoing the inputs-layer guarantee above.
    """
    module = import_module(NEW_ADAPTERS[model_id])
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path / model_id
    seen: list[list[str]] = []

    def fake_run(_adapter, operation, image_paths, texts, categories):
        seen.append(list(texts))
        return np.zeros((len(image_paths), DIMENSIONS[model_id]), dtype=np.float32)

    # The transport lives in the module that DEFINES ``encode_queries``, which for the two
    # CLIP4Cir variants is the shared implementation, not the thin registry-facing module.
    owner = sys.modules[module.Adapter.encode_queries.__globals__["__name__"]]
    monkeypatch.setattr(owner, "_run", fake_run, raising=False)
    images = [str(tmp_path / f"image-{index}.png") for index in range(3)]
    sent = ["", "an edit", "another edit"]
    adapter.encode_queries(images, sent, [None, None, None])
    assert seen, "encode_queries did not reach the transport layer"
    assert seen[-1] == sent, (
        f"{model_id} rewrote the query texts: sent {sent!r}, transported {seen[-1]!r}")
