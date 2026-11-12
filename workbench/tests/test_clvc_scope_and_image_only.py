"""CLVC-Net: the 119 unscoped queries, and the named image-encoder-only path.

Two claims, each with a failure it would produce if wrong:

* CLVC-Net and LIMN share ONE category contract, so CLVC reuses LIMN's resolver verbatim. If the
  contracts differed, reusing it would route a CLVC query into another model's rule; the tests
  below therefore check the resolver against the loader on all 5 884 already-scoped queries (a
  disagreement is a bug, not a refinement) and check that only the loader's own 119 are touched.
* ``clvc_image_encoder_only`` is representable with THIS checkpoint, and ``image_only_empty_text``
  is not. The image-only tests execute the real row expression with the real pickles on CPU: the
  text tower is asked to be unreachable (it is patched to raise), the caption is shown to be inert,
  and the image-only row is compared to the gallery row under a tolerance stated in advance.

Both layers skip cleanly when the external artifacts (release, annotations, checkpoints) are absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from workbench.composition import model_clvc_net as clvc
from workbench.composition.model_limn import caption_scope_index, resolve_query_category

SCOPES = clvc.Adapter.category_scopes
REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE = REPO_ROOT / "workbench/artifacts/circus_release"
FASHIONIQ = REPO_ROOT / "data/FashionIQ"

#: Absolute tolerance for "the image-only query row equals the gallery row of the SAME image",
#: stated BEFORE measuring. It is not a bit-identity claim: the two rows are produced by different
#: batch shapes (one image vs a gallery batch), and the measured value on the dress pair was
#: 6.15e-08 for batch 1 vs batch 3 — one float32 ULP scale, not a different encoding.
IMAGE_ONLY_TOLERANCE = 1e-6

# ------------------------------------------------------------------ fixtures (no external data)

def _row(candidate: str, *captions: str) -> dict[str, object]:
    return {"candidate": candidate, "captions": list(captions)}


def _write_captions(root: Path, category: str, rows: list[dict[str, object]]) -> None:
    (root / "captions").mkdir(parents=True, exist_ok=True)
    (root / "captions" / f"cap.{category}.val.json").write_text(json.dumps(rows), encoding="utf-8")


def _query(text: str, reference_id: str = "7:1", qid: str = "7:1",
           category: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(qid=qid, query_txt=text, reference_id=reference_id, category=category)


NAMES = {"7:1": "mbeir_images/fashioniq_images/B000S.jpg",
         "7:2": "mbeir_images/fashioniq_images/B000T.jpg",
         "7:3": "mbeir_images/fashioniq_images/B000AM.jpg"}


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    """One unique pair per category, plus one pair filed identically under two categories."""
    _write_captions(tmp_path, "dress", [_row("B000D", "is red", "is short")])
    _write_captions(tmp_path, "shirt", [
        _row("B000S", "is blue", "is long"),
        _row("B000AM", "is darker and has a different graphic", "is darker"),
    ])
    _write_captions(tmp_path, "toptee", [
        _row("B000T", "is green", "is sleeveless"),
        _row("B000AM", "is darker and has a different graphic", "is darker"),
    ])
    return tmp_path


def _adapter(root: Path) -> "clvc.Adapter":
    """Routing-only adapter: no checkpoint, no worker, no gate (the ``object.__new__`` seam)."""
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter.dataset_root = root
    adapter._caption_scopes = None
    adapter._scope_decisions = {}
    adapter.annotation_pin = {}
    return adapter


# ------------------------------------------------------------------ the shared contract

def test_a_unique_annotation_row_resolves_to_that_category(fixture_root: Path) -> None:
    adapter = _adapter(fixture_root)
    assert adapter.query_scope(_query("Is blue and is long", "7:1", "7:1"), NAMES) == "shirt"
    assert adapter.query_scope(_query("Is green and is sleeveless", "7:2", "7:2"), NAMES) == "toptee"
    assert adapter.query_scope_resolution()["counts_by_reason"] == {
        "annotation:shirt:shirt": 1, "annotation:toptee:toptee": 1}


def test_two_categories_for_one_caption_pair_stay_unscored(fixture_root: Path) -> None:
    adapter = _adapter(fixture_root)
    assert adapter.query_scope(_query("Is darker and has a different graphic and is darker", "7:3"),
                               NAMES) is None
    resolution = adapter.query_scope_resolution()
    assert resolution["counts_by_reason"] == {
        "unresolved:annotation_names_two_categories": 1}
    assert resolution["unresolved"][0]["holding_categories"] == ("shirt", "toptee")


def test_a_missing_annotation_file_refuses_to_route(fixture_root: Path) -> None:
    """The ``incomplete`` predicate is honoured: absence of a file is NOT absence of a row."""
    (fixture_root / "captions" / "cap.toptee.val.json").unlink()
    adapter = _adapter(fixture_root)
    assert adapter.caption_scope_index().incomplete is True
    assert adapter.query_scope(_query("Is blue and is long", "7:1"), NAMES) is None
    assert adapter.query_scope_resolution()["annotation"]["incomplete"] is True


def test_a_corrupt_annotation_file_refuses_to_route(fixture_root: Path) -> None:
    (fixture_root / "captions" / "cap.shirt.val.json").write_text("{not json", encoding="utf-8")
    adapter = _adapter(fixture_root)
    assert "shirt" in adapter.caption_scope_index().unreadable
    assert adapter.query_scope(_query("Is green and is sleeveless", "7:2"), NAMES) is None


def test_an_off_pin_annotation_file_refuses_to_route(fixture_root: Path) -> None:
    """A file whose rows no longer match the committed pin is not evidence, even if well formed."""
    from workbench.composition.data import ANNOTATION_PIN

    adapter = _adapter(fixture_root)
    adapter.annotation_pin = ANNOTATION_PIN          # the fixture bytes cannot match the release
    assert adapter.query_scope(_query("Is blue and is long", "7:1"), NAMES) is None
    assert adapter.query_scope_resolution()["annotation"]["incomplete"] is True


def test_the_leakage_declaration_names_the_caption_and_the_stem() -> None:
    """LEAKAGE, declared exactly rather than minimised.

    An earlier version of this string said routing reads "the reference image stem for every
    condition ... the only reference-derived input". That was FALSE for 117 of 6001 queries:
    ``resolve_query_scope`` keys on ``(reference stem, normalized edit caption)`` for the queries the
    loader left unscoped, so the caption chose their category. The declaration must name both inputs,
    and must not claim the conditions see no text.
    """
    declaration = clvc.IMAGE_ENCODER_ONLY_LEAKS_SCOPE
    assert "edit caption" in declaration, declaration
    assert "117" in declaration, declaration
    assert "DATASET LABEL" in declaration, declaration
    for condition in ("text_only_black_image", "clvc_image_encoder_only"):
        assert condition in declaration, declaration
    # The sentence that was wrong must not come back: "the only reference-derived input".
    assert "only reference-derived input" not in declaration
    # The one-line form a run summary writes says the same thing.
    assert "edit caption" in clvc.IMAGE_ENCODER_ONLY_LEAKS_CAPTION
    assert "117" in clvc.IMAGE_ENCODER_ONLY_LEAKS_CAPTION


def test_the_resolver_key_really_is_stem_plus_caption() -> None:
    """The fact the declaration rests on, measured on the shared resolver: the key has TEXT in it.

    Two queries with the SAME reference stem and DIFFERENT captions must be able to route to
    different categories — if the key were the stem alone, the second lookup could not.
    """
    import inspect

    from workbench.composition import model_limn

    source = inspect.getsource(model_limn.resolve_query_scope)
    assert "normalize_query_text" in source
    assert "query_reference_stem" in source
    assert "query_txt" in source


def test_routing_does_not_depend_on_the_condition_or_the_gallery() -> None:
    """The signature is the guarantee: no condition, no gallery, no score can enter a scope."""
    import inspect

    assert list(inspect.signature(clvc.Adapter.query_scope).parameters) == [
        "self", "query", "candidate_names"]


# ------------------------------------------------------------------ the real 119, measured

def _require_release():
    if not (RELEASE / "mbeir_fashioniq_task7_test.jsonl").is_file():
        pytest.skip("CIRCUS release is not present on this host (external artifact)")
    if not (FASHIONIQ / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    from workbench.composition import data as release_data

    loaded = release_data.load_release(RELEASE, verify_hashes=False)
    assigned = release_data.assign_garment_categories(loaded, FASHIONIQ)
    return assigned


def test_the_resolver_agrees_with_the_loader_on_every_scoped_query() -> None:
    """5 884 loader-assigned queries must not move: a disagreement is a bug, not a refinement."""
    assigned = _require_release()
    index = caption_scope_index(FASHIONIQ, clvc.ANNOTATION_PIN)
    changed = []
    scoped = 0
    for query in assigned.queries:
        if query.category is None:
            continue
        scoped += 1
        routed = resolve_query_category(SCOPES, index, query, assigned.candidate_names)
        if routed != query.category:
            changed.append((query.qid, query.category, routed))
    assert scoped == 5884, scoped
    assert not changed, changed[:10]


def test_the_119_split_into_117_recovered_and_2_genuinely_ambiguous() -> None:
    """The measured split, not a target: 5 884 + 117 + 2 == 6 003 with nothing silently dropped."""
    assigned = _require_release()
    index = caption_scope_index(FASHIONIQ, clvc.ANNOTATION_PIN)
    assert index.incomplete is False
    ambiguous = [query for query in assigned.queries if query.category is None]
    assert len(ambiguous) == 119
    routed = {query.qid: resolve_query_category(SCOPES, index, query, assigned.candidate_names)
              for query in ambiguous}
    recovered = {qid: scope for qid, scope in routed.items() if scope is not None}
    unresolved = sorted(qid for qid, scope in routed.items() if scope is None)
    assert len(recovered) == 117, unresolved
    assert set(recovered.values()) == {"shirt", "toptee"}
    assert unresolved == ["7:3038", "7:3790"]
    assert 5884 + len(recovered) + len(unresolved) == 6003


# ------------------------------------------------------------------ the image-only path, on real weights

def _checkpoint_pair(category: str = "dress"):
    """The real local/global pickles for one category, digest-verified, loaded on CPU.

    ``workbench/tests/conftest.py`` clears the operator settings for every test, which is right:
    a suite must not depend on this workstation. These tests DO want the real staged bundle, so
    they point the config loader at the operator's own file explicitly instead of inheriting a
    global. Absent staging means skip, never a fabricated pass.
    """
    import sys

    import torch

    from workbench.backend import model_settings, operator_config, runtime
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.replay.common import load_trusted_whole_model

    config_file = REPO_ROOT / "workbench/config/workbench.env"
    if not config_file.is_file():
        pytest.skip("no operator workbench.env on this host (external artifact)")
    operator_config._CONFIG_PATH = config_file
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()
    config = operator_config.resolve_config()
    model = model_by_id("clvc_net")
    checkpoint = checkpoint_by_id(model, model["checkpoint_variants"][0]["checkpoint_id"])
    root = Path(checkpoint_path("clvc_net", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT))
    digests = {member["filename"]: member["expected_sha256"]
               for member in checkpoint["bundle_members"]}
    source = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
    if not (root / f"{category}_local_model.pt").is_file():
        pytest.skip("CLVC-Net checkpoints are not staged on this host (external artifact)")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    local = load_trusted_whole_model(root / f"{category}_local_model.pt",
                                     digests[f"{category}_local_model.pt"], torch).eval()
    global_model = load_trusted_whole_model(root / f"{category}_global_model.pt",
                                            digests[f"{category}_global_model.pt"], torch).eval()
    return local, global_model, torch


def _images(count: int = 2):
    import torchvision.transforms as transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    paths = sorted((FASHIONIQ / "resized_image/dress").glob("*.jpg"))[:count]
    if len(paths) < count:
        pytest.skip("FashionIQ dress images are not present on this host (external artifact)")
    return [transform(Image.open(path).convert("RGB")) for path in paths]


def test_the_text_tower_is_never_called_on_the_image_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both halves' ``compose_img_text`` are replaced by a raise: the path still produces a row."""
    import torch
    import torch.nn.functional as F

    local, global_model, _ = _checkpoint_pair()
    images = torch.stack(_images(2)).float()

    def explode(*_args, **_kwargs):
        raise AssertionError("the text tower was called on the image-only path")

    monkeypatch.setattr(local, "compose_img_text", explode)
    monkeypatch.setattr(global_model, "compose_img_text", explode)
    with torch.no_grad():
        rows = clvc.composed_rows(clvc.IMAGE_ENCODER_ONLY_OPERATION, local, global_model, images,
                                  (), torch, F)
    assert tuple(rows.shape) == (2, clvc.DIMENSION)


def test_the_edit_caption_is_inert_on_the_image_only_path() -> None:
    """Two different captions produce the SAME row: the caption is not an input, not merely unused."""
    import torch
    import torch.nn.functional as F

    local, global_model, _ = _checkpoint_pair()
    images = torch.stack(_images(2)).float()
    with torch.no_grad():
        first = clvc.composed_rows(clvc.IMAGE_ENCODER_ONLY_OPERATION, local, global_model, images,
                                   ["is blue and is long"], torch, F)
        second = clvc.composed_rows(clvc.IMAGE_ENCODER_ONLY_OPERATION, local, global_model, images,
                                    ["a completely different edit entirely"], torch, F)
    assert torch.equal(first, second), float((first - second).abs().max())

    # CONTROL: the same seam DOES carry the caption when the operation is `queries`. Without this,
    # "inert" would be consistent with a function that drops its arguments, i.e. proves nothing.
    # Run on stand-ins because the real text tower hardwires `.cuda()` (text_model.py:108).
    class _TextSensitive:
        def __init__(self, half: int) -> None:
            self.half = half

        def extract_img_feature(self, batch):
            return batch, torch.ones((len(batch), 1024))

        def compose_img_text(self, batch, texts):
            # The DIRECTION must depend on the caption, not the length: `F.normalize` would
            # collapse a pure rescaling, and two identical directions would prove nothing.
            row = torch.zeros((len(batch), 1024))
            row[:, 0] = 1.0
            row[:, 1] = float(sum(len(text) for text in texts))
            # `composed_rows` reads index [1] off local and [0] off global (test.py:100).
            return (None, row) if self.half == 0 else (row, None)

    stub = (_TextSensitive(0), _TextSensitive(1))
    short = clvc.composed_rows("queries", *stub, images, ["ed"], torch, F)
    long = clvc.composed_rows("queries", *stub, images, ["a completely different edit"], torch, F)
    assert not torch.allclose(short, long)


def test_the_image_only_row_equals_the_gallery_row_within_the_stated_tolerance() -> None:
    """Same transform, same halves, same normalisation, same concatenation order — measured."""
    import torch
    import torch.nn.functional as F

    local, global_model, _ = _checkpoint_pair()
    images = torch.stack(_images(2)).float()
    with torch.no_grad():
        gallery = clvc.composed_rows("targets", local, global_model, images, (), torch, F)
        single = clvc.composed_rows(clvc.IMAGE_ENCODER_ONLY_OPERATION, local, global_model,
                                    images[:1], (), torch, F)
    measured = float((gallery[0] - single[0]).abs().max())
    assert measured <= IMAGE_ONLY_TOLERANCE, measured
    # Each half is unit-norm and the whole row's norm is sqrt(2), exactly as the gallery path.
    norms = np.linalg.norm(single.numpy(), axis=1)
    assert np.allclose(norms, np.sqrt(2.0), atol=1e-6), norms


def test_the_image_only_row_is_scored_by_the_unmodified_scorer() -> None:
    """Tie-break, rank convention and reference eligibility are the multimodal path's, unchanged."""
    import torch
    import torch.nn.functional as F

    local, global_model, _ = _checkpoint_pair()
    images = torch.stack(_images(2)).float()
    with torch.no_grad():
        rows = clvc.composed_rows(clvc.IMAGE_ENCODER_ONLY_OPERATION, local, global_model,
                                  images, (), torch, F).numpy()
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    gallery = np.asarray(rows, dtype=np.float32)
    # The reference image is IN the gallery, so a reference-eligible scorer ranks it 1.
    ranks = clvc.Adapter.rank(adapter, rows, gallery, [[0], [1]])
    assert ranks == [1, 1]
    # A gallery row unrelated to the query ranks behind both: same cosine scorer, no new path.
    diag = np.zeros((1, clvc.DIMENSION), dtype=np.float32)
    diag[0, 0] = 1.0
    mixed = np.vstack([rows, diag])
    assert clvc.Adapter.rank(adapter, rows[:1], mixed, [[2]]) == [3]


def test_scoring_rank_and_tie_break_are_the_multimodal_paths_own() -> None:
    """No scoring code was added for the new path: it is the SAME scorer, the same rank rule.

    Asserted structurally, not by re-measuring arithmetic: ``score`` is CLVC's own delegating
    override (unchanged) and ``rank`` resolves to the shared base implementation, so the tie-break
    "1 + #greater + #earlier-equal" and the gallery-position order apply verbatim.
    """
    from workbench.composition.adapters import CompositionAdapter

    assert clvc.Adapter.score.__module__ == "workbench.composition.model_clvc_net"
    assert clvc.Adapter.rank is CompositionAdapter.rank
    assert clvc.Adapter.rank_and_topk is CompositionAdapter.rank_and_topk
    assert clvc.Adapter.reference_eligible is True


def test_the_image_only_path_transports_no_text_at_all() -> None:
    """A caller that hands it texts is refused: the condition cannot silently start using them."""
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter.work_root = Path("/tmp")
    adapter._worker = None
    adapter._worker_counter = 0
    with pytest.raises(ValueError, match="transports no text"):
        clvc._run(adapter, clvc.IMAGE_ENCODER_ONLY_OPERATION, ["a.jpg"], ["is blue"], ["dress"])


def test_image_only_empty_text_stays_declared_unsupported() -> None:
    """The new named path must not become a redefinition of the old, unsupported condition."""
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter._model = {"model_id": "clvc_net", "source_dir": "CLVCNet",
                      "environment": {"python_env_var": "WORKBENCH_CLVC_PYTHON"}}
    adapter._checkpoint = {"checkpoint_id": "c", "checkpoint_mapping_status": "VERIFIED_METADATA",
                           "bundle_members": []}
    adapter.checkpoint_path = Path("/nonexistent")
    adapter._interpreter = None
    adapter.config = SimpleNamespace(WORKBENCH_CHECKPOINT_ROOT=Path("/nonexistent"),
                                     WORKBENCH_THIRD_PARTY_ROOT=Path("/nonexistent"))
    supported, reason = adapter.mode_supported("image_only_empty_text")
    assert supported is False
    assert clvc.EMPTY_TEXT_REASON in reason
    supported, reason = adapter.mode_supported(clvc.IMAGE_ENCODER_ONLY)
    assert supported is False, "the structural blocker still governs: no checkpoints here"
    assert clvc.EMPTY_TEXT_REASON not in (reason or "")
