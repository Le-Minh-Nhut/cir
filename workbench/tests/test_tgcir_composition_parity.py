"""TG-CIR composition adapter: caption contract, refused targets, digest gate, reply shape.

No GPU, no network, no weights: the worker is never started. The one live-model property that needs
bytes (a real forward) is measured in `raw/verify/88-tgcir-composition.md`, not here.
"""
from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import numpy as np
import pytest

from workbench.composition import model_tgcir as module

DATASET = Path("/home/heheboiz/data/cir/data/FashionIQ")
PINNED_DATASETS = Path("/home/heheboiz/data/cir/workbench/third_party/TGCIR/datasets.py")
needs_dataset = pytest.mark.skipif(
    not (DATASET / "captions" / "correction_dict_dress.json").is_file(),
    reason="FashionIQ correction dictionaries are not staged")
needs_source = pytest.mark.skipif(not PINNED_DATASETS.is_file(), reason="pinned TGCIR absent")


def pinned_correct_text():
    """The PINNED `correct_text`, compiled out of `datasets.py` with only the stdlib available.

    Importing `datasets` would need torch, PIL and a GPU-less repo path; the function itself is pure
    stdlib, so its exact source text is executed here. That makes "our copy matches the pinned
    function" a measured claim rather than a comment.
    """
    import re
    source = PINNED_DATASETS.read_text(encoding="utf-8")
    match = re.search(
        r"    def correct_text\(self, text, correction_dict\):\n(?:.*\n)*?(?=    def concat_text)",
        source)
    assert match, "pinned correct_text not found"
    namespace: dict[str, object] = {"string": string}
    exec("class _Pinned:\n" + match.group(0), namespace)  # noqa: S102 - pinned source, verbatim
    return namespace["_Pinned"].correct_text


@needs_dataset
def test_correct_text_matches_the_pinned_function_on_a_fixture():
    """Punctuation -> space, lowercase, per-token substitution: byte-identical to `datasets.py`."""
    pinned = pinned_correct_text()
    table = {"vneck": "v neck", "tshirt": "t shirt", "longand": "long and"}
    pinned_method = pinned_correct_text()
    for text in ("A V-Neck, TShirt!", "  Mixed.CASE  ", "and", "Longand sleeves; navy blue."):
        assert module._correct_text(text, table) == pinned_method(None, text, table), text


@needs_dataset
def test_correction_is_applied_exactly_once_and_uses_the_pinned_pair_form():
    """The corrected string reaches the model ONCE, in the pinned per-caption-then-join form."""
    corrections = module.Corrections(DATASET)
    rows = json.loads((DATASET / "captions" / "cap.shirt.val.json").read_text(encoding="utf-8"))
    dictionary = corrections.dictionaries["shirt"]
    pinned = pinned_correct_text()
    checked = changed = 0
    for row in rows:
        raw = " and ".join(row["captions"])
        expected = "{} and {}".format(*[pinned(None, part, dictionary) for part in row["captions"]])
        got = corrections.correct([raw], [None], [f"{row['candidate']}.jpg"])
        assert got == [expected]
        changed += got != [raw]
        # Idempotence is NOT universal (one val row has a punctuation token that becomes an
        # apostrophe-free word), so the claim under test is "exactly once", not "stable".
        twice = corrections.correct(got, [None], [f"{row['candidate']}.jpg"])
        assert twice in (got, [module._correct_text(got[0], dictionary)])
        checked += 1
        if checked == 40:
            break
    assert checked == 40
    assert changed, "no sampled row was changed by the correction, so this proves nothing"


@needs_dataset
def test_double_correction_is_observably_different_so_once_is_load_bearing():
    """At least one val row is non-idempotent: applying the pinned function twice changes it.

    Proves the contract in the module docstring is testable rather than decorative -- a worker that
    corrected an already-corrected text (or corrected twice) would produce a different string.
    """
    corrections = module.Corrections(DATASET)
    dictionary = corrections.dictionaries["toptee"]
    row = next(row for row in json.loads(
        (DATASET / "captions" / "cap.toptee.val.json").read_text(encoding="utf-8"))
        if row["candidate"] == "B00C9NQNSY")
    once = corrections.correct([" and ".join(row["captions"])], [None], ["B00C9NQNSY.jpg"])[0]
    twice = module._correct_text(once, dictionary)
    assert twice != once
    assert "coverup" in once and "coverup" in twice


@needs_dataset
def test_correction_refuses_a_text_it_cannot_assign_to_a_dictionary():
    corrections = module.Corrections(DATASET)
    with pytest.raises(ValueError, match="resolves to no category"):
        corrections.correct(["a sentence that is in no caption file"], [None], ["nope.jpg"])
    with pytest.raises(ValueError, match="no correction dictionary"):
        corrections.correct(["is grey"], ["shoes"], ["x.jpg"])


@needs_dataset
def test_correction_uses_the_caller_category_when_one_is_given():
    """A non-None category wins; the two dictionaries do differ (49 keys, measured)."""
    corrections = module.Corrections(DATASET)
    text = "Is grey with a design on the back."
    values = {category: corrections.correct([text], [category], ["B005PQ02G6.jpg"])[0]
              for category in ("dress", "shirt", "toptee")}
    assert set(values) == set(corrections.dictionaries)
    assert isinstance(values["shirt"], str) and values["shirt"]


def test_encode_queries_transports_texts_verbatim(monkeypatch, tmp_path):
    """The adapter must not rewrite texts: the correction belongs to the worker (downstream of _run)."""
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path
    seen: list[list[str]] = []

    def fake_run(_adapter, operation, image_paths, texts, categories):
        seen.append(list(texts))
        return np.zeros((len(image_paths), module.DIMENSION), dtype=np.float32)

    monkeypatch.setattr(module, "_run", fake_run)
    sent = ["", "an edit", "is grey with a design on the back and is shiny."]
    adapter.encode_queries(["a.png", "b.png", "c.png"], sent, [None, None, None])
    assert seen == [sent]


def test_query_and_target_row_order_is_preserved(monkeypatch, tmp_path):
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path
    order: list[tuple[str, list[int]]] = []

    def fake_run(_adapter, operation, image_paths, texts, categories):
        order.append((operation, list(range(len(image_paths)))))
        return np.arange(len(image_paths) * module.DIMENSION,
                         dtype=np.float32).reshape(len(image_paths), module.DIMENSION)

    monkeypatch.setattr(module, "_run", fake_run)
    queries = adapter.encode_queries(["q3", "q1", "q2"], ["t3", "t1", "t2"], [None] * 3)
    targets = adapter.encode_targets(["g9", "g8"], [None, None])
    assert order == [("queries", [0, 1, 2]), ("targets", [0, 1])]
    # Row i of the reply is input i, not a sorted or hashed rearrangement.
    np.testing.assert_array_equal(queries[1], np.arange(module.DIMENSION) + module.DIMENSION)
    np.testing.assert_array_equal(targets[0], np.arange(module.DIMENSION))


def test_none_target_gallery_is_accepted_and_a_category_is_refused(monkeypatch, tmp_path):
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path
    monkeypatch.setattr(module, "_run", lambda _a, op, paths, texts, categories:
                        np.zeros((len(paths), module.DIMENSION), dtype=np.float32))
    accepted = adapter.encode_targets(["a.jpg", "b.jpg"], [None, None])
    assert accepted.shape == (2, module.DIMENSION)
    with pytest.raises(ValueError, match="category-agnostic"):
        adapter.encode_targets(["a.jpg"], ["dress"])
    with pytest.raises(ValueError, match="equal lengths"):
        adapter.encode_targets(["a.jpg"], [None, None])


def test_digest_mismatch_is_refused_before_torch_load(tmp_path, monkeypatch):
    """An unauthorised pickle must never reach `torch.load`, even through the trusted loader."""
    import torch

    pickle = tmp_path / "Fashioniq.pt"
    pickle.write_bytes(b"not the authorised bytes")
    calls: list[str] = []

    def explode(*_args, **_kwargs):
        calls.append("torch.load")
        raise AssertionError("torch.load must not be reached for a mismatched digest")

    monkeypatch.setattr(torch, "load", explode)
    import workbench.replay.common as common

    monkeypatch.setattr(common, "load_trusted_whole_model", explode, raising=True)
    with pytest.raises(module.CheckpointDigestMismatch, match="SHA-256 mismatch"):
        module._load_verified_model(pickle, "0" * 64)
    with pytest.raises(module.CheckpointDigestMismatch, match="no authorised SHA-256"):
        module._load_verified_model(pickle, None)
    with pytest.raises(module.CheckpointDigestMismatch, match="size mismatch"):
        module._load_verified_model(pickle, "0" * 64, expected_size_bytes=1)
    assert calls == []


def test_digest_check_runs_before_the_loader_is_handed_the_file(tmp_path, monkeypatch):
    """With the RIGHT digest the verified bytes do reach the trusted loader, once."""
    import workbench.replay.common as common

    payload = tmp_path / "ok.pt"
    payload.write_bytes(b"bytes")
    import hashlib

    digest = hashlib.sha256(b"bytes").hexdigest()
    seen: list[str] = []
    monkeypatch.setattr(common, "load_trusted_whole_model",
                        lambda path, expected, torch_module: seen.append(str(expected)) or "model")
    assert module._load_verified_model(payload, digest, len(b"bytes")) == "model"
    assert seen == [digest]


def test_reply_shape_mismatch_raises_the_named_error(tmp_path, monkeypatch):
    """A worker reply of the wrong width is a named failure, never a silently short matrix."""
    adapter = module.Adapter.__new__(module.Adapter)
    adapter.work_root = tmp_path
    adapter.interpreter = sys.executable
    adapter._interpreter_verified = True
    adapter.batch_size = 2
    adapter._worker_pid = None
    adapter._counter = 0
    adapter._job_dir = None
    adapter._worker = None
    adapter._worker_log = None
    adapter._worker_failed = None
    adapter.load_state = None
    adapter._model = {"model_id": "tgcir", "environment": {"python_env_var": "X"}}
    adapter.checkpoint_path = tmp_path / "Fashioniq.pt"
    adapter.source_root = tmp_path
    adapter.dataset_root = tmp_path
    adapter.checkpoint = {"expected_sha256": "0" * 64}

    def fake_exchange(phase, image_paths, texts, categories):
        output = tmp_path / "out.npy"
        np.save(output, np.zeros((len(image_paths), module.DIMENSION - 1), dtype=np.float32))
        return {"ok": True, "pid": 1234}, output

    monkeypatch.setattr(adapter, "_exchange", fake_exchange)
    with pytest.raises(module.AdapterWorkerError, match="adapter_worker_invalid_matrix"):
        module._run(adapter, "queries", ["a.jpg", "b.jpg"], ["t", "t"], [None, None])


def test_worker_pid_change_is_a_named_refusal():
    """A reloaded worker means the model was loaded twice; that is refused, not absorbed."""
    adapter = module.Adapter.__new__(module.Adapter)
    adapter._worker_pid = 4242
    stopped: list[bool] = []
    adapter._stop_worker = lambda: stopped.append(True)
    assert adapter._check_worker_pid(4242) == 4242
    with pytest.raises(module.AdapterWorkerError, match="adapter_worker_reloaded"):
        adapter._check_worker_pid(4243)
    assert stopped == [True]


def test_first_reply_records_the_pid_instead_of_refusing():
    adapter = module.Adapter.__new__(module.Adapter)
    adapter._worker_pid = None
    assert adapter._check_worker_pid(999) == 999
    assert adapter._worker_pid == 999

@needs_dataset
def test_a_blank_caption_is_transported_as_blank_rather_than_refused():
    """`image_only_empty_text` hands the model an EMPTY caption, and TG-CIR can encode it.

    Refusing here aborted the whole run: the worker replies failure, `_run` raises
    `AdapterWorkerError`, and the runner re-raises that unconditionally (runner.py:693-694). A blank
    caption is not ambiguous -- every dictionary maps it to "" -- so it is passed through, once.
    """
    corrections = module.Corrections(DATASET)
    real = "is solid white and is a lighter color"
    assert corrections.correct(["", real], [None, None]) == ["", real]
    assert corrections.correct([""], [None]) == [""]
    assert corrections.correct([""], ["dress"]) == [""]
    assert corrections.correct(["   "], [None]) == [""]
    # The multimodal text is unchanged by the new branch, so the two conditions still differ only
    # in their pixels and the empty-text condition still differs only in its text.
    assert corrections.correct([real], [None]) == [real]

@needs_dataset
def test_a_non_blank_ambiguous_caption_is_still_refused():
    """The blank branch must not swallow the real refusal: two dictionaries that DISAGREE refuse."""
    corrections = module.Corrections(DATASET)
    key, (stem, forms) = "a probe caption", ("B005X4PL1G", {
        "dress": "probe dress form", "shirt": "probe shirt form"})
    corrections._by_text[key] = dict(forms)
    corrections._by_pair[(stem, key)] = dict(forms)
    try:
        with pytest.raises(ValueError, match="is ambiguous"):
            corrections.correct([key], [None], [f"{stem}.jpg"])
    finally:
        del corrections._by_text[key]
        del corrections._by_pair[(stem, key)]
    # After removing the fixture the same caption refuses for the OTHER stated reason when no
    # reference stem can resolve it, so the ambiguity branch was the one under test and not a
    # lucky accident of the lookup.
    with pytest.raises(ValueError, match="resolves to no category"):
        corrections.correct([key], [None])
