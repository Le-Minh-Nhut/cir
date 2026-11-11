"""Decidable-without-weights checks for the CLVC-Net composition adapter.

Everything here runs on CPU with no network, no real checkpoint and no GPU: the properties that
need real embeddings are measured on the host instead (evidence
``docs/reviews/20261111-master-audit/raw/verify/87-clvc-composition.md``), because copying a number
into a test would assert the past rather than the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from workbench.composition import model_clvc_net as clvc


def _adapter(tmp_path: Path):
    """An adapter with no worker: work_root set, gate bypassed only for INPUT validation tests.

    The gate itself is exercised by ``test_circus_env_gate_omp.py``; these tests never launch a
    worker, so the flag only stops ``_run`` from asking for an interpreter before it reaches the
    refusal under test.
    """
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter.work_root = tmp_path
    adapter.batch_size = 4
    adapter._worker = None
    adapter._worker_log = None
    adapter._worker_failed = None
    adapter._job_root = None
    adapter._worker_counter = 0
    adapter.worker_pids = []
    adapter._interpreter_verified = True
    adapter._interpreter = sys.executable
    adapter._model = {"model_id": "clvc_net", "source_dir": "CLVCNet",
                      "environment": {"python_env_var": "WORKBENCH_CLVC_PYTHON"}}
    adapter._checkpoint = {"checkpoint_id": "author_drive_unresolved",
                           "checkpoint_mapping_status": "VERIFIED_METADATA",
                           "bundle_members": [
                               {"filename": f"{category}_{member}_model.pt",
                                "expected_sha256": "0" * 64}
                               for category in clvc.CATEGORIES
                               for member in ("local", "global")]}
    adapter.checkpoint_path = tmp_path / "checkpoints"
    adapter.checkpoint_path.mkdir(exist_ok=True)
    adapter.source_root = tmp_path / "source"
    adapter.config = _Config(tmp_path)
    return adapter


class _Config:
    """Enough of the operator config for the structural check: no interpreter, no artifacts."""

    def __init__(self, root: Path) -> None:
        self.WORKBENCH_CHECKPOINT_ROOT = root
        self.WORKBENCH_THIRD_PARTY_ROOT = root
        self.CIR_REPO_ROOT = root


# ------------------------------------------------------------------ row order across categories

def test_scatter_puts_rows_back_at_input_positions() -> None:
    """Interleaved categories: grouped-order output must land on the right rows."""
    categories = ["dress", "shirt", "dress", "toptee", "shirt"]
    grouped = {name: np.full((count, 2), number, dtype=np.float32)
               for number, (name, count) in enumerate(
                   {"dress": 2, "shirt": 2, "toptee": 1}.items(), start=1)}
    rows = clvc.scatter_rows(np.zeros((5, 2), dtype=np.float32), categories, grouped)
    for index, category in enumerate(categories):
        assert rows[index, 0] == grouped[category][0, 0], (index, category)


def test_group_indices_are_the_input_positions_in_first_appearance_order() -> None:
    assert clvc.group_indices(["shirt", "dress", "shirt"]) == {"shirt": [0, 2], "dress": [1]}


def test_scatter_refuses_a_row_count_that_does_not_cover_the_inputs() -> None:
    """A short block would silently leave a zero row where an image's feature belongs."""
    with pytest.raises(ValueError, match="clvc_worker_row_count"):
        clvc.scatter_rows(np.zeros((3, 2), np.float32), ["dress"] * 3,
                          {"dress": np.ones((2, 2), np.float32)})


@pytest.mark.parametrize("categories", [["dress", "shirt"], ["toptee", "dress", "shirt"]])
def test_interleaved_multi_category_order_holds_for_every_split(categories) -> None:
    grouped = {name: np.full((categories.count(name), 1), number, dtype=np.float32)
               for number, name in enumerate(sorted(set(categories)), start=1)}
    rows = clvc.scatter_rows(np.zeros((len(categories), 1), np.float32), categories, grouped)
    expected = [grouped[name][0, 0] for name in categories]
    assert rows[:, 0].tolist() == expected


# ------------------------------------------------------------------ refusals

def test_none_query_category_is_refused_by_name(tmp_path: Path) -> None:
    """A query's category is never inferred: the refusal has to name the reason."""
    adapter = _adapter(tmp_path)
    with pytest.raises(clvc.Refused, match="adapter_query_category_missing"):
        clvc._run(adapter, "queries", [str(tmp_path / "a.jpg")], ["an edit"], [None])


def test_unknown_query_category_is_refused_by_name(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(clvc.Refused, match="adapter_query_category_unknown"):
        clvc._run(adapter, "queries", [str(tmp_path / "a.jpg")], ["an edit"], ["trousers"])


def test_mixed_category_target_gallery_is_refused(tmp_path: Path) -> None:
    """The adapter encodes ONE category per gallery; its own check must fire before any launch."""
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter.work_root = tmp_path
    with pytest.raises(ValueError, match="scopes must not be mixed"):
        clvc.Adapter.encode_targets(adapter, ["a.jpg", "b.jpg"], ["dress", "shirt"])


def test_target_gallery_scope_outside_the_staged_categories_is_refused(tmp_path: Path) -> None:
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter.work_root = tmp_path
    with pytest.raises(ValueError, match="is not one of"):
        clvc.Adapter.encode_targets(adapter, ["a.jpg"], ["trousers"])


def test_empty_edit_is_refused_by_name(tmp_path: Path) -> None:
    """``image_only_empty_text`` is unrepresentable: tokenising to nothing indexes x[0][0]."""
    adapter = _adapter(tmp_path)
    with pytest.raises(clvc.Refused, match="adapter_empty_edit_unrepresentable"):
        clvc._run(adapter, "queries", [str(tmp_path / "a.jpg")], [""], ["dress"])


def test_work_root_is_required_before_encoding(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.work_root = None
    with pytest.raises(clvc.AdapterWorkRootError):
        clvc._run(adapter, "targets", ["a.jpg"], (), ["dress"])


# ------------------------------------------------------------------ digest before unpickling

def test_digest_mismatch_is_refused_before_torch_load(tmp_path: Path) -> None:
    """The REAL loader chain: hash first, ``torch.load`` only on authorised bytes.

    ``torch.load`` is instrumented, not monkeypatched away: if the mismatch path ever reached it,
    the recorded call would be non-empty and the assertion would say so.
    """
    member = tmp_path / "dress_local_model.pt"
    member.write_bytes(b"not the author's weights")
    calls: list[str] = []

    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=True):  # noqa: ARG004 - signature mirror
            calls.append(str(path))
            raise AssertionError("torch.load reached for an unauthorised file")

    with pytest.raises(clvc.CheckpointDigestMismatch, match="clvc_checkpoint_digest_mismatch"):
        clvc.load_category_pair(
            "dress", [{"path": str(member), "sha256": "0" * 64, "member": "local"}], FakeTorch)
    assert calls == []


def test_a_member_without_a_declared_digest_is_refused(tmp_path: Path) -> None:
    member = tmp_path / "dress_local_model.pt"
    member.write_bytes(b"x")
    with pytest.raises(clvc.CheckpointDigestMismatch):
        clvc.load_category_pair(
            "dress", [{"path": str(member), "sha256": None, "member": "local"}], object())


# ------------------------------------------------------------------ score identity

@pytest.mark.parametrize("seed", [0, 7])
def test_score_equals_the_mean_of_the_two_half_dot_products(seed: int) -> None:
    """``test.py:100-129`` on rows of two unit 1024-d halves, carried to float32.

    ``test.py`` normalizes each 1024-d half, concatenates, divides the 2048-d row by its norm
    (exactly ``sqrt(2)``) and dots. That is ``(dot_local + dot_global) / 2`` — the identity the
    adapter's docstring claims, so it is asserted here on synthetic unit halves.
    """
    rng = np.random.default_rng(seed)
    def halves(count: int) -> np.ndarray:
        blocks = rng.normal(size=(count, 2, clvc.HALF_DIMENSION)).astype(np.float32)
        blocks /= np.linalg.norm(blocks, axis=2, keepdims=True)
        return blocks.reshape(count, clvc.DIMENSION)

    queries, targets = halves(4), halves(6)
    expected = ((queries[:, :1024] @ targets[:, :1024].T)
                + (queries[:, 1024:] @ targets[:, 1024:].T)) / np.float32(2.0)
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    scores = adapter.score(queries, targets)
    assert scores.shape == (4, 6)
    np.testing.assert_allclose(scores, expected, atol=1e-6)
    # And it is a cosine: bounded by 1, no token axis to max over.
    assert float(scores.max()) <= 1.0 + 1e-5


def test_rows_from_the_encoders_are_unit_norm_per_half() -> None:
    """The identity above only holds for rows the encoders produced; pin what that means."""
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    row = np.zeros((1, clvc.DIMENSION), dtype=np.float32)
    row[0, :1024] = 1.0 / np.sqrt(1024)
    row[0, 1024:] = 1.0 / np.sqrt(1024)
    half_row = np.stack([row[0, :1024], row[0, 1024:]])
    np.testing.assert_allclose(np.linalg.norm(half_row, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(row), np.sqrt(2.0), atol=1e-6)
    assert adapter.score(row, row)[0, 0] == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ worker reply contract

def _fake_worker(call, tmp_path: Path, matrix):
    """A ``Popen`` stand-in: writes ``matrix`` where the job asked, replies with the marker."""
    import io

    class FakeWorker:
        pid = 4321
        returncode = None
        stdin = None

        def __init__(self) -> None:
            self.stdout = io.StringIO(clvc.WORKER_MARKER + clvc.json.dumps(
                {"ok": True, "worker_pid": self.pid}) + "\n")

        def poll(self):
            return -1

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    def popen(argv, **kwargs):
        payload = clvc.json.loads(Path(argv[-1]).read_text())
        call.append(payload)
        np.save(payload["output_path"], matrix(len(payload["image_paths"])))
        return FakeWorker()

    return popen


def test_worker_reply_shape_mismatch_raises_the_named_error(tmp_path: Path, monkeypatch) -> None:
    """A reply whose matrix is the wrong width must fail loudly, naming the worker log."""
    adapter = _adapter(tmp_path)
    # The bundle's bytes are irrelevant here: this test is about the reply, not the artifact.
    monkeypatch.setattr("workbench.backend.registry.checkpoint_is_present", lambda *a, **k: True)
    calls: list[dict] = []
    monkeypatch.setattr(clvc.subprocess, "Popen",
                        _fake_worker(calls, tmp_path,
                                     lambda n: np.zeros((n, clvc.DIMENSION - 1), np.float32)))
    with pytest.raises(clvc.AdapterWorkerError, match="adapter_worker_invalid_matrix"):
        clvc._run(adapter, "targets", [str(tmp_path / "a.jpg")], (), ["dress"])
    assert calls, "the job was never written"
    adapter._worker = None
    adapter._worker_log = None


def test_worker_job_payload_keeps_texts_verbatim(tmp_path: Path, monkeypatch) -> None:
    """No rewrite on the way out: the runner already applied the condition."""
    adapter = _adapter(tmp_path)
    monkeypatch.setattr("workbench.backend.registry.checkpoint_is_present", lambda *a, **k: True)
    calls: list[dict] = []
    monkeypatch.setattr(clvc.subprocess, "Popen",
                        _fake_worker(calls, tmp_path,
                                     lambda n: np.zeros((n, clvc.DIMENSION), np.float32)))
    sent = ["an edit", "another edit"]
    clvc._run(adapter, "queries", [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
              sent, ["dress", "dress"])
    assert calls[-1]["texts"] == sent
    adapter._worker = None
    adapter._worker_log = None


# ------------------------------------------------------------------ declared contract

def test_the_adapter_declares_the_native_shape() -> None:
    assert clvc.Adapter.model_id == "clvc_net"
    assert clvc.Adapter.score_dimension == 2048
    assert clvc.Adapter.category_scopes == ("dress", "shirt", "toptee")
    assert clvc.Adapter.reference_eligible is True
    assert clvc.DIMENSION == 2 * clvc.HALF_DIMENSION == 2048


def test_the_module_keeps_the_transport_signature_the_runner_calls() -> None:
    import inspect

    signature = inspect.signature(clvc._run)
    assert list(signature.parameters) == [
        "adapter", "operation", "image_paths", "texts", "categories"]
