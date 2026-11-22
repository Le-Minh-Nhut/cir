"""Tests for TG-CIR protocol parity and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.tgcir import (
    MODEL,
    preflight,
)



def test_tgcir_pair_cli_passes_selected_category(tmp_path: Path, monkeypatch, capsys):
    import sys
    from workbench.replay import tgcir_pair

    called = {}
    monkeypatch.setattr(tgcir_pair, "run_pair",
                        lambda *args, **kwargs: called.update(kwargs) or {"category": kwargs["category"]})
    monkeypatch.setattr(sys, "argv", ["tgcir_pair", "--dataset-root", str(tmp_path),
                                      "--checkpoint-root", str(tmp_path), "--source-root", str(tmp_path),
                                      "--output", str(tmp_path / "out"), "--category", "shirt"])

    assert tgcir_pair.main() == 0
    assert called["category"] == "shirt"
    assert '"category": "shirt"' in capsys.readouterr().out
def test_tgcir_native_stage_consumer_validates_atomic_artifact_and_child_exit(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    import numpy as np

    from workbench.replay import tgcir_pair

    dataset = tmp_path / "dataset"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    rows = [{"candidate": f"a{i}", "target": f"b{i}", "captions": ["blue"]} for i in range(5)]
    (dataset / "captions/cap.dress.val.json").write_text(json.dumps(rows))
    (dataset / "image_splits/split.dress.val.json").write_text(json.dumps([f"{prefix}{i}" for prefix in ("a", "b") for i in range(5)]))
    output = tmp_path / "stage"
    output.mkdir()

    def child(_command, **_kwargs):
        arrays = output / "native-arrays.npz"
        tgcir_pair._atomic_npz(
            arrays, queries=np.zeros((5, 512), np.float32), targets=np.zeros((10, 512), np.float32),
            scores=np.zeros((5, 10), np.float32), ranks=np.tile(np.arange(10), (5, 1)),
            g1_token_ids=np.zeros((5, 77), np.int64))
        population = tgcir_pair.g1_sample(dataset, "dress", None)
        metadata = {"native_pid": os.getpid() + 1,
                    "query_ids": population["query_image_names"],
                    "gallery_ids": population["gallery_image_names"],
                    "reference_indices": population["reference_indices"],
                    "target_indices": population["target_indices"],
                    "artifact_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest()}
        (output / "native-stage.json").write_text(json.dumps(metadata))

    from workbench.backend import runtime
    monkeypatch.setattr(runtime, "require_verified_model_interpreter",
                        lambda *_a, **_k: "/verified/tgcir/python")
    monkeypatch.setattr(tgcir_pair.subprocess, "run", child)
    result = tgcir_pair._run_native_stage(dataset, tmp_path, tmp_path, output, 16, "dress")
    assert result["native_pid"] != os.getpid()

    def failed_child(*_args, **_kwargs):
        raise tgcir_pair.subprocess.CalledProcessError(7, ["python", "native-stage"])

    monkeypatch.setattr(tgcir_pair.subprocess, "run", failed_child)
    with pytest.raises(tgcir_pair.subprocess.CalledProcessError):
        tgcir_pair._run_native_stage(dataset, tmp_path, tmp_path, output, 16, "dress")


def test_tgcir_g1_sample_preserves_annotation_order_and_pair_union(tmp_path: Path):
    import json
    from workbench.replay.tgcir_pair import g1_sample

    captions = tmp_path / "captions"
    splits = tmp_path / "image_splits"
    captions.mkdir()
    splits.mkdir()
    (captions / "cap.dress.val.json").write_text(json.dumps([
        {"candidate": "a", "target": "b", "captions": ["blue", "soft"]},
        {"candidate": "b", "target": "c", "captions": ["long"]},
    ]))
    (splits / "split.dress.val.json").write_text(json.dumps(["a", "b", "c"]))

    sample = g1_sample(tmp_path, "dress", limit=2)

    assert sample["query_image_names"] == ["dress_a", "dress_b"]
    assert sample["gallery_image_names"] == ["dress_a", "dress_b", "dress_c"]
    assert sample["reference_indices"] == [0, 1]
    assert sample["target_indices"] == [1, 2]
    assert sample["category"] == "dress"
    with pytest.raises(ValueError):
        g1_sample(tmp_path, "dress", limit=6)
    assert sample["query_texts"] == ["blue and soft", "long"]

def test_tgcir_scoring_query_ids_distinguish_repeated_candidate_rows():
    from workbench.replay.tgcir_pair import scoring_query_ids

    assert scoring_query_ids(["shirt_a", "shirt_a", "shirt_b"]) == [
        "shirt_a#row-0", "shirt_a#row-1", "shirt_b#row-2"]


def test_tgcir_v3_plan_keeps_repeated_source_candidates_as_distinct_queries(tmp_path: Path):
    import json
    from workbench.replay.tgcir_v3 import ROLE_SIDES, build_plan

    (tmp_path / "captions").mkdir()
    (tmp_path / "image_splits").mkdir()
    (tmp_path / "captions/cap.dress.val.json").write_text(json.dumps([
        {"candidate": "a", "target": "b", "captions": ["blue"]},
        {"candidate": "a", "target": "c", "captions": ["dark"]},
    ]))
    (tmp_path / "image_splits/split.dress.val.json").write_text(json.dumps(["a", "b", "c"]))
    plan = build_plan(dataset_root=tmp_path, category="dress", checkpoint_sha256="a" * 64,
                      configuration_sha256="b" * 64, environment="/pin/python 3.9.25 torch 2.7.1",
                      protocol="upstream:test.py", code_sha256={"source": "c" * 64},
                      output_root=tmp_path / "roles", cohort_id="fashioniq-val-dress")

    assert set(plan["runs"]) == set(ROLE_SIDES)
    assert plan["population_scope"]["query_ids"] == ["dress_a#row-0", "dress_a#row-1"]
    assert plan["population_scope"]["gallery_ids"] == ["dress_a", "dress_b", "dress_c"]
    assert {run["execution"]["query_batch_size"] for run in plan["runs"].values()} == {1, 32}


def test_tgcir_v3_native_g1_capture_roundtrips_adapter_validation(tmp_path: Path):
    import json
    from datetime import datetime, timezone

    import numpy as np

    from workbench.replay.common import sha256_file
    from workbench.replay.tgcir_v3 import build_plan, capture_role, validate_native_g1_sample

    dataset = tmp_path / "dataset"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    (dataset / "captions/cap.dress.val.json").write_text(json.dumps([
        {"candidate": "a", "target": "b", "captions": ["blue"]},
        {"candidate": "a", "target": "c", "captions": ["dark"]},
    ]))
    (dataset / "image_splits/split.dress.val.json").write_text(json.dumps(["a", "b", "c"]))
    plan = build_plan(dataset_root=dataset, category="dress", checkpoint_sha256="a" * 64,
                      configuration_sha256="b" * 64, environment="python=3.9.25 torch=2.7.1 /pin/python",
                      protocol="upstream:test.py", code_sha256={"source": "c" * 64},
                      output_root=tmp_path / "roles", cohort_id="fashioniq-val-dress")
    qids, gids = plan["population_scope"]["query_ids"], plan["population_scope"]["gallery_ids"]
    labels = np.asarray([gids.index(name) for name in plan["population_scope"]["target_ids"]], dtype=np.int64)
    scores = np.asarray([[0.9, 0.2, 0.1], [0.2, 0.9, 0.1]], dtype=np.float32)
    ranks = np.argsort(-scores, axis=1, kind="stable")
    arrays = {"query_features": np.ones((2, 512), dtype=np.float32),
              "gallery_features": np.ones((3, 512), dtype=np.float32), "scores": scores,
              "ranks": ranks, "production_ranks": np.ones(2, dtype=np.int64), "labels": labels}
    ordered_ids = {name: qids for name in arrays}
    ordered_ids["gallery_features"] = gids
    g1 = {"query_pixels": np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3),
          "gallery_pixels": np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3),
          "token_ids": np.ones((1, 77), dtype=np.int32), "token_mask": np.ones((1, 77), dtype=bool),
          "query_texts": ["blue"], "raw_query_texts": ["blue"],
          "query_paths": ["q.jpg"], "gallery_paths": ["r.jpg", "t.jpg"],
          "query_image_names": ["q"], "gallery_image_names": ["r", "t"]}
    native_dir = tmp_path / "roles/native_adapter_native"
    receipt = capture_role(plan=plan, plan_sha256="d" * 64, role="native_adapter_native",
                           started_at=datetime.now(timezone.utc).isoformat(),
                           execution=plan["runs"]["native_adapter_native"]["execution"],
                           arrays=arrays, ordered_ids=ordered_ids, output=native_dir, g1=g1)
    artifact_path = native_dir / "role-arrays.npz"
    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert artifact["g1_query_image_names"].astype(str).tolist() == g1["query_image_names"]
        assert artifact["g1_gallery_image_names"].astype(str).tolist() == g1["gallery_image_names"]
        assert artifact["g1_raw_query_texts"].astype(str).tolist() == g1["raw_query_texts"]
    binding = validate_native_g1_sample(
        plan=plan, role="native_adapter_adapter", native_g1_receipt=artifact_path,
        g1={key: value for key, value in g1.items()}, paired_receipt=receipt,
        native_receipt_path=native_dir / "role-receipt.json")
    assert binding["native_role"]["role"] == "native_adapter_native"
    assert binding["native_role"]["role_receipt_sha256"] == sha256_file(native_dir / "role-receipt.json")
    assert binding["native_role_artifact_sha256"] == sha256_file(artifact_path)


def test_tgcir_v3_synthetic_artifact_contract_reports_missing_native_identity(tmp_path: Path):
    import json
    import time
    import numpy as np
    from datetime import datetime, timezone
    from workbench.backend import parity_gate
    from workbench.replay.tgcir_v3 import ROLE_SIDES, build_plan, capture_role, preregister, aggregate

    dataset = tmp_path / "dataset"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    (dataset / "captions/cap.dress.val.json").write_text(json.dumps([
        {"candidate": "a", "target": "b", "captions": ["blue"]},
        {"candidate": "a", "target": "c", "captions": ["dark"]},
    ]))
    (dataset / "image_splits/split.dress.val.json").write_text(json.dumps(["a", "b", "c"]))
    plan = build_plan(dataset_root=dataset, category="dress", checkpoint_sha256="a" * 64,
                      configuration_sha256="b" * 64, environment="python=3.9.25 torch=2.7.1 /pin/python",
                      protocol="upstream:test.py", code_sha256={"source": "c" * 64},
                      output_root=tmp_path / "roles", cohort_id="fashioniq-val-dress")
    root = tmp_path / "gate"
    root.mkdir()
    plan_path = root / "g1_g2.tgcir.dress.v3-plan.json"
    plan_sha = preregister(plan_path, plan)
    time.sleep(0.02)
    qids, gids = plan["population_scope"]["query_ids"], plan["population_scope"]["gallery_ids"]
    labels = np.asarray([gids.index(name) for name in plan["population_scope"]["target_ids"]], dtype=np.int64)
    scores = np.asarray([[0.0, 0.9, 0.2], [0.0, 0.2, 0.9]], dtype=np.float32)
    scores[:, 0] = -10e10
    order = np.argsort(-scores, axis=1, kind="stable")
    arrays = {"query_features": np.ones((2, 512), dtype=np.float32),
              "gallery_features": np.ones((3, 512), dtype=np.float32), "scores": scores,
              "ranks": order, "production_ranks": np.asarray([1, 1], dtype=np.int64), "labels": labels}
    ordered = {name: qids for name in arrays}
    ordered["gallery_features"] = gids
    g1 = {"query_pixels": np.zeros((1, 2, 2, 3), dtype=np.uint8),
          "gallery_pixels": np.zeros((2, 2, 2, 3), dtype=np.uint8),
          "token_ids": np.ones((1, 77), dtype=np.int32), "token_mask": np.ones((1, 77), dtype=bool),
          "query_texts": ["blue"], "raw_query_texts": ["blue"],
          "query_paths": ["q.jpg"], "gallery_paths": ["r.jpg", "t.jpg"],
          "query_image_names": ["q"], "gallery_image_names": ["r", "t"]}
    receipts = {}
    role_order = ["native_repeat_1", "native_adapter_native",
                  *(role for role in ROLE_SIDES if role not in ("native_repeat_1", "native_adapter_native"))]
    for role in role_order:
        role_dir = tmp_path / "roles" / role
        role_g1 = dict(g1)
        if role.startswith("adapter_") or role == "native_adapter_adapter":
            from workbench.replay.common import sha256_file
            native_role = "native_adapter_native" if role == "native_adapter_adapter" else "native_repeat_1"
            native_receipt = receipts[native_role]
            role_g1["native_role"] = {"role": native_role, "run_id": plan["runs"][native_role]["run_id"],
                                      "source_id": plan["runs"][native_role]["source_id"],
                                      "role_receipt_sha256": sha256_file(native_receipt)}
            role_g1["native_role_artifact_sha256"] = sha256_file(native_receipt.parent / "role-arrays.npz")
        capture_role(plan=plan, plan_sha256=plan_sha, role=role,
                     started_at=datetime.now(timezone.utc).isoformat(),
                     execution=plan["runs"][role]["execution"], arrays=arrays,
                     ordered_ids=ordered, output=role_dir, g1=role_g1)
        receipts[role] = role_dir / "role-receipt.json"
    result = aggregate(plan_path, receipts, root)
    assert result["status"] == "complete_gate_artifacts"
    plan_copy = root / "g1_g2.tgcir.dress.v3-plan.json"
    native_path = root / "g1_g2.tgcir.dress.native.json"
    native_artifact = json.loads(native_path.read_text())
    assert native_artifact["parity_v3"]["plan_path"] == plan_copy.name
    assert plan_copy.read_bytes() == plan_path.read_bytes()
    assert len(result["controls"]) == 5
    assert native_artifact["g1_observation"]["native_adapter_native"]["coverage"] == {
        "query": {"captured_count": 1, "population_count": 2, "scope": "sample"},
        "gallery": {"captured_count": 2, "population_count": 3, "scope": "sample"},
    }
    adapter_artifact = json.loads((root / "g1_g2.tgcir.dress.adapter.json").read_text())
    assert adapter_artifact["g1_observation"] == native_artifact["g1_observation"]
    verdict = parity_gate.evaluate_model("tgcir", categories=("dress",), root=root,
                                         standard=parity_gate.STANDARD_ABS_REL_V3)
    assert verdict.status == "PARITY_MANDATORY_SUBCHECK_FAILED"
    failed = {check.name for check in verdict.failures}
    assert failed == {"dress.identity.gallery_order_identical",
                      "dress.identity.model_text_identical_to_author_composition",
                      "dress.identity.reference_ids_identical", "dress.identity.target_ids_identical"}
    with pytest.raises(ValueError, match="exactly ten"):
        aggregate(plan_path, {key: value for key, value in receipts.items() if key != "adapter_batch_n"},
                  tmp_path / "incomplete")

def test_tgcir_score_rank_proof_binds_labels_and_full_orders(tmp_path: Path):
    import hashlib
    import json
    import numpy as np
    from workbench.replay.tgcir_pair import score_proof_identity, write_score_rank_proof

    queries = ["dress_q1", "dress_q2"]
    gallery = ["dress_ref1", "dress_target1", "dress_ref2", "dress_target2"]
    targets, refs = [1, 3], [0, 2]
    target_ids = [gallery[i] for i in targets]
    ref_ids = [gallery[i] for i in refs]
    identity = score_proof_identity("dress", "a" * 64, {}, "runtime-pin", queries, gallery,
                                    target_ids, ref_ids)
    native_scores = np.array([[0.0, 0.9, 0.1, 0.2], [0.2, 0.1, 0.3, 0.8]], dtype=np.float32)
    adapter_scores = native_scores.copy()
    native_order = np.argsort(-native_scores, axis=1)
    adapter_order = native_order.copy()
    path, proof_sha, labels_sha = write_score_rank_proof(
        tmp_path, identity, queries, gallery, targets, refs,
        native_scores, adapter_scores, native_order, adapter_order)

    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    assert hashlib.sha256(raw).hexdigest() == proof_sha
    assert records[0]["schema"] == "cir.score-ranks/v1"
    proof_identity = records[0]["identity"]
    assert proof_identity["category"] == "dress"
    assert proof_identity["query_count"] == len(queries)
    assert proof_identity["gallery_count"] == len(gallery)
    assert len(proof_identity["query_order_sha256"]) == 64
    assert len(proof_identity["gallery_order_sha256"]) == 64
    assert proof_identity["environment"] == "runtime-pin"
    assert records[0]["identity"]["reference_exclusions_sha256"] == labels_sha
    assert records[1]["native_scores"] == native_scores[0].tolist()
    assert records[2]["adapter_order"] == adapter_order[1].tolist()

def test_tgcir_protocol_is_val_split():
    assert MODEL["native_protocol"] == "fashioniq_val_split"


def test_tgcir_test_function_signature_and_gallery_policy():
    """Verify test.py:6 signature (params, model, testset, category) and -10e10 reference exclusion."""
    tgcir_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "TGCIR"
    test_py = tgcir_source / "test.py"
    assert test_py.is_file()

    content = test_py.read_text()
    assert "def test(params, model, testset, category):" in content
    assert "sims[i, test_targets_id.index(t['source_img_id'])] = -10e10" in content


def test_tgcir_datasets_pair_union():
    """Verify datasets.py constructs pair union gallery (endpoints only, not full split)."""
    tgcir_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "TGCIR"
    datasets_py = tgcir_source / "datasets.py"
    assert datasets_py.is_file()

    content = datasets_py.read_text()
    # Check that test_targets_id is constructed from source_img_id and target_img_id in test_queries
    assert "test_targets_id = []" in content
    assert "i['source_img_id'] not in test_targets_id" in content
    assert "i['target_img_id'] not in test_targets_id" in content


def test_tgcir_preflight_blocks_empty_environment(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(source_dir, data_dir, ckpt_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("raw input missing" in b.lower() for b in blockers)
    assert any("checkpoint missing or empty" in b.lower() for b in blockers)
    # The replacement contract, both halves: the blocker must come from the BYTES, and a registry
    # that is not VERIFIED_METADATA must still block even when the file is present and hashed.
    assert not any("mapping unresolved" in b.lower() for b in blockers)


def test_tgcir_preflight_fails_closed_on_a_wrong_digest(tmp_path: Path):
    """A staged file whose bytes differ from the declared digest is refused, not warned about."""
    import hashlib
    from unittest.mock import patch

    from workbench.replay import tgcir as module

    (tmp_path / module.CHECKPOINT["filename"]).write_bytes(b"not the author's weights")
    variant = dict(module.CHECKPOINT)
    variant["expected_sha256"] = hashlib.sha256(b"the author's weights").hexdigest()
    variant["expected_size_bytes"] = None
    with patch.object(module, "CHECKPOINT", variant):
        blockers = module.validate_checkpoint(tmp_path)
    assert any("digest mismatch" in b for b in blockers), blockers


def test_tgcir_preflight_still_blocks_when_the_registry_is_not_verified(tmp_path: Path):
    """A verified digest plus an unverified registry status is still a refusal."""
    import hashlib
    from unittest.mock import patch

    from workbench.replay import tgcir as module

    payload = b"author bytes"
    (tmp_path / module.CHECKPOINT["filename"]).write_bytes(payload)
    variant = dict(module.CHECKPOINT)
    variant["expected_sha256"] = hashlib.sha256(payload).hexdigest()
    variant["expected_size_bytes"] = len(payload)
    variant["checkpoint_mapping_status"] = "UNVERIFIED"
    with patch.object(module, "CHECKPOINT", variant):
        blockers = module.validate_checkpoint(tmp_path)
    assert any("mapping unresolved" in b.lower() for b in blockers), blockers


def test_tgcir_evaluate_categories_returns_native_metrics_and_complete_macro(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from workbench.replay import tgcir

    class Model:
        def modules(self):
            return []

        def to(self, _device):
            return self

        def eval(self):
            return self

    class Torch:
        class nn:
            class MultiheadAttention:
                pass

        class Tensor:
            pass

        class cuda:
            @staticmethod
            def is_available():
                return False

    class Clip:
        @staticmethod
        def load(_name, *, device, jit):
            assert device == "cpu" and jit is False
            return object(), object()

    dataset = object()
    calls = []

    class Datasets:
        @staticmethod
        def FashionIQ(*, path, transform):
            assert path.endswith("/") and transform is not None
            return dataset

    class Evaluator:
        @staticmethod
        def test(params, model, testset, category):
            calls.append((params.batch_size, model, testset, category))
            base = {"dress": 10, "shirt": 20, "toptee": 30}[category]
            return [(f"{category}_r{k}", base + k) for k in (1, 10, 50)]

    context_receipts = []
    def image_context(*args):
        context_receipts.append(args)
        return _ReceiptContext()
    class _ReceiptContext:
        def __enter__(self):
            self.receipt = {"expected_image_tensors": 5, "stored_image_tensors": 0}
            return self.receipt
        def __exit__(self, *_exc):
            self.receipt.update(stored_image_tensors=5, spool_file_removed=True)
            return False
    def model_context(model, categories):
        assert isinstance(model, Model)
        assert categories == CATEGORIES
        return nullcontext({"captured": True})
    monkeypatch.setattr(tgcir, "_load_tgcir", lambda *_: (Torch, (Clip, Datasets, Evaluator), []))
    monkeypatch.setattr(Torch, "load", lambda *_args, **_kwargs: Model(), raising=False)
    monkeypatch.setattr(tgcir, "run_cache_root", lambda *_args, **_kwargs: nullcontext(tmp_path))
    monkeypatch.setattr(tgcir, "cache_requirement", lambda *_args, **_kwargs: {})

    results, aggregate = tgcir.evaluate_categories(
        tmp_path, tmp_path, tmp_path, batch_size=7, device="cpu",
        image_context_factory=image_context, model_context_factory=model_context)

    assert len(context_receipts) == 1
    assert len(context_receipts[0]) == 3
    assert results[0].provenance["image_spool"]["stored_image_tensors"] == 5
    assert results[0].provenance["image_spool"]["spool_file_removed"] is True
    assert results[0].provenance["native_model_observation"]["captured"] is True

    assert [row.category for row in results] == list(CATEGORIES)
    assert [(row.r1, row.r10, row.r50, row.complete) for row in results] == [
        (11, 20, 60, True), (21, 30, 70, True), (31, 40, 80, True)]
    assert aggregate == {"macro_r10": 30, "macro_r50": 70, "macro_mean": 50, "complete": True}
    assert [call[0] for call in calls] == [7, 7, 7]
