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
    adapter.capture_path = None
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



def test_capture_selector_preserves_requested_ids_and_refuses_unknown_ids() -> None:
    from workbench.replay.clvc_net import select_capture_rows

    rows = [{"sample_id": "b", "value": 2}, {"sample_id": "a", "value": 1}]
    assert select_capture_rows(rows, ("a", "b")) == [rows[1], rows[0]]
    with pytest.raises(ValueError, match="absent from native rows"):
        select_capture_rows(rows, ("missing",))


def test_clvc_reference_exclusions_bind_each_native_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    rows = [
        {"candidate": "source-a", "target": "target-a", "captions": ["a", "b"]},
        {"candidate": "source-b", "target": "target-b", "captions": ["c", "d"]},
    ]
    monkeypatch.setattr(plans, "_read_json", lambda *args: rows)
    assert plans._clvc_reference_exclusions("dress", ["q-a", "q-b"]) == [
        {"query_id": "q-a", "reference_id": "source-a", "score": -10e10},
        {"query_id": "q-b", "reference_id": "source-b", "score": -10e10},
    ]


def test_native_plan_gallery_preserves_source_order_and_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    captions = {
        "dress": [{"candidate": "a", "target": "b", "captions": ["one", "two"]}],
        "shirt": [],
        "toptee": [],
    }
    splits = {"dress": ["a", "b"], "shirt": ["c", "shared"],
              "toptee": ["shared", "d"]}

    def read_json(path: Path, label: str) -> list:
        category = path.stem.split(".")[1]
        return captions[category] if "captions" in path.parts else splits[category]

    monkeypatch.setattr(plans, "_read_json", read_json)
    query_ids, clvc_gallery = plans._clvc_rows("dress")
    assert len(query_ids) == 1
    assert clvc_gallery == ["a", "b"]
    assert plans._tme_rows("dress")[1] == ["a", "b"]
    assert plans._tme_rows("shirt")[1] == ["c", "shared"]
    assert plans._tme_rows("toptee")[1] == ["shared", "d"]


def test_direct_tme_adapter_role_uses_frozen_category_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from types import SimpleNamespace
    from workbench.composition import model_tme
    from workbench.scripts import clvc_tme_v3_parity as producer

    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True)
    (data / "images").mkdir()
    rows = [{"candidate": "ref", "target": "target", "captions": ["long sleeve.", "brighter"]}]
    (data / "captions/cap.dress.val.json").write_text(json.dumps(rows))
    for name in ("ref", "target"):
        (data / "images" / f"{name}.jpg").touch()
    from workbench.replay import clvc_tme_parity

    monkeypatch.setattr(clvc_tme_parity, "AdapterArtifactSink",
                        lambda *args, **kwargs: lambda **sink_args: None)
    from workbench.scripts.clvc_tme_v3_parity import _ordered_ids_sha256
    fixture_query_ids, fixture_gallery_ids = ["query"], ["ref", "target"]
    identity = {"query_count": 1, "gallery_count": 2,
                "query_order_sha256": _ordered_ids_sha256(fixture_query_ids),
                "gallery_order_sha256": _ordered_ids_sha256(fixture_gallery_ids)}
    plan = {"model_id": "tme", "category": "dress", "identity": identity,
            "runs": {"native_adapter_adapter": {"side": "adapter", "run_id": "run",
                "source_id": "adapter:source", "identity": identity, "execution": {
                    "partition_id": "p", "query_partition_sha256": "d",
                    "gallery_partition_sha256": "e", "query_batch_size": 2,
                    "gallery_batch_size": 2}}}}
    monkeypatch.setattr("workbench.replay.prepare_clvc_tme_v3_plans._tme_rows",
                        lambda category: (["query"], ["ref", "target"]))
    monkeypatch.setattr("workbench.backend.operator_config.resolve_config",
                        lambda: SimpleNamespace(FASHIONIQ_ROOT=data))
    monkeypatch.setattr(producer, "_plan", lambda path: (plan, "f" * 64))

    class FakeAdapter:
        def __init__(self, config, checkpoint_id):
            assert checkpoint_id == "fashioniq_0"
            self.work_root = None
            self.batch_size = 0

        def require_verified_interpreter(self):
            return "/verified/tme/python"

        def encode_queries(self, paths, texts, categories):
            assert [path.name for path in paths] == ["ref.jpg"]
            assert texts == ["Long sleeve and brighter"]
            return np.ones((1, 256), dtype=np.float32)

        def encode_targets(self, paths, categories):
            assert [path.name for path in paths] == ["ref.jpg", "target.jpg"]
            return np.ones((2, 32, 256), dtype=np.float32)

        def score(self, queries, targets):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def close(self):
            pass
    monkeypatch.setattr(model_tme, "Adapter", FakeAdapter)
    destination = tmp_path / "out.npz"
    producer.run_adapter(tmp_path / "plan.json", "native_adapter_adapter", destination)
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["query_ids"].tolist() == ["query"]
        assert archive["gallery_ids"].tolist() == ["ref", "target"]
        assert archive["scores"].tolist() == [[1.0, 0.0]]
        assert archive["ranks"].tolist() == [2]
    receipt = json.loads(destination.with_suffix(".run.json").read_text())
    assert receipt["worker_interpreter"] == "/verified/tme/python"

def test_native_replay_command_passes_invocation_id_inside_lock_child() -> None:
    import subprocess
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    commands = [
        [sys.executable, "-c", (
            "import argparse; p=argparse.ArgumentParser(); "
            "p.add_argument('--invocation-id', required=True); "
            "v=p.parse_args().invocation_id; "
            "print(v); assert v and len(v)==36")],
        [sys.executable, "-c", (
            "import argparse; p=argparse.ArgumentParser(); "
            "p.add_argument('--capture-selection', required=True); "
            "p.add_argument('--invocation-id', required=True); "
            "a=p.parse_args(); print(a.invocation_id); "
            "assert a.capture_selection=='selection.json' and len(a.invocation_id)==36")],
    ]
    for argv in commands:
        command = plans._native_replay_command(argv, Path(sys.executable))
        result = subprocess.run(["flock", "/tmp/cir-test-native-invocation.lock", "-c", command],
                                text=True, capture_output=True, check=True)
        assert len(result.stdout.strip()) == 36

def test_tme_family_commands_route_native_and_adapter_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    roles = {}
    for name in plans.ROLES:
        side = "native" if name.startswith("native_") and name != "native_adapter_adapter" else "adapter"
        full = name.startswith("native_adapter_")
        execution = {"query_batch_size": 1, "gallery_batch_size": 1}
        if not full:
            execution["sample_query_count"] = 1
            execution["sample_query_ids"] = [f"{name}-q"]
        roles[name] = {"side": side, "run_id": f"run-{name}", "source_id": f"source-{name}",
                       "identity": {}, "execution": execution}
    category_plans, plan_paths = {}, {}
    for category in plans.CATEGORIES:
        path = tmp_path / f"g1_g2.tme.{category}.v3-plan.json"
        path.write_text("{}\n")
        category_plans[category] = {
            "runs": roles,
            "commands": {"roles": {name: {"output": str(tmp_path / category / f"{name}.npz")}
                                     for name in plans.ROLES}},
            "identity": {},
        }
        plan_paths[category] = path
    monkeypatch.setattr(plans, "_tme_rows", lambda category: ([f"{category}-q"], ["ref", "target"]))
    monkeypatch.setattr(plans, "_g1_selection", lambda *args: {"queries": [], "gallery": []})
    monkeypatch.setattr("workbench.backend.operator_config.resolve_config", lambda: {})
    monkeypatch.setattr("workbench.backend.registry.model_by_id", lambda model: {"id": model})
    monkeypatch.setattr("workbench.backend.runtime.resolve_model_interpreter", lambda model, config: sys.executable)

    family = plans._tme_family(category_plans, plan_paths)
    assert set(family["roles"]) == set(plans.ROLES)
    assert set(family["commands"]) == set(plans.ROLES)
    for role_name in plans.ROLES:
        assert family["roles"][role_name]["side"] == roles[role_name]["side"]
        assert family["roles"][role_name]["query_ids"] == {
            category: roles[role_name]["execution"].get("sample_query_ids", [f"{category}-q"])
            for category in plans.CATEGORIES}
    for name, command in family["commands"].items():
        if name.startswith("adapter_") or name == "native_adapter_adapter":
            assert "--phase adapter --plan" in command
        else:
            assert "--phase adapter" not in command
        if name == "native_adapter_native":
            assert "--phase native" in command and "--phase bind-tme-family" in command
            assert "--capture-selection" in command
            assert "--sample-queries" not in command
        elif name.startswith("adapter_"):
            assert command.count("--phase adapter") == 3
            assert "--sample-queries" in command


def test_tme_score_and_control_captures_reach_real_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture-only producer-shaped score/control artifacts exercise aggregate through the gate."""
    import datetime
    import hashlib
    import importlib.util
    import json

    from workbench.backend import parity_gate as gate
    from workbench.replay import aggregate_fashioniq_v3 as aggregator
    from workbench.replay import prepare_clvc_tme_v3_plans as plans
    from workbench.replay.run_fashioniq_v3 import _v3_capture

    spec = importlib.util.spec_from_file_location(
        "fixture_parity_v3", Path(__file__).with_name("test_parity_absolute_relative_v2.py"))
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    t4 = fixture._producer("t4_parity")
    query_ids, gallery_ids = [f"q{i}" for i in range(4)], [f"q{i}" for i in range(4)] + [f"g{i}" for i in range(16)]
    sample_ids = query_ids[:2]
    monkeypatch.setattr(plans, "_tme_rows", lambda _: (query_ids, gallery_ids))
    monkeypatch.setattr(aggregator, "_t4", lambda: t4)

    root = tmp_path / "evidence"
    root.mkdir()
    captions = [{"candidate": f"ref{i}", "target": query_ids[i], "captions": ["blue", "soft"]}
                for i in range(len(query_ids))]
    query_source = root / "captions.json"
    query_source.write_text(json.dumps(captions))
    labels = {"query_targets": [[query_ids[i], query_ids[i]] for i in range(len(query_ids))],
              "reference_exclusions": [[query_id, []] for query_id in query_ids]}
    identity = {"checkpoint": hashlib.sha256(b"fixture weights").hexdigest(),
                "configuration": "cfg", "environment": "env", "protocol": "fashioniq_val_split",
                "category": "dress", "cohort_id": "fixture-tme", "precision": "float32", "seed": 0,
                "query_order_sha256": aggregator.hashlib.sha256(json.dumps(query_ids, separators=(",", ":")).encode()).hexdigest(),
                "gallery_order_sha256": aggregator.hashlib.sha256(json.dumps(gallery_ids, separators=(",", ":")).encode()).hexdigest(),
                "reference_exclusions_sha256": hashlib.sha256(json.dumps(labels, ensure_ascii=False,
                    separators=(",", ":")).encode()).hexdigest(),
                "query_count": len(query_ids), "gallery_count": len(gallery_ids)}
    roles = {}
    sample_digest = hashlib.sha256(json.dumps(sample_ids, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()
    gallery_digest = identity["gallery_order_sha256"]
    query_digest = identity["query_order_sha256"]
    for role in aggregator.ROLE_ORDER:
        full = role.startswith("native_adapter_")
        side = "adapter" if role.startswith("adapter_") or role == "native_adapter_adapter" else "native"
        qids = query_ids if full else sample_ids
        cohort_by_role = {"native_repeat_1": "native-repeat", "native_repeat_2": "native-repeat",
                          "adapter_repeat_1": "adapter-repeat", "adapter_repeat_2": "adapter-repeat",
                          "native_batch_1": "native-batch", "native_batch_n": "native-batch",
                          "adapter_batch_1": "adapter-batch", "adapter_batch_n": "adapter-batch"}
        cohort = identity["cohort_id"] if full else f"fixture-{cohort_by_role[role]}"
        role_identity = {**identity, "cohort_id": cohort,
                         "query_count": len(qids),
                         "query_order_sha256": hashlib.sha256(json.dumps(qids, ensure_ascii=False,
                             separators=(",", ":")).encode()).hexdigest()}
        batch = 1 if role.endswith("batch_1") else 8
        execution = {"partition_id": f"fixture:{'full' if full else 'sample'}",
                     "query_partition_sha256": role_identity["query_order_sha256"],
                     "gallery_partition_sha256": gallery_digest,
                     "query_batch_size": batch, "gallery_batch_size": batch}
        if not full:
            execution.update(sample_query_count=len(sample_ids), sample_query_ids=sample_ids)
        roles[role] = {"run_id": f"fixture-{role}", "side": side,
                       "source_id": f"{side}:fixture:{role}",
                       "identity": role_identity, "execution": execution}
    control_specs = {
        "native_repeat": ("native_repeat_1", "native_repeat_2"),
        "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2"),
        "native_adapter": ("native_adapter_native", "native_adapter_adapter"),
        "native_batch_1_vs_n": ("native_batch_1", "native_batch_n"),
        "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n"),
    }
    controls = {}
    for name, (ref_role, actual_role) in control_specs.items():
        qids = query_ids if name == "native_adapter" else sample_ids
        qdigest = query_digest if name == "native_adapter" else sample_digest
        cohort = roles[ref_role]["identity"]["cohort_id"]
        coverage = "full" if name == "native_adapter" else "sample"
        controls[name] = {"reference_run_id": roles[ref_role]["run_id"],
                          "actual_run_id": roles[actual_role]["run_id"],
                          "reference_side": roles[ref_role]["side"],
                          "dimension": "batch_parity" if "batch" in name else "forward_parity",
                          "dtype": "float32", "tolerance": {"atol": 1e-5, "rtol": 1e-4},
                          "scopes": {
                              "query": {"coverage": coverage, "population_count": len(query_ids),
                                  "sample_count": None if coverage == "full" else len(qids),
                                  "tensor_shape": [len(qids), 256], "component_shape": [len(qids), 256],
                                  "ordered_ids_sha256": qdigest, "cohort_id": cohort},
                              "gallery": {"coverage": "full", "population_count": len(gallery_ids),
                                  "sample_count": None, "tensor_shape": [len(gallery_ids), 32, 256],
                                  "component_shape": [len(gallery_ids), 32 * 256],
                                  "ordered_ids_sha256": gallery_digest, "cohort_id": cohort}}}
    plan = {"standard": gate.STANDARD_ABS_REL_V3, "model_id": "tme", "category": "dress",
            "identity": identity, "runs": roles, "controls": controls,
            "supported_batch_policy": {"supported": True, "batch_sizes": [1, 8]},
            "population_scope": {"coverage": "full", "query_count": len(query_ids),
                "gallery_count": len(gallery_ids), "query_ids": query_ids, "gallery_ids": gallery_ids,
                "reference_ids": [row["candidate"] for row in captions],
                "target_ids": [row["target"] for row in captions], "query_source": str(query_source)},
            "input_capture": {"query_ids": sample_ids, "gallery_ids": gallery_ids[:2]},
            "scoring_labels": labels,
            "scoring_proof_path": "g1_g2.tme.dress.native.score-ranks.jsonl",
            "tolerances": fixture.TOLERANCES,
            "score_order_conventions": {"native": "fp32_one_minus_score_ascending",
                                        "adapter": "fp32_descending_score"}}
    plan_path = root / "g1_g2.tme.dress.v3-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log = root / "roles.jsonl"
    metrics = {"recall_at1": 100.0, "recall_at5": 100.0, "recall_at10": 100.0,
               "recall_at50": 100.0, "acc": 100.0, "mean": 100.0}
    features_q = np.ones((len(query_ids), 256), np.float32)
    features_g = np.ones((len(gallery_ids), 32, 256), np.float32)
    with log.open("w", encoding="utf-8") as stream:
        for role, run in roles.items():
            qids = query_ids if role.startswith("native_adapter_") else sample_ids
            labels_at = np.asarray([int(qid[1:]) for qid in qids], np.int64)
            ranks = np.tile(np.arange(len(gallery_ids), dtype=np.int64), (len(qids), 1))
            for index, label in enumerate(labels_at):
                ranks[index] = np.roll(ranks[index], -int(label))
            scores = np.zeros((len(qids), len(gallery_ids)), np.float32)
            for row, order in enumerate(ranks):
                scores[row, order] = np.arange(len(gallery_ids), 0, -1, dtype=np.float32)
            qfeat = features_q[[query_ids.index(qid) for qid in qids]]
            arrays = {"query_features": (qfeat, qids), "target_features": (features_g.reshape(len(gallery_ids), -1), gallery_ids),
                      "scores": (scores, qids), "ranks": (ranks, qids), "labels": (labels_at, qids)}
            score_path = root / f"{plan_path.stem}.{role}.{run['side']}.npz"
            np.savez_compressed(score_path, query_ids=np.asarray(qids), gallery_ids=np.asarray(gallery_ids),
                query_features=qfeat, target_features=features_g.reshape(len(gallery_ids), -1),
                scores=scores, ranks=ranks, labels=labels_at, metrics=np.asarray(json.dumps(metrics)))
            receipt = _v3_capture(plan, role, run["side"], started, plan_path, plan_sha, arrays)
            receipt.update({"role": role, "started_at": started,
                            "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if role in ("native_adapter_native", "native_adapter_adapter"):
                from workbench.scripts.clvc_tme_v3_parity import _score_order_digest
                receipt["score_order_sha256"] = _score_order_digest(qids, scores, ranks)
                receipt["identity_evidence"] = {"reference_ids": plan["population_scope"]["reference_ids"],
                    "target_ids": plan["population_scope"]["target_ids"], "gallery_order_ids": gallery_ids,
                    "model_texts": ["Blue and soft"] * len(query_ids)}
                capture = {}
                for scope, ids in (("query", sample_ids), ("gallery", gallery_ids[:2])):
                    raw_ids = ids
                    arrays_in = {"transformed_pixels": np.ones((len(ids), 3, 2, 2), np.float32),
                                 "token_ids": np.ones((len(ids), 4), np.int64),
                                 "attention_mask": np.ones((len(ids), 36), np.int64),
                                 "processed_text": np.asarray(["processed"] * len(ids)),
                                 "raw_text": np.asarray(["Blue and soft"] * len(ids))}
                    capture_path = root / f"{role}.{scope}.npz"
                    np.savez_compressed(capture_path, ordered_ids=np.asarray(raw_ids), **arrays_in)
                    capture[scope] = {"path": str(capture_path), "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                        "ordered_ids": raw_ids, "arrays": {key: list(value.shape) for key, value in arrays_in.items()},
                        "array_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                         for key, value in arrays_in.items()},
                        "raw_text": arrays_in["raw_text"].tolist(), "processed_text": arrays_in["processed_text"].tolist()}
                receipt["input_capture"] = capture
            stream.write(json.dumps({"role": role, "v3_receipt": receipt}) + "\n")
            control_root = root / "control-captures" / role
            control_root.mkdir(parents=True, exist_ok=True)
            selected = query_ids if role.startswith("native_adapter_") else sample_ids
            qselected = features_q[[query_ids.index(qid) for qid in selected]]
            control_path = control_root / "dress.npz"
            np.savez_compressed(control_path, query_ids=np.asarray(selected), gallery_ids=np.asarray(gallery_ids),
                query_features=qselected, gallery_features=features_g)
            control_receipt = {"model_id": "tme", "category": "dress", "role": role,
                "side": run["side"], "run_id": run["run_id"], "source_id": run["source_id"],
                "plan_sha256": plan_sha, "identity": run["identity"], "execution": run["execution"],
                "invocation_id": f"fixture-{role}", "started_at": started,
                "source_capture_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "source_population": {"query_count": len(query_ids), "gallery_count": len(gallery_ids),
                                      "evaluation_scope": "full-source-evaluator", "selected_query_count": len(selected)},
                "capture_sha256": hashlib.sha256(control_path.read_bytes()).hexdigest(), "arrays": {}}
            with np.load(control_path, allow_pickle=False) as blob:
                control_receipt["arrays"] = {key: {"dtype": str(blob[key].dtype), "shape": list(blob[key].shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(blob[key]).tobytes()).hexdigest()}
                    for key in ("query_ids", "gallery_ids", "query_features", "gallery_features")}
            control_path.with_suffix(".run.json").write_text(json.dumps(control_receipt))

    result = aggregator.aggregate(plan_path, [log])
    assert result["status"] == "ARTIFACTS_ASSEMBLED"
    assert result["checks"]["all_five_control_feature_tensors_within_preregistered_tolerance"]
    verdict = gate.evaluate_model("tme", root=root, categories=("dress",),
                                  standard=gate.STANDARD_ABS_REL_V3)
    assert verdict.status == gate.PASS, verdict.detail
def test_clvc_tme_controls_share_full_population_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    monkeypatch.setattr(plans, "PLAN_ROOT", tmp_path)
    query_ids = [f"q{index}" for index in range(40)]
    monkeypatch.setattr(plans, "_clvc_rows", lambda category: (query_ids, ["a", "b"]))
    monkeypatch.setattr(plans, "_tme_rows", lambda category: (query_ids, ["a", "b"]))
    monkeypatch.setattr(plans, "_read_json", lambda *args: [{"candidate": "a", "target": "b"}] * 40)
    monkeypatch.setattr(plans, "_g1_selection", lambda model, category, queries, gallery: {"queries": [queries[0]], "gallery": [gallery[0]]})
    monkeypatch.setattr(plans, "_source_identity", lambda model: {"source.py": "a" * 64})
    monkeypatch.setattr(plans, "_environment_identity", lambda model: {"python": "/python", "python_sha256": "b" * 64})
    from workbench.backend import runtime
    from workbench.backend import operator_config
    from workbench.backend import registry
    monkeypatch.setattr(operator_config, "resolve_config", lambda: {})
    monkeypatch.setattr(registry, "model_by_id", lambda model_id: {"id": model_id})
    monkeypatch.setattr(runtime, "resolve_model_interpreter", lambda model, config: "/usr/bin/python3")
    for model_id in ("clvc_net", "tme"):
        hashes = {f"dress_{scope}_model.pt": "x" for scope in ("local", "global")}
        hashes["fashioniq_0_best_model.pth"] = "x"
        plan = plans._plan(model_id, "dress", hashes)
        partitions = {role["execution"]["partition_id"] for role in plan["runs"].values()}
        assert f"{model_id}:dress:full-population-v3" in partitions
        if model_id == "clvc_net":
            assert f"{model_id}:dress:sample32-v1" in partitions
            continue
        assert len(partitions) == 2
        for name, roles in plans.V3_CONTROL_ROLES.items():
            control = plan["controls"][name]
            reference, actual = (plan["runs"][role] for role in roles)
            assert control["reference_run_id"] == reference["run_id"]
            assert control["actual_run_id"] == actual["run_id"]
            assert control["reference_side"] == reference["side"]
            assert control["dimension"] == ("batch_parity" if "batch" in name else "forward_parity")
            assert set(control["scopes"]) == {"query", "gallery"}
            for scope_name, scope in control["scopes"].items():
                population = plan["identity"]["query_count" if scope_name == "query" else "gallery_count"]
                assert scope["coverage"] == ("full" if name == "native_adapter" or scope_name == "gallery" else "sample")
                assert scope["population_count"] == population
                assert scope["tensor_shape"] == [population, *scope["component_shape"]]
                assert scope["selected_tensor_shape"][0] == scope.get("sample_count", population)
                assert len(scope["ordered_ids_sha256"]) == 64
                assert scope["cohort_id"] == reference["identity"]["cohort_id"]
            assert reference["identity"]["cohort_id"] == actual["identity"]["cohort_id"]
            assert reference["identity"]["query_order_sha256"] == actual["identity"]["query_order_sha256"]

def test_g1_selection_links_queries_to_gallery_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    from workbench.replay import prepare_clvc_tme_v3_plans as plans

    rows = [{"candidate": "g0", "target": "g1"},
            {"candidate": "g2", "target": "g3"},
            {"candidate": "g4", "target": "g5"},
            {"candidate": "g6", "target": "g7"},
            {"candidate": "g8", "target": "g9"}]
    monkeypatch.setattr(plans, "_read_json", lambda *args: rows)
    selected = plans._g1_selection("clvc_net", "dress", [f"q{i}" for i in range(5)],
                                   [f"g{i}" for i in range(10)])
    assert selected == {"queries": ["q0", "q4"],
                        "gallery": ["g0", "g1", "g8", "g9"]}


def test_native_capture_binding_hashes_frozen_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from workbench.scripts import clvc_tme_v3_parity as producer

    plan = {"model_id": "tme", "category": "dress", "identity": {"category": "dress"},
            "runs": {"native_adapter_native": {"side": "native", "run_id": "run-1",
                "source_id": "native-source", "identity": {"category": "dress"},
                "execution": {"partition_id": "full"}}}}
    plan_sha = "a" * 64
    monkeypatch.setattr(producer, "_plan", lambda _: (plan, plan_sha))
    capture = tmp_path / "dress.npz"
    np.savez(capture, metadata=np.asarray(json.dumps({"invocation_id": "run-uuid"})))
    receipt_path = producer.bind_native(tmp_path / "plan.json", capture, "native_adapter_native")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["plan_sha256"] == plan_sha
    assert receipt["capture_sha256"] == producer._sha(capture)
    assert receipt["source_id"] == "native-source"
    assert receipt["invocation_id"] == "run-uuid"

def test_tme_family_binding_requires_one_nonempty_invocation_and_emits_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from workbench.replay import prepare_clvc_tme_v3_plans as plans
    from workbench.scripts import clvc_tme_v3_parity as producer

    category_plans, family_categories = {}, {}
    family = {"schema": "cir-tme-family-plan-v1", "roles": {"native_adapter_native": {
        "query_ids": {category: [f"{category}-q"] for category in plans.CATEGORIES}}},
        "categories": family_categories}
    for category in plans.CATEGORIES:
        identity = {"category": category, "query_count": 1, "gallery_count": 1,
                    "query_order_sha256": producer._ordered_ids_sha256([f"{category}-q"]),
                    "gallery_order_sha256": producer._ordered_ids_sha256(["gallery"])}
        plan = {"standard": producer.STANDARD, "model_id": "tme", "category": category,
                "identity": identity, "runs": {"native_adapter_native": {
                    "side": "native", "run_id": f"run-{category}", "source_id": f"source-{category}",
                    "identity": identity, "execution": {}}}}
        path = tmp_path / f"{category}.plan.json"
        path.write_text(json.dumps(plan))
        category_plans[category] = plan
        plan_sha = producer._sha(path)
        family_categories[category] = {"plan_path": str(path), "plan_sha256": plan_sha,
                                       "identity": plan["identity"], "query_ids": [f"{category}-q"]}
    monkeypatch.setattr(producer, "_plan", lambda path: (category_plans[Path(path).stem.split(".")[0]],
                                                           producer._sha(Path(path))))
    monkeypatch.setattr(plans, "_tme_rows", lambda category: ([f"{category}-q"], ["gallery"]))
    monkeypatch.setattr(producer, "_tme_rows", lambda category: ([f"{category}-q"], ["gallery"]))
    family_path = tmp_path / "family.json"
    family_path.write_text(json.dumps(family))

    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    for category in plans.CATEGORIES:
        invocation_id = "shared-run" if category != "shirt" else "different-run"
        np.savez(capture_root / f"{category}.npz", query_ids=[f"{category}-q"],
                 gallery_ids=["gallery"], scores=np.asarray([[1.0]], dtype=np.float32),
                 native_order=np.asarray([[0]], dtype=np.int64),
                 query_features=np.ones((1, 256), dtype=np.float32),
                 gallery_features=np.ones((1, 32, 256), dtype=np.float32), metadata=np.asarray(json.dumps({
                     "invocation_id": invocation_id, "started_at": "2026-11-21T00:00:00+00:00",
                     "model_id": "tme", "category": category})))
    (capture_root / "result.json").write_text(json.dumps({"invocation_id": "shared-run"}))
    with pytest.raises(ValueError, match="do not share one native invocation"):
        producer.bind_tme_family(family_path, capture_root, "native_adapter_native")
    with np.load(capture_root / "shirt.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    metadata.pop("invocation_id")
    np.savez(capture_root / "shirt.npz", query_ids=["shirt-q"], gallery_ids=["gallery"],
             scores=np.asarray([[1.0]], dtype=np.float32), native_order=np.asarray([[0]], dtype=np.int64),
             metadata=np.asarray(json.dumps(metadata)))
    with pytest.raises(ValueError, match="lacks pre-run invocation ID: shirt"):
        producer.bind_tme_family(family_path, capture_root, "native_adapter_native")

    for category in plans.CATEGORIES:
        capture_root = tmp_path / "captures"
        with np.load(capture_root / f"{category}.npz", allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        metadata["invocation_id"] = "shared-run"
        np.savez(capture_root / f"{category}.npz", query_ids=[f"{category}-q"],
                 gallery_ids=["gallery"], scores=np.asarray([[1.0]], dtype=np.float32),
                 native_order=np.asarray([[0]], dtype=np.int64),
                 query_features=np.ones((1, 256), dtype=np.float32),
                 gallery_features=np.ones((1, 32, 256), dtype=np.float32),
                 metadata=np.asarray(json.dumps(metadata)))
    receipts = producer.bind_tme_family(family_path, tmp_path / "captures", "native_adapter_native")
    assert len(receipts) == len(plans.CATEGORIES)
    assert [json.loads(path.read_text())["invocation_id"] for path in receipts] == ["shared-run"] * 3
    native_control = tmp_path / "control-captures/native_adapter_native/dress.npz"
    with np.load(native_control, allow_pickle=False) as control:
        assert control["gallery_features"].shape == (1, 32, 256)
    native_receipt = json.loads(native_control.with_suffix(".run.json").read_text())
    assert native_receipt["role"] == "native_adapter_native"
    assert native_receipt["side"] == "native"
    with pytest.raises(FileExistsError):
        producer.bind_tme_family(family_path, tmp_path / "captures", "native_adapter_native")

def test_tme_control_capture_projects_ordered_queries_and_keeps_3d_gallery(tmp_path: Path) -> None:
    import json
    from workbench.scripts import clvc_tme_v3_parity as producer

    source = tmp_path / "source.npz"
    output = tmp_path / "native_batch.npz"
    query_ids = ["q0", "q1", "q2"]
    gallery_ids = ["g0", "g1"]
    queries = np.arange(3 * 256, dtype=np.float32).reshape(3, 256)
    gallery = np.arange(2 * 32 * 256, dtype=np.float32).reshape(2, 32, 256)
    np.savez(source, query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
             query_features=queries, gallery_features=gallery)
    selected = ["q0", "q2"]
    role = {"run_id": "run-1", "source_id": "native-source", "identity": {
        "query_count": 2, "gallery_count": 2,
        "query_order_sha256": producer._ordered_ids_sha256(selected),
        "gallery_order_sha256": producer._ordered_ids_sha256(gallery_ids)},
        "execution": {"query_batch_size": 1}}
    receipt_path = producer.write_tme_control_capture(
        source, output, role=role, role_name="native_batch_1", plan_sha="a" * 64,
        category="dress", query_ids=selected, gallery_ids=gallery_ids, side="native",
        invocation_id="same-eval", started_at="2026-11-21T00:00:00+00:00")
    with np.load(output, allow_pickle=False) as capture:
        assert capture["query_ids"].tolist() == selected
        assert capture["gallery_features"].shape == (2, 32, 256)
        assert np.array_equal(capture["query_features"], queries[[0, 2]])
        assert np.array_equal(capture["gallery_features"], gallery)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["role"] == "native_batch_1" and receipt["side"] == "native"
    assert receipt["arrays"]["gallery_features"]["shape"] == [2, 32, 256]
    assert receipt["source_capture_sha256"] == producer._sha(source)

def test_adapter_artifact_sink_streams_memmaps_and_binds_identity(tmp_path: Path) -> None:

    from workbench.replay import clvc_tme_parity

    query_ids = ("q0", "q1")
    gallery_ids = ("g0", "g1", "g2")
    digest = clvc_tme_parity._canonical_digest
    identity = {
        "checkpoint": "sha256:" + "a" * 64,
        "configuration": "cfg-1",
        "environment": "env-1",
        "protocol": "fashioniq-val",
        "category": "dress",
        "cohort_id": "cohort-1",
        "precision": "float32",
        "seed": 0,
        "query_order_sha256": digest(query_ids),
        "gallery_order_sha256": digest(gallery_ids),
        "reference_exclusions_sha256": "b" * 64,
        "query_count": len(query_ids),
        "gallery_count": len(gallery_ids),
    }
    features = np.lib.format.open_memmap(tmp_path / "features.npy", mode="w+",
                                         dtype=np.float32, shape=(3, 4))
    features[:] = np.arange(12, dtype=np.float32).reshape(3, 4)
    queries = np.lib.format.open_memmap(tmp_path / "queries.npy", mode="w+",
                                        dtype=np.float32, shape=(2, 4))
    queries[:] = np.arange(8, dtype=np.float32).reshape(2, 4)
    plan_path = Path("g1_g2.clvc_net.dress.v3-plan.json")
    role = {"run_id": "run-1", "source_id": "source-1", "plan_path": plan_path,
            "plan_sha256": "c" * 64, "identity": identity}
    sink = clvc_tme_parity.AdapterArtifactSink(
        tmp_path / "out", parity_run_for=lambda *_: role)
    sink(model_id="clvc_net", category="dress", condition="image_only",
         run_id="run-1", plan_path=plan_path, plan_sha256="c" * 64,
         started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), identity=identity, source_id="source-1",
         query_ids=query_ids, gallery_ids=gallery_ids, query_features=queries,
         gallery_features=features, ranks={"q0": 2, "q1": 1}, metrics={"r1": 0.5},
         execution={"partition_id": "p0", "query_partition_sha256": "d" * 64,
                    "gallery_partition_sha256": "e" * 64, "query_batch_size": 2,
                    "gallery_batch_size": 3})
    archive = np.load(tmp_path / "out/adapter/clvc_net/dress.image_only.npz", allow_pickle=False)
    assert archive["query_ids"].tolist() == list(query_ids)
    assert np.array_equal(archive["gallery_features"], features)
    assert archive["rank_qids"].tolist() == list(query_ids)
    assert archive["ranks"].tolist() == [2, 1]
    import json

    document = json.loads((tmp_path / "out/adapter/clvc_net/dress.image_only.json").read_text())
    assert document["identity"] == identity
    assert document["metrics"] == {"r1": 0.5}
    native_dir = tmp_path / "out/native/clvc_net"
    native_dir.mkdir(parents=True)
    native_json = native_dir / "dress.image_only.json"
    native_npz = native_dir / "dress.image_only.npz"
    native_doc = dict(document, source_id="native-source")
    native_json.write_text(json.dumps(native_doc))
    import shutil

    shutil.copyfile(tmp_path / "out/adapter/clvc_net/dress.image_only.npz", native_npz)
    comparison = clvc_tme_parity.compare_artifacts(native_json,
        tmp_path / "out/adapter/clvc_net/dress.image_only.json", atol=0, rtol=0)
    assert comparison["status"] == "PASS"
    assert comparison["rank_changed_count"] == 0
    assert comparison["feature_comparison"]["query_features"]["mismatch_count"] == 0

def test_tme_mmap_cache_preserves_source_indexing_and_category_gallery(tmp_path: Path) -> None:
    import json
    import torch
    from workbench.replay.tme import _load_feature_cache

    cache = tmp_path / "val_images.pt"
    mapping_path = tmp_path / "val_name_to_idx.json"
    cached_names = ["shared", "dress", "shirt", "shared", "toptee"]
    values = torch.arange(len(cached_names) * 2, dtype=torch.float32).reshape(len(cached_names), 2)
    torch.save(values, cache)
    mapping = {name: index for index, name in enumerate(cached_names)}
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    loaded, loaded_mapping = _load_feature_cache(cache, mapping_path)

    assert tuple(loaded.shape) == (5, 2)
    assert loaded_mapping == {"shared": 3, "dress": 1, "shirt": 2, "toptee": 4}
    from workbench.third_party.TME.src.data import FashionIQ

    dataset = FashionIQ.__new__(FashionIQ)
    dataset.split = "val"
    dataset.type_to_triplets = {category: [category] for category in ("dress", "shirt", "toptee")}
    dataset.image_type_names = {"dress": ["dress", "shared"],
                                "shirt": ["shirt", "shared"],
                                "toptee": ["toptee"]}
    dataset.load_image_features_tensor_and_dict(loaded, loaded_mapping)
    dataset.set_specific_dress_type("dress")
    assert dataset.current_triplets == ["dress"]
    assert dataset.current_image_names == ["dress", "shared"]
    assert torch.equal(dataset.get_image_features(dataset.current_image_names), values[[1, 3]])
    assert dataset.name_to_idx["shared"] == 3

def test_tme_observer_archives_selected_rows_across_native_batches(tmp_path: Path, monkeypatch) -> None:
    import json
    import sys
    import types
    import torch
    from workbench.replay import tme

    rows = [{"candidate": f"ref{i}", "target": f"image{i}", "captions": [f"c{i}", "d"]}
            for i in range(3)]
    class Dataset:
        current_triplets = rows
        current_image_names = [f"image{i}" for i in range(4)]

    class Projection:
        def register_forward_hook(self, hook):
            self.hook = hook
            return types.SimpleNamespace(remove=lambda: None)

    class Model:
        vision_proj = Projection()
        def encode_image(self, images):
            output = torch.as_tensor(images, dtype=torch.float32)
            self.vision_proj.hook(self.vision_proj, (), output)
            return output
        def encode_fusion(self, reference, tokens):
            return reference
        def inference(self, references, gallery, captions):
            tokens = types.SimpleNamespace(input_ids=torch.ones((len(captions), 3), dtype=torch.long),
                                           attention_mask=torch.ones((len(captions), 3), dtype=torch.long))
            fused = self.encode_fusion(references, tokens)
            scores = fused @ gallery.T
            return scores[0] if len(captions) == 1 else scores

    def native_metric(model, dataset, preprocessor, category):
        model.encode_image(torch.eye(4)[:2])
        model.encode_image(torch.eye(4)[2:])
        distances = []
        for first in (0, 2):
            count = min(2, len(rows) - first)
            scores = model.inference(torch.ones((count, 4)), torch.eye(4), ["x"] * count)
            distances.append(1 - scores)
        torch.argsort(torch.vstack(distances), dim=-1)
        return {"recall_at1": 100.0}

    monkeypatch.setitem(sys.modules, "precompute_evaluation",
                        types.SimpleNamespace(compute_metric_features=native_metric))
    selected = {tme._tme_source_id("dress", rows[index]) for index in (0, 2)}
    model = Model()
    dataset = Dataset()
    restore = tme._install_tme_observer(model, dataset, tmp_path,
                                        {"dress": selected}, {}, "run-test")
    try:
        sys.modules["precompute_evaluation"].compute_metric_features(model, dataset, None, "dress")
    finally:
        restore()
    with np.load(tmp_path / "dress.npz", allow_pickle=False) as capture:
        query_ids = capture["selected_query_ids"].tolist()
        assert query_ids == [tme._tme_source_id("dress", rows[index]) for index in (0, 2)]
        assert query_ids == capture["selected_scores_row_ids"].tolist()
        assert query_ids == capture["selected_fusion_features_row_ids"].tolist()
        assert capture["scores"].shape == (3, 4)
        assert capture["selected_scores"].shape == (2, 4)
        assert capture["selected_fusion_features"].shape == (2, 4)
        assert capture["selected_input_ids"].shape == (2, 3)
        metadata = json.loads(str(capture["metadata"].item()))
        assert metadata["invocation_id"] == "run-test"
        assert [batch["input_shape"] for batch in metadata["selected_score_batches"]] == [[2, 4], [4]]
        assert [batch["selected_query_ids"] for batch in metadata["selected_score_batches"]] == [
            [query_ids[0]], [query_ids[1]]]

def test_tme_evaluation_restores_cwd_when_native_evaluation_fails(tmp_path: Path, monkeypatch) -> None:
    from workbench.replay import tme

    source = tmp_path / "source"
    source.mkdir()
    original = Path.cwd()
    monkeypatch.setattr(tme, "source_root_for", lambda _: source)
    monkeypatch.setattr(tme, "_prior_feature_provenance", lambda _: {})

    def fail(*args, **kwargs):
        raise RuntimeError("native evaluation failed")

    monkeypatch.setattr(tme, "_evaluate_categories", fail)
    with pytest.raises(RuntimeError, match="native evaluation failed"):
        tme.evaluate_categories(source, tmp_path, tmp_path, config=object())
    assert Path.cwd() == original

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
        clvc._run(adapter, "targets", [str(tmp_path / "a.jpg")], (), ["dress"],
                  capture_ids=None, selected_capture_ids=None)
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
              sent, ["dress", "dress"], capture_ids=None, selected_capture_ids=None)
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


# --------------------------------------------------- the unsupported condition is DECLARED, not failed

def test_the_empty_text_condition_is_declared_unsupported_by_name() -> None:
    """`mode_supported` must answer False for `image_only_empty_text`: the runner SKIPS a condition
    it answers False for (runner.py:1920-1925) and records `ABLATION_UNSUPPORTED`, so declaring it
    is what stops 6003 guaranteed failures from being run and reported as a plan that lied.
    The cause is the model's text tower (`text_model.py:84-86`), and the reason must say so.
    """
    adapter = clvc.Adapter.__new__(clvc.Adapter)
    adapter._model = {"model_id": "clvc_net",
                      "environment": {"python_env_var": "WORKBENCH_CLVC_PYTHON"}}
    adapter._checkpoint = {"checkpoint_id": "x", "checkpoint_mapping_status": "VERIFIED_METADATA",
                           "bundle_members": [{"filename": f"{c}_{m}_model.pt",
                                               "expected_sha256": "0" * 64}
                                              for c in clvc.CATEGORIES for m in ("local", "global")]}
    adapter.checkpoint_path = Path("/nonexistent")
    adapter._interpreter = None
    adapter.config = None
    supported, reason = adapter.mode_supported("image_only_empty_text")
    assert supported is False
    assert reason is not None and clvc.EMPTY_TEXT_REASON in reason
    # The other two conditions are unaffected by the declaration.
    assert clvc._mode_support("multimodal", None) == (True, None)
    assert clvc._mode_support("text_only_black_image", None) == (True, None)
    assert clvc._mode_support("not-a-condition", None)[0] is False

def test_status_per_mode_agrees_with_mode_supported_for_the_empty_condition() -> None:
    """One predicate, so the status row cannot say True where `mode_supported` says False."""
    sealed = clvc._mode_support("image_only_empty_text", None)
    assert sealed[0] is False and sealed[1] is not None and clvc.EMPTY_TEXT_REASON in sealed[1]
    assert clvc._mode_support("image_only_empty_text", "structural x") == (
        False, f"structural x; {clvc.EMPTY_TEXT_REASON}")
