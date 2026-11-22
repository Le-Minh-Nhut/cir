"""Composition cache identity must bind model provenance.

Upstream commit, approved source patches, and vendored source modifications
must invalidate the composition run identity so stale ranks are not reused.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
from PIL import Image

from workbench.composition.adapters import CompositionAdapter
from workbench.composition.data import Gallery, Query, ReleaseData
from workbench.composition.runner import ModelSpec, run_composition


class _DummyAdapter(CompositionAdapter):
    """Minimal composition adapter satisfying runner contract."""

    model_id = "testmodel"

    def encode_queries(
        self,
        image_paths: Sequence[str | Path],
        texts: Sequence[str],
        categories: Sequence[str | None],
    ) -> np.ndarray:
        return np.ones((len(image_paths), 2), dtype=np.float32)

    def encode_targets(
        self,
        image_paths: Sequence[str | Path],
        categories: Sequence[str | None],
    ) -> np.ndarray:
        return np.ones((len(image_paths), 2), dtype=np.float32)


def _fixture(tmp_path: Path) -> tuple[ReleaseData, SimpleNamespace]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    third_party = tmp_path / "third_party"
    third_party.mkdir()

    img_path = dataset / "img1.png"
    Image.new("RGB", (10, 10), "red").save(img_path)

    gallery = Gallery(protocol_id="circus_fashioniq_release_v1", image_ids=("img1",))
    queries = (
        Query(
            qid="q1",
            index=0,
            query_txt="query",
            reference_id="img1",
            positive_ids=("img1",),
            category=None,
            subsets=("full",),
        ),
    )
    candidate_names = {"img1": "img1.png"}
    data = ReleaseData(
        gallery=gallery,
        queries=queries,
        candidate_names=candidate_names,
        provenance={"is_synthetic": True},
    )
    config = SimpleNamespace(
        WORKBENCH_ARTIFACTS_ROOT=str(artifacts),
        FASHIONIQ_ROOT=str(dataset),
        WORKBENCH_THIRD_PARTY_ROOT=str(third_party),
    )
    return data, config


def _resolver(identity: str, candidate_names: dict[str, str], dataset: Path) -> Path | None:
    filename = candidate_names.get(identity, identity)
    p = dataset / filename
    return p if p.is_file() else None


def test_loaded_code_identity_is_stable_across_disk_rewrite_and_tracks_names(tmp_path: Path) -> None:
    import sys
    from types import ModuleType

    from workbench.composition.runner import _loaded_module_source_hash

    module_path = tmp_path / "loaded_identity_fixture.py"
    module_path.write_text("VALUE = 1\ndef forward(x):\n    return x + VALUE, ('nested', (1, 2))\n")
    loaded = ModuleType("loaded_identity_fixture")
    loaded.__file__ = str(module_path)
    sys.modules[loaded.__name__] = loaded
    try:
        exec(compile(module_path.read_text(), str(module_path), "exec"), loaded.__dict__)
        before = _loaded_module_source_hash(loaded)
        module_path.write_text("VALUE = 2\ndef forward(x):\n    return x + VALUE, ('rewritten', (1, 2))\n")
        assert _loaded_module_source_hash(loaded) == before
        exec(compile("def forward(x):\n    return x + RENAMED, ('nested', (1, 2))\n",
                     str(module_path), "exec"), loaded.__dict__)
        assert _loaded_module_source_hash(loaded) != before
    finally:
        sys.modules.pop(loaded.__name__, None)


def test_upstream_commit_change_invalidates_composition_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda i, c: _resolver(i, c, dataset)

    spec_a = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "upstream_commit_sha": "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )
    spec_b = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "upstream_commit_sha": "1111111111111111111111111111111111111111",
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )

    rep_a = run_composition(data, [spec_a], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    rep_b = run_composition(data, [spec_b], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)

    run_id_a = rep_a["models"][0]["run_id"]
    run_id_b = rep_b["models"][0]["run_id"]
    assert run_id_a != run_id_b, "different upstream commit sha must invalidate composition identity"


def test_approved_patch_change_invalidates_composition_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda i, c: _resolver(i, c, dataset)

    spec_unpatched = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "upstream_commit_sha": "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
            "approved_source_patches": [],
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )
    spec_patched = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "upstream_commit_sha": "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
            "approved_source_patches": [
                {
                    "path": "src/lavis/models/__init__.py",
                    "patch_sha256": "0e703584a7ecb5df3781574c2516fd406745c414491b6aa487fbb3e76532b9bd",
                    "derived_sha256": "aa68fe5e20bcda5e17d978799deb8ec0ff4636e5b3b2a6ef637ce5e385b0c818",
                }
            ],
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )

    rep_u = run_composition(data, [spec_unpatched], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    rep_p = run_composition(data, [spec_patched], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)

    run_id_u = rep_u["models"][0]["run_id"]
    run_id_p = rep_p["models"][0]["run_id"]
    assert run_id_u != run_id_p, "different approved patches must invalidate composition identity"


def test_vendored_source_mutation_invalidates_composition_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda i, c: _resolver(i, c, dataset)

    model_src = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "CSMCIR"
    model_src.mkdir(parents=True)
    source_file = model_src / "model.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")

    spec = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "source_dir": "CSMCIR",
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )

    rep1 = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    run_id_1 = rep1["models"][0]["run_id"]

    # Mutate vendored file content
    source_file.write_text("VALUE = 2\n", encoding="utf-8")

    rep2 = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    run_id_2 = rep2["models"][0]["run_id"]

    assert run_id_1 != run_id_2, "mutated vendored source must invalidate composition identity"



def test_adapter_prepare_identity_invalidates_composition_cache(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda identity, candidates: _resolver(identity, candidates, dataset)
    layout = {"version": 1}

    class IdentityAdapter(_DummyAdapter):
        def prepare_identity(self) -> dict[str, Any]:
            return {"layout": dict(layout)}

    spec = ModelSpec(
        model_id="testmodel", checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: IdentityAdapter(cfg, ckpt),
        registry_model={"model_id": "testmodel", "fashioniq_layout": "fashioniq_standard",
                        "environment": {"python_env_var": None}},
        checkpoint_identity="dummy_ckpt_hash",
    )
    first = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"],
                            image_resolver=resolve)["models"][0]["run_id"]
    layout["version"] = 2
    second = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"],
                             image_resolver=resolve)["models"][0]["run_id"]

    assert first != second


def test_capture_selection_runs_production_encoders_and_partitions_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda identity, candidates: _resolver(identity, candidates, dataset)
    observed: list[tuple[str, list[str], list[str]]] = []

    class CapturingAdapter(_DummyAdapter):
        capture_path: Path | None = None

        def encode_queries(self, image_paths, texts, categories, *, capture_ids: Sequence[str] | None = None,
                           selected_capture_ids: Sequence[str] | None = None):
            observed.append(("queries", list(capture_ids or ()), list(selected_capture_ids or ())))
            if self.capture_path is not None:
                np.savez(self.capture_path, ids=np.asarray(selected_capture_ids or ()))
            return super().encode_queries(image_paths, texts, categories)

        def encode_targets(self, image_paths, categories, *, capture_ids: Sequence[str] | None = None,
                           selected_capture_ids: Sequence[str] | None = None):
            if self.capture_path is not None:
                assert capture_ids is not None and selected_capture_ids is not None
                observed.append(("gallery", list(capture_ids), list(selected_capture_ids)))
                np.savez(self.capture_path, ids=np.asarray(selected_capture_ids))
            return super().encode_targets(image_paths, categories)

    spec = ModelSpec(
        model_id="capturemodel", checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: CapturingAdapter(cfg, ckpt),
        registry_model={"model_id": "capturemodel", "fashioniq_layout": "fashioniq_standard",
                        "environment": {"python_env_var": None}},
        checkpoint_identity="capture-ckpt",
    )
    query_capture = run_composition(
        data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve,
        capture_selection={"queries": ["q1"]})["models"][0]
    gallery_capture = run_composition(
        data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve,
        capture_selection={"gallery": ["img1"]})["models"][0]
    mapped_capture = run_composition(
        data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve,
        capture_selection={"queries": ["mapped:q1"]},
        capture_id_for_query=lambda query, model_id, category: f"mapped:{query.qid}")["models"][0]

    assert observed == [("queries", ["q1"], ["q1"]), ("gallery", ["img1"], ["img1"]),
                        ("queries", [], []), ("queries", ["mapped:q1"], ["mapped:q1"])]
    assert query_capture["run_id"] != gallery_capture["run_id"] != mapped_capture["run_id"]


def test_parity_sink_receives_actual_ranked_memmaps_and_propagates_failure(tmp_path: Path) -> None:
    import hashlib
    import json

    from dataclasses import replace

    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    Image.new("RGB", (10, 10), "blue").save(dataset / "img2.png")
    data = replace(data, gallery=Gallery(data.gallery.protocol_id, ("img1", "img2")),
                   queries=(replace(data.queries[0], positive_ids=("img2",)),),
                   candidate_names={**data.candidate_names, "img2": "img2.png"})
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(b'{"frozen":true}\n')
    digest_ids = lambda values: hashlib.sha256(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    gallery_ids = list(data.gallery.image_ids)
    query_ids = [query.qid for query in data.queries]
    identity = {
        "checkpoint": "ckpt1", "configuration": "cfg", "environment": "python",
        "protocol": data.gallery.protocol_id, "category": "dress", "cohort_id": "full:test:dress",
        "precision": "float32", "seed": 0,
        "query_order_sha256": digest_ids(query_ids), "gallery_order_sha256": digest_ids(gallery_ids),
        "reference_exclusions_sha256": digest_ids([]), "query_count": len(query_ids),
        "gallery_count": len(gallery_ids),
    }
    execution = {"partition_id": "full:test:dress", "query_batch_size": 32,
                 "gallery_batch_size": 32, "query_partition_sha256": digest_ids(query_ids),
                 "gallery_partition_sha256": digest_ids(gallery_ids)}
    role = {"run_id": "test-run", "source_id": "source-sha", "plan_path": str(plan_path),
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "identity": identity, "execution": execution}

    query_calls: list[tuple[list[str], list[str], list[str | None]]] = []
    target_calls: list[list[str]] = []

    class SinkAdapter(_DummyAdapter):
        category_scopes = ("dress",)
        query_dimension = 3
        score_dimension = 2
        cache_policy = "nonreusable"

        def prepare_identity(self):
            return {"native_model_mode": "train", "dropout_seed": 0,
                    "feature_cache_policy": self.cache_policy, "gallery_before_queries": True}

        def query_scope(self, query, candidate_names):
            return "dress"

        def encode_queries(self, image_paths, texts, categories):
            query_calls.append(([str(path) for path in image_paths], list(texts), list(categories)))
            return np.tile(np.asarray([[1.0, 0.0, 9.0]], dtype=np.float32), (len(image_paths), 1))

        def encode_targets(self, image_paths, categories):
            target_calls.append([str(path) for path in image_paths])
            return np.asarray([[1.0, 0.0] if Path(path).name == "img2.png" else [0.0, 1.0]
                               for path in image_paths], dtype=np.float32)

        def score(self, queries, targets, categories=None):
            assert queries.shape == (1, 3)
            assert targets.shape == (2, 2)
            result = queries[:, :2] @ targets.T
            assert result.shape == (1, 2)
            assert np.argmax(result[0]) == 1
            return result

    spec = ModelSpec(
        model_id="sinkmodel", checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: SinkAdapter(cfg, ckpt),
        registry_model={"model_id": "sinkmodel", "fashioniq_layout": "fashioniq_standard",
                        "environment": {"python_env_var": None}},
        checkpoint_identity="sink-checkpoint",
    )
    captured: list[dict[str, Any]] = []

    def sink(**payload: Any) -> None:
        captured.append(payload)

    def run(callback=None, *, batch_size=32):
        kwargs = {"artifact_sink": callback, "parity_run_for": lambda model_id, category, condition: role} \
            if callback is not None else {}
        return run_composition(
            data, [spec], config, subsets=["full"], conditions=["multimodal"], batch_size=batch_size,
            image_resolver=lambda identity, candidates: _resolver(identity, dict(candidates), dataset),
            **kwargs)

    report = run(sink)
    assert report["models"][0]["status"] == "COMPLETED", report["models"][0]
    assert len(query_calls) == 1 and query_calls[0][1:] == (["query"], ["dress"])
    assert target_calls and [Path(path).name for path in target_calls[0]] == ["img1.png", "img2.png"]
    assert len(captured) == 1
    item = captured[0]
    assert item["run_id"] == "test-run" and item["plan_sha256"] == role["plan_sha256"]
    assert item["query_ids"] == query_ids and list(item["gallery_ids"]) == gallery_ids
    assert isinstance(item["query_features"], np.memmap)
    assert isinstance(item["gallery_features"], np.memmap)
    assert item["ranks"] == {"q1": 1}
    assert item["metrics"]["query_count"] == 1 and item["metrics"]["scored_count"] == 1
    assert item["query_features"].shape == (1, 3)
    assert item["gallery_features"].shape == (2, 2)
    assert item["execution"]["started_at"] == item["started_at"]
    assert item["execution"]["captured_at"] >= item["started_at"]
    same_execution = run(sink)
    assert same_execution["models"][0]["run_id"] == report["models"][0]["run_id"]
    assert same_execution["models"][0]["reused"] is False
    assert len(query_calls) == 2 and len(target_calls) == 2
    first_run_id = report["models"][0]["run_id"]
    without_parity = run()
    assert without_parity["models"][0]["run_id"] != first_run_id
    role["execution"]["partition_id"] = "full:test:dress:changed-execution"
    changed_execution = run(sink)
    assert changed_execution["models"][0]["run_id"] != first_run_id

    def fail_sink(**payload: Any) -> None:
        raise RuntimeError("sink write failure")

    try:
        run(fail_sink)
    except RuntimeError as error:
        assert "parity artifact sink failed" in str(error)
    else:
        raise AssertionError("parity sink failure must abort publication")
    role["execution"]["query_batch_size"] = 64
    role["execution"]["gallery_batch_size"] = 64
    rejected = run(sink)
    assert rejected["models"][0]["status"] == "FAILED"
    assert "parity execution batch sizes must match runner batch_size" in rejected["models"][0]["reason"]
def test_unchanged_provenance_preserves_identity_and_resumes(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda i, c: _resolver(i, c, dataset)

    spec = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={
            "model_id": "testmodel",
            "upstream_commit_sha": "774f94e2076ff17ea91703a6239d2a08f0e1a44e",
            "approved_source_patches": [],
            "fashioniq_layout": "fashioniq_standard",
            "environment": {"python_env_var": None},
        },
        checkpoint_identity="dummy_ckpt_hash",
    )

    rep1 = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    run_id_1 = rep1["models"][0]["run_id"]

    rep2 = run_composition(data, [spec], config, subsets=["full"], conditions=["multimodal"], resume=True, image_resolver=resolve)
    run_id_2 = rep2["models"][0]["run_id"]

    assert run_id_1 == run_id_2, "identical provenance must produce identical composition identity"
    assert rep2["models"][0]["reused"] is True, "resume must reuse cached ranks when provenance unchanged"


def test_absent_provenance_degrades_gracefully_to_unknown(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda i, c: _resolver(i, c, dataset)

    spec_empty = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={},
        checkpoint_identity="dummy_ckpt_hash",
    )

    rep = run_composition(data, [spec_empty], config, subsets=["full"], conditions=["multimodal"], image_resolver=resolve)
    assert rep["models"][0]["status"] == "COMPLETED"
    run_id = rep["models"][0]["run_id"]
    assert isinstance(run_id, str) and len(run_id) == 24
