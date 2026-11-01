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
