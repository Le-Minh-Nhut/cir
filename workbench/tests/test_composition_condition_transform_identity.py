"""Composition cache identity must bind condition-transform source."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from PIL import Image

from workbench.composition.adapters import CompositionAdapter
from workbench.composition.data import Gallery, Query, ReleaseData
from workbench.composition.runner import ModelSpec, run_composition
import workbench.composition.inputs as inputs


class _DummyAdapter(CompositionAdapter):
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
    Image.new("RGB", (10, 10), "red").save(dataset / "img1.png")
    data = ReleaseData(
        gallery=Gallery(protocol_id="circus_fashioniq_release_v1", image_ids=("img1",)),
        queries=(Query(qid="q1", index=0, query_txt="query", reference_id="img1",
                       positive_ids=("img1",), category=None, subsets=("full",)),),
        candidate_names={"img1": "img1.png"},
        provenance={"is_synthetic": True},
    )
    config = SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=str(artifacts),
                             FASHIONIQ_ROOT=str(dataset),
                             WORKBENCH_THIRD_PARTY_ROOT=str(third_party))
    return data, config


def test_condition_transform_source_changes_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    source = Path(inputs.__file__)
    original = source.read_bytes()
    spec = ModelSpec(
        model_id="testmodel",
        checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: _DummyAdapter(cfg, ckpt),
        registry_model={"model_id": "testmodel", "fashioniq_layout": "fashioniq_standard",
                         "environment": {"python_env_var": None}},
        checkpoint_identity="dummy_ckpt_hash",
    )

    def run() -> str:
        report = run_composition(data, [spec], config, subsets=["full"],
                                 conditions=["text_only_black_image"],
                                 image_resolver=lambda identity, names: dataset / names[identity])
        return report["models"][0]["run_id"]

    try:
        identity_a = run()
        assert run() == identity_a, "unchanged condition-transform source must preserve identity"
        source.write_bytes(original + b"\n# condition transform identity mutation\n")
        identity_b = run()
        assert identity_b != identity_a, "mutated condition-transform source must invalidate identity"
    finally:
        source.write_bytes(original)
