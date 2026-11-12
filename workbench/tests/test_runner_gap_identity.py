"""The condition list is part of the run identity: adding a condition mints a different run id.

Also proves the end-to-end shape of the new plumbing on a CPU fixture: a run that names
`clvc_image_encoder_only` encodes it through the adapter's dedicated method, publishes its ranks and
resolves a gap over the triple that actually carries ranks.

CPU only. The adapter is a NumPy fixture; no GPU, no checkpoint, no model interpreter.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pytest
from PIL import Image

from workbench.composition.adapters import CompositionAdapter
from workbench.composition.data import Gallery, Query, ReleaseData
from workbench.composition.runner import ModelSpec, run_composition

IMAGE_ONLY = "clvc_image_encoder_only"
DIMENSION = 4

class FixtureAdapter(CompositionAdapter):
    """Image-only row differs from the caption row, so a routing mistake is visible in the ranks."""

    model_id = "fixtureclvc"
    category_scopes = (None,)

    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                       (len(image_paths), 1))

    def encode_image_only_queries(self, image_paths: Sequence[str | Path],
                                  categories: Sequence[str | None]) -> np.ndarray:
        return np.tile(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                       (len(image_paths), 1))

    def encode_targets(self, image_paths: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        rows = [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)]
        return np.stack(rows[:len(image_paths)])

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return True, None

def _fixture(tmp_path: Path) -> tuple[ReleaseData, SimpleNamespace]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    third_party = tmp_path / "third_party"
    third_party.mkdir()
    names = ["g0.png", "g1.png", "g2.png"]
    for index, name in enumerate(names):
        Image.new("RGB", (8, 8), (index * 40, 10, 10)).save(dataset / name)
    data = ReleaseData(
        gallery=Gallery(protocol_id="circus_fashioniq_release_v1", image_ids=("g0", "g1", "g2")),
        queries=(Query(qid="q1", index=0, query_txt="make it blue", reference_id="g0",
                       positive_ids=("g1",), category=None, subsets=("full",)),),
        candidate_names={"g0": names[0], "g1": names[1], "g2": names[2]},
        provenance={"is_synthetic": True},
    )
    config = SimpleNamespace(WORKBENCH_ARTIFACTS_ROOT=str(artifacts),
                             FASHIONIQ_ROOT=str(dataset),
                             WORKBENCH_THIRD_PARTY_ROOT=str(third_party))
    return data, config

def _spec() -> ModelSpec:
    return ModelSpec(
        model_id="fixtureclvc", checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: FixtureAdapter(cfg, ckpt),
        registry_model={"model_id": "fixtureclvc", "fashioniq_layout": "fashioniq_standard",
                        "environment": {"python_env_var": None}},
        checkpoint_identity="dummy_ckpt_hash",
    )

def test_condition_list_is_part_of_the_run_identity(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda identity, names: dataset / names[identity] if identity in names else None
    two = ["multimodal", "text_only_black_image"]
    three = list(two) + ["image_only_empty_text"]
    four = list(two) + [IMAGE_ONLY]

    ids = {}
    for label, conditions in (("two", two), ("three", three), ("ablation", four)):
        report = run_composition(data, [_spec()], config, subsets=["full"], conditions=conditions,
                                 image_resolver=resolve)
        row = report["models"][0]
        assert row["status"] in {"COMPLETED", "PARTIAL"}, row.get("reason")
        ids[label] = row["run_id"]

    assert ids["two"] != ids["three"], "a longer condition list must mint a different run id"
    assert ids["two"] != ids["ablation"]
    assert ids["three"] != ids["ablation"], \
        "swapping the image condition must mint a different run id, not reuse the protocol one"

def test_ablation_condition_encodes_routes_and_publishes_a_gap(tmp_path: Path) -> None:
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda identity, names: dataset / names[identity] if identity in names else None
    conditions = ["multimodal", "text_only_black_image", IMAGE_ONLY]
    report = run_composition(data, [_spec()], config, subsets=["full"], conditions=conditions,
                             image_resolver=resolve)
    row = report["models"][0]
    assert row["status"] in {"COMPLETED", "PARTIAL"}, row.get("reason")

    import json
    summary = json.loads(Path(row["summaries"]["full"]).read_text(encoding="utf-8"))
    assert IMAGE_ONLY in summary["conditions"]
    # `image_only_empty_text` was never requested, so it is not inherited from the protocol tuple.
    assert "image_only_empty_text" not in summary["conditions"]
    assert summary["condition_status"][IMAGE_ONLY]["status"] == "COMPLETED"

    metrics = summary["metrics"]
    assert metrics["gap_conditions"] == [IMAGE_ONLY if index == 2 else conditions[index]
                                         for index in range(3)]
    assert metrics["gap_role_reason"] and "image role filled by" in metrics["gap_role_reason"]
    assert metrics["composition_gap_ndcg"] is not None
    # image-only row scores the g1 candidate highest for a reference image equal to it; the exact
    # value is not asserted here (the fixture's ranks are), only that the number exists and the
    # image role is the ablation's.
    assert metrics["image_only"]["condition"] == IMAGE_ONLY

def test_an_adapter_without_the_method_refuses_the_condition_by_name_only(tmp_path: Path) -> None:
    """Requesting the ablation on an adapter that lacks it must not cost the protocol conditions.

    The refusal is per condition (`ABLATION_UNSUPPORTED`, carrying `adapter_encoder_missing`), it
    happens BEFORE any encode, and the other two conditions still complete.
    """
    data, config = _fixture(tmp_path)
    dataset = Path(config.FASHIONIQ_ROOT)
    resolve = lambda identity, names: dataset / names[identity] if identity in names else None

    class NoImageOnly(FixtureAdapter):
        model_id = "fixturenoimageonly"
        # The historical adapter shape: the attribute simply is not there.
        encode_image_only_queries = None

    spec = ModelSpec(
        model_id="fixturenoimageonly", checkpoint_id="ckpt1",
        adapter_factory=lambda cfg, ckpt: NoImageOnly(cfg, ckpt),
        registry_model={"model_id": "fixturenoimageonly", "fashioniq_layout": "fashioniq_standard",
                        "environment": {"python_env_var": None}},
        checkpoint_identity="dummy_ckpt_hash",
    )
    report = run_composition(data, [spec], config, subsets=["full"],
                             conditions=["multimodal", "text_only_black_image", IMAGE_ONLY],
                             image_resolver=resolve)
    row = report["models"][0]
    import json
    summary = json.loads(Path(row["summaries"]["full"]).read_text(encoding="utf-8"))
    status = summary["condition_status"]
    assert status[IMAGE_ONLY]["status"] == "ABLATION_UNSUPPORTED"
    assert "adapter_encoder_missing" in status[IMAGE_ONLY]["reason"]
    assert IMAGE_ONLY in status[IMAGE_ONLY]["reason"]
    for condition in ("multimodal", "text_only_black_image"):
        assert status[condition]["status"] == "COMPLETED", condition
    # The refused condition carried no ranks, so it did NOT fill the image role and the triple stays
    # the protocol one; the gap is null with the existing `missing condition(s)` reason, and the role
    # resolution says which role is empty rather than silently substituting a name.
    assert summary["metrics"]["composition_gap_ndcg"] is None
    assert summary["metrics"]["gap_conditions"] == ["multimodal", "text_only_black_image",
                                                    "image_only_empty_text"]
    assert "role 'image' has no scored condition" in summary["metrics"]["gap_role_reason"]
    assert summary["metrics"]["gap_unavailable_reason"] == \
        "missing condition(s): image_only_empty_text"
