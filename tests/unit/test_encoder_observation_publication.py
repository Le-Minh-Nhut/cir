import json
import subprocess
import sys

import torch

from workbench.replay.encoder_export import observation_payload
from workbench.replay.encoder_observer import observe_category
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.backend.index import load_runs
from workbench.backend.schemas.results import ResultRun


def test_encoder_observer_preserves_native_gallery_and_query_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    (tmp_path / "image_splits").mkdir()
    (tmp_path / "captions").mkdir()
    (tmp_path / "image_splits/split.dress.val.json").write_text(json.dumps(["ref", "target"]))
    (tmp_path / "captions/cap.dress.val.json").write_text(json.dumps([
        {"candidate": "ref", "target": "target", "captions": ["one", "two"]},
    ]))
    (tmp_path / "captions/correction_dict_dress.json").write_text("{}")

    query = {"source_img_id": 0, "target_img_id": 1, "source_img_data": torch.tensor([1., 0.]),
             "mod": {"str": "one and two"}}
    targets = [
        {"target_img_id": 0, "target_img_data": torch.tensor([1., 0.])},
        {"target_img_id": 1, "target_img_data": torch.tensor([0., 1.])},
    ]

    class Model:
        def extract_retrieval_compose(self, images, texts):
            return torch.tensor([[0., 1.] for _ in texts])

        def extract_retrieval_target(self, images):
            return images

    observed = observe_category("dress", Model(), type("Dataset", (), {
        "test_queries_dress": [query], "test_targets_dress": targets,
    })(), tmp_path, batch_size=1, top_k=2)
    assert observed.queries[0].reference_id == "ref"
    assert observed.queries[0].target_id == "target"
    assert observed.queries[0].target_rank == 1
    assert [item[1] for item in observed.queries[0].top_results] == ["target"]
    payload = observation_payload([observed], {}, run_id="dev-encoder-observer-fixture",
                                  top_k=1, checkpoint={
                                      "checkpoint_id": "fashioniq", "filename": "fashioniq.pt",
                                      "expected_sha256": "a" * 64,
                                  })
    payload_path = tmp_path / "observer.payload.json"
    payload_path.write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, "-m", "workbench.replay.encoder_publish", "--payload",
         str(payload_path), "--results-root", str(tmp_path / "observer-results")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    artifact = ResultRun.model_validate_json(open(json.loads(result.stdout)["published"]).read())
    assert artifact.queries[0].reference_id == "ref"
    assert artifact.queries[0].target_id == "target"
    assert artifact.queries[0].target_rank == 1


def test_encoder_payload_publishes_canonical_result_in_subprocess(tmp_path):
    category = ObservedCategory(
        category="dress", gallery_size=2, eligible_count=1,
        queries=(ObservedQuery(
            annotation_index=0, reference_id="ref", target_id="target",
            raw_captions=("first", "second"), model_input_text="first and second",
            target_rank=1, top_results=((1, "target", 0.9),), category="dress",
        ),),
    )
    checkpoint = {
        "checkpoint_id": "fashioniq", "filename": "fashioniq.pt",
        "expected_sha256": "a" * 64,
    }
    payload = observation_payload([category], {}, run_id="dev-encoder-fixture",
                                  top_k=1, checkpoint=checkpoint)
    payload["command_digest"] = "b" * 64
    payload["environment_digest"] = "c" * 64
    payload_path = tmp_path / "observation.payload.json"
    payload_path.write_text(json.dumps(payload))

    result = subprocess.run(
        [sys.executable, "-m", "workbench.replay.encoder_publish", "--payload",
         str(payload_path), "--results-root", str(tmp_path / "results")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    published = json.loads(result.stdout)["published"]
    artifact = ResultRun.model_validate_json(open(published).read())
    assert artifact.run.model_id == "encoder"
    assert artifact.run.method_name == "ENCODER"
    assert artifact.run.checkpoint_id == "fashioniq"
    indexed = load_runs(tmp_path / "results")
    assert len(indexed) == 1
    assert indexed[0].run.model_id == "encoder"
    assert artifact.run.data_kind == "mock"
    assert artifact.queries[0].target_id == "target"
    assert artifact.queries[0].target_rank == 1
