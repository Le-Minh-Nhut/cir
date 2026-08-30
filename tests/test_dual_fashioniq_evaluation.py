from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import pytest
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

import evaluation.fashioniq as fashioniq_evaluation
import training.engine as training_engine
from data.images import ImageBatch
from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery
from evaluation.fashioniq import (
    evaluate_fashioniq_protocols,
    evaluate_fashioniq_recall,
    select_gallery_features,
)
from training.engine import PrecisionPolicy, fit, trainable_parameters


def _annotations_for_sixty_images() -> list[FashionIQAnnotation]:
    ordering = [7, 42, 2, 51, 13, 37, *[index for index in range(60) if index not in {7, 42, 2, 51, 13, 37}]]
    return [
        FashionIQAnnotation(
            reference_id=f"image-{ordering[2 * index]}",
            target_id=f"image-{ordering[2 * index + 1]}",
            captions=("first", "second"),
            category="dress",
            index=index,
        )
        for index in range(30)
    ]


def test_ordered_original_gallery_subset_matches_direct_val_evaluation() -> None:
    torch.manual_seed(501)
    original_ids = [f"image-{index}" for index in range(60)]
    annotations = _annotations_for_sixty_images()
    val_ids = build_pair_union_gallery(annotations)
    feature_by_id = {
        image_id: torch.randn(24) for image_id in original_ids
    }
    original_features = torch.stack([feature_by_id[image_id] for image_id in original_ids])

    selected = select_gallery_features(original_features, original_ids, val_ids)
    directly_encoded = torch.stack([feature_by_id[image_id] for image_id in val_ids])

    assert val_ids[:6] == ["image-7", "image-42", "image-2", "image-51", "image-13", "image-37"]
    torch.testing.assert_close(selected, directly_encoded, atol=0.0, rtol=0.0)
    target_ids = [str(annotation.target_id) for annotation in annotations]
    reference_ids = [annotation.reference_id for annotation in annotations]
    queries = torch.stack([feature_by_id[target_id] for target_id in target_ids])
    scores_from_subset = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(
        selected, dim=-1
    ).T
    scores_from_direct = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(
        directly_encoded, dim=-1
    ).T
    assert [val_ids.index(target_id) for target_id in target_ids] == [
        2 * index + 1 for index in range(30)
    ]
    subset_metrics = evaluate_fashioniq_recall(
        scores_from_subset, target_ids, val_ids, reference_ids
    )
    direct_metrics = evaluate_fashioniq_recall(
        scores_from_direct, target_ids, val_ids, reference_ids
    )
    assert subset_metrics == direct_metrics == {"recall_at_10": 100.0, "recall_at_50": 100.0}
    with pytest.raises(ValueError, match="not a subset"):
        select_gallery_features(original_features, original_ids, [*val_ids, "missing-image"])


def test_dual_evaluation_encodes_queries_and_original_gallery_once(
    tmp_path, monkeypatch
) -> None:
    original_ids = [f"image-{index}" for index in range(60)]
    annotations = _annotations_for_sixty_images()
    val_ids = build_pair_union_gallery(annotations)
    (tmp_path / "split.dress.val.json").write_text(
        json.dumps(original_ids), encoding="utf-8"
    )
    feature_by_id = {
        image_id: torch.eye(60)[index] for index, image_id in enumerate(original_ids)
    }
    target_ids = [str(annotation.target_id) for annotation in annotations]
    queries = torch.stack([feature_by_id[target_id] for target_id in target_ids])

    class QueryModel:
        def __init__(self) -> None:
            self.training = True
            self.query_calls = 0

        def eval(self):
            self.training = False
            return self

        def __call__(self, *args: Tensor) -> SimpleNamespace:
            del args
            assert not torch.is_grad_enabled()
            self.query_calls += 1
            return SimpleNamespace(final_query=queries)

    model = QueryModel()
    gallery_calls: list[list[str]] = []

    def fake_encode_gallery(model_arg, image_ids, *args, **kwargs):
        del model_arg, args, kwargs
        gallery_calls.append(list(image_ids))
        return torch.stack([feature_by_id[image_id] for image_id in image_ids])

    monkeypatch.setattr(fashioniq_evaluation, "encode_gallery", fake_encode_gallery)
    batch = ImageBatch(
        sample_ids=[f"sample-{index}" for index in range(30)],
        reference_ids=[annotation.reference_id for annotation in annotations],
        target_ids=target_ids,
        modification_texts=["first and second"] * 30,
        categories=["dress"] * 30,
        reference_pixels=torch.zeros(30, 1),
        target_pixels=None,
        input_ids=torch.zeros(30, 2, dtype=torch.long),
        attention_mask=torch.ones(30, 2, dtype=torch.bool),
        content_mask=torch.ones(30, 2, dtype=torch.bool),
    )

    results = evaluate_fashioniq_protocols(
        model,
        {"dress": [batch]},
        {"dress": annotations},
        protocols=("fashioniq_original", "fashioniq_val"),
        split_root=tmp_path,
        split="val",
        image_store=None,
        image_processor=None,
        device=torch.device("cpu"),
        gallery_batch_size=128,
        num_workers=0,
    )

    assert model.query_calls == 1
    assert gallery_calls == [original_ids]
    assert not model.training
    assert set(results) == {"fashioniq_original", "fashioniq_val"}
    assert results["fashioniq_val"]["mean_recall"] == 100.0
    assert val_ids != original_ids


class _TinyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.backbone = SimpleNamespace(
            backbone_type="openclip",
            checkpoint="ViT-B-16",
            revision="laion2b_s34b_b88k",
            library="open_clip_torch",
            library_version="3.3.0",
            weights_repository="repository",
            weights_revision="immutable-sha",
        )


def _metrics(mean_recall: float) -> dict[str, float]:
    return {
        "recall_at_10": mean_recall,
        "recall_at_50": mean_recall,
        "mean_recall": mean_recall,
    }


def test_dual_checkpoint_selection_is_independent_and_evaluation_is_read_only(
    tmp_path, monkeypatch
) -> None:
    model = _TinyTrainingModel()
    objective = nn.Linear(1, 1, bias=False)
    optimizer = SGD(trainable_parameters(model, objective), lr=0.1)
    optimizer_ids_before = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    trainability_before = [
        parameter.requires_grad for parameter in trainable_parameters(model, objective)
    ]
    precision = PrecisionPolicy("fp32", False, None, False)
    initial_model_weight = model.weight.detach().clone()
    successful_steps = 0
    training_modes: list[bool] = []
    observed_losses: list[float] = []

    def fake_train_one_epoch(
        model_arg, objective_arg, loader, optimizer_arg, scaler, device, *, precision, epoch
    ):
        del loader, scaler, device
        nonlocal successful_steps
        model_arg.train()
        objective_arg.train()
        training_modes.append(model_arg.training)
        optimizer_arg.zero_grad(set_to_none=True)
        loss = model_arg.weight.sum() + objective_arg.weight.sum()
        loss.backward()
        optimizer_arg.step()
        successful_steps += 1
        value = float(epoch + 1)
        observed_losses.append(value)
        assert precision.name == "fp32"
        return {"total": value}

    monkeypatch.setattr(training_engine, "train_one_epoch", fake_train_one_epoch)
    sequence = iter(((20.0, 30.0), (25.0, 29.0), (24.0, 35.0)))

    def evaluate(model_arg):
        original, val = next(sequence)
        before = model_arg.weight.detach().clone()
        with torch.no_grad():
            model_arg.eval()
            assert not torch.is_grad_enabled()
        torch.testing.assert_close(model_arg.weight, before)
        return {
            "fashioniq_original": _metrics(original),
            "fashioniq_val": _metrics(val),
        }

    fit(
        model,
        objective,
        DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
        optimizer,
        evaluate,
        epochs=3,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        precision=precision,
    )

    original_checkpoint = torch.load(tmp_path / "best_original.pt", weights_only=True)
    val_checkpoint = torch.load(tmp_path / "best_val.pt", weights_only=True)
    last_checkpoint = torch.load(tmp_path / "last.pt", weights_only=True)
    assert original_checkpoint["epoch"] == 2
    assert val_checkpoint["epoch"] == 3
    assert last_checkpoint["epoch"] == 3
    assert not (tmp_path / "best.pt").exists()
    assert original_checkpoint["metadata"]["selection_protocol"] == "fashioniq_original"
    assert original_checkpoint["metadata"]["selection_metric_value"] == 25.0
    assert val_checkpoint["metadata"]["selection_protocol"] == "fashioniq_val"
    assert val_checkpoint["metadata"]["selection_metric_value"] == 35.0
    assert last_checkpoint["metadata"]["selection_protocol"] is None
    assert last_checkpoint["metric"] is None
    assert last_checkpoint["metadata"]["validation_metrics"] == {
        "fashioniq_original": _metrics(24.0),
        "fashioniq_val": _metrics(35.0),
    }

    optimizer_ids_after = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids_after == optimizer_ids_before
    assert successful_steps == 3
    assert observed_losses == [1.0, 2.0, 3.0]
    assert training_modes == [True, True, True]
    assert [parameter.requires_grad for parameter in trainable_parameters(model, objective)] == (
        trainability_before
    )
    torch.testing.assert_close(model.weight, initial_model_weight - 0.3)
    assert precision == PrecisionPolicy("fp32", False, None, False)
