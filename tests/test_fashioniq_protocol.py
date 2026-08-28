from __future__ import annotations

import random
import json
from pathlib import Path

import pytest
import torch
import yaml

from datasets.fashioniq import (
    FashionIQAnnotation,
    compose_fashioniq_caption,
    normalize_fashioniq_caption,
    resolve_fashioniq_correction_dicts,
    validate_correction_policy,
)
from evaluation.fashioniq import (
    apply_fashioniq_protocol_mask,
    build_fashioniq_gallery,
    compare_fashioniq_rankings,
    evaluate_fashioniq_ranking,
    evaluate_fashioniq_recall,
    fashioniq_target_ranks,
    macro_average_fashioniq,
    mask_fashioniq_reference_scores,
)


def test_validated_caption_order_and_correction_semantics() -> None:
    captions = ("Is BRIGT, red.", "Has sleeves?")
    corrections = {"brigt": "bright"}
    assert compose_fashioniq_caption(captions, "ordered_and") == "Is brigt, red and Has sleeves"
    assert normalize_fashioniq_caption(captions[0], corrections) == "is bright red"
    assert (
        compose_fashioniq_caption(captions, "normalized_ordered_and", corrections)
        == "is bright red and has sleeves"
    )
    # Determinism is controlled by the dataset's seeded local RNG; branch semantics remain exact.
    assert compose_fashioniq_caption(captions, "randomized_four_way", rng=random.Random(1)) == "Is brigt, red and Has sleeves"
    assert validate_correction_policy("fashioniq") == "fashioniq"
    assert validate_correction_policy("none") == "none"


def test_fashioniq_hand_ranking_and_macro_metrics() -> None:
    gallery_ids = tuple(f"image-{index}" for index in range(60))
    scores = torch.zeros(2, 60)
    scores[0, 7] = 10
    scores[1, 55] = 10
    scores[1, :49] = torch.arange(49, dtype=torch.float32) + 20
    metrics = evaluate_fashioniq_recall(scores, ("image-7", "image-55"), gallery_ids)
    assert metrics == {
        "recall_at_1": 50.0,
        "recall_at_10": 50.0,
        "recall_at_50": 100.0,
    }
    macro = macro_average_fashioniq(
        {
            "a": metrics,
            "b": {
                "recall_at_1": 100.0,
                "recall_at_10": 100.0,
                "recall_at_50": 100.0,
            },
        }
    )
    assert macro == {
        "recall_at_1": 75.0,
        "recall_at_10": 75.0,
        "recall_at_50": 100.0,
        "mean_recall": 87.5,
    }


def test_val_reference_exclusion_promotes_target_without_masking_it() -> None:
    gallery = ("A", "B", "C", "D")
    scores = torch.tensor([[10.0, 9.0, 2.0, 1.0]])
    masked = mask_fashioniq_reference_scores(
        scores, ("A",), gallery, target_ids=("B",)
    )
    assert scores[0, 0] == 10.0
    assert torch.isneginf(masked[0, 0])
    assert masked[0, 1] == scores[0, 1]
    assert fashioniq_target_ranks(scores, ("B",), gallery).item() == 2
    assert fashioniq_target_ranks(
        scores,
        ("B",),
        gallery,
        protocol="fashioniq_val",
        reference_ids=("A",),
    ).item() == 1


def test_val_reference_exclusion_is_per_query_and_fails_loudly() -> None:
    gallery = ("A", "B", "C", "D", "E")
    scores = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0],
            [6.0, 7.0, 10.0, 9.0, 8.0],
        ]
    )
    masked = mask_fashioniq_reference_scores(
        scores,
        ("A", "C"),
        gallery,
        target_ids=("B", "D"),
    )
    assert torch.isneginf(masked[0, 0]) and masked[0, 2] == scores[0, 2]
    assert torch.isneginf(masked[1, 2]) and masked[1, 0] == scores[1, 0]
    assert masked[0, 1] == scores[0, 1]
    assert masked[1, 3] == scores[1, 3]
    with pytest.raises(ValueError, match="references absent from gallery"):
        mask_fashioniq_reference_scores(scores[:1], ("missing",), gallery)
    with pytest.raises(ValueError, match="reference_id == target_id"):
        mask_fashioniq_reference_scores(
            scores[:1], ("A",), gallery, target_ids=("A",)
        )


def test_protocol_mask_is_val_only_and_applies_to_all_variant_scores() -> None:
    gallery = tuple(f"image-{index}" for index in range(60))
    references = ("image-0", "image-2")
    targets = ("image-1", "image-3")
    base = torch.full((2, 60), -10.0)
    base[0, 0], base[0, 1] = 10.0, 9.0
    base[1, 2], base[1, 3] = 10.0, 9.0
    for offset in (0.0, 0.5, 1.0, 1.5):  # dynamic/frozen/repeat/clone
        metrics = evaluate_fashioniq_ranking(
            base + offset,
            targets,
            gallery,
            protocol="fashioniq_val",
            reference_ids=references,
        )
        assert metrics["recall_at_1"] == 100.0
        assert metrics["mrr"] == 1.0
    original = apply_fashioniq_protocol_mask(
        "fashioniq_original", base, targets, gallery, references
    )
    val = apply_fashioniq_protocol_mask(
        "fashioniq_val", base, targets, gallery, references
    )
    torch.testing.assert_close(original, base)
    assert torch.isneginf(val[0, 0]) and torch.isneginf(val[1, 2])
    with pytest.raises(ValueError, match="requires reference_ids"):
        apply_fashioniq_protocol_mask(
            "fashioniq_val", base, targets, gallery, reference_ids=None
        )


def test_val_gallery_is_ordered_pair_union_and_distinct_from_original(tmp_path) -> None:
    annotations = [
        FashionIQAnnotation("A", "B", ("x", "y"), "dress", 0),
        FashionIQAnnotation("C", "B", ("x", "y"), "dress", 1),
        FashionIQAnnotation("D", "E", ("x", "y"), "dress", 2),
    ]
    (tmp_path / "split.dress.val.json").write_text(
        json.dumps(["official-0", "A", "official-1", "E"]), encoding="utf-8"
    )
    val_gallery = build_fashioniq_gallery(
        "fashioniq_val", tmp_path, "dress", annotations, "val"
    )
    original_gallery = build_fashioniq_gallery(
        "fashioniq_original", tmp_path, "dress", annotations, "val"
    )
    assert val_gallery == ["A", "B", "C", "D", "E"]
    assert len(val_gallery) == len(set(val_gallery))
    assert {annotation.target_id for annotation in annotations}.issubset(val_gallery)
    assert original_gallery == ["official-0", "A", "official-1", "E"]
    assert val_gallery != original_gallery
    with pytest.raises(ValueError, match="Unsupported FashionIQ protocol"):
        build_fashioniq_gallery("fashioniq_unknown", tmp_path, "dress", annotations, "val")


def test_val_protocol_config_is_explicit_and_primary_taper_configs_use_it() -> None:
    root = Path(__file__).parents[1]
    protocol = yaml.safe_load(
        (root / "conf" / "protocol" / "fashioniq_val.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol == {
        "name": "fashioniq_val",
        "dataset_name": "fashioniq",
        "split": "val",
    }
    for name in (
        "taper_mag_v4_base.yaml",
        "taper_mag_v4_frozen_text.yaml",
        "taper_mag_v4_full_text.yaml",
        "taper_mag_v4_smoke.yaml",
    ):
        config = yaml.safe_load((root / "conf" / name).read_text(encoding="utf-8"))
        assert config["data"]["validation_protocol"] == "fashioniq_val"


def test_dynamic_frozen_comparison_uses_gallery_rank_not_target_score_alone() -> None:
    gallery = tuple(f"image-{index}" for index in range(60))
    target = ("image-0",)
    gallery_embeddings = torch.eye(60)
    dynamic_query = torch.full((1, 60), -((1.0 - 0.15**2) / 59) ** 0.5)
    dynamic_query[0, 0] = 0.15
    frozen_query = torch.full((1, 60), -((1.0 - 0.2**2 - 10 * 0.21**2) / 49) ** 0.5)
    frozen_query[0, 0] = 0.2
    frozen_query[0, 1:11] = 0.21
    torch.testing.assert_close(dynamic_query.norm(dim=-1), torch.ones(1))
    torch.testing.assert_close(frozen_query.norm(dim=-1), torch.ones(1))
    dynamic_scores = dynamic_query @ gallery_embeddings.T
    frozen_scores = frozen_query @ gallery_embeddings.T
    comparison = compare_fashioniq_rankings(
        dynamic_scores,
        frozen_scores,
        target,
        gallery,
        protocol="fashioniq_val",
        reference_ids=("image-59",),
    )
    assert dynamic_scores[0, 0] < frozen_scores[0, 0]  # target cosine points the wrong way
    assert comparison["dynamic"]["recall_at_10"] == 100.0
    assert comparison["frozen"]["recall_at_10"] == 0.0
    assert comparison["delta"]["mean_recall"] > 0
    assert comparison["target_rank_improved_fraction"] == 1.0
    assert comparison["same_gallery"] is True


def test_correction_resolver_fails_clearly_without_silent_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        resolve_fashioniq_correction_dicts(tmp_path, "fashioniq")
    message = str(error.value)
    assert "correction_dict_dress.json" in message
    assert "correction_dict_shirt.json" in message
    assert "correction_dict_toptee.json" in message
    assert str(tmp_path.resolve()) in message
    assert "correction_policy=none" in message
    assert resolve_fashioniq_correction_dicts(tmp_path, "none") is None


def test_correction_resolver_loads_all_audited_files(tmp_path) -> None:
    for category in ("dress", "shirt", "toptee"):
        (tmp_path / f"correction_dict_{category}.json").write_text(
            json.dumps({"brigt": "bright"}), encoding="utf-8"
        )
    resolved = resolve_fashioniq_correction_dicts(tmp_path, "fashioniq")
    assert resolved == {category: {"brigt": "bright"} for category in ("dress", "shirt", "toptee")}
