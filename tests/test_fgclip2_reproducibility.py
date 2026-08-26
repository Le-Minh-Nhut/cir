from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from backbones.fgclip2 import (
    FGCLIP2Backbone,
    FGCLIP2_DYNAMIC_PATCH_BUDGETS,
    FGCLIP2_LARGE_MODEL_ID,
    FGCLIP2_LARGE_REVISION,
    FGCLIP2_PATCH_POLICY_NAME,
    determine_max_num_patches,
)
from cache.features import validate_feature_manifest
from precompute_fgclip2_images import build_image_manifest
from precompute_fgclip2_text import build_text_manifest


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeImageProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int]]] = []

    def __call__(self, *, images, max_num_patches, return_tensors):
        image_ids = [image.getpixel((0, 0))[0] for image in images]
        self.calls.append((max_num_patches, image_ids))
        return _FakeBatch(pixel_values=torch.tensor(image_ids, dtype=torch.long))


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=1024),
            vision_config=SimpleNamespace(hidden_size=1024),
        )

    def get_image_features(self, *, pixel_values):
        features = torch.zeros(pixel_values.shape[0], 1024)
        features[torch.arange(pixel_values.shape[0]), pixel_values] = 1.0
        return features


def _image_with_patch_count(patch_count: int, image_id: int = 0) -> Image.Image:
    return Image.new("RGB", (max(patch_count * 16, 1), 16), color=(image_id, 0, 0))


class FGCLIP2PatchPolicyTest(unittest.TestCase):
    def test_official_strict_threshold_boundaries(self) -> None:
        cases = {
            0: 128,
            128: 128,
            129: 256,
            256: 256,
            257: 576,
            576: 576,
            577: 784,
            784: 784,
            785: 1024,
        }
        for patch_count, expected_budget in cases.items():
            with self.subTest(patch_count=patch_count):
                image = _image_with_patch_count(patch_count)
                self.assertEqual(determine_max_num_patches(image), expected_budget)

        # Width flooring must occur before multiplication, exactly as upstream.
        floored = Image.new("RGB", (129 * 16 - 1, 16))
        self.assertEqual(determine_max_num_patches(floored), 128)

    def test_grouping_restores_original_order(self) -> None:
        fake_model = _FakeModel()
        fake_processor = _FakeImageProcessor()
        with (
            patch("backbones.fgclip2.AutoModelForCausalLM.from_pretrained", return_value=fake_model),
            patch("backbones.fgclip2.AutoTokenizer.from_pretrained", return_value=object()),
            patch(
                "backbones.fgclip2.AutoImageProcessor.from_pretrained",
                return_value=fake_processor,
            ),
        ):
            backbone = FGCLIP2Backbone()

        images = [
            _image_with_patch_count(128, 11),
            _image_with_patch_count(257, 22),
            _image_with_patch_count(64, 33),
            _image_with_patch_count(785, 44),
        ]
        features = backbone.encode_image_global(images)

        self.assertEqual(features.argmax(dim=1).tolist(), [11, 22, 33, 44])
        self.assertEqual(
            fake_processor.calls,
            [(128, [11, 33]), (576, [22]), (1024, [44])],
        )
        self.assertEqual(features.shape, (4, 1024))
        self.assertTrue(torch.allclose(features.norm(dim=1), torch.ones(4)))
        self.assertFalse(features.requires_grad)


class FGCLIP2RevisionAndManifestTest(unittest.TestCase):
    def test_revision_propagates_to_all_hf_loaders(self) -> None:
        fake_model = _FakeModel()
        with (
            patch(
                "backbones.fgclip2.AutoModelForCausalLM.from_pretrained",
                return_value=fake_model,
            ) as model_loader,
            patch(
                "backbones.fgclip2.AutoTokenizer.from_pretrained",
                return_value=object(),
            ) as tokenizer_loader,
            patch(
                "backbones.fgclip2.AutoImageProcessor.from_pretrained",
                return_value=_FakeImageProcessor(),
            ) as processor_loader,
        ):
            backbone = FGCLIP2Backbone()

        expected = {
            "revision": FGCLIP2_LARGE_REVISION,
            "trust_remote_code": True,
        }
        model_loader.assert_called_once_with(FGCLIP2_LARGE_MODEL_ID, **expected)
        tokenizer_loader.assert_called_once_with(FGCLIP2_LARGE_MODEL_ID, **expected)
        processor_loader.assert_called_once_with(FGCLIP2_LARGE_MODEL_ID, **expected)
        self.assertEqual(backbone.revision, FGCLIP2_LARGE_REVISION)
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_image_and_text_manifest_contracts(self) -> None:
        backbone = SimpleNamespace(
            model_id=FGCLIP2_LARGE_MODEL_ID,
            revision=FGCLIP2_LARGE_REVISION,
            max_text_length=64,
        )
        image_manifest = build_image_manifest(
            split="val",
            backbone=backbone,
            num_samples=4,
            images_shape=(4, 1, 1024),
            images_dtype="float32",
            patch_budget_counts=Counter({128: 2, 576: 1, 1024: 1}),
            parity_samples=3,
            parity_max_abs_error=0.0,
        )
        text_manifest = build_text_manifest(
            split="val",
            backbone=backbone,
            num_samples=4,
            states_shape=(4, 64, 1024),
            attention_shape=(4, 64),
            content_shape=(4, 64),
            states_dtype="float32",
            mask_dtype="bool",
            token_audit={"num_samples": 4},
            parity_samples=3,
            parity_max_abs_error=0.0,
        )

        for manifest in (image_manifest, text_manifest):
            self.assertEqual(manifest["model_id"], FGCLIP2_LARGE_MODEL_ID)
            self.assertEqual(manifest["revision"], FGCLIP2_LARGE_REVISION)
        self.assertEqual(
            image_manifest["preprocessing"]["patch_policy"],
            FGCLIP2_PATCH_POLICY_NAME,
        )
        self.assertEqual(
            image_manifest["preprocessing"]["possible_patch_budgets"],
            list(FGCLIP2_DYNAMIC_PATCH_BUDGETS),
        )
        self.assertEqual(
            image_manifest["patch_budget_counts"],
            {"128": 2, "256": 0, "576": 1, "784": 0, "1024": 1},
        )
        with self.assertRaises(ValueError):
            validate_feature_manifest(
                {**text_manifest, "revision": "0" * 40},
                model_id=image_manifest["model_id"],
                revision=image_manifest["revision"],
                cache_name="val/text",
            )


if __name__ == "__main__":
    unittest.main()
