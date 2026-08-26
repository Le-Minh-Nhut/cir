from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from backbones.fgclip2 import FGCLIP2_LARGE_MODEL_ID, FGCLIP2_LARGE_REVISION
from cache.features import validate_feature_manifest, validate_text_cache_subdir
from datasets.fashioniq import (
    CORRECTION_POLICIES,
    FashionIQDataset,
    compose_fashioniq_caption,
    normalize_fashioniq_caption,
    validate_correction_policy,
)
from precompute_fgclip2_text import build_text_manifest


class FashionIQCorrectionPolicyTest(unittest.TestCase):
    def test_normalization_with_correction(self) -> None:
        self.assertEqual(
            normalize_fashioniq_caption(
                "Redd shirt!!!",
                correction_dict={"redd": "red"},
            ),
            "red shirt",
        )

    def test_normalization_without_correction(self) -> None:
        self.assertEqual(
            normalize_fashioniq_caption("Redd shirt!!!", correction_dict=None),
            "redd shirt",
        )

    def test_composition_differs_only_by_token_substitution(self) -> None:
        captions = ("Redd shirt.", "Long sleeves!")
        corrected = compose_fashioniq_caption(
            captions,
            policy="normalized_ordered_and",
            correction_dict={"redd": "red"},
        )
        uncorrected = compose_fashioniq_caption(
            captions,
            policy="normalized_ordered_and",
            correction_dict=None,
        )
        self.assertEqual(corrected, "red shirt and long sleeves")
        self.assertEqual(uncorrected, "redd shirt and long sleeves")

    def test_dataset_accepts_normalized_caption_without_dictionary(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            annotation_root = Path(directory)
            (annotation_root / "cap.dress.val.json").write_text(
                json.dumps(
                    [
                        {
                            "candidate": "reference",
                            "target": "target",
                            "captions": ["Redd shirt.", "Long sleeves!"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            dataset = FashionIQDataset(
                annotation_root=annotation_root,
                split="val",
                categories=["dress"],
                caption_policy="normalized_ordered_and",
                correction_dicts=None,
            )
            self.assertEqual(
                dataset[0].modification_text,
                "redd shirt and long sleeves",
            )

    def test_invalid_correction_policy_fails_loudly(self) -> None:
        self.assertEqual(CORRECTION_POLICIES, {"fashioniq", "none"})
        with self.assertRaises(ValueError):
            validate_correction_policy("raw")

    def test_experiment_defaults_preserve_corrected_baseline(self) -> None:
        config = yaml.safe_load(
            Path("conf/experiment/taper_e2e.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["train_caption_policy"], "normalized_ordered_and")
        self.assertEqual(config["val_caption_policy"], "normalized_ordered_and")
        self.assertEqual(config["correction_policy"], "fashioniq")
        self.assertEqual(config["text_cache_subdir"], "text")


class FashionIQCorrectionManifestTest(unittest.TestCase):
    def _manifest(self, correction_policy: str) -> dict:
        return build_text_manifest(
            split="val",
            backbone=SimpleNamespace(
                model_id=FGCLIP2_LARGE_MODEL_ID,
                revision=FGCLIP2_LARGE_REVISION,
                max_text_length=64,
            ),
            num_samples=1,
            states_shape=(1, 64, 1024),
            attention_shape=(1, 64),
            content_shape=(1, 64),
            states_dtype="float32",
            mask_dtype="bool",
            token_audit={"num_samples": 1},
            correction_policy=correction_policy,
            parity_samples=1,
            parity_max_abs_error=0.0,
        )

    def test_manifests_record_explicit_correction_policy(self) -> None:
        corrected = self._manifest("fashioniq")
        uncorrected = self._manifest("none")
        self.assertEqual(corrected["caption_policy"], "normalized_ordered_and")
        self.assertEqual(uncorrected["caption_policy"], "normalized_ordered_and")
        self.assertEqual(corrected["correction_policy"], "fashioniq")
        self.assertEqual(uncorrected["correction_policy"], "none")
        self.assertIn("correction_dictionary_files", corrected)
        self.assertNotIn("correction_dictionary_files", uncorrected)

    def test_correction_policy_mismatch_is_rejected(self) -> None:
        corrected = self._manifest("fashioniq")
        with self.assertRaises(ValueError):
            validate_feature_manifest(
                corrected,
                model_id=FGCLIP2_LARGE_MODEL_ID,
                revision=FGCLIP2_LARGE_REVISION,
                cache_name="val/text_no_correction",
                correction_policy="none",
            )

    def test_legacy_baseline_manifest_is_explicitly_warned(self) -> None:
        legacy = self._manifest("fashioniq")
        del legacy["correction_policy"]
        with self.assertWarnsRegex(UserWarning, "Legacy corrected"):
            validate_feature_manifest(
                legacy,
                model_id=FGCLIP2_LARGE_MODEL_ID,
                revision=FGCLIP2_LARGE_REVISION,
                cache_name="val/text",
                correction_policy="fashioniq",
            )
        with self.assertRaises(ValueError):
            validate_feature_manifest(
                legacy,
                model_id=FGCLIP2_LARGE_MODEL_ID,
                revision=FGCLIP2_LARGE_REVISION,
                cache_name="val/text_no_correction",
                correction_policy="none",
            )

    def test_no_correction_cannot_target_baseline_or_image_cache(self) -> None:
        self.assertEqual(
            validate_text_cache_subdir("text_no_correction", "none"),
            "text_no_correction",
        )
        with self.assertRaises(ValueError):
            validate_text_cache_subdir("text", "none")
        with self.assertRaises(ValueError):
            validate_text_cache_subdir("images", "none")
        with self.assertRaises(ValueError):
            validate_text_cache_subdir("../text", "none")
        with self.assertRaises(ValueError):
            validate_text_cache_subdir("text_other", "raw")


if __name__ == "__main__":
    unittest.main()
