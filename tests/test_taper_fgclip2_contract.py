from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache.features import get_text_features_by_sample_ids, load_text_features
from models.taper import TAPER


class TAPERFGCLIP2ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TAPER(
            text_dim=1024,
            slot_dim=1024,
            reference_dim=1024,
            query_dim=1024,
            state_dim=512,
            num_slots=4,
            num_primitives=8,
            qasa_tau=0.5,
            qasa_rho=0.8,
            qasa_mu=0.3,
        )
        self.reference = torch.randn(2, 1024)
        self.text = torch.randn(2, 8, 1024)
        self.attention = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]],
            dtype=torch.bool,
        )
        self.content = torch.tensor(
            [[0, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0]],
            dtype=torch.bool,
        )

    def test_forward_ownership_qasa_and_backward(self) -> None:
        output = self.model(
            self.reference,
            self.text,
            self.attention,
            text_content_mask=self.content,
        )
        valid = self.attention & self.content

        self.assertEqual(output["edit_slots"].shape, (2, 4, 1024))
        self.assertEqual(output["slot_masks"].shape, (2, 4, 8))
        self.assertEqual(output["slot_semantics"].shape, (2, 4, 1024))
        self.assertEqual(output["routing_masks"].shape, (2, 4, 8))
        self.assertEqual(output["routing_slot_semantics"].shape, (2, 4, 1024))
        self.assertEqual(output["routing_support_count"].shape, (2, 4))
        self.assertEqual(output["reference_state"].shape, (2, 512))
        self.assertEqual(output["final_state"].shape, (2, 512))
        self.assertEqual(output["q0"].shape, (2, 1024))
        self.assertEqual(output["qasa_attention"].dtype, torch.float32)
        self.assertTrue(
            torch.allclose(
                output["qasa_attention"],
                output["slot_masks"].float(),
                atol=1e-6,
            )
        )
        self.assertEqual(self.model.router[0].in_features, 2560)

        invalid_ownership = output["slot_masks"] * (~valid[:, None, :])
        self.assertEqual(torch.count_nonzero(invalid_ownership).item(), 0)
        valid_ownership = output["slot_masks"].sum(dim=1)[valid]
        self.assertTrue(
            torch.allclose(valid_ownership, torch.ones_like(valid_ownership), atol=1e-6)
        )
        self.assertTrue(torch.isfinite(output["slot_masks"]).all())
        self.assertTrue(torch.isfinite(output["slot_mass"]).all())
        self.assertTrue(torch.isfinite(output["routing_masks"]).all())
        self.assertTrue((output["routing_masks"] >= 0).all())
        invalid_routing = output["routing_masks"] * (~valid[:, None, :])
        self.assertEqual(torch.count_nonzero(invalid_routing).item(), 0)
        unselected_routing = output["routing_masks"] * (
            ~output["qasa_selected_mask"][:, :, None]
        )
        self.assertEqual(torch.count_nonzero(unselected_routing).item(), 0)
        active = output["qasa_selected_mask"] & valid.any(dim=1, keepdim=True)
        routing_mass = output["routing_masks"].sum(dim=-1)
        self.assertTrue(
            torch.allclose(
                routing_mass[active],
                torch.ones_like(routing_mass[active]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output["edit_slots"],
                output["routing_slot_semantics"]
                * output["routing_slot_activity"].unsqueeze(-1),
            )
        )

        hard_sum = output["qasa_inference_hard_regions"].sum(dim=1)
        self.assertTrue(torch.equal(hard_sum[valid], torch.ones_like(hard_sum[valid])))
        self.assertEqual(torch.count_nonzero(hard_sum[~valid]).item(), 0)

        losses = self.model.compute_loss(
            {
                "reference_features": self.reference,
                "target_features": torch.randn(2, 1, 1024),
                "text_states": self.text,
                "text_attention_mask": self.attention,
                "text_content_mask": self.content,
                "target_ids": ["target-a", "target-b"],
            }
        )
        loss = losses["retrieval_loss"]
        self.assertTrue(torch.isfinite(loss))
        for name in (
            "routing_support_mean",
            "routing_support_max",
            "routing_support_fraction_mean",
            "routing_zero_fraction",
            "routing_active_slot_count",
            "routing_support_overlap_mean",
        ):
            self.assertIn(f"diagnostic/{name}", losses)
        for slot_id in range(self.model.num_slots):
            self.assertIn(f"diagnostic/routing_slot_{slot_id}_support_mean", losses)
        loss.backward()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.grad is not None for parameter in trainable))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in trainable))

    def test_empty_content_has_zero_slots(self) -> None:
        empty = self.model.build_edit_slots(
            self.text,
            self.attention,
            text_content_mask=torch.zeros_like(self.content),
        )
        for key in (
            "slot_masks",
            "slot_mass",
            "slot_activity",
            "routing_masks",
            "routing_slot_mass",
            "routing_slot_activity",
            "routing_slot_semantics",
            "routing_support_count",
            "edit_slots",
        ):
            self.assertEqual(torch.count_nonzero(empty[key]).item(), 0, key)
        self.assertEqual(empty["qasa_attention"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(empty["qasa_attention"]).all())
        self.assertEqual(
            torch.count_nonzero(empty["qasa_inference_hard_regions"]).item(),
            0,
        )


class TextFeatureCacheContractTest(unittest.TestCase):
    def test_cache_has_only_fgclip2_text_arrays_and_preserves_caption_parity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            np.save(root / "states.npy", np.zeros((2, 4, 1024), dtype=np.float32))
            np.save(root / "attention_mask.npy", np.ones((2, 4), dtype=np.bool_))
            np.save(root / "content_mask.npy", np.ones((2, 4), dtype=np.bool_))
            (root / "sample_to_idx.json").write_text(
                json.dumps({"a": 0, "b": 1}), encoding="utf-8"
            )
            (root / "captions.json").write_text(
                json.dumps({"a": "red", "b": "blue"}), encoding="utf-8"
            )
            (root / "manifest.json").write_text(
                json.dumps({"model_id": "qihoo360/fg-clip2-large"}), encoding="utf-8"
            )

            cache = load_text_features(root)
            result = get_text_features_by_sample_ids(["b"], ["blue"], cache)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0].shape, (1, 4, 1024))
            with self.assertRaises(RuntimeError):
                get_text_features_by_sample_ids(["b"], ["not blue"], cache)


if __name__ == "__main__":
    unittest.main()
