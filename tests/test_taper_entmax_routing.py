from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from entmax import entmax15

from models.taper import R1_ROUTING_SUPPORT_EPS, TAPER


def tiny_taper(*, num_slots: int = 2, dim: int = 4) -> TAPER:
    return TAPER(
        text_dim=dim,
        slot_dim=dim,
        reference_dim=dim,
        query_dim=dim,
        state_dim=dim,
        num_slots=num_slots,
        num_primitives=2,
        qasa_tau=0.5,
        qasa_rho=0.8,
        qasa_mu=0.3,
    )


class TAPEREntmaxRoutingTest(unittest.TestCase):
    def test_clear_gap_produces_exact_sparse_token_support(self) -> None:
        model = tiny_taper()
        logits = torch.tensor(
            [[[4.0, 0.0, -4.0, 99.0], [-4.0, 0.0, 4.0, 99.0]]],
            requires_grad=True,
        )
        valid = torch.tensor([[True, True, True, False]])
        selected = torch.tensor([[True, True]])

        routing = model._token_entmax_routing(logits, valid, selected)

        self.assertEqual(routing.shape, (1, 2, 4))
        self.assertTrue(torch.isfinite(routing).all())
        self.assertTrue((routing >= 0).all())
        self.assertEqual(torch.count_nonzero(routing[:, :, ~valid[0]]).item(), 0)
        self.assertTrue(
            torch.allclose(
                routing[0, :, :3],
                entmax15(logits[0, :, :3], dim=-1),
            )
        )
        self.assertTrue(torch.allclose(routing.sum(dim=-1), torch.ones(1, 2)))
        self.assertFalse(
            torch.allclose(
                routing.sum(dim=1)[valid],
                torch.ones_like(routing.sum(dim=1)[valid]),
            )
        )
        self.assertGreater(
            int((routing[:, :, :3] <= R1_ROUTING_SUPPORT_EPS).sum().item()),
            0,
        )

    def test_unselected_and_zero_content_rows_are_zero(self) -> None:
        model = tiny_taper()
        logits = torch.randn(2, 2, 5, requires_grad=True)
        valid = torch.tensor(
            [[True, True, False, False, False], [False, False, False, False, False]]
        )
        selected = torch.tensor([[True, False], [True, True]])

        routing = model._token_entmax_routing(logits, valid, selected)

        self.assertTrue(torch.allclose(routing[0, 0].sum(), torch.tensor(1.0)))
        self.assertEqual(torch.count_nonzero(routing[0, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(routing[1]).item(), 0)
        self.assertTrue(torch.isfinite(routing).all())

    def test_edit_slots_are_pooled_from_entmax_routing(self) -> None:
        model = tiny_taper(dim=2)
        text = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        logits = torch.tensor([[[4.0, 0.0, -4.0], [-4.0, 0.0, 4.0]]])
        soft_ownership = F.softmax(logits, dim=1)

        def competitive(_text: torch.Tensor, _valid: torch.Tensor):
            return logits, soft_ownership

        def select_all(attention: torch.Tensor, _valid: torch.Tensor):
            return {
                "qasa_quality": torch.ones(1, 2),
                "qasa_selected_mask": torch.ones(1, 2, dtype=torch.bool),
                "qasa_selected_count": torch.tensor([2]),
                "qasa_final_coverage": torch.ones(1),
                "qasa_novelty_skip_count": torch.zeros(1),
            }

        with (
            patch.object(model, "_competitive_ownership", side_effect=competitive),
            patch.object(model, "_qasa_select_slots", side_effect=select_all),
        ):
            output = model.build_edit_slots(text, valid, text_content_mask=valid)

        expected = torch.tensor([[[1.0, 0.0], [10.0, 10.0]]])
        dense_edit_slots = output["slot_semantics"] * output["slot_activity"].unsqueeze(-1)
        self.assertTrue(torch.allclose(output["edit_slots"], expected, atol=1e-6))
        self.assertFalse(torch.allclose(output["edit_slots"], dense_edit_slots))
        self.assertTrue(
            torch.allclose(
                output["edit_slots"],
                output["routing_slot_semantics"]
                * output["routing_slot_activity"].unsqueeze(-1),
            )
        )

    def test_qasa_outputs_match_independent_pre_sparse_computation(self) -> None:
        torch.manual_seed(11)
        model = tiny_taper()
        text = torch.randn(2, 5, 4)
        attention = torch.ones(2, 5, dtype=torch.bool)
        content = torch.tensor(
            [[False, True, True, True, False], [False, True, True, False, False]]
        )
        _, valid = model._validate_text_inputs(text, attention, content)
        expected_attention = model._qasa_attention_fp32(text, valid)
        expected_qasa = model._qasa_select_slots(expected_attention, valid)

        output = model.build_edit_slots(text, attention, text_content_mask=content)

        self.assertTrue(torch.equal(output["qasa_attention"], expected_attention))
        self.assertTrue(torch.equal(output["qasa_quality"], expected_qasa["qasa_quality"]))
        self.assertTrue(
            torch.equal(
                output["qasa_selected_mask"],
                expected_qasa["qasa_selected_mask"],
            )
        )

    def test_routing_diagnostics_use_binary_support_jaccard(self) -> None:
        model = tiny_taper()
        soft_masks = torch.full((1, 2, 4), 0.5)
        routing_masks = torch.tensor(
            [[[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]]
        )
        diagnostics = model._assignment_diagnostics(
            slot_masks=soft_masks,
            slot_mass=soft_masks.sum(dim=-1),
            routing_masks=routing_masks,
            routing_slot_mass=routing_masks.sum(dim=-1),
            routing_support_count=torch.tensor([[2, 2]]),
            qasa_selected_mask=torch.tensor([[True, True]]),
            qasa_quality=torch.ones(1, 2),
            qasa_final_coverage=torch.ones(1),
            hard_active_slot_mask=torch.tensor([[True, True]]),
            text_attention_mask=torch.ones(1, 4, dtype=torch.bool),
            text_content_mask=torch.ones(1, 4, dtype=torch.bool),
        )

        self.assertAlmostEqual(float(diagnostics["routing_support_mean"]), 2.0)
        self.assertAlmostEqual(float(diagnostics["routing_support_max"]), 2.0)
        self.assertAlmostEqual(
            float(diagnostics["routing_support_fraction_mean"]),
            0.5,
        )
        self.assertAlmostEqual(float(diagnostics["routing_zero_fraction"]), 0.5)
        self.assertAlmostEqual(float(diagnostics["routing_active_slot_count"]), 2.0)
        self.assertAlmostEqual(
            float(diagnostics["routing_support_overlap_mean"]),
            1.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(diagnostics["routing_slot_0_support_mean"]),
            2.0,
        )
        self.assertAlmostEqual(
            float(diagnostics["routing_slot_1_support_mean"]),
            2.0,
        )

    def test_sparse_routing_path_has_finite_parameter_gradients(self) -> None:
        torch.manual_seed(19)
        model = tiny_taper()
        text = torch.randn(2, 6, 4)
        attention = torch.ones(2, 6, dtype=torch.bool)
        content = torch.tensor(
            [[False, True, True, True, True, False], [False, True, True, True, False, False]]
        )

        output = model.build_edit_slots(text, attention, text_content_mask=content)
        self.assertTrue(output["qasa_selected_mask"].any())
        output["edit_slots"].square().sum().backward()

        parameters = (
            model.slot_queries,
            model.slot_query_projection.weight,
            model.text_key_projection.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_autocast_routing_is_finite_and_qasa_remains_fp32(self) -> None:
        torch.manual_seed(23)
        model = tiny_taper()
        reference = torch.randn(2, 4)
        text = torch.randn(2, 6, 4)
        attention = torch.ones(2, 6, dtype=torch.bool)
        content = torch.tensor(
            [[False, True, True, True, True, False], [False, True, True, True, False, False]]
        )

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(
                reference,
                text,
                attention,
                text_content_mask=content,
            )

        self.assertEqual(output["qasa_attention"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["routing_masks"]).all())
        self.assertTrue(torch.isfinite(output["edit_slots"]).all())
        output["q0"][:, 0].sum().backward()
        for parameter in (
            model.slot_queries,
            model.slot_query_projection.weight,
            model.text_key_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
