from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from models.taper import TAPER


def tiny_taper(
    *,
    capacity_enabled: bool = False,
    capacity: float = 1.0,
    theta: float = 0.25,
    r4_lambda: float = 1.0,
    solver_iters: int = 64,
    routing_mode: str = "qisca",
    candidate_mode: str = "qasa_selected",
) -> TAPER:
    torch.manual_seed(41)
    return TAPER(
        text_dim=4,
        slot_dim=4,
        reference_dim=4,
        query_dim=4,
        state_dim=4,
        num_slots=4,
        num_primitives=2,
        routing_mode=routing_mode,
        r4_theta=theta,
        r4_lambda=r4_lambda,
        r4_capacity_enabled=capacity_enabled,
        r4_candidate_mode=candidate_mode,
        r4_slot_capacity=capacity,
        r4_solver_iters=solver_iters,
    )


class SubSimplexProjectionTest(unittest.TestCase):
    def test_already_feasible_clamps_only_negative_values(self) -> None:
        values = torch.tensor([[0.2, -0.1, 0.3]])
        projected = TAPER._project_nonnegative_l1_ball(values, 1.0)
        torch.testing.assert_close(projected, torch.tensor([[0.2, 0.0, 0.3]]))

    def test_excess_mass_projects_to_equality_simplex(self) -> None:
        values = torch.tensor([[0.8, 0.7, 0.1]])
        projected = TAPER._project_nonnegative_l1_ball(values, 1.0)
        self.assertTrue((projected >= 0).all())
        torch.testing.assert_close(projected.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(projected, torch.tensor([[0.55, 0.45, 0.0]]))

    def test_all_negative_and_zero_radius_are_zero(self) -> None:
        negative = torch.tensor([[-2.0, -1.0, -3.0]])
        self.assertEqual(
            torch.count_nonzero(
                TAPER._project_nonnegative_l1_ball(negative, 1.0)
            ).item(),
            0,
        )
        values = torch.tensor([[2.0, 1.0, -3.0]])
        self.assertEqual(
            torch.count_nonzero(
                TAPER._project_nonnegative_l1_ball(values, 0.0)
            ).item(),
            0,
        )


class MassAwarePoolingTest(unittest.TestCase):
    def test_assignment_mass_is_applied_exactly_once(self) -> None:
        states = torch.tensor([[[2.0, -4.0]]])
        cases = (
            (0.0, [0.0, 0.0], 0.0, [0.0, 0.0]),
            (0.2, [2.0, -4.0], 0.2, [0.4, -0.8]),
            (1.0, [2.0, -4.0], 1.0, [2.0, -4.0]),
            (2.0, [2.0, -4.0], 1.0, [2.0, -4.0]),
        )
        for mass, expected_semantics, expected_activity, expected_edit in cases:
            with self.subTest(mass=mass):
                masks = torch.tensor([[[mass]]])
                semantics, slot_mass, activity = TAPER._mass_aware_slot_pool(
                    states,
                    masks,
                )
                edit_slots = semantics * activity.unsqueeze(-1)
                torch.testing.assert_close(slot_mass, torch.tensor([[mass]]))
                torch.testing.assert_close(
                    semantics,
                    torch.tensor([[expected_semantics]]),
                )
                torch.testing.assert_close(
                    activity,
                    torch.tensor([[expected_activity]]),
                )
                torch.testing.assert_close(
                    edit_slots,
                    torch.tensor([[expected_edit]]),
                )

    def test_r1_unit_mass_pooling_is_unchanged(self) -> None:
        states = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
        masks = torch.tensor([[[0.25, 0.75]]])
        semantics, mass, activity = TAPER._mass_aware_slot_pool(states, masks)
        torch.testing.assert_close(mass, torch.ones(1, 1))
        torch.testing.assert_close(activity, torch.ones(1, 1))
        torch.testing.assert_close(semantics, torch.tensor([[[4.0, 6.0]]]))


class TAPERQISCARoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = torch.tensor([[True, True, True, True]])
        self.selected = torch.tensor([[True, True, True, True]])

    def test_r4_configuration_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "routing_mode"):
            tiny_taper(routing_mode="invalid")
        with self.assertRaisesRegex(ValueError, "r4_theta"):
            tiny_taper(theta=1.0)
        with self.assertRaisesRegex(ValueError, "r4_lambda"):
            TAPER(
                text_dim=2,
                slot_dim=2,
                reference_dim=2,
                query_dim=2,
                r4_lambda=0.0,
            )
        with self.assertRaisesRegex(ValueError, "r4_slot_capacity"):
            tiny_taper(capacity_enabled=True, capacity=0.0)
        with self.assertRaisesRegex(ValueError, "r4_solver_iters"):
            tiny_taper(solver_iters=0)
        with self.assertRaisesRegex(ValueError, "r4_candidate_mode"):
            tiny_taper(candidate_mode="invalid")

    def test_r4a_token_budget_and_exact_row_projection(self) -> None:
        model = tiny_taper(theta=0.10, capacity_enabled=False)
        competition = torch.tensor(
            [
                [
                    [0.55, 0.20, 0.40, 0.25],
                    [0.25, 0.50, 0.20, 0.25],
                    [0.10, 0.20, 0.30, 0.25],
                    [0.10, 0.10, 0.10, 0.25],
                ]
            ]
        )
        routing = model._qisca_routing(competition, self.valid, self.selected)
        utility = (competition - model.r4_theta) / model.r4_lambda
        expected = TAPER._project_nonnegative_l1_ball(
            utility.transpose(1, 2),
            1.0,
        ).transpose(1, 2)

        torch.testing.assert_close(routing, expected)
        self.assertTrue((routing >= 0).all())
        self.assertTrue((routing.sum(dim=1) <= 1.0 + 1e-6).all())

    def _token_budget_diagnostics(
        self,
        model: TAPER,
        competition: torch.Tensor,
        preprojection: torch.Tensor,
        routing: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = torch.ones(1, competition.shape[-1], dtype=torch.bool)
        return model._assignment_diagnostics(
            slot_masks=competition,
            slot_mass=competition.sum(dim=-1),
            routing_masks=routing,
            routing_slot_mass=routing.sum(dim=-1),
            routing_support_count=(routing > 1e-6).sum(dim=-1),
            qasa_selected_mask=self.selected,
            qasa_quality=torch.ones(1, 4),
            qasa_final_coverage=torch.ones(1),
            hard_active_slot_mask=routing.sum(dim=-1) > 1e-6,
            text_attention_mask=valid,
            text_content_mask=valid,
            r4_preprojection=preprojection,
        )

    def test_default_lambda_can_leave_token_budget_inactive(self) -> None:
        model = tiny_taper(theta=0.25, r4_lambda=1.0)
        competition = torch.tensor([[[0.70], [0.10], [0.10], [0.10]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        preprojection = model._qisca_preprojection(
            competition,
            valid,
            self.selected,
        )
        routing = model._qisca_routing(competition, valid, self.selected)
        diagnostics = self._token_budget_diagnostics(
            model,
            competition,
            preprojection,
            routing,
        )

        self.assertLessEqual(float(preprojection.sum(dim=1)), 1.0)
        torch.testing.assert_close(routing, preprojection)
        self.assertEqual(
            float(
                diagnostics[
                    "r4_preprojection_token_budget_violation_fraction"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(diagnostics["r4_preprojection_token_budget_excess_mean"]),
            0.0,
        )
        self.assertEqual(
            float(diagnostics["r4_token_budget_binding_fraction"]),
            0.0,
        )

    def test_smaller_lambda_activates_token_budget_projection(self) -> None:
        model = tiny_taper(theta=0.25, r4_lambda=0.25)
        competition = torch.tensor([[[0.70], [0.10], [0.10], [0.10]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        preprojection = model._qisca_preprojection(
            competition,
            valid,
            self.selected,
        )
        routing = model._qisca_routing(competition, valid, self.selected)
        diagnostics = self._token_budget_diagnostics(
            model,
            competition,
            preprojection,
            routing,
        )

        self.assertGreater(float(preprojection.sum(dim=1)), 1.0)
        torch.testing.assert_close(routing.sum(dim=1), torch.ones(1, 1))
        self.assertEqual(
            float(
                diagnostics[
                    "r4_preprojection_token_budget_violation_fraction"
                ]
            ),
            1.0,
        )
        self.assertGreater(
            float(diagnostics["r4_preprojection_token_budget_excess_mean"]),
            0.0,
        )
        self.assertEqual(
            float(diagnostics["r4_token_budget_binding_fraction"]),
            1.0,
        )

    def test_implicit_rejection_and_clear_winner(self) -> None:
        rejected_model = tiny_taper(theta=0.30)
        uniform = torch.full((1, 4, 1), 0.25)
        valid = torch.ones(1, 1, dtype=torch.bool)
        rejected = rejected_model._qisca_routing(
            uniform,
            valid,
            self.selected,
        )
        self.assertEqual(torch.count_nonzero(rejected).item(), 0)

        winner_model = tiny_taper(theta=0.25)
        clear = torch.tensor([[[0.8], [0.1], [0.05], [0.05]]])
        routed = winner_model._qisca_routing(clear, valid, self.selected)
        self.assertGreater(float(routed[0, 0, 0]), 0.0)
        self.assertEqual(torch.count_nonzero(routed[0, 1:, 0]).item(), 0)

    def test_qasa_selected_zero_evidence_slot_never_executes(self) -> None:
        model = TAPER(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=2,
            routing_mode="qisca",
            r4_theta=0.30,
            r4_lambda=1.0,
            r4_capacity_enabled=False,
        )
        reference = torch.tensor([[0.2, -0.1]])
        text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        competition = torch.tensor([[[0.80, 0.80], [0.20, 0.20]]])
        logits = competition.log()

        def select_both(
            _attention: torch.Tensor,
            _valid: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "qasa_quality": torch.ones(1, 2),
                "qasa_selected_mask": torch.ones(1, 2, dtype=torch.bool),
                "qasa_selected_count": torch.tensor([2]),
                "qasa_final_coverage": torch.ones(1),
                "qasa_novelty_skip_count": torch.zeros(1),
            }

        with (
            patch.object(
                model,
                "_competitive_ownership",
                return_value=(logits, competition),
            ),
            patch.object(model, "_qasa_select_slots", side_effect=select_both),
        ):
            output = model(
                reference,
                text,
                valid,
                text_content_mask=valid,
            )

        self.assertTrue(bool(output["qasa_selected_mask"][0, 1]))
        self.assertGreater(float(output["routing_slot_mass"][0, 0]), 0.0)
        self.assertEqual(float(output["routing_slot_mass"][0, 1]), 0.0)
        self.assertFalse(bool(output["routing_active_mask"][0, 1]))
        self.assertFalse(bool(output["execution_selected_mask"][0, 1]))
        self.assertFalse(bool(output["hard_active_slot_mask"][0, 1]))
        self.assertEqual(int(output["slot_to_step"][0, 1]), -1)
        self.assertEqual(int(output["trace_valid_mask"].sum()), 1)

        changed_slots = output["edit_slots"].clone()
        changed_slots[:, 1] = 10_000.0
        changed_execution = model.execute(
            changed_slots,
            output["execution_selected_mask"],
            output["z0"],
            output["reference_state"],
        )
        torch.testing.assert_close(
            changed_execution["final_state"],
            output["final_state"],
        )

    def test_qasa_mask_does_not_renormalize_selected_slot(self) -> None:
        model = tiny_taper(theta=0.30)
        competition = torch.tensor([[[0.28], [0.25], [0.24], [0.23]]])
        selected = torch.tensor([[True, False, False, False]])
        routing = model._qisca_routing(
            competition,
            torch.ones(1, 1, dtype=torch.bool),
            selected,
        )
        self.assertEqual(torch.count_nonzero(routing).item(), 0)

    def test_all_real_slots_capacity_redistributes_to_secondary_slot(self) -> None:
        competition = torch.tensor([[[0.70, 0.70], [0.30, 0.30]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        qasa_only_s0 = torch.tensor([[True, False]])
        common = dict(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=1,
            routing_mode="qisca",
            r4_theta=0.0,
            r4_lambda=0.5,
            r4_slot_capacity=0.8,
            r4_solver_iters=64,
        )
        no_capacity = TAPER(
            **common,
            r4_capacity_enabled=False,
            r4_candidate_mode="all_real_slots",
        )
        spillover = TAPER(
            **common,
            r4_capacity_enabled=True,
            r4_candidate_mode="all_real_slots",
        )
        hard_pruned = TAPER(
            **common,
            r4_capacity_enabled=True,
            r4_candidate_mode="qasa_selected",
        )

        unconstrained = no_capacity._qisca_routing(
            competition,
            valid,
            qasa_only_s0,
        )
        redistributed = spillover._qisca_routing(
            competition,
            valid,
            qasa_only_s0,
        )
        rejected = hard_pruned._qisca_routing(
            competition,
            valid,
            qasa_only_s0,
        )

        self.assertGreater(
            float(unconstrained[:, 0].sum()),
            float(unconstrained[:, 1].sum()),
        )
        torch.testing.assert_close(
            redistributed.sum(dim=-1),
            torch.tensor([[0.8, 0.8]]),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertGreater(
            float(redistributed[:, 1].sum()),
            float(unconstrained[:, 1].sum()),
        )
        self.assertEqual(torch.count_nonzero(rejected[:, 1]).item(), 0)
        torch.testing.assert_close(
            rejected[:, 0].sum(),
            torch.tensor(0.8),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertGreater(
            float(redistributed.sum()),
            float(rejected.sum()),
        )

    def test_non_qasa_spillover_slot_executes_and_is_traced(self) -> None:
        model = TAPER(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=2,
            routing_mode="qisca",
            r4_theta=0.0,
            r4_lambda=0.5,
            r4_capacity_enabled=True,
            r4_candidate_mode="all_real_slots",
            r4_slot_capacity=0.8,
        )
        reference = torch.tensor([[0.2, -0.1]])
        text = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        competition = torch.tensor([[[0.70, 0.70], [0.30, 0.30]]])
        logits = competition.log()

        def select_s0(
            _attention: torch.Tensor,
            _valid: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "qasa_quality": torch.ones(1, 2),
                "qasa_selected_mask": torch.tensor([[True, False]]),
                "qasa_selected_count": torch.tensor([1]),
                "qasa_final_coverage": torch.ones(1),
                "qasa_novelty_skip_count": torch.zeros(1),
            }

        with (
            patch.object(
                model,
                "_competitive_ownership",
                return_value=(logits, competition),
            ),
            patch.object(model, "_qasa_select_slots", side_effect=select_s0),
        ):
            output = model(
                reference,
                text,
                valid,
                text_content_mask=valid,
            )
            dropped = model.slot_drop_queries(
                reference_features=reference,
                text_states=text,
                text_attention_mask=valid,
                text_content_mask=valid,
            )

        self.assertFalse(bool(output["qasa_selected_mask"][0, 1]))
        self.assertGreater(float(output["routing_slot_mass"][0, 1]), 0.0)
        self.assertTrue(bool(output["routing_active_mask"][0, 1]))
        self.assertTrue(bool(output["execution_selected_mask"][0, 1]))
        self.assertTrue(bool(output["hard_active_slot_mask"][0, 1]))
        self.assertGreaterEqual(int(output["slot_to_step"][0, 1]), 0)
        self.assertEqual(int(output["trace_valid_mask"].sum()), 2)
        torch.testing.assert_close(
            dropped["execution_selected_mask"],
            output["execution_selected_mask"],
        )
        torch.testing.assert_close(dropped["slot_to_step"], output["slot_to_step"])
        torch.testing.assert_close(
            dropped["trace_valid_mask"],
            output["trace_valid_mask"],
        )

        diagnostics = model._assignment_diagnostics(
            slot_masks=output["slot_masks"],
            slot_mass=output["slot_mass"],
            routing_masks=output["routing_masks"],
            routing_slot_mass=output["routing_slot_mass"],
            routing_support_count=output["routing_support_count"],
            qasa_selected_mask=output["qasa_selected_mask"],
            qasa_quality=output["qasa_quality"],
            qasa_final_coverage=output["qasa_final_coverage"],
            hard_active_slot_mask=output["hard_active_slot_mask"],
            text_attention_mask=valid,
            text_content_mask=valid,
            r4_preprojection=output["r4_preprojection"],
            routing_candidate_mask=output["routing_candidate_mask"],
        )
        self.assertEqual(
            float(diagnostics["routing_non_qasa_active_slot_count"]),
            1.0,
        )
        self.assertGreater(float(diagnostics["routing_non_qasa_mass_mean"]), 0.0)
        self.assertGreater(
            float(diagnostics["routing_non_qasa_mass_fraction"]),
            0.0,
        )
        self.assertEqual(
            float(diagnostics["execution_non_qasa_active_slot_count"]),
            1.0,
        )

    def test_zero_mass_non_qasa_candidate_does_not_execute(self) -> None:
        model = TAPER(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=1,
            routing_mode="qisca",
            r4_theta=0.25,
            r4_lambda=1.0,
            r4_capacity_enabled=True,
            r4_candidate_mode="all_real_slots",
            r4_slot_capacity=1.0,
        )
        competition = torch.tensor([[[0.80], [0.20]]])
        selected = torch.tensor([[True, False]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        routing = model._qisca_routing(competition, valid, selected)
        active = routing.sum(dim=-1) > 1e-6
        execution = model._r4_candidate_mask(selected, valid) & active
        self.assertEqual(float(routing[0, 1, 0]), 0.0)
        self.assertFalse(bool(active[0, 1]))
        self.assertFalse(bool(execution[0, 1]))

    def test_invalid_inactive_and_zero_content_are_exactly_zero(self) -> None:
        model = tiny_taper(theta=0.10)
        competition = torch.full((2, 4, 3), 0.25)
        valid = torch.tensor([[True, False, True], [False, False, False]])
        selected = torch.tensor(
            [[True, False, True, False], [True, True, True, True]]
        )
        routing = model._qisca_routing(competition, valid, selected)

        self.assertTrue(torch.isfinite(routing).all())
        self.assertEqual(torch.count_nonzero(routing[0, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(routing[0, 3]).item(), 0)
        self.assertEqual(torch.count_nonzero(routing[:, :, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(routing[1]).item(), 0)

    def test_all_real_slots_admits_non_qasa_slots_but_not_invalid_tokens(self) -> None:
        model = tiny_taper(theta=0.10, candidate_mode="all_real_slots")
        competition = torch.full((1, 4, 3), 0.25)
        valid = torch.tensor([[True, False, True]])
        selected = torch.tensor([[True, False, False, False]])
        routing = model._qisca_routing(competition, valid, selected)

        self.assertTrue((routing[0, 1:, valid[0]] > 0).all())
        self.assertEqual(torch.count_nonzero(routing[:, :, ~valid[0]]).item(), 0)

    def test_capacity_toggle_controls_only_column_constraint(self) -> None:
        competition = torch.tensor(
            [[[[0.90, 0.90, 0.90, 0.90]], [[0.10, 0.10, 0.10, 0.10]]]]
        ).reshape(1, 2, 4)
        selected = torch.tensor([[True, True]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        off = TAPER(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=1,
            routing_mode="qisca",
            r4_theta=0.25,
            r4_capacity_enabled=False,
            r4_slot_capacity=1.0,
        )
        on = TAPER(
            text_dim=2,
            slot_dim=2,
            reference_dim=2,
            query_dim=2,
            state_dim=2,
            num_slots=2,
            num_primitives=1,
            routing_mode="qisca",
            r4_theta=0.25,
            r4_capacity_enabled=True,
            r4_slot_capacity=1.0,
            r4_solver_iters=64,
        )
        routing_off = off._qisca_routing(competition, valid, selected)
        routing_on = on._qisca_routing(competition, valid, selected)

        self.assertGreater(float(routing_off[:, 0].sum()), 1.0)
        self.assertLessEqual(float(routing_on[:, 0].sum()), 1.0 + 1e-5)
        self.assertFalse(torch.allclose(routing_off, routing_on))

    def test_capacity_disabled_ignores_capacity_value(self) -> None:
        competition = torch.rand(1, 4, 4)
        competition = competition / competition.sum(dim=1, keepdim=True)
        small = tiny_taper(capacity_enabled=False, capacity=0.1, theta=0.15)
        large = tiny_taper(capacity_enabled=False, capacity=1000.0, theta=0.15)
        routing_small = small._qisca_routing(
            competition,
            self.valid,
            self.selected,
        )
        routing_large = large._qisca_routing(
            competition,
            self.valid,
            self.selected,
        )
        torch.testing.assert_close(routing_small, routing_large)

    def test_r4b_feasibility_and_dykstra_convergence(self) -> None:
        torch.manual_seed(47)
        competition = torch.rand(2, 4, 7)
        competition = competition / competition.sum(dim=1, keepdim=True)
        valid = torch.tensor(
            [
                [True, True, True, True, True, False, False],
                [True, True, True, True, True, True, False],
            ]
        )
        selected = torch.tensor(
            [[True, True, False, True], [True, False, True, True]]
        )
        production = tiny_taper(
            capacity_enabled=True,
            capacity=0.8,
            theta=0.12,
            solver_iters=64,
        )
        reference = tiny_taper(
            capacity_enabled=True,
            capacity=0.8,
            theta=0.12,
            solver_iters=512,
        )
        routed = production._qisca_routing(competition, valid, selected)
        routed_reference = reference._qisca_routing(
            competition,
            valid,
            selected,
        )

        self.assertTrue((routed >= 0).all())
        self.assertTrue((routed.sum(dim=1) <= 1.0 + 1e-5).all())
        self.assertTrue((routed.sum(dim=-1) <= 0.8 + 1e-4).all())
        self.assertEqual(
            torch.count_nonzero(routed * (~valid[:, None, :])).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(routed * (~selected[:, :, None])).item(),
            0,
        )
        torch.testing.assert_close(routed, routed_reference, atol=2e-5, rtol=2e-5)

        utility = competition - production.r4_theta
        objective = (utility * routed).sum() - 0.5 * routed.square().sum()
        reference_objective = (
            (utility * routed_reference).sum()
            - 0.5 * routed_reference.square().sum()
        )
        torch.testing.assert_close(
            objective,
            reference_objective,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_projection_objective_identity(self) -> None:
        model = tiny_taper(theta=0.25)
        competition = torch.tensor([[[0.8], [0.1], [0.05], [0.05]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        routing = model._qisca_routing(competition, valid, self.selected)
        utility = competition - model.r4_theta
        scaled = utility / model.r4_lambda
        maximization = (utility * routing).sum() - (
            model.r4_lambda / 2.0
        ) * routing.square().sum()
        projection_form = -(
            model.r4_lambda / 2.0
        ) * (routing - scaled).square().sum() + (
            utility.square().sum() / (2.0 * model.r4_lambda)
        )

        torch.testing.assert_close(maximization, projection_form)
        self.assertGreater(float(maximization), 0.0)

    def test_qisca_gradient_reaches_ownership_parameters(self) -> None:
        model = tiny_taper(theta=0.10, candidate_mode="all_real_slots")
        text = torch.randn(2, 6, 4)
        valid = torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, False, False],
            ]
        )
        logits, competition = model._competitive_ownership(text, valid)
        del logits
        selected = torch.tensor(
            [[True, False, False, False], [True, False, False, False]]
        )
        routing = model._qisca_routing(competition, valid, selected)
        weights = torch.arange(
            routing.numel(),
            dtype=routing.dtype,
        ).reshape_as(routing)
        (routing * weights).sum().backward()

        for parameter in (
            model.slot_queries,
            model.slot_query_projection.weight,
            model.text_key_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_qisca_full_forward_is_finite_under_autocast(self) -> None:
        model = tiny_taper(
            theta=0.10,
            capacity_enabled=True,
            capacity=0.8,
            candidate_mode="all_real_slots",
        )
        reference = torch.randn(2, 4)
        text = torch.randn(2, 6, 4)
        attention = torch.ones(2, 6, dtype=torch.bool)
        content = torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, False, False],
            ]
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(
                reference,
                text,
                attention,
                text_content_mask=content,
            )

        self.assertEqual(output["qasa_attention"].dtype, torch.float32)
        self.assertEqual(output["routing_masks"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["routing_masks"]).all())
        self.assertTrue(torch.isfinite(output["edit_slots"]).all())
        output["q0"].square().sum().backward()
        for parameter in (
            model.slot_queries,
            model.slot_query_projection.weight,
            model.text_key_projection.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_qasa_is_invariant_across_candidate_and_capacity_modes(self) -> None:
        qasa_candidates = tiny_taper(
            capacity_enabled=False,
            candidate_mode="qasa_selected",
            theta=0.10,
        )
        all_candidates = tiny_taper(
            capacity_enabled=True,
            capacity=0.8,
            candidate_mode="all_real_slots",
            theta=0.10,
        )
        all_candidates.load_state_dict(qasa_candidates.state_dict())
        text = torch.randn(2, 6, 4)
        attention = torch.ones(2, 6, dtype=torch.bool)
        content = torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, False, False],
            ]
        )
        qasa_output = qasa_candidates.build_edit_slots(
            text,
            attention,
            text_content_mask=content,
        )
        all_output = all_candidates.build_edit_slots(
            text,
            attention,
            text_content_mask=content,
        )

        for name in (
            "ownership_logits",
            "slot_masks",
            "qasa_attention",
            "qasa_quality",
            "qasa_selected_mask",
            "qasa_selected_count",
            "qasa_final_coverage",
        ):
            torch.testing.assert_close(qasa_output[name], all_output[name])

    def test_r1_regression_and_qasa_invariance_across_routing_modes(self) -> None:
        entmax_model = tiny_taper(routing_mode="entmax15")
        qisca_model = tiny_taper(routing_mode="qisca", theta=0.10)
        qisca_model.load_state_dict(entmax_model.state_dict())
        text = torch.randn(2, 6, 4)
        attention = torch.ones(2, 6, dtype=torch.bool)
        content = torch.tensor(
            [
                [False, True, True, True, True, False],
                [False, True, True, True, False, False],
            ]
        )
        entmax_output = entmax_model.build_edit_slots(
            text,
            attention,
            text_content_mask=content,
        )
        qisca_output = qisca_model.build_edit_slots(
            text,
            attention,
            text_content_mask=content,
        )

        for name in (
            "ownership_logits",
            "slot_masks",
            "qasa_attention",
            "qasa_quality",
            "qasa_selected_mask",
        ):
            torch.testing.assert_close(entmax_output[name], qisca_output[name])
        self.assertTrue(bool(entmax_output["routing_mode_entmax15"]))
        self.assertTrue(bool(qisca_output["routing_mode_qisca"]))
        active = entmax_output["qasa_selected_mask"] & content.any(
            dim=1,
            keepdim=True,
        )
        torch.testing.assert_close(
            entmax_output["routing_masks"].sum(dim=-1)[active],
            torch.ones_like(
                entmax_output["routing_masks"].sum(dim=-1)[active]
            ),
            atol=1e-6,
            rtol=1e-6,
        )
        entmax_forward = entmax_model(
            torch.randn(2, 4),
            text,
            attention,
            text_content_mask=content,
        )
        self.assertTrue(
            torch.equal(
                entmax_forward["execution_selected_mask"],
                entmax_forward["qasa_selected_mask"],
            )
        )
        self.assertTrue(
            torch.equal(
                entmax_forward["hard_active_slot_mask"],
                entmax_forward["qasa_selected_mask"],
            )
        )

    def test_r4_diagnostics_report_rejection_and_capacity_na(self) -> None:
        model = tiny_taper(theta=0.30, capacity_enabled=False)
        soft = torch.full((1, 4, 3), 0.25)
        routing = torch.zeros_like(soft)
        diagnostics = model._assignment_diagnostics(
            slot_masks=soft,
            slot_mass=soft.sum(dim=-1),
            routing_masks=routing,
            routing_slot_mass=routing.sum(dim=-1),
            routing_support_count=torch.zeros(1, 4, dtype=torch.long),
            qasa_selected_mask=torch.ones(1, 4, dtype=torch.bool),
            qasa_quality=torch.ones(1, 4),
            qasa_final_coverage=torch.ones(1),
            hard_active_slot_mask=torch.zeros(1, 4, dtype=torch.bool),
            text_attention_mask=torch.ones(1, 3, dtype=torch.bool),
            text_content_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(float(diagnostics["routing_token_mass_mean"]), 0.0)
        self.assertEqual(float(diagnostics["routing_unassigned_mass_mean"]), 1.0)
        self.assertEqual(
            float(diagnostics["routing_fully_unassigned_token_fraction"]),
            1.0,
        )
        self.assertTrue(
            math.isnan(float(diagnostics["routing_capacity_utilization_mean"]))
        )
        self.assertTrue(
            math.isnan(float(diagnostics["routing_capacity_binding_fraction"]))
        )

    def test_r4_capacity_diagnostics_are_meaningful_when_enabled(self) -> None:
        model = tiny_taper(capacity_enabled=True, capacity=0.5)
        soft = torch.full((1, 4, 2), 0.25)
        routing = torch.tensor(
            [[[0.25, 0.25], [0.10, 0.10], [0.0, 0.0], [0.0, 0.0]]]
        )
        diagnostics = model._assignment_diagnostics(
            slot_masks=soft,
            slot_mass=soft.sum(dim=-1),
            routing_masks=routing,
            routing_slot_mass=routing.sum(dim=-1),
            routing_support_count=(routing > 1e-6).sum(dim=-1),
            qasa_selected_mask=torch.ones(1, 4, dtype=torch.bool),
            qasa_quality=torch.ones(1, 4),
            qasa_final_coverage=torch.ones(1),
            hard_active_slot_mask=torch.ones(1, 4, dtype=torch.bool),
            text_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            text_content_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        self.assertTrue(
            math.isfinite(float(diagnostics["routing_capacity_utilization_mean"]))
        )
        self.assertAlmostEqual(
            float(diagnostics["routing_capacity_binding_fraction"]),
            0.25,
        )

    def test_capacity_diagnostics_use_all_real_solver_candidates(self) -> None:
        model = tiny_taper(
            capacity_enabled=True,
            capacity=0.5,
            candidate_mode="all_real_slots",
        )
        soft = torch.full((1, 4, 2), 0.25)
        routing = torch.tensor(
            [[[0.25, 0.25], [0.125, 0.125], [0.0, 0.0], [0.0, 0.0]]]
        )
        qasa_selected = torch.tensor([[True, False, False, False]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        candidates = model._r4_candidate_mask(qasa_selected, valid)
        diagnostics = model._assignment_diagnostics(
            slot_masks=soft,
            slot_mass=soft.sum(dim=-1),
            routing_masks=routing,
            routing_slot_mass=routing.sum(dim=-1),
            routing_support_count=(routing > 1e-6).sum(dim=-1),
            qasa_selected_mask=qasa_selected,
            qasa_quality=torch.ones(1, 4),
            qasa_final_coverage=torch.ones(1),
            hard_active_slot_mask=routing.sum(dim=-1) > 1e-6,
            text_attention_mask=valid,
            text_content_mask=valid,
            routing_candidate_mask=candidates,
        )
        self.assertAlmostEqual(
            float(diagnostics["routing_capacity_utilization_mean"]),
            0.375,
        )
        self.assertAlmostEqual(
            float(diagnostics["routing_capacity_binding_fraction"]),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
