from __future__ import annotations

import torch

from training.taper_mag_audit import TeacherShadowAuditor, validate_teacher_shadow_report, write_json
from test_taper_mag_training_contract import _end_to_end_fixture


def test_teacher_shadow_is_deterministic_and_updates_no_parameters(tmp_path) -> None:
    fg, taper, policy, supervision, engine = _end_to_end_fixture()
    taper.eval()
    fg.eval()
    encoded = engine.encode_policy(policy)
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixed checkpoint audit identity")

    def run() -> tuple[dict, list[dict]]:
        auditor = TeacherShadowAuditor(
            taper,
            fg.model,
            engine.negative_bank,
            engine.teacher,
            seed=17,
            near_tie_band=0.01,
        )
        auditor.update(
            encoded,
            supervision,
            sample_ids=("s0", "s1"),
            reference_ids=policy.reference_ids,
            modification_texts=policy.modification_texts,
        )
        return auditor.finalize(
            checkpoint_path=checkpoint,
            cache_manifest_hashes={"train_global": "abc"},
        ), auditor.traces

    before_actor = {name: value.detach().clone() for name, value in taper.state_dict().items()}
    before_critic = {name: value.detach().clone() for name, value in taper.utility.state_dict().items()}
    first, first_traces = run()
    second, second_traces = run()
    assert first == second
    assert first_traces == second_traces
    assert not first["parameter_updates"]["changed"]
    assert first["numerical_health"]["finite"]
    assert first["sample_count"] == 2
    assert first["candidate_space"]["oracle_action_realized_gain"] >= first["candidate_space"]["random_action_realized_gain"]
    assert first["clone_controls"]["query_delta_arithmetic_used"] is False
    assert first["clone_controls"]["execution_contract"] == "operator_to_executor_to_state_to_readout"
    assert "repeat_best_causal_teacher_gain" in first["clone_controls"]
    assert "target_id" not in first_traces[0]["policy"]
    assert "target_id" in first_traces[0]["supervision_audit"]
    for name, value in taper.state_dict().items():
        torch.testing.assert_close(value, before_actor[name])
    for name, value in taper.utility.state_dict().items():
        torch.testing.assert_close(value, before_critic[name])
    report_path = tmp_path / "teacher_shadow_report.json"
    write_json(report_path, first)
    assert len(validate_teacher_shadow_report(report_path)) == 64
