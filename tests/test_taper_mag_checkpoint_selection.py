from __future__ import annotations

from training.checkpoint_selection import CheckpointSelectionState


def _functional(*, firewall=True, numerical=True, collapsed=False, rank=2.0):
    return {
        "firewall": {"pass": firewall},
        "numerical_health": {"pass": numerical},
        "candidate_space": {"catastrophic_exact_collapse": collapsed},
        "dynamic_policy": {"valid": True},
        "repeat": {"valid": True},
        "response_rank": {"mean_effective_rank": rank},
    }


def test_checkpoint_selection_rules_and_deterministic_ties() -> None:
    state = CheckpointSelectionState()
    first = state.select(
        retrieval={"mean_recall": 10.0},
        policy={"mean_regret": 0.5, "median_regret": 0.4},
        functional=_functional(),
    )
    assert set(first) == {
        "last.ckpt",
        "best_retrieval_valid.ckpt",
        "best_policy_regret.ckpt",
        "best_functional_health.ckpt",
    }
    tied = state.select(
        retrieval={"mean_recall": 10.0},
        policy={"mean_regret": 0.5, "median_regret": 0.4},
        functional=_functional(),
    )
    assert tied == {"last.ckpt": "latest completed epoch"}
    improved_policy = state.select(
        retrieval={"mean_recall": 9.0},
        policy={"mean_regret": 0.4, "median_regret": 0.5},
        functional=_functional(),
    )
    assert "best_policy_regret.ckpt" in improved_policy
    assert "best_retrieval_valid.ckpt" not in improved_policy


def test_functional_and_retrieval_selection_reject_failures_and_last_updates() -> None:
    state = CheckpointSelectionState()
    rejected = state.select(
        retrieval={"mean_recall": 100.0},
        policy={"mean_regret": 0.0, "median_regret": 0.0},
        functional=_functional(firewall=False),
    )
    assert rejected == {"last.ckpt": "latest completed epoch"}
    collapsed = state.select(
        retrieval={"mean_recall": 10.0},
        policy={"mean_regret": 1.0, "median_regret": 1.0},
        functional=_functional(collapsed=True),
    )
    assert "best_retrieval_valid.ckpt" in collapsed
    assert "best_functional_health.ckpt" not in collapsed
