from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CheckpointSelectionState:
    best_retrieval: float = -float("inf")
    best_policy_key: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    best_functional_key: tuple[float, float] = (-float("inf"), -float("inf"))

    def state_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["best_policy_key"] = list(self.best_policy_key)
        result["best_functional_key"] = list(self.best_functional_key)
        return result

    @classmethod
    def from_state_dict(cls, state: dict[str, Any] | None) -> CheckpointSelectionState:
        if state is None:
            return cls()
        return cls(
            best_retrieval=float(state["best_retrieval"]),
            best_policy_key=tuple(float(value) for value in state["best_policy_key"]),
            best_functional_key=tuple(
                float(value) for value in state["best_functional_key"]
            ),
        )

    def select(
        self,
        *,
        retrieval: dict[str, float],
        policy: dict[str, Any],
        functional: dict[str, Any],
    ) -> dict[str, str]:
        """Stable predeclared selection; `last` is always emitted."""
        selected = {"last.ckpt": "latest completed epoch"}
        firewall_ok = bool(functional["firewall"]["pass"])
        numerical_ok = bool(functional["numerical_health"]["pass"])
        mean_recall = float(retrieval["mean_recall"])
        if firewall_ok and numerical_ok and mean_recall > self.best_retrieval:
            self.best_retrieval = mean_recall
            selected["best_retrieval_valid.ckpt"] = (
                "highest validation mean_recall among firewall/numerical survivors"
            )

        policy_key = (
            float(policy.get("mean_regret", float("inf"))),
            float(policy.get("median_regret", float("inf"))),
            -mean_recall,
        )
        if firewall_ok and numerical_ok and policy_key < self.best_policy_key:
            self.best_policy_key = policy_key
            selected["best_policy_regret.ckpt"] = (
                "lowest mean regret; tie-break median regret then higher validation mean_recall"
            )

        candidate_ok = not bool(
            functional["candidate_space"].get("catastrophic_exact_collapse", False)
        )
        dynamic_valid = bool(functional["dynamic_policy"].get("valid", True))
        repeat_valid = bool(functional["repeat"].get("valid", True))
        response_rank = float(
            functional["response_rank"].get("mean_effective_rank", 0.0)
        )
        functional_key = (mean_recall, response_rank)
        if (
            firewall_ok
            and numerical_ok
            and candidate_ok
            and dynamic_valid
            and repeat_valid
            and functional_key > self.best_functional_key
        ):
            self.best_functional_key = functional_key
            selected["best_functional_health.ckpt"] = (
                "lexicographic survivors: firewall, numerical, no exact candidate collapse, "
                "valid dynamic/repeat metrics; then retrieval, response rank"
            )
        return selected
