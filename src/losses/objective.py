from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from .retrieval import TerminalRetrievalLoss, positive_mask_from_ids


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    retrieval_temperature: float = 0.07


def _mean_batch_l2(values: Tensor) -> Tensor:
    return values.detach().float().flatten(1).norm(dim=-1).mean()


class IAGSRMEObjective(nn.Module):
    """Final-state retrieval supervision with sequential execution diagnostics."""

    def __init__(self, config: ObjectiveConfig = ObjectiveConfig()) -> None:
        super().__init__()
        self.config = config
        self.terminal = TerminalRetrievalLoss(config.retrieval_temperature)

    def forward(
        self,
        output: Mapping[str, object],
        target_embeddings: Tensor,
        target_ids: Sequence[str],
    ) -> Mapping[str, Tensor]:
        query = output["query"]
        initial_state = output["initial_state"]
        steps = output["steps"]
        assert isinstance(query, Tensor)
        assert isinstance(initial_state, Tensor)
        assert isinstance(steps, list)

        positive = positive_mask_from_ids(target_ids, target_embeddings.device)
        terminal = self.terminal(query, target_embeddings, positive)
        components: dict[str, Tensor] = {
            "terminal": terminal,
            "total": terminal,
        }

        for step in steps:
            slot = int(step["slot"])
            prefix = f"slot_{slot}"
            edit = step["edit"]
            action = step["action"]
            exec_mask = step["exec_mask"]
            delta = step["delta"]
            parent = step["parent_state"]
            state = step["state"]
            assert isinstance(edit, Tensor)
            assert isinstance(action, Tensor)
            assert isinstance(exec_mask, Tensor)
            assert isinstance(delta, Tensor)
            assert isinstance(parent, Tensor)
            assert isinstance(state, Tensor)

            components[f"{prefix}_delta_l2"] = _mean_batch_l2(delta)
            components[f"{prefix}_delta_patch_l2_mean"] = (
                delta.detach().float().norm(dim=-1).mean()
            )
            components[f"{prefix}_exec_mask_mean"] = exec_mask.detach().float().mean()
            components[f"{prefix}_exec_mask_support"] = (
                exec_mask.detach().float().sum(dim=-1).mean()
            )
            components[f"{prefix}_action_l2"] = _mean_batch_l2(action)
            components[f"{prefix}_edit_l2"] = _mean_batch_l2(edit)
            components[f"{prefix}_state_drift_l2"] = _mean_batch_l2(state - parent)
            components[f"{prefix}_cumulative_drift_l2"] = _mean_batch_l2(
                state - initial_state
            )

        return components
