import inspect

import torch


def test_target_permutation_cannot_change_forward(core, synthetic_encoded) -> None:
    core.eval()
    target_ids = torch.tensor([0, 1, 2])
    first = core(synthetic_encoded)
    target_ids = target_ids[torch.tensor([2, 0, 1])]
    second = core(synthetic_encoded)
    assert target_ids.tolist() == [2, 0, 1]
    for name in ("intents", "supports", "final_state", "final_query"):
        assert torch.equal(getattr(first, name), getattr(second, name))
    for left, right in zip(first.trace, second.trace, strict=True):
        for name in ("contexts", "delta_z", "candidate_states", "candidate_queries", "scores"):
            assert torch.equal(getattr(left, name), getattr(right, name))
    assert "target" not in inspect.signature(core.forward).parameters

