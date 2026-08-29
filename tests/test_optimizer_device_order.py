import pytest
import torch
from torch import nn
from torch.optim import AdamW

from training.engine import assert_training_setup, trainable_parameters


def test_optimizer_owns_exact_post_placement_parameter_objects() -> None:
    device = torch.device("cpu")
    model = nn.Linear(4, 3).to(device)
    objective = nn.Linear(3, 1).to(device)
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-3)

    assert_training_setup(model, objective, optimizer, device)
    expected_ids = {id(parameter) for parameter in trainable_parameters(model, objective)}
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == expected_ids


def test_setup_rejects_optimizer_with_pre_migration_or_stale_parameters() -> None:
    model = nn.Linear(4, 3)
    objective = nn.Linear(3, 1)
    optimizer_built_too_early = AdamW(trainable_parameters(model, objective), lr=1e-3)
    # Simulate a placement/transformation that replaces a live Parameter object.
    model.weight = nn.Parameter(model.weight.detach().clone())

    with pytest.raises(RuntimeError, match="do not match"):
        assert_training_setup(model, objective, optimizer_built_too_early, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_optimizer_references_final_cuda_parameters() -> None:
    device = torch.device("cuda:0")
    model = nn.Linear(4, 3).to(device)
    objective = nn.Linear(3, 1).to(device)
    optimizer = AdamW(trainable_parameters(model, objective), lr=1e-3)
    assert_training_setup(model, objective, optimizer, device)
    assert all(parameter.device == device for parameter in trainable_parameters(model, objective))
