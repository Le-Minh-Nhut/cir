import torch
import pytest

from workbench.composition.model_csmcir import _load_csmcir_checkpoint


def test_checkpoint_loads_real_linear_weights_and_ignores_training_head():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, device="cpu", dtype=torch.float32)
        _load_csmcir_checkpoint(model, {
            "weight": torch.tensor([[2.0, 3.0]], device="cpu", dtype=torch.float32),
            "bias": torch.tensor([1.0], device="cpu", dtype=torch.float32),
            "training_head.weight": torch.tensor([[9.0, 9.0]], device="cpu", dtype=torch.float32),
        })
        actual = model(torch.tensor([[4.0, 5.0]], device="cpu", dtype=torch.float32))
    torch.testing.assert_close(actual, torch.tensor([[24.0]], device="cpu", dtype=torch.float32))


def test_checkpoint_missing_model_weight_fails():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, device="cpu", dtype=torch.float32)
        with pytest.raises(ValueError):
            _load_csmcir_checkpoint(model, {
                "bias": torch.tensor([1.0], device="cpu", dtype=torch.float32),
            })
