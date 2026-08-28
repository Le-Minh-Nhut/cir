from __future__ import annotations

import pytest

from training.progress import (
    training_progress_description,
    training_progress_postfix,
)


def test_training_progress_description_exposes_curriculum_state() -> None:
    assert (
        training_progress_description(1, 60, "actor_warmup", 1)
        == "Epoch 1/60 [actor_warmup T=1]"
    )
    with pytest.raises(ValueError, match="epoch"):
        training_progress_description(0, 60, "actor_warmup", 1)


def test_training_progress_postfix_is_observational() -> None:
    postfix = training_progress_postfix(
        running_loss=4.446,
        processed_batches=3,
        global_step=91,
        max_updates=4260,
        learning_rate=1.8e-4,
        micro_step=3,
        accumulation=8,
    )
    assert postfix == {
        "loss": "1.4820",
        "opt_step": "91/4260",
        "lr": "1.80e-04",
        "accum": "4/8",
    }
