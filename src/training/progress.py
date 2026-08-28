from __future__ import annotations


def training_progress_description(
    epoch: int,
    total_epochs: int,
    phase: str,
    horizon: int,
) -> str:
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within [1,total_epochs]")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return f"Epoch {epoch}/{total_epochs} [{phase} T={horizon}]"


def training_progress_postfix(
    *,
    running_loss: float,
    processed_batches: int,
    global_step: int,
    max_updates: int,
    learning_rate: float,
    micro_step: int,
    accumulation: int,
) -> dict[str, str]:
    if processed_batches <= 0 or accumulation <= 0 or max_updates <= 0:
        raise ValueError("progress counters must be positive")
    return {
        "loss": f"{running_loss / processed_batches:.4f}",
        "opt_step": f"{global_step}/{max_updates}",
        "lr": f"{learning_rate:.2e}",
        "accum": f"{micro_step % accumulation + 1}/{accumulation}",
    }
