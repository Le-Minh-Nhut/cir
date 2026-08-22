from collections import defaultdict
from pathlib import Path

import torch
import wandb
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.common import CIRBatch
from cache.features import TextFeatureCache, get_features_by_ids, get_text_features_by_sample_ids


def _compute_total_loss(loss_dict: dict[str, Tensor], loss_weights: dict[str, float]) -> Tensor:
    """Combine model-specific loss components."""

    if not loss_weights:
        raise ValueError("loss_weights must not be empty")

    total_loss = 0.0

    for name, weight in loss_weights.items():
        loss = loss_dict[name]
        weighted_loss = weight * loss

        total_loss = total_loss + weighted_loss

    return total_loss


def _set_epoch(train_loader: DataLoader, epoch: int) -> None:
    """Propagate epoch state to Dataset/Sampler when supported."""

    # một sample khi được lấy ra thì nó trông như thế nào
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)

    # lấy sample nào trước, sample nào sau
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)


def train_one_epoch(
    prepare_batch_fn,
    model, # model được train 
    train_loader: DataLoader, # DataLoader của tập train
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler, # dùng cho amp
    device: torch.device,
    epoch: int, # epoch hiện tại -> hiển thị
    loss_weights: dict[str, float], # trọng số từng hàm loss
    use_amp: bool = True, # bật tắt mixed precision
) -> dict[str, float]:
    """Train model for one epoch."""

    model.train()
    amp_enabled = use_amp and device.type == "cuda" # kiểm tra xem cho chép dùng amp không 
    running_total_loss = 0.0
    running_components = defaultdict(float) # nghĩa là dictionary mà key chưa tồn tại thì tự mặc định = 0.0
    num_steps = 0 # đến số batch
    progress = tqdm(train_loader, desc=f"Train [{epoch + 1}]", dynamic_ncols=True)

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch = prepare_batch_fn(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss_dict = model.compute_loss(batch) # model nhận batch, chạy forward và tính các loss riêng

            total_loss = _compute_total_loss(loss_dict=loss_dict, loss_weights=loss_weights,)

        scaler.scale(total_loss).backward() # phóng to loss rồi cho backward
        scaler.step(optimizer) # tương đương optimizer.step() cho cập nhật trọng số 
        scaler.update() # sau mỗi batch scaler tự điều chỉnh hệ số scale cho batch sau.

        total_loss_value = total_loss.item()

        running_total_loss += total_loss_value
        num_steps += 1

        for name, loss in loss_dict.items():
            running_components[name] += loss.item()

        progress.set_postfix(loss=f"{total_loss_value:.4f}") # cập nhật progress bar 

    metrics = {
        "total_loss": running_total_loss / num_steps,
    }

    for name, value in running_components.items():
        metrics[name] = value / num_steps

    return metrics

def prepare_batch(
    batch: CIRBatch,
    device: torch.device,
    retrieval_features: torch.Tensor,
    native_features: torch.Tensor,
    retrieval_name_to_idx,
    native_name_to_idx,
    text_cache: TextFeatureCache,
) -> dict[str, object]:
    target_ids = list(batch.target_ids)

    if any(target_id is None for target_id in target_ids):
        raise ValueError("Training sample is missing target_id")
    
    reference_native = get_features_by_ids(batch.reference_ids, native_features, native_name_to_idx).to(device, dtype=torch.float32)
    target_features = get_features_by_ids(target_ids, retrieval_features, retrieval_name_to_idx).to(device, dtype=torch.float32)
    reference_features = reference_native[:, 0, :]
    (text_states, teacher_text_states, attention_mask, content_mask) = get_text_features_by_sample_ids(batch.sample_ids, batch.modification_texts, text_cache)
    text_states = text_states.to(
        device=device,
        dtype=torch.float32,
        # non_blocking=True,
    )

    teacher_text_states = teacher_text_states.to(
        device=device,
        dtype=torch.float32,
        # non_blocking=True,
    )

    attention_mask = attention_mask.to(
        device=device,
        dtype=torch.bool,
        # non_blocking=True,
    )

    content_mask = content_mask.to(
        device=device,
        dtype=torch.bool,
        # non_blocking=True,
    )

    return {
        "reference_features": reference_features,
        "teacher_reference_features": reference_native,
        "target_features": target_features,
        "text_states": text_states,
        "teacher_text_states": teacher_text_states,
        "text_attention_mask": attention_mask,
        "text_content_mask": content_mask,
        "target_ids": target_ids,
    }

def taper_state_dict(model):
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("teacher.")
    }

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    evaluate_fn,
    *,
    num_epochs: int,
    device: torch.device,
    loss_weights: dict[str, float],
    primary_metric: str,
    output_dir: str | Path,
    use_amp: bool = True,
    prepare_batch_fn
) -> None:

    model.to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_metric = float("-inf")
    best_epoch = 0

    best_model_path = output_dir / "best.pt"
    last_model_path = output_dir / "last.pt"

    for epoch in range(num_epochs):
        _set_epoch(train_loader=train_loader, epoch=epoch)

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            loss_weights=loss_weights,
            use_amp=use_amp,
            prepare_batch_fn=prepare_batch_fn,
        )

        model.eval()

        with torch.no_grad():
            val_metrics = dict(evaluate_fn(model))

        current_metric = float(val_metrics[primary_metric])

        # Always keep the latest model.
        torch.save(taper_state_dict(model), last_model_path)

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1

            torch.save(taper_state_dict(model), best_model_path)

            print(f"Saved best.pt | {primary_metric}={best_metric:.4f}")

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss={train_metrics['total_loss']:.4f} | "
            f"{primary_metric}={current_metric:.4f} | "
            f"best={best_metric:.4f} | "
            f"null={train_metrics.get('diagnostic/null_ownership_rate', float('nan')):.3f} | "
            f"null_argmax={train_metrics.get('diagnostic/null_argmax_fraction', float('nan')):.3f} | "
            f"all_null={train_metrics.get('diagnostic/all_null_argmax_sample_fraction', float('nan')):.3f} | "
            f"active_slots={train_metrics.get('diagnostic/ownership_active_slot_count', float('nan')):.2f} | "
            f"hard_active={train_metrics.get('diagnostic/execution_hard_active_slot_count', float('nan')):.2f} | "
            f"dominant={train_metrics.get('diagnostic/dominant_slot_share', float('nan')):.3f} | "
            f"monopoly={train_metrics.get('diagnostic/near_monopoly_fraction', float('nan')):.3f}"
        )

        print(
            "A4 VAL | "
            f"soft_H={val_metrics.get('slot/soft_entropy', float('nan')):.3f} | "
            f"margin={val_metrics.get('slot/soft_top1_top2_margin', float('nan')):.3f} | "
            f"conf={val_metrics.get('slot/soft_winner_confidence', float('nan')):.3f} | "
            f"multi={val_metrics.get('slot/multi_active_sample_rate', float('nan')):.3f} | "
            f"sem_cos={val_metrics.get('slot/active_pair_semantic_cosine', float('nan')):.3f} | "
            f"effect_cos={val_metrics.get('slot/active_pair_teacher_effect_cosine', float('nan')):.3f} | "
            f"edit_cos={val_metrics.get('slot/active_pair_edit_slot_cosine', float('nan')):.3f} | "
            f"exec_steps={val_metrics.get('slot/valid_execution_steps_mean', float('nan')):.2f}"
        )

        for slot_id in range(model.num_slots):
            print(
                f"  S{slot_id} | "
                f"nonempty={val_metrics.get(f'slot/slot_{slot_id}_nonempty_rate', float('nan')):.3f} | "
                f"gate={val_metrics.get(f'slot/slot_{slot_id}_gate_mean', float('nan')):.3f} | "
                f"hard_active={val_metrics.get(f'slot/slot_{slot_id}_hard_active_rate', float('nan')):.3f} | "
                f"exec={val_metrics.get(f'slot/slot_{slot_id}_execution_rate', float('nan')):.3f}"
            )


        if wandb.run is not None:
            log_data = {
                "epoch": epoch + 1,
                "train/lr": optimizer.param_groups[0]["lr"],
                "best/metric": best_metric,
                "best/epoch": best_epoch,
            }

            # Total + component training losses.
            for name, value in train_metrics.items():
                log_data[f"train/{name}"] = value

            # Retrieval metrics and optional validation losses.
            for name, value in val_metrics.items():
                log_data[f"val/{name}"] = value

            wandb.log(log_data)