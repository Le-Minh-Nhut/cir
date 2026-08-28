from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW

from backbones.fgclip2_base import FGCLIP2BaseBackbone
from models.taper_mag.model import TaperMAG


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    actor_lr: float = 2e-4
    utility_lr: float = 3e-4
    text_lr: float = 5e-6
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    gradient_clip: float = 1.0
    ema_decay: float = 0.999


def _uses_weight_decay(name: str, parameter: nn.Parameter) -> bool:
    lowered = name.lower()
    if parameter.ndim < 2 or lowered.endswith("bias"):
        return False
    if any(token in lowered for token in ("norm", "queries", "layerscale", "support_bias")):
        return False
    return True


def build_optimizer(
    model: TaperMAG,
    backbone: FGCLIP2BaseBackbone,
    config: OptimizerConfig | None = None,
) -> AdamW:
    config = config or OptimizerConfig()
    categorized: dict[tuple[str, bool], list[nn.Parameter]] = {}
    seen: set[int] = set()

    def add(category: str, name: str, parameter: nn.Parameter) -> None:
        if not parameter.requires_grad:
            return
        if id(parameter) in seen:
            raise RuntimeError(f"Optimizer parameter duplicated: {name}")
        seen.add(id(parameter))
        categorized.setdefault((category, _uses_weight_decay(name, parameter)), []).append(parameter)

    for name, parameter in model.named_parameters():
        category = "utility" if name.startswith("utility.") else "actor"
        add(category, name, parameter)
    for name, parameter in backbone.model.named_parameters():
        add("text", f"backbone.{name}", parameter)

    learning_rates = {
        "actor": config.actor_lr,
        "utility": config.utility_lr,
        "text": config.text_lr,
    }
    groups = []
    for (category, decay), parameters in categorized.items():
        groups.append(
            {
                "params": parameters,
                "lr": learning_rates[category],
                "weight_decay": config.weight_decay if decay else 0.0,
                "group_name": f"{category}_{'decay' if decay else 'no_decay'}",
            }
        )
    return AdamW(groups, betas=config.betas, eps=config.eps)


def format_parameter_report(backbone: FGCLIP2BaseBackbone, model: TaperMAG) -> str:
    taper_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = backbone.parameter_report(taper_parameters=taper_trainable)
    lines = [
        f"total FG-CLIP2 params: {report['total_fgclip2_params']:,}",
        f"trainable FG-CLIP2 params: {report['trainable_fgclip2_params']:,}",
        f"trainable text params: {report['trainable_text_params']:,}",
        f"trainable vision params: {report['trainable_vision_params']:,}",
        f"trainable TAPER params: {report['trainable_taper_params']:,}",
        "unfrozen text blocks:",
        *[f"  {name}" for name in report["unfrozen_text_blocks"]],
    ]
    return "\n".join(lines)
