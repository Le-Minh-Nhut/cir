"""Tests for SPRC registered model name, backbone selection, and evaluator audit."""
from __future__ import annotations

from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
SPRC_DIR = REPO / "workbench" / "third_party" / "SPRC"
EVALUATOR = SPRC_DIR / "src" / "blip_validate.py"
ALIGN_PROMPT_MODEL = (
    SPRC_DIR / "src" / "lavis" / "models" / "blip2_models" / "blip2_qformer_cir_align_prompt.py"
)


def test_sprc_model_registration_and_backbones() -> None:
    assert ALIGN_PROMPT_MODEL.is_file(), f"Missing {ALIGN_PROMPT_MODEL}"
    content = ALIGN_PROMPT_MODEL.read_text(encoding="utf-8")
    assert '@registry.register_model("blip2_cir_align_prompt")' in content, (
        "blip2_cir_align_prompt not registered in blip2_qformer_cir_align_prompt.py"
    )
    assert '"pretrain": "configs/models/blip2/blip2_pretrain.yaml"' in content, (
        "Missing pretrain config (ViT-g)"
    )
    assert '"pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml"' in content, (
        "Missing pretrain_vitL config (ViT-L)"
    )


def test_sprc_evaluator_flags_and_missing_save_memory() -> None:
    assert EVALUATOR.is_file(), f"Missing {EVALUATOR}"
    content = EVALUATOR.read_text(encoding="utf-8")
    assert '"--dataset"' in content
    assert '"--blip-model-name"' in content
    assert '"--backbone"' in content
    assert '"--model-path"' in content
    # The pinned evaluator defaults to an unregistered model name
    assert 'default="blip2_cir_rerank_learn"' in content
    # The pinned standalone evaluator blip_validate.py lacks --save-memory
    assert "--save-memory" not in content
