from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _compose(backbone: str):
    config_dir = str(Path(__file__).parents[1] / "conf")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=[f"backbone={backbone}"])


def test_track_b_readout_configs_differ_only_in_declared_ablation_identity() -> None:
    qg = _compose("fgclip_base_full_qg")
    native = _compose("fgclip_base_full_native_cls")
    qg_backbone = OmegaConf.to_container(qg.backbone, resolve=True)
    native_backbone = OmegaConf.to_container(native.backbone, resolve=True)

    assert qg_backbone.pop("global_readout_mode") == "learned_qg"
    assert native_backbone.pop("global_readout_mode") == "native_cls"
    assert qg_backbone.pop("readout_experiment") == "R0-QG"
    assert native_backbone.pop("readout_experiment") == "R0-NCLS"
    assert qg_backbone.pop("experiment_identity") == "R0-QG-FULL"
    assert native_backbone.pop("experiment_identity") == "R0-NCLS-FULL"
    qg_backbone.pop("name")
    native_backbone.pop("name")
    assert qg_backbone == native_backbone

    # All non-backbone experiment/model/objective/protocol settings remain identical.
    for group in ("dataset", "model", "objective", "experiment", "protocol"):
        assert OmegaConf.to_container(qg[group], resolve=True) == OmegaConf.to_container(
            native[group], resolve=True
        )


def test_text_only_configs_change_only_finetune_policy_and_identity() -> None:
    pairs = (
        ("fgclip_base_full_qg", "fgclip_base_text_qg", "R0-QG-FULL", "R0-QG-TEXT"),
        (
            "fgclip_base_full_native_cls",
            "fgclip_base_text_native_cls",
            "R0-NCLS-FULL",
            "R0-NCLS-TEXT",
        ),
    )
    for full_name, text_name, full_identity, text_identity in pairs:
        full = _compose(full_name)
        text = _compose(text_name)
        full_backbone = OmegaConf.to_container(full.backbone, resolve=True)
        text_backbone = OmegaConf.to_container(text.backbone, resolve=True)

        assert full_backbone.pop("train_vision") is True
        assert text_backbone.pop("train_vision") is False
        assert full_backbone.pop("finetune_policy") == "full"
        assert text_backbone.pop("finetune_policy") == "text_only"
        assert full_backbone.pop("experiment_identity") == full_identity
        assert text_backbone.pop("experiment_identity") == text_identity
        full_backbone.pop("name")
        text_backbone.pop("name")
        assert full_backbone == text_backbone

        for group in ("dataset", "model", "objective", "experiment", "protocol"):
            assert OmegaConf.to_container(full[group], resolve=True) == OmegaConf.to_container(
                text[group], resolve=True
            )
