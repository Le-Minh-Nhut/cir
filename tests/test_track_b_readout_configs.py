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
    qg_backbone.pop("name")
    native_backbone.pop("name")
    assert qg_backbone == native_backbone

    # All non-backbone experiment/model/objective/protocol settings remain identical.
    for group in ("dataset", "model", "objective", "experiment", "protocol"):
        assert OmegaConf.to_container(qg[group], resolve=True) == OmegaConf.to_container(
            native[group], resolve=True
        )
