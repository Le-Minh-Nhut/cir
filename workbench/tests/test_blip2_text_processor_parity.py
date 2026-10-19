"""Superseded F06 test: kept only as the record of the config-level question it answered.

F06 is applied now, so the executed check lives in `workbench/tests/test_r9_defect_regressions.py`
(the AST assertion that no wrapper calls `load_model_and_preprocess`, the pinned-config assertion,
and the real `BlipCaptionProcessor` output) plus `workbench/scripts/lavis_processor_parity.py`.

The execution half was deferred here because `lavis.processors` appeared to pull `torchaudio` and be
uninstallable. That diagnosis was incomplete: the `torchaudio` import is a single dead line in
`lavis/processors/audio_processors.py`, reachable only through `lavis/processors/__init__.py`, and the
pinned processor itself needs nothing from it. `workbench/replay/lavis_processors.py` loads the real
pinned package with that one module left out, which is what made the executed check possible.

What remains in this file is the config-level resolution, which is still true and still cheap.
"""
import re
from pathlib import Path
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "model_dir,config_rel_path",
    [
        ("HINT", "lavis/configs/models/blip2/blip2_pretrain.yaml"),
        ("HABIT", "lavis/configs/models/blip2/blip2_pretrain.yaml"),
        ("INTENT", "lavis/configs/models/blip2/blip2_pretrain.yaml"),
        ("ConeSep", "lavis/configs/models/blip2/blip2_pretrain.yaml"),
        ("AirKnow", "lavis/configs/models/blip2/blip2_pretrain.yaml"),
    ],
)
def test_all_five_models_resolve_identical_blip_caption_eval_processor(model_dir: str, config_rel_path: str):
    config_file = REPO_ROOT / "workbench" / "third_party" / model_dir / config_rel_path
    assert config_file.is_file(), f"Config file missing: {config_file}"

    cfg = OmegaConf.load(config_file)
    preprocess_cfg = cfg.get("preprocess")
    assert preprocess_cfg is not None

    txt_proc_cfg = preprocess_cfg.get("text_processor")
    assert txt_proc_cfg is not None

    eval_cfg = txt_proc_cfg.get("eval")
    assert eval_cfg is not None
    assert eval_cfg.get("name") == "blip_caption"
    # prompt and max_words are unstated in yaml, defaulting to "" and 50
    assert eval_cfg.get("prompt", "") == ""
    assert eval_cfg.get("max_words", 50) == 50


def test_the_five_wrappers_agree_on_the_text_processor_the_configs_declare():
    """All five models resolve the SAME text preprocessing.

    The configs are read from the pinned checkouts. The executed half of F06 -- the processor's own
    output, and the absence of any `load_model_and_preprocess` call -- is in
    `test_r9_defect_regressions.py`; see this file's module docstring for why it was moved.
    """
    import re
    from pathlib import Path

    import pytest
    from omegaconf import OmegaConf

    resolved = {}
    for model_dir in ("HINT", "HABIT", "INTENT", "ConeSep", "AirKnow"):
        config_file = (REPO_ROOT / "workbench" / "third_party" / model_dir
                       / "lavis" / "configs" / "models" / "blip2" / "blip2_pretrain.yaml")
        assert config_file.is_file(), config_file
        config = OmegaConf.load(config_file)
        eval_cfg = config.get("preprocess", {}).get("text_processor", {}).get("eval")
        assert eval_cfg is not None, config_file
        assert eval_cfg.get("name") == "blip_caption", config_file
        # The values the processor is constructed with, resolved through the config default.
        resolved[model_dir] = (eval_cfg.get("prompt", ""), eval_cfg.get("max_words", 50))

    assert len(set(resolved.values())) == 1, f"the five models disagree: {resolved}"
    prompt, max_words = next(iter(resolved.values()))
    assert prompt == ""
    assert max_words == 50

    # And the pinned processor really implements that contract: the pattern and truncation are read
    # from the source rather than restated, so a change upstream moves this assertion.
    source = (REPO_ROOT / "workbench" / "third_party" / "HINT" / "lavis" / "processors"
              / "blip_processors.py").read_text()
    assert re.search(r"class BlipCaptionProcessor", source)
    assert "max_words" in source
    assert source.count("re.sub") >= 2
