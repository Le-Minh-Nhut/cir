"""CPU-only checks that every CIRCUS worker spawn inherits the resolved Hugging Face cache root.

The registry declares `HF_HOME` as an operator setting, and `workbench.env` may hold it with no
process-environment copy. The controller previously spawned model workers with the default
(inherited) environment, so a file-only `HF_HOME` never reached the child and the child resolved a
different cache root than the controller. These checks drive a real child through the TME adapter's
own spawn site and read `huggingface_hub.constants.HF_HOME` back out of it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from workbench.backend import model_settings, operator_config
from workbench.composition import model_tme

CACHE_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")

CHILD = r"""
import json, sys
from pathlib import Path

import huggingface_hub.constants as constants
import numpy as np

job_path = Path(sys.argv[sys.argv.index("--worker-job") + 1])
job = json.loads(job_path.read_text(encoding="utf-8"))
(job_path.parent / "probe.json").write_text(json.dumps({"hf_home": constants.HF_HOME}))
np.save(job["output_path"], np.zeros((len(job["image_paths"]), 256), dtype=np.float32),
        allow_pickle=False)
print("CIRCUS_WORKER:" + json.dumps({"ok": True}), flush=True)
for line in sys.stdin:
    if json.loads(line).get("stop"):
        break
"""


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path: Path):
    """Point `workbench.env` at a temp file, clear env+file caches, restore both after."""
    for name in CACHE_VARS:
        monkeypatch.delenv(name, raising=False)
    config_path = tmp_path / "workbench.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", config_path)
    model_settings.file_settings.cache_clear()
    try:
        yield config_path
    finally:
        model_settings.file_settings.cache_clear()


def _run_probe(tmp_path: Path, monkeypatch, label: str, return_env: bool = False):
    """Spawn the real child through the TME adapter spawn site and return its HF_HOME."""
    adapter = model_tme.Adapter()
    adapter.interpreter = sys.executable
    adapter._verified_interpreter = sys.executable
    adapter.work_root = tmp_path / label
    image = tmp_path / f"{label}.png"
    image.write_bytes(b"")
    real_popen = subprocess.Popen
    seen_env: list[dict[str, str]] = []

    def start(argv, **kwargs):
        seen_env.append(kwargs["env"])
        return real_popen([sys.executable, "-c", CHILD, *argv], **kwargs)

    monkeypatch.setattr(model_tme.subprocess, "Popen", start)
    result = adapter.encode_queries([str(image)], ["edit"], [None])
    assert result.shape == (1, 256)
    probes = sorted(adapter._job_dir.parent.rglob("probe.json"))
    assert len(probes) == 1, "the child must have written exactly one probe"
    hf_home = json.loads(probes[0].read_text(encoding="utf-8"))["hf_home"]
    adapter.close()
    assert len(seen_env) == 1
    return seen_env[0] if return_env else hf_home


def test_file_only_hf_home_reaches_child(tmp_path, monkeypatch, clean_cache_env) -> None:
    root = tmp_path / "file-cache"
    clean_cache_env.write_text(f"HF_HOME={root}\n", encoding="utf-8")
    model_settings.file_settings.cache_clear()

    assert _run_probe(tmp_path, monkeypatch, "file") == str(root)


def test_environment_only_hf_home_reaches_child(tmp_path, monkeypatch, clean_cache_env) -> None:
    root = tmp_path / "env-cache"
    monkeypatch.setenv("HF_HOME", str(root))

    assert _run_probe(tmp_path, monkeypatch, "env") == str(root)


def test_environment_wins_over_file(tmp_path, monkeypatch, clean_cache_env) -> None:
    clean_cache_env.write_text(f"HF_HOME={tmp_path / 'file-cache'}\n", encoding="utf-8")
    model_settings.file_settings.cache_clear()
    env_root = tmp_path / "env-cache"
    monkeypatch.setenv("HF_HOME", str(env_root))

    assert _run_probe(tmp_path, monkeypatch, "conflict") == str(env_root)


def test_nothing_configured_keeps_huggingface_default(tmp_path, monkeypatch, clean_cache_env) -> None:
    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    expected = os.path.expanduser(
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"))

    assert _run_probe(tmp_path, monkeypatch, "default") == expected


def test_relative_file_setting_resolves_absolute(tmp_path, monkeypatch, clean_cache_env) -> None:
    clean_cache_env.write_text("HF_HOME=relcache\n", encoding="utf-8")
    model_settings.file_settings.cache_clear()
    expected = model_settings.resolve_path_setting("HF_HOME")
    assert expected is not None and expected.is_absolute()

    child_hf_home = _run_probe(tmp_path, monkeypatch, "relative")
    assert child_hf_home == str(expected)
    assert Path(child_hf_home).is_absolute()

def test_sibling_env_cache_does_not_suppress_file_hf_home(tmp_path, monkeypatch, clean_cache_env) -> None:
    """(f) An exported sibling cache var must not hide a file-only `HF_HOME`."""
    file_root = tmp_path / "file-cache"
    clean_cache_env.write_text(f"HF_HOME={file_root}\n", encoding="utf-8")
    model_settings.file_settings.cache_clear()
    sibling = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(sibling))

    child_hf_home = _run_probe(tmp_path, monkeypatch, "sibling")
    assert child_hf_home == str(file_root)

    child_env = _run_probe(tmp_path, monkeypatch, "sibling-env", return_env=True)
    assert child_env["HF_HOME"] == str(file_root)
    assert child_env["HF_HUB_CACHE"] == str(sibling)
