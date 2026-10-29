"""CPU-only checks for TME cwd preparation and native vision precision boundary."""
from __future__ import annotations

import ast
import os
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from workbench.composition import model_tme


def test_worker_cwd_resolves_paths_before_entering_tme_source(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "checkout"
    (source_root / "src").mkdir(parents=True)
    (source_root / "weight").mkdir()
    (source_root / "weight" / "eva_vit_g.pth").touch()
    (source_root / "weight" / "blip2_pretrained.pth").touch()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "checkpoint.pth").touch()
    (tmp_path / "image.jpg").touch()
    monkeypatch.chdir(tmp_path)
    job = {
        "source_root": "checkout", "checkpoint_path": "checkpoint.pth",
        "dataset_root": "dataset", "image_paths": ["image.jpg"], "output_path": "output.npy",
    }

    resolved_source = model_tme._prepare_worker_cwd(job)

    assert resolved_source == source_root
    assert Path.cwd() == source_root
    assert all(Path(path).is_absolute() for path in (
        job["source_root"], job["checkpoint_path"], job["dataset_root"],
        *job["image_paths"], job["output_path"],
    ))
    assert Path("weight/eva_vit_g.pth").is_file()
    assert Path("weight/blip2_pretrained.pth").is_file()


def test_tme_worker_vision_autocast_boundary_source_structure() -> None:
    """Source-structure check pins autocast to TME's vision call without requiring GPU."""
    source = Path(model_tme.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    worker = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_worker_main")
    parents: dict[ast.AST, ast.AST] = {
        child: parent for parent in ast.walk(worker) for child in ast.iter_child_nodes(parent)
    }
    vision_call = next(node for node in ast.walk(worker)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "vit_encode")
    ancestor = parents[vision_call]
    while not isinstance(ancestor, ast.With):
        ancestor = parents[ancestor]
    assert ast.unparse(ancestor.items[0].context_expr) == "torch.cuda.amp.autocast()"

def test_tme_worker_loads_checkpoint_on_cpu_source_structure() -> None:
    """Source-structure check pins checkpoint unpickling to CPU without requiring GPU."""
    module = ast.parse(Path(model_tme.__file__).read_text(encoding="utf-8"))
    worker = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_worker_main")
    loads = [node for node in ast.walk(worker)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "load" and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "torch"]
    assert len(loads) == 1
    assert ast.literal_eval(next(keyword.value for keyword in loads[0].keywords
                                 if keyword.arg == "map_location")) == "cpu"

def test_worker_skips_empty_stdin_lines(tmp_path: Path, monkeypatch, capsys) -> None:
    source_root = tmp_path / "source"
    (source_root / "src").mkdir(parents=True)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    job_path = tmp_path / "job.json"
    output_path = tmp_path / "output.npy"
    job_path.write_text(json.dumps({
        "source_root": str(source_root), "checkpoint_path": str(tmp_path / "checkpoint"),
        "dataset_root": str(dataset_root), "image_paths": [], "output_path": str(output_path),
        "operation": "targets", "batch_size": 1, "checkpoint": {"checkpoint_id": "fashioniq_0"},
    }), encoding="utf-8")

    network = SimpleNamespace(
        load_state_dict=lambda *_args, **_kwargs: SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        eval=lambda: None,
    )
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    torch.device = lambda value: SimpleNamespace(type=value)
    torch.no_grad = contextlib.nullcontext
    torch.load = lambda *_args, **_kwargs: {}
    lavis = types.ModuleType("lavis")
    lavis.__path__ = []
    lavis_models = types.ModuleType("lavis.models")
    lavis_models.load_model_and_preprocess = lambda **_kwargs: (network, None, {"eval": str})
    utility = types.ModuleType("utility")
    utility.targetpad_transform = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "lavis", lavis)
    monkeypatch.setitem(sys.modules, "lavis.models", lavis_models)
    monkeypatch.setitem(sys.modules, "utility", utility)
    from workbench.backend import registry
    from workbench.replay import tme

    monkeypatch.setattr(registry, "checkpoint_is_present", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tme, "source_root_for", lambda _checkout: source_root)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n{\"stop\": true}\n"))
    original_cwd = Path.cwd()
    try:
        assert model_tme._worker_main(["--worker-job", str(job_path)]) == 0
    finally:
        os.chdir(original_cwd)
    assert np.load(output_path, allow_pickle=False).shape == (0, 32, 256)
    assert capsys.readouterr().out.count("CIRCUS_WORKER:") == 1
