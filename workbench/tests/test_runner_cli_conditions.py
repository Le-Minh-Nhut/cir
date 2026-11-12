"""`--conditions` on the composition CLI: parsing, exit codes, and a plan that matches the run.

The defect this closes: `composition_eval.main` never passed `conditions=` to `run_composition`, so
the runner always used the protocol triple and `clvc_image_encoder_only` could not be requested at
all -- the role-resolution code in the runner would never see it. `_plan` also iterated the module
constant, so a planned variant run listed an output it would not write and omitted one it would.

CPU only. No GPU, no composition run: the plan path writes nothing (`--plan` asserts that in the
existing suite) and the parser tests never reach a release.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from workbench.composition.inputs import ABLATION_CONDITIONS, CONDITIONS

CLI = Path(__file__).resolve().parents[1] / "scripts" / "composition_eval.py"
IMAGE_ONLY = "clvc_image_encoder_only"

def _load_cli():
    spec = importlib.util.spec_from_file_location("composition_eval_conditions_test", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def test_the_flag_exists_and_documents_the_ablation_names(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args(["--help"])
    assert raised.value.code == 0
    # argparse wraps long help at the terminal width, so a condition name can be split across two
    # lines; compare with the wrapping removed rather than pretending it is not there.
    text = "".join(capsys.readouterr().out.split())
    assert "--conditions" in text
    for name in CONDITIONS + ABLATION_CONDITIONS:
        assert name in text, f"{name} must be named in --help, not only in code"

@pytest.mark.parametrize(("raw", "expected"), [
    (None, CONDITIONS),
    ("multimodal", ("multimodal",)),
    ("multimodal,text_only_black_image", ("multimodal", "text_only_black_image")),
    # order preserved, duplicates collapsed
    ("text_only_black_image, multimodal, text_only_black_image", ("text_only_black_image", "multimodal")),
    # the ablation name, alone and beside the protocol conditions
    (IMAGE_ONLY, (IMAGE_ONLY,)),
    (f"multimodal,text_only_black_image,{IMAGE_ONLY}",
     ("multimodal", "text_only_black_image", IMAGE_ONLY)),
    # an all-four request keeps all four, in the caller's order
    (f"multimodal,text_only_black_image,image_only_empty_text,{IMAGE_ONLY}",
     ("multimodal", "text_only_black_image", "image_only_empty_text", IMAGE_ONLY)),
])
def test_resolution_keeps_order_and_collapses_duplicates(raw, expected) -> None:
    cli = _load_cli()
    assert cli._resolve_conditions(raw) == expected

@pytest.mark.parametrize("raw", ["", " ", ",", " , "])
def test_an_empty_list_is_refused(raw: str) -> None:
    cli = _load_cli()
    with pytest.raises(ValueError, match="was empty"):
        cli._resolve_conditions(raw)

def test_an_empty_list_exits_2_without_touching_the_release(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    assert cli.main(["--conditions", ""]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "was empty" in payload["error"]

def test_an_unknown_name_exits_2_and_names_the_allowed_set(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    assert cli.main(["--conditions", "multimodal,image_only_empty_text_typo"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert "image_only_empty_text_typo" in payload["error"]
    for name in CONDITIONS + ABLATION_CONDITIONS:
        assert name in payload["error"]

def test_the_protocol_name_is_still_refused_when_it_is_not_a_condition(capsys: pytest.CaptureFixture[str]) -> None:
    """A name that is not a condition at all never becomes a silent no-op."""
    cli = _load_cli()
    assert cli.main(["--conditions", "mm"]) == 2
    assert "unknown condition(s): mm" in json.loads(capsys.readouterr().out)["error"]

def test_plan_lists_exactly_the_requested_conditions(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    """A plan that disagrees with the run it describes is a lie; this asserts they agree."""
    cli = _load_cli()
    cli_tests = _load_cli_tests()
    config, release, adapters, artifacts = cli_tests._fixture(tmp_path, ("flat",))

    for requested in (("multimodal", "text_only_black_image"),
                      ("multimodal", "text_only_black_image", IMAGE_ONLY)):
        rc, report, error = cli_tests._run(config, release, adapters, artifacts, monkeypatch,
                                           capsys, "--models", "flat", "--plan", "--subsets", "full",
                                           "--conditions", ",".join(requested))
        assert rc == 0, error
        assert report["conditions"] == list(requested)
        paths = report["expected_output_paths"]
        for condition in requested:
            assert any(path.endswith(f"/full/{condition}.jsonl") for path in paths), condition
        for absent in set(CONDITIONS) - set(requested):
            assert not any(path.endswith(f"/full/{absent}.jsonl") for path in paths), absent
        assert not artifacts.exists(), "a plan must not write anything"

def test_default_plan_is_the_protocol_triple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    cli_tests = _load_cli_tests()
    config, release, adapters, artifacts = cli_tests._fixture(tmp_path, ("flat",))
    rc, report, error = cli_tests._run(config, release, adapters, artifacts, monkeypatch, capsys,
                                       "--models", "flat", "--plan", "--subsets", "full")
    assert rc == 0, error
    assert report["conditions"] == list(CONDITIONS)
    for condition in CONDITIONS:
        assert any(path.endswith(f"/full/{condition}.jsonl") for path in report["expected_output_paths"])

def test_main_passes_the_resolved_list_to_the_runner() -> None:
    """The runner must receive `conditions=` derived from the flag, not its own default."""
    source = CLI.read_text(encoding="utf-8")
    assert "conditions=conditions" in source, "run_composition must be called with conditions="
    # and the plan must use the same variable, not the module constant
    assert "for condition in conditions:" in source
    assert "for condition in composition_data.CONDITIONS:" not in source

def _load_cli_tests():
    path = Path(__file__).resolve().parent / "test_circus_cli.py"
    spec = importlib.util.spec_from_file_location("circus_cli_fixture_source_conditions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
