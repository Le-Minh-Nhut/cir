"""A31-03 recipe guards: the `manage_environment.py` flag rules the PC recipe documents.

`workbench/docs/CIRCUS_COMPOSITION_SPEC.md` is in `test_handoff_commands.py`'s `DOCS_EXEMPT`, so its
bash blocks are never parsed against the real argparse. These tests pin the parser facts the rewritten
recipe relies on, so a silent regression in the recipe (or the parser) fails here instead.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_MANAGER = REPO / "workbench" / "scripts" / "manage_environment.py"
_SPEC = REPO / "workbench" / "docs" / "CIRCUS_COMPOSITION_SPEC.md"


def _manage():
    spec = importlib.util.spec_from_file_location("_manage_environment_recipe_guard", _MANAGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_rejects_the_documented_usage_errors(capsys):
    """Each invalid pairing the recipe warns about must exit 2 at the parser boundary."""
    module = _manage()

    # `--list` and `--model` share a required mutually-exclusive group.
    with pytest.raises(SystemExit) as list_and_model:
        module.main(["--model", "encoder", "--list"])
    assert list_and_model.value.code == 2

    # `--create` without its explicit authorization is a usage error, raised before any work.
    with pytest.raises(SystemExit) as create_ungated:
        module.main(["--model", "encoder", "--create"])
    assert create_ungated.value.code == 2

    # The same mutually-exclusive group rejects `--list --json` when combined the errant way.
    with pytest.raises(SystemExit) as list_model_json:
        module.main(["--model", "encoder", "--list", "--json"])
    assert list_model_json.value.code == 2

    capsys.readouterr()  # swallow argparse usage text


def test_json_with_create_parses_and_provisions_nothing(monkeypatch):
    """`--json` + `--create` is a VALID parse that prints status JSON and creates no environment.

    The recipe documents this as an intentional no-op: the `if args.json:` branch returns before
    `create_env` is reachable. The parser must keep accepting the combination (no SystemExit), and
    the control flow must keep the json branch ahead of the create branch.
    """
    module = _manage()

    # Parser-level: the combination is not rejected. `--allow-env-install` is supplied so the
    # authorization guard (which fires before the json branch) does not mask the parse check.
    called: list[object] = []
    monkeypatch.setattr(module, "resolve_config", lambda: object())
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [{"model_id": "encoder"}]})
    monkeypatch.setattr(module, "status", lambda model, config: {"ready": False})
    monkeypatch.setattr(module, "create_env", lambda *a, **k: called.append(a) or True)

    result = module.main(["--model", "encoder", "--create", "--json", "--allow-env-install"])

    assert result in (0, 1)
    assert called == []  # the json branch returned before create_env

    # Source-level: the json branch precedes the create call in `main`.
    source = _MANAGER.read_text(encoding="utf-8")
    main_region = source[source.index("def main(") :]
    assert main_region.index("if args.json:") < main_region.index("create_env(")


def test_spec_recipe_never_documents_a_usage_error_pairing():
    """Text-level guard on the recipe: no manager line pairs `--model` with `--list`, or `--create`
    with `--json`."""
    lines = [line for line in _SPEC.read_text(encoding="utf-8").splitlines()
             if "manage_environment.py" in line]
    assert lines, "the recipe no longer mentions manage_environment.py"

    for line in lines:
        tokens = set(line.split())
        assert not {"--model", "--list"} <= tokens, line
        assert not ("--create" in tokens and "--json" in tokens), line
