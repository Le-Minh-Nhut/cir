"""One lookup for per-model operator settings, with the documented precedence.

Three places can name a model's interpreter or an external asset's location:

1. the process environment (`export WORKBENCH_LIMN_PYTHON=/…/bin/python`),
2. an operator's local `workbench/config/workbench.env` file, and
3. the provisioning step's persisted assignment (`<artifacts>/environment_assignment.json`).

`workbench.env.example` documents the file, and `AUTOPILOT_GUIDE.md` documents the exported
variable, both saying the environment wins. Only (1) actually worked: `operator_config._read_env`
parsed the file, `resolve_config` stored the ten path/port keys and discarded every per-model value,
and every consumer read `os.environ` directly. A fixture containing only
`WORKBENCH_LIMN_PYTHON=/fixture/dedicated/bin/python` resolved to `None`.

This module is that missing lookup. It is deliberately small: one read, one source label for
diagnostics. It does NOT mutate `os.environ`, so a file setting cannot leak into a child process
through an unrelated name; a consumer that wants a child to see the value has to pass it, which is
how the autopilot already builds its `child_environment`.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

#: Names that are a *setting for this workbench*, as opposed to a variable the model's own code
#: reads for its own reasons. Derived from the registry declarations rather than maintained by hand:
#: hand-maintenance is what left `WORKBENCH_TME_PYTHON` and `WORKBENCH_PTHA_MTST_PYTHON` rejected by
#: the file parser while the registry declared them (models.yaml) and the consumers looked for them.
def _declared_names() -> frozenset[str]:
    names: set[str] = set()
    try:
        from workbench.backend.registry import load_registry

        for model in load_registry()["models"]:
            variable = (model.get("environment") or {}).get("python_env_var")
            if isinstance(variable, str) and variable:
                names.add(variable)
    except Exception:
        pass
    try:
        from workbench.backend.registry import load_preparation_contracts

        for contract in load_preparation_contracts().values():
            for asset in contract.get("external_assets", []):
                # Only the asset LOCATION override is a workbench setting. A `cache_root_env_var`
                # (TORCH_HOME, HF_HOME, XDG_CACHE_HOME) belongs to the library that reads it, so it
                # is neither advertised nor accepted as a local workbench.env setting.
                value = asset.get("location_env_var")
                if isinstance(value, str) and value:
                    names.add(value)
    except Exception:
        pass
    return frozenset(names)


@lru_cache(maxsize=1)
def declared_setting_names() -> frozenset[str]:
    """The registry-declared setting names, computed once per process."""
    return _declared_names()


@lru_cache(maxsize=1)
def file_settings() -> Mapping[str, str]:
    """The operator's `workbench.env` values, read once per process.

    Cached for the same reason `operator_config.resolve_config` reads it once: every consumer asks
    the same question, and re-reading per lookup would put file I/O on the interpreter-resolution
    path that the pipeline fingerprints. A process that changes the file mid-run is not a supported
    state; the API's own writers reload the module.
    """
    from workbench.backend.operator_config import read_env_file

    return dict(read_env_file())


def setting(name: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The effective value of *name*, or None when nothing sets it.

    Process environment first (the documented precedence), then the local file. An empty value is
    treated as unset in both, because `export WORKBENCH_LIMN_PYTHON=` and `WORKBENCH_LIMN_PYTHON=` in
    a file mean "no interpreter", not "an interpreter named empty string".
    """
    source = environ if environ is not None else os.environ
    configured = source.get(name)
    if configured is not None and configured.strip():
        return configured.strip()
    file_value = file_settings().get(name)
    if file_value is not None and file_value.strip():
        return file_value.strip()
    return None


def setting_source(name: str, environ: Mapping[str, str] | None = None) -> str:
    """Where `setting` would read this name from: `process_environment`, `workbench_env`, `unset`."""
    source = environ if environ is not None else os.environ
    configured = source.get(name)
    if configured is not None and configured.strip():
        return "process_environment"
    file_value = file_settings().get(name)
    if file_value is not None and file_value.strip():
        return "workbench_env"
    return "unset"


def resolve_path_setting(name: str, root: Path | None = None,
                         environ: Mapping[str, str] | None = None) -> Path | None:
    """A path-valued setting, resolved the way `operator_config._path` resolves file entries.

    Relative values are relative to the repository root, matching the documented meaning of a
    relative path in `workbench.env`.
    """
    value = setting(name, environ)
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    from workbench.backend.operator_config import _DEFAULT_REPO_ROOT

    return ((root or _DEFAULT_REPO_ROOT) / path).resolve()
