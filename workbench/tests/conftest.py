"""Suite-wide isolation from the operator's machine state.

The suite must give the same result on a fresh clone as on this workstation. Two pieces of
per-machine state were deciding outcomes instead:

* ``workbench/config/workbench.env`` is untracked and read by
  ``operator_config.resolve_config()`` on every call that does not repoint it. A staged
  checkpoint under ``WORKBENCH_CHECKPOINT_ROOT`` therefore made an adapter that a fresh clone
  reports as *unready* report ``ready=True``, and an absolute ``WORKBENCH_ARTIFACTS_ROOT``
  moved a subprocess's ledger and reports out of the sandbox the test asserts on.
* ``CIR_*``/``WORKBENCH_*`` variables exported in the operator's shell outrank that file.

Both are cleared for every test here. A test that needs settings points ``_CONFIG_PATH`` at its
own file and clears ``file_settings``; the A31/A33/A34/``test_operator_config`` tests already do.

Subprocesses do not inherit ``_CONFIG_PATH`` (it is a Python attribute), so a test that spawns
one does pin the keys it asserts on into the child environment itself. This fixture only
guarantees that nothing leaks *in*, never that a test's own sandbox disappears.
"""
from __future__ import annotations

import os

import pytest

#: Prefixes the operator file and the operator's shell may carry. The file itself is handled by
#: repointing ``_CONFIG_PATH``; these are the exported forms that outrank it.
_OPERATOR_ENV_PREFIXES = ("CIR_", "WORKBENCH_")


@pytest.fixture(autouse=True)
def isolate_operator_state(monkeypatch, tmp_path_factory):
    """Run every test as if the checkout were fresh: no operator file, no exported settings.

    Only setup clears the caches. A teardown clear would run *before* ``monkeypatch`` undoes a
    test's own patch of ``file_settings``, so it would call ``cache_clear`` on that test's stand-in
    (``test_a36_assigned_identity`` replaces it with a plain lambda) and error during teardown.
    Clearing at setup is sufficient: every test starts from the empty cache it should.
    """
    from workbench.backend import model_settings, operator_config, runtime

    for name in [name for name in os.environ if name.startswith(_OPERATOR_ENV_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    # A path that never exists is the file's "not configured" state, which is what a fresh
    # clone has. Placing it under the session tmp dir keeps it from accidentally existing.
    monkeypatch.setattr(operator_config, "_CONFIG_PATH",
                        tmp_path_factory.mktemp("operator-config") / "workbench.env")
    model_settings.file_settings.cache_clear()
    runtime.forget_environment()
