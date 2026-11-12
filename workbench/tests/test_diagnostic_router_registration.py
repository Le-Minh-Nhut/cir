"""The CLVC diagnostic routes must be REGISTERED on the shipped application.

`limn_diagnostic.py` declares them on an `APIRouter`; a router nobody includes is dead code that
looks like a feature. The live backend answered 404 for `/api/composition/diagnostics/clvc-dual-scope`
while LIMN's literal route worked, and nothing in the test suite noticed, because every existing test
imports `load_diagnostic` directly instead of going through the app.

FastAPI >= 0.122 nests an included router lazily: `app.routes` holds one `_IncludedRouter` entry, and
the concrete paths appear in the EFFECTIVE route table (`app.openapi()`), which is what dispatch and
the OpenAPI document use. Asserting on `app.routes` alone would pass on a build that never dispatches
the route, so both facts are asserted here.

These paths are literal on purpose. A parameterised `/{model_id}-dual-scope` would shadow LIMN's
literals depending on registration order; this test pins the four literal paths so that "fix" fails.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CLVC_PATHS = ["/api/composition/diagnostics/clvc-dual-scope",
              "/api/composition/diagnostics/clvc-dual-scope/candidates"]
LIMN_PATHS = ["/api/composition/diagnostics/limn-dual-scope",
              "/api/composition/diagnostics/limn-dual-scope/candidates"]


def test_all_four_diagnostic_paths_are_served_by_the_real_app() -> None:
    from workbench.backend.main import app

    documented = {path for path in app.openapi()["paths"] if "diagnostics" in path}
    assert documented == set(CLVC_PATHS) | set(LIMN_PATHS), sorted(documented)


def test_the_app_is_the_shipped_module_and_its_router_is_included() -> None:
    """A route registered on a *different* app object is the failure this pins.
    """
    from workbench.backend import main

    assert Path(main.__file__).resolve() == REPO / "workbench" / "backend" / "main.py"
    assert any(type(route).__name__ == "_IncludedRouter" for route in main.app.routes), (
        "no included router on the app: `app.include_router(diagnostic_router)` is missing")


def test_no_parameterised_dual_scope_route_exists() -> None:
    """`{model_id}-dual-scope` would resolve by registration order, shadowing LIMN.
    """
    from workbench.backend.limn_diagnostic import diagnostic_router

    paths = [route.path for route in diagnostic_router.routes]
    assert paths == CLVC_PATHS, paths
    assert not any("{" in path for path in paths), "a parameterised diagnostic path was declared"
