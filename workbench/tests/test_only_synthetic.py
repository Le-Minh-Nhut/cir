"""Synthetic test fixtures are reachable ONLY through explicit test registration.

Production routing must never treat a synthetic fixture as a real reproduction
path; the synthetic adapter must not exist in the default registry.
"""
from __future__ import annotations

from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter, register_test_adapter


def test_synthetic_adapter_absent_from_production_registry():
    assert "synthetic_model" not in ADAPTERS


def test_synthetic_model_not_routed_by_metric_extraction():
    import workbench.backend.metrics_extraction as mx
    result = mx.extract_aggregate_metrics("synthetic_model", '{"average_recall": 1.0}', "", 0)
    assert result.extraction_status == "AGGREGATE_PARSER_UNAVAILABLE"


def test_register_test_adapter_is_explicit_and_reversible():
    class _TempAdapter(OfficialScriptAdapter):
        model_id = "synthetic_test_only"
        script = "src/evaluator.py"

        def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
            return ["python", str(source / self.script)]

    try:
        register_test_adapter(_TempAdapter)
        assert "synthetic_test_only" in ADAPTERS
    finally:
        ADAPTERS.pop("synthetic_test_only", None)
    assert "synthetic_test_only" not in ADAPTERS
