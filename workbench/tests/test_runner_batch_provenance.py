"""End-to-end (CPU fixture) proof that a summary states its batch/partition provenance.

Reuses the SYNTHETIC CLI fixture from `test_circus_cli.py` by import, not by copying: this test is
about what lands in the summary, not about building a release. No GPU, no checkpoint, no model
interpreter — the adapters are NumPy fixtures.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from workbench.composition import runner

CLI_TESTS = Path(__file__).resolve().parent / "test_circus_cli.py"

def _cli_tests():
    spec = importlib.util.spec_from_file_location("circus_cli_fixture_source", CLI_TESTS)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_summary_records_batch_and_partition_provenance(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    cli = _cli_tests()
    config, release, adapters, artifacts = cli._fixture(tmp_path, ("flat",))
    rc, report, error = cli._run(config, release, adapters, artifacts, monkeypatch, capsys,
                                 "--models", "flat", "--batch-size", "2")
    assert rc == 0, error
    summary = json.loads(Path(report["models"][0]["summaries"]["full"]).read_text(encoding="utf-8"))

    block = summary["batch_partition"]
    # `--batch-size` sets the ADAPTER's forward batch (env), not the runner's scoring group: the
    # two are different knobs and both are recorded.
    assert block["runner_batch_size"] == 32
    assert block["adapter_batch_size"] == 2
    assert block["call_window"] == runner.ADAPTER_CALL_WINDOW
    assert "ascending" in block["query_order"]
    assert "feature_batch_fallback" in block["partition_policy"]
    conditions = block["conditions"]
    assert set(conditions) == set(summary["conditions"])
    for record in conditions.values():
        # Every condition either ran the canonical/mixed partition or says why it never encoded.
        assert record["partition"] in {runner.FEATURE_PARTITION_CANONICAL,
                                       runner.FEATURE_PARTITION_MIXED, "not_encoded"}
        assert record["adapter_batch_size"] in {2, None}
    assert conditions["multimodal"]["partition"] == runner.FEATURE_PARTITION_CANONICAL
    assert conditions["multimodal"]["fallback_used"] is False
    assert conditions["multimodal"]["fallback_qids"] == []
    # Scoring fallback keeps its own, unchanged meaning.
    for status in summary["condition_status"].values():
        assert status.get("fallback_used") is False
        assert "feature_batch_fallback" in status

def test_summary_partition_names_the_affected_qids(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """Force the ENCODE fallback through the real runner and read the record it publishes."""
    cli = _cli_tests()
    config, release, adapters, artifacts = cli._fixture(tmp_path, ("flat",))
    original = runner._encode_queries

    def flaky(*args, **kwargs):
        adapter = args[0]

        class Strict:
            def __init__(self, inner):
                self.inner = inner

            def encode_queries(self, image_paths, texts, categories):
                if len(image_paths) > 1:
                    raise RuntimeError("fixture batch rejection")
                return self.inner.encode_queries(image_paths, texts, categories)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        strict = Strict(adapter)
        return original(strict, *args[1:], **kwargs)

    monkeypatch.setattr(runner, "_encode_queries", flaky)
    rc, report, error = cli._run(config, release, adapters, artifacts, monkeypatch, capsys,
                                 "--models", "flat", "--batch-size", "2")
    assert rc == 0, error
    summary = json.loads(Path(report["models"][0]["summaries"]["full"]).read_text(encoding="utf-8"))
    multimodal = summary["condition_status"]["multimodal"]["feature_batch_fallback"]
    assert multimodal["partition"] == runner.FEATURE_PARTITION_MIXED
    assert multimodal["fallback_used"] is True
    assert multimodal["fallback_qids"], "the fallback rows must be named, not just counted"
    assert "fixture batch rejection" in multimodal["fallback_reason"]
    assert multimodal["adapter_batch_size"] == 2
    assert multimodal["runner_batch_size"] == 32
    assert summary["batch_partition"]["conditions"]["multimodal"]["fallback_qids"] == \
        multimodal["fallback_qids"]
