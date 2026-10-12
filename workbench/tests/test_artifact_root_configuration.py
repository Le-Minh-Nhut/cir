from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from workbench.backend import index, runtime, storage
from workbench.backend.operator_config import resolve_config
from workbench.scripts import evaluate_models as em
from workbench.scripts import pipeline, rebuild_index


def test_configured_artifact_root_contains_evaluation_and_index(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "code-checkout"
    artifacts = tmp_path / "operator-artifacts"
    repo.mkdir()
    monkeypatch.setattr("workbench.backend.operator_config._CONFIG_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("CIR_REPO_ROOT", str(repo))
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "test-ledger-key")

    config = resolve_config()
    assert config.CIR_REPO_ROOT == repo
    assert config.WORKBENCH_ARTIFACTS_ROOT == artifacts

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "import json\n"
        "print('evaluating')\n"
        "print(json.dumps({"
        "'average_recall_at10': 55.0, 'average_recall_at50': 75.0, 'average_recall': 65.0,"
        "'dress_recall_at10': 60.0, 'dress_recall_at50': 80.0,"
        "'shirt_recall_at10': 55.0, 'shirt_recall_at50': 75.0,"
        "'toptee_recall_at10': 50.0, 'toptee_recall_at50': 70.0}))\n",
        encoding="utf-8",
    )
    plan = em.EvaluationPlan(
        model_id="csmcir",
        checkpoint_id="ckpt_v1",
        protocol="fashioniq_original_split",
        dataset_root=tmp_path / "dataset",
        source=tmp_path / "source",
        checkpoint=checkpoint,
        cwd=tmp_path,
        command=[sys.executable, str(evaluator)],
        pin="a" * 40,
        actual_source_commit="a" * 40,
    )
    run_id = "configured-artifact-root"

    assert em.execute(plan, config, run_id=run_id) == 0

    run_directory = em.log_dir(config, plan, em.datetime.now(em.UTC), run_id)
    ledger = em.evidence_ledger_path(config)
    proof = em.completion_proof_path(config, plan, run_id)
    assert run_directory.is_dir()
    assert ledger.is_file()
    assert proof.is_file()
    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert Path(proof_payload["run_directory"]) == run_directory
    assert run_directory.is_relative_to(artifacts)
    assert ledger.is_relative_to(artifacts)
    assert rebuild_index.main(["--results-root", str(config.WORKBENCH_RESULTS_ROOT)]) == 0
    capsys.readouterr()
    index_path = artifacts / "workbench.duckdb"
    assert index_path.is_file()
    assert index_path.is_relative_to(artifacts)
    assert storage._database_path(None) == index_path
    assert storage.list_runs() == []

    assert runtime.gpu_capability_directory(config) == artifacts / "gpu_capabilities"
    assert pipeline.database_path(config) == index.configured_database_path(config)
    assert pipeline.stage_receipt_path(config, pipeline.Stage("evaluation")).is_relative_to(artifacts)

    second_artifacts = tmp_path / "second-operator-artifacts"
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(second_artifacts))
    importlib.reload(index)
    importlib.reload(storage)
    assert index.rebuild_index() == 0
    second_index = tmp_path / "second-operator-artifacts" / "workbench.duckdb"
    assert second_index.is_file()
    assert storage._database_path(None) == second_index
    assert storage.list_runs() == []
    assert index.DATABASE_PATH == second_index

    assert not (repo / "workbench" / "artifacts").exists()
