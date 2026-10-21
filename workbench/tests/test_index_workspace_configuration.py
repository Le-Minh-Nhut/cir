"""An isolated workspace must index ITS OWN results against ITS OWN receipts.

The defect this covers: `rebuild_index` accepts a `config` and threads it to the verifier receipts,
but the two `pipeline.run_validation_index` branches, the API rebuild endpoint and the CLI all
called it without one. The result files and the database came from the operator's configuration
while the receipts that PROMOTE a run came from the process default, so a workspace could be
indexed while the receipt for its run was read out of a different artifacts tree.

The tests below drive the real functions against two workspaces on disk and read the promotion out
of the resulting DuckDB database. The receipt is supplied through `verifier_verdicts` -- the same
seam `test_limn_query_export_publication.py` uses -- so the differing answer between the two roots
is what the assertion measures, not the parameter name.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from workbench.backend import index as index_module
from workbench.backend.operator_config import resolve_config
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import build_result_run, publish
from workbench.scripts import evaluate_models as evaluator
from workbench.scripts import pipeline
from workbench.scripts import rebuild_index as rebuild_cli

RUN_ID = "20260101T000000000000Z_limn_base_iter0_all_categories_fashioniq_val_split"


def _workspace(tmp_path: Path, monkeypatch, name: str, *, artifacts: Path | None = None) -> Path:
    """Point the operator configuration at a self-contained workspace and return its artifacts root."""
    artifacts = artifacts or tmp_path / name / "artifacts"
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / name / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / name / "home"))
    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / name / key.lower()))
    return artifacts


def _query(category: str, number: int, gallery_size: int, target_rank: int, top_depth: int) -> ObservedQuery:
    reference, target = f"{category}-ref-{number}", f"{category}-target-{number}"
    top = []
    for rank in range(1, top_depth + 1):
        image_id = target if rank == target_rank else f"distractor-{number}-{rank}"
        top.append((rank, image_id, round(1.0 - rank / (gallery_size + 10), 6)))
    return ObservedQuery(
        annotation_index=number, reference_id=reference, target_id=target,
        raw_captions=("change one", "detail two"), model_input_text="change one and detail two",
        target_rank=target_rank, top_results=tuple(top), category=category)


def _category(category: str, gallery_size: int, count: int = 3, target_rank: int = 2,
              top_depth: int = 200) -> ObservedCategory:
    """A category fixture at the ELIGIBLE depth the schema requires."""
    eligible = max(gallery_size - 1, 0)
    depth = min(top_depth, eligible)
    return ObservedCategory(
        category=category, gallery_size=gallery_size, eligible_count=eligible,
        queries=tuple(_query(category, index, gallery_size, target_rank, depth)
                      for index in range(count)))


def _publish_into(config, *, run_id: str = RUN_ID) -> tuple[Path, str]:
    """Publish a canonical per-query result into this workspace; return its path and digest."""
    results = config.WORKBENCH_RESULTS_ROOT
    results.mkdir(parents=True, exist_ok=True)
    result = build_result_run(
        [_category("dress", 300), _category("shirt", 350), _category("toptee", 400)],
        run_id=run_id, model_id="limn", method_name="LIMN",
        checkpoint_id="base_iter0_all_categories", protocol_id="fashioniq_val_split",
        top_k_saved=200, source_commit="a" * 40, checkpoint_sha256="b" * 64,
        command_digest="c" * 64, environment_digest="d" * 64, bundle_manifest_digest="e" * 64,
        instrumentation_parity="INSTRUMENTATION_PARITY_VERIFIED", data_kind="experiment")
    publish(result, results)
    matches = [path for path in results.rglob("*.json") if run_id in path.name]
    assert len(matches) == 1, matches
    return matches[0], evaluator.sha256_file(matches[0])


def _corroborate_aggregate(config, *, run_id: str = RUN_ID) -> Path:
    """Write an aggregate report plus the ledger entry that authenticates it."""
    reports = evaluator.reports_root(config)
    reports.mkdir(parents=True, exist_ok=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "run_id": run_id,
              "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split"}
    path = reports / f"{run_id}_aggregate.json"
    path.write_text(json.dumps(report))
    evaluator.append_evidence_entry(config, {
        "run_id": run_id, "report_digest": evaluator.sha256_file(path), "model_id": "limn",
        "checkpoint_id": "base_iter0_all_categories", "protocol_id": "fashioniq_val_split",
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
    return path


def _evidence_status(database: Path, run_id: str = RUN_ID) -> str:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?", [run_id]).fetchone()
    finally:
        connection.close()
    assert row is not None, f"{run_id} was not indexed at all in {database}"
    return row[0]


def _verdict_for(digest: str) -> dict:
    """The receipt shape `main._aggregate_reports` attaches, naming THIS artifact's bytes."""
    return {"certified": True, "is_research_evidence": True, "per_query_artifact_digest": digest,
            "model_id": "limn", "checkpoint_id": "base_iter0_all_categories"}


def _receipts_only_for(monkeypatch, config, verdicts: dict) -> None:
    """Serve `verdicts` when the caller passes THIS configuration, and nothing for any other.

    This is the seam the defect lived in: `rebuild_index(config=None)` asked the process default,
    while a caller that forwarded its configuration asked that configuration's own reports.
    """
    original = index_module.verifier_verdicts

    def only_for(theirs=None):
        if theirs is not None and theirs.WORKBENCH_ARTIFACTS_ROOT == config.WORKBENCH_ARTIFACTS_ROOT:
            return verdicts
        return original(theirs)

    monkeypatch.setattr(index_module, "verifier_verdicts", only_for)


def test_the_pipeline_promotes_from_the_workspace_it_was_given(tmp_path, monkeypatch, capsys):
    """`run_validation_index(cfg)` must read the receipts of `cfg`.

    The promoting receipt is served ONLY for the workspace under test. A stage that drops its
    configuration asks the process default instead, gets nothing, and leaves the run unverified --
    so this fails if the forwarding is removed.
    """
    custom = _workspace(tmp_path, monkeypatch, "custom")
    config = resolve_config()
    _, digest = _publish_into(config)
    _receipts_only_for(monkeypatch, config, {RUN_ID: _verdict_for(digest)})

    assert pipeline.run_validation_index(config) == 0, capsys.readouterr().err
    database = pipeline.database_path(config)
    assert database.is_relative_to(custom), "the index belongs to the configured workspace"
    assert _evidence_status(database) == "PER_QUERY_VERIFIED", \
        "the stage must consult the receipts of the configuration it was given"


def test_a_receipt_from_an_unrelated_workspace_cannot_promote_this_one(tmp_path, monkeypatch, capsys):
    """The inverse: the process default holds the receipt, the given workspace holds none.

    The unrelated root gets a corroborated report and a verdict that binds this artifact's exact
    bytes. Because the receipt lives in the OTHER tree, it must not promote this workspace's run:
    a report nobody in this root produced is not evidence about this root. An unrelated default
    workspace must also never be written to.
    """
    unrelated = tmp_path / "unrelated" / "artifacts"
    unrelated.mkdir(parents=True)
    custom = _workspace(tmp_path, monkeypatch, "custom")
    config = resolve_config()
    artifact_path, digest = _publish_into(config)

    # The unrelated workspace: a corroborated report and a receipt binding these very bytes.
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(unrelated))
    _workspace(tmp_path, monkeypatch, "unrelated", artifacts=unrelated)
    unrelated_config = resolve_config()
    assert unrelated_config.WORKBENCH_ARTIFACTS_ROOT != config.WORKBENCH_ARTIFACTS_ROOT
    _corroborate_aggregate(unrelated_config)
    _receipts_only_for(monkeypatch, unrelated_config, {RUN_ID: _verdict_for(digest)})

    # From here on the process default IS the unrelated workspace.
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(unrelated))
    assert pipeline.run_validation_index(config) == 0, capsys.readouterr().err

    database = pipeline.database_path(config)
    assert database.is_relative_to(custom)
    assert _evidence_status(database) == "PER_QUERY_UNVERIFIED", (
        "a receipt read from an unrelated artifacts root promoted this workspace's run; "
        f"the artifact indexed was {artifact_path}")
    assert not (unrelated / "workbench.duckdb").exists(), \
        "the custom rebuild must not write an index into the unrelated workspace"


def test_the_empty_results_branch_also_uses_the_given_workspace(tmp_path, monkeypatch, capsys):
    """An aggregate-only workspace has its own rebuild call, with the same root-mixing risk.

    No per-query run exists, so the stage must still write its declared database -- into the
    configured workspace -- and the unrelated default must stay untouched.
    """
    unrelated = tmp_path / "default-artifacts"
    unrelated.mkdir(parents=True)
    custom = _workspace(tmp_path, monkeypatch, "aggregate-only")
    config = resolve_config()
    _corroborate_aggregate(config)
    assert index_module.load_runs(config.WORKBENCH_RESULTS_ROOT) == []

    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(unrelated))
    assert pipeline.run_validation_index(config) == 0, capsys.readouterr().err

    database = pipeline.database_path(config)
    assert database.is_relative_to(custom) and database.is_file()
    assert not (unrelated / "workbench.duckdb").exists()
    assert index_module.load_runs(config.WORKBENCH_RESULTS_ROOT) == []


def test_the_cli_rebuild_forwards_its_configuration(tmp_path, monkeypatch, capsys):
    """`rebuild_index.py` resolved a configuration and then indexed without it."""
    custom = _workspace(tmp_path, monkeypatch, "cli")
    config = resolve_config()
    _, digest = _publish_into(config)
    _receipts_only_for(monkeypatch, config, {RUN_ID: _verdict_for(digest)})

    assert rebuild_cli.main(["--results-root", str(config.WORKBENCH_RESULTS_ROOT)]) == 0
    output = capsys.readouterr().out
    database = config.WORKBENCH_ARTIFACTS_ROOT / "workbench.duckdb"
    assert database.is_file() and database.is_relative_to(custom)
    assert output.strip().endswith(f"runs into {database}")
    assert _evidence_status(database) == "PER_QUERY_VERIFIED"
    assert not (custom / "results").joinpath("results").exists()


def test_explicit_path_overrides_still_win_over_the_configuration(tmp_path, monkeypatch):
    """Compatibility: a caller naming both roots keeps control of both."""
    custom = _workspace(tmp_path, monkeypatch, "configured")
    config = resolve_config()
    _, digest = _publish_into(config)
    _corroborate_aggregate(config)
    _receipts_only_for(monkeypatch, config, {RUN_ID: _verdict_for(digest)})

    elsewhere = tmp_path / "explicit"
    elsewhere.mkdir()
    explicit_database = elsewhere / "explicit.duckdb"

    assert index_module.rebuild_index(results_root=config.WORKBENCH_RESULTS_ROOT,
                                      database_path=explicit_database, config=config) >= 1
    assert explicit_database.is_file()
    assert not (custom / "workbench.duckdb").exists()
    assert _evidence_status(explicit_database) == "PER_QUERY_VERIFIED"


def test_rebuild_index_defaults_both_roots_from_the_supplied_configuration(tmp_path, monkeypatch):
    """With no explicit paths, the configuration decides results AND database.

    The process environment is then repointed at a DIFFERENT workspace, so resolving the defaults
    from the process would write the index somewhere else. Only a function that honours the
    configuration it was handed can pass.
    """
    custom = _workspace(tmp_path, monkeypatch, "defaulted")
    config = resolve_config()
    _, digest = _publish_into(config)
    _receipts_only_for(monkeypatch, config, {RUN_ID: _verdict_for(digest)})

    elsewhere = tmp_path / "elsewhere" / "artifacts"
    elsewhere.mkdir(parents=True)
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(elsewhere))
    monkeypatch.setenv("WORKBENCH_RESULTS_ROOT", str(elsewhere / "results"))
    assert resolve_config().WORKBENCH_ARTIFACTS_ROOT == elsewhere, "the process default moved"

    assert index_module.rebuild_index(config=config) >= 1
    database = config.WORKBENCH_ARTIFACTS_ROOT / "workbench.duckdb"
    assert database.is_file() and database.is_relative_to(custom), \
        "the supplied configuration, not the process default, must decide the defaulted roots"
    assert not (elsewhere / "workbench.duckdb").exists()
    assert _evidence_status(database) == "PER_QUERY_VERIFIED"


def test_the_api_rebuild_endpoint_forwards_its_configuration(tmp_path, monkeypatch, capsys):
    """`POST /api/index/rebuild` resolved a configuration and then indexed without it.

    The endpoint called `rebuild_index(...)` with both paths but no configuration, so the receipt
    that promotes each run was still read from the process default. The receipt here is served only
    for the workspace the endpoint's own configuration names.
    """
    from workbench.backend import main as api

    custom = _workspace(tmp_path, monkeypatch, "api")
    config = resolve_config()
    _, digest = _publish_into(config)
    _receipts_only_for(monkeypatch, config, {RUN_ID: _verdict_for(digest)})

    assert api.index() == {"runs_indexed": 1}, capsys.readouterr().err
    assert _evidence_status(config.WORKBENCH_ARTIFACTS_ROOT / "workbench.duckdb") == "PER_QUERY_VERIFIED"


def test_an_ordinary_deployment_rebuild_never_promotes_without_a_proof(tmp_path, monkeypatch, capsys):
    """The documented operator path, end to end, with the real verifier and no injection.

    A corroborated report with no completion proof is authenticated but not research evidence, so
    nothing may be promoted; the run is indexed and readable, which is what an operator rebuild
    has to deliver either way.
    """
    custom = _workspace(tmp_path, monkeypatch, "deployment")
    config = resolve_config()
    published, _ = _publish_into(config)
    _corroborate_aggregate(config)

    assert rebuild_cli.main([]) == 0
    database = config.WORKBENCH_ARTIFACTS_ROOT / "workbench.duckdb"
    assert database.is_relative_to(custom)
    assert _evidence_status(database) == "PER_QUERY_UNVERIFIED", \
        "without a validated completion proof the authoritative verifier cannot promote a run"

    # The indexed row is the artifact that was published, read back from the same workspace.
    assert [run.run.run_id for run in index_module.load_runs(config.WORKBENCH_RESULTS_ROOT)] == [RUN_ID]
    assert published.is_file()
    assert index_module.rebuild_index(config=config) >= 1
    assert _evidence_status(database) == "PER_QUERY_UNVERIFIED"
