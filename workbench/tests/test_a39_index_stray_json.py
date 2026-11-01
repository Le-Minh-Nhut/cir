from pathlib import Path

from workbench.backend.index import rebuild_index


def test_index_rebuild_skips_and_reports_stray_json(tmp_path: Path, caplog) -> None:
    results = tmp_path / "results"
    results.mkdir()
    stray = results / "summary.json"
    stray.write_text('{"model_id": "limn", "status": "COMPLETED"}', encoding="utf-8")
    database = tmp_path / "workbench.duckdb"

    count = rebuild_index(results_root=results, database_path=database)

    assert count == 0
    assert database.is_file()
    assert str(stray) in caplog.text
    assert "Skipping invalid result JSON" in caplog.text
