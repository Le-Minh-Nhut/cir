"""CPU-only end-to-end contracts for the opt-in CIRCUS CLI."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from workbench.composition import data as composition_data
from workbench.composition import runner
from workbench.composition.metrics import ndcg_full


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_cli():
    spec = importlib.util.spec_from_file_location("composition_eval_test", SCRIPTS / "composition_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, model_ids: tuple[str, ...] = ("fixture3d",)) -> tuple[Path, Path, Path, Path]:
    release = tmp_path / "release"
    (release / "shortcut_audit").mkdir(parents=True)
    (release / "shortcut_free_subsets/fashioniq/shortcut_free").mkdir(parents=True)
    for part in ("validated_solved", "validated_unsolved"):
        (release / "final_dataset/query_jsonl/fashioniq" / part).mkdir(parents=True)

    candidates = [{"did": f"g{index}", "img_path": f"gallery/img{index}.jpg"}
                  for index in range(5)]
    queries = [
        {"qid": f"q{index}", "query_txt": f"p={index + 1};r={index + 1}",
         "query_img_path": "g4" if index == 2 else "g0",
         "pos_cand_list": [f"g{index + 1}"]}
        for index in range(4)
    ]

    def jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    jsonl(release / "mbeir_fashioniq_task7_cand_pool.jsonl", candidates)
    jsonl(release / "mbeir_fashioniq_task7_test.jsonl", queries)
    jsonl(release / "shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl",
          queries[:3])
    jsonl(release / "final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl",
          queries[:2])
    jsonl(release / "final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl",
          queries[2:3])
    audit = release / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"
    audit.write_text(json.dumps({"gallery_paths": [
        f"<path_to_M-BEIR>/gallery/img{index}.jpg" for index in range(5)]}), encoding="utf-8")

    fashioniq = tmp_path / "FashionIQ"
    (fashioniq / "images").mkdir(parents=True)
    (fashioniq / "captions").mkdir()
    (fashioniq / "image_splits").mkdir()
    for index in range(5):
        Image.new("RGB", (3, 2), (index, index, index)).save(fashioniq / "images" / f"img{index}.jpg")
    for category in ("dress", "shirt", "toptee"):
        refs = ["img0.jpg", "img1.jpg", "img3.jpg"] if category == "dress" else []
        (fashioniq / "image_splits" / f"split.{category}.val.json").write_text(json.dumps(refs), encoding="utf-8")
        (fashioniq / "captions" / f"cap.{category}.test.json").write_text("[]", encoding="utf-8")

    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    for model_id in model_ids:
        module_name = f"synthetic_{model_id}"
        (adapter_root / f"{module_name}.py").write_text(
            "from __future__ import annotations\n"
            "import json, os, re\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "from workbench.composition.adapters import CompositionAdapter, AdapterWorkerError\n"
            f"MODEL_ID = {model_id!r}\n"
            "def record(kind, shape, values=None):\n"
            "    path = os.environ.get('CIRCUS_TEST_LOG')\n"
            "    if path:\n"
            "        entry = {'model': MODEL_ID, 'kind': kind, 'shape': list(shape)}\n"
            "        if values is not None: entry['values'] = values\n"
            "        with open(path, 'a', encoding='utf-8') as stream:\n"
            "            stream.write(json.dumps(entry) + '\\n')\n"
            "class Synthetic(CompositionAdapter):\n"
            "    model_id = MODEL_ID\n"
            "    category_scopes = ('dress',) if MODEL_ID == 'scoped' else (None,)\n"
            "    @classmethod\n"
            "    def status(cls, config=None):\n"
            "        from workbench.composition.adapters import AdapterStatus\n"
            "        return AdapterStatus(cls.model_id, cls.model_id != 'notready',\n"
            "            None if cls.model_id != 'notready' else 'synthetic adapter deliberately blocked',\n"
            "            ('fixture',), {}, 'synthetic test status')\n"
            "    def __init__(self, config, checkpoint_id=None):\n"
            "        super().__init__(config, checkpoint_id)\n"
            "        record('checkpoint', (checkpoint_id,))\n"
            "    def mode_supported(self, condition): return True, None\n"
            "    def encode_targets(self, image_paths, categories):\n"
            "        record('targets', (len(image_paths),))\n"
            "        if MODEL_ID == 'flat':\n"
            "            values = np.zeros((len(image_paths), 8192), dtype=np.float32)\n"
            "            values[:, 0] = [int(Path(path).stem.removeprefix('img')) for path in image_paths]\n"
            "            return values\n"
            "        values = np.zeros((len(image_paths), 32, 256), dtype=np.float32)\n"
            "        values[:, 0, 0] = [int(Path(path).stem.removeprefix('img')) for path in image_paths]\n"
            "        return values\n"
            "    def encode_queries(self, images, texts, categories):\n"
            "        record('queries', (len(images),), texts)\n"
            "        if MODEL_ID in {'workerfail', 'workersuccess'}:\n"
            "            worker_dir = Path(self.work_root) / MODEL_ID / 'worker-test'\n"
            "            worker_dir.mkdir(parents=True, exist_ok=True)\n"
            "            (worker_dir / 'worker.stderr.log').write_text('synthetic worker traceback\\n')\n"
            "        if MODEL_ID == 'workerfail': raise AdapterWorkerError('synthetic adapter worker failure')\n"
            "        fail_once = int(os.environ.get('CIRCUS_TEST_FAIL_ONCE', '0'))\n"
            "        counter = Path(os.environ['CIRCUS_TEST_LOG']).with_suffix('.fail-count') if os.environ.get('CIRCUS_TEST_LOG') else None\n"
            "        if fail_once and counter:\n"
            "            calls = int(counter.read_text() if counter.exists() else '0')\n"
            "            counter.write_text(str(calls + 1))\n"
            "            if calls < fail_once: raise RuntimeError('synthetic transient encoding failure')\n"
            "        if MODEL_ID in {'fail', 'blocked'}: raise RuntimeError('synthetic encoding failure')\n"
            "        width = 8192 if MODEL_ID == 'flat' else 256\n"
            "        result = np.zeros((len(images), width), dtype=np.float32)\n"
            "        for index, text in enumerate(texts):\n"
            "            match = re.fullmatch(r'p=(\\d+);r=(\\d+)', text)\n"
            "            if not text:\n"
            "                result[index, :2] = [1, 1]\n"
            "                continue\n"
            "            result[index, :2] = [int(match.group(1)), int(match.group(2))]\n"
            "        return result\n"
            "    def score(self, queries, targets, categories=None):\n"
            "        candidates = targets[:, 0, 0] if targets.ndim == 3 else targets[:, 0]\n"
            "        scores = np.zeros((len(queries), len(candidates)), dtype=np.float32)\n"
            "        for row, query in enumerate(queries):\n"
            "            positive, rank = int(query[0]), int(query[1])\n"
            "            positions = np.flatnonzero(candidates == positive)\n"
            "            assert len(positions) == 1\n"
            "            scores[row, positions[0]] = 10\n"
            "            competitors = [index for index in range(len(candidates)) if index != positions[0]]\n"
            "            scores[row, competitors[:rank - 1]] = 11\n"
            "        return scores\n"
            "ADAPTER = Synthetic\n", encoding="utf-8")

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    models = []
    for model_id in model_ids:
        directory = checkpoints / model_id
        directory.mkdir()
        (directory / "fixture.pt").write_bytes(b"synthetic checkpoint")
        models.append({
            "model_id": model_id,
            "composition_adapter": f"synthetic_{model_id}",
            "native_protocol": "fashioniq_original_split",
            "supported_protocols": ["fashioniq_original_split"],
            "literature_split_label": "original",
            "source_available": False,
            "checkpoint_variants": [{"checkpoint_id": "fixture", "filename": "fixture.pt",
                                      "evaluation_noise_pct": 0, "checkpoint_training_noise_pct": None,
                                      "checkpoint_training_dataset": "FashionIQ",
                                      "checkpoint_mapping_status": "VERIFIED_METADATA"}],
            "environment": {"python_env_var": "WORKBENCH_TEST_MODEL_PYTHON", "python": "3.13",
                            "packages": [], "cuda": "UNKNOWN"},
        })
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({
        "schema_version": 1,
        "protocols": {"fashioniq_original_split": {"label": "test", "literature_split_label": "original"}},
        "models": models,
    }), encoding="utf-8")

    artifacts = tmp_path / "artifacts"
    config = tmp_path / "autopilot.yaml"
    config.write_text(yaml.safe_dump({
        "schema_version": 1,
        "selected_models": list(model_ids),
        "paths": {"registry": str(registry), "dataset_root": str(fashioniq),
                  "artifacts": str(artifacts), "checkpoints": str(checkpoints),
                  "results": str(tmp_path / "results"), "third_party": str(tmp_path / "third_party")},
        "budgets": {"max_download_bytes": 0, "max_models": max(1, len(model_ids)), "top_k": 10},
        "checkpoint_policy": "include_flagged_conditions",
        "permissions": {name: False for name in (
            "allow_network", "allow_large_downloads", "allow_env_install", "allow_preparation",
            "allow_gpu_eval", "allow_gpu_preparation", "allow_index_write")},
    }), encoding="utf-8")
    return config, release, adapter_root, artifacts




def _run(config: Path, release: Path, adapters: Path, artifacts: Path, monkeypatch: pytest.MonkeyPatch,
         capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, dict | None, str]:
    # These adapters are NumPy fixtures, not model interpreters. Importing the host's Torch in
    # their provenance probe can time out under parallel load and change an otherwise identical
    # resume key. Real interpreter identity and real worker launches have separate tests.
    monkeypatch.setattr(runner, "model_interpreter_identity", lambda *_args, **_kwargs: {
        "path": str(Path(sys.executable).absolute()),
        "version": f"Python {sys.version.split()[0]}; synthetic NumPy fixture",
        "source": "process_environment", "verified": False,
        "verification_status": "SYNTHETIC_FIXTURE",
    })
    monkeypatch.setattr(runner, "_gpu_identity", lambda: None)
    monkeypatch.syspath_prepend(str(adapters))
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(config.parent / "registry.yaml"))
    monkeypatch.setenv("WORKBENCH_CIRCUS_RELEASE_ROOT", str(release))
    monkeypatch.setenv("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE", "1")
    monkeypatch.setenv("WORKBENCH_TEST_MODEL_PYTHON", sys.executable)
    monkeypatch.setenv("CIRCUS_TEST_LOG", str(config.parent / "encodes.jsonl"))
    from workbench.scripts import composition_eval
    rc = composition_eval.main(["--config", str(config), *args])
    captured = capsys.readouterr()
    try:
        return rc, json.loads(captured.out), captured.err
    except json.JSONDecodeError:
        return rc, None, captured.err + captured.out


def _log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_cli_3d_full_gallery_resume_and_gallery_invalidation(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    gallery_shapes = []
    query_shapes = []
    projections = []
    encode_gallery = runner._encode_gallery
    encode_queries = runner._encode_queries
    disk_projection = runner._disk_projection

    def capture_gallery(*args, **kwargs):
        mapped, projection = encode_gallery(*args, **kwargs)
        gallery_shapes.append(mapped.shape)
        return mapped, projection

    def capture_queries(*args, **kwargs):
        mapped, *rest = encode_queries(*args, **kwargs)
        query_shapes.append(mapped.shape)
        return (mapped, *rest)

    def capture_projection(*args, **kwargs):
        result = disk_projection(*args, **kwargs)
        projections.append((args[1], tuple(args[2]), args[4], args[5], result["projected_bytes"]))
        return result

    monkeypatch.setattr(runner, "_encode_gallery", capture_gallery)
    monkeypatch.setattr(runner, "_encode_queries", capture_queries)
    monkeypatch.setattr(runner, "_disk_projection", capture_projection)
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d")
    assert rc == 0, error
    row = report["models"][0]
    summary_path = Path(row["summaries"]["full"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["is_synthetic"] is True
    assert summary["synthetic"] is True
    assert summary["gallery_count"] == 5
    assert summary["runtime"]["model_interpreter"]["path"] == str(Path(sys.executable).absolute())
    assert "Python " in summary["runtime"]["model_interpreter"]["version"]
    full_result_path = next(path for path in (artifacts / "composition/results").rglob("full/multimodal.jsonl")
                            if path.parent.parent.name == row["attempt"])
    assert "SYNTHETIC" in full_result_path.parts
    full_rows = [json.loads(line) for line in full_result_path.read_text().splitlines()]
    assert [row["rank"] for row in full_rows] == [1, 2, 3, 4]
    assert all(row["synthetic"] is True for row in full_rows)
    assert summary_path.parent.name.startswith("SYNTHETIC_")
    assert summary["checkpoint_labels"] == {
        "checkpoint_training_noise_pct": None, "evaluation_noise_pct": 0,
        "checkpoint_training_dataset": "FashionIQ", "checkpoint_transfer": None,
        "checkpoint_mapping_status": "VERIFIED_METADATA", "reproduction_status": None,
        "labels_source": "registry"}
    assert runner._result_outputs(artifacts, "circus_fashioniq_release_v1", "fixture3d", "fixture",
                                  row["attempt"], "full", "multimodal", True) == full_result_path
    assert gallery_shapes == [(5, 32, 256)]
    assert query_shapes == [(4, 256)] * 3
    assert projections == [(5, (32, 256), 4, 3, 5 * 32 * 256 * 4 + 4 * 256 * 4 * 3)]

    log_path = tmp_path / "encodes.jsonl"
    before = [entry for entry in _log(log_path) if entry["kind"] != "checkpoint"]
    rank_file = next((artifacts / "composition/runs").rglob("ranks/fixture3d_fixture.json"))
    rank_digest, rank_mtime = rank_file.read_bytes(), rank_file.stat().st_mtime_ns
    rc, resumed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "fixture3d", "--resume")
    assert rc == 0, error
    assert resumed["models"][0]["status"] == "REUSED"
    assert resumed["models"][0]["attempt"] == row["attempt"]
    assert resumed["models"][0]["summaries"]["full"] == str(summary_path)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["execution_status"] == "REUSED"
    assert [entry for entry in _log(log_path) if entry["kind"] != "checkpoint"] == before
    assert (rank_file.read_bytes(), rank_file.stat().st_mtime_ns) == (rank_digest, rank_mtime)
    module_file = adapters / "synthetic_fixture3d.py"
    module_file.write_text(module_file.read_text(encoding="utf-8") + "\n# changed adapter worker source\n",
                           encoding="utf-8")
    rc, changed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "fixture3d", "--resume")
    assert rc == 0, error
    assert changed["models"][0]["status"] == "COMPLETED"
    changed_count = len(_log(log_path))
    assert changed_count > len(before)


    candidate_path = release / "mbeir_fashioniq_task7_cand_pool.jsonl"
    candidates = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    candidates.append({"did": "g5", "img_path": "gallery/img5.jpg"})
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8")
    audit_path = release / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["gallery_paths"].append("<path_to_M-BEIR>/gallery/img5.jpg")
    Image.new("RGB", (3, 2), (5, 5, 5)).save(Path(config.parent) / "FashionIQ/images/img5.jpg")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    rc, recomputed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                 "--models", "fixture3d", "--resume")
    assert rc == 0, error
    assert recomputed["models"][0]["status"] == "COMPLETED"
    assert len(_log(log_path)) > changed_count
    assert recomputed["models"][0]["gallery_count"] == 6


def test_cli_2d_gallery_memmap_and_weighted_subsets(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("flat",))
    gallery_shapes = []
    query_shapes = []
    encode_gallery, encode_queries = runner._encode_gallery, runner._encode_queries

    def capture_gallery(*args, **kwargs):
        mapped, info = encode_gallery(*args, **kwargs)
        gallery_shapes.append(mapped.shape)
        return mapped, info

    def capture_queries(*args, **kwargs):
        mapped, *rest = encode_queries(*args, **kwargs)
        query_shapes.append(mapped.shape)
        return (mapped, *rest)

    monkeypatch.setattr(runner, "_encode_gallery", capture_gallery)
    monkeypatch.setattr(runner, "_encode_queries", capture_queries)
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "flat")
    assert rc == 0, error
    assert gallery_shapes == [(5, 8192)]
    assert query_shapes == [(4, 8192)] * 3
    summaries = report["models"][0]["summaries"]
    rows = {}
    for subset in ("full", "sf", "validated"):
        result_path = next(path for path in (artifacts / "composition/results").rglob(
            f"{subset}/multimodal.jsonl") if path.parent.parent.name == report["models"][0]["attempt"])
        rows[subset] = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert [len(rows[name]) for name in ("full", "sf", "validated")] == [4, 3, 3]
    assert len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] == "queries"]) == 3
    subset_summary = json.loads(Path(summaries["validated"]).read_text(encoding="utf-8"))
    ndcgs = [ndcg_full(row["rank"]) for row in rows["validated"]]
    assert subset_summary["metrics"]["multimodal"]["ndcg_full"] == pytest.approx(sum(ndcgs) / 3)
    solved = rows["validated"][:2]
    unsolved = rows["validated"][2:]
    solved_mean = sum(ndcg_full(row["rank"]) for row in solved) / 2
    unsolved_mean = sum(ndcg_full(row["rank"]) for row in unsolved)
    assert subset_summary["metrics"]["multimodal"]["ndcg_full"] == pytest.approx((2 * solved_mean + unsolved_mean) / 3)
    assert subset_summary["metrics"]["multimodal"]["ndcg_full"] != pytest.approx((solved_mean + unsolved_mean) / 2)

def test_cli_subset_filter_preserves_gallery_and_resume_fingerprint(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    rc, validated, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                "--models", "fixture3d", "--subsets", "validated")
    assert rc == 0, error
    model = validated["models"][0]
    summary = json.loads(Path(model["summaries"]["validated"]).read_text(encoding="utf-8"))
    assert model["gallery_count"] == summary["gallery_count"] == 5
    assert summary["query_count"] == 3
    query_calls = [entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] == "queries"]
    assert len(query_calls) == 3
    assert {text for entry in query_calls for text in entry["values"] if text} == {
        "p=1;r=1", "p=2;r=2", "p=3;r=3"}

    rc, full, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                           "--models", "fixture3d", "--subsets", "full,sf,validated", "--resume")
    assert rc == 0, error
    full_summary = json.loads(Path(full["models"][0]["summaries"]["full"]).read_text(encoding="utf-8"))
    assert full["models"][0]["status"] == "COMPLETED"
    assert full["models"][0]["run_id"] != model["run_id"]
    assert full_summary["resume_key"] != summary["resume_key"]
    query_calls = [entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] == "queries"]
    assert len(query_calls) == 6



def test_cli_unsupported_and_selected_cohort_exit_codes(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, _artifacts = _fixture(tmp_path / "unsupported")
    registry_path = config.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["models"][0].pop("composition_adapter")
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    rc, unsupported, error = _run(config, release, adapters, _artifacts, monkeypatch, capsys,
                                  "--models", "fixture3d")
    assert rc == 1, error
    assert unsupported["models"][0]["status"] == "UNSUPPORTED"

    config, release, adapters, artifacts = _fixture(tmp_path / "cohort-blocked", ("notready",))
    rc, blocked, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "selected")
    assert rc == 1, error
    assert blocked["models"]
    assert all(row["status"] == "SELECTED_BUT_BLOCKED" for row in blocked["models"])

    config, release, adapters, artifacts = _fixture(tmp_path / "cohort-mixed", ("notready", "fixture3d"))
    rc, mixed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "selected")
    assert rc == 0, error
    assert {row["status"] for row in mixed["models"]} == {"SELECTED_BUT_BLOCKED", "COMPLETED"}


def test_selected_cohort_resume_exits_successfully(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    rc, first, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "selected")
    assert rc == 0, error
    before = [entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"]
    rc, resumed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "selected", "--resume")
    assert rc == 0, error
    assert resumed["models"][0]["status"] == "REUSED"
    assert [entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"] == before
    summary = json.loads(Path(resumed["models"][0]["summaries"]["full"]).read_text())
    assert summary["execution_status"] == "REUSED"
def test_transient_failure_retries_on_resume(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    monkeypatch.setenv("CIRCUS_TEST_FAIL_ONCE", "15")
    rc, failed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "selected")
    assert rc == 1, error
    failed_model = failed["models"][0]
    assert failed_model["status"] == "MODEL_UNAVAILABLE"
    failed_summary = json.loads(Path(failed_model["summaries"]["full"]).read_text())
    assert failed_summary["execution_status"] == "FAILED"
    assert failed_summary["condition_status"]["multimodal"]["status"] == "MODEL_UNAVAILABLE"
    before_retry = len(_log(tmp_path / "encodes.jsonl"))
    monkeypatch.delenv("CIRCUS_TEST_FAIL_ONCE")
    rc, recovered, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                "--models", "selected", "--resume")
    assert rc == 0, error
    assert recovered["models"][0]["status"] == "COMPLETED"
    assert recovered["models"][0]["attempt"] != failed_model["attempt"]
    assert len(_log(tmp_path / "encodes.jsonl")) > before_retry
    before_reuse = len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])
    rc, reused, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "selected", "--resume")
    assert rc == 0, error
    assert reused["models"][0]["status"] == "REUSED"
    assert len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"]) == before_reuse

def test_partial_transient_failure_retries_and_then_reuses(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    monkeypatch.setenv("CIRCUS_TEST_FAIL_ONCE", "2")
    rc, partial, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "selected")
    assert rc == 1, error
    assert partial["models"][0]["status"] == "PARTIAL"
    monkeypatch.delenv("CIRCUS_TEST_FAIL_ONCE")
    before = len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])
    rc, recovered, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                "--models", "selected", "--resume")
    assert rc == 0, error
    assert recovered["models"][0]["status"] == "COMPLETED"
    assert json.loads(Path(recovered["models"][0]["summaries"]["full"]).read_text(
        encoding="utf-8"))["metrics"]["multimodal"]["coverage"] == 1.0
    after = len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])
    assert after > before
    rc, reused, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "selected", "--resume")
    assert rc == 0, error
    assert reused["models"][0]["status"] == "REUSED"
    assert len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"]) == after

def test_worker_stderr_preserved_only_for_failed_model(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("workerfail", "workersuccess"))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "workerfail,workersuccess")

    assert rc == 1, error
    failed, successful = report["models"]
    failed_log = (artifacts / "composition/logs" / "workerfail_fixture" / failed["attempt"] /
                  "worker.stderr.log")
    assert failed["status"] == "MODEL_UNAVAILABLE"
    assert failed_log.read_text(encoding="utf-8") == "synthetic worker traceback\n"
    success_log = (artifacts / "composition/logs" / "workersuccess_fixture" / successful["attempt"] /
                   "worker.stderr.log")
    assert successful["status"] == "COMPLETED"
    assert not success_log.exists()




def test_attempt_outputs_preserve_full_smoke_resume_and_failure(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    rc, full, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                           "--models", "fixture3d")
    assert rc == 0, error
    full_model = full["models"][0]
    full_path = next(path for path in (artifacts / "composition/results").rglob("full/multimodal.jsonl")
                     if path.parent.parent.name == full_model["attempt"])
    assert len(_log(full_path)) == 4
    rc, smoke, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "fixture3d", "--smoke", "1")
    assert rc == 0, error
    smoke_model = smoke["models"][0]
    smoke_path = next(path for path in (artifacts / "composition/results").rglob("full/multimodal.jsonl")
                      if path.parent.parent.name == smoke_model["attempt"])
    assert smoke_model["attempt"] != full_model["attempt"]
    assert len(_log(full_path)) == 4 and len(_log(smoke_path)) == 1
    assert json.loads(Path(full_model["summaries"]["full"]).read_text())["query_count"] == 4
    assert json.loads(Path(smoke_model["summaries"]["full"]).read_text())["query_count"] == 1

    encode_queries = runner._encode_queries
    monkeypatch.setattr(runner, "_encode_queries",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic transient encoding failure")))
    rc, failed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d")
    monkeypatch.setattr(runner, "_encode_queries", encode_queries)
    assert rc == 1, error
    failed_model = failed["models"][0]
    assert failed_model["status"] == "MODEL_UNAVAILABLE"
    assert failed_model["attempt"] not in {full_model["attempt"], smoke_model["attempt"]}
    assert len(_log(full_path)) == 4
    failure_summary = json.loads(Path(failed_model["summaries"]["full"]).read_text())
    assert failure_summary["execution_status"] == "FAILED"
    assert failure_summary["condition_status"]["multimodal"]["status"] == "MODEL_UNAVAILABLE"
    failure_path = next(path for path in (artifacts / "composition/results").rglob("full/multimodal.jsonl")
                        if path.parent.parent.name == failed_model["attempt"])
    assert len(_log(failure_path)) == 4
    rc, recovered, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                "--models", "fixture3d", "--resume")
    assert rc == 0, error
    assert len(_log(full_path)) == 4
    assert recovered["models"][0]["attempt"] != failed_model["attempt"]

def test_category_scope_failure_cache_is_stable(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("scoped",))
    rc, first, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "scoped", "--resume")
    assert rc == 1, error
    encode_count = len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])
    for _ in range(2):
        rc, resumed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                  "--models", "scoped", "--resume")
        assert rc == 1, error
        assert resumed["models"][0]["status"] == "PARTIAL"
        assert len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"]) == encode_count

def test_partial_and_wholly_blocked_failures_are_unscored(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("fail", "fixture3d", "blocked"))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fail,fixture3d")
    assert rc == 1
    failed, good = report["models"]
    assert failed["status"] == "MODEL_UNAVAILABLE"
    assert "synthetic encoding failure" in failed["reason"]
    assert good["status"] == "COMPLETED"
    failed_result = next(path for path in (artifacts / "composition/results").rglob("multimodal.jsonl")
                         if "fail" in path.parts and path.parent.parent.name == failed["attempt"])
    failed_rows = [json.loads(line) for line in failed_result.read_text(encoding="utf-8").splitlines()]
    assert all(row["rank"] is None and row["scored"] is False for row in failed_rows)
    failed_summary = json.loads(Path(failed["summaries"]["full"]).read_text(encoding="utf-8"))
    assert failed_summary["metrics"]["multimodal"]["coverage"] == 0.0
    summary = json.loads(Path(good["summaries"]["full"]).read_text(encoding="utf-8"))
    assert summary["metrics"]["multimodal"]["coverage"] == 1.0

    rc, blocked, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "blocked")
    assert rc == 1
    model = blocked["models"][0]
    assert model["status"] == "MODEL_UNAVAILABLE"
    condition_status = model["condition_status"]["multimodal"]
    assert condition_status["status"] == "MODEL_UNAVAILABLE"
    assert "synthetic encoding failure" in condition_status["reason"]

    config, release, adapters, artifacts = _fixture(tmp_path / "blocked-selection",
                                                     ("notready", "fixture3d"))
    rc, selected, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                               "--models", "notready,fixture3d")
    assert rc == 1
    blocked_model, completed_model = selected["models"]
    assert blocked_model["model_id"] == "notready"
    assert blocked_model["status"] == "SELECTED_BUT_BLOCKED"
    assert "synthetic adapter deliberately blocked" in blocked_model["reason"]
    assert completed_model["model_id"] == "fixture3d"
    assert completed_model["status"] == "COMPLETED"


def test_category_scope_failures_are_separate_and_keep_other_queries(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("scoped",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "scoped")
    assert rc == 1
    assert report["models"][0]["status"] == "PARTIAL"
    row = report["models"][0]
    result_path = next((artifacts / "composition/results").rglob("full/multimodal.jsonl"))
    results = {item["qid"]: item for item in map(json.loads, result_path.read_text().splitlines())}
    assert results["q2"]["rank"] is None
    assert "batch_indices=(2,)" in results["q2"]["failure"]
    assert "qids=['q2']" in results["q2"]["failure"]
    assert all(results[qid]["rank"] is not None for qid in ("q0", "q1", "q3"))
    summary_path = Path(row["summaries"]["full"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"]["multimodal"]["coverage"] == pytest.approx(0.75)
    assert summary["category_scope_failures"]["multimodal"] == 1
    assert summary["condition_status"]["multimodal"]["category_scope_failure_count"] == 1


def test_query_chunking_preserves_candidate_dependent_scoring() -> None:
    class CandidateNormalized:
        def __init__(self) -> None:
            self.gallery_sizes: list[int] = []

        def score(self, queries, targets, categories=None):
            self.gallery_sizes.append(len(targets))
            raw = np.asarray(queries) @ np.asarray(targets).T
            scale = np.maximum(raw.std(axis=1, keepdims=True), np.finfo(np.float32).tiny)
            return (raw - raw.mean(axis=1, keepdims=True)) / scale

    adapter = CandidateNormalized()
    queries = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 4.0]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0], [-1.0, 3.0]], dtype=np.float32)
    expected = adapter.score(queries, targets)
    actual = runner.score_in_query_chunks(adapter, queries, targets, chunk_size=2)
    np.testing.assert_array_equal(actual, expected)
    assert adapter.gallery_sizes == [4, 4, 4]

def test_model_interpreter_identity_preserves_virtualenv_path(monkeypatch: pytest.MonkeyPatch) -> None:
    variable = "WORKBENCH_CIRCUS_TEST_PYTHON"
    monkeypatch.setenv(variable, sys.executable)
    model = {"model_id": "fixture", "environment": {"python_env_var": variable}}

    identity = runner.model_interpreter_identity(model, None, verify=False)

    assert identity["path"] == str(Path(sys.executable).absolute())
    assert identity["version"].startswith(f"Python {sys.version.split()[0]}")


def test_inventory_resolves_flat_png_for_resized_layout(tmp_path: Path) -> None:
    _config, release, _adapters, _artifacts = _fixture(tmp_path)
    dataset_root = tmp_path / "FashionIQ"
    images = dataset_root / "images"
    (images / "img0.jpg").rename(images / "X.png")
    candidates = release / "mbeir_fashioniq_task7_cand_pool.jsonl"
    candidates.write_text(json.dumps({"did": "X", "img_path": "X.jpg"}) + "\n", encoding="utf-8")
    query = {"qid": "q0", "query_txt": "test", "query_img_path": "X", "pos_cand_list": ["X"]}
    for path in (release / "mbeir_fashioniq_task7_test.jsonl",
                 release / "shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl",
                 release / "final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl"):
        path.write_text(json.dumps(query) + "\n", encoding="utf-8")
    (release / "final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl").write_text(
        "", encoding="utf-8")
    audit = release / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"
    audit.write_text(json.dumps({"gallery_paths": ["<path_to_M-BEIR>/X.jpg"]}), encoding="utf-8")
    data = composition_data.load_release(release, verify_hashes=False, expect_released_counts=False)

    inventory = runner.inspect_image_inventory(
        data, [{"model_id": "fixture", "fashioniq_layout": "fashioniq_ilearn_resized_training"}],
        dataset_root)

    assert inventory["status"] == "extension_mismatch"
    assert inventory["unresolvable_gallery_count"] == 0
    assert inventory["extension_resolution_counts"][".png"] == 1


def test_plan_reports_blocked_selection_without_persisting(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("notready",))
    (config.parent / "FashionIQ/images/img2.jpg").unlink()
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "notready", "--plan")
    assert rc == 0, error
    assert report["missing_assets"]["unresolvable_gallery_count"] == 1
    assert report["missing_assets"]["unresolvable_gallery_examples"] == ["gallery/img2.jpg"]
    assert report["gallery_count"] == 5
    readiness = report["readiness"][0]
    assert readiness["ready"] is False
    assert readiness["status"] == "SELECTED_BUT_BLOCKED"
    assert readiness["reason"] == "synthetic adapter deliberately blocked"
    runtime = readiness["runtime"]["model_interpreter"]
    assert runtime["path"] == str(Path(sys.executable).absolute())
    assert runtime["verified"] is False
    assert not artifacts.exists()


def test_output_containment_and_synthetic_guard(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="inside.*composition"):
        runner.assert_composition_destination(tmp_path / "artifacts", tmp_path / "outside.json")

    config, release, adapters, _artifacts = _fixture(tmp_path / "guard")
    monkeypatch.syspath_prepend(str(adapters))
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(config.parent / "registry.yaml"))
    monkeypatch.setenv("WORKBENCH_CIRCUS_RELEASE_ROOT", str(release))
    monkeypatch.delenv("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE", raising=False)
    from workbench.scripts import composition_eval
    rc = composition_eval.main(["--config", str(config), "--models", "fixture3d", "--plan"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "candidate count 5 != expected 74381" in captured.err

def test_bundle_dispatch_and_partial_member_refusal(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    registry_path = config.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    model = registry["models"][0]
    variants = []
    for member in ("dress", "shirt", "toptee"):
        checkpoint_id = f"member_{member}"
        filename = f"{checkpoint_id}.pt"
        (config.parent / "checkpoints" / "fixture3d" / filename).write_bytes(b"bundle member")
        variants.append({"checkpoint_id": checkpoint_id, "filename": filename,
                         "evaluation_noise_pct": 0, "checkpoint_training_noise_pct": None,
                         "checkpoint_mapping_status": "VERIFIED_METADATA"})
    model["checkpoint_variants"] = variants
    model["checkpoint_bundles"] = [{"bundle_id": "bundle-all",
        "required_checkpoint_ids": [item["checkpoint_id"] for item in variants],
        "structure": "category_specific_whole_models", "mapping_status": "VERIFIED_METADATA",
        "notes": "synthetic test bundle"}]
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    rc, explicit, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                               "--models", "fixture3d")
    assert rc == 0, error
    assert len(explicit["models"]) == 1
    assert explicit["models"][0]["checkpoint_id"] == "bundle-all"
    checkpoints = [entry["shape"][0] for entry in _log(tmp_path / "encodes.jsonl")
                   if entry["kind"] == "checkpoint"]
    assert checkpoints == ["bundle-all"]
    rc, selected, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                               "--models", "selected")
    assert rc == 0, error
    assert len(selected["models"]) == 1
    assert selected["models"][0]["checkpoint_id"] == "bundle-all"
    checkpoints = [entry["shape"][0] for entry in _log(tmp_path / "encodes.jsonl")
                   if entry["kind"] == "checkpoint"]
    assert checkpoints == ["bundle-all", "bundle-all"]
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["checkpoint_overrides"] = {"fixture3d": ["member_dress"]}
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    rc, partial, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "fixture3d")
    assert rc == 1, error
    assert partial["models"][0]["status"] == "UNSUPPORTED"
    assert "full checkpoint bundle 'bundle-all'" in partial["models"][0]["reason"]
    assert len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] == "checkpoint"]) == 2



def _append_independent_variant(config: Path, checkpoint_id: str, *, staged: bool = True,
                                download_url: str | None = None) -> None:
    """Register a second clean, clean-mapping checkpoint variant on the fixture model."""
    registry_path = config.parent / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    filename = f"{checkpoint_id}.pt"
    variant = {"checkpoint_id": checkpoint_id, "filename": filename, "evaluation_noise_pct": 0,
               "checkpoint_training_noise_pct": None, "checkpoint_training_dataset": "FashionIQ",
               "checkpoint_mapping_status": "VERIFIED_METADATA"}
    if download_url is not None:
        variant["download_url"] = download_url
    registry["models"][0]["checkpoint_variants"].append(variant)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    if staged:
        (config.parent / "checkpoints" / "fixture3d" / filename).write_bytes(b"synthetic checkpoint")


def _override_checkpoints(config: Path, model_id: str, checkpoint_ids: list[str]) -> None:
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_data["checkpoint_overrides"] = {model_id: checkpoint_ids}
    config.write_text(yaml.safe_dump(config_data), encoding="utf-8")


def test_explicit_multiple_independent_checkpoints_run_each(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    _append_independent_variant(config, "second")
    _override_checkpoints(config, "fixture3d", ["fixture", "second"])
    log_path = tmp_path / "encodes.jsonl"
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d")
    assert rc == 0, error
    assert len(report["models"]) == 2
    rows = {row["checkpoint_id"]: row for row in report["models"]}
    assert set(rows) == {"fixture", "second"}
    assert all(row["status"] == "COMPLETED" for row in rows.values())
    assert rows["fixture"]["attempt"] != rows["second"]["attempt"]
    for checkpoint_id, row in rows.items():
        assert row["run_id"]
        assert checkpoint_id in Path(row["summaries"]["full"]).parent.name
        assert Path(row["summaries"]["full"]).is_file()
    assert [entry["shape"][0] for entry in _log(log_path)
            if entry["kind"] == "checkpoint"] == ["fixture", "second"]
    before = [entry for entry in _log(log_path) if entry["kind"] != "checkpoint"]
    attempts = {key: row["attempt"] for key, row in rows.items()}
    summaries = {key: row["summaries"]["full"] for key, row in rows.items()}
    rc, resumed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "fixture3d", "--resume")
    assert rc == 0, error
    resumed_rows = {row["checkpoint_id"]: row for row in resumed["models"]}
    assert set(resumed_rows) == {"fixture", "second"}
    for checkpoint_id, row in resumed_rows.items():
        assert row["status"] == "REUSED"
        assert row["attempt"] == attempts[checkpoint_id]
        assert row["summaries"]["full"] == summaries[checkpoint_id]
    assert [entry for entry in _log(log_path) if entry["kind"] != "checkpoint"] == before


def test_selected_route_runs_each_selected_variant(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    _append_independent_variant(config, "second")
    _override_checkpoints(config, "fixture3d", ["fixture", "second"])
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "selected")
    assert rc == 0, error
    assert len(report["models"]) == 2
    assert {row["checkpoint_id"] for row in report["models"]} == {"fixture", "second"}
    assert all(row["status"] == "COMPLETED" for row in report["models"])
    assert report["run_count"] == 2
    assert [entry["shape"][0] for entry in _log(tmp_path / "encodes.jsonl")
            if entry["kind"] == "checkpoint"] == ["fixture", "second"]


def test_multi_variant_with_one_missing_checkpoint_reports_row(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path)
    _append_independent_variant(config, "second", staged=False,
                                download_url="https://example.invalid/second.pt")
    _override_checkpoints(config, "fixture3d", ["fixture", "second"])
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d")
    assert rc == 1, error
    rows = {row["checkpoint_id"]: row for row in report["models"]}
    assert set(rows) == {"fixture", "second"}
    assert rows["fixture"]["status"] == "COMPLETED"
    missing = rows["second"]
    assert missing["checkpoint_id"] == "second"
    assert missing["status"] == "SELECTED_BUT_BLOCKED"
    assert "checkpoint not staged" in missing["reason"]
    assert not (config.parent / "checkpoints/fixture3d/second.pt").exists()
    assert [entry["shape"][0] for entry in _log(tmp_path / "encodes.jsonl")
            if entry["kind"] == "checkpoint"] == ["fixture"]
