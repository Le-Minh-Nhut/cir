"""Tests for PTHA+MTST source integrity, evaluator flags, and preflight blockers."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.replay.ptha_mtst import evaluator_path, preflight
MTST_DIR = REPO / "workbench" / "third_party" / "MTST_PTHA"
DATA_UTILS = MTST_DIR / "MTST_PTHA" / "src" / "data_utils.py"
MODELS_INIT = MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "__init__.py"
EVALUATOR = MTST_DIR / "MTST_PTHA" / "src" / "validate_blip.py"
DISPATCHER = MTST_DIR / "MTST_PTHA" / "src" / "blip_validate.py"


def test_mtst_data_root_fallback() -> None:
    assert DATA_UTILS.is_file(), f"Missing {DATA_UTILS}"
    content = DATA_UTILS.read_text(encoding="utf-8")
    assert 'os.environ.get("MTST_DATA_ROOT", "path_to_data_dir")' in content, (
        "data_utils.py lacks MTST_DATA_ROOT fallback"
    )


def test_guarded_import_in_models_init() -> None:
    assert MODELS_INIT.is_file(), f"Missing {MODELS_INIT}"
    content = MODELS_INIT.read_text(encoding="utf-8")
    assert "Blip2Qformer4TGIRCESIMZYN = None" in content, (
        "models/__init__.py lacks guarded import for Blip2Qformer4TGIRCESIMZYN"
    )
    assert "@registry.register_model(\"blip2_tgir_ce_sim\")" in (
        MTST_DIR / "MTST_PTHA" / "src" / "lavis" / "models" / "blip2_models" / "blip2_qformer_tgir_concatembedding_sim.py"
    ).read_text(encoding="utf-8")


def test_evaluator_flags() -> None:
    assert DISPATCHER.is_file(), f"Missing {DISPATCHER}"
    content = DISPATCHER.read_text(encoding="utf-8")
    for flag in ["--dataset", "--blip-model-name", "--backbone", "--model-path"]:
        assert f'"{flag}"' in content, f"Missing flag {flag} in {DISPATCHER}"


def test_validate_blip_cli_only_dispatches_cirr() -> None:
    content = EVALUATOR.read_text(encoding="utf-8")
    main = content[content.index("def main():"):]
    assert "if args.dataset == 'CIRR':" in main
    assert "args.dataset == 'fashionIQ'" not in main


def _dataset_root(root: Path) -> Path:
    for category in ("dress", "shirt", "toptee"):
        for relative in (
            f"captions/cap.{category}.val.json",
            f"image_splits/split.{category}.val.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


def test_preflight_uses_pinned_validate_blip_module() -> None:
    assert evaluator_path(MTST_DIR) == EVALUATOR
    assert EVALUATOR.is_file()
    assert DISPATCHER.is_file()
    blockers = preflight(MTST_DIR, REPO / "data" / "fashionIQ_dataset", REPO / "workbench" / "artifacts" / "checkpoints")
    assert not any("pinned evaluator missing" in blocker for blocker in blockers)


def test_cirr_checkpoint_requires_explicit_transfer_opt_in(tmp_path: Path) -> None:
    # The checkpoint root IS the member root: `_replay_command` passes `checkpoint_member_root` of the
    # resolved path, which for a single file is its own directory, and every wrapper then opens
    # `<root>/<registry filename>`. Nesting the model directory here encoded the doubled-id bug this
    # test is supposed to catch, so the fixture now matches the real convention.
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "tuned_mtst_cirr.pt").write_bytes(b"placeholder")
    dataset_root = _dataset_root(tmp_path / "fashioniq")

    normal = preflight(MTST_DIR, dataset_root, checkpoint_root)
    assert any("CIRR weights require explicit transfer opt-in" in blocker for blocker in normal)

    transfer = preflight(
        MTST_DIR,
        dataset_root,
        checkpoint_root,
        transfer_cirr_to_fashioniq=True,
    )
    assert not any("CIRR weights require explicit transfer opt-in" in blocker for blocker in transfer)
    assert not any("CIRR checkpoint missing" in blocker for blocker in transfer)


def test_preflight_blocks_on_missing_fashioniq_weights(tmp_path: Path) -> None:
    blockers = preflight(
        source_root=MTST_DIR,
        dataset_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    assert any("FashionIQ fine-tuned checkpoint missing" in blocker for blocker in blockers)
    assert any("tuned_mtst_cirr.pt" in blocker for blocker in blockers)
    # And the path it names must be `<root>/<filename>`, with no model directory appended twice: a
    # doubled root made a correctly placed checkpoint unreadable, which is the defect R19-02 records.
    assert not any("ptha_mtst/ptha_mtst/" in blocker for blocker in blockers), blockers


def test_the_nested_source_layout_resolves_to_the_directory_holding_src() -> None:
    """The vendored checkout keeps the repository's own top directory inside itself.

    `third_party/MTST_PTHA/` contains `MTST_PTHA/src/`, while `requirements.txt` and `provenance.json`
    sit at the outer level. Both call sites used to name `<checkout>/MTST_PTHA/src` -- wait, they named
    `<checkout>/src`, which does not exist -- so preflight reported the pinned evaluator missing for a
    checkout that contains it, and the import path would have failed on `data_utils`. ONE resolver now
    answers for both, and this test pins the layout it must handle: nested when the tree is nested,
    flat when a future vendoring flattens it.
    """
    from pathlib import Path

    from workbench.replay import ptha_mtst

    real = Path("workbench/third_party/MTST_PTHA")
    if not real.is_dir():
        return pytest.skip("vendored MTST_PTHA checkout is not present")
    tree = ptha_mtst.source_tree(real)
    assert (tree / "src" / "validate_blip.py").is_file(), (
        f"{tree} does not hold src/validate_blip.py; the resolver picked the wrong level")
    # `evaluator_path` must agree with the resolver, since they are read by different callers.
    assert ptha_mtst.evaluator_path(real) == tree / "src" / "validate_blip.py"


def test_a_transfer_run_needs_the_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive cache fixture isolates setup; the final refusal asserts the opt-in gate."""
    from pathlib import Path

    from workbench.replay import ptha_mtst

    source = Path("workbench/third_party/MTST_PTHA")
    if not source.is_dir():
        return pytest.skip("vendored MTST_PTHA checkout is not present")
    dataset = tmp_path / "FashionIQ"
    (dataset / "images").mkdir(parents=True, exist_ok=True)
    for relative in ptha_mtst.RAW_INPUTS:
        (dataset / relative).parent.mkdir(parents=True, exist_ok=True)
        (dataset / relative).write_text("[]")

    # Flat member root, matching `checkpoint_member_root` and every wrapper's `root / filename`:
    # a nested `ptha_mtst/` here made a correctly placed checkpoint read as missing.
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / ptha_mtst.PUBLISHED_CIRR_WEIGHT).write_bytes(b"weights")

    torch_home = tmp_path / "torch"
    eva = torch_home / "hub/checkpoints/eva_vit_g.pth"
    eva.parent.mkdir(parents=True)
    eva.write_bytes(b"eva")
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_EVA_VIT_G_PTH", "3")

    from workbench.backend.registry import auxiliary_assets_for_model, hf_hub_repo_dir

    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    revision = next(asset["source_revision"] for asset in auxiliary_assets_for_model("ptha_mtst")
                    if asset.get("hub_repo_id") == "bert-base-uncased")
    repo_dir = hf_hub_repo_dir("bert-base-uncased", hf_home)
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(revision)
    (snapshot / "config.json").write_text('{"model_type":"bert","vocab_size":7}')
    (snapshot / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"BertTokenizer","do_lower_case":true}')
    (snapshot / "tokenizer.json").write_text(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],'
        '"normalizer":{"type":"BertNormalizer","clean_text":true,'
        '"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},'
        '"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":null,'
        '"decoder":null,"model":{"type":"WordPiece","unk_token":"[UNK]",'
        '"continuing_subword_prefix":"##","max_input_chars_per_word":100,'
        '"vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,'
        '"hello":5,"world":6}}}')
    (snapshot / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    (snapshot / "pytorch_model.bin").write_bytes(b"tiny-model-payload\x00\xff")
    payload_size = sum((snapshot / name).stat().st_size for name in (
        "config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", str(payload_size))

    # The positive case proves opt-in permits the transfer; the negative case below is the gate.
    assert ptha_mtst.preflight(source, dataset, checkpoint_root,
                               transfer_cirr_to_fashioniq=True) == []
    refused = ptha_mtst.preflight(source, dataset, checkpoint_root,
                                  transfer_cirr_to_fashioniq=False)
    assert refused and "transfer opt-in" in refused[0], refused


def test_adapter_runtime_preflight_cli_uses_configured_security_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from workbench.replay import ptha_mtst

    called: list[str] = []

    def verify(checkpoint_id: str) -> str:
        called.append(checkpoint_id)
        return "/verified/model/python"

    monkeypatch.setattr(ptha_mtst, "_verify_adapter_runtime", verify)
    args = ["--source-root", str(tmp_path / "source"), "--dataset-root", str(tmp_path / "data"),
            "--checkpoint-root", str(tmp_path / "checkpoints"), "--checkpoint-id", "cirr_transfer",
            "--verify-adapter-runtime"]
    assert ptha_mtst.main(args) == 0
    assert called == ["cirr_transfer"]




def test_ptha_transfer_batch_receipts_reject_mismatched_observed_partitions() -> None:
    from workbench.replay import ptha_mtst

    ptha_mtst._validate_native_batch_observations(
        {"query": [1, 1], "gallery": [2, 1]}, 2, 3, 1, 2)
    with pytest.raises(ValueError, match="native observed batch partitions"):
        ptha_mtst._validate_native_batch_observations(
            {"query": [2], "gallery": [2, 1]}, 2, 3, 1, 2)
    with pytest.raises(ValueError, match="native observed batch partitions"):
        ptha_mtst._validate_native_batch_observations(
            {"query": [1, 1], "gallery": [3]}, 2, 3, 1, 2)

    ptha_mtst._validate_adapter_batch_receipt(
        {"batch_size": 2, "batch_lengths": [2, 1]}, 3, 2, "query")
    with pytest.raises(ValueError, match="Adapter query batch receipt mismatch"):
        ptha_mtst._validate_adapter_batch_receipt(
            {"batch_size": 2, "batch_lengths": [3]}, 3, 2, "query")
    ptha_mtst._validate_adapter_score_batches([2, 1], 3, 2)
    with pytest.raises(ValueError, match="Adapter score query chunks"):
        ptha_mtst._validate_adapter_score_batches([3], 3, 2)

def test_native_capture_publication_and_paired_consumer_roundtrip(tmp_path: Path) -> None:
    import json
    import numpy as np

    from workbench.replay.ptha_mtst import (
        _load_native_capture, _publish_native_capture, _publish_paired_capture,
    )
    from workbench.composition.topk import topk_from_scores

    root = tmp_path / "capture"
    native_meta = {
        "query_sha256": ["q-hash"], "gallery_sha256": ["g0-hash", "g1-hash"],
        "source": {"commit": "pinned-source"},
    }
    native_arrays = {
        "query_ids": np.asarray(["query.png"]),
        "gallery_ids": np.asarray(["g0.png", "g1.png"]),
        "native_scores": np.asarray([[0.9, 0.1]], dtype=np.float32),
        "native_ranks": np.asarray([[0, 1]], dtype=np.int64),
        "native_top10": np.asarray([[0, 1]], dtype=np.int64),
    }
    _publish_native_capture(root, "dress", native_arrays, native_meta)
    loaded_meta, loaded = _load_native_capture(root, "dress")
    assert loaded_meta == native_meta
    assert np.array_equal(loaded["native_scores"], native_arrays["native_scores"])
    assert loaded["native_ranks"].shape == (1, 2)

    from workbench.composition.adapters import CompositionAdapter
    from workbench.replay.ptha_mtst import _adapter_rank_and_topk

    adapter_scores = np.asarray([[0.1, 0.9]], dtype=np.float32)
    adapter_positive_ranks, adapter_top, native_positive_ranks, adapter_hits, rank_equal = (
        _adapter_rank_and_topk(adapter_scores, [[1]], np.asarray([[0, 1]]),
                               CompositionAdapter._ranks_from_scores))
    assert adapter_positive_ranks.tolist() == [1]
    assert native_positive_ranks.tolist() == [2]
    assert not rank_equal
    assert adapter_hits == {"10": 1, "50": 1}
    assert adapter_top.tolist() == [[1, 0]]
    assert [position for position, _score in topk_from_scores(adapter_scores[0], 2)] == [1, 0]
    paired_metadata = {
        "positive_rank_equal": rank_equal,
        "adapter_scorer": "actual Adapter.score output captured by adapter_scores",
        "adapter_rank_policy": "actual Adapter._ranks_from_scores: positive rank",
        "adapter_topk_policy": "actual topk_from_scores over Adapter.score rows",
    }
    _publish_paired_capture(root, "dress", {
        "adapter_scores": adapter_scores,
        "adapter_positive_ranks": adapter_positive_ranks,
        "native_positive_ranks": native_positive_ranks,
        "adapter_top10": adapter_top[:, :1],
        "adapter_top50": adapter_top,
    }, paired_metadata)
    with np.load(root / "dress.paired.npz", allow_pickle=False) as artifact:
        assert artifact["adapter_positive_ranks"].shape == (1,)
        assert np.array_equal(artifact["adapter_positive_ranks"], [1])
        assert np.array_equal(artifact["native_positive_ranks"], [2])
        assert np.array_equal(artifact["adapter_top10"], [[1]])
        assert np.array_equal(artifact["adapter_top50"], [[1, 0]])
    assert json.loads((root / "dress.paired.json").read_text()) == paired_metadata
    assert (root / "dress.native.json").is_file()


def test_adapter_capture_consumer_reads_native_and_writes_production_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    import numpy as np

    from workbench.composition.adapters import CompositionAdapter
    from workbench.replay import ptha_mtst

    class FakeAdapter:
        def __init__(self, _config: object, checkpoint_id: str) -> None:
            assert checkpoint_id == "cirr_transfer"
            self.batch_size = 1
            self._last_batch_receipt: dict[str, object] = {}
            self.work_root = tmp_path

        def encode_queries(self, _paths: list[str], _texts: list[str], _categories: list[None]) -> np.ndarray:
            self._last_batch_receipt = {"batch_size": 16, "batch_lengths": [1]}
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        def encode_targets(self, _paths: list[str], _categories: list[None]) -> np.ndarray:
            self._last_batch_receipt = {"batch_size": 64, "batch_lengths": [2]}
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        @staticmethod
        def score(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
            return queries @ targets.T

        _ranks_from_scores = staticmethod(CompositionAdapter._ranks_from_scores)

        def close(self) -> None:
            pass

    monkeypatch.setattr(ptha_mtst, "_verify_adapter_runtime", lambda _checkpoint: "/verified/python")
    monkeypatch.setattr("workbench.composition.model_ptha_mtst.Adapter", FakeAdapter)
    monkeypatch.setattr("workbench.backend.operator_config.resolve_config", lambda: object())

    root = tmp_path / "native"
    image_root = tmp_path / "FashionIQ" / "fashionIQ_dataset" / "images"
    image_root.mkdir(parents=True)
    for category in ptha_mtst.CATEGORIES:
        gallery_ids = ["B0084Y8XIU", f"{category}-g1"]
        query_id, first_id, second_id = f"{category}-query", *gallery_ids
        paths = [image_root / f"{name}.png" for name in (query_id, first_id, second_id)]
        for index, path in enumerate(paths):
            payload = b"same bare-ID gallery image" if path.name == "B0084Y8XIU.png" else f"{category}-{index}".encode()
            path.write_bytes(payload)
        query_sha, first_sha, second_sha = [ptha_mtst.sha256_file(path) for path in paths]
        native_meta = {
            "query_sha256": [query_sha], "gallery_sha256": [first_sha, second_sha],
            "source": {"commit": "pinned-source"}, "native_batch_sizes": {"query": 16, "gallery": 64},
            "num_workers": {"query": 0, "gallery": 0},
            "native_hits": {"10": 1, "50": 1},
            "native_metric": {"10": {"hits": 1, "denominator": 1},
                              "50": {"hits": 1, "denominator": 1}},
            "native_sort": {"device": "cpu", "dtype": "torch.int64", "shape": [1, 2], "dim": -1},
        }
        native_arrays = {
            "query_ids": np.asarray([query_id]), "gallery_ids": np.asarray([first_id, second_id]),
            "target_ids": np.asarray([first_id]), "reference_ids": np.asarray([query_id]),
            "query_texts": np.asarray(["change caption"]), "processed_texts": np.asarray(["change caption"]),
            "native_g1_pixels": np.zeros((3, 2, 2), dtype=np.float32),
            "native_g1_pixel_gallery_id": np.asarray(first_id),
            "native_g1_gallery_features": np.zeros((32, 768), dtype=np.float32),
            "native_query_reference_embeds": np.zeros((257, 1408), dtype=np.float32),
            "native_gallery_features": np.zeros((2, 32, 256), dtype=np.float32),
            "native_query_features": np.zeros((1, 256), dtype=np.float32),
            "native_g1_input_ids": np.zeros((1, 32), dtype=np.int64),
            "native_g1_attention_masks": np.ones((1, 32), dtype=np.int64),
            "native_scores": np.asarray([[0.9, 0.1]], dtype=np.float32),
            "native_ranks": np.asarray([[0, 1]], dtype=np.int64),
            "native_top10": np.asarray([[0, 1]], dtype=np.int64),
            "native_top50": np.asarray([[0, 1]], dtype=np.int64),
        }
        ptha_mtst._publish_native_capture(root, category, native_arrays, native_meta)

    ptha_mtst._evaluate_adapter_capture(root, tmp_path / "FashionIQ", "cirr_transfer")
    for category in ptha_mtst.CATEGORIES:
        with np.load(root / f"{category}.paired.npz", allow_pickle=False) as paired:
            assert paired["adapter_positive_ranks"].shape == (1,)
            assert np.array_equal(paired["adapter_positive_ranks"], [1])
            assert paired["adapter_top10"].shape == (1, 2)
            assert np.array_equal(paired["adapter_top10"], [[0, 1]])
            assert np.array_equal(paired["adapter_top50"], [[0, 1]])
            assert paired["query_ids"].tolist() == [f"{category}-query"]
            assert paired["gallery_ids"].tolist() == ["B0084Y8XIU", f"{category}-g1"]
        metadata = json.loads((root / f"{category}.paired.json").read_text())
        assert metadata["gallery_sha256"] == [ptha_mtst.sha256_file(image_root / "B0084Y8XIU.png"),
                                                ptha_mtst.sha256_file(image_root / f"{category}-g1.png")]
        assert "actual Adapter.score" in metadata["adapter_scorer"]
        assert "no derived full permutation" in metadata["adapter_topk_policy"]
        from workbench.backend.parity_gate import _metrics_checks

        identity = {
            "protocol_id": "fashioniq_val_transfer",
            "checkpoint_id": "cirr_transfer",
            "query_order_sha256": ptha_mtst._ordered_ids_sha256([f"{category}-query"]),
            "gallery_order_sha256": ptha_mtst._ordered_ids_sha256(["B0084Y8XIU", f"{category}-g1"]),
        }
        native_artifact = {
            **identity,
            "native_metrics_parity_detail": metadata["native_metrics_parity_detail"],
            "native_metrics_match_feature_rerank": metadata["native_metrics_match_feature_rerank"],
        }
        adapter_artifact = dict(identity)
        checks = _metrics_checks(native_artifact, adapter_artifact, category)
        assert checks
        assert all(check.ok for check in checks)

    from workbench.backend import parity_gate

    metadata = json.loads((root / "dress.paired.json").read_text())
    category = "dress"
    gate_root = tmp_path / "transfer-gate"
    gate_root.mkdir()
    identity = {
        "model_id": "ptha_mtst", "category": category,
        "protocol_id": "fashioniq_val_cirr_transfer_exploratory",
        "checkpoint_id": "cirr_transfer",
        "query_order_sha256": ptha_mtst._ordered_ids_sha256(["dress-query"]),
        "gallery_order_sha256": ptha_mtst._ordered_ids_sha256(["B0084Y8XIU", "dress-g1"]),
    }
    native_doc = {**identity,
                  "native_metrics_parity_detail": metadata["native_metrics_parity_detail"],
                  "native_metrics_match_feature_rerank": metadata["native_metrics_match_feature_rerank"]}
    adapter_doc = {**identity, "adapter": {"transfer_comparison": "exploratory only"}}
    (gate_root / f"g1_g2.ptha_mtst.{category}.native.json").write_text(json.dumps(native_doc))
    (gate_root / f"g1_g2.ptha_mtst.{category}.adapter.json").write_text(json.dumps(adapter_doc))
    verdict = parity_gate.evaluate_model("ptha_mtst", categories=(category,), root=gate_root)
    assert verdict.status == "PARITY_MANDATORY_SUBCHECK_FAILED"
    metric_check = next(check for check in verdict.checks
                        if check.name == f"{category}.metrics.hit_count_equality")
    assert metric_check.ok



def test_native_capture_controller_dispatches_through_verified_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    from workbench.replay import ptha_mtst

    calls: list[list[str]] = []
    monkeypatch.setattr(ptha_mtst, "preflight", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ptha_mtst, "_verify_adapter_runtime", lambda _checkpoint: "/verified/native/python")

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ptha_mtst.subprocess, "run", run)
    capture = tmp_path / "capture"
    args = ["--source-root", str(MTST_DIR), "--dataset-root", str(tmp_path / "data"),
            "--checkpoint-root", str(tmp_path / "weights"), "--checkpoint-id", "cirr_transfer",
            "--capture-directory", str(capture), "--capture-stage", "native",
            "--transfer-cirr-to-fashioniq"]
    assert ptha_mtst.main(args) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[0] == "/verified/native/python"
    assert str(Path(ptha_mtst.__file__).resolve()) == command[1]
    assert "--native-capture-child" in command
    assert command[command.index("--capture-stage") + 1] == "native"
    assert command[command.index("--capture-directory") + 1] == str(capture)
    assert "--gallery-probe-batches" not in command


def test_transfer_v3_plan_is_immutable_and_role_receipts_bind_capture(tmp_path: Path) -> None:
    import json
    import numpy as np

    from workbench.replay.ptha_mtst import (
        _canonical_sha256, _ordered_ids_sha256, _publish_v3_role_receipt, _score_proof,
        _v3_capture_receipt, publish_transfer_v3_plan, transfer_v3_role_receipt,
    )

    query_ids, gallery_ids, target_ids = ["q1", "q2"], ["g1", "g2", "g3"], ["g1", "g2"]
    identity = {
        "checkpoint": "checkpoint-sha256", "configuration": "config-sha256",
        "environment": "env-sha256", "protocol": "protocol-sha256", "category": "dress",
        "cohort_id": "fixture", "precision": "float32", "seed": 0,
        "query_order_sha256": _ordered_ids_sha256(query_ids),
        "gallery_order_sha256": _ordered_ids_sha256(gallery_ids),
        "reference_exclusions_sha256": _canonical_sha256({
            "query_targets": [[query, target] for query, target in zip(query_ids, target_ids)],
            "reference_exclusions": [[query, []] for query in query_ids]}),
        "query_count": 2, "gallery_count": 3,
    }
    plan_path = tmp_path / "plans" / "dress.json"
    plan_hash = publish_transfer_v3_plan(plan_path, identity, query_ids, gallery_ids, target_ids)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_transfer_v3_plan(plan_path, identity, query_ids, gallery_ids, target_ids)
    receipt = _v3_capture_receipt(plan_path, "native_adapter_native", "native",
                                  "dress", query_ids, gallery_ids, target_ids, query_ids)
    assert receipt is not None and receipt["plan_sha256"] == plan_hash
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["score_order_conventions"] == {
        "native": "fp32_one_minus_score_ascending",
        "adapter": "fp32_descending_score",
    }
    assert set(plan["runs"]) == {
        "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
        "adapter_batch_1", "adapter_batch_n", "native_repeat_1", "native_repeat_2",
        "adapter_repeat_1", "adapter_repeat_2"}
    assert set(plan["controls"]) == {
        "native_adapter", "native_repeat", "adapter_repeat",
        "native_batch_1_vs_n", "adapter_batch_1_vs_n"}
    assert plan["supported_batch_policy"]["supported"] is True
    assert plan["supported_batch_policy"]["native_query_batch"] == 16
    assert plan["supported_batch_policy"]["native_gallery_batch"] == 64
    assert plan["supported_batch_policy"]["batch_sizes"] == [1, 16, 64]
    repeat_receipt = transfer_v3_role_receipt(plan_path, "adapter_repeat_1", "adapter")
    assert repeat_receipt["execution"]["query_batch_size"] == 16
    assert repeat_receipt["execution"]["gallery_batch_size"] == 64
    native_b1 = transfer_v3_role_receipt(plan_path, "native_batch_1", "native")
    adapter_b1 = transfer_v3_role_receipt(plan_path, "adapter_batch_1", "adapter")
    assert native_b1["execution"]["query_batch_size"] == 1
    assert native_b1["execution"]["gallery_batch_size"] == 1
    assert adapter_b1["execution"]["query_batch_size"] == 1
    assert adapter_b1["execution"]["gallery_batch_size"] == 1

    artifact = tmp_path / "dress.native.npz"
    artifact.write_bytes(b"fixture capture")
    scores = np.asarray([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]], dtype=np.float32)
    import torch

    ranks = torch.argsort(1.0 - torch.from_numpy(scores), dim=-1).numpy()
    proof = _score_proof(receipt, scores, ranks)
    assert proof is not None and proof["role"] == "native_adapter_native"
    assert proof["score_order_convention"] == "fp32_one_minus_score_ascending"
    with pytest.raises(ValueError, match=r"torch.argsort\(1 - pred_sim\)"):
        _score_proof(receipt, scores, ranks[:, ::-1].copy())
    adapter_receipt = transfer_v3_role_receipt(plan_path, "native_adapter_adapter", "adapter")
    adapter_receipt["positive_positions"] = [0, 1]
    adapter_proof = _score_proof(adapter_receipt, scores, np.asarray([1, 1], dtype=np.int64))
    assert adapter_proof is not None and adapter_proof["role"] == "native_adapter_adapter"
    assert adapter_proof["score_order_convention"] == "fp32_descending_score"
    assert adapter_proof["rank_shape"] == [2]
    assert adapter_proof["rank_semantics"] == "positive_rank_1based"
    with pytest.raises(ValueError, match="Adapter ranks contradict"):
        _score_proof(adapter_receipt, scores, np.asarray([2, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="must bind"):
        _publish_v3_role_receipt(tmp_path, "dress", receipt, artifact, adapter_proof)
    assert proof["score_order_convention"] == "fp32_one_minus_score_ascending"
    with pytest.raises(ValueError, match="receipt category/stage"):
        _publish_v3_role_receipt(tmp_path, "shirt", receipt, artifact, proof)
    _publish_v3_role_receipt(tmp_path, "dress", receipt, artifact, proof)
    stored = json.loads((tmp_path / "dress.native_adapter_native.v3-role.json").read_text())
    assert stored["artifact_sha256"]
    assert stored["plan_sha256"] == plan_hash
    with pytest.raises(ValueError, match="already exists"):
        _publish_v3_role_receipt(tmp_path, "dress", receipt, artifact, proof)
    from workbench.replay.aggregate_fashioniq_v3 import aggregate

    with pytest.raises(ValueError, match="missing actual role receipts"):
        aggregate(plan_path, [])

    wrong_identity = dict(identity, query_order_sha256=_ordered_ids_sha256(["q2", "q1"]))
    other_plan = tmp_path / "other" / "dress.json"
    publish_transfer_v3_plan(other_plan, wrong_identity, query_ids, gallery_ids, target_ids)
    with pytest.raises(ValueError, match="does not match captured"):
        _v3_capture_receipt(other_plan, "native_adapter_native", "native",
                            "dress", query_ids, gallery_ids)


def test_transfer_v3_identity_tracks_annotation_and_filtered_gallery(tmp_path: Path) -> None:
    import hashlib
    import json
    from workbench.replay.ptha_mtst import _transfer_v3_category_identity

    images = tmp_path / "images"
    images.mkdir()
    (images / "present.png").write_bytes(b"present")
    annotation = tmp_path / "cap.dress.val.json"
    rows = [{"candidate": "query.jpg", "target": "target.jpg", "captions": ["red", "dress"]}]
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    annotation_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    first = _transfer_v3_category_identity(
        "dress", rows, ["missing", "present"], images, annotation_sha256=annotation_hash,
        gallery_sha256="a" * 64, checkpoint_sha256="b" * 64, checkpoint_id="cirr_transfer",
        checkpoint_bytes=7, source_hashes={"source.py": "c" * 64}, runtime_digest="d" * 64,
        checkpoint_model_key="Blip2QformerCirCat", model_name="blip2_cir_cat", model_type="pretrain")
    changed_rows = [{**rows[0], "target": "new-target.jpg", "captions": ["blue", "dress"]}]
    annotation.write_text(json.dumps(changed_rows), encoding="utf-8")
    changed_hash = hashlib.sha256(annotation.read_bytes()).hexdigest()
    second = _transfer_v3_category_identity(
        "dress", changed_rows, ["missing", "present"], images, annotation_sha256=changed_hash,
        gallery_sha256="a" * 64, checkpoint_sha256="b" * 64, checkpoint_id="cirr_transfer",
        checkpoint_bytes=7, source_hashes={"source.py": "c" * 64}, runtime_digest="d" * 64,
        checkpoint_model_key="Blip2QformerCirCat", model_name="blip2_cir_cat", model_type="pretrain")
    assert first["configuration"] != second["configuration"]
    assert first["reference_exclusions_sha256"] != second["reference_exclusions_sha256"]
    assert first["gallery_count"] == 1
    assert first["gallery_order_sha256"] == second["gallery_order_sha256"]

def test_transfer_v3_receipt_binds_duplicate_query_rows_and_labels(tmp_path: Path) -> None:
    import hashlib
    from workbench.replay.ptha_mtst import (
        _canonical_sha256, _ordered_ids_sha256, _transfer_v3_category_identity,
        _v3_capture_receipt, publish_transfer_v3_plan,
    )

    images = tmp_path / "images"
    images.mkdir()
    for image_id in ("gallery", "target-a", "target-b"):
        (images / f"{image_id}.png").write_bytes(image_id.encode())
    pairs = [
        {"candidate": "same.jpg", "target": "target-a.jpg", "captions": ["red", "coat"]},
        {"candidate": "same.jpg", "target": "target-b.jpg", "captions": ["blue", "coat"]},
    ]
    annotation = tmp_path / "annotation.json"
    annotation.write_text("[]", encoding="utf-8")
    identity = _transfer_v3_category_identity(
        "dress", pairs, ["gallery", "target-a", "target-b"], images,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
        gallery_sha256="a" * 64, checkpoint_sha256="b" * 64, checkpoint_id="cirr_transfer",
        checkpoint_bytes=7, source_hashes={"source.py": "c" * 64}, runtime_digest="d" * 64,
        checkpoint_model_key="Blip2QformerCirCat", model_name="blip2_cir_cat", model_type="pretrain")
    plan_path = tmp_path / "plans" / "dress.json"
    query_ids = ["same.png#row-0", "same.png#row-1"]
    gallery_ids, target_ids = ["gallery", "target-a.png", "target-b.png"], ["target-a.png", "target-b.png"]
    identity.update(query_order_sha256=_ordered_ids_sha256(query_ids),
                    gallery_order_sha256=_ordered_ids_sha256(gallery_ids),
                    reference_exclusions_sha256=_canonical_sha256({
                        "query_targets": [[query, target] for query, target in zip(query_ids, target_ids)],
                        "reference_exclusions": [[query, []] for query in query_ids]}))
    publish_transfer_v3_plan(plan_path, identity, query_ids, gallery_ids, target_ids)
    receipt = _v3_capture_receipt(
        plan_path, "native_adapter_native", "native", "dress", query_ids, gallery_ids,
        target_ids, query_ids)
    assert receipt is not None
    with pytest.raises(ValueError, match="does not match captured"):
        _v3_capture_receipt(
            plan_path, "native_adapter_native", "native", "dress",
            ["same.png#row-0", "same.png#row-1"], gallery_ids,
            ["target-b.png", "target-a.png"], ["same.png#row-0", "same.png#row-1"])
