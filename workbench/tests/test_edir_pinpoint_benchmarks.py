"""Comprehensive tests for EDIR and PinPoint splits, dataloaders, and evaluators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from workbench.benchmarks import (
    CIRCorpusDataset,
    CIRQueryEvalDataset,
    CIRTrainingDataset,
    build_dataloaders,
    evaluate_edir,
    evaluate_pinpoint,
    load_split_bundle,
    load_split_ids,
    rank_embeddings,
    verify_split_integrity,
)

REPO_ROOT = Path("/home/heheboiz/data/cir")
TRACKED_MANIFEST_DIR = (
    REPO_ROOT / "workbench" / "docs" / "reviews" / "20261123-edir-pinpoint-splits" / "manifests"
)


def _tensor_transform():
    return transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


def test_edir_split_integrity_and_zero_leakage():
    manifest = verify_split_integrity("edir")
    assert manifest["counts"] == {"train": 3500, "val": 497, "test": 1003}
    assert sum(manifest["counts"].values()) == 5000

    # Plain-text ID files match tracked manifests and JSONL splits
    all_ids = []
    bundles = {}
    for sp in ("train", "val", "test"):
        ids = load_split_ids("edir", sp)
        tracked_ids = [
            l.strip()
            for l in (TRACKED_MANIFEST_DIR / f"edir_{sp}_ids.txt").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert ids == tracked_ids == manifest["splits"][sp]
        b = load_split_bundle("edir", sp, corpus_scope="full")
        assert [q.query_id for q in b.queries] == ids
        assert len(b.corpus) == 28645
        bundles[sp] = b
        all_ids.extend(ids)

    assert len(all_ids) == len(set(all_ids)) == 5000

    # Re-verify strict 0% leakage directly from loaded bundles
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        refs_a = {q.ref_id for q in bundles[a].queries}
        refs_b = {q.ref_id for q in bundles[b].queries}
        pos_a = {p for q in bundles[a].queries for p in q.positive_ids}
        pos_b = {p for q in bundles[b].queries for p in q.positive_ids}
        neg_a = {n for q in bundles[a].queries for n in q.negative_ids}
        neg_b = {n for q in bundles[b].queries for n in q.negative_ids}
        base_a = {q.metadata["stem_base"] for q in bundles[a].queries}
        base_b = {q.metadata["stem_base"] for q in bundles[b].queries}
        assert len(refs_a & refs_b) == 0
        assert len(pos_a & pos_b) == 0
        assert len(neg_a & neg_b) == 0
        assert len(base_a & base_b) == 0


def test_pinpoint_split_integrity_cleaning_and_disjoint_refs():
    manifest = verify_split_integrity("pinpoint")
    assert manifest["counts"] == {"train": 4724, "val": 679, "test": 1337}
    assert manifest["multi_image_eval_count"] == 895
    assert manifest["missing_403_neg_affected_queries"]["test"] == 0

    bundles = {}
    all_single_ids = []
    for sp in ("train", "val", "test", "multi_image_eval"):
        ids = load_split_ids("pinpoint", sp)
        tracked_ids = [
            l.strip()
            for l in (TRACKED_MANIFEST_DIR / f"pinpoint_{sp}_ids.txt").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert ids == tracked_ids
        b = load_split_bundle("pinpoint", sp, corpus_scope="full")
        assert [q.query_id for q in b.queries] == ids
        assert len(b.corpus) == 109598
        bundles[sp] = b
        if sp != "multi_image_eval":
            all_single_ids.extend(ids)

        # Verify cleaned training targets: no None, no duplicates, no pos&neg conflicts, all exist on disk
        for q in b.queries:
            assert q.positive_ids, f"{q.query_id} missing positives"
            assert len(q.positive_ids) == len(set(q.positive_ids))
            assert len(q.negative_ids) == len(set(q.negative_ids))
            assert not (set(q.positive_ids) & set(q.negative_ids))
            assert all(p and p in b.corpus_index for p in q.positive_ids)
            assert all(n and n in b.corpus_index for n in q.negative_ids)
            if sp == "multi_image_eval":
                assert not q.is_single and q.ref2_id is not None
            else:
                assert q.is_single and q.ref2_id is None

    assert len(all_single_ids) == len(set(all_single_ids)) == 6740

    # Zero ref/paraphrase overlap between train, val, test
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        refs_a = {q.ref_id for q in bundles[a].queries}
        refs_b = {q.ref_id for q in bundles[b].queries}
        assert len(refs_a & refs_b) == 0


def test_edir_and_pinpoint_dataloaders_load_real_batches():
    tf = _tensor_transform()

    # EDIR with hard negatives
    edir_loaders = build_dataloaders(
        "edir",
        train_image_transform=tf,
        eval_image_transform=tf,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        positive_mode="expand",
        num_hard_negatives=2,
        corpus_scope="split",
        verify_checksums=True,
    )
    tb = next(iter(edir_loaders["train_loader"]))
    assert tb["ref_image"].shape == (4, 3, 64, 64)
    assert tb["target_image"].shape == (4, 3, 64, 64)
    assert tb["hard_neg_images"].shape == (4, 2, 3, 64, 64)
    assert len(tb["text"]) == 4

    qb = next(iter(edir_loaders["val_query_loader"]))
    assert qb["ref_image"].shape == (4, 3, 64, 64)
    cb = next(iter(edir_loaders["val_corpus_loader"]))
    assert cb["image"].shape == (4, 3, 64, 64)

    # PinPoint: check both random (4724) and expand (exact 39,886 deduplicated pairs) modes
    pp_train = load_split_bundle("pinpoint", "train", corpus_scope="split")
    ds_rand = CIRTrainingDataset(pp_train, image_transform=tf, positive_mode="random", num_hard_negatives=2)
    ds_exp = CIRTrainingDataset(pp_train, image_transform=tf, positive_mode="expand", num_hard_negatives=0)
    assert len(ds_rand) == 4724
    assert len(ds_exp) == 39886
    s0 = ds_rand[0]
    assert s0["ref_image"].shape == (3, 64, 64)
    assert s0["target_image"].shape == (3, 64, 64)
    assert len(s0["hard_neg_images"]) == 2

    # PinPoint multi_image_eval loads both ref_image and ref2_image
    pp_multi = load_split_bundle("pinpoint", "multi_image_eval", corpus_scope="split")
    ds_multi = CIRQueryEvalDataset(pp_multi, image_transform=tf)
    m0 = ds_multi[0]
    assert m0["ref_image"].shape == (3, 64, 64)
    assert m0["ref2_image"].shape == (3, 64, 64)


def test_edir_evaluator_and_deterministic_rank_embeddings():
    b = load_split_bundle("edir", "val", corpus_scope="split")
    recs = b.queries[:20]
    # Construct synthetic embeddings where even-indexed queries match their positive exactly
    c_ids = [c.image_id for c in b.corpus]
    c_idx = {cid: i for i, cid in enumerate(c_ids)}
    dim = 16
    rng = np.random.default_rng(123)
    c_emb = rng.standard_normal((len(c_ids), dim)).astype(np.float32)
    q_emb = np.zeros((len(recs), dim), dtype=np.float32)
    for i, r in enumerate(recs):
        if i % 2 == 0:
            q_emb[i] = c_emb[c_idx[r.positive_ids[0]]]
        else:
            q_emb[i] = -c_emb[c_idx[r.positive_ids[0]]]

    preds = rank_embeddings(q_emb, [r.query_id for r in recs], c_emb, c_ids, top_k=10)
    res = evaluate_edir(preds, recs, ks=(1, 5, 10))
    assert res["overall"]["matched_queries"] == 20
    assert res["overall"]["Recall@1"] == pytest.approx(0.5, abs=1e-9)

    # Verify tie-breaking stability (identical scores preserve corpus order)
    tie_q = np.ones((1, 4), dtype=np.float32)
    tie_c = np.ones((5, 4), dtype=np.float32)
    tie_preds = rank_embeddings(tie_q, ["q0"], tie_c, ["c0", "c1", "c2", "c3", "c4"], top_k=5)
    assert tie_preds["q0"] == ["c0", "c1", "c2", "c3", "c4"]


def test_pinpoint_evaluator_matches_official_script_bit_exact():
    """Verify evaluate_pinpoint matches pinpoint-dataset/evaluate.py to 0.00e+00."""
    pp_repo = REPO_ROOT / "pinpoint-dataset"
    sample_res_files = sorted((pp_repo / "standardized_results").glob("*.json"))
    if not sample_res_files:
        pytest.skip("No standardized_results JSON found in pinpoint-dataset")

    # Import official metrics & filter_out_negatives directly from pinpoint-dataset
    sys.path.insert(0, str(pp_repo))
    try:
        from utils.data_utils import filter_out_negatives, normalize_query_id
        from utils.metrics import calculate_metrics_at_k, calculate_neg_recall_at_k
    finally:
        sys.path.pop(0)

    preds = json.loads(sample_res_files[0].read_text(encoding="utf-8"))
    test_bundle = load_split_bundle("pinpoint", "test", corpus_scope="split")
    ours = evaluate_pinpoint(preds, test_bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True)["overall"]

    # Compute reference directly using pinpoint-dataset's exact functions over the same test queries
    ks = [1, 5, 10, 50]
    ap = {k: [] for k in ks}
    ap_nn = {k: [] for k in ks}
    prec = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    neg_r = {k: [] for k in ks}
    variations = {}
    for q in test_bundle.queries:
        nid = normalize_query_id(q.query_id)
        if nid not in preds:
            continue
        ret = preds[nid].get("retrieved_items", [])
        rel = list(q.positive_ids_raw)
        neg = list(q.negative_ids_raw)
        ret_nn = filter_out_negatives(ret, neg)
        for k in ks:
            rk = calculate_metrics_at_k(ret, rel, k)
            ap[k].append(rk["ap"])
            prec[k].append(rk["precision"])
            rec[k].append(rk["recall"])
            neg_r[k].append(calculate_neg_recall_at_k(ret, neg, k))
            ap_nn[k].append(calculate_metrics_at_k(ret_nn, rel, k)["ap"])
        img2_raw = q.metadata.get("query_image_signature2_raw")
        vkey = (q.ref_id, img2_raw) if img2_raw else (q.ref_id, None)
        variations.setdefault(vkey, []).append(calculate_metrics_at_k(ret, rel, 10)["precision"])

    assert ours["total_queries"] == len(ap[10]) > 0
    for k in ks:
        assert ours[f"mAP@{k}"] == pytest.approx(float(np.mean(ap[k])), abs=1e-12)
        assert ours[f"mAP@{k}_noNeg"] == pytest.approx(float(np.mean(ap_nn[k])), abs=1e-12)
        assert ours[f"precision@{k}"] == pytest.approx(float(np.mean(prec[k])), abs=1e-12)
        assert ours[f"recall@{k}"] == pytest.approx(float(np.mean(rec[k])), abs=1e-12)
        assert ours[f"NegRecall@{k}"] == pytest.approx(float(np.mean(neg_r[k])), abs=1e-12)
    ref_ranges = [max(v) - min(v) for v in variations.values() if len(v) > 1]
    ref_stds = [float(np.std(v)) for v in variations.values() if len(v) > 1]
    assert ours["ling_sens_range"] == pytest.approx(float(np.mean(ref_ranges)), abs=1e-12)
    assert ours["ling_sens_std"] == pytest.approx(float(np.mean(ref_stds)), abs=1e-12)


def test_torch_dataset_inheritance_multiworker_and_nan_guard():
    from torch.utils.data import DataLoader, Dataset, Subset

    tf = _tensor_transform()
    b = load_split_bundle("pinpoint", "train", corpus_scope="split")
    ds_train = CIRTrainingDataset(b, image_transform=tf, positive_mode="random", num_hard_negatives=2, seed=42)
    ds_q = CIRQueryEvalDataset(b, image_transform=tf)
    ds_c = CIRCorpusDataset(b, image_transform=tf)

    assert isinstance(ds_train, Dataset)
    assert isinstance(ds_q, Dataset)
    assert isinstance(ds_c, Dataset)

    # Verify multi-worker DataLoader produces fresh samples across epochs (not frozen RNG)
    sub = Subset(ds_train, list(range(32)))
    gen = torch.Generator()
    gen.manual_seed(42)
    from workbench.benchmarks import collate_train_batch, seed_worker

    dl = DataLoader(
        sub,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=gen,
        collate_fn=collate_train_batch,
    )
    ep1 = [(q, t, tuple(n)) for batch in dl for q, t, n in zip(batch["query_id"], batch["target_id"], batch["hard_neg_ids"])]
    ep2 = [(q, t, tuple(n)) for batch in dl for q, t, n in zip(batch["query_id"], batch["target_id"], batch["hard_neg_ids"])]
    assert len(ep1) == len(ep2) == 32
    assert ep1 != ep2, "Multi-worker DataLoader sampled identical targets/negatives across epochs!"

    # Verify NaN/Inf guard in rank_embeddings
    with pytest.raises(ValueError, match="NaN or Inf"):
        rank_embeddings(np.array([[np.nan, 1.0]], dtype=np.float32), ["q0"], np.ones((2, 2), dtype=np.float32), ["c0", "c1"])
