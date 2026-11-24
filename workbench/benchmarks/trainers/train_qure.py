#!/usr/bin/env python3
"""Upstream-faithful QuRe (Blip2QuRe, ICML 2025) trainer for EDIR and PinPoint canonical splits.

Initialized strictly from base pretrained BLIP-2 ViT-L weights (`pretrain_vitL`, NEVER FashionIQ/CIRR).
Implements QuRe's exact pairwise KL reward objective + periodic hard negative mining between the two
steepest relevance drops strictly within the training split's target pool (zero val/test leakage).

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/QuRe \
    workbench/envs/qure310-sm120/bin/python \
    workbench/benchmarks/trainers/train_qure.py --dataset edir
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
QURE_ROOT = REPO_ROOT / "workbench" / "third_party" / "QuRe"
for p in (str(REPO_ROOT), str(QURE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.utils import targetpad_transform  # noqa: E402
from lavis.models import load_model_and_preprocess  # noqa: E402

from workbench.benchmarks import (  # noqa: E402
    CIRCorpusDataset,
    CIRQueryEvalDataset,
    collate_corpus,
    collate_eval_queries,
    evaluate_edir,
    evaluate_pinpoint,
    load_split_bundle,
    seed_worker,
    verify_split_integrity,
)
from workbench.benchmarks.schema import DEFAULT_SPLIT_TAG, SplitBundle  # noqa: E402
from workbench.benchmarks.trainers.common import (  # noqa: E402
    append_jsonl,
    compute_val_selection_score,
    default_run_dir,
    get_git_sha,
    probe_batch_size_and_accum,
    seed_everything,
    utc_now_iso,
    write_json,
)
from workbench.benchmarks.trainers.train_qure_upstream_ckpt import (  # noqa: E402
    compute_score_matrix_chunked,
    cosine_lr_schedule,
    encode_qure_images,
    encode_qure_queries,
    select_qure_hard_negatives_for_query,
)


@torch.no_grad()
def extract_qure_corpus_features(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_corpus, 32, 256] target features (CPU fp32), exact Blip2QuRe match."""
    model.eval()
    ds = CIRCorpusDataset(bundle, image_transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_corpus,
    )
    feats_list: List[torch.Tensor] = []
    ids: List[str] = []
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        with model.maybe_autocast():
            img_emb = model.ln_vision(model.visual_encoder(imgs))
        img_emb = img_emb.float()
        img_atts = torch.ones(img_emb.size()[:-1], dtype=torch.long, device=device)
        q_tok = model.query_tokens.expand(img_emb.shape[0], -1, -1)
        q_out = model.Qformer.bert(
            query_embeds=q_tok,
            encoder_hidden_states=img_emb,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        bf = F.normalize(model.vision_proj(q_out.last_hidden_state), dim=-1)
        feats_list.append(bf.float().cpu())
        ids.extend(batch["image_id"])
    return torch.cat(feats_list, dim=0), ids


@torch.no_grad()
def extract_qure_query_features(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_q, 256] query features (CPU fp32), exact Blip2QuRe match."""
    model.eval()
    ds = CIRQueryEvalDataset(bundle, image_transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_eval_queries,
    )
    feats_list: List[torch.Tensor] = []
    qids: List[str] = []
    for batch in dl:
        imgs = batch["ref_image"].to(device, non_blocking=True)
        captions = [txt_proc(c) for c in batch["raw_text"]]
        text_tokens = model.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        with model.maybe_autocast():
            ref_emb = model.ln_vision(model.visual_encoder(imgs))
        ref_emb = ref_emb.float()
        ref_atts = torch.ones(ref_emb.size()[:-1], dtype=torch.long, device=device)
        q_tok = model.query_tokens.expand(ref_emb.shape[0], -1, -1)
        q_atts = torch.ones(q_tok.size()[:-1], dtype=torch.long, device=device)
        attn_mask = torch.cat([q_atts, text_tokens.attention_mask], dim=1)
        q_out = model.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=q_tok,
            attention_mask=attn_mask,
            encoder_hidden_states=ref_emb,
            encoder_attention_mask=ref_atts,
            return_dict=True,
        )
        q_embeds = q_out.last_hidden_state[:, : q_tok.size(1), :]
        qf = F.normalize(model.text_proj(q_embeds).mean(dim=1), dim=-1)
        feats_list.append(qf.float().cpu())
        qids.extend(batch["query_id"])
    return torch.cat(feats_list, dim=0), qids


@torch.no_grad()
def rank_qure_multitoken(
    query_feats: torch.Tensor,
    query_ids: List[str],
    target_feats: torch.Tensor,
    corpus_ids: List[str],
    device: torch.device,
    top_k: int = 50,
    q_chunk: int = 128,
    c_chunk: int = 4096,
) -> Dict[str, List[str]]:
    """Exact max-over-32-tokens scoring (immune to OOM on 109k PinPoint corpus)."""
    if not torch.isfinite(query_feats).all() or not torch.isfinite(target_feats).all():
        raise ValueError("Non-finite features detected in rank_qure_multitoken")
    n_q = query_feats.shape[0]
    n_c = target_feats.shape[0]
    k = min(int(top_k), n_c)
    results: Dict[str, List[str]] = {}
    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        qf = query_feats[qs:qe].to(device, non_blocking=True)
        sims_row = np.empty((qe - qs, n_c), dtype=np.float32)
        for cs in range(0, n_c, c_chunk):
            ce = min(cs + c_chunk, n_c)
            tf = target_feats[cs:ce].to(device, non_blocking=True)
            sim_blk = torch.einsum("ik,jlk->ijl", qf, tf).max(dim=-1).values
            sims_row[:, cs:ce] = sim_blk.cpu().numpy()
        order = np.argsort(-sims_row, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[qs:qe]):
            results[str(qid)] = [str(corpus_ids[ci]) for ci in order[row_i]]
    return results


def evaluate_qure_split(
    dataset: str,
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 8,
    cached_corpus: Optional[Tuple[torch.Tensor, List[str]]] = None,
    top_k: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Tuple[torch.Tensor, List[str]]]:
    if cached_corpus is None:
        target_feats, corpus_ids = extract_qure_corpus_features(
            model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        target_feats, corpus_ids = cached_corpus
    query_feats, query_ids = extract_qure_query_features(
        model, bundle, preprocess, txt_proc, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_qure_multitoken(query_feats, query_ids, target_feats, corpus_ids, device=device, top_k=top_k)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (target_feats, corpus_ids)


class QuReCanonicalTrainDataset(Dataset):
    """Canonical EDIR/PinPoint training dataset with strictly train-only hard negative pool."""

    def __init__(self, bundle: SplitBundle, preprocess: Callable):
        self.bundle = bundle
        self.preprocess = preprocess
        self.queries = list(bundle.queries)
        self.id_to_path: Dict[str, Path] = {
            cid: rec.image_path(bundle.data_dir) for cid, rec in bundle.corpus_index.items()
        }
        # Strictly training-only target pool (all positive IDs appearing in train queries, NEVER val/test!)
        pool_set: List[str] = []
        seen = set()
        for q in self.queries:
            for pid in q.positive_ids:
                if pid not in self.id_to_path:
                    cand = bundle.resolve_image_path(pid)
                    if cand.is_file():
                        self.id_to_path[pid] = cand
                if pid in self.id_to_path and pid not in seen:
                    seen.add(pid)
                    pool_set.append(pid)
        self.train_target_pool: List[str] = pool_set
        self.use_hard_negatives = False
        self.hard_images: List[List[str]] = []
        self.negative_selection_history: Dict[int, List[str]] = {}
        self.epoch_positives: List[str] = []
        self.epoch_negatives: List[str] = []

    def prepare_epoch(self, rng: random.Random) -> None:
        pos_list: List[str] = []
        neg_list: List[str] = []
        for idx, q in enumerate(self.queries):
            valid_pos = [p for p in q.positive_ids if p in self.id_to_path]
            pos_id = rng.choice(valid_pos)
            pos_list.append(pos_id)
            forbidden = set(valid_pos)

            if not self.use_hard_negatives or not self.hard_images:
                neg_id = rng.choice(self.train_target_pool)
                while neg_id in forbidden:
                    neg_id = rng.choice(self.train_target_pool)
                neg_list.append(neg_id)
            else:
                hn = self.hard_images[idx]
                hist = self.negative_selection_history.setdefault(idx, [])
                if len(hist) >= len(hn):
                    hist.clear()
                avail = [n for n in hn if n not in hist]
                if not avail:
                    avail = hn
                neg_id = rng.choice(avail)
                hist.append(neg_id)
                neg_list.append(neg_id)
        self.epoch_positives = pos_list
        self.epoch_negatives = neg_list

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        q = self.queries[idx]
        with PIL.Image.open(q.ref_path(self.bundle.data_dir)) as im:
            ref_img = self.preprocess(im.convert("RGB"))
        with PIL.Image.open(self.id_to_path[self.epoch_positives[idx]]) as im:
            tar_img = self.preprocess(im.convert("RGB"))
        with PIL.Image.open(self.id_to_path[self.epoch_negatives[idx]]) as im:
            neg_img = self.preprocess(im.convert("RGB"))
        return ref_img, tar_img, neg_img, q.text


@torch.no_grad()
def rerank_canonical_train(
    model: nn.Module,
    train_ds: QuReCanonicalTrainDataset,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    consecutive_threshold: int = 1,
    eval_bs: int = 128,
    num_workers: int = 8,
) -> Dict[str, float]:
    """Exact QuRe hard-negative mining strictly within train_ds.train_target_pool."""
    t0 = time.time()
    model.eval()
    ref_paths = [q.ref_path(train_ds.bundle.data_dir) for q in train_ds.queries]
    captions = [q.text for q in train_ds.queries]
    q_feats = encode_qure_queries(model, ref_paths, captions, preprocess, txt_proc, device, batch_size=eval_bs, num_workers=num_workers)

    cand_names = train_ds.train_target_pool
    cand_paths = [train_ds.id_to_path[cid] for cid in cand_names]
    t_feats, _ = encode_qure_images(model, cand_paths, cand_names, preprocess, device, batch_size=eval_bs, num_workers=num_workers)

    scores = compute_score_matrix_chunked(q_feats, t_feats, device=device)
    cand_idx_map = {cid: i for i, cid in enumerate(cand_names)}

    hard_images: List[List[str]] = []
    fallbacks = 0
    for qi, q in enumerate(train_ds.queries):
        row = scores[qi]
        order = np.argsort(-row, kind="stable")
        sorted_scores = row[order]
        pos_ids = [p for p in q.positive_ids if p in cand_idx_map]
        pos_indices = [cand_idx_map[p] for p in pos_ids]
        hn, used_fb = select_qure_hard_negatives_for_query(
            sorted_indices=order,
            sorted_scores=sorted_scores,
            target_positions=pos_indices,
            candidate_names=cand_names,
            forbidden_names=set(pos_ids),
            consecutive_threshold=consecutive_threshold,
        )
        hard_images.append(hn)
        if used_fb:
            fallbacks += 1

    train_ds.hard_images = hard_images
    train_ds.negative_selection_history = {}
    train_ds.use_hard_negatives = True

    sizes = [len(h) for h in hard_images]
    avg_sz = float(np.mean(sizes))
    std_sz = float(np.std(sizes))
    dt = time.time() - t0
    print(
        f"  [QuRe Train Rerank] queries={len(hard_images)} train_pool={len(cand_names)} in {dt:.1f}s | "
        f"avg_hn={avg_sz:.1f} +/- {std_sz:.1f} | min={min(sizes)} max={max(sizes)} | fallbacks={fallbacks}",
        flush=True,
    )
    del q_feats, t_feats, scores
    gc.collect()
    torch.cuda.empty_cache()
    return {"avg_hn": round(avg_sz, 2), "std_hn": round(std_sz, 2), "fallbacks": fallbacks, "sec": round(dt, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QuRe (Blip2QuRe, pretrain_vitL) on EDIR or PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--backbone", type=str, default="pretrain_vitL", choices=["pretrain", "pretrain_vitL"])
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--negative-def-num", type=int, default=6)
    parser.add_argument("--consecutive-threshold", type=int, default=1)
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_workers = max(0, min(int(args.num_workers), 8))

    manifest = verify_split_integrity(args.dataset, split_tag=args.split_tag)
    train_bundle = load_split_bundle(args.dataset, "train", corpus_scope="full", split_tag=args.split_tag)
    val_bundle = load_split_bundle(args.dataset, "val", corpus_scope=args.val_corpus_scope, split_tag=args.split_tag)
    test_bundle = load_split_bundle(args.dataset, "test", corpus_scope="full", split_tag=args.split_tag)

    run_dir = args.run_dir or default_run_dir(args.dataset, "qure_vitl", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"

    # Load base pretrained BLIP-2 ViT-L ONLY (NEVER FashionIQ or CIRR checkpoint!)
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_qure", model_type=args.backbone, is_eval=False, device=device
    )
    txt_proc = txt_processors["eval"]
    preprocess = targetpad_transform(args.target_ratio, 224)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[QuRe] dataset={args.dataset} backbone={args.backbone} "
        f"trainable={trainable_params/1e6:.1f}M / total={total_params/1e6:.1f}M",
        flush=True,
    )

    # Parity self-check vs upstream Blip2QuRe.score()
    model.eval()
    with torch.no_grad():
        dq = F.normalize(torch.randn(4, 256), dim=-1)
        dt = F.normalize(torch.randn(9, 32, 256), dim=-1)
        ref_sim = model.score(dq, dt).numpy()
        our_sim = compute_score_matrix_chunked(dq, dt, device=device, q_chunk=2, c_chunk=4)
        max_diff = float(np.abs(ref_sim - our_sim).max())
        assert max_diff < 1e-5, f"QuRe scorer mismatch: {max_diff}"
        print(f"[QuRe Scorer Parity Self-Check] max_abs_diff vs model.score() = {max_diff:.2e} (OK)", flush=True)

    # VRAM auto-probe (target effective_bs=32, exact upstream batch_size=32)
    base_non_vit = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("visual_encoder.")
    }

    def probe_qure(bs: int) -> None:
        model.train()
        model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tar = torch.randn(bs, 3, 224, 224, device=device)
        d_neg = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = [txt_proc("change the shirt color to dark blue")] * bs
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            sc = model(d_ref, d_tar, d_txt, d_neg, True)
            lsm = F.log_softmax(sc, dim=1)
            tgt = torch.zeros_like(lsm)
            tgt[:, 0] = 1.0
            loss = F.kl_div(lsm, tgt, reduction="batchmean")
        loss.backward()
        model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_qure, default_effective_bs=32, allow_upgrade_to_32=True, max_micro_bs=32
    )
    model.load_state_dict(base_non_vit, strict=False)
    del base_non_vit
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[QuRe VRAM Probe] micro_bs={micro_bs} grad_accum={grad_accum}: {probe_log}", flush=True)

    if args.smoke_only:
        smoke_ds = QuReCanonicalTrainDataset(train_bundle, preprocess=preprocess)
        smoke_ds.prepare_epoch(random.Random(args.seed))
        r_im, t_im, n_im, cap = smoke_ds[0]
        assert r_im.shape == (3, 224, 224) and t_im.shape == (3, 224, 224) and n_im.shape == (3, 224, 224)
        print(
            f"[SMOKE PASS] train_qure.py ({args.dataset}) micro_bs={micro_bs} grad_accum={grad_accum} "
            f"train_q={len(smoke_ds)} train_pool={len(smoke_ds.train_target_pool)} cap={cap[:40]!r}",
            flush=True,
        )
        return

    if history_path.exists():
        history_path.unlink()

    # Stage 0: Zero-Shot pretrained BLIP-2 ViT-L baseline on val & test
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_qure_split(
        args.dataset, model, val_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_qure_split(
        args.dataset, model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    del zs_val_corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Stage 0 Zero-Shot BLIP-2 ViT-L] val {sel_name}={zs_val_score:.4f} | "
        f"test primary={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {
            "stage": "stage0_zeroshot",
            "epoch": 0,
            "val_selection_metric": sel_name,
            "val_selection_score": zs_val_score,
            "val_metrics": zs_val_metrics,
        },
    )

    train_ds = QuReCanonicalTrainDataset(train_bundle, preprocess=preprocess)
    # Leakage assertion: ensure zero val/test query IDs in train_ds
    train_qids = {q.query_id for q in train_ds.queries}
    val_qids = {q.query_id for q in val_bundle.queries}
    test_qids = {q.query_id for q in test_bundle.queries}
    assert train_qids.isdisjoint(val_qids) and train_qids.isdisjoint(test_qids), "Query split leak detected!"

    write_json(run_dir / "config.json", {
        "model_id": "qure_vitl",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "backbone": args.backbone,
            "learning_rate": args.learning_rate,
            "min_lr": args.learning_rate / 100.0,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "negative_def_num": args.negative_def_num,
            "consecutive_threshold": args.consecutive_threshold,
            "micro_bs": micro_bs,
            "grad_accum": grad_accum,
            "effective_bs": micro_bs * grad_accum,
            "target_ratio": args.target_ratio,
            "val_corpus_scope": args.val_corpus_scope,
            "seed": args.seed,
            "val_selection_metric": sel_name,
            "train_target_pool_size": len(train_ds.train_target_pool),
        },
        "vram_probe": probe_log,
    })

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(args.seed)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / "best_qure.pt"

    def save_trainable_ckpt(epoch: int, score: float) -> None:
        sd = model.state_dict()
        compact = {k: v.cpu() for k, v in sd.items() if not k.startswith("visual_encoder.")}
        torch.save({"epoch": epoch, "val_score": score, "state_dict_non_vit": compact}, best_ckpt_path)

    save_trainable_ckpt(0, best_val_score)
    no_improve = 0
    rerank_interval = max(1, args.num_epochs // args.negative_def_num)
    min_lr = args.learning_rate / 100.0

    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        zero_ep = epoch - 1
        rerank_info = None
        if zero_ep > 0 and (zero_ep % rerank_interval) == 0:
            rerank_info = rerank_canonical_train(
                model,
                train_ds,
                preprocess,
                txt_proc,
                device,
                consecutive_threshold=args.consecutive_threshold,
                eval_bs=args.eval_bs,
                num_workers=effective_workers,
            )

        train_ds.prepare_epoch(rng)
        train_loader = DataLoader(
            train_ds,
            batch_size=micro_bs,
            shuffle=True,
            drop_last=False,
            num_workers=effective_workers,
            pin_memory=True,
            prefetch_factor=2 if effective_workers > 0 else None,
            worker_init_fn=seed_worker if effective_workers > 0 else None,
            generator=gen,
        )

        cur_lr = cosine_lr_schedule(optimizer, zero_ep, args.num_epochs, args.learning_rate, min_lr)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        n_samples = 0
        steps_in_window = 0

        for step_idx, (ref_imgs, tar_imgs, neg_imgs, caps) in enumerate(train_loader):
            ref_imgs = ref_imgs.to(device, non_blocking=True)
            tar_imgs = tar_imgs.to(device, non_blocking=True)
            neg_imgs = neg_imgs.to(device, non_blocking=True)
            proc_caps = [txt_proc(c) for c in caps]
            bs_curr = ref_imgs.size(0)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                scores = model(ref_imgs, tar_imgs, proc_caps, neg_imgs, True)
                log_sm = F.log_softmax(scores, dim=1)
                tgt = torch.zeros_like(log_sm)
                tgt[:, 0] = 1.0
                loss = F.kl_div(log_sm, tgt, reduction="batchmean")

            l_val = float(loss.detach().cpu().item())
            if not np.isfinite(l_val):
                raise RuntimeError(f"Non-finite QuRe loss at epoch={epoch} step={step_idx}: {l_val}")

            scaler.scale(loss / grad_accum).backward()
            loss_sum += l_val * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_in_window = 0

        avg_loss = loss_sum / max(n_samples, 1)
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        temp_val = float(model.temp.detach().cpu().item())

        val_metrics, _, _ = evaluate_qure_split(
            args.dataset, model, val_bundle, preprocess, txt_proc, device,
            eval_bs=args.eval_bs, num_workers=effective_workers,
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_trainable_ckpt(epoch, val_score)
        else:
            no_improve += 1

        print(
            f"[QuRe Ep {epoch:02d}/{args.num_epochs}] loss={avg_loss:.4f} lr={cur_lr:.2e} temp={temp_val:.4f} | "
            f"val {sel_name}={val_score:.4f} (best={best_val_score:.4f}@ep{best_epoch}) | "
            f"vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "train",
                "epoch": epoch,
                "lr": cur_lr,
                "temp": round(temp_val, 6),
                "rerank": rerank_info,
                "train_losses": {"total": round(avg_loss, 6)},
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_val_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        # Allow at least 1 hard-negative rerank cycle before early-stopping!
        if epoch > rerank_interval + 1 and no_improve >= args.patience:
            print(f"[QuRe] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Final evaluation ONCE on test using best val checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    msg = model.load_state_dict(ckpt["state_dict_non_vit"], strict=False)
    non_vit_missing = [k for k in msg.missing_keys if not k.startswith("visual_encoder.")]
    assert not non_vit_missing, f"Unexpected missing non-ViT keys: {non_vit_missing}"
    assert not msg.unexpected_keys, f"Unexpected keys in checkpoint: {msg.unexpected_keys}"

    final_test_metrics, final_test_preds, _ = evaluate_qure_split(
        args.dataset, model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "qure_vitl",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_zeroshot_test": zs_test_metrics,
        "final_qure_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== QuRe FINAL TEST SUMMARY ===", flush=True)
    print(f"  Zero-Shot BLIP-2 Test : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final QuRe Test       : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
