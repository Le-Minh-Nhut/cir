#!/usr/bin/env python3
"""Upstream-faithful CLIP4Cir (Stage 1 CLIP RN50x4 fine-tune + Stage 2 Combiner) trainer
for EDIR and PinPoint canonical splits.

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/CLIP4Cir/src \
    workbench/envs/clip4cir310-sm120-pep610/bin/python \
    workbench/benchmarks/trainers/train_clip4cir.py --dataset edir
"""

from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path("/home/heheboiz/data/cir")
CLIP4CIR_SRC = REPO_ROOT / "workbench" / "third_party" / "CLIP4Cir" / "src"
for p in (str(REPO_ROOT), str(CLIP4CIR_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import clip  # noqa: E402
from combiner import Combiner  # noqa: E402
from data_utils import targetpad_transform  # noqa: E402

from workbench.benchmarks import (  # noqa: E402
    CIRCorpusDataset,
    CIRQueryEvalDataset,
    CIRTrainingDataset,
    collate_corpus,
    collate_eval_queries,
    collate_train_batch,
    evaluate_edir,
    evaluate_pinpoint,
    load_split_bundle,
    rank_embeddings,
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


def element_wise_sum(image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    return F.normalize(image_features + text_features, dim=-1)


@torch.no_grad()
def extract_corpus_embeddings(
    clip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    clip_model.eval().float()
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
    feats_list: List[np.ndarray] = []
    ids: List[str] = []
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            f = clip_model.encode_image(imgs)
        f = F.normalize(f.float(), dim=-1)
        feats_list.append(f.cpu().numpy())
        ids.extend(batch["image_id"])
    return np.concatenate(feats_list, axis=0), ids


@torch.no_grad()
def extract_query_embeddings(
    clip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    combining_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 4,
) -> Tuple[np.ndarray, List[str]]:
    clip_model.eval().float()
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
    feats_list: List[np.ndarray] = []
    qids: List[str] = []
    for batch in dl:
        imgs = batch["ref_image"].to(device, non_blocking=True)
        tokens = clip.tokenize(batch["raw_text"], context_length=77, truncate=True).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            img_f = clip_model.encode_image(imgs)
            txt_f = clip_model.encode_text(tokens)
        pred_f = combining_fn(img_f.float(), txt_f.float())
        pred_f = F.normalize(pred_f.float(), dim=-1)
        feats_list.append(pred_f.cpu().numpy())
        qids.extend(batch["query_id"])
    return np.concatenate(feats_list, axis=0), qids


def evaluate_split(
    dataset: str,
    clip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    combining_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 4,
    cached_corpus: Optional[Tuple[np.ndarray, List[str]]] = None,
    top_k: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Tuple[np.ndarray, List[str]]]:
    if cached_corpus is None:
        corpus_emb, corpus_ids = extract_corpus_embeddings(
            clip_model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        corpus_emb, corpus_ids = cached_corpus
    query_emb, query_ids = extract_query_embeddings(
        clip_model, bundle, preprocess, combining_fn, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_embeddings(
        query_emb, query_ids, corpus_emb, corpus_ids, top_k=top_k, l2_normalize=True
    )
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (corpus_emb, corpus_ids)


@torch.no_grad()
def extract_train_triplet_features(
    clip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 4,
    positive_mode: str = "expand",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-extract frozen Stage-1 CLIP features for deterministic TargetPad preprocessing.

    Because TargetPad has zero random augmentation and CLIP is strictly frozen during
    Stage-2 Combiner training, extracting once per unique image/text is mathematically
    equivalent to re-running frozen CLIP every epoch while being ~50x faster.
    """
    clip_model.eval().float()
    ds = CIRTrainingDataset(
        bundle, image_transform=preprocess, positive_mode=positive_mode, load_images=True, seed=42
    )
    dl = DataLoader(
        ds,
        batch_size=min(eval_bs, 64),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_train_batch,
    )
    ref_list, txt_list, tgt_list = [], [], []
    for batch in dl:
        ref_imgs = batch["ref_image"].to(device, non_blocking=True)
        tgt_imgs = batch["target_image"].to(device, non_blocking=True)
        tokens = clip.tokenize(batch["raw_text"], context_length=77, truncate=True).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            rf = clip_model.encode_image(ref_imgs).float()
            tf = clip_model.encode_text(tokens).float()
            gf = clip_model.encode_image(tgt_imgs).float()
        ref_list.append(rf.cpu())
        txt_list.append(tf.cpu())
        tgt_list.append(gf.cpu())
    return torch.cat(ref_list, dim=0), torch.cat(txt_list, dim=0), torch.cat(tgt_list, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP4Cir (Stage 1 CLIP + Stage 2 Combiner) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--clip-model-name", type=str, default="RN50x4")
    parser.add_argument("--clip-lr", type=float, default=2e-6)
    parser.add_argument("--clip-epochs", type=int, default=25)
    parser.add_argument("--clip-patience", type=int, default=5)
    parser.add_argument("--combiner-lr", type=float, default=2e-5)
    parser.add_argument("--combiner-epochs", type=int, default=30)
    parser.add_argument("--combiner-patience", type=int, default=6)
    parser.add_argument("--combiner-bs", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=640 * 4)
    parser.add_argument("--hidden-dim", type=int, default=640 * 8)
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--positive-mode", type=str, default="random", choices=["random", "expand"])
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_workers = max(0, min(args.num_workers, 16))

    # 1. Verify split integrity & load bundles (val uses configurable scope; test ALWAYS full corpus!)
    manifest = verify_split_integrity(args.dataset, split_tag=args.split_tag)
    train_bundle = load_split_bundle(args.dataset, "train", corpus_scope="full", split_tag=args.split_tag)
    val_bundle = load_split_bundle(args.dataset, "val", corpus_scope=args.val_corpus_scope, split_tag=args.split_tag)
    test_bundle = load_split_bundle(args.dataset, "test", corpus_scope="full", split_tag=args.split_tag)

    run_dir = args.run_dir or default_run_dir(args.dataset, f"clip4cir_{args.clip_model_name.lower()}", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    # 2. Load base pretrained OpenAI CLIP (NEVER FashionIQ/CIRR checkpoint)
    clip_model, _ = clip.load(args.clip_model_name, device=device, jit=False)
    # Upstream CLIP4Cir keeps CLIP in eval().float() even during fine-tuning (preserves RN50x4 BN stats)
    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim
    preprocess = targetpad_transform(args.target_ratio, input_dim)

    print(
        f"[CLIP4Cir] dataset={args.dataset} model={args.clip_model_name} "
        f"input_dim={input_dim} feature_dim={feature_dim} device={device}",
        flush=True,
    )

    # 3. Stage 0: Zero-shot pretrained CLIP baseline (Before training) on val & test
    t0 = time.time()
    zs_val_metrics, _, cached_val_corpus = evaluate_split(
        args.dataset, clip_model, val_bundle, preprocess, element_wise_sum, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_split(
        args.dataset, clip_model, test_bundle, preprocess, element_wise_sum, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=cached_val_corpus if args.val_corpus_scope == "full" else None,
    )
    print(
        f"[Stage 0 Zero-Shot CLIP] val {sel_name}={zs_val_score:.4f} | "
        f"test R@1={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
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

    # 4. Probe VRAM for Stage 1 CLIP fine-tuning (default 16 -> upgrade 32 if headroom -> fallback grad_accum)
    criterion = nn.CrossEntropyLoss()
    base_state = copy.deepcopy(clip_model.state_dict())

    def probe_stage1(bs: int) -> None:
        clip_model.zero_grad(set_to_none=True)
        dummy_ref = torch.randn(bs, 3, input_dim, input_dim, device=device)
        dummy_tgt = torch.randn(bs, 3, input_dim, input_dim, device=device)
        dummy_txt = clip.tokenize(["a photo with a red hat"] * bs, context_length=77, truncate=True).to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            rf = clip_model.encode_image(dummy_ref)
            tf = clip_model.encode_text(dummy_txt)
            pf = element_wise_sum(rf, tf)
            gf = F.normalize(clip_model.encode_image(dummy_tgt), dim=-1)
            logits = 100.0 * pf @ gf.T
            loss = criterion(logits, torch.arange(bs, dtype=torch.long, device=device))
        loss.backward()
        clip_model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_stage1, default_effective_bs=16, allow_upgrade_to_32=True
    )
    clip_model.load_state_dict(base_state)
    del base_state
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Stage 1 VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": f"clip4cir_{args.clip_model_name.lower()}",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "clip_model_name": args.clip_model_name,
            "clip_lr": args.clip_lr,
            "clip_epochs": args.clip_epochs,
            "clip_patience": args.clip_patience,
            "stage1_micro_bs": micro_bs,
            "stage1_grad_accum": grad_accum,
            "stage1_effective_bs": micro_bs * grad_accum,
            "combiner_lr": args.combiner_lr,
            "combiner_epochs": args.combiner_epochs,
            "combiner_patience": args.combiner_patience,
            "combiner_bs": args.combiner_bs,
            "projection_dim": args.projection_dim,
            "hidden_dim": args.hidden_dim,
            "target_ratio": args.target_ratio,
            "positive_mode": args.positive_mode,
            "val_corpus_scope": args.val_corpus_scope,
            "seed": args.seed,
            "val_selection_metric": sel_name,
        },
        "vram_probe": probe_log,
    }
    write_json(run_dir / "config.json", config_payload)

    # 5. Stage 1: Fine-tune CLIP (both encoders, exact upstream clip_fine_tune.py)
    train_ds = CIRTrainingDataset(
        train_bundle,
        image_transform=preprocess,
        positive_mode=args.positive_mode,
        load_images=True,
        seed=args.seed,
    )
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_bs,
        shuffle=True,
        drop_last=True,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=2 if effective_workers > 0 else None,
        collate_fn=collate_train_batch,
        worker_init_fn=seed_worker if effective_workers > 0 else None,
        generator=gen,
    )

    optimizer = optim.AdamW(
        [{"params": [p for p in clip_model.parameters() if p.requires_grad], "lr": args.clip_lr, "betas": (0.9, 0.999), "eps": 1e-7}]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_s1_score = zs_val_score
    best_s1_epoch = 0
    best_s1_path = run_dir / "best_stage1_clip.pt"
    torch.save({"epoch": 0, "CLIP": clip_model.state_dict(), "val_score": best_s1_score}, best_s1_path)
    no_improve = 0

    print(f"[Stage 1] Starting CLIP fine-tune: epochs={args.clip_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.clip_epochs + 1):
        ep_t0 = time.time()
        clip_model.eval().float()  # Upstream CLIP4Cir keeps BN in eval mode
        train_ds.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_samples = 0
        steps_in_window = 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            tokens = clip.tokenize(batch["raw_text"], context_length=77, truncate=True).to(device, non_blocking=True)
            bs_curr = ref_imgs.size(0)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                ref_f = clip_model.encode_image(ref_imgs)
                txt_f = clip_model.encode_text(tokens)
                pred_f = element_wise_sum(ref_f, txt_f)
                tgt_f = F.normalize(clip_model.encode_image(tgt_imgs), dim=-1)
                logits = 100.0 * pred_f @ tgt_f.T
                gt = torch.arange(bs_curr, dtype=torch.long, device=device)
                loss = criterion(logits, gt)

            loss_val = float(loss.detach().cpu().item())
            if not np.isfinite(loss_val):
                raise RuntimeError(f"Non-finite loss in Stage 1 epoch={epoch} step={step_idx}: {loss_val}")

            scaler.scale(loss / grad_accum).backward()
            running_loss += loss_val * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_in_window = 0

        train_loss = running_loss / max(n_samples, 1)
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

        val_metrics, _, _ = evaluate_split(
            args.dataset, clip_model, val_bundle, preprocess, element_wise_sum, device,
            eval_bs=args.eval_bs, num_workers=effective_workers,
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_s1_score
        if improved:
            best_s1_score = val_score
            best_s1_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "CLIP": clip_model.state_dict(), "val_score": val_score}, best_s1_path)
        else:
            no_improve += 1

        print(
            f"[Stage 1 Ep {epoch:02d}/{args.clip_epochs}] loss={train_loss:.4f} | "
            f"val {sel_name}={val_score:.4f} (best={best_s1_score:.4f}@ep{best_s1_epoch}) | "
            f"vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "stage1_clip_finetune",
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_s1_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve >= args.clip_patience:
            print(f"[Stage 1] Early stopping at epoch {epoch} (patience={args.clip_patience})", flush=True)
            break

    # Free Stage-1 training loader & optimizer state before full-corpus test eval & Stage 2
    del train_loader, train_ds, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    # Load best Stage 1 CLIP weights and evaluate Stage 1 on test (for clean ablation!)
    s1_ckpt = torch.load(best_s1_path, map_location=device)
    clip_model.load_state_dict(s1_ckpt["CLIP"])
    clip_model.eval().float()
    for p in clip_model.parameters():
        p.requires_grad = False

    s1_val_metrics, _, s1_val_corpus = evaluate_split(
        args.dataset, clip_model, val_bundle, preprocess, element_wise_sum, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    s1_test_metrics, _, s1_full_corpus = evaluate_split(
        args.dataset, clip_model, test_bundle, preprocess, element_wise_sum, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=s1_val_corpus if args.val_corpus_scope == "full" else None,
    )

    # 6. Stage 2: Train Combiner on frozen best Stage 1 CLIP features (exact upstream combiner_train.py)
    print("[Stage 2] Extracting frozen Stage-1 CLIP features for Combiner training...", flush=True)
    # Always expand all clean (query, pos) pairs for Stage-2 Combiner since CLIP is frozen
    comb_pos_mode = "expand"
    ref_feats_tr, txt_feats_tr, tgt_feats_tr = extract_train_triplet_features(
        clip_model, train_bundle, preprocess, device, eval_bs=args.eval_bs, num_workers=effective_workers,
        positive_mode=comb_pos_mode,
    )
    # Also pre-extract val query raw (ref_f, txt_f) once since CLIP is frozen in Stage 2!
    val_q_ds = CIRQueryEvalDataset(val_bundle, image_transform=preprocess)
    val_q_dl = DataLoader(val_q_ds, batch_size=args.eval_bs, shuffle=False, num_workers=effective_workers, pin_memory=True, collate_fn=collate_eval_queries)
    val_rf_list, val_tf_list, val_qids = [], [], []
    with torch.no_grad():
        for batch in val_q_dl:
            imgs = batch["ref_image"].to(device, non_blocking=True)
            tokens = clip.tokenize(batch["raw_text"], context_length=77, truncate=True).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                val_rf_list.append(clip_model.encode_image(imgs).float().cpu())
                val_tf_list.append(clip_model.encode_text(tokens).float().cpu())
            val_qids.extend(batch["query_id"])
    val_rf_all = torch.cat(val_rf_list, dim=0)
    val_tf_all = torch.cat(val_tf_list, dim=0)

    combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device)
    comb_opt = optim.Adam(combiner.parameters(), lr=args.combiner_lr)
    comb_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    feat_ds = TensorDataset(ref_feats_tr, txt_feats_tr, tgt_feats_tr)
    comb_gen = torch.Generator()
    comb_gen.manual_seed(args.seed + 100)
    comb_bs = min(args.combiner_bs, len(feat_ds))
    feat_loader = DataLoader(feat_ds, batch_size=comb_bs, shuffle=True, drop_last=True, generator=comb_gen)

    best_s2_score = -1.0
    best_s2_epoch = 0
    best_s2_path = run_dir / "best_stage2_combiner.pt"
    no_improve_s2 = 0

    print(f"[Stage 2] Training Combiner: epochs={args.combiner_epochs}, bs={comb_bs}, batches/ep={len(feat_loader)}", flush=True)
    for epoch in range(1, args.combiner_epochs + 1):
        ep_t0 = time.time()
        combiner.train()
        running_loss = 0.0
        n_samples = 0
        for rf, tf, gf in feat_loader:
            rf = rf.to(device, non_blocking=True)
            tf = tf.to(device, non_blocking=True)
            gf = gf.to(device, non_blocking=True)
            bs_curr = rf.size(0)
            comb_opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = combiner(rf, tf, gf)
                gt = torch.arange(bs_curr, dtype=torch.long, device=device)
                loss = criterion(logits, gt)
            comb_scaler.scale(loss).backward()
            comb_scaler.step(comb_opt)
            comb_scaler.update()
            running_loss += float(loss.detach().cpu().item()) * bs_curr
            n_samples += bs_curr

        train_loss = running_loss / max(n_samples, 1)
        combiner.eval()
        with torch.no_grad():
            pred_chunks = []
            for start in range(0, len(val_qids), args.eval_bs):
                end = min(start + args.eval_bs, len(val_qids))
                pf = combiner.combine_features(val_rf_all[start:end].to(device), val_tf_all[start:end].to(device))
                pred_chunks.append(pf.float().cpu().numpy())
            val_pred_emb = np.concatenate(pred_chunks, axis=0)
        val_preds = rank_embeddings(val_pred_emb, val_qids, s1_val_corpus[0], s1_val_corpus[1], top_k=50, l2_normalize=True)
        val_metrics = (
            evaluate_edir(val_preds, val_bundle.queries, ks=(1, 5, 10, 50), strict=True)
            if args.dataset == "edir"
            else evaluate_pinpoint(val_preds, val_bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_s2_score
        if improved:
            best_s2_score = val_score
            best_s2_epoch = epoch
            no_improve_s2 = 0
            torch.save({"epoch": epoch, "Combiner": combiner.state_dict(), "val_score": val_score}, best_s2_path)
        else:
            no_improve_s2 += 1

        print(
            f"[Stage 2 Ep {epoch:02d}/{args.combiner_epochs}] loss={train_loss:.4f} | "
            f"val {sel_name}={val_score:.4f} (best={best_s2_score:.4f}@ep{best_s2_epoch}) | "
            f"{dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "stage2_combiner",
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_s2_score,
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve_s2 >= args.combiner_patience:
            print(f"[Stage 2] Early stopping at epoch {epoch} (patience={args.combiner_patience})", flush=True)
            break

    # 7. Final evaluation ONCE on test using best Stage-2 Combiner + best Stage-1 CLIP
    s2_ckpt = torch.load(best_s2_path, map_location=device)
    combiner.load_state_dict(s2_ckpt["Combiner"])
    combiner.eval()

    final_test_metrics, final_test_preds, _ = evaluate_split(
        args.dataset, clip_model, test_bundle, preprocess, combiner.combine_features, device,
        eval_bs=args.eval_bs, num_workers=effective_workers, cached_corpus=s1_full_corpus,
    )

    result_payload = {
        "model_id": f"clip4cir_{args.clip_model_name.lower()}",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_stage1_epoch": best_s1_epoch,
        "best_stage1_val_score": best_s1_score,
        "best_stage2_epoch": best_s2_epoch,
        "best_stage2_val_score": best_s2_score,
        "stage0_zeroshot_test": zs_test_metrics,
        "stage1_clip_finetuned_test": s1_test_metrics,
        "final_clip4cir_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== CLIP4Cir FINAL TEST SUMMARY ===", flush=True)
    print(f"  Zero-Shot CLIP Test : {zs_test_metrics['overall']}", flush=True)
    print(f"  Stage-1 CLIP Test   : {s1_test_metrics['overall']}", flush=True)
    print(f"  Final CLIP4Cir Test : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
