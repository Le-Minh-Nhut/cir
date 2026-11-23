#!/usr/bin/env python3
"""Upstream-faithful ENCODER (ViT-B-32 LAION-2B + Entity/Action Binding) trainer
for EDIR and PinPoint canonical splits.

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/ENCODER \
    workbench/envs/encoder310-adapted/bin/python \
    workbench/benchmarks/trainers/train_encoder.py --dataset edir
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/heheboiz/data/cir")
ENCODER_DIR = REPO_ROOT / "workbench" / "third_party" / "ENCODER"
for p in (str(REPO_ROOT), str(ENCODER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import open_clip  # noqa: E402
import model_try2  # noqa: E402

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


def set_bn_eval(m: nn.Module) -> None:
    if m.__class__.__name__.find("BatchNorm") != -1:
        m.eval()


def build_encoder_model(
    hidden_dim: int = 512,
    dropout: float = 0.5,
    local_token_num: int = 1,
    wc: int = 2,
    n_p: int = 1,
    tau_: float = 0.1,
    weighted: bool = True,
    device: torch.device = torch.device("cuda"),
) -> Tuple[nn.Module, Callable, Callable]:
    """Instantiate upstream ENCODER from base LAION-2B ViT-B-32 weights (NO FashionIQ/CIRR ckpt)."""
    cwd_before = os.getcwd()
    try:
        os.chdir(str(ENCODER_DIR))
        _, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(ENCODER_DIR / "open_clip_pytorch_model.bin")
        )
        model = model_try2.Encoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            local_token_num=local_token_num,
            t=tau_,
            wc=wc,
            N_p=n_p,
            weighted=weighted,
        ).to(device)
    finally:
        os.chdir(cwd_before)

    # Upstream bug note (model_try2.py:93): BindingDecoder.learnable_relation is a plain CUDA Tensor
    # (due to nn.Parameter(...).cuda()), so it is NOT in state_dict(). Save/restore it explicitly
    # so reloading best_model.pt reproduces the exact same learnable_relation tensor!
    return model, train_preprocess, val_preprocess


def get_encoder_full_state(model: nn.Module) -> Dict[str, Any]:
    state = {
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "binding_decoder") and hasattr(backbone.binding_decoder, "learnable_relation"):
        state["learnable_relation"] = backbone.binding_decoder.learnable_relation.detach().cpu().clone()
    return state


def load_encoder_full_state(model: nn.Module, state: Dict[str, Any], device: torch.device) -> None:
    model.load_state_dict(state["model_state_dict"], strict=True)
    backbone = getattr(model, "backbone", model)
    if "learnable_relation" in state and hasattr(backbone, "binding_decoder"):
        backbone.binding_decoder.learnable_relation = state["learnable_relation"].to(device)


@torch.no_grad()
def extract_corpus_embeddings(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    ds = CIRCorpusDataset(bundle, image_transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_corpus,
    )
    feats_list: List[np.ndarray] = []
    ids: List[str] = []
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            f = model.extract_retrieval_target(imgs)
        feats_list.append(f.float().cpu().numpy())
        ids.extend(batch["image_id"])
    return np.concatenate(feats_list, axis=0), ids


@torch.no_grad()
def extract_query_embeddings(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    ds = CIRQueryEvalDataset(bundle, image_transform=preprocess)
    dl = DataLoader(
        ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_eval_queries,
    )
    feats_list: List[np.ndarray] = []
    qids: List[str] = []
    for batch in dl:
        imgs = batch["ref_image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            f = model.extract_retrieval_compose(imgs, batch["raw_text"])
        feats_list.append(f.float().cpu().numpy())
        qids.extend(batch["query_id"])
    return np.concatenate(feats_list, axis=0), qids


def evaluate_encoder_split(
    dataset: str,
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 16,
    cached_corpus: Optional[Tuple[np.ndarray, List[str]]] = None,
    top_k: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Tuple[np.ndarray, List[str]]]:
    if cached_corpus is None:
        corpus_emb, corpus_ids = extract_corpus_embeddings(
            model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        corpus_emb, corpus_ids = cached_corpus
    query_emb, query_ids = extract_query_embeddings(
        model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_embeddings(query_emb, query_ids, corpus_emb, corpus_ids, top_k=top_k, l2_normalize=True)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (corpus_emb, corpus_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ENCODER (ViT-B-32) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--clip-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-decay", type=int, default=5)
    parser.add_argument("--lr-div", type=float, default=0.1)
    parser.add_argument("--max-decay-epoch", type=int, default=10)
    parser.add_argument("--local-token-num", type=int, default=1)
    parser.add_argument("--wc", type=int, default=2)
    parser.add_argument("--n-p", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
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

    manifest = verify_split_integrity(args.dataset, split_tag=args.split_tag)
    train_bundle = load_split_bundle(args.dataset, "train", corpus_scope="full", split_tag=args.split_tag)
    val_bundle = load_split_bundle(args.dataset, "val", corpus_scope=args.val_corpus_scope, split_tag=args.split_tag)
    test_bundle = load_split_bundle(args.dataset, "test", corpus_scope="full", split_tag=args.split_tag)

    run_dir = args.run_dir or default_run_dir(args.dataset, "encoder_vitb32", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    model, train_preprocess, val_preprocess = build_encoder_model(
        local_token_num=args.local_token_num,
        wc=args.wc,
        n_p=args.n_p,
        tau_=args.tau,
        device=device,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ENCODER] dataset={args.dataset} backbone=open_clip/ViT-B-32 trainable={trainable/1e6:.1f}M", flush=True)

    # Stage 0 baseline (before training)
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_encoder_split(
        args.dataset, model, val_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_encoder_split(
        args.dataset, model, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    print(
        f"[Stage 0 ENCODER Init] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_init", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    # VRAM auto-probe
    base_state = get_encoder_full_state(model)

    def probe_step(bs: int) -> None:
        model.train()
        model.apply(set_bn_eval)
        model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = ["change the background to a sunny beach"] * bs
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss_dict = model.compute_loss(d_ref, d_txt, d_tgt)
            loss = loss_dict["stu_rank"] + args.kappa * loss_dict["kl"] + loss_dict["entity"] + loss_dict["ortho"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=64
    )
    load_encoder_full_state(model, base_state, device)
    del base_state
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[ENCODER VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": "encoder_vitb32",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "lr": args.lr,
            "clip_lr": args.clip_lr,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "lr_div": args.lr_div,
            "max_decay_epoch": args.max_decay_epoch,
            "local_token_num": args.local_token_num,
            "wc": args.wc,
            "n_p": args.n_p,
            "tau": args.tau,
            "kappa": args.kappa,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "micro_bs": micro_bs,
            "grad_accum": grad_accum,
            "effective_bs": micro_bs * grad_accum,
            "positive_mode": args.positive_mode,
            "val_corpus_scope": args.val_corpus_scope,
            "seed": args.seed,
            "val_selection_metric": sel_name,
        },
        "vram_probe": probe_log,
    }
    write_json(run_dir / "config.json", config_payload)

    # DataLoader & exact upstream AdamW dual-LR optimizer
    train_ds = CIRTrainingDataset(
        train_bundle, image_transform=train_preprocess, positive_mode=args.positive_mode, load_images=True, seed=args.seed
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
        prefetch_factor=4 if effective_workers > 0 else None,
        collate_fn=collate_train_batch,
        worker_init_fn=seed_worker if effective_workers > 0 else None,
        generator=gen,
    )

    clip_params = [p for n, p in model.named_parameters() if "clip" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "clip" not in n and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": clip_params, "lr": args.clip_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / "best_encoder.pt"
    torch.save({"epoch": 0, "val_score": best_val_score, **get_encoder_full_state(model)}, best_ckpt_path)
    no_improve = 0

    print(f"[ENCODER] Starting training: epochs={args.num_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        model.apply(set_bn_eval)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        loss_sums = {"total": 0.0, "stu_rank": 0.0, "kl": 0.0, "entity": 0.0, "ortho": 0.0}
        n_samples = 0
        steps_in_window = 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            captions = batch["raw_text"]
            bs_curr = ref_imgs.size(0)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss_dict = model.compute_loss(ref_imgs, captions, tgt_imgs)
                loss = (
                    loss_dict["stu_rank"]
                    + args.kappa * loss_dict["kl"]
                    + loss_dict["entity"]
                    + loss_dict["ortho"]
                )

            l_val = float(loss.detach().cpu().item())
            if not np.isfinite(l_val):
                raise RuntimeError(f"Non-finite ENCODER loss at epoch={epoch} step={step_idx}: {loss_dict}")

            scaler.scale(loss / grad_accum).backward()
            loss_sums["total"] += l_val * bs_curr
            for lk in ("stu_rank", "kl", "entity", "ortho"):
                loss_sums[lk] += float(loss_dict[lk].detach().cpu().item()) * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_in_window = 0

        # Exact upstream step LR decay (train.py:171-173)
        if epoch % args.lr_decay == 0 and (epoch - 1) < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g["lr"] *= args.lr_div

        avg_losses = {k: round(v / max(n_samples, 1), 6) for k, v in loss_sums.items()}
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        val_metrics, _, _ = evaluate_encoder_split(
            args.dataset, model, val_bundle, val_preprocess, device,
            eval_bs=args.eval_bs, num_workers=effective_workers,
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "val_score": val_score, **get_encoder_full_state(model)}, best_ckpt_path)
        else:
            no_improve += 1

        print(
            f"[ENCODER Ep {epoch:02d}/{args.num_epochs}] loss={avg_losses['total']:.4f} "
            f"(rank={avg_losses['stu_rank']:.3f}, kl={avg_losses['kl']:.3f}, ent={avg_losses['entity']:.3f}) | "
            f"lr_clip={optimizer.param_groups[0]['lr']:.1e} | val {sel_name}={val_score:.4f} "
            f"(best={best_val_score:.4f}@ep{best_epoch}) | vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "train",
                "epoch": epoch,
                "clip_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "train_losses": avg_losses,
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_val_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve >= args.patience:
            print(f"[ENCODER] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Reload best checkpoint and evaluate ONCE on full test corpus
    ckpt = torch.load(best_ckpt_path, map_location=device)
    load_encoder_full_state(model, ckpt, device)
    final_test_metrics, final_test_preds, _ = evaluate_encoder_split(
        args.dataset, model, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "encoder_vitb32",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_init_test": zs_test_metrics,
        "final_encoder_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== ENCODER FINAL TEST SUMMARY ===", flush=True)
    print(f"  Init ENCODER Test  : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final ENCODER Test : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
