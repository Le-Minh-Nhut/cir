#!/usr/bin/env python3
"""Upstream-faithful CLVC-Net (dual ResNet50 + LSTM local/global mutual learning) trainer
for EDIR and PinPoint canonical splits.

Builds SimpleVocab strictly from the training split captions only (zero val/test leak)
and runs in FP32 to avoid GeM (p=3) fp16 underflow NaNs.

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/CLVCNet \
    workbench/envs/clvc39-la/bin/python \
    workbench/benchmarks/trainers/train_clvcnet.py --dataset edir
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path("/home/heheboiz/data/cir")
CLVC_DIR = REPO_ROOT / "workbench" / "third_party" / "CLVCNet"
for p in (str(REPO_ROOT), str(CLVC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as clvc_model_mod  # noqa: E402
import text_model as clvc_text_mod  # noqa: E402

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


_VOCAB_HELPER = clvc_text_mod.SimpleVocab()


def _safe_caption(text: str) -> str:
    """Guard against empty/punctuation-only strings crashing CLVCNet TextLSTMModel."""
    toks = _VOCAB_HELPER.tokenize_text(text)
    return text.strip() if toks else "unknown"


@torch.no_grad()
def extract_corpus_embeddings(
    local_model: nn.Module,
    global_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    local_model.eval()
    global_model.eval()
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
        loc_f = F.normalize(local_model.extract_img_feature(imgs)[1].float(), dim=-1)
        glb_f = F.normalize(global_model.extract_img_feature(imgs)[1].float(), dim=-1)
        cat_f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
        feats_list.append(cat_f.cpu().numpy())
        ids.extend(batch["image_id"])
    return np.concatenate(feats_list, axis=0), ids


@torch.no_grad()
def extract_query_embeddings(
    local_model: nn.Module,
    global_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 16,
) -> Tuple[np.ndarray, List[str]]:
    local_model.eval()
    global_model.eval()
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
        mods = [_safe_caption(c) for c in batch["raw_text"]]
        # Upstream CLVCNet/test.py:100: local[1], global[0]
        loc_f = F.normalize(local_model.compose_img_text(imgs, mods)[1].float(), dim=-1)
        glb_f = F.normalize(global_model.compose_img_text(imgs, mods)[0].float(), dim=-1)
        cat_f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
        feats_list.append(cat_f.cpu().numpy())
        qids.extend(batch["query_id"])
    return np.concatenate(feats_list, axis=0), qids


def evaluate_clvc_split(
    dataset: str,
    local_model: nn.Module,
    global_model: nn.Module,
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
            local_model, global_model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        corpus_emb, corpus_ids = cached_corpus
    query_emb, query_ids = extract_query_embeddings(
        local_model, global_model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_embeddings(query_emb, query_ids, corpus_emb, corpus_ids, top_k=top_k, l2_normalize=True)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (corpus_emb, corpus_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLVC-Net (ResNet50+LSTM) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-decay", type=int, default=10)
    parser.add_argument("--lr-div", type=float, default=0.1)
    parser.add_argument("--max-decay-epoch", type=int, default=20)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--img-weight", type=float, default=1.0)
    parser.add_argument("--mul-kl-weight", type=float, default=1.0)
    parser.add_argument("--local-t", type=float, default=7.0)
    parser.add_argument("--global-t", type=float, default=4.0)
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

    run_dir = args.run_dir or default_run_dir(args.dataset, "clvcnet_resnet50", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    train_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Build vocab STRICTLY from training captions only (zero val/test leak!)
    train_texts = [_safe_caption(q.text) for q in train_bundle.queries]
    local_model = clvc_model_mod.compose_local(texts=train_texts, T=args.local_t).to(device)
    global_model = clvc_model_mod.compose_global(texts=train_texts, T=args.global_t).to(device)

    vocab_size = local_model.text_model.vocab.get_size()
    trainable = sum(p.numel() for p in local_model.parameters() if p.requires_grad) + sum(
        p.numel() for p in global_model.parameters() if p.requires_grad
    )
    print(f"[CLVC-Net] dataset={args.dataset} vocab={vocab_size} trainable={trainable/1e6:.1f}M (FP32)", flush=True)

    # Stage 0 baseline
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_clvc_split(
        args.dataset, local_model, global_model, val_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_clvc_split(
        args.dataset, local_model, global_model, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    print(
        f"[Stage 0 CLVC-Net Init] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_init", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    # VRAM auto-probe (FP32 safe for GeM p=3)
    base_loc = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    base_glb = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    def probe_step(bs: int) -> None:
        local_model.train()
        global_model.train()
        local_model.zero_grad(set_to_none=True)
        global_model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = ["change color to blue"] * bs
        with torch.no_grad():
            g_t0, g_t1 = global_model.extract_img_feature(d_tgt)
            g_q0 = global_model.compose_img_text(d_ref, d_txt)[0]
        l_dict = local_model.compute_loss(d_ref, d_txt, d_tgt, g_t0, g_t1, g_q0)
        l_loss = args.class_weight * l_dict["class"] + l_dict["perceptual"] + args.img_weight * l_dict["img"] + args.mul_kl_weight * l_dict["mul_kl"]
        l_loss.backward()
        with torch.no_grad():
            l_t0, l_t1 = local_model.extract_img_feature(d_tgt)
            l_q1 = local_model.compose_img_text(d_ref, d_txt)[1]
        g_dict = global_model.compute_loss(d_ref, d_txt, d_tgt, l_t0, l_t1, l_q1)
        g_loss = args.class_weight * g_dict["class"] + args.img_weight * g_dict["img"] + args.mul_kl_weight * g_dict["mul_kl"]
        g_loss.backward()
        local_model.zero_grad(set_to_none=True)
        global_model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=64
    )
    local_model.load_state_dict(base_loc, strict=True)
    global_model.load_state_dict(base_glb, strict=True)
    del base_loc, base_glb
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[CLVC-Net VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": "clvcnet_resnet50",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "vocab_size": vocab_size,
        "hyperparams": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_decay": args.lr_decay,
            "lr_div": args.lr_div,
            "max_decay_epoch": args.max_decay_epoch,
            "class_weight": args.class_weight,
            "img_weight": args.img_weight,
            "mul_kl_weight": args.mul_kl_weight,
            "local_t": args.local_t,
            "global_t": args.global_t,
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

    opt_local = optim.Adam(local_model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
    opt_global = optim.Adam(global_model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / "best_clvcnet.pt"

    def save_ckpt(epoch: int, score: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "val_score": score,
                "vocab_word2id": local_model.text_model.vocab.word2id,
                "local_state_dict": {k: v.cpu() for k, v in local_model.state_dict().items()},
                "global_state_dict": {k: v.cpu() for k, v in global_model.state_dict().items()},
            },
            best_ckpt_path,
        )

    save_ckpt(0, best_val_score)
    no_improve = 0

    print(f"[CLVC-Net] Starting training: epochs={args.num_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        train_ds.set_epoch(epoch)
        local_model.train()
        global_model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        opt_local.zero_grad(set_to_none=True)
        opt_global.zero_grad(set_to_none=True)
        loc_loss_sum = 0.0
        glb_loss_sum = 0.0
        n_samples = 0
        steps_in_window = 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            mods = [_safe_caption(c) for c in batch["raw_text"]]
            bs_curr = ref_imgs.size(0)

            # 1) Local model step (exact alternating mutual learning from CLVCNet/train.py:195-226)
            with torch.no_grad():
                g_t0, g_t1 = global_model.extract_img_feature(tgt_imgs)
                g_q0 = global_model.compose_img_text(ref_imgs, mods)[0]
            l_dict = local_model.compute_loss(ref_imgs, mods, tgt_imgs, g_t0, g_t1, g_q0)
            l_loss = (
                args.class_weight * l_dict["class"]
                + l_dict["perceptual"]
                + args.img_weight * l_dict["img"]
                + args.mul_kl_weight * l_dict["mul_kl"]
            )
            lv = float(l_loss.detach().cpu().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"Non-finite CLVC-Net local loss at ep={epoch} step={step_idx}: {l_dict}")
            (l_loss / grad_accum).backward()

            # 2) Global model step
            with torch.no_grad():
                l_t0, l_t1 = local_model.extract_img_feature(tgt_imgs)
                l_q1 = local_model.compose_img_text(ref_imgs, mods)[1]
            g_dict = global_model.compute_loss(ref_imgs, mods, tgt_imgs, l_t0, l_t1, l_q1)
            g_loss = (
                args.class_weight * g_dict["class"]
                + args.img_weight * g_dict["img"]
                + args.mul_kl_weight * g_dict["mul_kl"]
            )
            gv = float(g_loss.detach().cpu().item())
            if not np.isfinite(gv):
                raise RuntimeError(f"Non-finite CLVC-Net global loss at ep={epoch} step={step_idx}: {g_dict}")
            (g_loss / grad_accum).backward()

            loc_loss_sum += lv * bs_curr
            glb_loss_sum += gv * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                opt_local.step()
                opt_global.step()
                opt_local.zero_grad(set_to_none=True)
                opt_global.zero_grad(set_to_none=True)
                steps_in_window = 0

        if epoch % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in opt_local.param_groups:
                g["lr"] *= args.lr_div
            for g in opt_global.param_groups:
                g["lr"] *= args.lr_div

        avg_loc = loc_loss_sum / max(n_samples, 1)
        avg_glb = glb_loss_sum / max(n_samples, 1)
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        val_metrics, _, _ = evaluate_clvc_split(
            args.dataset, local_model, global_model, val_bundle, val_preprocess, device,
            eval_bs=args.eval_bs, num_workers=effective_workers,
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            no_improve = 0
            save_ckpt(epoch, val_score)
        else:
            no_improve += 1

        print(
            f"[CLVC-Net Ep {epoch:02d}/{args.num_epochs}] loc_loss={avg_loc:.4f} glb_loss={avg_glb:.4f} | "
            f"lr={opt_local.param_groups[0]['lr']:.1e} | val {sel_name}={val_score:.4f} "
            f"(best={best_val_score:.4f}@ep{best_epoch}) | vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "train",
                "epoch": epoch,
                "lr": opt_local.param_groups[0]["lr"],
                "train_losses": {"local": round(avg_loc, 6), "global": round(avg_glb, 6)},
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_val_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve >= args.patience:
            print(f"[CLVC-Net] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    local_model.load_state_dict(ckpt["local_state_dict"], strict=True)
    global_model.load_state_dict(ckpt["global_state_dict"], strict=True)
    final_test_metrics, final_test_preds, _ = evaluate_clvc_split(
        args.dataset, local_model, global_model, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "clvcnet_resnet50",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_init_test": zs_test_metrics,
        "final_clvcnet_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== CLVC-Net FINAL TEST SUMMARY ===", flush=True)
    print(f"  Init CLVC-Net Test  : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final CLVC-Net Test : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
