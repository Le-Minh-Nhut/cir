#!/usr/bin/env python3
"""Upstream-faithful MTST_PTHA (blip2_cir_cat, BLIP-2 EVA-ViT-g + Q-Former) trainer
for EDIR and PinPoint canonical splits.

Uses Stage-1 single-caption `blip2_cir_cat` (no external LLM reverse-caption dependency)
with consistent mean-pooling `[:, :33, :].mean(dim=1)` matching `forward()`.

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/MTST_PTHA/MTST_PTHA/src \
    workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python \
    workbench/benchmarks/trainers/train_mtst_ptha.py --dataset edir
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/heheboiz/data/cir")
MTST_SRC = REPO_ROOT / "workbench" / "third_party" / "MTST_PTHA" / "MTST_PTHA" / "src"
for p in (str(REPO_ROOT), str(MTST_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import targetpad_transform  # noqa: E402
from lavis.models import load_model_and_preprocess  # noqa: E402

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


@torch.no_grad()
def extract_mtst_corpus_features(
    blip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_corpus, 32, 256] target features matching `forward()` lines 143-156."""
    blip_model.eval()
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
        with blip_model.maybe_autocast():
            img_embeds = blip_model.ln_vision(blip_model.visual_encoder(imgs))
        img_embeds = img_embeds.float()
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=device)
        q_tok = blip_model.query_tokens.expand(img_embeds.shape[0], -1, -1)
        out = blip_model.Qformer.bert(
            query_embeds=q_tok,
            encoder_hidden_states=img_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        feats = F.normalize(blip_model.vision_proj(out.last_hidden_state), dim=-1)
        feats_list.append(feats.float().cpu())
        ids.extend(batch["image_id"])
    return torch.cat(feats_list, dim=0), ids


@torch.no_grad()
def extract_mtst_query_features(
    blip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_query, 256] query features matching `blip2_cir_cat.forward()` lines 98-139."""
    blip_model.eval()
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
        with blip_model.maybe_autocast():
            ref_embeds = blip_model.ln_vision(blip_model.visual_encoder(imgs))
        ref_embeds = ref_embeds.float()
        img_atts = torch.ones(ref_embeds.size()[:-1], dtype=torch.long, device=device)
        q_tok = blip_model.query_tokens.expand(ref_embeds.shape[0], -1, -1)
        q_atts = torch.ones(q_tok.size()[:-1], dtype=torch.long, device=device)
        t_tok = blip_model.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=blip_model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        attn_mask = torch.cat([q_atts, t_tok.attention_mask], dim=1)
        fusion_out = blip_model.Qformer.bert(
            t_tok.input_ids,
            query_embeds=q_tok,
            attention_mask=attn_mask,
            encoder_hidden_states=ref_embeds,
            encoder_attention_mask=img_atts,
            return_dict=True,
        )
        text_out = blip_model.Qformer.bert(
            t_tok.input_ids,
            query_embeds=fusion_out.last_hidden_state[:, : q_tok.size(1), :],
            attention_mask=attn_mask,
            return_dict=True,
        )
        # Consistent with blip2_qformer_cir_cat.py:137-139 training objective
        fusion_feats = F.normalize(blip_model.text_proj(text_out.last_hidden_state[:, :33, :].mean(dim=1)), dim=-1)
        feats_list.append(fusion_feats.float().cpu())
        qids.extend(batch["query_id"])
    return torch.cat(feats_list, dim=0), qids


@torch.no_grad()
def rank_mtst_multitoken(
    query_feats: torch.Tensor,
    query_ids: List[str],
    target_feats: torch.Tensor,
    corpus_ids: List[str],
    device: torch.device,
    top_k: int = 50,
    q_chunk: int = 128,
    c_chunk: int = 4096,
) -> Dict[str, List[str]]:
    if not torch.isfinite(query_feats).all() or not torch.isfinite(target_feats).all():
        raise ValueError("Non-finite features in rank_mtst_multitoken")
    n_q, n_c = query_feats.shape[0], target_feats.shape[0]
    k = min(int(top_k), n_c)
    results: Dict[str, List[str]] = {}
    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        qf = query_feats[qs:qe].to(device, non_blocking=True)
        sims_row = np.empty((qe - qs, n_c), dtype=np.float32)
        for cs in range(0, n_c, c_chunk):
            ce = min(cs + c_chunk, n_c)
            tf = target_feats[cs:ce].to(device, non_blocking=True)
            sims_row[:, cs:ce] = torch.einsum("bd,nkd->bnk", qf, tf).max(dim=-1).values.cpu().numpy()
        order = np.argsort(-sims_row, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[qs:qe]):
            results[str(qid)] = [str(corpus_ids[ci]) for ci in order[row_i]]
    return results


def evaluate_mtst_split(
    dataset: str,
    blip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
    cached_corpus: Optional[Tuple[torch.Tensor, List[str]]] = None,
    top_k: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Tuple[torch.Tensor, List[str]]]:
    if cached_corpus is None:
        target_feats, corpus_ids = extract_mtst_corpus_features(
            blip_model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        target_feats, corpus_ids = cached_corpus
    query_feats, query_ids = extract_mtst_query_features(
        blip_model, bundle, preprocess, txt_proc, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_mtst_multitoken(query_feats, query_ids, target_feats, corpus_ids, device=device, top_k=top_k)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (target_feats, corpus_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTST_PTHA (blip2_cir_cat) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--positive-mode", type=str, default="random", choices=["random", "expand"])
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--eval-bs", type=int, default=64)
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

    run_dir = args.run_dir or default_run_dir(args.dataset, "mtst_ptha_blip2", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    blip_model, _, txt_processors = load_model_and_preprocess(
        name="blip2_cir_cat", model_type="pretrain", is_eval=False, device=device
    )
    txt_proc = txt_processors["eval"]
    preprocess = targetpad_transform(args.target_ratio, 224)

    trainable = sum(p.numel() for p in blip_model.parameters() if p.requires_grad)
    print(f"[MTST_PTHA] dataset={args.dataset} model=blip2_cir_cat trainable={trainable/1e6:.1f}M", flush=True)

    # Stage 0 zero-shot baseline
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_mtst_split(
        args.dataset, blip_model, val_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_mtst_split(
        args.dataset, blip_model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    del zs_val_corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Stage 0 MTST_PTHA] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_zeroshot", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    # VRAM auto-probe
    base_non_vit = {
        k: v.detach().cpu().clone() for k, v in blip_model.state_dict().items() if not k.startswith("visual_encoder.")
    }

    def probe_step(bs: int) -> None:
        blip_model.train()
        blip_model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = [txt_proc("change the hat to green")] * bs
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            loss = blip_model({"image": d_ref, "target": d_tgt, "text_input": d_txt})["loss_itc"]
        loss.backward()
        blip_model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=64
    )
    blip_model.load_state_dict(base_non_vit, strict=False)
    del base_non_vit
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[MTST_PTHA VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": "mtst_ptha_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "blip_model_name": "blip2_cir_cat",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "micro_bs": micro_bs,
            "grad_accum": grad_accum,
            "effective_bs": micro_bs * grad_accum,
            "target_ratio": args.target_ratio,
            "positive_mode": args.positive_mode,
            "val_corpus_scope": args.val_corpus_scope,
            "seed": args.seed,
            "val_selection_metric": sel_name,
        },
        "vram_probe": probe_log,
    }
    write_json(run_dir / "config.json", config_payload)

    train_ds = CIRTrainingDataset(
        train_bundle, image_transform=preprocess, positive_mode=args.positive_mode, load_images=True, seed=args.seed
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
        [
            {
                "params": [p for p in blip_model.parameters() if p.requires_grad],
                "lr": args.learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": args.weight_decay,
            }
        ]
    )
    opt_steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_opt_steps = opt_steps_per_epoch * args.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        pct_start=min(0.3, 1.5 / max(args.num_epochs, 1)),
        div_factor=100.0,
        total_steps=total_opt_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / "best_mtst_ptha.pt"

    def save_ckpt(epoch: int, score: float) -> None:
        compact = {k: v.cpu() for k, v in blip_model.state_dict().items() if not k.startswith("visual_encoder.")}
        torch.save({"epoch": epoch, "val_score": score, "state_dict_non_vit": compact}, best_ckpt_path)

    save_ckpt(0, best_val_score)
    no_improve = 0
    sched_steps = 0

    print(f"[MTST_PTHA] Starting training: epochs={args.num_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        train_ds.set_epoch(epoch)
        blip_model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        loss_sum, n_samples, steps_in_window = 0.0, 0, 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            captions = [txt_proc(c) for c in batch["raw_text"]]
            bs_curr = ref_imgs.size(0)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = blip_model({"image": ref_imgs, "target": tgt_imgs, "text_input": captions})["loss_itc"]

            lv = float(loss.detach().cpu().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"Non-finite MTST_PTHA loss at ep={epoch} step={step_idx}: {lv}")

            scaler.scale(loss / grad_accum).backward()
            loss_sum += lv * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before and sched_steps < total_opt_steps:
                    scheduler.step()
                    sched_steps += 1
                steps_in_window = 0

        avg_loss = loss_sum / max(n_samples, 1)
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        cur_lr = float(optimizer.param_groups[0]["lr"])
        val_metrics, _, _ = evaluate_mtst_split(
            args.dataset, blip_model, val_bundle, preprocess, txt_proc, device,
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
            f"[MTST_PTHA Ep {epoch:02d}/{args.num_epochs}] loss={avg_loss:.4f} | lr={cur_lr:.2e} | "
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
                "train_loss": round(avg_loss, 6),
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_val_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve >= args.patience:
            print(f"[MTST_PTHA] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    msg = blip_model.load_state_dict(ckpt["state_dict_non_vit"], strict=False)
    assert not [k for k in msg.missing_keys if not k.startswith("visual_encoder.")]
    assert not msg.unexpected_keys

    final_test_metrics, final_test_preds, _ = evaluate_mtst_split(
        args.dataset, blip_model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "mtst_ptha_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_zeroshot_test": zs_test_metrics,
        "final_mtst_ptha_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== MTST_PTHA FINAL TEST SUMMARY ===", flush=True)
    print(f"  Zero-Shot Test : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final Test     : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
