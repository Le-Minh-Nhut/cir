#!/usr/bin/env python3
"""Upstream-faithful TME (blip2_cir_image_diff_features, BLIP-2 + 3-stage warmup + GMM Partitioner)
trainer for EDIR and PinPoint canonical splits.

Computes frozen EVA-ViT-g embeddings online via `model.vit_encode()` (zero multi-GB feature
cache needed, mathematically identical to `precompute.py`).

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/TME/src \
    workbench/envs/tme-adapted/bin/python \
    workbench/benchmarks/trainers/train_tme.py --dataset edir
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/heheboiz/data/cir")
TME_DIR = REPO_ROOT / "workbench" / "third_party" / "TME"
TME_SRC = TME_DIR / "src"
for p in (str(REPO_ROOT), str(TME_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

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
def extract_tme_corpus_features(
    blip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_corpus, 32, 256] target features matching precompute_evaluation.py."""
    blip_model.eval()
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
    feats_list: List[torch.Tensor] = []
    ids: List[str] = []
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            vit_e = blip_model.vit_encode(imgs)
            f_img = blip_model.encode_image(vit_e)
            z_tgt = F.normalize(blip_model.vision_proj(f_img), dim=-1)
        feats_list.append(z_tgt.float().cpu())
        ids.extend(batch["image_id"])
    return torch.cat(feats_list, dim=0), ids


@torch.no_grad()
def extract_tme_query_features(
    blip_model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_query, 256] query features matching precompute_evaluation.py."""
    blip_model.eval()
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
    feats_list: List[torch.Tensor] = []
    qids: List[str] = []
    for batch in dl:
        imgs = batch["ref_image"].to(device, non_blocking=True)
        caps = [txt_proc(c) for c in batch["raw_text"]]
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            vit_e = blip_model.vit_encode(imgs)
            f_r = blip_model.encode_image(vit_e)
            text_tokens = blip_model.tokenizer(
                caps,
                padding="max_length",
                truncation=True,
                max_length=blip_model.max_txt_len,
                return_tensors="pt",
            ).to(device)
            z_rm = blip_model.encode_fusion(f_r, text_tokens)
        feats_list.append(z_rm.float().cpu())
        qids.extend(batch["query_id"])
    return torch.cat(feats_list, dim=0), qids


@torch.no_grad()
def rank_tme_multitoken(
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
        raise ValueError("Non-finite features in rank_tme_multitoken")
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


def evaluate_tme_split(
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
        target_feats, corpus_ids = extract_tme_corpus_features(
            blip_model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        target_feats, corpus_ids = cached_corpus
    query_feats, query_ids = extract_tme_query_features(
        blip_model, bundle, preprocess, txt_proc, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_tme_multitoken(query_feats, query_ids, target_feats, corpus_ids, device=device, top_k=top_k)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (target_feats, corpus_ids)


@torch.no_grad()
def fit_gmm_clean_mask(
    blip_model: nn.Module,
    train_ds: CIRTrainingDataset,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Exact upstream TME 2-component GMM clean/noisy partitioner keyed by query_id."""
    blip_model.eval()
    dl = DataLoader(
        train_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_train_batch,
    )
    losses_all: List[float] = []
    qids_all: List[str] = []
    for batch in dl:
        ref_imgs = batch["ref_image"].to(device, non_blocking=True)
        tgt_imgs = batch["target_image"].to(device, non_blocking=True)
        caps = [txt_proc(c) for c in batch["raw_text"]]
        if ref_imgs.size(0) == 1:
            # Avoid B==1 squeeze bug inside per_loss
            losses_all.append(0.0)
            qids_all.extend(batch["query_id"])
            continue
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            ref_e = blip_model.vit_encode(ref_imgs)
            tgt_e = blip_model.vit_encode(tgt_imgs)
            per_l, _ = blip_model.per_loss(ref_e, tgt_e, caps)
        losses_all.extend(per_l.float().cpu().tolist())
        qids_all.extend(batch["query_id"])

    arr = np.asarray(losses_all, dtype=np.float64)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max - arr_min < 1e-12:
        return {qid: 1.0 for qid in qids_all}
    normed = ((arr - arr_min) / (arr_max - arr_min)).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, random_state=42)
    gmm.fit(normed)
    clean_comp = int(gmm.means_.argmin())
    probs = gmm.predict_proba(normed)[:, clean_comp]
    mask = (probs > threshold).astype(np.float32)
    # Guard against degenerate all-zero mask
    if mask.sum() < max(1, int(0.05 * len(mask))):
        mask = np.ones_like(mask)
    return {qid: float(m) for qid, m in zip(qids_all, mask)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TME (blip2_cir_image_diff_features) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lrm", type=float, default=1.0)
    parser.add_argument("--lpm", type=float, default=1.0)
    parser.add_argument("--lsa", type=float, default=1.0)
    parser.add_argument("--lrd", type=float, default=0.2)
    parser.add_argument("--warmup-qformer", type=int, default=3)
    parser.add_argument("--warmup-proj", type=int, default=2)
    parser.add_argument("--warmup-last", type=int, default=1)
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

    run_dir = args.run_dir or default_run_dir(args.dataset, "tme_blip2", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    # TME loads 'weight/blip2_pretrained.pth' & 'weight/eva_vit_g.pth' relative to TME_DIR
    cwd_before = os.getcwd()
    try:
        os.chdir(str(TME_DIR))
        import utility as tme_utility  # noqa: E402
        from lavis.models import load_model_and_preprocess  # noqa: E402

        blip_model, _, txt_processors = load_model_and_preprocess(
            name="blip2_cir_image_diff_features", model_type="pretrain", is_eval=False, device=device
        )
        preprocess = tme_utility.targetpad_transform(target_ratio=args.target_ratio, dim=224)
    finally:
        os.chdir(cwd_before)

    txt_proc = txt_processors["eval"]
    trainable = sum(p.numel() for p in blip_model.parameters() if p.requires_grad)
    print(f"[TME] dataset={args.dataset} model=blip2_cir_image_diff_features trainable={trainable/1e6:.1f}M", flush=True)

    pn_loss = {
        "positive_loss": "RCL",
        "negative_loss": "None",
        "positive_align_loss": "MSE",
        "negative_align_loss": "None",
        "trade_off": 1.0,
        "trade_off_align": 1.0,
        "warmup_loss": "RCL",
        "warmup_align_loss": "MSE",
    }
    loss_balance = {"lrm": args.lrm, "lpm": args.lpm, "lsa": args.lsa, "lrd": args.lrd}

    # Stage 0 zero-shot baseline
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_tme_split(
        args.dataset, blip_model, val_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_tme_split(
        args.dataset, blip_model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    del zs_val_corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Stage 0 TME] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_zeroshot", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    # VRAM auto-probe in worst-case full phase (warmup=None -> 3 encode_image + 3 encode_fusion passes!)
    base_non_vit = {
        k: v.detach().cpu().clone() for k, v in blip_model.state_dict().items() if not k.startswith("visual_encoder.")
    }

    def probe_step(bs: int) -> None:
        blip_model.train()
        blip_model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = [txt_proc("replace the chair with a blue sofa")] * bs
        labels = torch.ones(bs, device=device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            with torch.no_grad():
                ref_e = blip_model.vit_encode(d_ref)
                tgt_e = blip_model.vit_encode(d_tgt)
            ld = blip_model({"image": ref_e, "target": tgt_e, "text_input": d_txt}, labels, pn_loss, None)
            loss = sum(loss_balance[k] * ld[k] for k in ld)
        loss.backward()
        blip_model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=48
    )
    blip_model.load_state_dict(base_non_vit, strict=False)
    del base_non_vit
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[TME VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": "tme_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss_balance": loss_balance,
            "warmup_qformer": args.warmup_qformer,
            "warmup_proj": args.warmup_proj,
            "warmup_last": args.warmup_last,
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
        prefetch_factor=4 if effective_workers > 0 else None,
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
    proj_params = list(blip_model.d2t_proj.parameters()) + [blip_model.prompt_tokens]
    optimizer_proj = optim.AdamW(
        [{"params": proj_params, "lr": args.learning_rate, "betas": (0.9, 0.98), "eps": 1e-7, "weight_decay": args.weight_decay}]
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
    best_ckpt_path = run_dir / "best_tme.pt"

    def save_ckpt(epoch: int, score: float) -> None:
        compact = {k: v.cpu() for k, v in blip_model.state_dict().items() if not k.startswith("visual_encoder.")}
        torch.save({"epoch": epoch, "val_score": score, "state_dict_non_vit": compact}, best_ckpt_path)

    save_ckpt(0, best_val_score)
    no_improve = 0
    sched_steps = 0
    warmup_total = args.warmup_qformer + args.warmup_proj + args.warmup_last

    print(f"[TME] Starting training: epochs={args.num_epochs}, warmup_total={warmup_total}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        ep0 = epoch - 1
        if ep0 < args.warmup_qformer:
            warmup_phase: Optional[str] = "qformer"
        elif ep0 < args.warmup_qformer + args.warmup_proj:
            warmup_phase = "proj"
        elif ep0 < warmup_total:
            warmup_phase = "last"
        else:
            warmup_phase = None

        train_ds.set_epoch(epoch)
        if warmup_phase is None:
            gmm_mask_map = fit_gmm_clean_mask(
                blip_model, train_ds, txt_proc, device, eval_bs=args.eval_bs, num_workers=effective_workers
            )
        else:
            gmm_mask_map = None

        blip_model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        active_opt = optimizer_proj if warmup_phase == "proj" else optimizer
        active_opt.zero_grad(set_to_none=True)
        loss_sums: Dict[str, float] = {"total": 0.0}
        n_samples, steps_in_window = 0, 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            caps = [txt_proc(c) for c in batch["raw_text"]]
            bs_curr = ref_imgs.size(0)
            if gmm_mask_map is not None:
                labels = torch.tensor([gmm_mask_map.get(qid, 1.0) for qid in batch["query_id"]], dtype=torch.float32, device=device)
                if labels.sum() == 0:
                    labels = torch.ones(bs_curr, device=device)
            else:
                labels = torch.ones(bs_curr, device=device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                with torch.no_grad():
                    ref_e = blip_model.vit_encode(ref_imgs)
                    tgt_e = blip_model.vit_encode(tgt_imgs)
                ld = blip_model({"image": ref_e, "target": tgt_e, "text_input": caps}, labels, pn_loss, warmup_phase)
                loss = sum(loss_balance[k] * ld[k] for k in ld)

            lv = float(loss.detach().cpu().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"Non-finite TME loss at ep={epoch} phase={warmup_phase} step={step_idx}: {ld}")

            scaler.scale(loss / grad_accum).backward()
            loss_sums["total"] += lv * bs_curr
            for lk, lval in ld.items():
                loss_sums[lk] = loss_sums.get(lk, 0.0) + float(lval.detach().cpu().item()) * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scale_before = scaler.get_scale()
                scaler.step(active_opt)
                scaler.update()
                active_opt.zero_grad(set_to_none=True)
                # Upstream precompute_train.py:152 steps main OneCycleLR scheduler every opt step
                if scaler.get_scale() >= scale_before and sched_steps < total_opt_steps:
                    if warmup_phase != "proj":
                        scheduler.step()
                    sched_steps += 1
                steps_in_window = 0

        avg_losses = {k: round(v / max(n_samples, 1), 6) for k, v in loss_sums.items()}
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        cur_lr = float(active_opt.param_groups[0]["lr"])
        val_metrics, _, _ = evaluate_tme_split(
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
            # Don't count early-stop patience while still inside warmup phases (proj freezes Q-Former!)
            if warmup_phase is None:
                no_improve += 1

        print(
            f"[TME Ep {epoch:02d}/{args.num_epochs} ({warmup_phase or 'full'})] loss={avg_losses['total']:.4f} | "
            f"lr={cur_lr:.2e} | val {sel_name}={val_score:.4f} (best={best_val_score:.4f}@ep{best_epoch}) | "
            f"vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": warmup_phase or "full",
                "epoch": epoch,
                "lr": cur_lr,
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
            print(f"[TME] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    msg = blip_model.load_state_dict(ckpt["state_dict_non_vit"], strict=False)
    assert not [k for k in msg.missing_keys if not k.startswith("visual_encoder.")]
    assert not msg.unexpected_keys

    final_test_metrics, final_test_preds, _ = evaluate_tme_split(
        args.dataset, blip_model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "tme_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_zeroshot_test": zs_test_metrics,
        "final_tme_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== TME FINAL TEST SUMMARY ===", flush=True)
    print(f"  Zero-Shot Test : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final Test     : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
