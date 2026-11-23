#!/usr/bin/env python3
"""Upstream-faithful DCNet (ResNet50 ImageNet + GloVe 300d + 7-expert TIRG/FusDiff + cyc3)
trainer for EDIR and PinPoint canonical splits.

Builds GloVe 300d vocab STRICTLY from training captions using the staged local
`en_vectors_web_lg-2.1.0` package (zero val/test leak).

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/DCNet \
    workbench/envs/dcnet310-sm120/bin/python \
    workbench/benchmarks/trainers/train_dcnet.py --dataset edir
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path("/home/heheboiz/data/cir")
DCNET_DIR = REPO_ROOT / "workbench" / "third_party" / "DCNet"
GLOVE_SPACY_DIR = REPO_ROOT / "workbench" / "artifacts" / "glove_recipe" / "en_vectors_web_lg-2.1.0" / "en_vectors_web_lg" / "en_vectors_web_lg-2.1.0"
DCNET_PREPARE_PY = REPO_ROOT / "workbench" / "envs" / "dcnet-prepare" / "bin" / "python"

for p in (str(REPO_ROOT), str(DCNET_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.loss import AdditiveMarginSoftmax  # noqa: E402
from model.model import CENet  # noqa: E402

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


def _tokenize_dcnet(caption: str) -> List[str]:
    clean = caption.replace("-", " ").replace(".", "").lower()
    return re.findall(r"[a-z0-9']+|[?!,;:]+", clean)


def build_or_load_glove_vocab(
    train_captions: List[str],
    cache_path: Path,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build 300-d GloVe vocab strictly from train captions using local en_vectors_web_lg."""
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    counts: Counter = Counter()
    for cap in train_captions:
        counts.update(_tokenize_dcnet(cap))

    words_list = sorted(counts.keys())
    tmp_words = cache_path.with_suffix(".words.json")
    tmp_words.parent.mkdir(parents=True, exist_ok=True)
    tmp_words.write_text(json.dumps({"words": words_list, "counts": dict(counts), "seed": seed}), encoding="utf-8")

    # Query local en_vectors_web_lg via dcnet-prepare python (has spacy) or fallback directly
    script = f"""
import json, pickle, numpy as np
import spacy
nlp = spacy.load({str(GLOVE_SPACY_DIR)!r})
data = json.loads(open({str(tmp_words)!r}).read())
rng = np.random.RandomState(data['seed'])
vocab = {{}}
hit = 0
oov_kept = 0
for w in data['words']:
    lex = nlp.vocab[w]
    if lex.has_vector and np.linalg.norm(lex.vector) > 0:
        vocab[w] = np.asarray(lex.vector, dtype=np.float32)
        hit += 1
    elif data['counts'][w] > 2:
        vocab[w] = rng.normal(0.0, 0.3, (300,)).astype(np.float32)
        oov_kept += 1
with open({str(cache_path)!r}, 'wb') as f:
    pickle.dump(vocab, f)
print(f'DCNet GloVe vocab built: total_train_words={{len(data[\"words\"])}}, glove_hit={{hit}}, oov_freq>2={{oov_kept}}, kept={{len(vocab)}}')
"""
    py_exe = str(DCNET_PREPARE_PY) if DCNET_PREPARE_PY.is_file() else sys.executable
    out = subprocess.check_output([py_exe, "-c", script], text=True)
    print(f"[DCNet] {out.strip()}", flush=True)
    tmp_words.unlink(missing_ok=True)
    with cache_path.open("rb") as f:
        return pickle.load(f)


def encode_captions_to_glove_batch(
    captions: List[str],
    vocab: Dict[str, np.ndarray],
    max_words: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of raw captions to (B, max_len, 300) + lengths (empty fallback -> 1 zero vec)."""
    seqs: List[np.ndarray] = []
    lengths: List[int] = []
    for cap in captions:
        toks = _tokenize_dcnet(cap)
        vecs = [vocab[t] for t in toks if t in vocab][:max_words]
        if not vecs:
            vecs = [np.zeros((300,), dtype=np.float32)]
        seqs.append(np.stack(vecs, axis=0))
        lengths.append(len(vecs))
    max_l = max(lengths)
    padded = np.zeros((len(captions), max_l, 300), dtype=np.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0], :] = s
    return torch.from_numpy(padded), torch.tensor(lengths, dtype=torch.long)


def build_dcnet_model(device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Instantiate exact upstream 7-expert CENet with ResNet50 ImageNet base weights."""
    modalities = ["im_feat", "inter-1", "spatial0", "spatial1", "spatial2", "spatial3", "spatial4"]
    text_dim = 300
    expert_dims = {
        "im_feat": 2048,
        "inter-1": 1024,
        "spatial0": 2048,
        "spatial1": 2048,
        "spatial2": 2048,
        "spatial3": 2048,
        "spatial4": 2048,
    }
    model = CENet(
        text_dim=text_dim,
        composition="tirg",
        correction="fd",
        fusion="hadamard",
        expert_dims=expert_dims,
        same_dim=1024,
        text_feat="glove",
        norm_scale=5.0,
        text_moe=False,
        backbone_pretrain="imagenet",
        backbone="resnet",
        text_pooling_module="sharedconv",
        metric_fn="ip",
        ref=False,
    ).to(device)
    return model, modalities


@torch.no_grad()
def evaluate_dcnet_split(
    dataset: str,
    model: nn.Module,
    modalities: List[str],
    vocab: Dict[str, np.ndarray],
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 128,
    q_bs: int = 8,
    c_chunk: int = 2048,
    num_workers: int = 16,
    cached_corpus: Optional[Tuple[Dict[str, torch.Tensor], List[str]]] = None,
    top_k: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Tuple[Dict[str, torch.Tensor], List[str]]]:
    """Exact DCNet inference: full-gallery log_softmax(comp) + log_softmax(corr) (chunked safely for 16GB)."""
    model.eval()
    if cached_corpus is None:
        c_ds = CIRCorpusDataset(bundle, image_transform=preprocess)
        c_dl = DataLoader(
            c_ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            collate_fn=collate_corpus,
        )
        gal_acc: Dict[str, List[torch.Tensor]] = {m: [] for m in modalities}
        corpus_ids: List[str] = []
        for batch in c_dl:
            imgs = batch["image"].to(device, non_blocking=True)
            exps = model.get_image_emb({"im_feat": imgs}, istgt=True)
            for m in modalities:
                gal_acc[m].append(exps[m].float().cpu())
            corpus_ids.extend(batch["image_id"])
        gal_experts = {m: torch.cat(gal_acc[m], dim=0) for m in modalities}
    else:
        gal_experts, corpus_ids = cached_corpus

    n_c = len(corpus_ids)
    k = min(int(top_k), n_c)

    # Pre-extract all query embeddings once (tiny: ~38 MB for 1,337 queries) so gallery chunks
    # are transferred CPU->GPU ONLY ONCE instead of Q/q_bs times (saves ~500 GB PCIe copy on PinPoint!).
    q_ds = CIRQueryEvalDataset(bundle, image_transform=preprocess)
    q_dl = DataLoader(
        q_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_eval_queries,
    )
    src_acc: Dict[str, List[torch.Tensor]] = {m: [] for m in modalities}
    txt_acc: Dict[str, List[torch.Tensor]] = {m: [] for m in modalities}
    trg_acc: Dict[str, List[torch.Tensor]] = {m: [] for m in modalities}
    query_ids: List[str] = []
    for batch in q_dl:
        ref_imgs = batch["ref_image"].to(device, non_blocking=True)
        txt_t, txt_l = encode_captions_to_glove_batch(batch["raw_text"], vocab)
        txt_t = txt_t.to(device, non_blocking=True)
        txt_l = txt_l.tolist()
        src_exp = model.get_image_emb({"im_feat": ref_imgs}, istgt=False)
        txt_exp = model.get_text_emb(txt_t, txt_l)
        mod_trg = model.get_composition(src_exp, txt_exp)
        for m in modalities:
            src_acc[m].append(src_exp[m].float().cpu())
            txt_acc[m].append(txt_exp[m].float().cpu())
            trg_acc[m].append(mod_trg[m].float().cpu())
        query_ids.extend(batch["query_id"])

    n_q = len(query_ids)
    all_src = {m: torch.cat(src_acc[m], dim=0) for m in modalities}
    all_txt = {m: torch.cat(txt_acc[m], dim=0) for m in modalities}
    all_trg = {m: torch.cat(trg_acc[m], dim=0) for m in modalities}

    comp_full = torch.empty((n_q, n_c), dtype=torch.float32)
    corr_full = torch.empty((n_q, n_c), dtype=torch.float32)

    # Outer loop over gallery chunks -> each gallery block moves to GPU ONCE!
    for cs in range(0, n_c, c_chunk):
        ce = min(cs + c_chunk, n_c)
        gal_blk = {m: gal_experts[m][cs:ce].to(device, non_blocking=True) for m in modalities}
        for qs in range(0, n_q, q_bs):
            qe = min(qs + q_bs, n_q)
            src_blk = {m: all_src[m][qs:qe].to(device, non_blocking=True) for m in modalities}
            txt_blk = {m: all_txt[m][qs:qe].to(device, non_blocking=True) for m in modalities}
            trg_blk = {m: all_trg[m][qs:qe].to(device, non_blocking=True) for m in modalities}
            comp_full[qs:qe, cs:ce] = model.get_score_matrix(gal_blk, trg_blk, modalities).cpu()
            mod_txt = model.get_correction(src_blk, gal_blk)
            corr_full[qs:qe, cs:ce] = model.get_score_matrix(mod_txt, txt_blk, modalities).cpu()

    # Full-gallery log_softmax fusion (exact match to trainer_joint.py:239!)
    preds: Dict[str, List[str]] = {}
    for qs in range(0, n_q, 128):
        qe = min(qs + 128, n_q)
        fused = F.log_softmax(comp_full[qs:qe], dim=-1) + F.log_softmax(corr_full[qs:qe], dim=-1)
        fused_np = fused.numpy()
        if not np.isfinite(fused_np).all():
            raise ValueError("Non-finite DCNet fused scores detected")
        order = np.argsort(-fused_np, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[qs:qe]):
            preds[str(qid)] = [str(corpus_ids[ci]) for ci in order[row_i]]

    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (gal_experts, corpus_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DCNet (ResNet50+GloVe) on EDIR/PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--joint-weight", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--positive-mode", type=str, default="random", choices=["random", "expand"])
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--q-bs", type=int, default=8)
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

    run_dir = args.run_dir or default_run_dir(args.dataset, "dcnet_resnet50", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    # Build GloVe vocab strictly from training split captions
    vocab_cache = run_dir / f"glove_vocab_{args.dataset}_{args.split_tag}.pkl"
    vocab = build_or_load_glove_vocab([q.text for q in train_bundle.queries], vocab_cache, seed=args.seed)

    # Include Resize((256, 256)) matching DCNet's offline resize_img.py
    train_preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    model, modalities = build_dcnet_model(device)
    criterion = AdditiveMarginSoftmax(margin=args.margin, dual=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DCNet] dataset={args.dataset} experts={len(modalities)} vocab={len(vocab)} trainable={trainable/1e6:.1f}M (FP32)", flush=True)

    # Stage 0 baseline
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_dcnet_split(
        args.dataset, model, modalities, vocab, val_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, q_bs=args.q_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_dcnet_split(
        args.dataset, model, modalities, vocab, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, q_bs=args.q_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    del zs_val_corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Stage 0 DCNet Init] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_init", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    def forward_dcnet_loss(ref_imgs: torch.Tensor, tgt_imgs: torch.Tensor, captions: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        txt_t, txt_l = encode_captions_to_glove_batch(captions, vocab)
        txt_t = txt_t.to(device, non_blocking=True)
        txt_l = txt_l.tolist()
        ref_exp = model.get_image_emb({"im_feat": ref_imgs}, istgt=False)
        trg_exp = model.get_image_emb({"im_feat": tgt_imgs}, istgt=True)
        txt_exp = model.get_text_emb(txt_t, txt_l)

        mod_trg = model.get_composition(ref_exp, txt_exp)
        score_comp = model.get_score_matrix(trg_exp, mod_trg, modalities)

        mod_txt = model.get_correction(ref_exp, trg_exp)
        score_corr = model.get_score_matrix(mod_txt, txt_exp, modalities)

        loss_comp = criterion(score_comp)
        loss_corr = criterion(score_corr)

        # Exact upstream cyc3 cycle loss (trainer_joint.py:72-111)
        bs_curr = ref_imgs.size(0)
        d_idx = torch.arange(bs_curr, device=device)
        d_mod_txt = {m: mod_txt[m][d_idx, d_idx] for m in modalities}
        d_mod_txt_mix = {m: 0.5 * (d_mod_txt[m] + txt_exp[m]) for m in modalities}
        aux_src = model.get_composition(ref_exp, d_mod_txt_mix)
        score_cyc = model.get_score_matrix(mod_trg, aux_src, modalities)
        loss_cyc = criterion(score_cyc)

        loss = loss_comp + loss_corr + args.joint_weight * loss_cyc
        return loss, {
            "loss_comp": float(loss_comp.detach().cpu().item()),
            "loss_corr": float(loss_corr.detach().cpu().item()),
            "loss_cyc": float(loss_cyc.detach().cpu().item()),
        }

    # VRAM auto-probe (FP32 exact upstream)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def probe_step(bs: int) -> None:
        model.train()
        model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = ["change the color to red"] * bs
        loss, _ = forward_dcnet_loss(d_ref, d_tgt, d_txt)
        loss.backward()
        model.zero_grad(set_to_none=True)

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=48
    )
    model.load_state_dict(base_state, strict=True)
    del base_state
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[DCNet VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": "dcnet_resnet50",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "vocab_size": len(vocab),
        "hyperparams": {
            "lr": args.lr,
            "gamma": args.gamma,
            "margin": args.margin,
            "joint_weight": args.joint_weight,
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / "best_dcnet.pt"
    torch.save({"epoch": 0, "val_score": best_val_score, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, best_ckpt_path)
    no_improve = 0

    print(f"[DCNet] Starting training: epochs={args.num_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        loss_sums = {"total": 0.0, "loss_comp": 0.0, "loss_corr": 0.0, "loss_cyc": 0.0}
        n_samples, steps_in_window = 0, 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            bs_curr = ref_imgs.size(0)

            loss, comps = forward_dcnet_loss(ref_imgs, tgt_imgs, batch["raw_text"])
            lv = float(loss.detach().cpu().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"Non-finite DCNet loss at ep={epoch} step={step_idx}: {comps}")

            (loss / grad_accum).backward()
            loss_sums["total"] += lv * bs_curr
            for ck, cv in comps.items():
                loss_sums[ck] += cv * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps_in_window = 0

        scheduler.step()
        avg_losses = {k: round(v / max(n_samples, 1), 6) for k, v in loss_sums.items()}
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        cur_lr = float(optimizer.param_groups[0]["lr"])

        val_metrics, _, _ = evaluate_dcnet_split(
            args.dataset, model, modalities, vocab, val_bundle, val_preprocess, device,
            eval_bs=args.eval_bs, q_bs=args.q_bs, num_workers=effective_workers,
        )
        _, val_score = compute_val_selection_score(args.dataset, val_metrics)
        dt = time.time() - ep_t0
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "val_score": val_score, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, best_ckpt_path)
        else:
            no_improve += 1

        print(
            f"[DCNet Ep {epoch:02d}/{args.num_epochs}] loss={avg_losses['total']:.4f} "
            f"(comp={avg_losses['loss_comp']:.3f}, corr={avg_losses['loss_corr']:.3f}, cyc={avg_losses['loss_cyc']:.3f}) | "
            f"lr={cur_lr:.2e} | val {sel_name}={val_score:.4f} (best={best_val_score:.4f}@ep{best_epoch}) | "
            f"vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "train",
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
            print(f"[DCNet] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    final_test_metrics, final_test_preds, _ = evaluate_dcnet_split(
        args.dataset, model, modalities, vocab, test_bundle, val_preprocess, device,
        eval_bs=args.eval_bs, q_bs=args.q_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": "dcnet_resnet50",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_init_test": zs_test_metrics,
        "final_dcnet_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print("\n=== DCNET FINAL TEST SUMMARY ===", flush=True)
    print(f"  Init DCNet Test  : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final DCNet Test : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
