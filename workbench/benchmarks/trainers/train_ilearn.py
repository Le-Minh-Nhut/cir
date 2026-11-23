#!/usr/bin/env python3
"""Unified upstream-faithful trainer for all 5 iLearn-Lab BLIP-2 CIR models:
  - HINT     (ilearn39-adapted)
  - HABIT    (ilearn39-adapted)
  - ConeSep  (ilearn39-adapted)
  - INTENT   (ilearn39-adapted)
  - AirKnow  (airknow310-la)

Run e.g.:
  PYTHONPATH=/home/heheboiz/data/cir \
    workbench/envs/ilearn39-adapted/bin/python \
    workbench/benchmarks/trainers/train_ilearn.py --model hint --dataset edir
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import sys
import types
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "hint": {"dir": "HINT", "lavis_name": "HINT", "default_lr": 2e-5},
    "habit": {"dir": "HABIT", "lavis_name": "HABIT", "default_lr": 2e-5},
    "airknow": {"dir": "AirKnow", "lavis_name": "Blip2QformerCir", "default_lr": 2e-5},
    "conesep": {"dir": "ConeSep", "lavis_name": "ConeSep", "default_lr": 1e-5},
    "intent": {"dir": "INTENT", "lavis_name": "Blip2QformerCir", "default_lr": 2e-5},
}


def _stub_torchaudio_if_missing() -> None:
    """Stub unused LAVIS audio/3D/xinstruct submodules so iLearn BLIP-2 CIR models import cleanly."""
    import importlib.machinery

    def _mk(name: str, attrs: Optional[Dict[str, Any]] = None) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        if attrs:
            for k, v in attrs.items():
                setattr(m, k, v)
        return m

    class _DummyUnused:
        pass

    stubs = {
        "lavis.datasets.builders": {"__all__": []},
        "lavis.tasks": {"__all__": []},
        "lavis.processors.audio_processors": {"BeatsAudioProcessor": _DummyUnused},
        "lavis.processors.ulip_processors": {"ULIPPCProcessor": _DummyUnused},
        "lavis.processors.instruction_text_processors": {"BlipInstructionProcessor": _DummyUnused},
        "lavis.models.blip2_models.blip2_vicuna_xinstruct": {"Blip2VicunaXInstruct": _DummyUnused},
        "lavis.models.blip_diffusion_models.blip_diffusion": {"BlipDiffusion": _DummyUnused},
        "lavis.models.clip_models.model": {"CLIP": _DummyUnused},
        "lavis.models.gpt_models.gpt_dialogue": {"GPTDialogue": _DummyUnused},
        "lavis.models.pnp_vqa_models.pnp_vqa": {"PNPVQA": _DummyUnused},
        "lavis.models.pnp_vqa_models.pnp_unifiedqav2_fid": {"PNPUnifiedQAv2FiD": _DummyUnused},
        "lavis.models.img2prompt_models.img2prompt_vqa": {"Img2PromptVQA": _DummyUnused},
    }
    for mod_name, attrs in stubs.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _mk(mod_name, attrs)


def _patch_bert_root_path() -> None:
    """Redirect hardcoded '/root/bert-base-uncased' in iLearn lavis/blip2.py to 'bert-base-uncased'."""
    _stub_torchaudio_if_missing()
    from transformers import BertConfig, BertTokenizer
    from lavis.models.blip2_models.Qformer import BertConfig as QformerBertConfig, BertLMHeadModel as QformerBertLMHeadModel
    from lavis.models.med import BertLMHeadModel as MedBertLMHeadModel

    seen = set()
    for cls in (BertTokenizer, BertConfig, QformerBertConfig, QformerBertLMHeadModel, MedBertLMHeadModel):
        if cls in seen:
            continue
        seen.add(cls)
        orig = cls.from_pretrained

        def _make_wrapper(fn):
            def _wrapped(pretrained_model_name_or_path, *args, **kwargs):
                if str(pretrained_model_name_or_path).strip() == "/root/bert-base-uncased":
                    pretrained_model_name_or_path = "bert-base-uncased"
                return fn(pretrained_model_name_or_path, *args, **kwargs)
            return staticmethod(_wrapped)

        cls.from_pretrained = _make_wrapper(orig)


def _patch_habit_dynamic_batch(model: nn.Module) -> None:
    """Patch 4 hardcoded `256` batch-size literals & `n_jobs=-1` in HABIT.py cleanly at runtime."""
    import re
    import lavis.models.blip2_models.HABIT as habit_mod

    # 1) Patch DBSCAN: cap n_jobs to 8 AND scale min_samples proportionally when bs < 256
    #    (upstream min_samples=10 assumes BS=256 (~3.9%); keeping 10 at BS=32-48 would mark most samples as noise!)
    _orig_dbscan = habit_mod.DBSCAN

    class _ScaledDBSCAN:
        def __init__(self, *args, **kwargs):
            nj = kwargs.get("n_jobs", 8)
            kwargs["n_jobs"] = min(int(nj) if (nj is not None and int(nj) > 0) else 8, 8)
            self._args = args
            self._kwargs = kwargs

        def fit_predict(self, X, y=None, sample_weight=None):
            kw = dict(self._kwargs)
            n = len(X)
            ms = int(kw.get("min_samples", 5))
            if n < 256 and ms > 2:
                kw["min_samples"] = max(2, int(round(ms * n / 256.0)))
            return _orig_dbscan(*self._args, **kw).fit_predict(X, y=y, sample_weight=sample_weight)

    habit_mod.DBSCAN = _ScaledDBSCAN

    # 2) Exact upstream robust_infoNCE with dynamic `bs` instead of hardcoded `256`
    def _robust_infoNCE_dyn(self, query, target, mask_sample):
        eps = 1e-7
        bs = query.size(0)
        x = torch.matmul(query, target).squeeze().to(query.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        sim_i2t, _ = x.max(-1)
        i2t = (sim_i2t / 0.07).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1 - eps)
        mask_t = torch.as_tensor(mask_sample, dtype=i2t.dtype, device=query.device)
        mask_t = mask_t.unsqueeze(1).expand(-1, bs)
        i2t = i2t * mask_t
        labels = torch.arange(bs, dtype=torch.long, device=query.device)
        mask = torch.ones_like(i2t, dtype=torch.float32, device=i2t.device)
        mask[torch.arange(bs, device=i2t.device), labels] = 0.0
        neg_log = -((1.0 - i2t).clamp(min=eps).log() * mask)
        loss = neg_log.sum() / bs
        row_losses = neg_log.sum(dim=1)
        min_loss_idx = row_losses.argmin()
        return loss, min_loss_idx, sim_i2t

    # 3) Exact upstream cal_Knowledge_Consistency with dynamic `bs` instead of hardcoded `256`
    def _cal_kc_dyn(self, sim_Historical, sim_Current, Historical_soft_labels, Current_soft_labels):
        bs = sim_Current.size(0)
        h_sl = Historical_soft_labels.unsqueeze(1).expand(-1, bs).detach()
        c_sl = Current_soft_labels.unsqueeze(1).expand(-1, bs)
        sim_h_masked = sim_Historical * h_sl * 10.0
        sim_c_masked = sim_Current * c_sl * 10.0
        sim_c_logprob = F.log_softmax(sim_c_masked, dim=1)
        sim_h_prob = F.softmax(sim_h_masked, dim=1)
        return self.KLDivLoss(sim_c_logprob, sim_h_prob)

    # 4) Exact upstream calculate_mk_transition_rate with zero-division guard on mi_anchor
    def _calc_mk_rate_dyn(self, anchor_img, anchor_text, img, text):
        mi_anchor = self.estimate_mutual_knowledge(anchor_img, anchor_text)
        mi_current = self.estimate_mutual_knowledge(img, text)
        mi_text = self.estimate_mutual_knowledge(anchor_img, text)
        mi_image = self.estimate_mutual_knowledge(img, anchor_text)
        denom = mi_anchor if abs(float(mi_anchor)) > 1e-12 else 1e-12
        return (
            abs(mi_anchor - mi_current) / denom,
            abs(mi_anchor - mi_text) / denom,
            abs(mi_anchor - mi_image) / denom,
        )

    model.robust_infoNCE = types.MethodType(_robust_infoNCE_dyn, model)
    model.cal_Knowledge_Consistency = types.MethodType(_cal_kc_dyn, model)
    model.calculate_mk_transition_rate = types.MethodType(_calc_mk_rate_dyn, model)

    # 5) Patch HABIT.forward: rewrite only `torch.ones(256, ...)` -> `torch.ones(image.size(0), ...)`
    src = inspect.getsource(habit_mod.HABIT.forward)
    lines = src.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    dedented = "\n".join(l[indent:] for l in lines)
    dedented = re.sub(r"torch\.ones\(\s*256\s*,", "torch.ones(image.size(0),", dedented)
    ns = dict(habit_mod.__dict__)
    exec(dedented, ns)
    model.forward = types.MethodType(ns["forward"], model)


def _patch_conesep_nan_guard() -> None:
    """Guard ConeSep.fidelity_measure against 0/0 NaN when all sims < boundary_estimation."""
    import lavis.models.blip2_models.ConeSep as cs_mod

    def _safe_fidelity(x, boundary_estimation):
        x = torch.clamp(x - boundary_estimation, min=0.0)
        xmax = x.max()
        if float(xmax.item()) <= 1e-12:
            return torch.zeros_like(x)
        return x / xmax

    cs_mod.fidelity_measure = _safe_fidelity


def set_bn_eval(m: nn.Module) -> None:
    if m.__class__.__name__.find("BatchNorm") != -1:
        m.eval()


@torch.no_grad()
def extract_ilearn_corpus_features(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_corpus, 32, 256] target features (stored on CPU fp32)."""
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
    feats_list: List[torch.Tensor] = []
    ids: List[str] = []
    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        # Upstream extract_retrieval_target returns (B, 256, 32) -> transpose to (B, 32, 256)
        f = model.extract_retrieval_target(imgs)
        if f.dim() == 3 and f.shape[1] == 256 and f.shape[2] == 32:
            f = f.permute(0, 2, 1)
        feats_list.append(f.float().cpu())
        ids.extend(batch["image_id"])
    return torch.cat(feats_list, dim=0), ids


@torch.no_grad()
def extract_ilearn_query_features(
    model: nn.Module,
    bundle: SplitBundle,
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 64,
    num_workers: int = 16,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract normalized [N_query, 256] composed query features."""
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
    feats_list: List[torch.Tensor] = []
    qids: List[str] = []
    for batch in dl:
        imgs = batch["ref_image"].to(device, non_blocking=True)
        mods = [txt_proc(c) for c in batch["raw_text"]]
        # Upstream extract_retrieval_compose returns (B, 1, 1, 256) -> reshape to (B, 256)
        f = model.extract_retrieval_compose(imgs, mods)
        f = f.reshape(imgs.size(0), -1)
        feats_list.append(f.float().cpu())
        qids.extend(batch["query_id"])
    return torch.cat(feats_list, dim=0), qids


@torch.no_grad()
def rank_multitoken_max(
    query_feats: torch.Tensor,
    query_ids: List[str],
    target_feats: torch.Tensor,
    corpus_ids: List[str],
    device: torch.device,
    top_k: int = 50,
    q_chunk: int = 128,
    c_chunk: int = 4096,
) -> Dict[str, List[str]]:
    """Exact max-over-32-tokens cosine scoring (immune to B==1 squeeze bug)."""
    if not torch.isfinite(query_feats).all() or not torch.isfinite(target_feats).all():
        raise ValueError("Non-finite embeddings detected in rank_multitoken_max")
    n_q = query_feats.shape[0]
    n_c = target_feats.shape[0]
    k = min(int(top_k), n_c)
    results: Dict[str, List[str]] = {}
    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        qf = query_feats[qs:qe].to(device, non_blocking=True)  # [B, 256]
        sims_row = np.empty((qe - qs, n_c), dtype=np.float32)
        for cs in range(0, n_c, c_chunk):
            ce = min(cs + c_chunk, n_c)
            tf = target_feats[cs:ce].to(device, non_blocking=True)  # [Nc, 32, 256]
            sim_blk = torch.einsum("bd,nkd->bnk", qf, tf).max(dim=-1).values
            sims_row[:, cs:ce] = sim_blk.cpu().numpy()
        order = np.argsort(-sims_row, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[qs:qe]):
            results[str(qid)] = [str(corpus_ids[ci]) for ci in order[row_i]]
    return results


def evaluate_ilearn_split(
    dataset: str,
    model: nn.Module,
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
        target_feats, corpus_ids = extract_ilearn_corpus_features(
            model, bundle, preprocess, device, eval_bs=eval_bs, num_workers=num_workers
        )
    else:
        target_feats, corpus_ids = cached_corpus
    query_feats, query_ids = extract_ilearn_query_features(
        model, bundle, preprocess, txt_proc, device, eval_bs=eval_bs, num_workers=num_workers
    )
    preds = rank_multitoken_max(query_feats, query_ids, target_feats, corpus_ids, device=device, top_k=top_k)
    if dataset == "edir":
        metrics = evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    else:
        metrics = evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)
    return metrics, preds, (target_feats, corpus_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train iLearn BLIP-2 model (HINT/HABIT/AirKnow/ConeSep/INTENT)")
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eki-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-ratio", type=float, default=1.25)
    parser.add_argument("--positive-mode", type=str, default="random", choices=["random", "expand"])
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--eval-bs", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-micro-bs", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    lr = args.lr if args.lr is not None else spec["default_lr"]
    repo_dir = REPO_ROOT / "workbench" / "third_party" / spec["dir"]
    sys.path.insert(0, str(repo_dir))

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_workers = max(0, min(args.num_workers, 16))

    manifest = verify_split_integrity(args.dataset, split_tag=args.split_tag)
    train_bundle = load_split_bundle(args.dataset, "train", corpus_scope="full", split_tag=args.split_tag)
    val_bundle = load_split_bundle(args.dataset, "val", corpus_scope=args.val_corpus_scope, split_tag=args.split_tag)
    test_bundle = load_split_bundle(args.dataset, "test", corpus_scope="full", split_tag=args.split_tag)

    run_dir = args.run_dir or default_run_dir(args.dataset, f"{args.model}_blip2", args.split_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    _patch_bert_root_path()
    from data_utils import targetpad_transform  # noqa: E402
    from lavis.models import load_model_and_preprocess  # noqa: E402

    model, _, txt_processors = load_model_and_preprocess(
        name=spec["lavis_name"], model_type="pretrain", is_eval=False, device=device
    )
    txt_proc = txt_processors["eval"]
    preprocess = targetpad_transform(args.target_ratio, 224)

    # Model-specific exact upstream init / safe patches
    if args.model == "conesep":
        model.prompt_load()
        _patch_conesep_nan_guard()
    elif args.model == "habit":
        _patch_habit_dynamic_batch(model)
    elif args.model == "airknow":
        model.alpha = 0.7

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{args.model.upper()}] dataset={args.dataset} trainable={trainable/1e6:.1f}M / total={total/1e6:.1f}M", flush=True)

    # Scorer parity self-check against exact upstream np.matmul(q, t).squeeze().max(-1)
    model.eval()
    with torch.no_grad():
        di = torch.randn(3, 3, 224, 224, device=device)
        q_up = model.extract_retrieval_compose(di, [txt_proc("a red shirt")] * 3).float().cpu().numpy()
        t_up = model.extract_retrieval_target(di).float().cpu().numpy()
        ref_sim = np.matmul(q_up, t_up).squeeze().max(-1)
        q_ours = torch.from_numpy(q_up).reshape(3, -1).to(device)
        t_ours = torch.from_numpy(t_up).permute(0, 2, 1).to(device)
        our_sim = torch.einsum("bd,nkd->bnk", q_ours, t_ours).max(dim=-1).values.cpu().numpy()
        max_diff = float(np.max(np.abs(ref_sim - our_sim)))
        assert max_diff < 1e-5, f"{args.model} scorer parity failed: {max_diff}"
        print(f"[{args.model.upper()} Scorer Parity Self-Check] max_abs_diff = {max_diff:.2e} (OK)", flush=True)
        del di, q_up, t_up, ref_sim, our_sim, q_ours, t_ours
        torch.cuda.empty_cache()

    # Stage 0 zero-shot baseline
    t0 = time.time()
    zs_val_metrics, _, zs_val_corpus = evaluate_ilearn_split(
        args.dataset, model, val_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )
    sel_name, zs_val_score = compute_val_selection_score(args.dataset, zs_val_metrics)
    zs_test_metrics, _, _ = evaluate_ilearn_split(
        args.dataset, model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
        cached_corpus=zs_val_corpus if args.val_corpus_scope == "full" else None,
    )
    del zs_val_corpus
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Stage 0 {args.model.upper()}] val {sel_name}={zs_val_score:.4f} | "
        f"test={zs_test_metrics['overall'].get('Recall@1', zs_test_metrics['overall'].get('mAP@10', 0)):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    append_jsonl(
        history_path,
        {"stage": "stage0_zeroshot", "epoch": 0, "val_selection_metric": sel_name, "val_selection_score": zs_val_score, "val_metrics": zs_val_metrics},
    )

    # Helper computing model-specific training loss for both VRAM probe & real epochs
    def compute_model_loss(
        ref_imgs: torch.Tensor,
        tgt_imgs: torch.Tensor,
        captions: List[str],
        epoch_zero_idx: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        samples: Dict[str, Any] = {"image": ref_imgs, "target": tgt_imgs, "text_input": captions}
        if args.model == "hint":
            ld = model(samples, device)
            loss = ld["loss_stu_rank"]
            return loss, {"loss_stu_rank": float(loss.detach().cpu().item())}
        if args.model == "intent":
            after_warmup = epoch_zero_idx >= 2
            ld = model(samples, device, after_warmup=after_warmup)
            if after_warmup:
                loss = ld["loss_stu"] + 0.5 * ld["loss_odl"]
            else:
                loss = ld["loss_stu"] + 0.5 * ld["loss_odl"] + 1.0 * ld["loss_caco"]
            return loss, {k: float(v.detach().cpu().item()) for k, v in ld.items() if torch.is_tensor(v)}
        if args.model == "conesep":
            model.epoch = epoch_zero_idx
            ld = model(
                samples,
                device,
                if_negative=(epoch_zero_idx < 2),
                omega=0.5,
                target_forgetting_loss_weight=0.5,
                target_oriented_weight=0.5,
                query_oriented_weight=0.8,
            )
            loss = sum(ld.values())
            return loss, {k: float(v.detach().cpu().item()) for k, v in ld.items() if torch.is_tensor(v)}
        if args.model == "habit":
            old_sl = (extra or {}).get("old_soft_labels", [])
            old_sim = (extra or {}).get("old_sim", [])
            ld = model(samples, device, old_sl, epoch_zero_idx, old_sim)
            loss = ld["loss_rank"] + 0.1 * ld["loss_soft"] + 10.0 * ld["loss_kl"]
            return loss, {k: float(v.detach().cpu().item()) for k, v in ld.items() if torch.is_tensor(v)}
        if args.model == "airknow":
            is_eki = (extra or {}).get("eki", False)
            if is_eki:
                samples["labels"] = extra["labels"]
                samples["eki"] = True
                ld = model(samples, device)
                loss = ld["loss_eki"]
                return loss, {"loss_eki": float(loss.detach().cpu().item())}
            samples["labels"] = None
            samples["eki"] = False
            ld = model(samples, device)
            loss = ld["loss_align"] + 0.6 * ld["loss_recon"]
            return loss, {k: float(v.detach().cpu().item()) for k, v in ld.items() if torch.is_tensor(v)}
        raise ValueError(args.model)

    # VRAM auto-probe (use worst-case memory phase: epoch 0 for INTENT/HINT/HABIT, epoch 2 for ConeSep, Stage 2 for AirKnow)
    base_non_vit = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("visual_encoder.")
    }
    probe_epoch = 2 if args.model == "conesep" else 0
    # HABIT DBSCAN uses min_samples=10, so require bs >= 16; ConeSep/HINT are heavier so cap max_micro_bs
    max_probe_bs = min(args.max_micro_bs, 32 if args.model in ("hint", "conesep", "intent") else 48)

    def probe_step(bs: int) -> None:
        model.train()
        model.apply(set_bn_eval)
        model.zero_grad(set_to_none=True)
        d_ref = torch.randn(bs, 3, 224, 224, device=device)
        d_tgt = torch.randn(bs, 3, 224, 224, device=device)
        d_txt = [txt_proc("change the color to bright yellow")] * bs
        if args.model == "habit":
            model.soft_labels = []
            model.sim = []
        # Note: INTENT colorful_spectrum_mix runs torch.fft.fft2 on fp32 images BEFORE autocast inside model or outside
        if args.model == "intent":
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss, _ = compute_model_loss(d_ref.float(), d_tgt.float(), d_txt, probe_epoch)
        else:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss, _ = compute_model_loss(d_ref, d_tgt, d_txt, probe_epoch)
        loss.backward()
        model.zero_grad(set_to_none=True)
        if args.model == "habit":
            model.soft_labels = []
            model.sim = []

    micro_bs, grad_accum, probe_log = probe_batch_size_and_accum(
        probe_step, default_effective_bs=16, allow_upgrade_to_32=True, max_micro_bs=max_probe_bs
    )
    model.load_state_dict(base_non_vit, strict=False)
    del base_non_vit
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{args.model.upper()} VRAM Probe] Selected micro_bs={micro_bs}, grad_accum={grad_accum}: {probe_log}", flush=True)

    config_payload = {
        "model_id": f"{args.model}_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "manifest_counts": manifest["counts"],
        "hyperparams": {
            "model": args.model,
            "lr": lr,
            "eki_lr": args.eki_lr if args.model == "airknow" else None,
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

    # Training DataLoader (AirKnow requests 1 explicit hard negative per sample for Stage-1 EKI warmup!)
    need_hneg = 1 if args.model == "airknow" else 0
    train_ds = CIRTrainingDataset(
        train_bundle,
        image_transform=preprocess,
        positive_mode=args.positive_mode,
        num_hard_negatives=need_hneg,
        load_images=True,
        seed=args.seed,
    )
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_bs,
        shuffle=(args.model != "habit"),  # HABIT aligns cross-epoch soft labels & sim matrices by batch index
        drop_last=True,
        num_workers=effective_workers,
        pin_memory=True,
        prefetch_factor=4 if effective_workers > 0 else None,
        collate_fn=collate_train_batch,
        worker_init_fn=seed_worker if effective_workers > 0 else None,
        generator=gen,
    )

    if args.model == "airknow":
        base_params = [p for n, p in model.named_parameters() if "eki" not in n and p.requires_grad]
        eki_params = [p for n, p in model.named_parameters() if "eki" in n and p.requires_grad]
        optimizer = optim.AdamW(
            [
                {"params": base_params, "lr": lr, "weight_decay": args.weight_decay},
                {"params": eki_params, "lr": args.eki_lr, "weight_decay": args.weight_decay},
            ]
        )
    else:
        optimizer = optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "lr": lr,
                    "betas": (0.9, 0.98),
                    "eps": 1e-7,
                    "weight_decay": args.weight_decay,
                }
            ]
        )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_score = zs_val_score
    best_epoch = 0
    best_ckpt_path = run_dir / f"best_{args.model}.pt"

    def save_ckpt(epoch: int, score: float) -> None:
        compact = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("visual_encoder.")}
        torch.save({"epoch": epoch, "val_score": score, "state_dict_non_vit": compact}, best_ckpt_path)

    save_ckpt(0, best_val_score)

    # AirKnow Stage 1: 3-epoch EKI proxy warmup using genuine clean (pos=1) vs explicit hard-neg (neg=0)
    if args.model == "airknow":
        eki_warmup_epochs = 3
        print(f"[AIRKNOW Stage 1] EKI Proxy warmup ({eki_warmup_epochs} epochs using clean pos=1 vs hard-neg=0)...", flush=True)
        for wep in range(1, eki_warmup_epochs + 1):
            t_w = time.time()
            train_ds.set_epoch(1000 + wep)
            model.train()
            model.apply(set_bn_eval)
            optimizer.zero_grad(set_to_none=True)
            eki_loss_sum, eki_n = 0.0, 0
            for batch in train_loader:
                ref_imgs = batch["ref_image"].to(device, non_blocking=True)
                pos_imgs = batch["target_image"].to(device, non_blocking=True)
                neg_imgs = batch["hard_neg_images"][:, 0].to(device, non_blocking=True)
                caps = [txt_proc(c) for c in batch["raw_text"]]
                bs_curr = ref_imgs.size(0)
                # Balanced 50% clean (label=1) + 50% explicit hard-negative (label=0)
                half = bs_curr // 2
                mixed_tgt = torch.cat([pos_imgs[:half], neg_imgs[half:]], dim=0)
                labels = torch.cat(
                    [torch.ones(half, dtype=torch.long, device=device), torch.zeros(bs_curr - half, dtype=torch.long, device=device)],
                    dim=0,
                )
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    loss, _ = compute_model_loss(ref_imgs, mixed_tgt, caps, 0, extra={"eki": True, "labels": labels})
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                eki_loss_sum += float(loss.detach().cpu().item()) * bs_curr
                eki_n += bs_curr
            print(f"[AIRKNOW EKI Warmup {wep}/{eki_warmup_epochs}] eki_loss={eki_loss_sum/max(eki_n,1):.4f} ({time.time()-t_w:.1f}s)", flush=True)

    # Main training loop
    no_improve = 0
    soft_label_old: Any = []
    similarity_matrix_old: Any = []

    print(f"[{args.model.upper()}] Starting main training: epochs={args.num_epochs}, batches/ep={len(train_loader)}", flush=True)
    for epoch in range(1, args.num_epochs + 1):
        ep_t0 = time.time()
        epoch_zero = epoch - 1
        train_ds.set_epoch(epoch)
        model.train()
        model.apply(set_bn_eval)
        if args.model == "habit":
            model.soft_labels = []
            model.sim = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        comp_sums: Dict[str, float] = {}
        n_samples = 0
        steps_in_window = 0

        for step_idx, batch in enumerate(train_loader):
            ref_imgs = batch["ref_image"].to(device, non_blocking=True)
            tgt_imgs = batch["target_image"].to(device, non_blocking=True)
            captions = [txt_proc(c) for c in batch["raw_text"]]
            bs_curr = ref_imgs.size(0)

            extra: Dict[str, Any] = {}
            if args.model == "habit" and epoch_zero >= 1 and len(similarity_matrix_old) > step_idx:
                start_i = step_idx * bs_curr
                end_i = start_i + bs_curr
                extra["old_soft_labels"] = soft_label_old[start_i:end_i]
                extra["old_sim"] = similarity_matrix_old[step_idx]

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss, comp_dict = compute_model_loss(ref_imgs, tgt_imgs, captions, epoch_zero, extra=extra)

            lv = float(loss.detach().cpu().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"Non-finite {args.model} loss at epoch={epoch} step={step_idx}: {comp_dict}")

            scaler.scale(loss / grad_accum).backward()
            total_loss_sum += lv * bs_curr
            for ck, cv in comp_dict.items():
                comp_sums[ck] = comp_sums.get(ck, 0.0) + cv * bs_curr
            n_samples += bs_curr
            steps_in_window += 1

            if steps_in_window == grad_accum or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps_in_window = 0

        if args.model == "habit":
            soft_label_old = model.get_soft_labels()
            similarity_matrix_old = model.get_similarity_matrix()

        avg_total = total_loss_sum / max(n_samples, 1)
        avg_comps = {k: round(v / max(n_samples, 1), 6) for k, v in comp_sums.items()}
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

        val_metrics, _, _ = evaluate_ilearn_split(
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
            save_ckpt(epoch, val_score)
        else:
            no_improve += 1

        print(
            f"[{args.model.upper()} Ep {epoch:02d}/{args.num_epochs}] loss={avg_total:.4f} {avg_comps} | "
            f"val {sel_name}={val_score:.4f} (best={best_val_score:.4f}@ep{best_epoch}) | "
            f"vram={peak_mib:.0f}MiB | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(
            history_path,
            {
                "stage": "train",
                "epoch": epoch,
                "train_loss": round(avg_total, 6),
                "train_loss_components": avg_comps,
                "val_selection_metric": sel_name,
                "val_selection_score": val_score,
                "best_val_score": best_val_score,
                "peak_vram_mib": round(peak_mib, 1),
                "epoch_sec": round(dt, 2),
                "val_metrics": val_metrics,
            },
        )
        if no_improve >= args.patience:
            print(f"[{args.model.upper()}] Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Reload best val checkpoint & evaluate ONCE on full test corpus
    ckpt = torch.load(best_ckpt_path, map_location=device)
    msg = model.load_state_dict(ckpt["state_dict_non_vit"], strict=False)
    non_vit_missing = [k for k in msg.missing_keys if not k.startswith("visual_encoder.")]
    assert not non_vit_missing, f"Unexpected missing non-ViT keys: {non_vit_missing}"
    assert not msg.unexpected_keys, f"Unexpected keys in checkpoint: {msg.unexpected_keys}"

    final_test_metrics, final_test_preds, _ = evaluate_ilearn_split(
        args.dataset, model, test_bundle, preprocess, txt_proc, device,
        eval_bs=args.eval_bs, num_workers=effective_workers,
    )

    result_payload = {
        "model_id": f"{args.model}_blip2",
        "dataset": args.dataset,
        "split_tag": args.split_tag,
        "finished_at": utc_now_iso(),
        "val_selection_metric": sel_name,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "stage0_zeroshot_test": zs_test_metrics,
        f"final_{args.model}_test": final_test_metrics,
    }
    write_json(run_dir / "test_results.json", result_payload)
    write_json(run_dir / "test_predictions_top50.json", final_test_preds)

    print(f"\n=== {args.model.upper()} FINAL TEST SUMMARY ===", flush=True)
    print(f"  Zero-Shot Test : {zs_test_metrics['overall']}", flush=True)
    print(f"  Final Test     : {final_test_metrics['overall']}", flush=True)
    print(f"  Saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
