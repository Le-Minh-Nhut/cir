#!/usr/bin/env python3
"""Upstream-faithful QuRe (ICML 2025, jaehyun-kwak/QuRe) trainer for FashionIQ & CIRR checkpoints.

Because the official Google Drive checkpoint folders in jaehyun-kwak/QuRe are empty
(confirmed via jaehyun-kwak/QuRe#3), this script reproduces the official FashionIQ and CIRR
checkpoints strictly following:
  - workbench/third_party/QuRe/configs/{fashionIQ,cirr}/train.json
  - workbench/third_party/QuRe/train_qure.py
  - workbench/third_party/QuRe/data/{fashionIQ,cirr}.py
  - workbench/third_party/QuRe/evaluate_qure/eval_utils.py

Without modifying any file inside workbench/third_party/QuRe.

Run with:
  PYTHONPATH=/home/heheboiz/data/cir:/home/heheboiz/data/cir/workbench/third_party/QuRe \
    workbench/envs/qure310-sm120/bin/python \
    workbench/benchmarks/trainers/train_qure_upstream_ckpt.py --dataset fashioniq
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean
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

from workbench.benchmarks.trainers.common import (  # noqa: E402
    append_jsonl,
    get_git_sha,
    seed_everything,
    utc_now_iso,
    write_json,
)


def cosine_lr_schedule(optimizer: torch.optim.Optimizer, epoch: int, max_epoch: int, init_lr: float, min_lr: float) -> float:
    """Exact upstream QuRe cosine LR schedule (train_qure.py lines 101-105)."""
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def format_fiq_caption(captions: List[str]) -> str:
    """Exact upstream FashionIQ caption join (data/fashionIQ.py line 105)."""
    return captions[0].strip(".?, ").capitalize() + " and " + captions[1].strip(".?, ")


def select_qure_hard_negatives_for_query(
    sorted_indices: np.ndarray,
    sorted_scores: np.ndarray,
    target_positions: List[int],
    candidate_names: List[str],
    forbidden_names: set,
    consecutive_threshold: int = 1,
) -> Tuple[List[str], bool]:
    """Exact upstream QuRe hard-negative boundary detection (between two steepest drops after target).

    Returns (hard_negative_names, used_fallback).
    """
    n = len(sorted_indices)
    # Match upstream: locate target in descending sorted list
    pos_map = {int(idx): rank for rank, idx in enumerate(sorted_indices)}
    t_ranks = [pos_map[p] for p in target_positions if p in pos_map]
    target_sorted_index = min(t_ranks) if t_ranks else 0

    start_index = target_sorted_index + 1
    scores_after_target = sorted_scores[start_index:]

    start_of_hn: Optional[int] = None
    end_of_hn: Optional[int] = None
    if len(scores_after_target) >= 3:
        differences = np.diff(scores_after_target)
        diff_sorted_indices = np.argsort(differences, kind="stable")
        idx = 0
        while idx < len(diff_sorted_indices) - 1:
            idx1 = int(diff_sorted_indices[idx])
            idx2 = int(diff_sorted_indices[idx + 1])
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            if idx2 - idx1 > consecutive_threshold:
                start_of_hn = idx1 + start_index
                end_of_hn = idx2 + start_index
                break
            idx += 1

    if start_of_hn is not None and end_of_hn is not None:
        raw_indices = sorted_indices[start_of_hn + 1 : end_of_hn + 1]
        names = list(dict.fromkeys(candidate_names[int(i)] for i in raw_indices if candidate_names[int(i)] not in forbidden_names))
        if names:
            return names, False

    # Safe deterministic fallback if target sits at very bottom or drop window yields only target duplicates
    after = [candidate_names[int(i)] for i in sorted_indices[start_index : min(n, start_index + 50)] if candidate_names[int(i)] not in forbidden_names]
    names = list(dict.fromkeys(after))
    if not names:
        top_any = [candidate_names[int(i)] for i in sorted_indices[: min(n, 100)] if candidate_names[int(i)] not in forbidden_names]
        names = list(dict.fromkeys(top_any))
    return names, True


@torch.no_grad()
def encode_qure_images(
    model: nn.Module,
    image_paths: List[Path],
    image_names: List[str],
    preprocess: Callable,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    """Exact Blip2QuRe.extract_target_features -> [N, 32, 256] L2-normalized CPU fp32."""
    model.eval()

    class _ImgDS(Dataset):
        def __len__(self) -> int:
            return len(image_paths)

        def __getitem__(self, i: int):
            with PIL.Image.open(image_paths[i]) as im:
                tensor = preprocess(im.convert("RGB"))
            return tensor, image_names[i]

    dl = DataLoader(
        _ImgDS(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    feats: List[torch.Tensor] = []
    names_out: List[str] = []
    for imgs, nms in dl:
        imgs = imgs.to(device, non_blocking=True)
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
        feats.append(bf.float().cpu())
        names_out.extend(nms)
    return torch.cat(feats, dim=0), names_out


@torch.no_grad()
def encode_qure_queries(
    model: nn.Module,
    ref_paths: List[Path],
    captions: List[str],
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
) -> torch.Tensor:
    """Exact Blip2QuRe query feature extraction -> [N, 256] L2-normalized CPU fp32."""
    model.eval()

    class _QDS(Dataset):
        def __len__(self) -> int:
            return len(ref_paths)

        def __getitem__(self, i: int):
            with PIL.Image.open(ref_paths[i]) as im:
                tensor = preprocess(im.convert("RGB"))
            return tensor, captions[i]

    dl = DataLoader(
        _QDS(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    feats: List[torch.Tensor] = []
    for imgs, caps in dl:
        proc_caps = [txt_proc(c) for c in caps]
        text_tokens = model.tokenizer(
            proc_caps,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(device)
        imgs = imgs.to(device, non_blocking=True)
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
        q_feat = F.normalize(model.text_proj(q_embeds).mean(dim=1), dim=-1)
        feats.append(q_feat.float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def compute_score_matrix_chunked(
    query_feats: torch.Tensor,
    target_feats: torch.Tensor,
    device: torch.device,
    q_chunk: int = 256,
    c_chunk: int = 4096,
) -> np.ndarray:
    """Exact equivalent of Blip2QuRe.score(query_feats, target_feats) without OOM."""
    n_q = query_feats.shape[0]
    n_c = target_feats.shape[0]
    out = np.empty((n_q, n_c), dtype=np.float32)
    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        qf = query_feats[qs:qe].to(device, non_blocking=True)
        for cs in range(0, n_c, c_chunk):
            ce = min(cs + c_chunk, n_c)
            tf = target_feats[cs:ce].to(device, non_blocking=True)
            sim = torch.einsum("ik,jlk->ijl", qf, tf).max(dim=-1).values
            out[qs:qe, cs:ce] = sim.cpu().numpy()
    return out


class QuReTripletTrainDataset(Dataset):
    """Training dataset with upstream QuRe random warmup + periodic hard-negative pool sampling."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        name_to_path: Dict[str, Path],
        domain_target_pools: Dict[str, List[str]],
        preprocess: Callable,
    ):
        self.records = records
        self.name_to_path = name_to_path
        self.domain_target_pools = domain_target_pools
        self.preprocess = preprocess
        self.use_hard_negatives = False
        self.hard_images: List[List[str]] = []
        # Deterministic per-epoch sampling: each epoch we pick 1 negative per triplet cleanly in main process
        # so multi-worker DataLoader shares history cleanly (fixes upstream multi-worker copy-on-write bug!).
        self.epoch_negatives: List[str] = []
        self.negative_selection_history: Dict[int, List[str]] = {}

    def prepare_epoch_negatives(self, rng: random.Random) -> None:
        chosen: List[str] = []
        for idx, rec in enumerate(self.records):
            target_name = rec["target"]
            if not self.use_hard_negatives or not self.hard_images:
                pool = self.domain_target_pools[rec["domain"]]
                neg = rng.choice(pool)
                while neg == target_name:
                    neg = rng.choice(pool)
                chosen.append(neg)
            else:
                hn_list = self.hard_images[idx]
                hist = self.negative_selection_history.setdefault(idx, [])
                if len(hist) >= len(hn_list):
                    hist.clear()
                avail = [n for n in hn_list if n not in hist]
                if not avail:
                    avail = hn_list
                neg = rng.choice(avail)
                hist.append(neg)
                chosen.append(neg)
        self.epoch_negatives = chosen

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        with PIL.Image.open(self.name_to_path[rec["reference"]]) as im:
            ref_img = self.preprocess(im.convert("RGB"))
        with PIL.Image.open(self.name_to_path[rec["target"]]) as im:
            tar_img = self.preprocess(im.convert("RGB"))
        neg_name = self.epoch_negatives[idx]
        with PIL.Image.open(self.name_to_path[neg_name]) as im:
            neg_img = self.preprocess(im.convert("RGB"))
        return ref_img, tar_img, neg_img, rec["caption"]


def load_fashioniq_data(fiq_root: Path):
    dress_types = ["dress", "shirt", "toptee"]
    img_dir = fiq_root / "images"
    name_to_path: Dict[str, Path] = {}

    train_records: List[Dict[str, Any]] = []
    domain_target_pools: Dict[str, List[str]] = {}
    domain_triplet_indices: Dict[str, List[int]] = {}

    for dt in dress_types:
        trips = json.loads((fiq_root / "captions" / f"cap.{dt}.train.json").read_text(encoding="utf-8"))
        start = len(train_records)
        pool: List[str] = []
        for t in trips:
            ref = t["candidate"]
            tar = t["target"]
            name_to_path[ref] = img_dir / f"{ref}.png"
            name_to_path[tar] = img_dir / f"{tar}.png"
            pool.append(tar)
            train_records.append({
                "reference": ref,
                "target": tar,
                "caption": format_fiq_caption(t["captions"]),
                "domain": dt,
            })
        domain_target_pools[dt] = pool
        domain_triplet_indices[dt] = list(range(start, len(train_records)))

    val_by_cat: Dict[str, Dict[str, Any]] = {}
    for dt in dress_types:
        val_trips = json.loads((fiq_root / "captions" / f"cap.{dt}.val.json").read_text(encoding="utf-8"))
        val_imgs = json.loads((fiq_root / "image_splits" / f"split.{dt}.val.json").read_text(encoding="utf-8"))
        val_by_cat[dt] = {
            "index_names": val_imgs,
            "index_paths": [img_dir / f"{n}.png" for n in val_imgs],
            "ref_paths": [img_dir / f"{t['candidate']}.png" for t in val_trips],
            "ref_names": [t["candidate"] for t in val_trips],
            "target_names": [t["target"] for t in val_trips],
            "captions": [format_fiq_caption(t["captions"]) for t in val_trips],
            "raw_captions": [t["captions"] for t in val_trips],
        }
    return train_records, name_to_path, domain_target_pools, domain_triplet_indices, val_by_cat


def load_cirr_data(cirr_root: Path):
    train_trips = json.loads((cirr_root / "captions" / "cap.rc2.train.json").read_text(encoding="utf-8"))
    train_map = json.loads((cirr_root / "image_splits" / "split.rc2.train.json").read_text(encoding="utf-8"))
    name_to_path = {k: (cirr_root / v) for k, v in train_map.items()}
    train_image_names = list(train_map.keys())

    train_records: List[Dict[str, Any]] = []
    for t in train_trips:
        train_records.append({
            "reference": t["reference"],
            "target": t["target_hard"],
            "caption": t["caption"],
            "domain": "cirr",
        })
    domain_target_pools = {"cirr": train_image_names}
    domain_triplet_indices = {"cirr": list(range(len(train_records)))}

    val_trips = json.loads((cirr_root / "captions" / "cap.rc2.val.json").read_text(encoding="utf-8"))
    val_map = json.loads((cirr_root / "image_splits" / "split.rc2.val.json").read_text(encoding="utf-8"))
    val_index_names = list(val_map.keys())
    val_data = {
        "index_names": val_index_names,
        "index_paths": [cirr_root / val_map[n] for n in val_index_names],
        "ref_paths": [cirr_root / val_map[t["reference"]] for t in val_trips],
        "ref_names": [t["reference"] for t in val_trips],
        "target_names": [t["target_hard"] for t in val_trips],
        "captions": [t["caption"] for t in val_trips],
        "group_members": [t["img_set"]["members"] for t in val_trips],
        "pair_ids": [t["pairid"] for t in val_trips],
    }
    return train_records, name_to_path, domain_target_pools, domain_triplet_indices, val_data


@torch.no_grad()
def rerank_hard_negatives(
    dataset_name: str,
    model: nn.Module,
    train_ds: QuReTripletTrainDataset,
    domain_triplet_indices: Dict[str, List[int]],
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    consecutive_threshold: int = 1,
    eval_bs: int = 128,
    num_workers: int = 8,
) -> Dict[str, float]:
    """Exact upstream QuRe rerank_score with unique-image encoding optimization."""
    t0 = time.time()
    model.eval()
    ref_paths = [train_ds.name_to_path[r["reference"]] for r in train_ds.records]
    captions = [r["caption"] for r in train_ds.records]
    q_feats = encode_qure_queries(model, ref_paths, captions, preprocess, txt_proc, device, batch_size=eval_bs, num_workers=num_workers)

    # Encode all unique candidate target images ONCE
    unique_cand_names = list(dict.fromkeys(
        n for pool in train_ds.domain_target_pools.values() for n in pool
    ))
    unique_cand_paths = [train_ds.name_to_path[n] for n in unique_cand_names]
    u_feats, _ = encode_qure_images(model, unique_cand_paths, unique_cand_names, preprocess, device, batch_size=eval_bs, num_workers=num_workers)
    u_idx_map = {n: i for i, n in enumerate(unique_cand_names)}

    hard_images: List[List[str]] = [[] for _ in range(len(train_ds.records))]
    fallback_count = 0

    for dom, q_indices in domain_triplet_indices.items():
        # Match upstream pool exactly:
        # FashionIQ: [triplet['target'] for triplet in domain_triplets[dt]] (deduplicating scores preserves exact unique image ranking?
        # Wait: upstream FashionIQ kept duplicate targets in pool, so let's match exact candidate list!)
        cand_names = train_ds.domain_target_pools[dom]
        # Map unique features -> domain candidate features efficiently by indexing unique scores!
        dom_unique_names = list(dict.fromkeys(cand_names))
        dom_u_tensor_idx = torch.tensor([u_idx_map[n] for n in dom_unique_names], dtype=torch.long)
        dom_q_feats = q_feats[q_indices]
        dom_u_feats = u_feats[dom_u_tensor_idx]

        # Compute [N_dom_q, N_dom_unique] scores on GPU
        u_scores = compute_score_matrix_chunked(dom_q_feats, dom_u_feats, device=device)

        if len(dom_unique_names) == len(cand_names):
            # CIRR: all unique!
            dom_scores = u_scores
            cand_list = dom_unique_names
        else:
            # FashionIQ: expand columns to match exact upstream domain_triplets target list
            u_local_map = {n: i for i, n in enumerate(dom_unique_names)}
            col_expand = np.array([u_local_map[n] for n in cand_names], dtype=np.int64)
            dom_scores = u_scores[:, col_expand]
            cand_list = cand_names

        # Map name -> first index in cand_list (matches upstream list.index(target_name))
        first_idx_of: Dict[str, int] = {}
        for ci, cn in enumerate(cand_list):
            if cn not in first_idx_of:
                first_idx_of[cn] = ci

        for local_i, global_q_idx in enumerate(q_indices):
            row = dom_scores[local_i]
            order = np.argsort(-row, kind="stable")
            sorted_scores = row[order]
            t_name = train_ds.records[global_q_idx]["target"]
            t_pos = [first_idx_of[t_name]]
            hn, used_fb = select_qure_hard_negatives_for_query(
                sorted_indices=order,
                sorted_scores=sorted_scores,
                target_positions=t_pos,
                candidate_names=cand_list,
                forbidden_names={t_name},
                consecutive_threshold=consecutive_threshold,
            )
            hard_images[global_q_idx] = hn
            if used_fb:
                fallback_count += 1

    train_ds.hard_images = hard_images
    train_ds.negative_selection_history = {}
    train_ds.use_hard_negatives = True

    sizes = [len(h) for h in hard_images]
    avg_sz = float(np.mean(sizes))
    std_sz = float(np.std(sizes))
    dt = time.time() - t0
    print(
        f"  [QuRe Rerank {dataset_name}] mined hard negatives for {len(hard_images)} queries in {dt:.1f}s | "
        f"avg_pool={avg_sz:.1f} +/- {std_sz:.1f} | min={min(sizes)} max={max(sizes)} | fallbacks={fallback_count}",
        flush=True,
    )
    del q_feats, u_feats
    gc.collect()
    torch.cuda.empty_cache()
    return {"avg_pool": round(avg_sz, 2), "std_pool": round(std_sz, 2), "fallbacks": fallback_count, "sec": round(dt, 1)}


@torch.no_grad()
def evaluate_fashioniq_val(
    model: nn.Module,
    val_by_cat: Dict[str, Dict[str, Any]],
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 8,
    return_top_results: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
    model.eval()
    r5s, r10s, r50s = [], [], []
    metrics: Dict[str, float] = {}
    per_query_dump: Dict[str, Any] = {}

    for dt, d in val_by_cat.items():
        idx_feats, idx_names = encode_qure_images(
            model, d["index_paths"], d["index_names"], preprocess, device, batch_size=eval_bs, num_workers=num_workers
        )
        q_feats = encode_qure_queries(
            model, d["ref_paths"], d["captions"], preprocess, txt_proc, device, batch_size=eval_bs, num_workers=num_workers
        )
        scores = compute_score_matrix_chunked(q_feats, idx_feats, device=device)
        order = np.argsort(-scores, axis=1, kind="stable")
        idx_arr = np.array(idx_names)
        sorted_names = idx_arr[order]
        target_arr = np.array(d["target_names"])[:, None]
        labels = (sorted_names == target_arr)
        assert int(labels.sum(axis=-1).min()) == 1 and int(labels.sum(axis=-1).max()) == 1

        r5 = float(labels[:, :5].any(axis=-1).mean() * 100.0)
        r10 = float(labels[:, :10].any(axis=-1).mean() * 100.0)
        r50 = float(labels[:, :50].any(axis=-1).mean() * 100.0)
        r5s.append(r5)
        r10s.append(r10)
        r50s.append(r50)
        metrics[f"{dt}_R@5"] = round(r5, 4)
        metrics[f"{dt}_R@10"] = round(r10, 4)
        metrics[f"{dt}_R@50"] = round(r50, 4)

        if return_top_results:
            q_records = []
            for qi in range(len(d["target_names"])):
                t_rank = int(np.nonzero(labels[qi])[0][0]) + 1
                top200_idx = order[qi, :200]
                top_res = [
                    {"rank": r_i + 1, "image_id": str(idx_names[ci]), "score": float(scores[qi, ci])}
                    for r_i, ci in enumerate(top200_idx)
                ]
                q_records.append({
                    "query_id": f"{dt}:{qi}:{d['ref_names'][qi]}:{d['target_names'][qi]}",
                    "category": dt,
                    "annotation_index": qi,
                    "reference_id": d["ref_names"][qi],
                    "target_id": d["target_names"][qi],
                    "raw_captions": d["raw_captions"][qi],
                    "model_input_text": d["captions"][qi],
                    "target_rank": t_rank,
                    "top_results": top_res,
                })
            per_query_dump[dt] = q_records

    metrics["avg_R@5"] = round(mean(r5s), 4)
    metrics["avg_R@10"] = round(mean(r10s), 4)
    metrics["avg_R@50"] = round(mean(r50s), 4)
    metrics["avg_recall"] = round((mean(r10s) + mean(r50s)) / 2.0, 4)
    return metrics, (per_query_dump if return_top_results else None)


@torch.no_grad()
def evaluate_cirr_val(
    model: nn.Module,
    val_data: Dict[str, Any],
    preprocess: Callable,
    txt_proc: Callable[[str], str],
    device: torch.device,
    eval_bs: int = 128,
    num_workers: int = 8,
) -> Dict[str, float]:
    model.eval()
    idx_feats, idx_names = encode_qure_images(
        model, val_data["index_paths"], val_data["index_names"], preprocess, device, batch_size=eval_bs, num_workers=num_workers
    )
    q_feats = encode_qure_queries(
        model, val_data["ref_paths"], val_data["captions"], preprocess, txt_proc, device, batch_size=eval_bs, num_workers=num_workers
    )
    scores = compute_score_matrix_chunked(q_feats, idx_feats, device=device)
    order = np.argsort(-scores, axis=1, kind="stable")
    idx_arr = np.array(idx_names)
    sorted_names = idx_arr[order]

    # Exact upstream CIRR val protocol: exclude reference image from ranking
    ref_arr = np.array(val_data["ref_names"])[:, None]
    ref_mask = sorted_names != ref_arr
    sorted_names = sorted_names[ref_mask].reshape(sorted_names.shape[0], sorted_names.shape[1] - 1)

    target_arr = np.array(val_data["target_names"])[:, None]
    labels = (sorted_names == target_arr)

    group_members = np.array(val_data["group_members"])
    group_mask = (sorted_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert int(labels.sum(axis=-1).min()) == 1 and int(labels.sum(axis=-1).max()) == 1
    assert int(group_labels.sum(axis=-1).min()) == 1 and int(group_labels.sum(axis=-1).max()) == 1

    r1 = float(labels[:, :1].any(axis=-1).mean() * 100.0)
    r5 = float(labels[:, :5].any(axis=-1).mean() * 100.0)
    r10 = float(labels[:, :10].any(axis=-1).mean() * 100.0)
    r50 = float(labels[:, :50].any(axis=-1).mean() * 100.0)
    rs1 = float(group_labels[:, :1].any(axis=-1).mean() * 100.0)
    rs2 = float(group_labels[:, :2].any(axis=-1).mean() * 100.0)
    rs3 = float(group_labels[:, :3].any(axis=-1).mean() * 100.0)
    return {
        "R@1": round(r1, 4),
        "R@5": round(r5, 4),
        "R@10": round(r10, 4),
        "R@50": round(r50, 4),
        "R_s@1": round(rs1, 4),
        "R_s@2": round(rs2, 4),
        "R_s@3": round(rs3, 4),
        "mean_R5_Rs1": round((r5 + rs1) / 2.0, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train upstream-faithful QuRe checkpoint on FashionIQ or CIRR")
    parser.add_argument("--dataset", type=str, required=True, choices=["fashioniq", "cirr"])
    parser.add_argument("--epochs", type=int, default=None, help="Default: 30 for fashioniq, 50 for cirr")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--init-lr", type=float, default=None, help="Default: 5e-6 for fashioniq, 1e-5 for cirr")
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--negative-def-num", type=int, default=6)
    parser.add_argument("--consecutive-threshold", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1000, help="Exact upstream random_seed=1000")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--smoke-steps", type=int, default=0, help="If >0, run quick N-step smoke check and exit")
    args = parser.parse_args()

    is_fiq = args.dataset == "fashioniq"
    max_epochs = args.epochs if args.epochs is not None else (30 if is_fiq else 50)
    init_lr = args.init_lr if args.init_lr is not None else (5e-6 if is_fiq else 1e-5)
    min_lr = init_lr / 100.0
    out_dir = args.out_dir or (REPO_ROOT / "workbench" / "artifacts" / "qure_checkpoints" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = targetpad_transform(1.25, 224)

    print(f"[QuRe Upstream Train] dataset={args.dataset} epochs={max_epochs} bs={args.batch_size} lr={init_lr} wd={args.weight_decay} seed={args.seed}", flush=True)
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_qure", model_type="pretrain_vitL", is_eval=False, device=device
    )
    txt_proc = txt_processors["eval"]

    # Scorer parity self-check vs model.score()
    model.eval()
    with torch.no_grad():
        dq = F.normalize(torch.randn(5, 256), dim=-1)
        dt = F.normalize(torch.randn(11, 32, 256), dim=-1)
        ref_s = model.score(dq, dt).numpy()
        our_s = compute_score_matrix_chunked(dq, dt, device=device, q_chunk=2, c_chunk=4)
        diff = float(np.abs(ref_s - our_s).max())
        assert diff < 1e-5, f"QuRe scorer parity mismatch: {diff}"
        print(f"[QuRe Scorer Parity Self-Check] max_abs_diff vs model.score() = {diff:.2e} (OK)", flush=True)

    if is_fiq:
        train_records, name_to_path, domain_pools, domain_triplets, val_obj = load_fashioniq_data(REPO_ROOT / "data" / "FashionIQ")
    else:
        train_records, name_to_path, domain_pools, domain_triplets, val_obj = load_cirr_data(REPO_ROOT / "data" / "CIRR")

    train_ds = QuReTripletTrainDataset(train_records, name_to_path, domain_pools, preprocess)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=init_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(args.seed)

    # Smoke mode: verify real data loading + forward + backward + step + rerank helper
    if args.smoke_steps > 0:
        train_ds.prepare_epoch_negatives(rng)
        dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=min(args.num_workers, 4), pin_memory=True, drop_last=True)
        model.train()
        t0 = time.time()
        for si, (ref_imgs, tar_imgs, neg_imgs, caps) in enumerate(dl):
            if si >= args.smoke_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            ref_imgs = ref_imgs.to(device, non_blocking=True)
            tar_imgs = tar_imgs.to(device, non_blocking=True)
            neg_imgs = neg_imgs.to(device, non_blocking=True)
            proc_caps = [txt_proc(c) for c in caps]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                scores = model(ref_imgs, tar_imgs, proc_caps, neg_imgs, True)
                log_sm = F.log_softmax(scores, dim=1)
                tgt = torch.zeros_like(log_sm)
                tgt[:, 0] = 1.0
                loss = F.kl_div(log_sm, tgt, reduction="batchmean")
            assert torch.isfinite(loss), f"Non-finite loss: {loss}"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        print(f"[SMOKE PASS] {args.dataset} {args.smoke_steps} steps in {time.time()-t0:.2f}s | loss={loss.item():.4f}", flush=True)
        return

    if history_path.exists():
        history_path.unlink()

    write_json(out_dir / "config.json", {
        "dataset": args.dataset,
        "model": "blip2_qure",
        "backbone": "pretrain_vitL",
        "max_epochs": max_epochs,
        "batch_size": args.batch_size,
        "init_lr": init_lr,
        "min_lr": min_lr,
        "weight_decay": args.weight_decay,
        "negative_definition_epoch_num": args.negative_def_num,
        "consecutive_threshold": args.consecutive_threshold,
        "seed": args.seed,
        "git_sha": get_git_sha(),
        "started_at": utc_now_iso(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_train_triplets": len(train_records),
    })

    # Stage 0 zero-shot eval on FashionIQ / CIRR val
    t0 = time.time()
    if is_fiq:
        s0_metrics, _ = evaluate_fashioniq_val(model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers)
        sel_key = "avg_recall"
    else:
        s0_metrics = evaluate_cirr_val(model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers)
        sel_key = "mean_R5_Rs1"
    best_score = s0_metrics[sel_key]
    best_epoch = 0
    print(f"[Stage 0 Zero-Shot {args.dataset}] {sel_key}={best_score:.2f} | {s0_metrics} ({time.time()-t0:.1f}s)", flush=True)
    append_jsonl(history_path, {"epoch": 0, "stage": "zeroshot", "val_metrics": s0_metrics, "best_score": best_score})

    best_ckpt_path = out_dir / "best_model.pth"
    last_ckpt_path = out_dir / "last_model.pth"
    torch.save(model.state_dict(), best_ckpt_path)

    rerank_interval = max(1, max_epochs // args.negative_def_num)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    for epoch in range(max_epochs):
        ep_t0 = time.time()
        rerank_info = None
        if epoch > 0 and (epoch % rerank_interval) == 0:
            rerank_info = rerank_hard_negatives(
                args.dataset,
                model,
                train_ds,
                domain_triplets,
                preprocess,
                txt_proc,
                device,
                consecutive_threshold=args.consecutive_threshold,
                eval_bs=args.eval_bs,
                num_workers=args.num_workers,
            )

        train_ds.prepare_epoch_negatives(rng)
        dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            generator=gen,
        )

        cur_lr = cosine_lr_schedule(optimizer, epoch, max_epochs, init_lr, min_lr)
        model.train()
        running_loss = 0.0
        n_batches = 0

        for ref_imgs, tar_imgs, neg_imgs, caps in dl:
            optimizer.zero_grad(set_to_none=True)
            ref_imgs = ref_imgs.to(device, non_blocking=True)
            tar_imgs = tar_imgs.to(device, non_blocking=True)
            neg_imgs = neg_imgs.to(device, non_blocking=True)
            proc_caps = [txt_proc(c) for c in caps]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                scores = model(ref_imgs, tar_imgs, proc_caps, neg_imgs, True)
                log_sm = F.log_softmax(scores, dim=1)
                tgt = torch.zeros_like(log_sm)
                tgt[:, 0] = 1.0
                loss = F.kl_div(log_sm, tgt, reduction="batchmean")

            l_val = float(loss.detach().cpu().item())
            if not np.isfinite(l_val):
                raise RuntimeError(f"Non-finite QuRe loss at epoch={epoch+1} batch={n_batches}: {l_val}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += l_val
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        if is_fiq:
            val_metrics, _ = evaluate_fashioniq_val(model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers)
        else:
            val_metrics = evaluate_cirr_val(model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers)

        cur_score = val_metrics[sel_key]
        improved = cur_score > best_score
        if improved:
            best_score = cur_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)

        dt = time.time() - ep_t0
        temp_val = float(model.temp.detach().cpu().item())
        print(
            f"[QuRe {args.dataset} Ep {epoch+1:02d}/{max_epochs}] loss={avg_loss:.4f} lr={cur_lr:.2e} temp={temp_val:.4f} | "
            f"val {sel_key}={cur_score:.2f} (best={best_score:.2f}@ep{best_epoch}) | {dt:.1f}s {'*BEST*' if improved else ''}",
            flush=True,
        )
        append_jsonl(history_path, {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "lr": cur_lr,
            "temp": round(temp_val, 6),
            "rerank": rerank_info,
            "val_metrics": val_metrics,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "epoch_sec": round(dt, 1),
        })

    torch.save(model.state_dict(), last_ckpt_path)

    # Reload best checkpoint cleanly (strict=True) and run final val verification (+ FIQ per-query dump)
    best_sd = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_sd, strict=True)
    if is_fiq:
        final_metrics, fiq_dump = evaluate_fashioniq_val(
            model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers, return_top_results=True
        )
        write_json(out_dir / "fashioniq_val_per_query_top200.json", fiq_dump)
    else:
        final_metrics = evaluate_cirr_val(model, val_obj, preprocess, txt_proc, device, eval_bs=args.eval_bs, num_workers=args.num_workers)
        # Also copy/symlink into standard cirr_checkpoints/qure/
        cirr_dst = REPO_ROOT / "workbench" / "artifacts" / "cirr_checkpoints" / "qure"
        cirr_dst.mkdir(parents=True, exist_ok=True)
        dst_link = cirr_dst / "qure_cirr_vitl.pth"
        if dst_link.exists() or dst_link.is_symlink():
            dst_link.unlink()
        dst_link.symlink_to(best_ckpt_path.resolve())

    summary = {
        "dataset": args.dataset,
        "finished_at": utc_now_iso(),
        "best_epoch": best_epoch,
        "selection_metric": sel_key,
        "best_val_score": best_score,
        "stage0_zeroshot_val": s0_metrics,
        "final_best_val": final_metrics,
        "best_ckpt": str(best_ckpt_path),
    }
    write_json(out_dir / "summary.json", summary)
    print(f"\n=== QuRe {args.dataset.upper()} FINISHED | best_ep={best_epoch} | {final_metrics} ===", flush=True)


if __name__ == "__main__":
    main()
