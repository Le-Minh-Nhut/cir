#!/usr/bin/env python3
"""Controlled GPU Training & Dynamics Suites: E02, E16, and E17.

Implements exact controlled causal training interventions and dynamics tracking:
  - E02 (F02, F14): Same-Scene Negative Intervention (0/10/25/50/100% same-scene siblings +
        CLIP-difficulty-matched other-scene hard negative control + ref-image negative arm)
        on EDIR and PinPoint with identical init and fixed step/epoch budget!
  - E16 (F18): Cross-Domain Gradient Conflict — exact pairwise inter-domain gradient cosine
        matrix at identical checkpoints + controlled train-domain x test-domain transfer matrix
        (single-domain, mixed-original, mixed-balanced, AND causal PCGrad gradient-projection arm
        at identical step budget!) + PinPoint demographic/domain fairness audit!
  - E17 (F02, F14, F18): Full 16-model history.jsonl trajectory forensics PLUS real multi-stage
        checkpoint probe trajectories (init / ep1 / ep2 / best / final) tracking RefCopy@1,
        Scene@1, within-scene CondR@1, sibling margins, and E04 linear probe accuracy!
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.benchmarks import evaluate_edir, evaluate_pinpoint, load_split_bundle  # noqa: E402
from workbench.probes.cohort import BENCH_RUNS, MODEL_BY_KEY, MODEL_SPECS, SPLIT_TAG  # noqa: E402

EDIR_DIR = REPO_ROOT / "data/EDIR"
EDIR_SPLIT = EDIR_DIR / "splits/70_10_20_seed42"
PP_DIR = REPO_ROOT / "data/PinPoint"
PP_SPLIT = PP_DIR / "splits/70_10_20_seed42"

TOP7_PP_DOMAINS = [
    "Women's Fashion",
    "Home Decor",
    "Art",
    "Beauty",
    "Entertainment",
    "Vehicles",
    "Animals",
]


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _det_hash_int(s: str) -> int:
    """Process-independent deterministic 32-bit hash (immune to PYTHONHASHSEED)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _open_rgb(path: str) -> PIL.Image.Image:
    with PIL.Image.open(path) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------------------
# E17 Part A: Full 16-model history.jsonl Dynamics Extraction (CPU-only!)
# ---------------------------------------------------------------------------

def run_e17_history_all16(out_root: Path) -> None:
    out_dir = out_root / "e17_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"suite": "E17_history_all16", "datasets": {}}

    for ds in ("edir", "pinpoint"):
        ds_out: Dict[str, Any] = {}
        for spec in MODEL_SPECS:
            run_dir = BENCH_RUNS / ds / SPLIT_TAG / spec.run_id
            hist_p = run_dir / "history.jsonl"
            res_p = run_dir / "test_results.json"
            if not hist_p.is_file():
                continue
            rows = _load_jsonl(hist_p)
            ep_curve = []
            for r in rows:
                vm = r.get("val_metrics", {}).get("overall", {})
                ep_curve.append({
                    "stage": r.get("stage", "train"),
                    "epoch": r.get("epoch"),
                    "val_selection_score": r.get("val_selection_score"),
                    "R@1": vm.get("Recall@1"),
                    "R@5": vm.get("Recall@5"),
                    "R@10": vm.get("Recall@10"),
                    "mAP@10": vm.get("mAP@10"),
                    "P@1": vm.get("precision@1"),
                    "NegRecall@10": vm.get("NegRecall@10"),
                    "train_loss": r.get("train_loss"),
                })
            # Extract init (ep 0), ep 1, best, and final values cleanly
            ep0 = next((x for x in ep_curve if x["epoch"] == 0), ep_curve[0] if ep_curve else None)
            ep1 = next((x for x in ep_curve if x["epoch"] == 1), None)
            ep_last = ep_curve[-1] if ep_curve else None
            best_row = max(ep_curve, key=lambda x: x["val_selection_score"] if x["val_selection_score"] is not None else -1.0) if ep_curve else None
            # Peak R@1 epoch on EDIR (to verify Epoch-1 Scene Lock & selection mismatch!)
            peak_r1_row = None
            if ds == "edir":
                valid_r1 = [x for x in ep_curve if x["R@1"] is not None]
                if valid_r1:
                    peak_r1_row = max(valid_r1, key=lambda x: x["R@1"])
            test_primary = None
            if res_p.is_file():
                td = json.loads(res_p.read_text())
                ov = td.get(spec.result_field, {}).get("overall", {})
                test_primary = ov.get("Recall@1" if ds == "edir" else "mAP@10")
            ds_out[spec.key] = {
                "family": spec.family,
                "n_epochs_logged": len(ep_curve),
                "init_ep0": ep0,
                "early_ep1": ep1,
                "best_val_selected": best_row,
                "peak_val_R1": peak_r1_row,
                "final_ep": ep_last,
                "test_primary": test_primary,
                "epoch_1_scene_lock_flag": bool(ds == "edir" and ep1 and ep1.get("R@10") is not None and ep1["R@10"] >= 0.90),
                "curve": ep_curve,
            }
        summary["datasets"][ds] = ds_out

    _write_json(out_dir / "e17_history_all16.json", summary)
    print(f"[E17-HIST] Wrote {out_dir / 'e17_history_all16.json'} ({len(summary['datasets']['edir'])} EDIR, {len(summary['datasets']['pinpoint'])} PinPoint models)", flush=True)


# ---------------------------------------------------------------------------
# Unified Trainable Model Wrapper for Controlled Arms (clip4cir, conesep, sprc)
# ---------------------------------------------------------------------------

class _TrainableCIR:
    """Clean trainable wrapper over clip4cir, conesep, or sprc initialized strictly from base pretrained weights."""

    def __init__(self, model_key: str, device: torch.device):
        self.model_key = model_key
        self.device = device
        self.txt_proc: Callable[[str], str] = lambda s: str(s)
        self.base_state: Dict[str, torch.Tensor] = {}
        self._build_base()

    def _build_base(self) -> None:
        k = self.model_key
        if k == "clip4cir":
            import clip
            from combiner import Combiner
            from data_utils import targetpad_transform

            self.clip_mod = clip
            self.clip_model, _ = clip.load("RN50x4", device=self.device, jit=False)
            self.clip_model.float()
            self.combiner = Combiner(self.clip_model.visual.output_dim, 2560, 5120).to(self.device)
            self.preprocess = targetpad_transform(1.25, self.clip_model.visual.input_resolution)
            # Freeze CLIP RN50x4 visual encoder for fast, clean, identical-init controlled arms?
            # Even better: fine-tune BOTH last CLIP blocks + Combiner (or full CLIP + Combiner at lr=2e-6 / 2e-5)!
            self.base_state = {
                "clip": {n: p.detach().cpu().clone() for n, p in self.clip_model.state_dict().items()},
                "comb": {n: p.detach().cpu().clone() for n, p in self.combiner.state_dict().items()},
            }
            self.lr = 2e-5
            self.clip_lr = 2e-6
            self.is_multivec = False

        elif k == "conesep":
            from workbench.benchmarks.trainers.train_ilearn import _patch_bert_root_path, _patch_conesep_nan_guard
            tp_dir = str(REPO_ROOT / "workbench/third_party/ConeSep")
            if tp_dir not in sys.path:
                sys.path.insert(0, tp_dir)
            _patch_bert_root_path()
            from data_utils import targetpad_transform
            from lavis.models import load_model_and_preprocess

            self.model, _, txt_procs = load_model_and_preprocess(
                name="ConeSep", model_type="pretrain", is_eval=False, device=self.device
            )
            self.model.prompt_load()
            _patch_conesep_nan_guard()
            # Keep visual_encoder frozen (exact match to upstream & train_ilearn.py!)
            for p in self.model.visual_encoder.parameters():
                p.requires_grad = False
            self.txt_proc = txt_procs["eval"]
            self.preprocess = targetpad_transform(1.25, 224)
            self.base_state = {
                n: p.detach().cpu().clone()
                for n, p in self.model.state_dict().items()
                if not n.startswith("visual_encoder.")
            }
            self.lr = 1e-5
            self.is_multivec = True

        elif k == "sprc":
            from data_utils import targetpad_transform
            from lavis.models import load_model_and_preprocess

            self.model, _, txt_procs = load_model_and_preprocess(
                name="blip2_cir_align_prompt", model_type="pretrain", is_eval=False, device=self.device
            )
            for p in self.model.visual_encoder.parameters():
                p.requires_grad = False
            self.txt_proc = txt_procs["eval"]
            self.preprocess = targetpad_transform(1.25, 224)
            self.base_state = {
                n: p.detach().cpu().clone()
                for n, p in self.model.state_dict().items()
                if not n.startswith("visual_encoder.")
            }
            self.lr = 1e-5
            self.is_multivec = True
        else:
            raise ValueError(k)

    def reset_to_init(self) -> None:
        if self.model_key == "clip4cir":
            self.clip_model.load_state_dict(self.base_state["clip"], strict=True)
            self.combiner.load_state_dict(self.base_state["comb"], strict=True)
        else:
            self.model.load_state_dict(self.base_state, strict=False)

    def snapshot_state(self) -> Dict[str, Any]:
        if self.model_key == "clip4cir":
            return {
                "clip": {n: p.detach().cpu().clone() for n, p in self.clip_model.state_dict().items()},
                "comb": {n: p.detach().cpu().clone() for n, p in self.combiner.state_dict().items()},
            }
        return {
            n: p.detach().cpu().clone()
            for n, p in self.model.state_dict().items()
            if not n.startswith("visual_encoder.")
        }

    def load_snapshot(self, snap: Dict[str, Any]) -> None:
        if self.model_key == "clip4cir":
            self.clip_model.load_state_dict(snap["clip"], strict=True)
            self.combiner.load_state_dict(snap["comb"], strict=True)
        else:
            self.model.load_state_dict(snap, strict=False)

    def load_trained_best(self, dataset: str) -> None:
        """Load the official trained best checkpoint on EDIR or PinPoint (for E16 gradient cosine at best ckpt!)."""
        run_dir = BENCH_RUNS / dataset / SPLIT_TAG / MODEL_BY_KEY[self.model_key].run_id
        if self.model_key == "clip4cir":
            s1 = torch.load(run_dir / "best_stage1_clip.pt", map_location=self.device, weights_only=False)
            s2 = torch.load(run_dir / "best_stage2_combiner.pt", map_location=self.device, weights_only=False)
            self.clip_model.load_state_dict(s1["CLIP"], strict=True)
            self.combiner.load_state_dict(s2["Combiner"], strict=True)
        else:
            sd = torch.load(run_dir / f"best_{self.model_key}.pt", map_location=self.device, weights_only=False)
            self.model.load_state_dict(sd["state_dict_non_vit"], strict=False)

    def make_optimizer(self) -> torch.optim.Optimizer:
        if self.model_key == "clip4cir":
            return torch.optim.AdamW(
                [
                    {"params": [p for p in self.clip_model.parameters() if p.requires_grad], "lr": self.clip_lr},
                    {"params": [p for p in self.combiner.parameters() if p.requires_grad], "lr": self.lr},
                ],
                weight_decay=0.05,
            )
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay=0.05)

    def trainable_params(self) -> List[nn.Parameter]:
        if self.model_key == "clip4cir":
            return [p for p in list(self.clip_model.parameters()) + list(self.combiner.parameters()) if p.requires_grad]
        return [p for p in self.model.parameters() if p.requires_grad]

    def set_train(self) -> None:
        if self.model_key == "clip4cir":
            self.clip_model.train()
            self.combiner.train()
        else:
            self.model.train()
            self.model.visual_encoder.eval()

    def set_eval(self) -> None:
        if self.model_key == "clip4cir":
            self.clip_model.eval()
            self.combiner.eval()
        else:
            self.model.eval()

    def forward_query_and_target(
        self,
        ref_imgs: torch.Tensor,
        texts: Sequence[str],
        tgt_imgs: torch.Tensor,
        extra_neg_imgs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Differentiable forward returning (q_feats, tgt_feats, extra_neg_feats)."""
        k = self.model_key
        if k == "clip4cir":
            toks = self.clip_mod.tokenize(list(texts), context_length=77, truncate=True).to(self.device)
            rf = self.clip_model.encode_image(ref_imgs).float()
            tf = self.clip_model.encode_text(toks).float()
            qf = F.normalize(self.combiner.combine_features(rf, tf).float(), dim=-1)
            gf = F.normalize(self.clip_model.encode_image(tgt_imgs).float(), dim=-1)
            nf = F.normalize(self.clip_model.encode_image(extra_neg_imgs).float(), dim=-1) if extra_neg_imgs is not None else None
            return qf, gf, nf

        m = self.model
        caps = [self.txt_proc(c) for c in texts]
        with torch.no_grad(), m.maybe_autocast():
            ref_vit = m.visual_encoder(ref_imgs)
            tgt_vit = m.visual_encoder(tgt_imgs)
            neg_vit = m.visual_encoder(extra_neg_imgs) if extra_neg_imgs is not None else None

        ref_e = m.ln_vision(ref_vit).float()
        tgt_e = m.ln_vision(tgt_vit).float()
        ia_r = torch.ones(ref_e.size()[:-1], dtype=torch.long, device=self.device)
        ia_t = torch.ones(tgt_e.size()[:-1], dtype=torch.long, device=self.device)
        qt = m.query_tokens.expand(ref_e.shape[0], -1, -1)
        qa = torch.ones(qt.size()[:-1], dtype=torch.long, device=self.device)
        tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
        am = torch.cat([qa, tt.attention_mask], dim=1)

        fo = m.Qformer.bert(tt.input_ids, query_embeds=qt, attention_mask=am, encoder_hidden_states=ref_e, encoder_attention_mask=ia_r, return_dict=True)
        if k == "sprc":
            to = m.Qformer.bert(tt.input_ids, query_embeds=fo.last_hidden_state[:, :32, :], attention_mask=am, return_dict=True)
            qf = F.normalize(m.text_proj(to.last_hidden_state[:, 32, :]), dim=-1)
        else:
            qf = F.normalize(m.text_proj(fo.last_hidden_state[:, 32, :]), dim=-1)

        t_out = m.Qformer.bert(query_embeds=qt, encoder_hidden_states=tgt_e, encoder_attention_mask=ia_t, return_dict=True)
        gf = F.normalize(m.vision_proj(t_out.last_hidden_state), dim=-1)  # [B, 32, 256]

        nf = None
        if neg_vit is not None:
            neg_e = m.ln_vision(neg_vit).float()
            ia_n = torch.ones(neg_e.size()[:-1], dtype=torch.long, device=self.device)
            qt_n = m.query_tokens.expand(neg_e.shape[0], -1, -1)
            n_out = m.Qformer.bert(query_embeds=qt_n, encoder_hidden_states=neg_e, encoder_attention_mask=ia_n, return_dict=True)
            nf = F.normalize(m.vision_proj(n_out.last_hidden_state), dim=-1)  # [M, 32, 256]
        return qf, gf, nf

    def compute_infonce_loss(
        self,
        ref_imgs: torch.Tensor,
        texts: Sequence[str],
        tgt_imgs: torch.Tensor,
        extra_neg_imgs: Optional[torch.Tensor] = None,
        extra_neg_mask: Optional[torch.Tensor] = None,
        false_neg_mask: Optional[torch.Tensor] = None,
        temp: float = 0.05,
    ) -> torch.Tensor:
        """Exact InfoNCE loss supporting per-row explicit hard negatives & off-diagonal false-negative masking!"""
        qf, gf, nf = self.forward_query_and_target(ref_imgs, texts, tgt_imgs, extra_neg_imgs=extra_neg_imgs)
        bs = qf.shape[0]
        if self.is_multivec:
            logits_ib = torch.einsum("bd,nkd->bnk", qf, gf).max(dim=-1).values / temp  # [B, B]
            if false_neg_mask is not None:
                logits_ib = logits_ib.masked_fill(false_neg_mask.bool(), -1e4)
            if nf is not None:
                # Extra negative for each row i: [B, M]
                logits_en = torch.einsum("bd,mkd->bmk", qf, nf).max(dim=-1).values / temp
                if extra_neg_mask is not None:
                    # Mask inactive rows so only selected fraction gets the explicit extra negative!
                    logits_en = logits_en.masked_fill(~extra_neg_mask.bool(), -1e4)
                logits = torch.cat([logits_ib, logits_en], dim=1)
            else:
                logits = logits_ib
        else:
            logits_ib = (qf @ gf.T) / temp
            if false_neg_mask is not None:
                logits_ib = logits_ib.masked_fill(false_neg_mask.bool(), -1e4)
            if nf is not None:
                logits_en = (qf @ nf.T) / temp
                if extra_neg_mask is not None:
                    logits_en = logits_en.masked_fill(~extra_neg_mask.bool(), -1e4)
                logits = torch.cat([logits_ib, logits_en], dim=1)
            else:
                logits = logits_ib
        labels = torch.arange(bs, device=self.device, dtype=torch.long)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_images_eval(self, img_paths: Sequence[str], bs: int = 96, num_workers: int = 6) -> torch.Tensor:
        self.set_eval()
        class _DS(Dataset):
            def __init__(self, ps, pp):
                self.ps, self.pp = list(ps), pp
            def __len__(self): return len(self.ps)
            def __getitem__(self, i): return self.pp(_open_rgb(self.ps[i]))
        dl = DataLoader(_DS(img_paths, self.preprocess), batch_size=bs, shuffle=False, num_workers=min(num_workers, 6), pin_memory=True)
        out = []
        for imgs in dl:
            imgs = imgs.to(self.device, non_blocking=True)
            if self.model_key == "clip4cir":
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    f = F.normalize(self.clip_model.encode_image(imgs).float(), dim=-1)
            else:
                m = self.model
                with m.maybe_autocast():
                    ve = m.ln_vision(m.visual_encoder(imgs))
                ve = ve.float()
                ia = torch.ones(ve.size()[:-1], dtype=torch.long, device=self.device)
                qt = m.query_tokens.expand(ve.shape[0], -1, -1)
                qo = m.Qformer.bert(query_embeds=qt, encoder_hidden_states=ve, encoder_attention_mask=ia, return_dict=True).last_hidden_state
                f = F.normalize(m.vision_proj(qo), dim=-1).float()
            out.append(f.cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_queries_eval(self, ref_paths: Sequence[str], texts: Sequence[str], bs: int = 64, num_workers: int = 6) -> torch.Tensor:
        self.set_eval()
        class _DS(Dataset):
            def __init__(self, ps, ts, pp):
                self.ps, self.ts, self.pp = list(ps), list(ts), pp
            def __len__(self): return len(self.ps)
            def __getitem__(self, i): return self.pp(_open_rgb(self.ps[i])), self.ts[i]
        dl = DataLoader(_DS(ref_paths, texts, self.preprocess), batch_size=bs, shuffle=False, num_workers=min(num_workers, 6), pin_memory=True)
        out = []
        for imgs, b_txts in dl:
            imgs = imgs.to(self.device, non_blocking=True)
            b_txts = list(b_txts)
            if self.model_key == "clip4cir":
                toks = self.clip_mod.tokenize(b_txts, context_length=77, truncate=True).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    rf = self.clip_model.encode_image(imgs).float()
                    tf = self.clip_model.encode_text(toks).float()
                f = F.normalize(self.combiner.combine_features(rf, tf).float(), dim=-1)
            else:
                m = self.model
                caps = [self.txt_proc(c) for c in b_txts]
                with m.maybe_autocast():
                    ve = m.ln_vision(m.visual_encoder(imgs))
                ve = ve.float()
                ia = torch.ones(ve.size()[:-1], dtype=torch.long, device=self.device)
                qt = m.query_tokens.expand(ve.shape[0], -1, -1)
                qa = torch.ones(qt.size()[:-1], dtype=torch.long, device=self.device)
                tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
                am = torch.cat([qa, tt.attention_mask], dim=1)
                fo = m.Qformer.bert(tt.input_ids, query_embeds=qt, attention_mask=am, encoder_hidden_states=ve, encoder_attention_mask=ia, return_dict=True)
                if self.model_key == "sprc":
                    to = m.Qformer.bert(tt.input_ids, query_embeds=fo.last_hidden_state[:, :32, :], attention_mask=am, return_dict=True)
                    f = F.normalize(m.text_proj(to.last_hidden_state[:, 32, :]), dim=-1).float()
                else:
                    f = F.normalize(m.text_proj(fo.last_hidden_state[:, 32, :]), dim=-1).float()
            out.append(f.cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def score_eval(self, q_feats: torch.Tensor, g_feats: torch.Tensor) -> np.ndarray:
        n_q, n_c = q_feats.shape[0], g_feats.shape[0]
        sims = np.empty((n_q, n_c), dtype=np.float32)
        if self.is_multivec:
            for qs in range(0, n_q, 128):
                qe = min(qs + 128, n_q)
                qf = q_feats[qs:qe].to(self.device)
                for cs in range(0, n_c, 4096):
                    ce = min(cs + 4096, n_c)
                    tf = g_feats[cs:ce].to(self.device)
                    sims[qs:qe, cs:ce] = torch.einsum("bd,nkd->bnk", qf, tf).max(dim=-1).values.cpu().numpy()
        else:
            qn = F.normalize(q_feats.float(), dim=-1)
            gn = F.normalize(g_feats.float(), dim=-1)
            for qs in range(0, n_q, 256):
                qe = min(qs + 256, n_q)
                sims[qs:qe] = (qn[qs:qe].to(self.device) @ gn.to(self.device).T).cpu().numpy()
        return sims


def _eval_edir_full_diagnostics(
    wrapper: _TrainableCIR,
    test_q: List[Dict[str, Any]],
    c_ids: List[str],
    c_paths: List[str],
    num_workers: int,
) -> Dict[str, Any]:
    """Evaluate EDIR test on FULL 28,645 corpus + RefCopy@1 + Scene@1 + within-scene CondR@1 + sibling margins!"""
    g_feats = wrapper.encode_images_eval(c_paths, bs=96, num_workers=num_workers)
    q_feats = wrapper.encode_queries_eval([str(EDIR_DIR / r["ref_image_rel"]) for r in test_q], [r["text"] for r in test_q], bs=64, num_workers=num_workers)
    # Also encode unique test ref images so we measure exact RefCopy@1 if refs were in the candidate pool!
    ref_ids = sorted({r["ref_id"] for r in test_q})
    ref_path_map = {r["ref_id"]: str(EDIR_DIR / r["ref_image_rel"]) for r in test_q}
    ref_feats = wrapper.encode_images_eval([ref_path_map[i] for i in ref_ids], bs=96, num_workers=num_workers)
    ref_idx_map = {rid: i for i, rid in enumerate(ref_ids)}

    sims = wrapper.score_eval(q_feats, g_feats)
    c_idx_map = {cid: i for i, cid in enumerate(c_ids)}
    top50 = np.argsort(-sims, axis=1, kind="stable")[:, :50]

    r1_list, r5_list, r10_list, scene1_list, cond_r1_list, ref_beats_pos_list = [], [], [], [], [], []
    sib_margin_list, ref_margin_list, within_rank_list = [], [], []
    for qi, r in enumerate(test_q):
        pos_id = r["positive_ids"][0]
        stem = r["metadata"]["stem_base"]
        top_ids = [c_ids[ci] for ci in top50[qi]]
        r1 = float(top_ids[0] == pos_id)
        r1_list.append(r1)
        r5_list.append(float(pos_id in top_ids[:5]))
        r10_list.append(float(pos_id in top_ids[:10]))
        sc1 = float(top_ids[0].startswith(stem))
        scene1_list.append(sc1)
        if sc1 > 0.5:
            cond_r1_list.append(r1)
        # Exact within-scene rank & sibling margin (pos vs best non-pos sibling)
        pos_s = float(sims[qi, c_idx_map[pos_id]])
        sib_scores = [float(sims[qi, c_idx_map[nid]]) for nid in r["negative_ids"] if nid in c_idx_map]
        best_sib = max(sib_scores) if sib_scores else pos_s
        sib_margin_list.append(pos_s - best_sib)
        within_rank_list.append(1 + sum(1 for s in sib_scores if s > pos_s))
        # Exact ref score
        r_vec = ref_feats[ref_idx_map[r["ref_id"]]]
        if wrapper.is_multivec:
            s_ref = float(torch.einsum("d,kd->k", q_feats[qi].float(), r_vec.float()).max().item())
        else:
            s_ref = float(torch.dot(F.normalize(q_feats[qi].float(), dim=-1), F.normalize(r_vec.float(), dim=-1)).item())
        ref_margin_list.append(pos_s - s_ref)
        ref_beats_pos_list.append(float(s_ref > pos_s))

    # Also run lightweight E04 linear probes (F01 query negation, F08 query geom/color, AND F08 gallery pos-ref diff!)
    probe_scores: Dict[str, float] = {}
    try:
        import re as _re
        neg_rx = _re.compile(r"\b(remove|without|no |not |don't|except|erase|delete|eliminate)\b", _re.I)
        # Fit L2 ridge-classifier strictly on deterministic scene-disjoint 50/50 split of test_q
        X_qry = F.normalize(q_feats.float(), dim=-1).numpy().astype(np.float64)
        # Gallery (pos - ref) representation for F08 gallery geometric probe across training epochs!
        g_pos_vecs = []
        r_ref_vecs = []
        for r in test_q:
            gv = g_feats[c_idx_map[r["positive_ids"][0]]].float()
            rv = ref_feats[ref_idx_map[r["ref_id"]]].float()
            if wrapper.is_multivec:
                gv = F.normalize(gv.mean(dim=0), dim=-1)
                rv = F.normalize(rv.mean(dim=0), dim=-1)
            else:
                gv = F.normalize(gv, dim=-1)
                rv = F.normalize(rv, dim=-1)
            g_pos_vecs.append(gv.numpy())
            r_ref_vecs.append(rv.numpy())
        X_gal_diff = (np.stack(g_pos_vecs, axis=0) - np.stack(r_ref_vecs, axis=0)).astype(np.float64)
        split_m = np.array([_det_hash_int(r["metadata"]["stem_base"]) % 2 == 0 for r in test_q])
        y_neg = np.array([1 if (r["category"] == "remove" or neg_rx.search(r["text"])) else 0 for r in test_q])
        y_geom = np.array([1 if r["category"] in ("shape", "viewpoint") else (0 if r["category"] in ("color", "material", "texture") else -1) for r in test_q])
        for pname, X_use, y_arr in [
            ("probe_F01_negation_acc", X_qry, y_neg),
            ("probe_F08_geom_vs_color_qry_acc", X_qry, y_geom),
            ("probe_F08_geom_vs_color_gal_diff_acc", X_gal_diff, y_geom),
        ]:
            valid = y_arr != -1
            tr_m = valid & split_m
            te_m = valid & (~split_m)
            if tr_m.sum() >= 20 and te_m.sum() >= 20:
                Xtr, ytr = X_use[tr_m], y_arr[tr_m].astype(np.float64)
                Xte, yte = X_use[te_m], y_arr[te_m]
                mu = Xtr.mean(axis=0, keepdims=True)
                Xc, Xtc = Xtr - mu, Xte - mu
                w = np.linalg.solve(Xc.T @ Xc + 10.0 * np.eye(Xc.shape[1]), Xc.T @ (ytr - ytr.mean()))
                pred = ((Xtc @ w + ytr.mean()) >= 0.5).astype(np.int64)
                probe_scores[pname] = float(np.mean(pred == yte))
    except Exception:
        pass

    del g_feats, q_feats, ref_feats, sims
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "n_queries": len(test_q),
        "n_corpus": len(c_ids),
        "R@1": float(np.mean(r1_list)),
        "R@5": float(np.mean(r5_list)),
        "R@10": float(np.mean(r10_list)),
        "Scene@1": float(np.mean(scene1_list)),
        "CondR@1_within_scene": float(np.mean(cond_r1_list)) if cond_r1_list else 0.0,
        "mean_within_scene_rank": float(np.mean(within_rank_list)),
        "mean_sibling_margin": float(np.mean(sib_margin_list)),
        "mean_ref_margin_pos_minus_ref": float(np.mean(ref_margin_list)),
        "RefBeatsPos_rate": float(np.mean(ref_beats_pos_list)),
        **probe_scores,
    }


def _eval_pinpoint_diagnostics(
    wrapper: _TrainableCIR,
    test_q: List[Dict[str, Any]],
    c_ids: List[str],
    c_paths: List[str],
    num_workers: int,
) -> Dict[str, Any]:
    """Evaluate PinPoint on split/test corpus (8,609 images — fast, exact, includes ALL test positives & negatives & refs!)."""
    g_feats = wrapper.encode_images_eval(c_paths, bs=96, num_workers=num_workers)
    q_feats = wrapper.encode_queries_eval([str(PP_DIR / r["ref_image_rel"]) for r in test_q], [r["text"] for r in test_q], bs=64, num_workers=num_workers)
    sims = wrapper.score_eval(q_feats, g_feats)
    top50 = np.argsort(-sims, axis=1, kind="stable")[:, :50]

    preds_raw = {test_q[i]["query_id"]: [c_ids[ci] for ci in top50[i]] for i in range(len(test_q))}
    preds_noref = {test_q[i]["query_id"]: [c_ids[ci] for ci in top50[i] if c_ids[ci] != test_q[i]["ref_id"]][:50] for i in range(len(test_q))}
    q_recs = load_split_bundle("pinpoint", "test", corpus_scope="split").queries
    ev_raw = evaluate_pinpoint(preds_raw, q_recs, ks=(1, 5, 10), use_raw_gt=True, strict=True)
    ev_nr = evaluate_pinpoint(preds_noref, q_recs, ks=(1, 5, 10), use_raw_gt=True, strict=True)
    ref_copy1 = float(np.mean([preds_raw[r["query_id"]][0] == r["ref_id"] for r in test_q]))

    # Per-domain mAP@10
    by_dom: Dict[str, List[float]] = defaultdict(list)
    for r in test_q:
        ps = set(r["positive_ids"])
        ranked = preds_raw[r["query_id"]][:10]
        hits, sap = 0, 0.0
        for rk_i, cid in enumerate(ranked, 1):
            if cid in ps:
                hits += 1
                sap += hits / rk_i
        ap = sap / min(10, max(len(ps), 1))
        dom = r["metadata"].get("l1_interest", "UNKNOWN")
        by_dom[dom].append(ap)

    del g_feats, q_feats, sims
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "n_queries": len(test_q),
        "n_corpus": len(c_ids),
        "raw_mAP@10": ev_raw["overall"]["mAP@10"],
        "raw_P@1": ev_raw["overall"]["precision@1"],
        "raw_NegRecall@10": ev_raw["overall"]["NegRecall@10"],
        "noref_mAP@10": ev_nr["overall"]["mAP@10"],
        "noref_P@1": ev_nr["overall"]["precision@1"],
        "RefCopy@1": ref_copy1,
        "per_category_mAP10": {c: v["mAP@10"] for c, v in ev_raw.get("per_category", {}).items()},
        "per_domain_mAP10": {d: float(np.mean(v)) for d, v in by_dom.items()},
    }


# ---------------------------------------------------------------------------
# E02: Controlled Same-Scene Negative & Reference-Negative Intervention (+ E17 checkpoints!)
# ---------------------------------------------------------------------------

def run_e02_and_e17_traj(model_key: str, out_root: Path, num_workers: int, force: bool) -> None:
    out_dir = out_root / "e02_same_scene" / model_key
    done_p = out_dir / "DONE.json"
    if done_p.is_file() and not force:
        print(f"[E02 SKIP] {model_key} already done ({done_p})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load EDIR splits & frozen CLIP arbiter for difficulty-matched other-scene hard negatives!
    edir_tr = _load_jsonl(EDIR_SPLIT / "train.jsonl")
    edir_te = _load_jsonl(EDIR_SPLIT / "test.jsonl")
    edir_corp = _load_jsonl(EDIR_SPLIT / "corpus_full.jsonl")
    edir_c_ids = [r["image_id"] for r in edir_corp]
    edir_c_paths = [str(EDIR_DIR / r["image_rel"]) for r in edir_corp]

    # Mine difficulty-matched other-scene hard negatives on EDIR train using frozen CLIP ViT-L/14!
    arb_dir = out_root / "arbiter_clip/edir"
    c_clip = np.load(arb_dir / "corpus_emb.npy").astype(np.float32)
    c_clip_ids = json.loads((arb_dir / "corpus_ids.json").read_text())
    c_clip_map = {cid: i for i, cid in enumerate(c_clip_ids)}
    r_clip = np.load(arb_dir / "extra_ref_emb.npy").astype(np.float32)
    r_clip_ids = json.loads((arb_dir / "extra_ref_ids.json").read_text())
    r_clip_map = {rid: i for i, rid in enumerate(r_clip_ids)}

    # Train-only corpus indices (strictly zero val/test leakage!)
    tr_corp = _load_jsonl(EDIR_SPLIT / "corpus_train.jsonl")
    tr_c_ids = [r["image_id"] for r in tr_corp]
    tr_c_clip_idxs = np.array([c_clip_map[cid] for cid in tr_c_ids], dtype=np.int64)
    tr_c_emb_t = torch.from_numpy(c_clip[tr_c_clip_idxs]).to(device)
    tr_c_stems = [cid.rsplit("_", 1)[0] for cid in tr_c_ids]

    tr_c_stems_arr = np.array(tr_c_stems, dtype=object)
    print(f"[E02/{model_key}] Mining global CLIP-difficulty-matched other-scene hard negatives on EDIR train ({len(edir_tr)} queries)...", flush=True)
    sib_clip_sims = []
    other_hard_map: Dict[str, Tuple[str, float]] = {}
    for s in range(0, len(edir_tr), 128):
        e = min(s + 128, len(edir_tr))
        rv = torch.from_numpy(np.stack([r_clip[r_clip_map[r["ref_id"]]] for r in edir_tr[s:e]], axis=0)).to(device)
        sc = (rv @ tr_c_emb_t.T).cpu().numpy()  # [B, N_tr_corp]
        for bi, r in enumerate(edir_tr[s:e]):
            stem = r["metadata"]["stem_base"]
            # Measure actual sibling CLIP similarity to ref
            sib_s = [float(np.dot(r_clip[r_clip_map[r["ref_id"]]], c_clip[c_clip_map[nid]])) for nid in r["negative_ids"] if nid in c_clip_map]
            target_sim = float(np.mean(sib_s)) if sib_s else 0.70
            sib_clip_sims.append(target_sim)
            # Global argmin |sim - target_sim| over ALL non-same-scene train-corpus images!
            diffs = np.abs(sc[bi] - target_sim)
            diffs[tr_c_stems_arr == stem] = np.inf
            best_ci = int(np.argmin(diffs))
            other_hard_map[r["query_id"]] = (tr_c_ids[best_ci], float(sc[bi, best_ci]))
    del tr_c_emb_t
    torch.cuda.empty_cache()

    wrapper = _TrainableCIR(model_key, device)
    # Fixed training budget across ALL EDIR arms: 4 epochs x 3,500 queries @ bs=16 (grad_accum=2 -> eff_bs=32, 436 steps!)
    n_epochs = 4
    micro_bs = 16
    grad_accum = 2

    edir_arms = [
        ("baseline_0pct", 0.0, "none"),
        ("samescene_10pct", 0.10, "same_scene"),
        ("samescene_25pct", 0.25, "same_scene"),
        ("samescene_50pct", 0.50, "same_scene"),
        ("samescene_100pct", 1.00, "same_scene"),
        ("otherscene_hard_matched", 1.00, "other_hard"),
        ("ref_image_negative", 1.00, "ref_neg"),
    ]

    class _EDIRArmDS(Dataset):
        def __init__(self, rows, pp, arm_ratio, neg_kind, seed=42):
            self.rows = rows
            self.pp = pp
            self.arm_ratio = arm_ratio
            self.neg_kind = neg_kind
            self.rng = random.Random(seed)
            # Deterministically assign which queries carry an explicit hard negative according to arm_ratio
            n_active = int(round(len(rows) * arm_ratio))
            idxs = list(range(len(rows)))
            self.rng.shuffle(idxs)
            self.active_set = set(idxs[:n_active])
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            ref_im = self.pp(_open_rgb(str(EDIR_DIR / r["ref_image_rel"])))
            pos_im = self.pp(_open_rgb(str(EDIR_DIR / "imgs_mc" / r["positive_ids"][0])))
            has_en = i in self.active_set and self.neg_kind != "none"
            if has_en and self.neg_kind == "same_scene":
                nid = self.rng.choice(r["negative_ids"])
                neg_p = str(EDIR_DIR / "imgs_mc" / nid)
            elif has_en and self.neg_kind == "other_hard":
                nid, _ = other_hard_map[r["query_id"]]
                neg_p = str(EDIR_DIR / "imgs_mc" / nid)
            elif has_en and self.neg_kind == "ref_neg":
                neg_p = str(EDIR_DIR / r["ref_image_rel"])
            else:
                neg_p = str(EDIR_DIR / "imgs_mc" / r["negative_ids"][0])
            neg_im = self.pp(_open_rgb(neg_p))
            return ref_im, pos_im, neg_im, r["text"], int(has_en)

    # First evaluate Stage 0 (init ep0) ONCE for E17 trajectory!
    wrapper.reset_to_init()
    t0 = time.time()
    init_metrics = _eval_edir_full_diagnostics(wrapper, edir_te, edir_c_ids, edir_c_paths, num_workers)
    print(f"  [E02/E17 {model_key} EDIR init ep0] R@1={init_metrics['R@1']:.4f} Scene@1={init_metrics['Scene@1']:.4f} CondR@1={init_metrics['CondR@1_within_scene']:.4f} RefBeatsPos={init_metrics['RefBeatsPos_rate']:.4f}", flush=True)

    edir_results: Dict[str, Any] = {
        "model": model_key,
        "dataset": "edir",
        "budget": {"epochs": n_epochs, "micro_bs": micro_bs, "grad_accum": grad_accum, "effective_bs": micro_bs * grad_accum},
        "difficulty_stats_clip": {
            "mean_same_scene_sib_clip_sim": float(np.mean(sib_clip_sims)),
            "mean_other_scene_hard_clip_sim": float(np.mean([v[1] for v in other_hard_map.values()])),
        },
        "init_ep0": init_metrics,
        "arms": {},
    }
    e17_edir_traj: List[Dict[str, Any]] = [{"stage": "init_ep0", "epoch": 0, **init_metrics}]

    for arm_name, ratio, neg_kind in edir_arms:
        _seed_all(42)
        wrapper.reset_to_init()
        opt = wrapper.make_optimizer()
        ds_arm = _EDIRArmDS(edir_tr, wrapper.preprocess, ratio, neg_kind, seed=42)
        achieved_ratio = len(ds_arm.active_set) / float(len(edir_tr)) if neg_kind != "none" else 0.0
        dl = DataLoader(ds_arm, batch_size=micro_bs, shuffle=True, drop_last=True, num_workers=min(num_workers, 6), pin_memory=True)

        t_arm = time.time()
        steps = 0
        for ep in range(1, n_epochs + 1):
            wrapper.set_train()
            opt.zero_grad(set_to_none=True)
            ep_loss = 0.0
            n_b = 0
            for bi, (ref_b, pos_b, neg_b, txt_b, has_en_b) in enumerate(dl):
                ref_b = ref_b.to(device, non_blocking=True)
                pos_b = pos_b.to(device, non_blocking=True)
                use_en = neg_kind != "none" and int(has_en_b.sum().item()) > 0
                neg_in = neg_b.to(device, non_blocking=True) if use_en else None
                # Mask matrix [B, B]: row i only attends to its OWN explicit hard negative column i when has_en_b[i]==1!
                en_mask = torch.diag(has_en_b.to(device).bool()) if use_en else None
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    loss = wrapper.compute_infonce_loss(ref_b, list(txt_b), pos_b, extra_neg_imgs=neg_in, extra_neg_mask=en_mask) / grad_accum
                loss.backward()
                ep_loss += float(loss.detach().cpu().item()) * grad_accum
                n_b += 1
                if (bi + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.trainable_params(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    steps += 1
            # For baseline_0pct, ALSO record full E17 diagnostic probe metrics at ep1, ep2, ep4 (early / mid / final)!
            if arm_name == "baseline_0pct" and ep in (1, 2, n_epochs):
                ev_ep = _eval_edir_full_diagnostics(wrapper, edir_te, edir_c_ids, edir_c_paths, num_workers)
                e17_edir_traj.append({"stage": f"ep{ep}", "epoch": ep, "train_loss": ep_loss / max(n_b, 1), **ev_ep})
                print(f"    [E17-TRAJ {model_key} EDIR ep{ep}] R@1={ev_ep['R@1']:.4f} Scene@1={ev_ep['Scene@1']:.4f} CondR@1={ev_ep['CondR@1_within_scene']:.4f} sib_margin={ev_ep['mean_sibling_margin']:.4f}", flush=True)

        ev_final = e17_edir_traj[-1] if arm_name == "baseline_0pct" else _eval_edir_full_diagnostics(wrapper, edir_te, edir_c_ids, edir_c_paths, num_workers)
        edir_results["arms"][arm_name] = {
            "requested_neg_ratio": ratio,
            "achieved_neg_ratio": achieved_ratio,
            "neg_kind": neg_kind,
            "optimizer_steps": steps,
            "elapsed_sec": round(time.time() - t_arm, 1),
            "test_metrics": ev_final,
        }
        print(f"  [E02 {model_key} EDIR {arm_name:<24}] steps={steps} R@1={ev_final['R@1']:.4f} Scene@1={ev_final['Scene@1']:.4f} CondR@1={ev_final['CondR@1_within_scene']:.4f} RefBeatsPos={ev_final['RefBeatsPos_rate']:.4f} ({time.time()-t_arm:.1f}s)", flush=True)

    # Now PinPoint F02 Reference-Negative vs Explicit-Negative vs Other-Hard-Negative Arms!
    pp_tr = _load_jsonl(PP_SPLIT / "train.jsonl")
    pp_te = _load_jsonl(PP_SPLIT / "test.jsonl")
    pp_te_corp = _load_jsonl(PP_SPLIT / "corpus_test.jsonl")
    # Ensure all test refs & positives are in evaluation corpus
    pp_c_map = {r["image_id"]: str(PP_DIR / r["image_rel"]) for r in pp_te_corp}
    for r in pp_te:
        pp_c_map.setdefault(r["ref_id"], str(PP_DIR / r["ref_image_rel"]))
    pp_c_ids = sorted(pp_c_map.keys())
    pp_c_paths = [pp_c_map[i] for i in pp_c_ids]
    pp_all_c_map = {r["image_id"]: str(PP_DIR / r["image_rel"]) for r in _load_jsonl(PP_SPLIT / "corpus_full.jsonl")}

    wrapper.reset_to_init()
    pp_init = _eval_pinpoint_diagnostics(wrapper, pp_te, pp_c_ids, pp_c_paths, num_workers)
    print(f"  [E02/E17 {model_key} PP init ep0] raw_mAP@10={pp_init['raw_mAP@10']:.4f} NoRef={pp_init['noref_mAP@10']:.4f} RefCopy@1={pp_init['RefCopy@1']:.4f}", flush=True)

    pp_arms = [
        ("pp_baseline", "none"),
        ("pp_ref_negative", "ref_neg"),
        ("pp_explicit_neg", "explicit_neg"),
        ("pp_random_extra_neg", "rand_extra"),
    ]
    pp_results: Dict[str, Any] = {"model": model_key, "dataset": "pinpoint", "corpus_scope": "corpus_test_8609", "init_ep0": pp_init, "arms": {}}
    e17_pp_traj: List[Dict[str, Any]] = [{"stage": "init_ep0", "epoch": 0, **pp_init}]

    class _PPArmDS(Dataset):
        def __init__(self, rows, pp, neg_kind, seed=42):
            self.rows = [r for r in rows if any(p in pp_all_c_map for p in r["positive_ids"])]
            self.valid_pos = [[p for p in r["positive_ids"] if p in pp_all_c_map] for r in self.rows]
            self.pp = pp
            self.neg_kind = neg_kind
            self.rng = random.Random(seed)
            self.tr_pos_pool = [p for vp in self.valid_pos for p in vp[:2]]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            ref_p = str(PP_DIR / r["ref_image_rel"])
            pos_id = self.rng.choice(self.valid_pos[i])
            pos_p = pp_all_c_map[pos_id]
            pos_set = set(r["positive_ids"])
            has_en = False
            neg_p = ref_p
            if self.neg_kind == "ref_neg" and r["ref_id"] not in pos_set:
                neg_p = ref_p
                has_en = True
            elif self.neg_kind == "explicit_neg":
                cands = [n for n in r.get("negative_ids", []) if n in pp_all_c_map and n != r["ref_id"] and n not in pos_set]
                if cands:
                    neg_p = pp_all_c_map[self.rng.choice(cands)]
                    has_en = True
            elif self.neg_kind == "rand_extra":
                nid = self.rng.choice(self.tr_pos_pool)
                if nid not in pos_set:
                    neg_p = pp_all_c_map[nid]
                    has_en = True
            return self.pp(_open_rgb(ref_p)), self.pp(_open_rgb(pos_p)), self.pp(_open_rgb(neg_p)), r["text"], int(has_en), int(i), pos_id

    # Fixed budget: 3 epochs x 4,724 queries (~441 steps @ eff_bs=32)
    pp_epochs = 3
    for arm_name, neg_kind in pp_arms:
        _seed_all(42)
        wrapper.reset_to_init()
        opt = wrapper.make_optimizer()
        ds_arm = _PPArmDS(pp_tr, wrapper.preprocess, neg_kind, seed=42)
        pos_sets_tr = [set(r["positive_ids"]) for r in ds_arm.rows]
        dl = DataLoader(ds_arm, batch_size=micro_bs, shuffle=True, drop_last=True, num_workers=min(num_workers, 6), pin_memory=True)
        t_arm = time.time()
        steps = 0
        for ep in range(1, pp_epochs + 1):
            wrapper.set_train()
            opt.zero_grad(set_to_none=True)
            for bi, (ref_b, pos_b, neg_b, txt_b, has_en_b, row_idx_b, pos_id_b) in enumerate(dl):
                ref_b = ref_b.to(device, non_blocking=True)
                pos_b = pos_b.to(device, non_blocking=True)
                use_en = neg_kind != "none" and int(has_en_b.sum().item()) > 0
                neg_in = neg_b.to(device, non_blocking=True) if use_en else None
                en_mask = torch.diag(has_en_b.to(device).bool()) if use_en else None
                # Mask off-diagonal false negatives when two rows in a PinPoint batch share a valid positive!
                b_len = len(pos_id_b)
                fn_mask = torch.zeros((b_len, b_len), dtype=torch.bool, device=device)
                r_idxs = row_idx_b.tolist()
                for ii in range(b_len):
                    pset_i = pos_sets_tr[r_idxs[ii]]
                    for jj in range(b_len):
                        if ii != jj and pos_id_b[jj] in pset_i:
                            fn_mask[ii, jj] = True
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    loss = wrapper.compute_infonce_loss(
                        ref_b, list(txt_b), pos_b, extra_neg_imgs=neg_in, extra_neg_mask=en_mask, false_neg_mask=fn_mask
                    ) / grad_accum
                loss.backward()
                if (bi + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(wrapper.trainable_params(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    steps += 1
            if arm_name == "pp_baseline" and ep in (1, 2, pp_epochs):
                ev_ep = _eval_pinpoint_diagnostics(wrapper, pp_te, pp_c_ids, pp_c_paths, num_workers)
                e17_pp_traj.append({"stage": f"ep{ep}", "epoch": ep, **ev_ep})
        ev_final = e17_pp_traj[-1] if arm_name == "pp_baseline" else _eval_pinpoint_diagnostics(wrapper, pp_te, pp_c_ids, pp_c_paths, num_workers)
        pp_results["arms"][arm_name] = {"neg_kind": neg_kind, "optimizer_steps": steps, "elapsed_sec": round(time.time() - t_arm, 1), "test_metrics": ev_final}
        print(f"  [E02 {model_key} PP {arm_name:<22}] raw_mAP@10={ev_final['raw_mAP@10']:.4f} NoRef={ev_final['noref_mAP@10']:.4f} RefCopy@1={ev_final['RefCopy@1']:.4f} ({time.time()-t_arm:.1f}s)", flush=True)

    _write_json(out_dir / "e02_edir_results.json", edir_results)
    _write_json(out_dir / "e02_pinpoint_results.json", pp_results)
    _write_json(out_root / "e17_dynamics" / f"e17_trajectory_{model_key}.json", {"suite": "E17_trajectory", "model": model_key, "edir_trajectory": e17_edir_traj, "pinpoint_trajectory": e17_pp_traj})
    _write_json(done_p, {"suite": "E02+E17_traj", "model": model_key, "status": "done"})


# ---------------------------------------------------------------------------
# E16: Cross-Domain Gradient Conflict + Transfer Matrix + Causal PCGrad Arm!
# ---------------------------------------------------------------------------

def run_e16_gradient_conflict(model_key: str, out_root: Path, num_workers: int, force: bool, grad_only: bool = False) -> None:
    out_dir = out_root / "e16_domain_conflict" / model_key
    done_p = out_dir / "DONE.json"
    if done_p.is_file() and not force and not grad_only:
        print(f"[E16 SKIP] {model_key} already done ({done_p})", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pp_tr = _load_jsonl(PP_SPLIT / "train.jsonl")
    pp_te = _load_jsonl(PP_SPLIT / "test.jsonl")
    pp_all_c_map = {r["image_id"]: str(PP_DIR / r["image_rel"]) for r in _load_jsonl(PP_SPLIT / "corpus_full.jsonl")}
    pp_te_corp = _load_jsonl(PP_SPLIT / "corpus_test.jsonl")
    pp_c_map = {r["image_id"]: str(PP_DIR / r["image_rel"]) for r in pp_te_corp}
    for r in pp_te:
        pp_c_map.setdefault(r["ref_id"], str(PP_DIR / r["ref_image_rel"]))
    pp_c_ids = sorted(pp_c_map.keys())
    pp_c_paths = [pp_c_map[i] for i in pp_c_ids]

    by_dom_tr: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pp_tr:
        if any(p in pp_all_c_map for p in r["positive_ids"]):
            by_dom_tr[r["metadata"].get("l1_interest", "UNKNOWN")].append(r)

    wrapper = _TrainableCIR(model_key, device)

    # Bit-exact deterministic image cache (capped at 4000 tensors ~2.4 GiB) + 6-thread parallel PIL loader
    # so GPU stays saturated at 95-100% instead of idling on single-thread JPEG decode!
    from concurrent.futures import ThreadPoolExecutor
    img_pool = ThreadPoolExecutor(max_workers=min(max(num_workers, 2), 6))
    img_cache: Dict[str, torch.Tensor] = {}

    def _dec_one(path: str) -> torch.Tensor:
        return wrapper.preprocess(_open_rgb(path))

    def _load_imgs_tensor(paths: Sequence[str]) -> torch.Tensor:
        misses = sorted({p for p in paths if p not in img_cache})
        if misses:
            loaded = list(img_pool.map(_dec_one, misses))
            for p, t in zip(misses, loaded):
                if len(img_cache) < 4000:
                    img_cache[p] = t
            miss_map = dict(zip(misses, loaded))
            return torch.stack([img_cache.get(p, miss_map.get(p)) for p in paths]).to(device, non_blocking=True)
        return torch.stack([img_cache[p] for p in paths]).to(device, non_blocking=True)

    def _sample_dom_batch(
        dom_rows: List[Dict[str, Any]], bs: int, rng: random.Random
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
        """Sample without replacement, draw a random valid positive per query, and build exact false-negative mask."""
        chosen = rng.sample(dom_rows, bs) if len(dom_rows) >= bs else [rng.choice(dom_rows) for _ in range(bs)]
        ref_ps, tgt_ps, txts, chosen_pids, psets = [], [], [], [], []
        for r in chosen:
            vpos = [p for p in r["positive_ids"] if p in pp_all_c_map]
            pid = rng.choice(vpos)
            ref_ps.append(str(PP_DIR / r["ref_image_rel"]))
            tgt_ps.append(pp_all_c_map[pid])
            txts.append(r["text"])
            chosen_pids.append(pid)
            psets.append(set(r["positive_ids"]))
        fn_mask = torch.zeros((bs, bs), dtype=torch.bool, device=device)
        for ii in range(bs):
            for jj in range(bs):
                if ii != jj and chosen_pids[jj] in psets[ii]:
                    fn_mask[ii, jj] = True
        return _load_imgs_tensor(ref_ps), txts, _load_imgs_tensor(tgt_ps), fn_mask

    def _flat_grad(params: List[nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.grad.detach().float().reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=device) for p in params])

    # Part 1: Pairwise Inter-Domain Gradient Cosine Matrix at BOTH `init` AND `trained_best` checkpoints!
    grad_matrices: Dict[str, Any] = {}
    for ckpt_tag in ("init_pretrained", "trained_pinpoint_best"):
        if ckpt_tag == "init_pretrained":
            wrapper.reset_to_init()
        else:
            wrapper.load_trained_best("pinpoint")
        wrapper.set_train()
        params = wrapper.trainable_params()
        dom_grads: Dict[str, torch.Tensor] = {}
        for dom in TOP7_PP_DOMAINS:
            rng_d = random.Random(42 + (_det_hash_int(dom) % 10000))
            g_acc = None
            n_batches = 4
            for _ in range(n_batches):
                for p in params:
                    p.grad = None
                rb, tb, gb, fn_m = _sample_dom_batch(by_dom_tr[dom], bs=16, rng=rng_d)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    loss = wrapper.compute_infonce_loss(rb, tb, gb, false_neg_mask=fn_m)
                loss.backward()
                fg = _flat_grad(params).cpu()
                g_acc = fg if g_acc is None else (g_acc + fg)
            dom_grads[dom] = g_acc / float(n_batches)
        mat = {}
        neg_pairs = []
        # Chunked IEEE-754 float64 dot products & norms: exact 15-digit precision with <64 MiB RAM overhead!
        def _dot_f64(a: torch.Tensor, b: torch.Tensor, ch: int = 4_000_000) -> float:
            acc = 0.0
            n_el = a.numel()
            for s_i in range(0, n_el, ch):
                e_i = min(s_i + ch, n_el)
                acc += float(torch.dot(a[s_i:e_i].double(), b[s_i:e_i].double()).item())
            return acc

        dom_norms_f64 = {d: math.sqrt(max(_dot_f64(g, g), 1e-30)) for d, g in dom_grads.items()}
        for d1 in TOP7_PP_DOMAINS:
            mat[d1] = {}
            for d2 in TOP7_PP_DOMAINS:
                if d1 == d2:
                    c = 1.0
                elif d2 in mat and d1 in mat[d2]:
                    c = mat[d2][d1]
                else:
                    denom = max(dom_norms_f64[d1] * dom_norms_f64[d2], 1e-18)
                    c = max(-1.0, min(1.0, _dot_f64(dom_grads[d1], dom_grads[d2]) / denom))
                mat[d1][d2] = round(c, 5)
                if d1 < d2 and c < 0.0:
                    neg_pairs.append({"d1": d1, "d2": d2, "cos": round(c, 5)})
        del dom_grads
        gc.collect()
        grad_matrices[ckpt_tag] = {
            "domains": TOP7_PP_DOMAINS,
            "cosine_matrix": mat,
            "mean_off_diag_cos": float(np.mean([mat[d1][d2] for d1 in TOP7_PP_DOMAINS for d2 in TOP7_PP_DOMAINS if d1 != d2])),
            "n_negative_pairs": len(neg_pairs),
            "negative_pairs": neg_pairs,
        }
        print(f"  [E16 {model_key} grad_cos {ckpt_tag}] mean_off_diag={grad_matrices[ckpt_tag]['mean_off_diag_cos']:.4f}, n_neg_pairs={len(neg_pairs)}", flush=True)

    if grad_only:
        img_pool.shutdown(wait=False)
        img_cache.clear()
        _write_json(out_dir / "e16_gradient_cosine.json", {"suite": "E16_grad_cosine", "model": model_key, "precision": "float64", **grad_matrices})
        print(f"  [E16 {model_key}] Updated float64 e16_gradient_cosine.json cleanly.", flush=True)
        return

    # Part 2: Controlled train-domain x test-domain Transfer Matrix + Causal PCGrad Arm!
    # Strictly matched across ALL 8 ARMS: 240 optimizer steps x 2 micro-batches x bs=16 = 7,680 total exposures,
    # identical 16-way InfoNCE denominator, identical 2-microbatch gradient averaging!
    total_steps = 240
    bs_arm = 16
    all_tr_valid = [r for d in by_dom_tr.values() for r in d]
    arms_e16 = [
        ("single_WomensFashion", "single", "Women's Fashion"),
        ("single_HomeDecor", "single", "Home Decor"),
        ("single_Animals", "single", "Animals"),
        ("single_Entertainment", "single", "Entertainment"),
        ("single_Art", "single", "Art"),
        ("mixed_original", "mixed_orig", None),
        ("mixed_balanced_top7", "mixed_bal", None),
        ("mixed_pcgrad_top7", "pcgrad", None),
    ]
    transfer_res: Dict[str, Any] = {
        "model": model_key,
        "dataset": "pinpoint",
        "corpus_scope": "corpus_test_8609",
        "matched_total_steps": total_steps,
        "micro_batch_size": bs_arm,
        "micro_batches_per_step": 2,
        "matched_total_exposures": total_steps * 2 * bs_arm,
        "arms": {},
    }

    for arm_name, mode, target_dom in arms_e16:
        _seed_all(42)
        rng_a = random.Random(42)
        wrapper.reset_to_init()
        opt = wrapper.make_optimizer()
        params = wrapper.trainable_params()
        dom_exposure: Counter = Counter()
        pcgrad_conflicts = 0
        t_a = time.time()

        wrapper.set_train()
        for st in range(total_steps):
            opt.zero_grad(set_to_none=True)
            if mode == "single":
                for _ in range(2):
                    rb, tb, gb, fn_m = _sample_dom_batch(by_dom_tr[target_dom], bs=bs_arm, rng=rng_a)
                    dom_exposure[target_dom] += bs_arm
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        loss = wrapper.compute_infonce_loss(rb, tb, gb, false_neg_mask=fn_m) * 0.5
                    loss.backward()
            elif mode == "mixed_orig":
                for _ in range(2):
                    chosen = rng_a.sample(all_tr_valid, bs_arm)
                    ref_ps, tgt_ps, txts, cpids, psets = [], [], [], [], []
                    for r in chosen:
                        dom_exposure[r["metadata"].get("l1_interest", "UNKNOWN")] += 1
                        vpos = [p for p in r["positive_ids"] if p in pp_all_c_map]
                        pid = rng_a.choice(vpos)
                        ref_ps.append(str(PP_DIR / r["ref_image_rel"]))
                        tgt_ps.append(pp_all_c_map[pid])
                        txts.append(r["text"])
                        cpids.append(pid)
                        psets.append(set(r["positive_ids"]))
                    fn_m = torch.zeros((bs_arm, bs_arm), dtype=torch.bool, device=device)
                    for ii in range(bs_arm):
                        for jj in range(bs_arm):
                            if ii != jj and cpids[jj] in psets[ii]:
                                fn_m[ii, jj] = True
                    rb = _load_imgs_tensor(ref_ps)
                    gb = _load_imgs_tensor(tgt_ps)
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        loss = wrapper.compute_infonce_loss(rb, txts, gb, false_neg_mask=fn_m) * 0.5
                    loss.backward()
            elif mode in ("mixed_bal", "pcgrad"):
                # Confound-free paired comparison: BOTH mixed_bal AND pcgrad sample the EXACT SAME
                # two domains (d1, d2) with IDENTICAL batch_size=bs_arm (16) and InfoNCE denominator!
                # The ONLY difference is causal PCGrad projection when dot(g1, g2) < 0!
                d1, d2 = rng_a.sample(TOP7_PP_DOMAINS, 2)
                dom_exposure[d1] += bs_arm
                dom_exposure[d2] += bs_arm
                opt.zero_grad(set_to_none=True)
                rb1, tb1, gb1, fn1 = _sample_dom_batch(by_dom_tr[d1], bs=bs_arm, rng=rng_a)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    l1 = wrapper.compute_infonce_loss(rb1, tb1, gb1, false_neg_mask=fn1)
                l1.backward()
                g1_list = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                opt.zero_grad(set_to_none=True)
                rb2, tb2, gb2, fn2 = _sample_dom_batch(by_dom_tr[d2], bs=bs_arm, rng=rng_a)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    l2 = wrapper.compute_infonce_loss(rb2, tb2, gb2, false_neg_mask=fn2)
                l2.backward()
                g2_list = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                if mode == "pcgrad":
                    dot = sum(float((a.float() * b.float()).sum().item()) for a, b in zip(g1_list, g2_list))
                    n1_sq = sum(float((a.float() * a.float()).sum().item()) for a in g1_list) + 1e-12
                    n2_sq = sum(float((b.float() * b.float()).sum().item()) for b in g2_list) + 1e-12
                    if dot < 0.0:
                        pcgrad_conflicts += 1
                        c1 = dot / n2_sq
                        c2 = dot / n1_sq
                        for p, g1, g2 in zip(params, g1_list, g2_list):
                            p.grad = 0.5 * ((g1 - c1 * g2) + (g2 - c2 * g1))
                    else:
                        for p, g1, g2 in zip(params, g1_list, g2_list):
                            p.grad = 0.5 * (g1 + g2)
                else:
                    for p, g1, g2 in zip(params, g1_list, g2_list):
                        p.grad = 0.5 * (g1 + g2)

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        ev = _eval_pinpoint_diagnostics(wrapper, pp_te, pp_c_ids, pp_c_paths, num_workers)
        transfer_res["arms"][arm_name] = {
            "total_steps": total_steps,
            "total_exposures": int(sum(dom_exposure.values())),
            "exposure_per_domain": dict(dom_exposure),
            "pcgrad_conflict_steps": pcgrad_conflicts,
            "elapsed_sec": round(time.time() - t_a, 1),
            "overall_raw_mAP10": ev["raw_mAP@10"],
            "overall_noref_mAP10": ev["noref_mAP@10"],
            "per_domain_mAP10": {d: ev["per_domain_mAP10"].get(d, float("nan")) for d in TOP7_PP_DOMAINS + ["Men's Fashion"]},
        }
        wf_val = ev["per_domain_mAP10"].get("Women's Fashion", 0.0)
        print(
            f"  [E16 {model_key} {arm_name:<22}] overall={ev['raw_mAP@10']:.4f} "
            f"Animals={ev['per_domain_mAP10'].get('Animals',0):.4f} "
            f"Ent={ev['per_domain_mAP10'].get('Entertainment',0):.4f} "
            f"WF={wf_val:.4f} "
            f"({time.time()-t_a:.1f}s)",
            flush=True,
        )

    img_pool.shutdown(wait=False)
    img_cache.clear()
    _write_json(out_dir / "e16_gradient_cosine.json", {"suite": "E16_grad_cosine", "model": model_key, **grad_matrices})
    _write_json(out_dir / "e16_transfer_matrix.json", transfer_res)
    _write_json(done_p, {"suite": "E16", "model": model_key, "status": "done"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled training suites E02, E16, E17")
    parser.add_argument("--suite", type=str, required=True, choices=["e02", "e16", "e16_grad_only", "e17_history", "e17_trajectory"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.suite == "e17_history":
        run_e17_history_all16(args.out_dir)
    elif args.suite in ("e02", "e17_trajectory"):
        run_e02_and_e17_traj(args.model.lower(), args.out_dir, num_workers=min(args.num_workers, 6), force=args.force)
    elif args.suite == "e16":
        run_e16_gradient_conflict(args.model.lower(), args.out_dir, num_workers=min(args.num_workers, 6), force=args.force, grad_only=False)
    elif args.suite == "e16_grad_only":
        run_e16_gradient_conflict(args.model.lower(), args.out_dir, num_workers=min(args.num_workers, 6), force=True, grad_only=True)


if __name__ == "__main__":
    main()
