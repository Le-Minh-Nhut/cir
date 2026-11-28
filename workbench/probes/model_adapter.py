"""Unified in-environment model adapter for all 16 CIR models across EDIR, PinPoint, and FashionIQ.

Runs inside each model's exact pinned virtualenv + PYTHONPATH. Provides:
  - Exact strict checkpoint loading (EDIR/PinPoint in-domain + FashionIQ author checkpoints)
  - Native gallery & query encoding + exact native scoring (single-vector, 32x256 MaxSim, DCNet 7-expert)
  - Exact unimodal paths: CIRCUS (empty-text / black-image) AND TRUE native unimodal towers!
  - Layer-wise query-branch AND gallery-branch representation hooks (E04, F28, F29)
  - Causal activation / token / spatial-ROI / block interventions with matched-norm & matched-area shams!
  - Temporary bicubic ViT pos_embed interpolation for E11 high-resolution testing (restored in try/finally)
  - Exact model tokenizer inspection for E12 / F25 Homophone Hijack
"""

from __future__ import annotations

import copy
import gc
import math
import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.probes.cohort import MODEL_BY_KEY, resolve_checkpoint_for  # noqa: E402


# ---------------------------------------------------------------------------
# Generic PIL / Path / Masked-Image Dataset helpers
# ---------------------------------------------------------------------------

def _open_pil(item: Union[str, Path, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(item, PIL.Image.Image):
        return item.convert("RGB")
    with PIL.Image.open(item) as im:
        return im.convert("RGB")


class _ImgListDataset(Dataset):
    def __init__(self, items: Sequence[Union[str, Path, PIL.Image.Image]], transform: Callable):
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        im = _open_pil(self.items[idx])
        return self.transform(im)


class _PairListDataset(Dataset):
    def __init__(
        self,
        imgs: Sequence[Union[str, Path, PIL.Image.Image]],
        texts: Sequence[str],
        transform: Callable,
        black_image: bool = False,
    ):
        self.imgs = list(imgs)
        self.texts = list(texts)
        self.transform = transform
        self.black_image = black_image

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        if self.black_image:
            im = PIL.Image.new("RGB", (224, 224), (0, 0, 0))
        else:
            im = _open_pil(self.imgs[idx])
        return self.transform(im), self.texts[idx]


def _make_loader(ds: Dataset, bs: int, num_workers: int) -> DataLoader:
    # PIL images in memory can't be cheaply forked across many workers; use 0 workers if items are PIL objects
    has_pil = len(ds) > 0 and hasattr(ds, "items") and isinstance(ds.items[0], PIL.Image.Image)
    has_pil2 = len(ds) > 0 and hasattr(ds, "imgs") and len(ds.imgs) > 0 and isinstance(ds.imgs[0], PIL.Image.Image)
    nw = 0 if (has_pil or has_pil2 or len(ds) < 64) else min(int(num_workers), 8)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if nw > 0 else None,
    )


def _infer_encoder_hparams(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    lw = int(state_dict["local_weight"].shape[0])
    ltn = int(state_dict["backbone.fac_net.weight.3.weight"].shape[0])
    n_p = int(state_dict["backbone.binding_decoder.mlp_p.0.weight"].shape[0])
    assert ltn + n_p + 1 == lw
    if "backbone.t_weight_phrases.5.weight" in state_dict:
        wc = 3
    elif "backbone.t_weight_phrases.3.weight" in state_dict:
        wc = 2
    else:
        wc = 1
    return ltn, wc, n_p


def _interpolate_pos_embed_1d(pos_embed: torch.Tensor, new_grid: int) -> torch.Tensor:
    """Bicubic interpolation of [1, 1 + G*G, D] or [1 + G*G, D] ViT positional embeddings."""
    squeeze = False
    pe = pos_embed
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)
        squeeze = True
    cls_tok = pe[:, :1, :]
    patch_tok = pe[:, 1:, :]
    n_old = patch_tok.shape[1]
    old_g = int(round(math.sqrt(n_old)))
    if old_g * old_g != n_old or old_g == new_grid:
        return pos_embed
    dim = patch_tok.shape[2]
    pt = patch_tok.reshape(1, old_g, old_g, dim).permute(0, 3, 1, 2).float()
    pt_new = F.interpolate(pt, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    pt_new = pt_new.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim).to(pos_embed.dtype)
    out = torch.cat([cls_tok, pt_new], dim=1)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Unified ModelHandle
# ---------------------------------------------------------------------------

class ModelHandle:
    """Unified runtime wrapper around any of the 16 CIR models."""

    def __init__(
        self,
        model_key: str,
        dataset: str,
        device: torch.device,
        fiq_category: Optional[str] = None,
        shared_vit: Optional[nn.Module] = None,
    ):
        self.model_key = model_key.lower().strip()
        self.dataset = dataset.lower().strip()
        self.device = device
        self.fiq_category = fiq_category
        self.spec = MODEL_BY_KEY[self.model_key]
        self.ckpt_info = resolve_checkpoint_for(self.model_key, self.dataset)
        self.model: Any = None
        self.extra: Dict[str, Any] = {}
        self.preprocess: Callable = None
        self.txt_proc: Callable[[str], str] = lambda s: str(s)
        self.load_report: Dict[str, Any] = {}
        self._load(shared_vit=shared_vit)

    @property
    def is_blip2(self) -> bool:
        return self.spec.family in ("blip2_evag", "blip2_vitl")

    def _load(self, shared_vit: Optional[nn.Module] = None) -> None:
        k = self.model_key
        ds = self.dataset
        ckpts = [Path(p) for p in self.ckpt_info["ckpts"]]

        # ------------------------------------------------------------------
        # 1. CLIP4Cir (RN50x4 + Combiner)
        # ------------------------------------------------------------------
        if k == "clip4cir":
            import clip
            from combiner import Combiner
            from data_utils import targetpad_transform

            clip_model, _ = clip.load("RN50x4", device=self.device, jit=False)
            clip_model.eval().float()
            c_state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            clip_model.load_state_dict(c_state["CLIP"], strict=True)
            combiner = Combiner(clip_model.visual.output_dim, 2560, 5120).to(self.device)
            comb_state = torch.load(ckpts[1], map_location=self.device, weights_only=False)
            combiner.load_state_dict(comb_state["Combiner"], strict=True)
            combiner.eval()
            self.model = clip_model
            self.extra["combiner"] = combiner
            self.extra["clip_mod"] = clip
            self.preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
            self.load_report = {"strict": True, "res": clip_model.visual.input_resolution}

        # ------------------------------------------------------------------
        # 2. ENCODER (OpenCLIP ViT-B-32)
        # ------------------------------------------------------------------
        elif k == "encoder":
            import open_clip
            from workbench.benchmarks.trainers.train_encoder import build_encoder_model, load_encoder_full_state

            raw = torch.load(ckpts[0], map_location="cpu", weights_only=False)
            if isinstance(raw, nn.Module):
                _, _, val_pp = build_encoder_model(device=self.device)
                model = raw.to(self.device).eval()
                self.load_report = {"whole_module": True}
            elif "learnable_relation" in raw and "model_state_dict" in raw:
                ltn, wc, n_p = _infer_encoder_hparams(raw["model_state_dict"])
                model, _, val_pp = build_encoder_model(local_token_num=ltn, wc=wc, n_p=n_p, device=self.device)
                load_encoder_full_state(model, raw, self.device)
                model.eval()
                self.load_report = {"strict": True, "ltn": ltn, "wc": wc, "n_p": n_p}
            else:
                sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
                ltn, wc, n_p = _infer_encoder_hparams(sd)
                torch.manual_seed(42)
                model, _, val_pp = build_encoder_model(local_token_num=ltn, wc=wc, n_p=n_p, device=self.device)
                res = model.load_state_dict(sd, strict=False)
                assert not res.missing_keys, f"ENCODER missing: {res.missing_keys}"
                model.eval()
                self.load_report = {"strict": False, "ltn": ltn, "wc": wc, "n_p": n_p}
            self.model = model
            self.preprocess = val_pp
            self.extra["open_clip"] = open_clip

        # ------------------------------------------------------------------
        # 3. TG-CIR (CLIP ViT-B/16)
        # ------------------------------------------------------------------
        elif k == "tgcir":
            import torch.nn.modules.linear as torch_linear
            if not hasattr(torch_linear, "_LinearWithBias"):
                torch_linear._LinearWithBias = torch.nn.Linear
            import clip
            import model as tgcir_mod

            _orig_tok = clip.tokenize
            def _safe_tok(texts, context_length=77, truncate=True):
                return _orig_tok(texts, context_length=context_length, truncate=True)
            clip.tokenize = _safe_tok
            tgcir_mod.clip.tokenize = _safe_tok

            raw = torch.load(ckpts[0], map_location="cpu", weights_only=False)
            _, pp = clip.load("ViT-B/16", device="cpu", jit=False)
            if isinstance(raw, nn.Module):
                for m in raw.modules():
                    if isinstance(m, nn.MultiheadAttention) and not hasattr(m, "batch_first"):
                        m.batch_first = False
                model = raw.to(self.device).eval()
                self.load_report = {"whole_module": True}
            else:
                model = tgcir_mod.VCG_CIR(img_encoder="ViT-B/16", hidden_dim=512, dropout=0.5, local_token_num=8, global_token_num=4).to(self.device)
                model.load_state_dict(raw["state_dict"], strict=True)
                model.eval()
                self.load_report = {"strict": True}
            self.model = model
            self.preprocess = pp
            self.extra["clip_mod"] = clip

        # ------------------------------------------------------------------
        # 4. LIMN (OpenCLIP ViT-L/14 DataComp-XL)
        # ------------------------------------------------------------------
        elif k == "limn":
            import open_clip
            _orig_create = open_clip.create_model_and_transforms
            def _patched_create(model_name, pretrained=None, *a, **kw):
                if isinstance(pretrained, str) and "[CLIP path]" in pretrained:
                    pretrained = "datacomp_xl_s13b_b90k"
                return _orig_create(model_name, pretrained=pretrained, *a, **kw)
            open_clip.create_model_and_transforms = _patched_create
            import model as limn_mod

            if ds == "fashioniq":
                cat = self.fiq_category or "dress"
                c_info = self.ckpt_info["per_category"][cat]
                dt_id = int(c_info["dt_id"])
                ckpt_p = Path(c_info["ckpt"])
                model = torch.load(ckpt_p, map_location=self.device, weights_only=False).to(self.device).eval()
                vis = model.backbone.clip.visual
                pp = open_clip.image_transform(vis.image_size, is_train=False, mean=vis.image_mean, std=vis.image_std)
                self.load_report = {"whole_module": True, "fiq_category": cat, "dt_id": dt_id}
            else:
                dt_id = 5
                raw = torch.load(ckpts[0], map_location="cpu", weights_only=False)
                if isinstance(raw, nn.Module):
                    model = raw.to(self.device).eval()
                else:
                    model = limn_mod.LIMN(prefix_num=10, head=8, layer=2, hidden_dim=768, temperature=0.1).to(self.device)
                    model.load_state_dict(raw["state_dict"], strict=True)
                    model.eval()
                vis = model.backbone.clip.visual
                pp = open_clip.image_transform(vis.image_size, is_train=False, mean=vis.image_mean, std=vis.image_std)
                self.load_report = {"strict": True, "dt_id": dt_id}
            self.model = model
            self.preprocess = pp
            self.extra["dt_id"] = dt_id
            self.extra["open_clip"] = open_clip

        # ------------------------------------------------------------------
        # 5. CLVC-Net (dual ResNet50 + LSTM)
        # ------------------------------------------------------------------
        elif k == "clvcnet":
            import model as clvc_mod
            import text_model as clvc_txt

            pp = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            if ds == "fashioniq":
                cat = self.fiq_category or "dress"
                c_info = self.ckpt_info["per_category"][cat]
                loc_m = torch.load(c_info["local"], map_location=self.device, weights_only=False).to(self.device).eval()
                glb_m = torch.load(c_info["global"], map_location=self.device, weights_only=False).to(self.device).eval()
                self.load_report = {"whole_module": True, "fiq_category": cat}
            else:
                raw = torch.load(ckpts[0], map_location="cpu", weights_only=False)
                word2id = raw["vocab_word2id"]
                vocab_words = [w for w, _ in sorted(word2id.items(), key=lambda kv: kv[1])]
                # Reconstruct exact vocab order from saved word2id!
                loc_m = clvc_mod.compose_local(texts=["dummy"], T=7.0)
                glb_m = clvc_mod.compose_global(texts=["dummy"], T=4.0)
                for m_sub, sd_key in [(loc_m, "local_state_dict"), (glb_m, "global_state_dict")]:
                    sd = raw[sd_key]
                    emb_w = sd["text_model.embedding_layer.weight"]
                    vocab_size = int(emb_w.shape[0])
                    m_sub.text_model.vocab.word2id = dict(word2id)
                    m_sub.text_model.vocab.wordcount = {w: 1 for w in word2id}
                    m_sub.text_model.embedding_layer = nn.Embedding(vocab_size, emb_w.shape[1])
                    m_sub.load_state_dict(sd, strict=True)
                    m_sub.to(self.device).eval()
                self.load_report = {"strict": True, "vocab_size": len(word2id)}
            self.model = loc_m
            self.extra["global_model"] = glb_m
            self.extra["vocab_helper"] = clvc_txt.SimpleVocab()
            self.preprocess = pp

        # ------------------------------------------------------------------
        # 6. DCNet (ResNet50 + GloVe 300d + 7-expert log_softmax)
        # ------------------------------------------------------------------
        elif k == "dcnet":
            from workbench.benchmarks.trainers.train_dcnet import (
                _tokenize_dcnet,
                build_dcnet_model,
                encode_captions_to_glove_batch,
            )

            model, modalities = build_dcnet_model(self.device)
            ckpt = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            sd = {kk.replace("module.", "", 1) if kk.startswith("module.") else kk: vv for kk, vv in sd.items()}
            model.load_state_dict(sd, strict=True)
            model.eval()
            if ds == "fashioniq":
                vocab_p = REPO_ROOT / "data/FashionIQ/captions/glove_vecs.pkl"
                if vocab_p.is_file():
                    with vocab_p.open("rb") as f:
                        vocab = pickle.load(f)
                else:
                    vocab = {}
            else:
                with Path(self.ckpt_info["glove_vocab"]).open("rb") as f:
                    vocab = pickle.load(f)
            pp = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.model = model
            self.extra["modalities"] = modalities
            self.extra["vocab"] = vocab
            self.extra["encode_glove"] = encode_captions_to_glove_batch
            self.extra["tokenize_dcnet"] = _tokenize_dcnet
            self.preprocess = pp
            self.load_report = {"strict": True, "modalities": modalities, "vocab_len": len(vocab)}

        # ------------------------------------------------------------------
        # 7. SPRC (blip2_cir_align_prompt: EVA-ViT-g on EDIR/PP, ViT-L on FIQ)
        # ------------------------------------------------------------------
        elif k == "sprc":
            from data_utils import targetpad_transform
            from lavis.models import load_model_and_preprocess

            backbone = self.ckpt_info.get("backbone", "pretrain")
            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_cir_align_prompt", model_type=backbone, is_eval=True, device=self.device
            )
            state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict_non_vit" in state:
                msg = model.load_state_dict(state["state_dict_non_vit"], strict=False)
                assert not [x for x in msg.missing_keys if not x.startswith("visual_encoder.")] and not msg.unexpected_keys
                self.load_report = {"strict_non_vit": True, "backbone": backbone}
            else:
                cls_key = model.__class__.__name__
                sd = state[cls_key] if isinstance(state, dict) and cls_key in state else state
                model.load_state_dict(sd, strict=True)
                self.load_report = {"strict": True, "backbone": backbone}
            model.eval()
            self.model = model
            self.txt_proc = txt_processors["eval"]
            self.preprocess = targetpad_transform(1.25, 224)

        # ------------------------------------------------------------------
        # 8. QuRe (blip2_qure, ViT-L)
        # ------------------------------------------------------------------
        elif k == "qure":
            from data.utils import targetpad_transform
            from lavis.models import load_model_and_preprocess

            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_qure", model_type="pretrain_vitL", is_eval=True, device=self.device
            )
            state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict_non_vit" in state:
                msg = model.load_state_dict(state["state_dict_non_vit"], strict=False)
                assert not [x for x in msg.missing_keys if not x.startswith("visual_encoder.")] and not msg.unexpected_keys
                self.load_report = {"strict_non_vit": True, "backbone": "pretrain_vitL"}
            else:
                model.load_state_dict(state, strict=True)
                self.load_report = {"strict": True, "backbone": "pretrain_vitL"}
            model.to(self.device).eval()
            self.model = model
            self.txt_proc = txt_processors["eval"]
            self.preprocess = targetpad_transform(1.25, 224)

        # ------------------------------------------------------------------
        # 9. MTST_PTHA (blip2_cir_cat, EVA-ViT-g)
        # ------------------------------------------------------------------
        elif k == "mtst_ptha":
            from data_utils import targetpad_transform
            from lavis.models import load_model_and_preprocess

            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_cir_cat", model_type="pretrain", is_eval=True, device=self.device
            )
            state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict_non_vit" in state:
                msg = model.load_state_dict(state["state_dict_non_vit"], strict=False)
                assert not [x for x in msg.missing_keys if not x.startswith("visual_encoder.")] and not msg.unexpected_keys
                self.load_report = {"strict_non_vit": True}
            else:
                cls_key = model.__class__.__name__
                sd = state[cls_key] if isinstance(state, dict) and cls_key in state else state
                model.load_state_dict(sd, strict=True)
                self.load_report = {"strict": True}
            if shared_vit is not None:
                model.visual_encoder = shared_vit
            model.eval()
            self.model = model
            self.txt_proc = txt_processors["eval"]
            self.preprocess = targetpad_transform(1.25, 224)

        # ------------------------------------------------------------------
        # 10. CSMCIR (blip2_cir_align_prompt_csmcir, EVA-ViT-g)
        # ------------------------------------------------------------------
        elif k == "csmcir":
            from data_utils_csmcir import targetpad_transform
            from lavis.models import load_model_and_preprocess

            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=True, device=self.device
            )
            state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict_non_vit" in state:
                msg = model.load_state_dict(state["state_dict_non_vit"], strict=False)
                assert not [x for x in msg.missing_keys if not x.startswith("visual_encoder.")]
                self.load_report = {"strict_non_vit": True}
            else:
                cls_key = model.__class__.__name__
                sd = state[cls_key] if isinstance(state, dict) and cls_key in state else next(v for kk, v in state.items() if kk != "epoch" and isinstance(v, dict))
                res = model.load_state_dict(sd, strict=False)
                assert not res.missing_keys
                self.load_report = {"strict": False, "unexpected": list(res.unexpected_keys)}
            if shared_vit is not None:
                model.visual_encoder = shared_vit
            model.eval()
            self.model = model
            self.txt_proc = txt_processors["eval"]
            self.preprocess = targetpad_transform(1.25, 224)

        # ------------------------------------------------------------------
        # 11. iLearn BLIP-2 family (HINT, HABIT, INTENT, ConeSep, AirKnow)
        # ------------------------------------------------------------------
        elif k in ("hint", "habit", "intent", "conesep", "airknow"):
            from workbench.benchmarks.trainers.train_ilearn import MODEL_SPECS as ILEARN_SPECS, _patch_bert_root_path

            # Ensure model's own third_party directory is at front of sys.path
            tp_dir = str(REPO_ROOT / "workbench/third_party" / ILEARN_SPECS[k]["dir"])
            if tp_dir not in sys.path:
                sys.path.insert(0, tp_dir)
            _patch_bert_root_path()
            from data_utils import targetpad_transform
            from lavis.models import load_model_and_preprocess
            from lavis.processors.blip_processors import BlipCaptionProcessor

            raw = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(raw, nn.Module):
                model = raw.to(self.device).eval()
                txt_proc = BlipCaptionProcessor()
                self.load_report = {"whole_module": type(model).__name__}
            else:
                model, _, txt_processors = load_model_and_preprocess(
                    name=ILEARN_SPECS[k]["lavis_name"], model_type="pretrain", is_eval=True, device=self.device
                )
                if k == "conesep":
                    model.prompt_load()
                elif k == "airknow":
                    model.alpha = 0.7
                sd = raw["state_dict_non_vit"] if "state_dict_non_vit" in raw else raw
                msg = model.load_state_dict(sd, strict=False)
                non_vit_miss = [x for x in msg.missing_keys if not x.startswith("visual_encoder.")]
                assert not non_vit_miss, f"{k} missing non-vit keys: {non_vit_miss}"
                txt_proc = txt_processors["eval"]
                self.load_report = {"strict_non_vit": True, "unexpected": list(msg.unexpected_keys)}
            if shared_vit is not None and not isinstance(raw, nn.Module):
                model.visual_encoder = shared_vit
            model.eval()
            self.model = model
            self.txt_proc = txt_proc
            self.preprocess = targetpad_transform(1.25, 224)

        # ------------------------------------------------------------------
        # 12. TME (blip2_cir_image_diff_features, EVA-ViT-g)
        # ------------------------------------------------------------------
        elif k == "tme":
            from workbench.benchmarks.trainers.train_tme import TME_DIR
            from lavis.models import load_model_and_preprocess

            cwd_before = os.getcwd()
            try:
                os.chdir(str(TME_DIR))
                model, _, txt_processors = load_model_and_preprocess(
                    name="blip2_cir_image_diff_features", model_type="pretrain", is_eval=False, device=self.device
                )
            finally:
                os.chdir(cwd_before)
            state = torch.load(ckpts[0], map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "state_dict_non_vit" in state:
                msg = model.load_state_dict(state["state_dict_non_vit"], strict=False)
                assert not [x for x in msg.missing_keys if not x.startswith("visual_encoder.")] and not msg.unexpected_keys
                self.load_report = {"strict_non_vit": True}
            else:
                sd = state.get("model", state) if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
                res = model.load_state_dict(sd, strict=False)
                assert not res.missing_keys and not res.unexpected_keys
                self.load_report = {"strict": True}
            model.eval()
            import utility as tme_utility
            self.model = model
            self.txt_proc = txt_processors["eval"]
            self.preprocess = tme_utility.targetpad_transform(target_ratio=1.25, dim=224)
        else:
            raise ValueError(f"Unsupported model key: {k}")

    # ------------------------------------------------------------------
    # Resolution interpolation context manager (E11)
    # ------------------------------------------------------------------

    @contextmanager
    def high_res_context(self, target_res: int = 336):
        """Temporarily interpolate ViT pos_embed and preprocessing for E11 high-res testing; always restores!"""
        if not self.spec.supports_res_interp or target_res == self.spec.native_resolution:
            yield False
            return

        orig_pp = self.preprocess
        saved_params: List[Tuple[nn.Module, str, nn.Parameter]] = []
        try:
            k = self.model_key
            if self.is_blip2:
                ve = self.model.visual_encoder
                if hasattr(ve, "patch_embed") and hasattr(ve, "pos_embed"):
                    patch_sz = int(ve.patch_embed.proj.kernel_size[0])
                    new_g = target_res // patch_sz
                    saved_params.append((ve, "pos_embed", ve.pos_embed))
                    ve.pos_embed = nn.Parameter(_interpolate_pos_embed_1d(ve.pos_embed.data, new_g), requires_grad=False)
                    if hasattr(ve.patch_embed, "img_size"):
                        saved_params.append((ve.patch_embed, "img_size", ve.patch_embed.img_size))
                        ve.patch_embed.img_size = (target_res, target_res)
                elif hasattr(ve, "conv1") and hasattr(ve, "positional_embedding"):
                    patch_sz = int(ve.conv1.kernel_size[0])
                    new_g = target_res // patch_sz
                    saved_params.append((ve, "positional_embedding", ve.positional_embedding))
                    ve.positional_embedding = nn.Parameter(_interpolate_pos_embed_1d(ve.positional_embedding.data, new_g), requires_grad=False)
                else:
                    yield False
                    return
                # Build matching targetpad transform at target_res
                if k == "tme":
                    import utility as tme_utility
                    self.preprocess = tme_utility.targetpad_transform(target_ratio=1.25, dim=target_res)
                elif k == "qure":
                    from data.utils import targetpad_transform
                    self.preprocess = targetpad_transform(1.25, target_res)
                elif k == "csmcir":
                    from data_utils_csmcir import targetpad_transform
                    self.preprocess = targetpad_transform(1.25, target_res)
                else:
                    from data_utils import targetpad_transform
                    self.preprocess = targetpad_transform(1.25, target_res)
            elif k in ("encoder", "limn"):
                vis = self.model.backbone.clip.visual
                patch_sz = int(vis.conv1.kernel_size[0])
                new_g = target_res // patch_sz
                saved_params.append((vis, "positional_embedding", vis.positional_embedding))
                vis.positional_embedding = nn.Parameter(_interpolate_pos_embed_1d(vis.positional_embedding.data, new_g), requires_grad=False)
                saved_params.append((vis, "image_size", vis.image_size))
                vis.image_size = (target_res, target_res)
                self.preprocess = self.extra["open_clip"].image_transform(
                    (target_res, target_res), is_train=False, mean=vis.image_mean, std=vis.image_std
                )
            elif k == "tgcir":
                vis = self.model.backbone.clip.visual
                patch_sz = int(vis.conv1.kernel_size[0])
                new_g = target_res // patch_sz
                saved_params.append((vis, "positional_embedding", vis.positional_embedding))
                vis.positional_embedding = nn.Parameter(_interpolate_pos_embed_1d(vis.positional_embedding.data, new_g), requires_grad=False)
                self.preprocess = transforms.Compose([
                    transforms.Resize(target_res, interpolation=PIL.Image.BICUBIC),
                    transforms.CenterCrop(target_res),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
            elif k == "clvcnet":
                self.preprocess = transforms.Compose([
                    transforms.Resize(int(round(target_res * 256 / 224))),
                    transforms.CenterCrop(target_res),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            yield True
        finally:
            self.preprocess = orig_pp
            for mod, attr, val in reversed(saved_params):
                setattr(mod, attr, val)

    # ------------------------------------------------------------------
    # Native Gallery Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_gallery_from_vit(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        """BLIP-2 fast path: compute [B, 32, 256] normalized target features directly from visual_encoder(imgs)."""
        k = self.model_key
        m = self.model
        if k == "tme":
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_img = m.encode_image(vit_embeds)
                return F.normalize(m.vision_proj(f_img), dim=-1).float()
        ie = m.ln_vision(vit_embeds).float()
        ia = torch.ones(ie.size()[:-1], dtype=torch.long, device=self.device)
        qt = m.query_tokens.expand(ie.shape[0], -1, -1)
        if k == "csmcir":
            # Match extract_target_caption_features(imgs, [""]*B, mode="mean")
            empty_toks = m.tokenizer([""] * ie.shape[0], padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
            q_atts = torch.ones(qt.size()[:-1], dtype=torch.long, device=self.device)
            am = torch.cat([q_atts, empty_toks.attention_mask], dim=1)
            out = m.Qformer.bert(empty_toks.input_ids, query_embeds=qt, attention_mask=am, encoder_hidden_states=ie, encoder_attention_mask=ia, return_dict=True)
            return F.normalize(m.vision_proj(out.last_hidden_state[:, : qt.size(1), :]), dim=-1).float()
        out = m.Qformer.bert(query_embeds=qt, encoder_hidden_states=ie, encoder_attention_mask=ia, return_dict=True)
        return F.normalize(m.vision_proj(out.last_hidden_state), dim=-1).float()

    @torch.no_grad()
    def encode_gallery(self, items: Sequence[Union[str, Path, PIL.Image.Image]], bs: int = 128, num_workers: int = 8) -> Any:
        """Encode gallery images into exact native target representations on CPU float32."""
        k = self.model_key
        ds = _ImgListDataset(items, self.preprocess)
        dl = _make_loader(ds, bs=min(bs, 128) if self.is_blip2 else bs, num_workers=num_workers)

        if k == "dcnet":
            mods = self.extra["modalities"]
            acc: Dict[str, List[torch.Tensor]] = {m_name: [] for m_name in mods}
            for imgs in dl:
                imgs = imgs.to(self.device, non_blocking=True)
                exps = self.model.get_image_emb({"im_feat": imgs}, istgt=True)
                for m_name in mods:
                    acc[m_name].append(exps[m_name].float().cpu())
            return {m_name: torch.cat(acc[m_name], dim=0) for m_name in mods}

        feats: List[torch.Tensor] = []
        for imgs in dl:
            imgs = imgs.to(self.device, non_blocking=True)
            if k == "clip4cir":
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    f = self.model.encode_image(imgs)
                f = F.normalize(f.float(), dim=-1)
            elif k in ("encoder", "tgcir"):
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    f = self.model.extract_retrieval_target(imgs)
                f = F.normalize(f.float(), dim=-1)
            elif k == "limn":
                dt = torch.full((imgs.size(0),), self.extra["dt_id"], dtype=torch.long, device=self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    f = self.model.compose_target(imgs, dt)[0]
                f = F.normalize(f.float(), dim=-1)
            elif k == "clvcnet":
                loc_f = F.normalize(self.model.extract_img_feature(imgs)[1].float(), dim=-1)
                glb_f = F.normalize(self.extra["global_model"].extract_img_feature(imgs)[1].float(), dim=-1)
                f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
            elif k == "sprc":
                f, _ = self.model.extract_target_features(imgs, mode="mean")
                f = f.float()
            elif k == "csmcir":
                f, _ = self.model.extract_target_caption_features(imgs, [""] * imgs.size(0), mode="mean")
                f = f.float()
            elif k in ("hint", "habit", "intent", "conesep", "airknow"):
                f = self.model.extract_retrieval_target(imgs)
                if f.dim() == 3 and f.shape[1] == 256 and f.shape[2] == 32:
                    f = f.permute(0, 2, 1)
                f = f.float()
            elif k in ("qure", "mtst_ptha", "tme"):
                if k == "tme":
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        ve = self.model.vit_encode(imgs)
                        f = F.normalize(self.model.vision_proj(self.model.encode_image(ve)), dim=-1).float()
                else:
                    with self.model.maybe_autocast():
                        ve = self.model.visual_encoder(imgs)
                    f = self.encode_gallery_from_vit(ve)
            else:
                raise ValueError(k)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    # ------------------------------------------------------------------
    # Native Query Encoding (Multimodal + CIRCUS Unimodal + TRUE Native Unimodal Towers!)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_blip2_query_batch(self, imgs: torch.Tensor, raw_texts: Sequence[str], mode: str = "multimodal") -> torch.Tensor:
        k = self.model_key
        m = self.model
        caps = [self.txt_proc(c) for c in raw_texts]

        # Path B: TRUE native BLIP-2 ITC unimodal towers (no OOD black-image distortion!)
        if mode == "native_img_tower":
            # Encode ref image through exact gallery encoder -> [B, 32, 256]
            if k == "tme":
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    return F.normalize(m.vision_proj(m.encode_image(m.vit_encode(imgs))), dim=-1).float()
            with m.maybe_autocast():
                ve = m.visual_encoder(imgs)
            return self.encode_gallery_from_vit(ve)

        if mode == "native_txt_tower":
            # Exact BLIP-2 ITC text-only tower: Q-Former BERT with NO visual cross-attention!
            text_tokens = m.tokenizer(
                caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt"
            ).to(self.device)
            text_out = m.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            return F.normalize(m.text_proj(text_out.last_hidden_state[:, 0, :]), dim=-1).float()

        # Standard multimodal / CIRCUS empty-text / CIRCUS black-image through native compose path
        if mode == "img_only_empty_text":
            caps = [self.txt_proc("")] * len(caps)
        if k in ("hint", "habit", "intent", "conesep", "airknow"):
            f = m.extract_retrieval_compose(imgs, caps)
            return f.reshape(imgs.size(0), -1).float()
        if k == "tme":
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_r = m.encode_image(m.vit_encode(imgs))
                tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
                return m.encode_fusion(f_r, tt).float()

        with m.maybe_autocast():
            ref_embeds = m.ln_vision(m.visual_encoder(imgs))
        ref_embeds = ref_embeds.float()
        img_atts = torch.ones(ref_embeds.size()[:-1], dtype=torch.long, device=self.device)
        q_tok = m.query_tokens.expand(ref_embeds.shape[0], -1, -1)
        q_atts = torch.ones(q_tok.size()[:-1], dtype=torch.long, device=self.device)
        tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
        am = torch.cat([q_atts, tt.attention_mask], dim=1)

        if k == "qure":
            q_out = m.Qformer.bert(tt.input_ids, query_embeds=q_tok, attention_mask=am, encoder_hidden_states=ref_embeds, encoder_attention_mask=img_atts, return_dict=True)
            return F.normalize(m.text_proj(q_out.last_hidden_state[:, : q_tok.size(1), :]).mean(dim=1), dim=-1).float()
        if k == "csmcir":
            fo = m.Qformer.bert(tt.input_ids, query_embeds=q_tok, attention_mask=am, encoder_hidden_states=ref_embeds, encoder_attention_mask=img_atts, return_dict=True)
            return F.normalize(m.text_proj(fo.last_hidden_state[:, 32, :]), dim=-1).float()
        if k in ("sprc", "mtst_ptha"):
            fo = m.Qformer.bert(tt.input_ids, query_embeds=q_tok, attention_mask=am, encoder_hidden_states=ref_embeds, encoder_attention_mask=img_atts, return_dict=True)
            to = m.Qformer.bert(tt.input_ids, query_embeds=fo.last_hidden_state[:, : q_tok.size(1), :], attention_mask=am, return_dict=True)
            if k == "sprc":
                return F.normalize(m.text_proj(to.last_hidden_state[:, 32, :]), dim=-1).float()
            return F.normalize(m.text_proj(to.last_hidden_state[:, :33, :].mean(dim=1)), dim=-1).float()
        raise ValueError(k)

    @torch.no_grad()
    def encode_queries(
        self,
        imgs: Sequence[Union[str, Path, PIL.Image.Image]],
        texts: Sequence[str],
        mode: str = "multimodal",
        bs: int = 64,
        num_workers: int = 8,
    ) -> Any:
        """Encode queries in specified mode:
        - multimodal
        - img_only_empty_text (CIRCUS)
        - txt_only_black_img (CIRCUS)
        - native_img_tower (true image-only encoder)
        - native_txt_tower (true text-only encoder)
        """
        k = self.model_key
        use_black = mode == "txt_only_black_img"
        ds = _PairListDataset(imgs, texts, self.preprocess, black_image=use_black)
        dl = _make_loader(ds, bs=min(bs, 64) if self.is_blip2 else bs, num_workers=num_workers)

        if k == "dcnet":
            mods = self.extra["modalities"]
            src_acc, txt_acc, trg_acc = {m: [] for m in mods}, {m: [] for m in mods}, {m: [] for m in mods}
            for b_imgs, b_txts in dl:
                b_imgs = b_imgs.to(self.device, non_blocking=True)
                txts_use = [""] * len(b_txts) if mode == "img_only_empty_text" else list(b_txts)
                tt, tl = self.extra["encode_glove"](txts_use, self.extra["vocab"])
                src_e = self.model.get_image_emb({"im_feat": b_imgs}, istgt=False)
                txt_e = self.model.get_text_emb(tt.to(self.device), tl.tolist())
                if mode == "native_img_tower":
                    trg_e = src_e
                elif mode == "native_txt_tower":
                    trg_e = txt_e
                else:
                    trg_e = self.model.get_composition(src_e, txt_e)
                for m_name in mods:
                    src_acc[m_name].append(src_e[m_name].float().cpu())
                    txt_acc[m_name].append(txt_e[m_name].float().cpu())
                    trg_acc[m_name].append(trg_e[m_name].float().cpu())
            return {
                "src": {m: torch.cat(src_acc[m], dim=0) for m in mods},
                "txt": {m: torch.cat(txt_acc[m], dim=0) for m in mods},
                "trg": {m: torch.cat(trg_acc[m], dim=0) for m in mods},
                "mode": mode,
            }

        out_list: List[torch.Tensor] = []
        for b_imgs, b_txts in dl:
            b_imgs = b_imgs.to(self.device, non_blocking=True)
            b_txts = list(b_txts)
            if self.is_blip2:
                f = self._encode_blip2_query_batch(b_imgs, b_txts, mode=mode)
            elif k == "clip4cir":
                clip_mod = self.extra["clip_mod"]
                txts_use = [""] * len(b_txts) if mode == "img_only_empty_text" else b_txts
                toks = clip_mod.tokenize(txts_use, context_length=77, truncate=True).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    img_f = self.model.encode_image(b_imgs).float()
                    txt_f = self.model.encode_text(toks).float()
                if mode == "native_img_tower":
                    f = F.normalize(img_f, dim=-1)
                elif mode == "native_txt_tower":
                    f = F.normalize(txt_f, dim=-1)
                else:
                    f = F.normalize(self.extra["combiner"].combine_features(img_f, txt_f).float(), dim=-1)
            elif k == "encoder":
                txts_use = [""] * len(b_txts) if mode == "img_only_empty_text" else b_txts
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    if mode == "native_img_tower":
                        f = self.model.extract_retrieval_target(b_imgs)
                    elif mode == "native_txt_tower":
                        toks = self.extra["open_clip"].tokenize(txts_use).to(self.device)
                        f = self.model.backbone.clip.encode_text(toks)
                    else:
                        f = self.model.extract_retrieval_compose(b_imgs, txts_use)
                f = F.normalize(f.float(), dim=-1)
            elif k == "tgcir":
                txts_use = [""] * len(b_txts) if mode == "img_only_empty_text" else b_txts
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    if mode == "native_img_tower":
                        f = self.model.extract_retrieval_target(b_imgs)
                    elif mode == "native_txt_tower":
                        toks = self.extra["clip_mod"].tokenize(txts_use, context_length=77, truncate=True).to(self.device)
                        f = self.model.backbone.clip.encode_text(toks)
                    else:
                        f = self.model.extract_retrieval_compose(b_imgs, txts_use)
                f = F.normalize(f.float(), dim=-1)
            elif k == "limn":
                dt = torch.full((b_imgs.size(0),), self.extra["dt_id"], dtype=torch.long, device=self.device)
                txts_use = [""] * len(b_txts) if mode == "img_only_empty_text" else b_txts
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    if mode == "native_img_tower":
                        f = self.model.compose_target(b_imgs, dt)[0]
                    elif mode == "native_txt_tower":
                        toks = self.extra["open_clip"].tokenize(txts_use).to(self.device)
                        f = self.model.backbone.clip.encode_text(toks)
                    else:
                        f = self.model.compose_query(b_imgs, txts_use, dt)[0]
                f = F.normalize(f.float(), dim=-1)
            elif k == "clvcnet":
                vh = self.extra["vocab_helper"]
                def _safe(t: str) -> str:
                    return t.strip() if vh.tokenize_text(t) else "unknown"
                txts_use = ["unknown"] * len(b_txts) if mode == "img_only_empty_text" else [_safe(t) for t in b_txts]
                if mode == "native_img_tower":
                    loc_f = F.normalize(self.model.extract_img_feature(b_imgs)[1].float(), dim=-1)
                    glb_f = F.normalize(self.extra["global_model"].extract_img_feature(b_imgs)[1].float(), dim=-1)
                    f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
                elif mode == "native_txt_tower":
                    # CLVC-Net text tower has no shared image-text cosine head; fall back explicitly to black-image compose
                    loc_f = F.normalize(self.model.compose_img_text(torch.zeros_like(b_imgs), txts_use)[1].float(), dim=-1)
                    glb_f = F.normalize(self.extra["global_model"].compose_img_text(torch.zeros_like(b_imgs), txts_use)[0].float(), dim=-1)
                    f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
                else:
                    loc_f = F.normalize(self.model.compose_img_text(b_imgs, txts_use)[1].float(), dim=-1)
                    glb_f = F.normalize(self.extra["global_model"].compose_img_text(b_imgs, txts_use)[0].float(), dim=-1)
                    f = F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1)
            else:
                raise ValueError(k)
            out_list.append(f.cpu())
        return torch.cat(out_list, dim=0)

    # ------------------------------------------------------------------
    # Exact Native Scoring Matrix
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_matrix(
        self,
        q_feats: Any,
        g_feats: Any,
        q_chunk: int = 128,
        c_chunk: int = 4096,
        return_active_tok_idx: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Compute exact native [N_q, N_g] similarity matrix (float32)."""
        k = self.model_key
        if k == "dcnet":
            mods = self.extra["modalities"]
            n_q = q_feats["trg"][mods[0]].shape[0]
            n_c = g_feats[mods[0]].shape[0]
            mode = q_feats.get("mode", "multimodal")
            comp_full = torch.empty((n_q, n_c), dtype=torch.float32)
            corr_full = torch.empty((n_q, n_c), dtype=torch.float32)
            q_bs = 12
            dc_c_chunk = min(max(c_chunk, 4096), 4096, n_c)
            for cs in range(0, n_c, dc_c_chunk):
                ce = min(cs + dc_c_chunk, n_c)
                gal_blk = {m: g_feats[m][cs:ce].to(self.device, non_blocking=True) for m in mods}
                for qs in range(0, n_q, q_bs):
                    qe = min(qs + q_bs, n_q)
                    trg_blk = {m: q_feats["trg"][m][qs:qe].to(self.device, non_blocking=True) for m in mods}
                    comp_full[qs:qe, cs:ce] = self.model.get_score_matrix(gal_blk, trg_blk, mods).cpu()
                    if mode not in ("native_img_tower", "native_txt_tower"):
                        src_blk = {m: q_feats["src"][m][qs:qe].to(self.device, non_blocking=True) for m in mods}
                        txt_blk = {m: q_feats["txt"][m][qs:qe].to(self.device, non_blocking=True) for m in mods}
                        mod_txt = self.model.get_correction(src_blk, gal_blk)
                        corr_full[qs:qe, cs:ce] = self.model.get_score_matrix(mod_txt, txt_blk, mods).cpu()
                        del mod_txt
            if mode in ("native_img_tower", "native_txt_tower"):
                sims = comp_full.numpy()
            else:
                sims = (F.log_softmax(comp_full, dim=-1) + F.log_softmax(corr_full, dim=-1)).numpy()
            return (sims, None) if return_active_tok_idx else sims

        n_q = q_feats.shape[0]
        n_c = g_feats.shape[0]
        sims = np.empty((n_q, n_c), dtype=np.float32)
        act_idx = np.empty((n_q, n_c), dtype=np.int16) if (return_active_tok_idx and g_feats.dim() == 3) else None

        if g_feats.dim() == 3:
            # Multi-vector [N_c, 32, 256]
            if q_feats.dim() == 3:
                # Native image-to-image MaxSim/Chamfer (native_img_tower for BLIP-2): mean_i max_j <q_i, g_j>
                for qs in range(0, n_q, min(q_chunk, 32)):
                    qe = min(qs + min(q_chunk, 32), n_q)
                    qf = F.normalize(q_feats[qs:qe].to(self.device, non_blocking=True), dim=-1)  # [B, 32, 256]
                    for cs in range(0, n_c, min(c_chunk, 1024)):
                        ce = min(cs + min(c_chunk, 1024), n_c)
                        tf = g_feats[cs:ce].to(self.device, non_blocking=True)  # [Nc, 32, 256]
                        sim_blk = torch.einsum("bmd,nkd->bnmk", qf, tf).max(dim=-1).values.mean(dim=-1)
                        sims[qs:qe, cs:ce] = sim_blk.cpu().numpy()
                return (sims, None) if return_active_tok_idx else sims

            for qs in range(0, n_q, q_chunk):
                qe = min(qs + q_chunk, n_q)
                qf = q_feats[qs:qe].to(self.device, non_blocking=True)  # [B, 256]
                for cs in range(0, n_c, c_chunk):
                    ce = min(cs + c_chunk, n_c)
                    tf = g_feats[cs:ce].to(self.device, non_blocking=True)  # [Nc, 32, 256]
                    all_tok = torch.einsum("bd,nkd->bnk", qf, tf)
                    mx, argm = all_tok.max(dim=-1)
                    sims[qs:qe, cs:ce] = mx.cpu().numpy()
                    if act_idx is not None:
                        act_idx[qs:qe, cs:ce] = argm.to(torch.int16).cpu().numpy()
        else:
            # Single-vector cosine [N_q, D] x [N_c, D]
            qn = F.normalize(q_feats.float(), dim=-1)
            gn = F.normalize(g_feats.float(), dim=-1)
            for qs in range(0, n_q, q_chunk):
                qe = min(qs + q_chunk, n_q)
                qf = qn[qs:qe].to(self.device, non_blocking=True)
                for cs in range(0, n_c, c_chunk * 4):
                    ce = min(cs + c_chunk * 4, n_c)
                    gf = gn[cs:ce].to(self.device, non_blocking=True)
                    sims[qs:qe, cs:ce] = (qf @ gf.T).cpu().numpy()
        return (sims, act_idx) if return_active_tok_idx else sims

    # ------------------------------------------------------------------
    # Layer-wise Representation Extraction (E04: BOTH Gallery & Query branches!)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_layer_reps_gallery(self, items: Sequence[Union[str, Path, PIL.Image.Image]], bs: int = 64, num_workers: int = 8) -> Dict[str, np.ndarray]:
        k = self.model_key
        ds = _ImgListDataset(items, self.preprocess)
        dl = _make_loader(ds, bs=min(bs, 48) if self.is_blip2 else bs, num_workers=num_workers)
        buckets: Dict[str, List[np.ndarray]] = {}

        def _add(name: str, t: torch.Tensor) -> None:
            buckets.setdefault(name, []).append(t.float().cpu().numpy())

        for imgs in dl:
            imgs = imgs.to(self.device, non_blocking=True)
            if self.is_blip2:
                m = self.model
                if k == "tme":
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        ve = m.vit_encode(imgs)
                        qf_out = m.encode_image(ve)
                        vp = F.normalize(m.vision_proj(qf_out), dim=-1)
                    _add("gal_vit_cls", ve[:, 0, :])
                    _add("gal_vit_patch_mean", ve[:, 1:, :].mean(dim=1))
                    _add("gal_qformer_preproj", qf_out.mean(dim=1))
                    _add("gal_vision_proj_mean", vp.mean(dim=1))
                else:
                    with m.maybe_autocast():
                        ve = m.ln_vision(m.visual_encoder(imgs))
                    ve = ve.float()
                    ia = torch.ones(ve.size()[:-1], dtype=torch.long, device=self.device)
                    qt = m.query_tokens.expand(ve.shape[0], -1, -1)
                    qo = m.Qformer.bert(query_embeds=qt, encoder_hidden_states=ve, encoder_attention_mask=ia, return_dict=True).last_hidden_state
                    vp = F.normalize(m.vision_proj(qo), dim=-1)
                    _add("gal_vit_cls", ve[:, 0, :])
                    _add("gal_vit_patch_mean", ve[:, 1:, :].mean(dim=1))
                    _add("gal_qformer_preproj", qo.mean(dim=1))
                    _add("gal_vision_proj_mean", vp.mean(dim=1))
            elif k == "clip4cir":
                # Capture ResNet layer3 pooled + final visual output
                cap_mid = {}
                h = self.model.visual.layer3.register_forward_hook(lambda mod, inp, out: cap_mid.__setitem__("mid", out.mean(dim=(2, 3))))
                try:
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        f = F.normalize(self.model.encode_image(imgs).float(), dim=-1)
                finally:
                    h.remove()
                _add("gal_vis_mid", cap_mid["mid"])
                _add("gal_vis_final", f)
            elif k in ("encoder", "limn"):
                vis = self.model.backbone.clip.visual
                blocks = vis.transformer.resblocks
                mid_idx = len(blocks) // 2
                cap_mid = {}
                def _hk(mod, inp, out):
                    # open_clip ViT: [N, L, D] or [L, N, D]
                    t = out.float()
                    cap_mid["mid"] = t.mean(dim=1) if t.shape[0] == imgs.shape[0] else t.mean(dim=0)
                h = blocks[mid_idx].register_forward_hook(_hk)
                try:
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        if k == "encoder":
                            f = F.normalize(self.model.extract_retrieval_target(imgs).float(), dim=-1)
                        else:
                            dt = torch.full((imgs.size(0),), self.extra["dt_id"], dtype=torch.long, device=self.device)
                            f = F.normalize(self.model.compose_target(imgs, dt)[0].float(), dim=-1)
                finally:
                    h.remove()
                _add("gal_vis_mid", cap_mid["mid"])
                _add("gal_vis_final", f)
            elif k == "tgcir":
                vis = self.model.backbone.clip.visual
                blocks = vis.transformer.resblocks
                mid_idx = len(blocks) // 2
                cap_mid = {}
                def _hk(mod, inp, out):
                    t = out.float()
                    cap_mid["mid"] = t.mean(dim=0) if t.shape[1] == imgs.shape[0] else t.mean(dim=1)
                h = blocks[mid_idx].register_forward_hook(_hk)
                try:
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        f = F.normalize(self.model.extract_retrieval_target(imgs).float(), dim=-1)
                finally:
                    h.remove()
                _add("gal_vis_mid", cap_mid["mid"])
                _add("gal_vis_final", f)
            elif k == "clvcnet":
                loc_f = F.normalize(self.model.extract_img_feature(imgs)[1].float(), dim=-1)
                glb_f = F.normalize(self.extra["global_model"].extract_img_feature(imgs)[1].float(), dim=-1)
                _add("gal_local_512", loc_f)
                _add("gal_global_512", glb_f)
                _add("gal_final_1024", F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1))
            elif k == "dcnet":
                exps = self.model.get_image_emb({"im_feat": imgs}, istgt=True)
                _add("gal_inter1", exps["inter-1"])
                _add("gal_im_feat", exps["im_feat"])

        return {name: np.concatenate(arrs, axis=0) for name, arrs in buckets.items()}

    @torch.no_grad()
    def extract_layer_reps_query(
        self,
        imgs: Sequence[Union[str, Path, PIL.Image.Image]],
        texts: Sequence[str],
        bs: int = 48,
        num_workers: int = 8,
    ) -> Dict[str, np.ndarray]:
        k = self.model_key
        ds = _PairListDataset(imgs, texts, self.preprocess, black_image=False)
        dl = _make_loader(ds, bs=min(bs, 48) if self.is_blip2 else bs, num_workers=num_workers)
        buckets: Dict[str, List[np.ndarray]] = {}

        def _add(name: str, t: torch.Tensor) -> None:
            buckets.setdefault(name, []).append(t.float().cpu().numpy())

        for b_imgs, b_txts in dl:
            b_imgs = b_imgs.to(self.device, non_blocking=True)
            b_txts = list(b_txts)
            if self.is_blip2:
                m = self.model
                caps = [self.txt_proc(c) for c in b_txts]
                # 1. Vision patch mean
                if k == "tme":
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        ve = m.vit_encode(b_imgs).float()
                else:
                    with m.maybe_autocast():
                        ve = m.ln_vision(m.visual_encoder(b_imgs)).float()
                _add("qry_vit_patch_mean", ve[:, 1:, :].mean(dim=1))

                # 2. Text-only Q-Former BERT CLS (no visual cross-attention)
                tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
                txt_only = m.Qformer.bert(tt.input_ids, attention_mask=tt.attention_mask, return_dict=True).last_hidden_state[:, 0, :]
                _add("qry_text_only_cls", txt_only)

                # 3. Q-Former fusion mid-layer (L6) & output (L12) before projection!
                ia = torch.ones(ve.size()[:-1], dtype=torch.long, device=self.device)
                qt = m.query_tokens.expand(ve.shape[0], -1, -1)
                qa = torch.ones(qt.size()[:-1], dtype=torch.long, device=self.device)
                am = torch.cat([qa, tt.attention_mask], dim=1)
                fo = m.Qformer.bert(
                    tt.input_ids,
                    query_embeds=qt,
                    attention_mask=am,
                    encoder_hidden_states=ve,
                    encoder_attention_mask=ia,
                    output_hidden_states=True,
                    return_dict=True,
                )
                mid_l = len(fo.hidden_states) // 2
                # Extract text-CLS position (idx 32) AND query-token mean ([:32])
                _add("qry_qformer_mid_L6", fo.hidden_states[mid_l][:, 32, :])
                _add("qry_qformer_out_L12_txt", fo.last_hidden_state[:, 32, :])
                _add("qry_qformer_out_L12_qmean", fo.last_hidden_state[:, :32, :].mean(dim=1))

                # 4. Exact final projected normalized query vector
                qf = self._encode_blip2_query_batch(b_imgs, b_txts, mode="multimodal")
                _add("qry_final_proj", qf)

            elif k == "clip4cir":
                toks = self.extra["clip_mod"].tokenize(b_txts, context_length=77, truncate=True).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    iv = F.normalize(self.model.encode_image(b_imgs).float(), dim=-1)
                    tv = F.normalize(self.model.encode_text(toks).float(), dim=-1)
                comb = F.normalize(self.extra["combiner"].combine_features(iv, tv).float(), dim=-1)
                _add("qry_vis_final", iv)
                _add("qry_txt_final", tv)
                _add("qry_fused_final", comb)

            elif k in ("encoder", "tgcir", "limn"):
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    if k == "encoder":
                        iv = F.normalize(self.model.extract_retrieval_target(b_imgs).float(), dim=-1)
                        tv = F.normalize(self.model.backbone.clip.encode_text(self.extra["open_clip"].tokenize(b_txts).to(self.device)).float(), dim=-1)
                        fv = F.normalize(self.model.extract_retrieval_compose(b_imgs, b_txts).float(), dim=-1)
                    elif k == "tgcir":
                        iv = F.normalize(self.model.extract_retrieval_target(b_imgs).float(), dim=-1)
                        tv = F.normalize(self.model.backbone.clip.encode_text(self.extra["clip_mod"].tokenize(b_txts, context_length=77, truncate=True).to(self.device)).float(), dim=-1)
                        fv = F.normalize(self.model.extract_retrieval_compose(b_imgs, b_txts).float(), dim=-1)
                    else:
                        dt = torch.full((b_imgs.size(0),), self.extra["dt_id"], dtype=torch.long, device=self.device)
                        iv = F.normalize(self.model.compose_target(b_imgs, dt)[0].float(), dim=-1)
                        tv = F.normalize(self.model.backbone.clip.encode_text(self.extra["open_clip"].tokenize(b_txts).to(self.device)).float(), dim=-1)
                        fv = F.normalize(self.model.compose_query(b_imgs, b_txts, dt)[0].float(), dim=-1)
                _add("qry_vis_final", iv)
                _add("qry_txt_final", tv)
                _add("qry_fused_final", fv)

            elif k == "clvcnet":
                vh = self.extra["vocab_helper"]
                mods = [t.strip() if vh.tokenize_text(t) else "unknown" for t in b_txts]
                iv = F.normalize(self.model.extract_img_feature(b_imgs)[1].float(), dim=-1)
                loc_f = F.normalize(self.model.compose_img_text(b_imgs, mods)[1].float(), dim=-1)
                glb_f = F.normalize(self.extra["global_model"].compose_img_text(b_imgs, mods)[0].float(), dim=-1)
                _add("qry_vis_512", iv)
                _add("qry_local_512", loc_f)
                _add("qry_fused_1024", F.normalize(torch.cat([loc_f, glb_f], dim=1), dim=-1))

            elif k == "dcnet":
                tt, tl = self.extra["encode_glove"](b_txts, self.extra["vocab"])
                src_e = self.model.get_image_emb({"im_feat": b_imgs}, istgt=False)
                txt_e = self.model.get_text_emb(tt.to(self.device), tl.tolist())
                trg_e = self.model.get_composition(src_e, txt_e)
                _add("qry_vis_im_feat", src_e["im_feat"])
                _add("qry_txt_im_feat", txt_e["im_feat"])
                _add("qry_fused_im_feat", trg_e["im_feat"])

        return {name: np.concatenate(arrs, axis=0) for name, arrs in buckets.items()}

    # ------------------------------------------------------------------
    # Causal Activation / Token / Spatial-ROI / Block Interventions w/ Shams (E04, E05, F28, F29)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_causal_interventions(
        self,
        ref_paths: Sequence[str],
        texts: Sequence[str],
        pos_paths: Sequence[str],
        neg_paths: Sequence[str],
        roi_boxes: Optional[Sequence[List[float]]] = None,
        sham_boxes: Optional[Sequence[List[float]]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Run controlled causal interventions measuring exact retrieval margin change (s(q,pos) - s(q,neg))."""
        n = len(texts)
        if n == 0:
            return {"n": 0}
        rng = np.random.default_rng(seed)

        # Pre-encode fixed positive & negative gallery targets natively
        g_pos = self.encode_gallery(pos_paths, bs=48, num_workers=4)
        g_neg = self.encode_gallery(neg_paths, bs=48, num_workers=4)

        def _paired_margins(q_rep: Any) -> np.ndarray:
            # Diagonal paired score s(q_i, pos_i) - s(q_i, neg_i)
            if self.model_key == "dcnet":
                # Evaluate each pair cleanly via score_matrix against concat([pos, neg])
                mods = self.extra["modalities"]
                g_cat = {m: torch.cat([g_pos[m], g_neg[m]], dim=0) for m in mods}
                sm = self.score_matrix(q_rep, g_cat)
                return np.array([sm[i, i] - sm[i, n + i] for i in range(n)], dtype=np.float32)
            elif g_pos.dim() == 3:
                sp = torch.einsum("bd,bkd->bk", q_rep.float(), g_pos.float()).max(dim=-1).values
                sn = torch.einsum("bd,bkd->bk", q_rep.float(), g_neg.float()).max(dim=-1).values
                return (sp - sn).cpu().numpy()
            else:
                qn = F.normalize(q_rep.float(), dim=-1)
                sp = (qn * F.normalize(g_pos.float(), dim=-1)).sum(dim=-1)
                sn = (qn * F.normalize(g_neg.float(), dim=-1)).sum(dim=-1)
                return (sp - sn).cpu().numpy()

        q_base = self.encode_queries(ref_paths, texts, mode="multimodal", bs=48, num_workers=4)
        m_base = _paired_margins(q_base)
        out: Dict[str, np.ndarray] = {"baseline_margin": m_base}

        # 1. BLIP-2 fine-grained internal interventions:
        #    (a) Q-Former visual cross-attention ablation vs text ablation vs matched-norm noise sham
        #    (b) ViT spatial ROI patch-token ablation vs matched-area sham patch-token ablation!
        #    (c) Gallery Q-Former top-8 active token drop vs 8 matched-norm random sham token drop!
        if self.is_blip2:
            m = self.model
            k = self.model_key
            q_vis_zero, q_txt_zero, q_sham_noise = [], [], []
            q_roi_drop, q_roi_sham = [], []
            bs = 32
            for s in range(0, n, bs):
                e = min(s + bs, n)
                imgs = torch.stack([self.preprocess(_open_pil(p)) for p in ref_paths[s:e]]).to(self.device)
                caps = [self.txt_proc(c) for c in texts[s:e]]
                if k == "tme":
                    with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                        ve = m.vit_encode(imgs).float()
                else:
                    with m.maybe_autocast():
                        ve = m.ln_vision(m.visual_encoder(imgs)).float()
                tt = m.tokenizer(caps, padding="max_length", truncation=True, max_length=m.max_txt_len, return_tensors="pt").to(self.device)
                ia = torch.ones(ve.size()[:-1], dtype=torch.long, device=self.device)
                qt = m.query_tokens.expand(ve.shape[0], -1, -1)
                qa = torch.ones(qt.size()[:-1], dtype=torch.long, device=self.device)

                def _run_qf(ve_in: torch.Tensor, ia_in: torch.Tensor, txt_mask_in: torch.Tensor) -> torch.Tensor:
                    am = torch.cat([qa, txt_mask_in], dim=1)
                    if k == "tme":
                        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                            qo = m.Qformer.bert(query_embeds=qt, encoder_hidden_states=ve_in, encoder_attention_mask=ia_in, return_dict=True).last_hidden_state
                            tt_cp = copy.copy(tt)
                            tt_cp["attention_mask"] = txt_mask_in
                            return m.encode_fusion(qo, tt_cp).float()
                    fo = m.Qformer.bert(tt.input_ids, query_embeds=qt, attention_mask=am, encoder_hidden_states=ve_in, encoder_attention_mask=ia_in, return_dict=True)
                    if k == "qure":
                        return F.normalize(m.text_proj(fo.last_hidden_state[:, :32, :]).mean(dim=1), dim=-1).float()
                    if k in ("sprc", "mtst_ptha"):
                        to = m.Qformer.bert(tt.input_ids, query_embeds=fo.last_hidden_state[:, :32, :], attention_mask=am, return_dict=True)
                        if k == "sprc":
                            return F.normalize(m.text_proj(to.last_hidden_state[:, 32, :]), dim=-1).float()
                        return F.normalize(m.text_proj(to.last_hidden_state[:, :33, :].mean(dim=1)), dim=-1).float()
                    # iLearn & CSMCIR single-pass Q-Former CLS at pos 32
                    return F.normalize(m.text_proj(fo.last_hidden_state[:, 32, :]), dim=-1).float()

                # (a) Cross-attention visual mask zeroed (ablate ref image inside Q-Former while keeping valid text!)
                q_vis_zero.append(_run_qf(ve, torch.zeros_like(ia), tt.attention_mask).cpu())
                # (b) Text tokens zeroed after BOS (ablate text inside Q-Former while keeping exact ref!)
                tm_zero = torch.zeros_like(tt.attention_mask)
                tm_zero[:, 0] = 1
                q_txt_zero.append(_run_qf(ve, ia, tm_zero).cpu())
                # (c) Matched-norm random orthogonal perturbation sham on 10% of vision tokens
                g_gen = torch.Generator(device=self.device).manual_seed(seed + s)
                noise = torch.randn(ve.shape, generator=g_gen, device=self.device, dtype=ve.dtype)
                # Scale noise to 15% of each token's L2 norm -> mild non-targeted perturbation sham
                ve_norm = ve.norm(dim=-1, keepdim=True)
                noise_unit = noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                ve_sham = ve + 0.15 * ve_norm * noise_unit
                q_sham_noise.append(_run_qf(ve_sham, ia, tt.attention_mask).cpu())

                # (d) ViT Spatial ROI patch-token ablation vs exact matched-area sham patch-token ablation!
                if roi_boxes is not None and sham_boxes is not None:
                    n_patches = ve.shape[1] - 1
                    grid = int(round(math.sqrt(n_patches)))
                    if grid * grid == n_patches:
                        ia_roi = ia.clone()
                        ia_shm = ia.clone()
                        for bi in range(e - s):
                            rx1, ry1, rx2, ry2 = roi_boxes[s + bi]
                            sx1, sy1, sx2, sy2 = sham_boxes[s + bi]
                            roi_idxs, shm_idxs = [], []
                            for gy in range(grid):
                                cy = (gy + 0.5) / grid
                                for gx in range(grid):
                                    cx = (gx + 0.5) / grid
                                    pidx = 1 + gy * grid + gx
                                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                                        roi_idxs.append(pidx)
                                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                                        shm_idxs.append(pidx)
                            k_drop = min(len(roi_idxs), len(shm_idxs))
                            if k_drop > 0:
                                ia_roi[bi, roi_idxs[:k_drop]] = 0
                                ia_shm[bi, shm_idxs[:k_drop]] = 0
                        q_roi_drop.append(_run_qf(ve, ia_roi, tt.attention_mask).cpu())
                        q_roi_sham.append(_run_qf(ve, ia_shm, tt.attention_mask).cpu())

            out["interv_block_vis_zero_margin"] = _paired_margins(torch.cat(q_vis_zero, dim=0))
            out["interv_block_txt_zero_margin"] = _paired_margins(torch.cat(q_txt_zero, dim=0))
            out["interv_sham_noise_margin"] = _paired_margins(torch.cat(q_sham_noise, dim=0))
            if q_roi_drop:
                out["interv_vit_roi_drop_margin"] = _paired_margins(torch.cat(q_roi_drop, dim=0))
                out["interv_vit_sham_drop_margin"] = _paired_margins(torch.cat(q_roi_sham, dim=0))

            # (e) Gallery 32-token MaxSim pruning: drop top-4 active MaxSim tokens vs 4 random sham tokens!
            # Directly tests whether BLIP-2 MaxSim relies on a narrow subset of 32 Q-Former tokens (F28)!
            sp_all = torch.einsum("bd,bkd->bk", q_base.float(), g_pos.float())  # [N, 32]
            sn_all = torch.einsum("bd,bkd->bk", q_base.float(), g_neg.float())  # [N, 32]
            # Drop top-4 tokens per sample
            sp_drop = sp_all.clone()
            sn_drop = sn_all.clone()
            top4_p = sp_all.topk(4, dim=-1).indices
            top4_n = sn_all.topk(4, dim=-1).indices
            sp_drop.scatter_(1, top4_p, -1e9)
            sn_drop.scatter_(1, top4_n, -1e9)
            out["interv_qformer_top4_tok_drop_margin"] = (sp_drop.max(dim=-1).values - sn_drop.max(dim=-1).values).cpu().numpy()
            # Matched-count random 4-token sham drop
            sp_shm = sp_all.clone()
            sn_shm = sn_all.clone()
            rand4 = torch.tensor(np.array([rng.choice(32, size=4, replace=False) for _ in range(n)]), dtype=torch.long)
            sp_shm.scatter_(1, rand4, -1e9)
            sn_shm.scatter_(1, rand4, -1e9)
            out["interv_qformer_sham4_tok_drop_margin"] = (sp_shm.max(dim=-1).values - sn_shm.max(dim=-1).values).cpu().numpy()
            # Also report mean-pool over 32 tokens (to contrast MaxSim vs MeanPool scorer readout for F28!)
            out["scorer_meanpool_32_margin"] = (sp_all.mean(dim=-1) - sn_all.mean(dim=-1)).cpu().numpy()

        # 2. CLIP4Cir Combiner exact block-level causal interventions (F29!)
        elif self.model_key == "clip4cir":
            comb = self.extra["combiner"]
            clip_mod = self.extra["clip_mod"]
            q_no_mlp, q_no_convex, q_sham = [], [], []
            for s in range(0, n, 48):
                e = min(s + 48, n)
                imgs = torch.stack([self.preprocess(_open_pil(p)) for p in ref_paths[s:e]]).to(self.device)
                toks = clip_mod.tokenize(list(texts[s:e]), context_length=77, truncate=True).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    iv = self.model.encode_image(imgs).float()
                    tv = self.model.encode_text(toks).float()
                tp = comb.dropout1(F.relu(comb.text_projection_layer(tv)))
                ip = comb.dropout2(F.relu(comb.image_projection_layer(iv)))
                raw_c = torch.cat((tp, ip), dim=-1)
                hid = comb.dropout3(F.relu(comb.combiner_layer(raw_c)))
                mlp_res = comb.output_layer(hid)
                dyn = comb.dynamic_scalar(raw_c)
                conv_part = dyn * tv + (1.0 - dyn) * iv
                # (a) Ablate nonlinear MLP residual (convex branch only!)
                q_no_mlp.append(F.normalize(conv_part, dim=-1).cpu())
                # (b) Ablate convex skip branch (MLP residual only!)
                q_no_convex.append(F.normalize(mlp_res, dim=-1).cpu())
                # (c) Matched-norm Gaussian sham added to MLP residual
                g_gen = torch.Generator(device=self.device).manual_seed(seed + s)
                ns = torch.randn(mlp_res.shape, generator=g_gen, device=self.device)
                ns = ns / ns.norm(dim=-1, keepdim=True).clamp(min=1e-6) * (0.15 * mlp_res.norm(dim=-1, keepdim=True))
                q_sham.append(F.normalize(mlp_res + ns + conv_part, dim=-1).cpu())
            out["interv_combiner_mlp_zero_margin"] = _paired_margins(torch.cat(q_no_mlp, dim=0))
            out["interv_combiner_convex_zero_margin"] = _paired_margins(torch.cat(q_no_convex, dim=0))
            out["interv_sham_noise_margin"] = _paired_margins(torch.cat(q_sham, dim=0))

        # 3. Other CLIP & Classic models: native unimodal block ablations + matched-norm sham
        else:
            q_img_only = self.encode_queries(ref_paths, texts, mode="native_img_tower", bs=48, num_workers=4)
            q_txt_only = self.encode_queries(ref_paths, texts, mode="native_txt_tower", bs=48, num_workers=4)
            out["interv_block_txt_zero_margin"] = _paired_margins(q_img_only)
            out["interv_block_vis_zero_margin"] = _paired_margins(q_txt_only)
            if isinstance(q_base, torch.Tensor):
                g_gen = torch.Generator().manual_seed(seed)
                ns = torch.randn(q_base.shape, generator=g_gen)
                ns = ns / ns.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 0.15
                out["interv_sham_noise_margin"] = _paired_margins(F.normalize(q_base + ns, dim=-1))

        return {kk: vv.tolist() for kk, vv in out.items()}

    # ------------------------------------------------------------------
    # Exact Model Tokenizer Inspection (E12 / F25 Homophone Hijack)
    # ------------------------------------------------------------------

    def tokenize_inspect(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        k = self.model_key
        res = []
        if self.is_blip2:
            tok = self.model.tokenizer
            for t in texts:
                enc = tok(self.txt_proc(t), truncation=True, max_length=getattr(self.model, "max_txt_len", 32))
                ids = list(enc["input_ids"])
                toks = tok.convert_ids_to_tokens(ids)
                res.append({"token_ids": ids, "tokens": toks, "n_tokens": len(ids)})
        elif k in ("clip4cir", "tgcir"):
            clip_mod = self.extra["clip_mod"]
            for t in texts:
                ids = clip_mod.tokenize([t], context_length=77, truncate=True)[0].tolist()
                nz = [i for i in ids if i != 0]
                res.append({"token_ids": nz, "tokens": None, "n_tokens": len(nz)})
        elif k in ("encoder", "limn"):
            oc = self.extra["open_clip"]
            for t in texts:
                ids = oc.tokenize([t])[0].tolist()
                nz = [i for i in ids if i != 0]
                res.append({"token_ids": nz, "tokens": None, "n_tokens": len(nz)})
        elif k == "clvcnet":
            vh = self.extra["vocab_helper"]
            w2i = self.model.text_model.vocab.word2id
            for t in texts:
                toks = vh.tokenize_text(t)
                ids = [int(w2i.get(w, 0)) for w in toks]
                res.append({"token_ids": ids, "tokens": toks, "n_tokens": len(ids), "oov_count": sum(1 for w in toks if w not in w2i)})
        elif k == "dcnet":
            tk = self.extra["tokenize_dcnet"]
            vocab = self.extra["vocab"]
            for t in texts:
                toks = tk(t)
                hits = [w for w in toks if w in vocab]
                res.append({"token_ids": None, "tokens": toks, "n_tokens": len(toks), "in_vocab_count": len(hits), "oov_count": len(toks) - len(hits)})
        return res
