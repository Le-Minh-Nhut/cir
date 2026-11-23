#!/usr/bin/env python3
"""Single-model cross-benchmark checkpoint transfer evaluator (inference-only).

Runs inside each model's exact pinned virtualenv + PYTHONPATH, strictly reusing
the audited preprocessors and scoring functions from `workbench/benchmarks/trainers/`.
Supports `--smoke-only` for fast GPU verification of strict checkpoint loading +
finite scoring across all 24 checkpoints before running full evaluations.
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
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.benchmarks import (  # noqa: E402
    CIRCorpusDataset,
    CIRQueryEvalDataset,
    collate_corpus,
    collate_eval_queries,
    evaluate_edir,
    evaluate_pinpoint,
    load_split_bundle,
    rank_embeddings,
    verify_split_integrity,
)
from workbench.benchmarks.schema import DEFAULT_SPLIT_TAG, SplitBundle  # noqa: E402
from workbench.benchmarks.trainers.common import (  # noqa: E402
    get_git_sha,
    seed_everything,
    utc_now_iso,
    write_json,
)


def _rank_mean_ensemble(
    head_pairs: List[Tuple[np.ndarray, np.ndarray]],
    query_ids: List[str],
    corpus_ids: List[str],
    top_k: int = 50,
    q_chunk: int = 256,
) -> Dict[str, List[str]]:
    """Compute exact mean cosine similarity across multiple heads (e.g., dress+shirt+toptee)."""
    n_q = len(query_ids)
    n_c = len(corpus_ids)
    k = min(int(top_k), n_c)
    norm_heads: List[Tuple[np.ndarray, np.ndarray]] = []
    for q_emb, c_emb in head_pairs:
        q = q_emb.astype(np.float32, copy=False)
        c = c_emb.astype(np.float32, copy=False)
        if not np.isfinite(q).all() or not np.isfinite(c).all():
            raise ValueError("Non-finite embeddings detected in multi-head ensemble")
        q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        c = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
        norm_heads.append((q, c))

    preds: Dict[str, List[str]] = {}
    n_heads = float(len(norm_heads))
    for qs in range(0, n_q, q_chunk):
        qe = min(qs + q_chunk, n_q)
        sims = np.zeros((qe - qs, n_c), dtype=np.float32)
        for q, c in norm_heads:
            sims += q[qs:qe] @ c.T
        sims /= n_heads
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        for row_i, qid in enumerate(query_ids[qs:qe]):
            preds[str(qid)] = [str(corpus_ids[ci]) for ci in order[row_i]]
    return preds


def _eval_preds(dataset: str, preds: Dict[str, List[str]], bundle: SplitBundle) -> Dict[str, Any]:
    if dataset == "edir":
        return evaluate_edir(preds, bundle.queries, ks=(1, 5, 10, 50), strict=True)
    return evaluate_pinpoint(preds, bundle.queries, ks=(1, 5, 10, 50), use_raw_gt=True, strict=True)


def _infer_encoder_hparams(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer exact (local_token_num, wc, N_p) directly from ENCODER checkpoint tensor shapes."""
    lw = int(state_dict["local_weight"].shape[0])
    local_token_num = int(state_dict["backbone.fac_net.weight.3.weight"].shape[0])
    n_p = int(state_dict["backbone.binding_decoder.mlp_p.0.weight"].shape[0])
    assert local_token_num + n_p + 1 == lw, (
        f"ENCODER shape invariant violated: local_token_num={local_token_num}, N_p={n_p}, local_weight={lw}"
    )
    if "backbone.t_weight_phrases.5.weight" in state_dict:
        wc = 3
    elif "backbone.t_weight_phrases.3.weight" in state_dict:
        wc = 2
    else:
        wc = 1
    return local_token_num, wc, n_p


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = max(0, min(int(args.num_workers), 8))
    model_key = args.model.lower().strip()
    ckpt_paths = [Path(p) for p in args.ckpts]
    for p in ckpt_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    if not args.smoke_only:
        verify_split_integrity(args.dataset, split_tag=args.split_tag)
        test_bundle = load_split_bundle(args.dataset, "test", corpus_scope="full", split_tag=args.split_tag)
    else:
        test_bundle = None

    t0 = time.time()
    load_report: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    preds: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # 1. CLIP4Cir (RN50x4 CLIP + Combiner)
    # ------------------------------------------------------------------
    if model_key == "clip4cir":
        import clip
        from combiner import Combiner
        from data_utils import targetpad_transform
        from workbench.benchmarks.trainers.train_clip4cir import evaluate_split

        assert len(ckpt_paths) == 2, "clip4cir requires [clip_ckpt, comb_ckpt]"
        clip_model, _ = clip.load("RN50x4", device=device, jit=False)
        clip_model.eval().float()
        c_state = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        clip_model.load_state_dict(c_state["CLIP"], strict=True)
        combiner = Combiner(clip_model.visual.output_dim, 2560, 5120).to(device)
        comb_state = torch.load(ckpt_paths[1], map_location=device, weights_only=False)
        combiner.load_state_dict(comb_state["Combiner"], strict=True)
        combiner.eval()
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        load_report = {"strict": True, "input_dim": clip_model.visual.input_resolution}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, clip_model.visual.input_resolution, clip_model.visual.input_resolution, device=device)
                toks = clip.tokenize(["change shirt to red"] * 4, context_length=77, truncate=True).to(device)
                rf = clip_model.encode_image(x).float()
                tf = clip_model.encode_text(toks).float()
                cf = combiner.combine_features(rf, tf)
                assert cf.shape == (4, 640) and torch.isfinite(cf).all()
            print(f"[SMOKE PASS] clip4cir ({args.variant}) shape={tuple(cf.shape)}", flush=True)
            return

        metrics, preds, _ = evaluate_split(
            args.dataset, clip_model, test_bundle, preprocess, combiner.combine_features,
            device, eval_bs=128, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 2. SPRC (BLIP-2 ViT-L author checkpoints: pretrain_vitL)
    # ------------------------------------------------------------------
    elif model_key == "sprc":
        from data_utils import targetpad_transform
        from lavis.models import load_model_and_preprocess
        from workbench.benchmarks.trainers.train_sprc import evaluate_sprc_split

        model, _, txt_processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt", model_type="pretrain_vitL", is_eval=True, device=device
        )
        state = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        cls_key = model.__class__.__name__
        sd = state[cls_key] if isinstance(state, dict) and cls_key in state else state
        model.load_state_dict(sd, strict=True)
        model.eval()
        txt_proc = txt_processors["eval"]
        preprocess = targetpad_transform(1.25, 224)
        load_report = {"strict": True, "model_type": "pretrain_vitL", "cls_key": cls_key}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                tf, re = model.extract_target_features(x, mode="mean")
                sim = model.inference(re, tf, [txt_proc("change to red")] * 4, None)
                assert sim.shape == (4, 4) and torch.isfinite(sim).all()
            print(f"[SMOKE PASS] sprc ({args.variant}) sim={tuple(sim.shape)}", flush=True)
            return

        metrics, preds, _ = evaluate_sprc_split(
            args.dataset, model, test_bundle, preprocess, txt_proc, device,
            eval_bs=64, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 3. MTST_PTHA (Blip2QformerCirCat, pretrain)
    # ------------------------------------------------------------------
    elif model_key == "mtst_ptha":
        from data_utils import targetpad_transform
        from lavis.models import load_model_and_preprocess
        from workbench.benchmarks.trainers.train_mtst_ptha import evaluate_mtst_split

        model, _, txt_processors = load_model_and_preprocess(
            name="blip2_cir_cat", model_type="pretrain", is_eval=True, device=device
        )
        state = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        cls_key = model.__class__.__name__
        sd = state[cls_key] if isinstance(state, dict) and cls_key in state else state
        model.load_state_dict(sd, strict=True)
        model.eval()
        txt_proc = txt_processors["eval"]
        preprocess = targetpad_transform(1.25, 224)
        load_report = {"strict": True, "cls_key": cls_key}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                with model.maybe_autocast():
                    ie = model.ln_vision(model.visual_encoder(x))
                ie = ie.float()
                ia = torch.ones(ie.size()[:-1], dtype=torch.long, device=device)
                qt = model.query_tokens.expand(4, -1, -1)
                qo = model.Qformer.bert(query_embeds=qt, encoder_hidden_states=ie, encoder_attention_mask=ia, return_dict=True)
                tf = F.normalize(model.vision_proj(qo.last_hidden_state), dim=-1)
                assert tf.shape == (4, 32, 256) and torch.isfinite(tf).all()
            print(f"[SMOKE PASS] mtst_ptha ({args.variant}) tf={tuple(tf.shape)}", flush=True)
            return

        metrics, preds, _ = evaluate_mtst_split(
            args.dataset, model, test_bundle, preprocess, txt_proc, device,
            eval_bs=64, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 4. CSMCIR (blip2_cir_align_prompt_csmcir, pretrain)
    # ------------------------------------------------------------------
    elif model_key == "csmcir":
        from data_utils_csmcir import targetpad_transform
        from lavis.models import load_model_and_preprocess
        from workbench.benchmarks.trainers.train_csmcir import evaluate_csmcir_split

        model, _, txt_processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=True, device=device
        )
        state = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        cls_key = model.__class__.__name__
        if isinstance(state, dict) and cls_key in state:
            sd = state[cls_key]
        elif isinstance(state, dict):
            dicts = [v for k, v in state.items() if k != "epoch" and isinstance(v, dict)]
            assert len(dicts) == 1, f"Ambiguous CSMCIR state keys: {list(state.keys())}"
            sd = dicts[0]
        else:
            raise TypeError(f"Unexpected CSMCIR checkpoint type: {type(state)}")
        res = model.load_state_dict(sd, strict=False)
        assert not res.missing_keys, f"CSMCIR missing keys: {res.missing_keys}"
        allowed_unexp = {"token_importance"}
        bad_unexp = [k for k in res.unexpected_keys if k not in allowed_unexp]
        assert not bad_unexp, f"CSMCIR unexpected keys: {bad_unexp}"
        model.eval()
        txt_proc = txt_processors["eval"]
        preprocess = targetpad_transform(1.25, 224)
        load_report = {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                tf, re = model.extract_target_caption_features(x, [""] * 4, mode="mean")
                sim = model.inference(re, tf, [txt_proc("make it blue")] * 4, None)
                assert sim.shape == (4, 4) and torch.isfinite(sim).all()
            print(f"[SMOKE PASS] csmcir ({args.variant}) sim={tuple(sim.shape)} unexp={res.unexpected_keys}", flush=True)
            return

        metrics, preds, _ = evaluate_csmcir_split(
            args.dataset, model, test_bundle, preprocess, txt_proc, device,
            eval_bs=64, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 5. iLearn BLIP-2 family (HINT, HABIT, INTENT, ConeSep, AirKnow)
    # ------------------------------------------------------------------
    elif model_key in ("hint", "habit", "intent", "conesep", "airknow"):
        from workbench.benchmarks.trainers.train_ilearn import (
            _patch_bert_root_path,
            evaluate_ilearn_split,
        )

        _patch_bert_root_path()
        from data_utils import targetpad_transform
        from lavis.processors.blip_processors import BlipCaptionProcessor

        obj = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"Expected whole pickled nn.Module for {model_key}, got {type(obj)}")
        model = obj.to(device).eval()
        txt_proc = BlipCaptionProcessor()
        preprocess = targetpad_transform(1.25, 224)
        load_report = {"whole_module_class": type(model).__name__}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                caps = [txt_proc("make it red")] * 4
                qf = model.extract_retrieval_compose(x, caps)
                tf = model.extract_retrieval_target(x)
                assert torch.isfinite(qf).all() and torch.isfinite(tf).all()
            print(f"[SMOKE PASS] {model_key} ({args.variant}) qf={tuple(qf.shape)} tf={tuple(tf.shape)}", flush=True)
            return

        metrics, preds, _ = evaluate_ilearn_split(
            args.dataset, model, test_bundle, preprocess, txt_proc, device,
            eval_bs=64, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 6. TME (blip2_cir_image_diff_features, pretrain)
    # ------------------------------------------------------------------
    elif model_key == "tme":
        from workbench.benchmarks.trainers.train_tme import TME_DIR, evaluate_tme_split
        from lavis.models import load_model_and_preprocess

        cwd_before = os.getcwd()
        try:
            os.chdir(str(TME_DIR))
            model, _, txt_processors = load_model_and_preprocess(
                name="blip2_cir_image_diff_features", model_type="pretrain", is_eval=False, device=device
            )
        finally:
            os.chdir(cwd_before)
        state = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        sd = state.get("model", state) if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict) else state
        res = model.load_state_dict(sd, strict=False)
        assert not res.missing_keys and not res.unexpected_keys, (
            f"TME incomplete load: missing={res.missing_keys}, unexpected={res.unexpected_keys}"
        )
        model.eval()
        txt_proc = txt_processors["eval"]
        import utility as tme_utility
        preprocess = tme_utility.targetpad_transform(target_ratio=1.25, dim=224)
        load_report = {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                vit_e = model.vit_encode(x)
                f_img = model.encode_image(vit_e)
                tf = F.normalize(model.vision_proj(f_img), dim=-1)
                toks = model.tokenizer(
                    [txt_proc("make it red")] * 4,
                    padding="max_length",
                    truncation=True,
                    max_length=model.max_txt_len,
                    return_tensors="pt",
                ).to(device)
                qf = model.encode_fusion(f_img, toks)
                assert torch.isfinite(tf).all() and torch.isfinite(qf).all()
            print(f"[SMOKE PASS] tme ({args.variant}) tf={tuple(tf.shape)} qf={tuple(qf.shape)}", flush=True)
            return

        metrics, preds, _ = evaluate_tme_split(
            args.dataset, model, test_bundle, preprocess, txt_proc, device,
            eval_bs=64, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 7. ENCODER (open_clip ViT-B-32)
    # ------------------------------------------------------------------
    elif model_key == "encoder":
        from workbench.benchmarks.trainers.train_encoder import (
            build_encoder_model,
            evaluate_encoder_split,
        )

        raw_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        if isinstance(raw_ckpt, nn.Module):
            _, _, val_preprocess = build_encoder_model(device=device)
            model = raw_ckpt.to(device).eval()
            load_report = {"whole_module": True}
        else:
            sd = raw_ckpt.get("model_state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
            ltn, wc, n_p = _infer_encoder_hparams(sd)
            torch.manual_seed(42)  # Deterministic relation seed per replay/encoder.py:499
            model, _, val_preprocess = build_encoder_model(
                local_token_num=ltn, wc=wc, n_p=n_p, device=device
            )
            res = model.load_state_dict(sd, strict=False)
            assert not res.missing_keys, f"ENCODER missing keys (ltn={ltn},wc={wc},Np={n_p}): {res.missing_keys}"
            model.eval()
            load_report = {
                "local_token_num": ltn,
                "wc": wc,
                "n_p": n_p,
                "missing_keys": list(res.missing_keys),
                "unexpected_keys": list(res.unexpected_keys),
            }

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                q = model.extract_retrieval_compose(x, ["make it red"] * 4)
                t = model.extract_retrieval_target(x)
                assert q.shape == (4, 512) and t.shape == (4, 512) and torch.isfinite(q).all()
            print(f"[SMOKE PASS] encoder ({args.variant}) {load_report}", flush=True)
            return

        metrics, preds, _ = evaluate_encoder_split(
            args.dataset, model, test_bundle, val_preprocess, device,
            eval_bs=128, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 8. TG-CIR (CLIP ViT-B/16; PyTorch 1.7 pickled nn.Module shims)
    # ------------------------------------------------------------------
    elif model_key == "tgcir":
        import torch.nn.modules.linear as torch_linear
        if not hasattr(torch_linear, "_LinearWithBias"):
            torch_linear._LinearWithBias = torch.nn.Linear
        import clip
        from workbench.benchmarks.trainers.train_tgcir import evaluate_tgcir_split

        obj = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        if not isinstance(obj, nn.Module):
            raise TypeError(f"Expected whole pickled VCG_CIR model for tgcir, got {type(obj)}")
        n_mha_patched = 0
        for m in obj.modules():
            if isinstance(m, nn.MultiheadAttention) and not hasattr(m, "batch_first"):
                m.batch_first = False
                n_mha_patched += 1
        model = obj.to(device).eval()
        _, preprocess = clip.load("ViT-B/16", device="cpu", jit=False)
        load_report = {"whole_module_class": type(model).__name__, "mha_batch_first_patched": n_mha_patched}

        if args.smoke_only:
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                q = model.extract_retrieval_compose(x, ["word " * 100] * 4)
                t = model.extract_retrieval_target(x)
                assert q.shape == (4, 512) and t.shape == (4, 512) and torch.isfinite(q).all()
            print(f"[SMOKE PASS] tgcir ({args.variant}) q={tuple(q.shape)} mha_patched={n_mha_patched}", flush=True)
            return

        metrics, preds, _ = evaluate_tgcir_split(
            args.dataset, model, test_bundle, preprocess, device,
            eval_bs=128, num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # 9. LIMN (single CIRR head id=5 OR 3-head FashionIQ ensemble 0/1/2)
    # ------------------------------------------------------------------
    elif model_key == "limn":
        import open_clip
        from workbench.benchmarks.trainers.train_limn import (
            extract_limn_corpus_embeddings,
            extract_limn_query_embeddings,
        )

        dt_ids = [int(x) for x in args.dataset_type_ids.split(",")]
        assert len(dt_ids) == len(ckpt_paths), "dataset_type_ids must match number of ckpts"
        head_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        qids_ref: Optional[List[str]] = None
        cids_ref: Optional[List[str]] = None

        for ckpt_p, dt_id in zip(ckpt_paths, dt_ids):
            model = torch.load(ckpt_p, map_location=device, weights_only=False).to(device).eval()
            vis = model.backbone.clip.visual
            img_sz = vis.image_size if isinstance(vis.image_size, tuple) else (int(vis.image_size), int(vis.image_size))
            val_preprocess = open_clip.image_transform(
                vis.image_size, is_train=False, mean=vis.image_mean, std=vis.image_std
            )
            if args.smoke_only:
                with torch.no_grad():
                    x = torch.randn(4, 3, img_sz[0], img_sz[1], device=device)
                    dt = torch.full((4,), dt_id, dtype=torch.long, device=device)
                    q = model.compose_query(x, ["change color to blue"] * 4, dt)[0]
                    t = model.compose_target(x, dt)[0]
                    assert q.shape == (4, 768) and t.shape == (4, 768) and torch.isfinite(q).all()
                del model
                torch.cuda.empty_cache()
                continue

            c_emb, c_ids = extract_limn_corpus_embeddings(
                model, test_bundle, val_preprocess, dataset_type_id=dt_id, device=device, eval_bs=128, num_workers=num_workers
            )
            q_emb, q_ids = extract_limn_query_embeddings(
                model, test_bundle, val_preprocess, dataset_type_id=dt_id, device=device, eval_bs=128, num_workers=num_workers
            )
            if qids_ref is None:
                qids_ref, cids_ref = q_ids, c_ids
            else:
                assert q_ids == qids_ref and c_ids == cids_ref
            head_pairs.append((q_emb, c_emb))
            del model
            gc.collect()
            torch.cuda.empty_cache()

        load_report = {"num_heads": len(ckpt_paths), "dataset_type_ids": dt_ids}
        if args.smoke_only:
            print(f"[SMOKE PASS] limn ({args.variant}) heads={len(ckpt_paths)} dt_ids={dt_ids}", flush=True)
            return

        if len(head_pairs) == 1:
            preds = rank_embeddings(head_pairs[0][0], qids_ref, head_pairs[0][1], cids_ref, top_k=50, l2_normalize=True)
        else:
            preds = _rank_mean_ensemble(head_pairs, qids_ref, cids_ref, top_k=50)
        metrics = _eval_preds(args.dataset, preds, test_bundle)

    # ------------------------------------------------------------------
    # 10. CLVC-Net (3-head FashionIQ ensemble: dress/shirt/toptee pairs)
    # ------------------------------------------------------------------
    elif model_key == "clvcnet":
        from workbench.benchmarks.trainers.train_clvcnet import (
            extract_corpus_embeddings,
            extract_query_embeddings,
        )

        assert len(ckpt_paths) % 2 == 0 and len(ckpt_paths) >= 2, "clvcnet expects pairs [local, global, ...]"
        # Native FashionIQ validation pipeline per replay/clvc_net.py:390-395
        val_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        head_pairs = []
        qids_ref = None
        cids_ref = None
        n_heads = len(ckpt_paths) // 2

        for hi in range(n_heads):
            loc_m = torch.load(ckpt_paths[2 * hi], map_location=device, weights_only=False).to(device).eval()
            glb_m = torch.load(ckpt_paths[2 * hi + 1], map_location=device, weights_only=False).to(device).eval()
            if args.smoke_only:
                with torch.no_grad():
                    x = torch.randn(4, 3, 224, 224, device=device)
                    caps = ["make the dress red", "unknown", "change shirt color", "longer sleeves"]
                    lf = F.normalize(loc_m.compose_img_text(x, caps)[1].float(), dim=-1)
                    gf = F.normalize(glb_m.compose_img_text(x, caps)[0].float(), dim=-1)
                    assert torch.isfinite(lf).all() and torch.isfinite(gf).all()
                del loc_m, glb_m
                torch.cuda.empty_cache()
                continue

            c_emb, c_ids = extract_corpus_embeddings(
                loc_m, glb_m, test_bundle, val_preprocess, device, eval_bs=128, num_workers=num_workers
            )
            q_emb, q_ids = extract_query_embeddings(
                loc_m, glb_m, test_bundle, val_preprocess, device, eval_bs=128, num_workers=num_workers
            )
            if qids_ref is None:
                qids_ref, cids_ref = q_ids, c_ids
            else:
                assert q_ids == qids_ref and c_ids == cids_ref
            head_pairs.append((q_emb, c_emb))
            del loc_m, glb_m
            gc.collect()
            torch.cuda.empty_cache()

        load_report = {"num_heads": n_heads}
        if args.smoke_only:
            print(f"[SMOKE PASS] clvcnet ({args.variant}) heads={n_heads}", flush=True)
            return

        if len(head_pairs) == 1:
            preds = rank_embeddings(head_pairs[0][0], qids_ref, head_pairs[0][1], cids_ref, top_k=50, l2_normalize=True)
        else:
            preds = _rank_mean_ensemble(head_pairs, qids_ref, cids_ref, top_k=50)
        metrics = _eval_preds(args.dataset, preds, test_bundle)

    # ------------------------------------------------------------------
    # 11. DCNet (FashionIQ trained_model.pth + GloVe 300d)
    # ------------------------------------------------------------------
    elif model_key == "dcnet":
        import json, pickle, subprocess
        from workbench.benchmarks.trainers.train_dcnet import (
            DCNET_PREPARE_PY,
            GLOVE_SPACY_DIR,
            _tokenize_dcnet,
            build_dcnet_model,
            encode_captions_to_glove_batch,
            evaluate_dcnet_split,
        )

        def _build_glove_hits_only(captions: List[str], cache_path: Path) -> Dict[str, np.ndarray]:
            """Transfer-safe GloVe 300d lookup: real pretrained GloVe vectors only (zero random OOV noise)."""
            if cache_path.is_file():
                with cache_path.open("rb") as f:
                    return pickle.load(f)
            words = sorted({tok for cap in captions for tok in _tokenize_dcnet(cap)})
            tmp_words = cache_path.with_suffix(".words.json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_words.write_text(json.dumps(words), encoding="utf-8")
            script = f"""
import json, pickle, numpy as np, spacy
nlp = spacy.load({str(GLOVE_SPACY_DIR)!r})
words = json.loads(open({str(tmp_words)!r}).read())
vocab = {{}}
for w in words:
    lex = nlp.vocab[w]
    if lex.has_vector and np.linalg.norm(lex.vector) > 0:
        vocab[w] = np.asarray(lex.vector, dtype=np.float32)
with open({str(cache_path)!r}, 'wb') as f:
    pickle.dump(vocab, f)
print(f'DCNet transfer GloVe hits: {{len(vocab)}}/{{len(words)}}')
"""
            py_exe = str(DCNET_PREPARE_PY) if DCNET_PREPARE_PY.is_file() else sys.executable
            out = subprocess.check_output([py_exe, "-c", script], text=True)
            print(f"[DCNet-Transfer] {out.strip()}", flush=True)
            tmp_words.unlink(missing_ok=True)
            with cache_path.open("rb") as f:
                return pickle.load(f)

        model, modalities = build_dcnet_model(device)
        ckpt = torch.load(ckpt_paths[0], map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        val_preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        load_report = {"strict": True, "modalities": modalities}

        if args.smoke_only:
            vocab = _build_glove_hits_only(
                ["change the shirt to bright red", "remove the car"], Path("/tmp/dcnet_smoke_vocab.pkl")
            )
            with torch.no_grad():
                x = torch.randn(4, 3, 224, 224, device=device)
                tt, tl = encode_captions_to_glove_batch(["change shirt to red"] * 4, vocab)
                re = model.get_image_emb({"im_feat": x}, istgt=False)
                te = model.get_image_emb({"im_feat": x}, istgt=True)
                tx = model.get_text_emb(tt.to(device), tl.tolist())
                sc = model.get_score_matrix(te, model.get_composition(re, tx), modalities)
                assert sc.shape == (4, 4) and torch.isfinite(sc).all()
            Path("/tmp/dcnet_smoke_vocab.pkl").unlink(missing_ok=True)
            print(f"[SMOKE PASS] dcnet ({args.variant}) sc={tuple(sc.shape)}", flush=True)
            return

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocab = _build_glove_hits_only(
            [q.text for q in test_bundle.queries],
            out_dir / f"glove_vocab_transfer_{args.dataset}.pkl",
        )
        metrics, preds, _ = evaluate_dcnet_split(
            args.dataset, model, modalities, vocab, test_bundle, val_preprocess, device,
            eval_bs=128, q_bs=8, c_chunk=4096, num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unsupported model: {model_key}")

    elapsed = time.time() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "variant": args.variant,
        "source_benchmark": args.source_benchmark,
        "target_dataset": args.dataset,
        "split_tag": args.split_tag,
        "checkpoints": [str(p) for p in ckpt_paths],
        "load_report": load_report,
        "git_sha": get_git_sha(),
        "finished_at": utc_now_iso(),
        "elapsed_sec": round(elapsed, 2),
        "test_metrics": metrics,
    }
    write_json(out_dir / "test_results.json", payload)
    write_json(out_dir / "test_predictions_top50.json", preds)
    primary = metrics["overall"].get("Recall@1", metrics["overall"].get("mAP@10", 0.0))
    print(f"[TRANSFER DONE] {model_key} ({args.variant}) -> {args.dataset} | primary={primary:.4f} | {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-benchmark CIR checkpoint transfer evaluator")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True, help="e.g. fiq_clean, fiq_n02, fiq_n05, cirr_clean, cirr_n02")
    parser.add_argument("--source-benchmark", type=str, required=True, choices=["fashioniq", "cirr"])
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default=DEFAULT_SPLIT_TAG)
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--dataset-type-ids", type=str, default="5", help="Comma-separated LIMN dataset_type IDs")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/transfer_eval_tmp"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
