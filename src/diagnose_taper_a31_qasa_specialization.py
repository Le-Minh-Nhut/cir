from __future__ import annotations

"""
Comprehensive diagnosis for TAPER A3.1 (no-NULL + QASA slot filtering).

Questions answered:
1) Is ownership sharp and decomposed, or diffuse / near-uniform?
2) Do the 4 Edit Slots encode different semantics/effects, or the same task repeatedly?
3) Is QASA pruning meaningful specialization or pruning diffuse attention?
4) Are slot contributions causally distinct?
5) Does QASA help vs all-slots / winner-active counterfactuals?
6) Do slots specialize in primitive routing / state transitions?

Outputs:
- JSON full report
- Markdown research report

Automatic verdict thresholds are HEURISTIC. Raw metrics + causal ablations are the evidence.
"""

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_features,
    load_text_features,
)
from datasets.common import collate_cir_samples
from datasets.fashioniq import FashionIQDataset, load_correction_dict
from evaluation.fashioniq import build_fashioniq_gallery
from models.taper import TAPER
from teachers.csmcir_compose import CSMCIRComposeTeacher

CATEGORIES = ("dress", "shirt", "toptee")
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        description="Comprehensive TAPER A3.1 QASA/no-NULL specialization diagnosis."
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument("--config", type=Path, default=Path("conf/experiment/taper_e2e.yaml"))
    p.add_argument(
        "--protocol",
        type=str,
        default="fashioniq_original",
        choices=("fashioniq_original", "fashioniq_val"),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--max-queries-per-category",
        type=int,
        default=512,
        help="0 = full validation; 512/category is a strong diagnostic pass.",
    )
    p.add_argument("--top-worst", type=int, default=20)
    p.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/taper_a31_qasa_specialization_diagnosis.json"),
    )
    p.add_argument(
        "--md-output",
        type=Path,
        default=Path("reports/taper_a31_qasa_specialization_diagnosis.md"),
    )
    return p.parse_args()


def safe_mean(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, d=4):
    x = float(x)
    return "nan" if not math.isfinite(x) else f"{x:.{d}f}"


class ScalarAccumulator:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, name, values):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor([values], dtype=torch.float32)
        values = values.detach().float().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        self.sum[name] += float(values.sum().item())
        self.count[name] += int(values.numel())

    def export(self):
        return {
            k: self.sum[k] / self.count[k]
            for k in sorted(self.sum)
            if self.count[k] > 0
        }


def newest_checkpoint(root: Path) -> Path:
    xs = list(root.rglob("best.pt"))
    if not xs:
        raise FileNotFoundError(
            f"No best.pt under {root}; pass --checkpoint explicitly."
        )
    return max(xs, key=lambda x: x.stat().st_mtime)


def correction_dicts(annotation_root: Path):
    out = {}
    for c in CATEGORIES:
        path = annotation_root / f"correction_dict_{c}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        out[c] = load_correction_dict(path)
    return out


def build_val_loaders(annotation_root, batch_size, num_workers, caption_policy, corrections):
    loaders, annotations = {}, {}
    for c in CATEGORIES:
        ds = FashionIQDataset(
            annotation_root=annotation_root,
            split="val",
            categories=[c],
            caption_policy=caption_policy,
            correction_dicts=corrections,
        )
        loaders[c] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_cir_samples,
            pin_memory=True,
        )
        annotations[c] = ds.annotations
    return loaders, annotations


def build_model(cfg, device):
    m = cfg.model
    teacher = CSMCIRComposeTeacher(
        csmcir_root=cfg.teacher.csmcir_root,
        checkpoint_path=cfg.teacher.checkpoint_path,
    ).to(device).eval()
    return TAPER(
        teacher,
        text_dim=m.text_dim,
        reference_dim=m.reference_dim,
        teacher_text_dim=m.teacher_text_dim,
        teacher_query_dim=m.teacher_query_dim,
        query_dim=m.query_dim,
        slot_dim=m.slot_dim,
        state_dim=m.state_dim,
        num_slots=m.num_slots,
        num_primitives=m.num_primitives,
        mask_temperature=m.mask_temperature,
        router_temperature=m.router_temperature,
        retrieval_temperature=m.retrieval_temperature,
        neutral_mode=m.neutral_mode,
        qasa_tau=m.qasa_tau,
        qasa_rho=m.qasa_rho,
        qasa_mu=m.qasa_mu,
        qasa_eps=m.qasa_eps,
        qasa_apply_at_eval=m.qasa_apply_at_eval,
        alpha_max=m.alpha_max,
        counterfactual_chunk_size=m.counterfactual_chunk_size,
    ).to(device).eval()


def load_checkpoint(model, path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("teacher.")]
    if bad_missing:
        raise RuntimeError("Missing non-teacher keys:\n" + "\n".join(bad_missing))
    if unexpected:
        raise RuntimeError("Unexpected keys:\n" + "\n".join(unexpected))
    print("Loaded:", path)


def pairwise_cosine(x, active=None, min_norm=1e-8):
    b, l, _ = x.shape
    norm = x.norm(dim=-1)
    eligible = norm > min_norm
    if active is not None:
        eligible &= active.bool()
    z = F.normalize(x, dim=-1, eps=1e-8)
    sim = z @ z.transpose(1, 2)
    upper = torch.triu(
        torch.ones(l, l, dtype=torch.bool, device=x.device), diagonal=1
    )
    mask = eligible[:, :, None] & eligible[:, None, :] & upper[None]
    den = mask.sum((1, 2))
    num = (sim * mask.to(sim.dtype)).sum((1, 2))
    out = torch.full((b,), float("nan"), device=x.device, dtype=sim.dtype)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def pairwise_js_token_maps(masks, valid):
    b, l, _ = masks.shape
    x = masks * valid[:, None, :].to(masks.dtype)
    p = x / x.sum(-1, keepdim=True).clamp_min(EPS)
    p = p.clamp_min(EPS)
    vals = []
    for i, j in combinations(range(l), 2):
        pi, pj = p[:, i], p[:, j]
        mid = 0.5 * (pi + pj)
        js = 0.5 * (
            (pi * (pi.log() - mid.log())).sum(-1)
            + (pj * (pj.log() - mid.log())).sum(-1)
        )
        vals.append(js / math.log(2.0))
    return torch.stack(vals, 1).mean(1) if vals else torch.zeros(b, device=masks.device)


def normalized_token_entropy(attn, valid):
    l = attn.shape[1]
    p = attn.clamp_min(EPS)
    h = -(p * p.log()).sum(1)
    if l > 1:
        h /= math.log(l)
    vf = valid.to(h.dtype)
    return (h * vf).sum(1) / vf.sum(1).clamp_min(1.0)


def top1_margin(attn, valid):
    top = attn.topk(k=min(2, attn.shape[1]), dim=1).values
    top1 = top[:, 0]
    margin = top[:, 0] - top[:, 1] if attn.shape[1] > 1 else top[:, 0]
    vf = valid.to(attn.dtype)
    den = vf.sum(1).clamp_min(1.0)
    return (top1 * vf).sum(1) / den, (margin * vf).sum(1) / den


def winner_stats(attn, valid):
    b, l, _ = attn.shape
    winner = attn.argmax(1)
    oh = F.one_hot(winner, num_classes=l).permute(0, 2, 1).bool()
    oh &= valid[:, None, :]
    counts = oh.sum(-1).to(attn.dtype)
    active = (counts > 0).sum(1).to(attn.dtype)
    p = counts / counts.sum(1, keepdim=True).clamp_min(1.0)
    h = -(p.clamp_min(EPS) * p.clamp_min(EPS).log()).sum(1)
    if l > 1:
        h /= math.log(l)
    return counts, active, h


def effective_rank(x):
    s = torch.linalg.svdvals(x.float())
    e = s.square()
    p = e / e.sum(1, keepdim=True).clamp_min(EPS)
    h = -(p.clamp_min(EPS) * p.clamp_min(EPS).log()).sum(1)
    r = torch.exp(h)
    return torch.where(e.sum(1) > EPS, r, torch.zeros_like(r))


def coeff_var(x, dim=1):
    return x.std(dim=dim, unbiased=False) / x.mean(dim=dim).abs().clamp_min(EPS)


def target_similarity(query, targets):
    targets = F.normalize(targets, dim=-1)
    return torch.einsum("bd,bkd->bk", query, targets).amax(-1)


def winner_active_mask(attn, valid):
    b, l, _ = attn.shape
    w = attn.argmax(1)
    oh = F.one_hot(w, num_classes=l).permute(0, 2, 1).bool()
    return (oh & valid[:, None, :]).any(-1)


def execute_query(model, slots, z0, reference_state, selected_mask, disabled=None):
    ex = model.execute(
        slots["edit_slots"],
        selected_mask,
        z0,
        reference_state,
        disabled_slots=disabled,
    )
    return model.make_query(ex["final_state"]), ex


def target_ranks(model, query, gallery, target_indices):
    scores = model._retrieval_scores(query, gallery)
    target_scores = scores.gather(1, target_indices[:, None])
    return (scores > target_scores).sum(1) + 1


def update_retrieval(stats, variant, category, ranks):
    d = stats[variant][category]
    d["n"] += int(ranks.numel())
    d["hit10"] += int((ranks <= 10).sum())
    d["hit50"] += int((ranks <= 50).sum())
    d["rank_sum"] += float(ranks.float().sum())


def summarize_retrieval(stats):
    out = {}
    for variant, cats in stats.items():
        by_cat = {}
        for c, d in cats.items():
            if not d["n"]:
                continue
            n = d["n"]
            by_cat[c] = {
                "n": n,
                "recall_at_10": 100 * d["hit10"] / n,
                "recall_at_50": 100 * d["hit50"] / n,
                "mean_rank": d["rank_sum"] / n,
            }
        if by_cat:
            r10 = safe_mean(x["recall_at_10"] for x in by_cat.values())
            r50 = safe_mean(x["recall_at_50"] for x in by_cat.values())
            out[variant] = {
                "recall_at_10": r10,
                "recall_at_50": r50,
                "mean_recall": (r10 + r50) / 2,
                "mean_rank": safe_mean(x["mean_rank"] for x in by_cat.values()),
                "categories": by_cat,
            }
    return out


def normalized_mi(counts):
    c = counts.double()
    total = c.sum()
    if total <= 0:
        return {"mi": float("nan"), "nmi": float("nan")}
    p = c / total
    ps, pp = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    denom = ps @ pp
    nz = p > 0
    mi = (p[nz] * (p[nz] / denom[nz]).log()).sum()
    hs = -(ps[ps > 0] * ps[ps > 0].log()).sum()
    hp = -(pp[pp > 0] * pp[pp > 0].log()).sum()
    norm = torch.minimum(hs, hp)
    return {
        "mi": float(mi),
        "nmi": float(mi / norm) if norm > 0 else float("nan"),
        "slot_entropy": float(hs),
        "primitive_entropy": float(hp),
    }


def build_verdict(m, router, num_slots):
    baseline = 1.0 / num_slots
    diffuse = {
        "token_entropy_near_uniform": m.get("ownership/token_entropy_norm", float("nan")) >= 0.85,
        "top1_margin_small": m.get("ownership/top1_margin", float("nan")) <= 0.12,
        "quality_near_uniform_baseline": m.get("qasa/quality_mean", float("nan")) <= baseline + 0.08,
        "ownership_maps_highly_similar": m.get("ownership/pairwise_cosine", float("nan")) >= 0.85,
    }
    monopoly = {
        "dominant_mass_large": m.get("ownership/dominant_mass_share", float("nan")) >= 0.75,
        "winner_active_slots_low": m.get("ownership/winner_active_slot_count", float("nan")) <= 1.5,
        "winner_balance_entropy_low": m.get("ownership/winner_balance_entropy", float("nan")) <= 0.35,
    }
    shared = {
        "slot_effects_highly_aligned": m.get("representation/slot_effect_pairwise_cosine", float("nan")) >= 0.80,
        "slot_effect_effective_rank_low": m.get("representation/slot_effect_effective_rank", float("nan")) <= 1.8,
        "forced_single_slot_effects_aligned": m.get("causal/forced_only_effect_pairwise_cosine", float("nan")) >= 0.80,
        "drop_contribution_directions_aligned": m.get("causal/drop_direction_pairwise_cosine", float("nan")) >= 0.80,
        "executed_state_changes_aligned": m.get("execution/state_change_pairwise_cosine", float("nan")) >= 0.80,
        "slot_primitive_dependence_low": (
            math.isfinite(router.get("nmi", float("nan")))
            and router["nmi"] <= 0.10
        ),
    }
    ds, ms, ss = map(lambda d: sum(bool(x) for x in d.values()), (diffuse, monopoly, shared))
    if ms >= 2:
        own = "MONOPOLY / WINNER COLLAPSE evidence"
    elif ds >= 3:
        own = "DIFFUSE / NEAR-SYMMETRIC ownership evidence"
    elif ds == 2:
        own = "MIXED ownership; specialization not convincing"
    else:
        own = "No strong diffuse/monopoly flag"

    if ss >= 4:
        fun = "STRONG shared-task / functional redundancy evidence"
    elif ss >= 2:
        fun = "MIXED-PARTIAL specialization; redundancy remains plausible"
    else:
        fun = "No strong shared-task evidence"

    if ms >= 2:
        overall = "FAILURE MODE: monopoly collapse"
    elif ds >= 3 and ss >= 2:
        overall = "FAILURE MODE: diffuse ownership + functional redundancy"
    elif ds >= 3:
        overall = "FAILURE MODE: diffuse ownership / QASA pruning before specialization"
    elif ss >= 4:
        overall = "FAILURE MODE: ownership differs but slot functions remain redundant"
    elif ss >= 2:
        overall = "PARTIAL decomposition; not cleanly specialized"
    else:
        overall = "No strong failure signature; inspect raw panel before claiming success"

    return {
        "overall": overall,
        "ownership": own,
        "functional": fun,
        "uniform_quality_baseline": baseline,
        "diffuse_flags": diffuse,
        "monopoly_flags": monopoly,
        "shared_task_flags": shared,
        "scores": {"diffuse": ds, "monopoly": ms, "shared_task": ss},
        "warning": "Heuristic verdict only; raw metrics and causal ablations are the evidence.",
    }


def add_records(records, category, batch, per_sample, slots, winner_counts, ranks):
    q = slots["qasa_quality"].detach().cpu()
    s = slots["qasa_selected_mask"].detach().cpu()
    wc = winner_counts.detach().cpu()
    for i in range(len(batch.sample_ids)):
        row = {
            "category": category,
            "sample_id": str(batch.sample_ids[i]),
            "modification_text": str(batch.modification_texts[i]),
            "target_id": str(batch.target_ids[i]),
            "qasa_selected_mask": s[i].tolist(),
            "qasa_quality": q[i].float().tolist(),
            "winner_counts": wc[i].float().tolist(),
        }
        for name, x in per_sample.items():
            row[name] = float(x[i].detach().float().item())
        for name, x in ranks.items():
            row[f"rank_{name}"] = int(x[i])
        row["qasa_rank_regret_vs_all"] = row["rank_qasa"] - row["rank_all_slots"]
        records.append(row)


@torch.no_grad()
def run(args):
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint = args.checkpoint or newest_checkpoint(args.outputs_root)
    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    model = build_model(cfg, device)
    load_checkpoint(model, checkpoint)
    model.eval()

    L, K = model.num_slots, model.num_primitives
    ann_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"
    corrections = correction_dicts(ann_root)
    loaders, annotations = build_val_loaders(
        ann_root, args.batch_size, args.num_workers, cfg.val_caption_policy, corrections
    )

    feature_root = args.cache_root / "fashioniq" / "csmcir" / "val"
    retrieval_features, retrieval_idx = load_features(feature_root / "retrieval")
    native_features, native_idx = load_features(feature_root / "native")
    text_cache = load_text_features(feature_root / "text")

    print("Device:", device)
    print("Checkpoint:", checkpoint)
    print("Val retrieval:", tuple(retrieval_features.shape))
    print("Val native:", tuple(native_features.shape))
    print("Val text:", tuple(text_cache.states.shape))
    print(
        f"QASA tau={model.qasa_tau} rho={model.qasa_rho} mu={model.qasa_mu} "
        f"apply_at_eval={model.qasa_apply_at_eval}"
    )

    acc = ScalarAccumulator()
    records = []
    primitive_counts = torch.zeros(L, K, dtype=torch.long)
    execution_counts = torch.zeros(L, dtype=torch.long)
    selected_counts = torch.zeros(L, dtype=torch.long)
    winner_active_counts = torch.zeros(L, dtype=torch.long)

    effect_proto_sum = torch.zeros(L, model.teacher_query_dim)
    effect_proto_count = torch.zeros(L)
    forced_proto_sum = torch.zeros(L, model.query_dim)
    forced_proto_count = torch.zeros(L)

    rstats = defaultdict(
        lambda: defaultdict(
            lambda: {"n": 0, "hit10": 0, "hit50": 0, "rank_sum": 0.0}
        )
    )
    processed_by_category = {}

    for category in CATEGORIES:
        gallery_ids = build_fashioniq_gallery(
            protocol=args.protocol,
            split_root=split_root,
            split="val",
            category=category,
            annotations=annotations[category],
        )
        gallery = get_features_by_ids(
            gallery_ids, retrieval_features, retrieval_idx
        ).to(device=device, dtype=torch.float32)
        gallery_index = {name: i for i, name in enumerate(gallery_ids)}

        processed = 0
        progress = tqdm(loaders[category], desc=f"Diagnose [{category}]", dynamic_ncols=True)
        for batch in progress:
            if args.max_queries_per_category and processed >= args.max_queries_per_category:
                break
            if any(t is None for t in batch.target_ids):
                raise ValueError("Validation target_id missing")

            reference_native = get_features_by_ids(
                batch.reference_ids, native_features, native_idx
            ).to(device=device, dtype=torch.float32)
            reference = reference_native[:, 0, :]
            targets = get_features_by_ids(
                list(batch.target_ids), retrieval_features, retrieval_idx
            ).to(device=device, dtype=torch.float32)

            text, teacher_text, attn_mask, content_mask = get_text_features_by_sample_ids(
                batch.sample_ids, batch.modification_texts, text_cache
            )
            text = text.to(device=device, dtype=torch.float32)
            teacher_text = teacher_text.to(device=device, dtype=torch.float32)
            attn_mask = attn_mask.to(device=device, dtype=torch.bool)
            content_mask = content_mask.to(device=device, dtype=torch.bool)

            slots = model.build_edit_slots(
                reference,
                text,
                attn_mask,
                text_content_mask=content_mask,
                teacher_reference_features=reference_native,
                teacher_text_states=teacher_text,
            )
            z0, ref_state = model.initialize_state(reference)
            q_ref = model.make_query(z0)

            valid = content_mask
            has_content = valid.any(1, keepdim=True)
            qasa_mask = slots["qasa_selected_mask"].bool()
            all_mask = has_content.expand(-1, L).clone()
            win_mask = winner_active_mask(slots["qasa_attention"], valid)

            q_qasa, ex_qasa = execute_query(model, slots, z0, ref_state, qasa_mask)
            q_all, _ = execute_query(model, slots, z0, ref_state, all_mask)
            q_win, _ = execute_query(model, slots, z0, ref_state, win_mask)

            # Ownership.
            qa = slots["qasa_attention"]
            ent = normalized_token_entropy(qa, valid)
            top1, margin = top1_margin(qa, valid)
            wcount, wactive, wbal = winner_stats(qa, valid)
            nvalid = valid.sum(1).to(qa.dtype).clamp_min(1.0)
            mass = (
                slots["slot_masks"] * valid[:, None, :].to(qa.dtype)
            ).sum(-1)
            mass_share = mass / nvalid[:, None]
            dom = mass_share.max(1).values
            own_cos = pairwise_cosine(
                slots["slot_masks"] * valid[:, None, :].to(qa.dtype)
            )
            own_js = pairwise_js_token_maps(slots["slot_masks"], valid)

            acc.add("ownership/token_entropy_norm", ent)
            acc.add("ownership/top1_probability", top1)
            acc.add("ownership/top1_margin", margin)
            acc.add("ownership/winner_active_slot_count", wactive)
            acc.add("ownership/winner_balance_entropy", wbal)
            acc.add("ownership/dominant_mass_share", dom)
            acc.add("ownership/pairwise_cosine", own_cos)
            acc.add("ownership/pairwise_js_token_map", own_js)

            for s in range(L):
                acc.add(f"slot/{s}/mass_share", mass_share[:, s])
                acc.add(f"slot/{s}/winner_count", wcount[:, s])
                acc.add(f"slot/{s}/quality", slots["qasa_quality"][:, s])
                selected_counts[s] += qasa_mask[:, s].sum().cpu()
                winner_active_counts[s] += win_mask[:, s].sum().cpu()

            # QASA.
            selected_k = qasa_mask.sum(1).to(qa.dtype)
            quality_mean = slots["qasa_quality"].mean(1)
            acc.add("qasa/selected_k", selected_k)
            acc.add("qasa/quality_mean", quality_mean)
            acc.add("qasa/final_coverage", slots["qasa_final_coverage"])
            acc.add("qasa/novelty_skip_count", slots["qasa_novelty_skip_count"])
            acc.add("qasa/selected_vs_winner_active_k_gap", selected_k - wactive)
            acc.add(
                "qasa/selected_mask_equals_winner_mask_fraction",
                (qasa_mask == win_mask).all(1).float(),
            )

            # Representation.
            sem_cos = pairwise_cosine(slots["slot_semantics"])
            eff_cos = pairwise_cosine(slots["slot_effects"])
            raw_cos = pairwise_cosine(slots["raw_edit_slots"])
            edit_cos = pairwise_cosine(slots["edit_slots"])
            sel_eff_cos = pairwise_cosine(slots["slot_effects"], qasa_mask)
            eff_rank = effective_rank(slots["slot_effects"])
            raw_rank = effective_rank(slots["raw_edit_slots"])
            eff_cv = coeff_var(slots["slot_effects"].norm(dim=-1))

            acc.add("representation/slot_semantic_pairwise_cosine", sem_cos)
            acc.add("representation/slot_effect_pairwise_cosine", eff_cos)
            acc.add("representation/raw_edit_slot_pairwise_cosine", raw_cos)
            acc.add("representation/edit_slot_pairwise_cosine", edit_cos)
            acc.add("representation/selected_slot_effect_pairwise_cosine", sel_eff_cos)
            acc.add("representation/slot_effect_effective_rank", eff_rank)
            acc.add("representation/raw_edit_slot_effective_rank", raw_rank)
            acc.add("representation/slot_effect_norm_cv", eff_cv)

            effect_proto_sum += F.normalize(
                slots["slot_effects"], dim=-1, eps=1e-8
            ).sum(0).cpu()
            effect_proto_count += slots["slot_effects"].shape[0]

            # Execution.
            vsteps = ex_qasa["trace_valid_mask"]
            state_cos = pairwise_cosine(
                ex_qasa["actual_state_changes"], vsteps, min_norm=1e-8
            )
            acc.add("execution/state_change_pairwise_cosine", state_cos)
            if vsteps.any():
                acc.add("execution/transition_strength_mean", ex_qasa["transition_strengths"][vsteps])
                acc.add("execution/route_confidence_mean", ex_qasa["route_confidences"][vsteps])
                acc.add(
                    "execution/actual_state_change_norm",
                    ex_qasa["actual_state_changes"].norm(dim=-1)[vsteps],
                )

            ts = ex_qasa["trace_slot_ids"].cpu()
            tp = ex_qasa["trace_primitive_ids"].cpu()
            tv = ex_qasa["trace_valid_mask"].cpu()
            for bi in range(ts.shape[0]):
                for step in range(ts.shape[1]):
                    if bool(tv[bi, step]):
                        s, pidx = int(ts[bi, step]), int(tp[bi, step])
                        primitive_counts[s, pidx] += 1
                        execution_counts[s] += 1

            # Causal single-drop (actual QASA policy).
            drop_queries, drop_dist = [], []
            for s in range(L):
                disabled = torch.zeros(
                    reference.shape[0], L, dtype=torch.bool, device=device
                )
                disabled[:, s] = True
                q_drop, _ = execute_query(
                    model, slots, z0, ref_state, qasa_mask, disabled
                )
                drop_queries.append(q_drop)
                drop_dist.append((q_qasa - q_drop).norm(dim=-1))
            drop_queries = torch.stack(drop_queries, 1)
            drop_dist = torch.stack(drop_dist, 1)
            drop_dirs = q_qasa[:, None, :] - drop_queries
            drop_cos = pairwise_cosine(drop_dirs, qasa_mask, min_norm=1e-6)
            acc.add("causal/drop_direction_pairwise_cosine", drop_cos)
            acc.add("causal/drop_query_distance_selected", drop_dist[qasa_mask])

            # Forced-only: probe every slot's latent function, regardless of QASA selection.
            only_queries = []
            for s in range(L):
                mask = torch.zeros(
                    reference.shape[0], L, dtype=torch.bool, device=device
                )
                mask[:, s] = has_content.squeeze(1)
                q_only, _ = execute_query(model, slots, z0, ref_state, mask)
                only_queries.append(q_only)
            only_queries = torch.stack(only_queries, 1)
            forced_effects = only_queries - q_ref[:, None, :]
            forced_cos = pairwise_cosine(forced_effects, all_mask, min_norm=1e-6)
            full_effect = q_qasa - q_ref
            only_to_full = F.cosine_similarity(
                forced_effects,
                full_effect[:, None, :].expand_as(forced_effects),
                dim=-1,
                eps=1e-8,
            )
            acc.add("causal/forced_only_effect_pairwise_cosine", forced_cos)
            acc.add("causal/full_effect_norm", full_effect.norm(dim=-1))
            for s in range(L):
                acc.add(f"slot/{s}/forced_only_effect_norm", forced_effects[:, s].norm(dim=-1))
                acc.add(f"slot/{s}/forced_only_to_full_effect_cosine", only_to_full[:, s])

            forced_proto_sum += F.normalize(
                forced_effects, dim=-1, eps=1e-8
            ).sum(0).cpu()
            forced_proto_count += forced_effects.shape[0]

            # Pair-drop redundancy / non-additivity heuristic.
            pair_red = []
            for i, j in combinations(range(L), 2):
                disabled = torch.zeros(
                    reference.shape[0], L, dtype=torch.bool, device=device
                )
                disabled[:, i] = True
                disabled[:, j] = True
                q_pair, _ = execute_query(
                    model, slots, z0, ref_state, qasa_mask, disabled
                )
                dij = (q_qasa - q_pair).norm(dim=-1)
                di, dj = drop_dist[:, i], drop_dist[:, j]
                red = (di + dj - dij) / (di + dj).clamp_min(1e-8)
                pair_on = qasa_mask[:, i] & qasa_mask[:, j]
                red = torch.where(
                    pair_on, red, torch.full_like(red, float("nan"))
                )
                pair_red.append(red)
            if pair_red:
                acc.add("causal/pair_drop_redundancy_index", torch.stack(pair_red, 1))

            # Target-sim ablations.
            sim_ref = target_similarity(q_ref, targets)
            sim_qasa = target_similarity(q_qasa, targets)
            sim_all = target_similarity(q_all, targets)
            sim_win = target_similarity(q_win, targets)
            acc.add("retrieval_target_sim/qasa", sim_qasa)
            acc.add("retrieval_target_sim/all_slots", sim_all)
            acc.add("retrieval_target_sim/winner_active", sim_win)
            acc.add("retrieval_target_sim/reference_only", sim_ref)
            acc.add("qasa/pruning_regret_target_sim_all_minus_qasa", sim_all - sim_qasa)
            acc.add("qasa/winner_counterfactual_target_sim_minus_qasa", sim_win - sim_qasa)

            for s in range(L):
                drop_loss = sim_qasa - target_similarity(drop_queries[:, s], targets)
                acc.add(
                    f"slot/{s}/drop_target_sim_loss",
                    drop_loss[qasa_mask[:, s]],
                )
                acc.add(
                    f"slot/{s}/forced_only_target_sim_gain_vs_ref",
                    target_similarity(only_queries[:, s], targets) - sim_ref,
                )

            # Gallery ranks for central policies.
            target_indices = torch.tensor(
                [gallery_index[str(t)] for t in batch.target_ids],
                device=device,
                dtype=torch.long,
            )
            rank_map = {
                "qasa": target_ranks(model, q_qasa, gallery, target_indices),
                "all_slots": target_ranks(model, q_all, gallery, target_indices),
                "winner_active": target_ranks(model, q_win, gallery, target_indices),
                "reference_only": target_ranks(model, q_ref, gallery, target_indices),
            }
            for name, ranks in rank_map.items():
                update_retrieval(rstats, name, category, ranks)

            rq, ra, rw = rank_map["qasa"], rank_map["all_slots"], rank_map["winner_active"]
            acc.add("qasa/fraction_all_slots_rank_better", (ra < rq).float())
            acc.add("qasa/fraction_qasa_rank_better_than_all", (rq < ra).float())
            acc.add("qasa/rank_regret_qasa_minus_all", rq.float() - ra.float())
            acc.add("qasa/fraction_winner_active_rank_better", (rw < rq).float())

            add_records(
                records,
                category,
                batch,
                {
                    "ownership_entropy": ent,
                    "ownership_pairwise_cosine": own_cos,
                    "ownership_pairwise_js": own_js,
                    "top1_probability": top1,
                    "top1_margin": margin,
                    "winner_active_count": wactive,
                    "qasa_selected_k": selected_k,
                    "qasa_quality_mean": quality_mean,
                    "qasa_coverage": slots["qasa_final_coverage"],
                    "slot_effect_cosine": eff_cos,
                    "slot_effect_effective_rank": eff_rank,
                    "forced_only_effect_cosine": forced_cos,
                    "drop_direction_cosine": drop_cos,
                    "state_change_cosine": state_cos,
                },
                slots,
                wcount,
                rank_map,
            )

            processed += len(batch.sample_ids)
            progress.set_postfix(n=processed)

        processed_by_category[category] = processed

    metrics = acc.export()

    effect_proto = effect_proto_sum / effect_proto_count[:, None].clamp_min(1)
    forced_proto = forced_proto_sum / forced_proto_count[:, None].clamp_min(1)
    metrics["dataset/slot_effect_prototype_pairwise_cosine"] = float(
        pairwise_cosine(effect_proto[None])[0]
    )
    metrics["dataset/forced_effect_prototype_pairwise_cosine"] = float(
        pairwise_cosine(forced_proto[None])[0]
    )

    router = normalized_mi(primitive_counts)
    router["primitive_counts"] = primitive_counts.tolist()
    router["slot_execution_counts"] = execution_counts.tolist()

    total_n = sum(processed_by_category.values())
    per_slot = {}
    for s in range(L):
        per_slot[str(s)] = {
            "selected_rate": float(selected_counts[s]) / max(total_n, 1),
            "winner_active_rate": float(winner_active_counts[s]) / max(total_n, 1),
            "mass_share": metrics.get(f"slot/{s}/mass_share", float("nan")),
            "winner_count": metrics.get(f"slot/{s}/winner_count", float("nan")),
            "quality": metrics.get(f"slot/{s}/quality", float("nan")),
            "forced_only_effect_norm": metrics.get(f"slot/{s}/forced_only_effect_norm", float("nan")),
            "forced_only_to_full_effect_cosine": metrics.get(
                f"slot/{s}/forced_only_to_full_effect_cosine", float("nan")
            ),
            "drop_target_sim_loss": metrics.get(f"slot/{s}/drop_target_sim_loss", float("nan")),
            "forced_only_target_sim_gain_vs_ref": metrics.get(
                f"slot/{s}/forced_only_target_sim_gain_vs_ref", float("nan")
            ),
        }

    retrieval = summarize_retrieval(rstats)
    verdict = build_verdict(metrics, router, L)
    uniform_k = math.ceil(model.qasa_tau * L - 1e-12)

    def top_by(key, reverse=True):
        xs = [r for r in records if math.isfinite(float(r.get(key, float("nan"))))]
        return sorted(xs, key=lambda r: float(r[key]), reverse=reverse)[: args.top_worst]

    worst = {
        "highest_ownership_entropy": top_by("ownership_entropy"),
        "highest_ownership_similarity": top_by("ownership_pairwise_cosine"),
        "highest_functional_redundancy_forced_only": top_by("forced_only_effect_cosine"),
        "lowest_slot_effect_effective_rank": top_by("slot_effect_effective_rank", reverse=False),
        "largest_qasa_rank_regret_vs_all": sorted(
            records, key=lambda r: r["qasa_rank_regret_vs_all"], reverse=True
        )[: args.top_worst],
    }

    report = {
        "checkpoint": str(checkpoint),
        "config": str(args.config),
        "protocol": args.protocol,
        "processed_by_category": processed_by_category,
        "qasa_context": {
            "num_slots": L,
            "tau": model.qasa_tau,
            "rho": model.qasa_rho,
            "mu": model.qasa_mu,
            "apply_at_eval": model.qasa_apply_at_eval,
            "uniform_quality_baseline": 1.0 / L,
            "uniform_attention_min_slots_to_reach_tau": uniform_k,
        },
        "verdict": verdict,
        "retrieval": retrieval,
        "metrics": metrics,
        "per_slot": per_slot,
        "router": router,
        "worst_examples": worst,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.md_output.write_text(render_markdown(report), encoding="utf-8")

    print("\n" + "=" * 78)
    print("HEADLINE VERDICT")
    print("=" * 78)
    print(verdict["overall"])
    print("Ownership:", verdict["ownership"])
    print("Functional:", verdict["functional"])
    print("\nCore panel:")
    for name in [
        "ownership/token_entropy_norm",
        "ownership/top1_probability",
        "ownership/top1_margin",
        "ownership/winner_active_slot_count",
        "ownership/dominant_mass_share",
        "ownership/pairwise_cosine",
        "ownership/pairwise_js_token_map",
        "qasa/selected_k",
        "qasa/quality_mean",
        "qasa/final_coverage",
        "representation/slot_effect_pairwise_cosine",
        "representation/slot_effect_effective_rank",
        "causal/forced_only_effect_pairwise_cosine",
        "causal/drop_direction_pairwise_cosine",
        "execution/state_change_pairwise_cosine",
    ]:
        print(f"  {name:54s} {fmt(metrics.get(name, float('nan')))}")

    print("\nRetrieval counterfactuals:")
    for name, x in retrieval.items():
        print(
            f"  {name:16s} MR={fmt(x['mean_recall'],2)} "
            f"R10={fmt(x['recall_at_10'],2)} R50={fmt(x['recall_at_50'],2)} "
            f"mean_rank={fmt(x['mean_rank'],2)}"
        )
    print("Router slot<->primitive NMI:", fmt(router["nmi"]))
    print("JSON:", args.json_output)
    print("MD:  ", args.md_output)
    return report


def render_markdown(report):
    m = report["metrics"]
    v = report["verdict"]
    q = report["qasa_context"]
    ret = report["retrieval"]
    ps = report["per_slot"]
    router = report["router"]

    lines = [
        "# TAPER A3.1 QASA / No-NULL — Comprehensive Specialization Diagnosis",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Protocol: `{report['protocol']}`",
        f"- Samples: `{report['processed_by_category']}`",
        "",
        "## 1. Headline verdict",
        "",
        f"**Overall:** {v['overall']}",
        "",
        f"**Ownership:** {v['ownership']}",
        "",
        f"**Functional specialization:** {v['functional']}",
        "",
        "> Automatic verdicts are heuristic. Raw metrics + causal ablations are the evidence.",
        "",
        "## 2. QASA failure-mode baseline",
        "",
        f"- Slots: **{q['num_slots']}**",
        f"- `tau={q['tau']}`, `rho={q['rho']}`, `mu={q['mu']}`",
        f"- `qasa_apply_at_eval={q['apply_at_eval']}`",
        f"- Uniform Quality baseline `1/L`: **{fmt(q['uniform_quality_baseline'])}**",
        (
            "- Perfectly uniform attention needs only "
            f"**{q['uniform_attention_min_slots_to_reach_tau']}** slots to reach `tau`."
        ),
        "",
        (
            "Therefore high QASA coverage by itself is **not evidence of decomposition**. "
            "If entropy is near 1 and quality is near `1/L`, QASA may simply be pruning a "
            "diffuse attention field."
        ),
        "",
        "## 3. Ownership decomposition",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
    ]
    ownership_rows = [
        ("Normalized token entropy", "ownership/token_entropy_norm", "1≈uniform; 0≈sharp"),
        ("Top-1 probability", "ownership/top1_probability", "Higher = sharper winner"),
        ("Top1-top2 margin", "ownership/top1_margin", "Near 0 = ambiguous"),
        ("Winner-active slots", "ownership/winner_active_slot_count", "Slots winning ≥1 token"),
        ("Winner balance entropy", "ownership/winner_balance_entropy", "1≈balanced winners"),
        ("Dominant mass share", "ownership/dominant_mass_share", "Uniform mass baseline=1/L"),
        ("Ownership cosine", "ownership/pairwise_cosine", "1 = same token pattern"),
        ("Token-map JS", "ownership/pairwise_js_token_map", "0 = identical token maps"),
    ]
    for label, key, note in ownership_rows:
        lines.append(f"| {label} | {fmt(m.get(key, float('nan')))} | {note} |")

    lines += [
        "",
        "## 4. QASA behavior",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for label, key in [
        ("Selected K", "qasa/selected_k"),
        ("Quality mean", "qasa/quality_mean"),
        ("Final coverage", "qasa/final_coverage"),
        ("Novelty skips", "qasa/novelty_skip_count"),
        ("Selected K - winner-active K", "qasa/selected_vs_winner_active_k_gap"),
        ("Selected mask == winner mask", "qasa/selected_mask_equals_winner_mask_fraction"),
        ("QASA rank regret vs all slots", "qasa/rank_regret_qasa_minus_all"),
        ("Fraction all-slots rank better", "qasa/fraction_all_slots_rank_better"),
        ("Fraction QASA rank better", "qasa/fraction_qasa_rank_better_than_all"),
    ]:
        lines.append(f"| {label} | {fmt(m.get(key, float('nan')))} |")

    lines += [
        "",
        "## 5. Representation specialization",
        "",
        "| Metric | Value | Same-task warning |",
        "|---|---:|---|",
    ]
    for label, key, note in [
        ("Slot semantic cosine", "representation/slot_semantic_pairwise_cosine", "High = same semantic content"),
        ("Slot effect cosine", "representation/slot_effect_pairwise_cosine", "High = teacher effects align"),
        ("Raw Edit Slot cosine", "representation/raw_edit_slot_pairwise_cosine", "High = redundant representation"),
        ("Edit Slot cosine", "representation/edit_slot_pairwise_cosine", "High = redundant post-activity slots"),
        ("Selected effect cosine", "representation/selected_slot_effect_pairwise_cosine", "High = QASA kept redundant slots"),
        ("Slot-effect effective rank", "representation/slot_effect_effective_rank", "Near 1 = common direction"),
        ("Raw-slot effective rank", "representation/raw_edit_slot_effective_rank", "Near 1 = low-dimensional"),
        ("Effect norm CV", "representation/slot_effect_norm_cv", "Near 0 = equal magnitude"),
        ("Dataset effect-prototype cosine", "dataset/slot_effect_prototype_pairwise_cosine", "High = same dataset-level role"),
    ]:
        lines.append(f"| {label} | {fmt(m.get(key, float('nan')))} | {note} |")

    lines += [
        "",
        "## 6. Causal test — are the 4 Edit Slots doing the same job?",
        "",
        "This section is the strongest evidence because it intervenes on execution rather than only comparing representations.",
        "",
        "1. **Drop-one:** remove each currently selected slot; compare query-change directions.",
        "2. **Forced-only:** force each slot to execute alone; compare its edit direction.",
        "3. **Pair-drop:** remove pairs and test non-additivity/redundancy.",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
    ]
    for label, key, note in [
        ("Drop contribution direction cosine", "causal/drop_direction_pairwise_cosine", "High = different drops perturb query similarly"),
        ("Forced-only effect cosine", "causal/forced_only_effect_pairwise_cosine", "High = different slots perform similar edit alone"),
        ("Dataset forced-effect prototype cosine", "dataset/forced_effect_prototype_pairwise_cosine", "High = same causal role across dataset"),
        ("Pair-drop redundancy index", "causal/pair_drop_redundancy_index", "Positive = overlapping/non-additive contribution"),
        ("Full edit effect norm", "causal/full_effect_norm", "Distance from reference-only query"),
    ]:
        lines.append(f"| {label} | {fmt(m.get(key, float('nan')))} | {note} |")

    lines += [
        "",
        "## 7. Execution / primitive routing",
        "",
        f"- Slot↔primitive MI: **{fmt(router.get('mi', float('nan')))}**",
        f"- Slot↔primitive NMI: **{fmt(router.get('nmi', float('nan')))}**",
        f"- State-change cosine: **{fmt(m.get('execution/state_change_pairwise_cosine', float('nan')))}**",
        f"- Route confidence: **{fmt(m.get('execution/route_confidence_mean', float('nan')))}**",
        f"- Transition strength: **{fmt(m.get('execution/transition_strength_mean', float('nan')))}**",
        "",
        "Primitive counts by slot:",
        "",
        "```text",
    ]
    for i, row in enumerate(router["primitive_counts"]):
        lines.append(f"S{i}: {row}")
    lines += ["```", ""]

    lines += [
        "## 8. Retrieval counterfactuals",
        "",
        "| Variant | R@10 | R@50 | Mean Recall | Mean Rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, x in ret.items():
        lines.append(
            f"| {name} | {fmt(x['recall_at_10'],2)} | {fmt(x['recall_at_50'],2)} "
            f"| {fmt(x['mean_recall'],2)} | {fmt(x['mean_rank'],2)} |"
        )
    lines += [
        "",
        "- `qasa`: actual current A3.1 policy.",
        "- `all_slots`: diagnostic counterfactual; directly tests whether QASA pruning hurts.",
        "- `winner_active`: diagnostic preview of the proposed next experiment only.",
        "- `reference_only`: no execution baseline.",
        "",
        "## 9. Per-slot panel",
        "",
        (
            "| Slot | Selected rate | Winner-active rate | Mass share | Winner count | Q | "
            "Only-effect norm | Only→full cos | Drop target loss | Only target gain |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s, x in ps.items():
        lines.append(
            f"| S{s} | {fmt(x['selected_rate'])} | {fmt(x['winner_active_rate'])} | "
            f"{fmt(x['mass_share'])} | {fmt(x['winner_count'])} | {fmt(x['quality'])} | "
            f"{fmt(x['forced_only_effect_norm'])} | "
            f"{fmt(x['forced_only_to_full_effect_cosine'])} | "
            f"{fmt(x['drop_target_sim_loss'])} | "
            f"{fmt(x['forced_only_target_sim_gain_vs_ref'])} |"
        )

    lines += ["", "## 10. Red-team flags", "", "### Diffuse / symmetric ownership"]
    for k, val in v["diffuse_flags"].items():
        lines.append(f"- {'FLAG' if val else 'pass'} `{k}` = `{val}`")
    lines += ["", "### Monopoly collapse"]
    for k, val in v["monopoly_flags"].items():
        lines.append(f"- {'FLAG' if val else 'pass'} `{k}` = `{val}`")
    lines += ["", "### Four-slots-same-task / functional redundancy"]
    for k, val in v["shared_task_flags"].items():
        lines.append(f"- {'FLAG' if val else 'pass'} `{k}` = `{val}`")

    lines += [
        "",
        "## 11. Worst-case samples",
        "",
        (
            "Full top-N records are in the JSON. Below are the first five from each view. "
            "Use them for manual trace inspection."
        ),
        "",
    ]
    for group, rows in report["worst_examples"].items():
        lines += [f"### {group}", ""]
        for row in rows[:5]:
            lines.append(
                f"- `{row['sample_id']}` — {row['modification_text']} "
                f"| H={fmt(row.get('ownership_entropy', float('nan')))} "
                f"| own_cos={fmt(row.get('ownership_pairwise_cosine', float('nan')))} "
                f"| effect_cos={fmt(row.get('slot_effect_cosine', float('nan')))} "
                f"| forced_cos={fmt(row.get('forced_only_effect_cosine', float('nan')))} "
                f"| rank QASA/all={row['rank_qasa']}/{row['rank_all_slots']}"
            )
        lines.append("")

    lines += [
        "## 12. Rule for claiming real specialization",
        "",
        "Do **not** claim success from one metric. A convincing decomposition should agree across:",
        "",
        "1. **Ownership:** multiple real winners, low ambiguity, non-identical token maps.",
        "2. **Representation:** slot effects are not nearly collinear; effective rank meaningfully >1.",
        "3. **Causality:** forced-only slots create different edit directions; drop-one effects are distinct.",
        "4. **Execution:** state changes / primitive routing are not effectively identical.",
        "5. **QASA:** high coverage comes with sharp ownership and quality above the uniform baseline.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run(parse_args())