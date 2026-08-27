from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from cache.features import (
    get_features_by_ids,
    get_text_features_by_sample_ids,
    load_feature_manifest,
    load_features,
    load_text_features,
    validate_feature_manifest,
    validate_text_cache_subdir,
)
from datasets.fashioniq import validate_correction_policy
from evaluate_qasa_inference import (
    CATEGORIES,
    build_model,
    build_val_loaders,
    load_checkpoint,
    load_correction_dicts,
)
from evaluation.fashioniq import build_fashioniq_gallery

ROUTING_EPS = 1e-6
CAPACITY_BINDING_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FG-CLIP2 TAPER forensic: retrieval, slot-drop, only-slot, routing and R4 spillover."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    p.add_argument("--cache-root", type=Path, default=Path("features"))
    p.add_argument("--config", type=Path, default=Path("conf/experiment/taper_e2e.yaml"))
    p.add_argument("--protocol", choices=("fashioniq_original", "fashioniq_val"), default="fashioniq_original")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-queries-per-category", type=int, default=0)
    p.add_argument("--correction-policy", default=None)
    p.add_argument("--text-cache-subdir", default=None)

    p.add_argument("--routing-mode", choices=("entmax15", "qisca"), default=None)
    p.add_argument("--r4-theta", type=float, default=None)
    p.add_argument("--r4-lambda", dest="r4_lambda", type=float, default=None)
    cap = p.add_mutually_exclusive_group()
    cap.add_argument("--r4-capacity-enabled", dest="r4_capacity_enabled", action="store_true")
    cap.add_argument("--r4-capacity-disabled", dest="r4_capacity_enabled", action="store_false")
    p.set_defaults(r4_capacity_enabled=None)
    p.add_argument("--r4-candidate-mode", choices=("qasa_selected", "all_real_slots"), default=None)
    p.add_argument("--r4-slot-capacity", type=float, default=None)
    p.add_argument("--r4-solver-iters", type=int, default=None)
    p.add_argument("--json-output", type=Path, default=Path("reports/taper_r4b_forensic.json"))
    return p.parse_args()


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    m = cfg.model
    for attr, value in (
        ("routing_mode", args.routing_mode),
        ("r4_theta", args.r4_theta),
        ("r4_lambda", args.r4_lambda),
        ("r4_capacity_enabled", args.r4_capacity_enabled),
        ("r4_candidate_mode", args.r4_candidate_mode),
        ("r4_slot_capacity", args.r4_slot_capacity),
        ("r4_solver_iters", args.r4_solver_iters),
    ):
        if value is not None:
            setattr(m, attr, value)


class RetrievalAccumulator:
    def __init__(self, variants: list[str]) -> None:
        self.stats = {
            v: {c: {"n": 0, "hit10": 0, "hit50": 0} for c in CATEGORIES}
            for v in variants
        }

    def update(self, variant: str, category: str, scores: torch.Tensor, target_indices: torch.Tensor) -> None:
        top10 = scores.topk(10, dim=1, largest=True, sorted=False).indices
        top50 = scores.topk(50, dim=1, largest=True, sorted=False).indices
        target = target_indices[:, None]
        s = self.stats[variant][category]
        s["n"] += scores.shape[0]
        s["hit10"] += int(top10.eq(target).any(dim=1).sum().item())
        s["hit50"] += int(top50.eq(target).any(dim=1).sum().item())

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for variant, cats in self.stats.items():
            per_cat = {}
            for category, s in cats.items():
                if s["n"] == 0:
                    continue
                r10 = 100.0 * s["hit10"] / s["n"]
                r50 = 100.0 * s["hit50"] / s["n"]
                per_cat[category] = {
                    "n": s["n"],
                    "recall_at_10": r10,
                    "recall_at_50": r50,
                    "mean_recall": 0.5 * (r10 + r50),
                }
            r10 = sum(x["recall_at_10"] for x in per_cat.values()) / len(per_cat)
            r50 = sum(x["recall_at_50"] for x in per_cat.values()) / len(per_cat)
            out[variant] = {
                "recall_at_10": r10,
                "recall_at_50": r50,
                "mean_recall": 0.5 * (r10 + r50),
                "categories": per_cat,
            }
        return out


class ForensicAccumulator:
    def __init__(self, num_slots: int, capacity: float | None) -> None:
        self.L = num_slots
        self.capacity = capacity
        self.samples = 0
        self.valid_tokens = 0
        self.route_mass = torch.zeros(self.L, dtype=torch.float64)
        self.route_active = torch.zeros(self.L, dtype=torch.float64)
        self.exec_active = torch.zeros(self.L, dtype=torch.float64)
        self.qasa_active = torch.zeros(self.L, dtype=torch.float64)
        self.soft_dom = torch.zeros(self.L, dtype=torch.float64)
        self.support_sum = torch.zeros(self.L, dtype=torch.float64)
        self.support_frac_sum = torch.zeros(self.L, dtype=torch.float64)
        self.active_den = torch.zeros(self.L, dtype=torch.float64)
        self.state_change_sum = torch.zeros(self.L, dtype=torch.float64)
        self.alpha_sum = torch.zeros(self.L, dtype=torch.float64)
        self.exec_den = torch.zeros(self.L, dtype=torch.float64)
        self.drop_cos_sum = torch.zeros(self.L, dtype=torch.float64)

        self.sum_route_k = self.sum_exec_k = self.sum_qasa_k = 0.0
        self.sum_non_qasa_active = self.sum_exec_non_qasa = 0.0
        self.sum_non_qasa_mass = self.sum_non_qasa_fraction = 0.0
        self.token_mass_sum = self.unassigned_sum = self.fully_unassigned = 0.0
        self.overlap_sum = self.overlap_pairs = 0.0
        self.cap_util_sum = self.cap_bind_sum = self.cap_candidate_count = 0.0

    def update(self, output: dict[str, torch.Tensor], valid: torch.Tensor, drop_queries: list[torch.Tensor]) -> None:
        b = valid.shape[0]
        self.samples += b
        self.valid_tokens += int(valid.sum().item())

        routing = output["routing_masks"]
        mass = output["routing_slot_mass"]
        active = output["routing_active_mask"].bool()
        execution = output["hard_active_slot_mask"].bool()
        qasa = output["qasa_selected_mask"].bool()

        self.route_mass += mass.double().sum(0).cpu()
        self.route_active += active.double().sum(0).cpu()
        self.exec_active += execution.double().sum(0).cpu()
        self.qasa_active += qasa.double().sum(0).cpu()
        dom = output["slot_mass"].argmax(1).cpu()
        self.soft_dom += torch.bincount(dom, minlength=self.L).double()

        support = ((routing > ROUTING_EPS) & valid[:, None, :]).sum(-1)
        valid_per_sample = valid.sum(1).clamp_min(1)
        support_frac = support.float() / valid_per_sample[:, None].float()
        for s in range(self.L):
            a = active[:, s]
            n = int(a.sum().item())
            if n:
                self.support_sum[s] += support[a, s].double().sum().cpu()
                self.support_frac_sum[s] += support_frac[a, s].double().sum().cpu()
                self.active_den[s] += n

        self.sum_route_k += float(active.float().sum().item())
        self.sum_exec_k += float(execution.float().sum().item())
        self.sum_qasa_k += float(qasa.float().sum().item())

        non_qasa = ~qasa
        non_qasa_active = active & non_qasa
        exec_non_qasa = execution & non_qasa
        non_qasa_mass = (mass * non_qasa.to(mass.dtype)).sum(1)
        total_mass = mass.sum(1)
        self.sum_non_qasa_active += float(non_qasa_active.float().sum().item())
        self.sum_exec_non_qasa += float(exec_non_qasa.float().sum().item())
        self.sum_non_qasa_mass += float(non_qasa_mass.sum().item())
        self.sum_non_qasa_fraction += float((non_qasa_mass / total_mass.clamp_min(1e-12)).sum().item())

        token_mass = routing.sum(1)
        vf = valid.to(token_mass.dtype)
        self.token_mass_sum += float((token_mass * vf).sum().item())
        self.unassigned_sum += float(((1.0 - token_mass).clamp(0, 1) * vf).sum().item())
        self.fully_unassigned += float(((token_mass <= ROUTING_EPS) & valid).sum().item())

        support_b = (routing > ROUTING_EPS) & valid[:, None, :]
        sf = support_b.float()
        inter = sf @ sf.transpose(1, 2)
        size = sf.sum(-1)
        union = size[:, :, None] + size[:, None, :] - inter
        upper = torch.triu(torch.ones(self.L, self.L, dtype=torch.bool, device=routing.device), diagonal=1)
        pairs = active[:, :, None] & active[:, None, :] & upper[None]
        if pairs.any():
            jac = inter / union.clamp_min(1.0)
            self.overlap_sum += float(jac[pairs].sum().item())
            self.overlap_pairs += float(pairs.sum().item())

        if self.capacity is not None:
            cand = output["routing_candidate_mask"].bool()
            count = float(cand.sum().item())
            if count:
                util = mass / self.capacity
                bind = (mass - self.capacity).abs() <= CAPACITY_BINDING_TOL
                self.cap_util_sum += float((util * cand.to(util.dtype)).sum().item())
                self.cap_bind_sum += float((bind & cand).sum().item())
                self.cap_candidate_count += count

        slot_to_step = output["slot_to_step"]
        changes = output["actual_state_changes"]
        alphas = output["transition_strengths"]
        for s in range(self.L):
            step = slot_to_step[:, s]
            ex = step >= 0
            n = int(ex.sum().item())
            if n:
                ids = torch.arange(b, device=step.device)[ex]
                st = step[ex]
                self.state_change_sum[s] += changes[ids, st].norm(dim=-1).double().sum().cpu()
                self.alpha_sum[s] += alphas[ids, st].double().sum().cpu()
                self.exec_den[s] += n

        full = output["q0"]
        for s, q in enumerate(drop_queries):
            self.drop_cos_sum[s] += (1.0 - F.cosine_similarity(full, q, dim=-1)).double().sum().cpu()

    def finalize(self) -> dict[str, Any]:
        n = max(self.samples, 1)
        vt = max(self.valid_tokens, 1)
        per_slot = {}
        for s in range(self.L):
            ad = max(float(self.active_den[s]), 1.0)
            ed = max(float(self.exec_den[s]), 1.0)
            per_slot[str(s)] = {
                "routing_mass_mean": float(self.route_mass[s] / n),
                "routing_active_frequency": float(self.route_active[s] / n),
                "execution_active_frequency": float(self.exec_active[s] / n),
                "qasa_selected_frequency": float(self.qasa_active[s] / n),
                "soft_dominant_frequency": float(self.soft_dom[s] / n),
                "routing_support_mean_when_active": float(self.support_sum[s] / ad),
                "routing_support_fraction_when_active": float(self.support_frac_sum[s] / ad),
                "state_change_norm_mean_when_executed": float(self.state_change_sum[s] / ed),
                "transition_strength_mean_when_executed": float(self.alpha_sum[s] / ed),
                "drop_query_cosine_change_mean": float(self.drop_cos_sum[s] / n),
            }
        out = {
            "samples": self.samples,
            "valid_tokens": self.valid_tokens,
            "qasa_selected_slot_count": self.sum_qasa_k / n,
            "routing_active_slot_count": self.sum_route_k / n,
            "execution_active_slot_count": self.sum_exec_k / n,
            "routing_non_qasa_active_slot_count": self.sum_non_qasa_active / n,
            "execution_non_qasa_active_slot_count": self.sum_exec_non_qasa / n,
            "routing_non_qasa_mass_mean": self.sum_non_qasa_mass / n,
            "routing_non_qasa_mass_fraction": self.sum_non_qasa_fraction / n,
            "routing_token_mass_mean": self.token_mass_sum / vt,
            "routing_unassigned_mass_mean": self.unassigned_sum / vt,
            "routing_fully_unassigned_token_fraction": self.fully_unassigned / vt,
            "routing_support_overlap_mean": self.overlap_sum / self.overlap_pairs if self.overlap_pairs else None,
            "per_slot": per_slot,
        }
        if self.capacity is not None and self.cap_candidate_count:
            out["routing_capacity_utilization_mean"] = self.cap_util_sum / self.cap_candidate_count
            out["routing_capacity_binding_fraction"] = self.cap_bind_sum / self.cap_candidate_count
        else:
            out["routing_capacity_utilization_mean"] = None
            out["routing_capacity_binding_fraction"] = None
        return out


def make_counterfactual_queries(model, output: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    b, L = output["execution_selected_mask"].shape
    device = output["edit_slots"].device
    drops, onlys = [], []
    for s in range(L):
        disabled = torch.zeros(b, L, dtype=torch.bool, device=device)
        disabled[:, s] = True
        ex = model.execute(output["edit_slots"], output["execution_selected_mask"], output["z0"], output["reference_state"], disabled_slots=disabled)
        drops.append(model.make_query(ex["final_state"]))

        disabled = torch.ones(b, L, dtype=torch.bool, device=device)
        disabled[:, s] = False
        ex = model.execute(output["edit_slots"], output["execution_selected_mask"], output["z0"], output["reference_state"], disabled_slots=disabled)
        onlys.append(model.make_query(ex["final_state"]))
    return drops, onlys


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args)

    correction_policy = validate_correction_policy(args.correction_policy or str(cfg.correction_policy))
    text_subdir = validate_text_cache_subdir(args.text_cache_subdir or str(cfg.text_cache_subdir), correction_policy)
    annotation_root = args.dataset_root / "captions"
    split_root = args.dataset_root / "image_splits"
    correction_dicts = load_correction_dicts(annotation_root) if correction_policy == "fashioniq" else None
    loaders = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=cfg.val_caption_policy,
        correction_dicts=correction_dicts,
    )
    annotations = {c: loaders[c].dataset.annotations for c in CATEGORIES}

    root = args.cache_root / "fashioniq" / "fgclip2-large" / "val"
    image_manifest = load_feature_manifest(root / "images")
    validate_feature_manifest(image_manifest, model_id=str(cfg.backbone.model_id), revision=str(cfg.backbone.revision), cache_name="val/images")
    images, image_idx = load_features(root / "images")
    text_cache = load_text_features(root / text_subdir)
    validate_feature_manifest(text_cache.manifest, model_id=str(cfg.backbone.model_id), revision=str(cfg.backbone.revision), cache_name=f"val/{text_subdir}", correction_policy=correction_policy)

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    capacity = float(model.r4_slot_capacity) if model.routing_mode == "qisca" and model.r4_capacity_enabled else None
    forensic = ForensicAccumulator(model.num_slots, capacity)
    variants = ["full", "reference_only"] + [f"drop_{s}" for s in range(model.num_slots)] + [f"only_{s}" for s in range(model.num_slots)]
    retrieval = RetrievalAccumulator(variants)

    print("Checkpoint:", args.checkpoint)
    print("Routing mode:", model.routing_mode)
    if model.routing_mode == "qisca":
        print("R4 theta:", model.r4_theta)
        print("R4 lambda:", model.r4_lambda)
        print("R4 capacity enabled:", model.r4_capacity_enabled)
        print("R4 candidate mode:", model.r4_candidate_mode)
        print("R4 slot capacity:", model.r4_slot_capacity)

    for category in CATEGORIES:
        gallery_ids = build_fashioniq_gallery(
            protocol=args.protocol,
            split_root=split_root,
            split="val",
            category=category,
            annotations=annotations[category],
        )
        gallery = get_features_by_ids(gallery_ids, images, image_idx).to(device=device, dtype=torch.float32)
        gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
        processed = 0

        for batch in tqdm(loaders[category], desc=f"Forensic [{category}]", dynamic_ncols=True):
            if args.max_queries_per_category and processed >= args.max_queries_per_category:
                break
            take = len(batch.sample_ids)
            if args.max_queries_per_category:
                take = min(take, args.max_queries_per_category - processed)
            if take <= 0:
                break

            ref_cache = get_features_by_ids(list(batch.reference_ids)[:take], images, image_idx).to(device=device, dtype=torch.float32)
            reference = ref_cache[:, 0, :]
            sample_ids = list(batch.sample_ids)[:take]
            mods = list(batch.modification_texts)[:take]
            text, attention, content = get_text_features_by_sample_ids(sample_ids, mods, text_cache)
            text = text.to(device=device, dtype=torch.float32)
            attention = attention.to(device=device, dtype=torch.bool)
            content = content.to(device=device, dtype=torch.bool)

            out = model.forward(reference, text, attention, text_content_mask=content)
            drops, onlys = make_counterfactual_queries(model, out)

            target_ids = list(batch.target_ids)[:take]
            target_idx = torch.tensor([gallery_index[t] for t in target_ids], device=device, dtype=torch.long)
            query_variants = {"full": out["q0"], "reference_only": out["q_reference_only"]}
            query_variants.update({f"drop_{s}": q for s, q in enumerate(drops)})
            query_variants.update({f"only_{s}": q for s, q in enumerate(onlys)})
            for variant, query in query_variants.items():
                scores = model._retrieval_scores(query, gallery)
                retrieval.update(variant, category, scores, target_idx)

            forensic.update(out, attention & content, drops)
            processed += take

    retrieval_result = retrieval.finalize()
    diag = forensic.finalize()
    full_mr = retrieval_result["full"]["mean_recall"]
    ref_mr = retrieval_result["reference_only"]["mean_recall"]

    functional = {}
    for s in range(model.num_slots):
        drop_mr = retrieval_result[f"drop_{s}"]["mean_recall"]
        only_mr = retrieval_result[f"only_{s}"]["mean_recall"]
        functional[str(s)] = {
            "drop_mean_recall": drop_mr,
            "drop_delta_vs_full": drop_mr - full_mr,
            "utility_full_minus_drop": full_mr - drop_mr,
            "only_slot_mean_recall": only_mr,
            "only_slot_gain_over_reference": only_mr - ref_mr,
            **diag["per_slot"][str(s)],
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "protocol": args.protocol,
        "runtime": {
            "routing_mode": model.routing_mode,
            "r4_theta": model.r4_theta,
            "r4_lambda": model.r4_lambda,
            "r4_capacity_enabled": model.r4_capacity_enabled,
            "r4_candidate_mode": model.r4_candidate_mode,
            "r4_slot_capacity": model.r4_slot_capacity,
            "r4_solver_iters": model.r4_solver_iters,
        },
        "retrieval": retrieval_result,
        "diagnostics": {k: v for k, v in diag.items() if k != "per_slot"},
        "per_slot_functional": functional,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    print("\n=== RETRIEVAL ===")
    print(f"FULL           R@10={retrieval_result['full']['recall_at_10']:.2f} R@50={retrieval_result['full']['recall_at_50']:.2f} MR={full_mr:.2f}")
    print(f"REFERENCE ONLY R@10={retrieval_result['reference_only']['recall_at_10']:.2f} R@50={retrieval_result['reference_only']['recall_at_50']:.2f} MR={ref_mr:.2f}")

    print("\n=== GLOBAL ROUTING / EXECUTION ===")
    for key, value in report["diagnostics"].items():
        print(f"{key}: {value}")

    print("\n=== PER-SLOT FUNCTIONAL FORENSIC ===")
    print("slot drop_MR utility only_MR only-ref route_f exec_f qasa_f soft_dom mass support_f drop_cos state_change")
    for s in range(model.num_slots):
        x = functional[str(s)]
        print(
            f"S{s} {x['drop_mean_recall']:.3f} {x['utility_full_minus_drop']:+.3f} "
            f"{x['only_slot_mean_recall']:.3f} {x['only_slot_gain_over_reference']:+.3f} "
            f"{x['routing_active_frequency']:.3f} {x['execution_active_frequency']:.3f} "
            f"{x['qasa_selected_frequency']:.3f} {x['soft_dominant_frequency']:.3f} "
            f"{x['routing_mass_mean']:.3f} {x['routing_support_fraction_when_active']:.3f} "
            f"{x['drop_query_cosine_change_mean']:.5f} {x['state_change_norm_mean_when_executed']:.5f}"
        )

    print("\nSaved:", args.json_output)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()