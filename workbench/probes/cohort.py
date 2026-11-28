"""Exact model cohort manifest, environment specs, and checkpoint resolution.

Resolves the 15-model probe specification against the actual repository state
at `research/cir-failure-workbench` (commit 4079669+):
- Core 15-model benchmark cohort (`run_benchmark_suite.py` MODEL_QUEUE[:15]):
    clip4cir, encoder, tgcir, clvcnet, dcnet, sprc, mtst_ptha,
    hint, intent, habit, conesep, airknow, csmcir, tme, limn
- Plus #16 `qure` (ICML 2025, added Nov 24, complete on EDIR, PinPoint, FashionIQ)
- FashionIQ in-domain author checkpoints (`FIQ_CHECKPOINTS` = exact 15 models:
    all 16 above EXCEPT `mtst_ptha`, which only has a CIRR-transfer checkpoint)
- Ambiguous / excluded registry entries explicitly documented:
    - `pair`: checkpoints `pair-B1.pt` / `pair-B2.pt` exist, never in EDIR/PinPoint/DuckDB
    - `combiner_rn50x4_noft`: frozen-CLIP ablation of `clip4cir_rn50x4_fullft` on FashionIQ only
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path("/home/heheboiz/data/cir")
ENVS_DIR = REPO_ROOT / "workbench" / "envs"
CIR_ENV_PY = REPO_ROOT / "workbench" / "artifacts" / "environments" / "cir-005ef4e1b02999a5" / "bin" / "python"
BENCH_RUNS = REPO_ROOT / "workbench" / "artifacts" / "benchmark_runs"
FIQ_ROOT = REPO_ROOT / "workbench" / "artifacts" / "checkpoint_selection" / "20261106T002617+0700"
QURE_CKPT_ROOT = REPO_ROOT / "workbench" / "artifacts" / "qure_checkpoints"
CIRR_ROOT = REPO_ROOT / "workbench" / "artifacts" / "cirr_checkpoints"

SPLIT_TAG = "70_10_20_seed42"


@dataclass
class ModelSpec:
    key: str
    run_id: str
    family: str           # classic | clip | blip2_vitl | blip2_evag
    scorer_type: str      # single_vec_cosine | multivec_maxsim_32x256 | dcnet_7expert_logsoftmax
    embed_dim: int
    env_group: str        # shared virtualenv group for safe frozen-ViT batching
    python: str
    pythonpath: str
    in_original_15_bench: bool
    in_fiq_15_indomain: bool
    result_field: str
    supports_res_interp: bool
    native_resolution: int
    notes: str = ""


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        key="clip4cir",
        run_id="clip4cir_rn50x4",
        family="clip",
        scorer_type="single_vec_cosine",
        embed_dim=640,
        env_group="clip4cir",
        python=str(ENVS_DIR / "clip4cir310-sm120-pep610/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CLIP4Cir/src",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_clip4cir_test",
        supports_res_interp=False,  # RN50x4 AttentionPool2d fixed 9x9 spatial grid -> 288x288
        native_resolution=288,
        notes="CLIP RN50x4 + Combiner MLP; native resolution 288x288.",
    ),
    ModelSpec(
        key="encoder",
        run_id="encoder_vitb32",
        family="clip",
        scorer_type="single_vec_cosine",
        embed_dim=512,
        env_group="encoder",
        python=str(ENVS_DIR / "encoder310-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/ENCODER",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_encoder_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="OpenCLIP ViT-B/32 + Entity/Action BindingDecoder.",
    ),
    ModelSpec(
        key="tgcir",
        run_id="tgcir_vitb16",
        family="clip",
        scorer_type="single_vec_cosine",
        embed_dim=512,
        env_group="tgcir",
        python=str(ENVS_DIR / "tgcir39-la/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/TGCIR",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_tgcir_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="CLIP ViT-B/16 + Target-Guided composition.",
    ),
    ModelSpec(
        key="limn",
        run_id="limn_vitl14",
        family="clip",
        scorer_type="single_vec_cosine",
        embed_dim=768,
        env_group="limn",
        python=str(ENVS_DIR / "limn310-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/LIMN/LIMN",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_limn_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="OpenCLIP ViT-L/14 DataComp-XL + LIMN prefix tokens (3 category heads on FIQ).",
    ),
    ModelSpec(
        key="clvcnet",
        run_id="clvcnet_resnet50",
        family="classic",
        scorer_type="single_vec_cosine",
        embed_dim=1024,
        env_group="clvcnet",
        python=str(ENVS_DIR / "clvc39-la/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CLVCNet",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_clvcnet_test",
        supports_res_interp=True,  # ResNet50 FCN + GeM supports arbitrary spatial sizes!
        native_resolution=224,
        notes="Dual ResNet50 + LSTM local/global mutual learning (concat 512+512=1024).",
    ),
    ModelSpec(
        key="dcnet",
        run_id="dcnet_resnet50",
        family="classic",
        scorer_type="dcnet_7expert_logsoftmax",
        embed_dim=1024,
        env_group="dcnet",
        python=str(ENVS_DIR / "dcnet310-sm120/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/DCNet",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_dcnet_test",
        supports_res_interp=False,  # Fixed spatial pool grids for spatial0..spatial4
        native_resolution=224,
        notes="ResNet50 + GloVe 300d + 7-expert TIRG/FusDiff full-gallery log_softmax scorer.",
    ),
    ModelSpec(
        key="qure",
        run_id="qure_vitl",
        family="blip2_vitl",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="qure",
        python=str(ENVS_DIR / "qure310-sm120/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/QuRe",
        in_original_15_bench=False,  # #16 added Nov 24
        in_fiq_15_indomain=True,
        result_field="final_qure_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="BLIP-2 ViT-L + Q-Former pairwise KL reward + hard-negative mining (ICML 2025).",
    ),
    ModelSpec(
        key="sprc",
        run_id="sprc_pretrain",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="sprc",
        python=str(ENVS_DIR / "sprc310-sm120/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/SPRC/src",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_sprc_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="BLIP-2 two-pass Q-Former alignment prompt (EVA-ViT-g on EDIR/PP; ViT-L on FIQ author ckpt).",
    ),
    ModelSpec(
        key="mtst_ptha",
        run_id="mtst_ptha_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="cir_evag",
        python=str(CIR_ENV_PY),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/MTST_PTHA/MTST_PTHA/src",
        in_original_15_bench=True,
        in_fiq_15_indomain=False,  # CIRR transfer only on FashionIQ
        result_field="final_mtst_ptha_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="BLIP-2 EVA-ViT-g two-pass Q-Former cat mean-33; CIRR-transfer only on FIQ.",
    ),
    ModelSpec(
        key="csmcir",
        run_id="csmcir_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="cir_evag",
        python=str(CIR_ENV_PY),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CSMCIR/src",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_csmcir_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="BLIP-2 EVA-ViT-g single-pass Q-Former + memory bank.",
    ),
    ModelSpec(
        key="hint",
        run_id="hint_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="ilearn_evag",
        python=str(ENVS_DIR / "ilearn39-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/HINT",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_hint_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="iLearn BLIP-2 EVA-ViT-g HINT.",
    ),
    ModelSpec(
        key="habit",
        run_id="habit_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="ilearn_evag",
        python=str(ENVS_DIR / "ilearn39-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/HABIT",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_habit_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="iLearn BLIP-2 EVA-ViT-g HABIT (FIQ author ckpt trained with 20% synthetic noise).",
    ),
    ModelSpec(
        key="intent",
        run_id="intent_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="ilearn_evag",
        python=str(ENVS_DIR / "ilearn39-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/INTENT",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_intent_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="iLearn BLIP-2 EVA-ViT-g INTENT (FIQ author ckpt trained with 20% synthetic noise).",
    ),
    ModelSpec(
        key="conesep",
        run_id="conesep_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="ilearn_evag",
        python=str(ENVS_DIR / "ilearn39-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/ConeSep",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_conesep_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="iLearn BLIP-2 EVA-ViT-g ConeSep (FIQ author ckpt trained with 20% synthetic noise).",
    ),
    ModelSpec(
        key="airknow",
        run_id="airknow_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="airknow",
        python=str(ENVS_DIR / "airknow310-la/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/AirKnow",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_airknow_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="iLearn BLIP-2 EVA-ViT-g AirKnow (FIQ author ckpt trained with 50% synthetic noise).",
    ),
    ModelSpec(
        key="tme",
        run_id="tme_blip2",
        family="blip2_evag",
        scorer_type="multivec_maxsim_32x256",
        embed_dim=256,
        env_group="tme",
        python=str(ENVS_DIR / "tme-adapted/bin/python"),
        pythonpath=f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/TME/src",
        in_original_15_bench=True,
        in_fiq_15_indomain=True,
        result_field="final_tme_test",
        supports_res_interp=True,
        native_resolution=224,
        notes="BLIP-2 EVA-ViT-g TME (image diff features + 3-stage GMM warmup).",
    ),
]

MODEL_BY_KEY: Dict[str, ModelSpec] = {m.key: m for m in MODEL_SPECS}

# Exact FashionIQ author checkpoint paths (15 in-domain + 1 mtst_ptha CIRR transfer)
FIQ_CKPT_MAP: Dict[str, Dict[str, Any]] = {
    "encoder": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "cohort/downloads/encoder/fashioniq.pt")]},
    "tgcir": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "downloads/tgcir/x/checkpoints/Fashioniq.pt")]},
    "clip4cir": {
        "variant": "fiq_clean",
        "protocol": "in_domain",
        "ckpts": [
            str(FIQ_ROOT / "checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt"),
            str(FIQ_ROOT / "checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt"),
        ],
    },
    "limn": {
        "variant": "fiq_3head_per_cat",
        "protocol": "in_domain",
        "per_category": {
            "dress": {"dt_id": 0, "ckpt": str(FIQ_ROOT / "cohort/downloads/limn/0_dress_best_model.pt")},
            "shirt": {"dt_id": 1, "ckpt": str(FIQ_ROOT / "cohort/downloads/limn/0_shirt_best_model.pt")},
            "toptee": {"dt_id": 2, "ckpt": str(FIQ_ROOT / "cohort/downloads/limn/0_toptee_best_model.pt")},
        },
        "ckpts": [
            str(FIQ_ROOT / "cohort/downloads/limn/0_dress_best_model.pt"),
            str(FIQ_ROOT / "cohort/downloads/limn/0_shirt_best_model.pt"),
            str(FIQ_ROOT / "cohort/downloads/limn/0_toptee_best_model.pt"),
        ],
    },
    "clvcnet": {
        "variant": "fiq_3head_per_cat",
        "protocol": "in_domain",
        "per_category": {
            "dress": {"local": str(FIQ_ROOT / "checkpoints/clvc/dress_local_model.pt"), "global": str(FIQ_ROOT / "checkpoints/clvc/dress_global_model.pt")},
            "shirt": {"local": str(FIQ_ROOT / "checkpoints/clvc/shirt_local_model.pt"), "global": str(FIQ_ROOT / "checkpoints/clvc/shirt_global_model.pt")},
            "toptee": {"local": str(FIQ_ROOT / "checkpoints/clvc/toptee_local_model.pt"), "global": str(FIQ_ROOT / "checkpoints/clvc/toptee_global_model.pt")},
        },
        "ckpts": [
            str(FIQ_ROOT / "checkpoints/clvc/dress_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/dress_global_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/shirt_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/shirt_global_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/toptee_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/toptee_global_model.pt"),
        ],
    },
    "dcnet": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "checkpoints/dcnet/trained_model.pth")]},
    "sprc": {"variant": "fiq_vitl_clean", "protocol": "in_domain", "backbone": "pretrain_vitL", "ckpts": [str(FIQ_ROOT / "checkpoints/sprc/sprc_fiq_vitl.pt")]},
    "qure": {"variant": "fiq_vitl_repro", "protocol": "in_domain", "backbone": "pretrain_vitL", "ckpts": [str(QURE_CKPT_ROOT / "fashioniq/best_model.pth")]},
    "hint": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "cohort/downloads/hint/fashioniq.pt")]},
    "habit": {"variant": "fiq_n02", "protocol": "in_domain_noisy_20pct", "ckpts": [str(FIQ_ROOT / "cohort/downloads/habit/HABIT-FIQ_N0.2.pt")]},
    "intent": {"variant": "fiq_n02", "protocol": "in_domain_noisy_20pct", "ckpts": [str(FIQ_ROOT / "cohort/downloads/intent/intent_fiq_0.2.pt")]},
    "conesep": {"variant": "fiq_n02", "protocol": "in_domain_noisy_20pct", "ckpts": [str(FIQ_ROOT / "cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt")]},
    "airknow": {"variant": "fiq_n05", "protocol": "in_domain_noisy_50pct", "ckpts": [str(FIQ_ROOT / "checkpoints/airknow/airknow_fiq_0.5.pt")]},
    "csmcir": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt")]},
    "tme": {"variant": "fiq_clean", "protocol": "in_domain", "ckpts": [str(FIQ_ROOT / "cohort/ckpt/tme/fashioniq_0_best_model.pth")]},
    "mtst_ptha": {"variant": "cirr_transfer", "protocol": "cirr_to_fiq_transfer_only", "ckpts": [str(CIRR_ROOT / "mtst_ptha/tuned_mtst_cirr.pt")]},
}

# Exact mapping of the 29 failures from CIR_one_run_probe_spec.md
FAILURES_29: List[Dict[str, Any]] = [
    {"id": "F01", "num": 1, "name": "Negation Inversion", "suites": ["E07", "E04", "E05"]},
    {"id": "F02", "num": 2, "name": "Reference-Copy", "suites": ["E02", "E17", "E01"]},
    {"id": "F03", "num": 3, "name": "Modality Imbalance", "suites": ["E05", "E06"]},
    {"id": "F04", "num": 4, "name": "Multi-Constraint Interference", "suites": ["E08", "E04", "E05"]},
    {"id": "F05", "num": 5, "name": "Attribute Misbinding", "suites": ["E09", "E04"]},
    {"id": "F06", "num": 6, "name": "Caption Primacy Bias", "suites": ["E08", "E12"]},
    {"id": "F07", "num": 7, "name": "Pseudo-Relative Reasoning", "suites": ["E06", "E01"]},
    {"id": "F08", "num": 8, "name": "Geometric Blindness", "suites": ["E04", "E11"]},
    {"id": "F09", "num": 9, "name": "Cardinality Blindness", "suites": ["E04", "E11"]},
    {"id": "F10", "num": 10, "name": "Fine-Grained Blindness", "suites": ["E11", "E04"]},
    {"id": "F11", "num": 11, "name": "Paraphrase Brittleness", "suites": ["E12", "E04"]},
    {"id": "F12", "num": 12, "name": "Cross-Category Lock", "suites": ["E13", "E01"]},
    {"id": "F13", "num": 13, "name": "Pragmatic Intent Blindness", "suites": ["E13", "E12"]},
    {"id": "F14", "num": 14, "name": "Same-Scene Lock", "suites": ["E02", "E17"]},
    {"id": "F15", "num": 15, "name": "Large-Scale Retrieval Degradation", "suites": ["E03", "E14"]},
    {"id": "F16", "num": 16, "name": "Embedding Hubness", "suites": ["E14", "E03"]},
    {"id": "F17", "num": 17, "name": "Uncertainty Blindness", "suites": ["E15"]},
    {"id": "F18", "num": 18, "name": "Negative Transfer & Domain Disparity", "suites": ["E16", "E17"]},
    {"id": "F19", "num": 19, "name": "Annotation Noise & False Negative", "suites": ["E18"]},
    {"id": "F20", "num": 20, "name": "Open-Ended Difference", "suites": ["E06", "E07"]},
    {"id": "F21", "num": 21, "name": "Photo-Presentation Gravity", "suites": ["E10"]},
    {"id": "F22", "num": 22, "name": "Co-Occurring Object Leakage", "suites": ["E10", "E05"]},
    {"id": "F23", "num": 23, "name": "Gaze Direction Blindness", "suites": ["E04", "E11"]},
    {"id": "F24", "num": 24, "name": "Limb Pose Blindness", "suites": ["E04", "E10"]},
    {"id": "F25", "num": 25, "name": "Homophone Hijack", "suites": ["E12", "E04"]},
    {"id": "F26", "num": 26, "name": "Gender-Identity Drop", "suites": ["E09", "E08"]},
    {"id": "F27", "num": 27, "name": "Visual Edit-Distance Cliff", "suites": ["E01", "E03"]},
    {"id": "F28", "num": 28, "name": "BLIP-2 Blind Spot", "suites": ["E04", "E11", "E05"]},
    {"id": "F29", "num": 29, "name": "Pure Fusion Failure", "suites": ["E05", "E04"]},
]


def resolve_checkpoint_for(model_key: str, dataset: str) -> Dict[str, Any]:
    """Return exact checkpoint path(s) and metadata for (model_key, dataset)."""
    spec = MODEL_BY_KEY[model_key]
    if dataset in ("edir", "pinpoint"):
        run_dir = BENCH_RUNS / dataset / SPLIT_TAG / spec.run_id
        if model_key == "clip4cir":
            ckpts = [str(run_dir / "best_stage1_clip.pt"), str(run_dir / "best_stage2_combiner.pt")]
        else:
            ckpts = [str(run_dir / f"best_{model_key}.pt")]
        extra: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "test_results_json": str(run_dir / "test_results.json"),
            "test_preds_json": str(run_dir / "test_predictions_top50.json"),
            "history_jsonl": str(run_dir / "history.jsonl"),
        }
        if model_key == "dcnet":
            extra["glove_vocab"] = str(run_dir / f"glove_vocab_{dataset}_{SPLIT_TAG}.pkl")
        if model_key == "sprc":
            extra["backbone"] = "pretrain"  # EDIR/PinPoint trained with EVA-ViT-g pretrain!
        if model_key == "qure":
            extra["backbone"] = "pretrain_vitL"
        return {
            "model_key": model_key,
            "dataset": dataset,
            "protocol": "in_domain",
            "variant": f"{dataset}_id_best",
            "ckpts": ckpts,
            "all_exist": all(Path(p).is_file() for p in ckpts),
            **extra,
        }
    elif dataset == "fashioniq":
        entry = dict(FIQ_CKPT_MAP[model_key])
        entry["model_key"] = model_key
        entry["dataset"] = "fashioniq"
        entry["all_exist"] = all(Path(p).is_file() for p in entry["ckpts"])
        return entry
    raise ValueError(f"Unknown dataset: {dataset}")


def build_cohort_manifest(out_path: Path) -> Dict[str, Any]:
    """Build and persist the frozen cohort manifest with full resolution audit."""
    models_resolved = []
    for spec in MODEL_SPECS:
        d = asdict(spec)
        d["python_exists"] = Path(spec.python).is_file()
        d["checkpoints"] = {
            ds: resolve_checkpoint_for(spec.key, ds) for ds in ("edir", "pinpoint", "fashioniq")
        }
        models_resolved.append(d)

    manifest = {
        "schema_version": "1.0.0",
        "split_tag": SPLIT_TAG,
        "cohort_resolution_note": (
            "The repo has two overlapping 15-model definitions plus a 16th model (qure, added Nov 24): "
            "(A) run_benchmark_suite.py original 15 EDIR/PinPoint models (all except qure); "
            "(B) run_transfer_suite.py FIQ_CHECKPOINTS 15 FashionIQ in-domain author checkpoints (all except mtst_ptha, "
            "which only has a CIRR transfer checkpoint). To ensure 100% coverage under every interpretation of the "
            "'15 model cohort', this probe run evaluates ALL 16 models on EDIR and PinPoint and ALL 15 in-domain "
            "(+1 mtst_ptha CIRR-transfer explicitly tagged) on FashionIQ."
        ),
        "ambiguous_or_excluded_registry_entries": [
            {
                "id": "pair",
                "status": "excluded_untrained_unresolved",
                "reason": "Staged checkpoints pair-B1.pt / pair-B2.pt exist, but PAIR has no completed native FashionIQ run in DuckDB, no EDIR/PinPoint in-domain training run, and failed native completeness gate in 20261109-native15-composition-limn/GATES.csv.",
            },
            {
                "id": "combiner_rn50x4_noft",
                "status": "excluded_ablation_subset",
                "reason": "Frozen-CLIP ablation of clip4cir_rn50x4_fullft on FashionIQ only; not an independent architecture in EDIR/PinPoint benchmark suite.",
            },
        ],
        "counts": {
            "total_evaluated_models": len(MODEL_SPECS),
            "original_15_bench_models": sum(1 for m in MODEL_SPECS if m.in_original_15_bench),
            "fiq_15_indomain_models": sum(1 for m in MODEL_SPECS if m.in_fiq_15_indomain),
            "failures_total": len(FAILURES_29),
            "suites_total": 18,
        },
        "models": models_resolved,
        "failures": FAILURES_29,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
