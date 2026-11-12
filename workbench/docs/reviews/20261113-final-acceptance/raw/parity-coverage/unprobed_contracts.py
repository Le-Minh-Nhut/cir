"""Step 3b: why the four remaining EXECUTABLE models have no pair, measured from their own code.

The acceptance rule is "a real paired probe for every executable configuration". Four models are
`EXECUTABLE` and unprobed: `ptha_mtst`, `clvc_net`, `tgcir`, `tme`. This file records, per model, the
CONCRETE reason a pair was not produced, by quoting the pinned code that makes it so. Each entry names
the seam a future producer would have to cross -- not a vague "not done".

Writes `unprobed_contracts.json`. Static reading plus cheap imports only: no checkpoint load, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
OUT = Path(__file__).resolve().parent

CONTRACTS = {
    "clvc_net": {
        "verdict": "NEEDS_A_DISTINCT_HOOK_NOT_A_PARAMETER",
        "why": (
            "CLVC-Net is not one forward. The pinned evaluator drives TWO models and concatenates "
            "their features: `test.py:48` and `test.py:116` do "
            "`torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), "
            "F.normalize(global_model.extract_img_feature(imgs)[1])], dim=1)` -> a 2048-d row from two "
            "4096-d halves. The shared hook contract is `native_features(paths, texts)` on ONE network, "
            "so this needs a new hook shape (a two-model capture), not another ModelScope field."
        ),
        "loader": "workbench/replay/clvc_net.py:125 `_load_replay_modules(source_root)` -> "
                  "(torch, datasets, test_module) -- a 3-tuple with NO data_utils, so the shared "
                  "`load_replay_modules()` index-2 `data_utils` read does not apply",
        "dataset": "workbench/third_party/CLVCNet/datasets.py:14 "
                   "`__init__(path, split='train', existed_npy=True, transform=None)` -- no "
                   "`noise_ratio`, and the keyword set differs from the LAVIS family",
        "scope": "per-category: `model_clvc_net.py:205 category_scopes = ('dress','shirt','toptee')`, "
                 "three separate checkpoint files",
        "also": "the adapter's `score()` is cosine over concatenated per-half-normalised 2048-d rows, "
                "so the comparator's gallery layout differs from the 32x256 BLIP-2 layout",
        "gpu_cost": "unknown, but two models per forward and a 74k-image gallery makes a full-population "
                    "encode the largest of the four",
    },
    "tme": {
        "verdict": "NO_SINGLE_CATEGORY_AND_NO_PER_SAMPLE_API",
        "why": (
            "`workbench/replay/tme.py:352-356` (the `raise` is at `:356`) refuses a category subset outright: 'The native "
            "evaluator always computes all three categories and has no single-category API. Refuse "
            "subsets rather than report metrics for categories the operator did not request.' A "
            "per-category pair — which is the shape every existing artifact uses — is therefore not "
            "expressible through this evaluator at all."
        ),
        "loader": "no `_load_replay_modules`: it imports INSIDE `_evaluate_categories` "
                  "(`tme.py:360-369`) via `load_model_and_preprocess`, `data`, "
                  "`precompute_evaluation.evaluate_features`, `utility.targetpad_transform`",
        "dataset": "`tme_data.FashioniqPath = str(dataset_root) + '/'` (`tme.py:365`), i.e. the "
                   "evaluator reads the dataset through a module-level global rather than a handle",
        "scope": "`category_scopes = (None,)` -- category-agnostic adapter",
        "also": "needs LAVIS BLIP-2 `pretrain` weights staged as an external runtime asset",
        "gpu_cost": "full three-category native evaluation in one process; the widest of the four",
    },
    "tgcir": {
        "verdict": "NEEDS_A_DEVICE_AND_CLIP_PREPROCESS_AWARE_HOOK",
        "why": (
            "`workbench/replay/tgcir.py:174 _load_tgcir(source_root, device)` returns "
            "`(torch, (clip, datasets, tg_test), adaptations)` -- 3 elements, and the middle is "
            "itself a 3-tuple. It also needs `clip.load('ViT-B/16')` PREPROCESS for the transform, not "
            "`data_utils.targetpad_transform(1.25, 224)`, so the shared transform/processor assumption "
            "is wrong for this model. The pinned source is CUDA-hard "
            "(`model.py:32,107,120,136,244,251`, `test.py:31,47,101,115`) with no device guard."
        ),
        "loader": "`workbench/replay/tgcir.py:174` -- see above; plus a torch-1.7 pickle alias patch "
                  "(`_LinearWithBias`) applied at load",
        "scope": "`category_scopes = (None,)` -- category-agnostic",
        "also": "the replay is a WORKBENCH-ADDED path over the pinned encoder; `test.py` has no CLI, "
                "so there is no author evaluator to be faithful to",
        "gpu_cost": "moderate: 512-d features, 607 MB checkpoint",
    },
    "ptha_mtst": {
        "verdict": "TRANSFER_CHECKPOINT_SO_A_PARITY_VERDICT_WOULD_MISLABEL_IT",
        "why": (
            "`cirr_transfer` is a CIRR-trained -> FashionIQ transfer checkpoint, not a FashionIQ "
            "reproduction (`models.yaml reproduction_status: SELECTED_TRANSFER_ONLY`; the adapter's own "
            "status string says so). A G1/G2 pair measures adapter-vs-native AGREEMENT for the same "
            "checkpoint, which is meaningful, but the registry's own disposition for this model is "
            "`TRANSFER_ONLY`, so a pair here would be measuring a model the campaign does not treat as "
            "a FashionIQ configuration. That is a scope decision for the acceptance owner, not a "
            "producer gap."
        ),
        "loader": "`workbench/replay/ptha_mtst.py` has no `_load_replay_modules`; it drives a "
                  "subprocess-shaped official evaluator through `evaluate_transfer` (`:291`) and "
                  "`preflight` (`:57`)",
        "adapter": "`model_ptha_mtst.py:360` "
                   "`model.extract_target_features(batch, mode='mean', target_only=True)` -- a "
                   "different target-side contract from the BLIP-2 32x256 layout",
        "gpu_cost": "moderate: 2 720 352 973 B checkpoint",
    },
}


def main() -> int:
    document = {
        "generated_by": "raw/parity-coverage/unprobed_contracts.py",
        "purpose": "why four EXECUTABLE models have no pair, quoted from their pinned code",
        "method": "static reading of the named file:line sites; no checkpoint load, no GPU",
        "models": CONTRACTS,
        "summary": {model: entry["verdict"] for model, entry in CONTRACTS.items()},
    }
    (OUT / "unprobed_contracts.json").write_text(json.dumps(document, indent=1) + "\n",
                                                encoding="utf-8")
    for model, entry in CONTRACTS.items():
        print(f"{model:12s} {entry['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
