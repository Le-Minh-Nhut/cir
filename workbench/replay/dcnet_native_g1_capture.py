"""Capture source-native DCNet features and component scores for four fixed dress queries."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUERY_ROWS = (657, 933, 1362, 1980)


def _pack(experts, modalities):
    return torch.cat([experts[mod] for mod in modalities], dim=1)


def _isolated_config(checkpoint: Path, config_root: Path) -> None:
    config_data = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config_data["trainer"]["save_dir"] = str(config_root / "native-work")
    config_root.joinpath("config.json").write_text(json.dumps(config_data), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    import tempfile
    scratch = tempfile.TemporaryDirectory(prefix="dcnet-g1-native-config-")
    config_root = Path(scratch.name)
    _isolated_config(args.checkpoint, config_root)
    import os
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = ["test.py", "--resume", str(config_root), "--mode", "val"]
    from parse_config import ConfigParser
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--resume", default="", type=str)
    config_parser.add_argument("--device", default="", type=str)
    config_parser.add_argument("--seeds", default="0", type=str)
    config_parser.add_argument("--mode", default="val", type=str)
    config = ConfigParser(config_parser, "eval")
    config["data_loader"]["args"]["mode"] = "val"
    config["data_loader"]["args"]["categories"] = ["dress"]
    import data_loader.data_loaders as module_data
    import model.model as module_arch
    from utils.util import compute_dims

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dims = compute_dims(config)
    loaders = config.init(name="data_loader", module=module_data, expert_dims=dims,
                          text_feat=config["experts"]["text_feat"],
                          text_dim=config["experts"]["text_dim"])
    model = config.init(name="arch", module=module_arch, expert_dims=dims,
                        text_dim=config["experts"]["text_dim"],
                        same_dim=config["experts"]["ce_shared_dim"],
                        text_feat=config["experts"]["text_feat"])
    state = torch.load(args.checkpoint / "trained_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    dataset = loaders["dress"].dataset
    target_dataset = loaders["dress_trg"].dataset
    gallery_ids = [str(item) for item in target_dataset.data]
    gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
    gallery_experts = {mod: [] for mod in model.modalities}
    with torch.inference_mode():
        for batch in loaders["dress_trg"]:
            pixels = batch["candidate_experts"]["im_feat"].to(device)
            experts = model.get_image_emb({"im_feat": pixels}, istgt=True)
            for mod in model.modalities:
                gallery_experts[mod].append(experts[mod].float().cpu().numpy())
        gallery = {mod: torch.from_numpy(np.concatenate(rows, axis=0)).to(device)
                   for mod, rows in gallery_experts.items()}
        selected = {}
        selected_set = set(QUERY_ROWS)
        source_row_offset = 0
        for batch in loaders["dress"]:
            rows = batch["meta_info"]
            batch_indices = [i for i in range(len(rows)) if source_row_offset + i in selected_set]
            if not batch_indices:
                source_row_offset += len(rows)
                continue
            row_positions = {i: source_row_offset + i for i in batch_indices}
            texts = batch["text"][batch_indices].to(device)
            lengths = [batch["text_lengths"][i] for i in batch_indices]
            pixels = batch["candidate_experts"]["im_feat"][batch_indices].to(device)
            source_experts = model.get_image_emb({"im_feat": pixels}, istgt=False)
            text_experts = model.get_text_emb(texts, lengths)
            composed = model.get_composition(source_experts, text_experts)
            for local, batch_i in enumerate(batch_indices):
                row = rows[batch_i]
                row_id = row_positions[batch_i]
                source_row_experts = {mod: value[local:local+1] for mod, value in source_experts.items()}
                text = {mod: value[local:local+1] for mod, value in text_experts.items()}
                composed_row = {mod: value[local:local+1] for mod, value in composed.items()}
                comp = model.get_score_matrix(trg_embds=gallery, src_embds=composed_row,
                                              subspaces=model.modalities, l2renorm=False,
                                              dist=False, val=True)
                correction_source = {mod: value.mean((1, 2)) if model.feat2d else value
                                    for mod, value in source_row_experts.items()}
                corrected = model.get_correction(correction_source, gallery)
                corr = model.get_score_matrix(trg_embds=corrected, src_embds=text,
                                              subspaces=model.modalities, l2renorm=False,
                                              dist=False, val=True)
                reference_index = gallery_index[row["candidate"]]
                comp[0, reference_index] = -10e10
                corr[0, reference_index] = -10e10
                selected[row_id] = {
                    "candidate": row["candidate"], "target": row["target"],
                    "pixels": pixels[local].cpu().numpy(),
                    "tokens": batch["text"][batch_i].cpu().numpy(),
                    "token_length": int(batch["text_lengths"][batch_i]),
                    "source": {mod: val[0].cpu().numpy() for mod, val in source_row_experts.items()},
                    "text": {mod: val[0].cpu().numpy() for mod, val in text.items()},
                    "composed": {mod: val[0].cpu().numpy() for mod, val in composed_row.items()},
                    "composition_scores": comp[0].cpu().numpy(),
                    "correction_scores": corr[0].cpu().numpy(),
                    "fused": (F.log_softmax(comp, -1) + F.log_softmax(corr, -1))[0].cpu().numpy(),
                }
            source_row_offset += len(rows)
    if sorted(selected) != list(QUERY_ROWS):
        raise RuntimeError(f"selected source rows differ: {sorted(selected)}")
    args.output.mkdir(parents=True, exist_ok=False)
    modalities = list(model.modalities)
    for position, row in selected.items():
        np.savez_compressed(args.output / f"row-{position}.npz", position=position,
                            candidate=row["candidate"], target=row["target"], pixels=row["pixels"],
                            tokens=row["tokens"], token_length=row["token_length"],
                            **{f"source.{mod}": row["source"][mod] for mod in modalities},
                            **{f"text.{mod}": row["text"][mod] for mod in modalities},
                            **{f"composed.{mod}": row["composed"][mod] for mod in modalities},
                            composition_scores=row["composition_scores"],
                            correction_scores=row["correction_scores"], fused=row["fused"])
    gallery_packed = np.concatenate([gallery[mod].cpu().numpy() for mod in modalities], axis=1)
    np.save(args.output / "gallery_context.npy", gallery_packed, allow_pickle=False)
    np.save(args.output / "gallery_ids.npy", np.asarray(gallery_ids), allow_pickle=False)
    (args.output / "receipt.json").write_text(json.dumps({
        "status": "native_selected_g1_complete", "rows": list(QUERY_ROWS),
        "gallery_count": len(gallery_ids), "modalities": modalities,
        "checkpoint": str(args.checkpoint.resolve()), "source_root": str(source),
        "interpreter": sys.executable, "torch": torch.__version__, "device": str(device),
        "capture": "source ConfigParser/DataLoaders, eval-mode actual model forward; native val target/source/text preprocessing and component scorer; native reference masking before component softmax",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
