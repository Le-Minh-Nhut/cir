#!/usr/bin/env python3
"""Model-independent frozen CLIP ViT-L/14 (DataComp-XL-13B) arbiter feature extractor.

Executed inside `workbench/envs/limn310-adapted/bin/python` (has open_clip + DataComp-XL cache).
Produces reusable, model-independent visual & text unit embeddings for:
  - E03: hard & mixed distractor pools (strictly independent of all 16 evaluated CIR models!)
  - E01 / F27: independent visual edit-distance `d_edit_clip = 1 - cos_CLIP(I_ref, I_target)`
  - E02: difficulty-matched other-scene hard negative mining on EDIR & PinPoint train splits
  - E06 / E07 / E09: candidate verification & visual similarity metadata

Saves compact fp16 `.npy` arrays + ID lists under `<out_dir>/arbiter_clip/<dataset>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PIL.Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _ImagePathDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        iid, path = self.items[idx]
        with PIL.Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return iid, x


def _encode_images(model, preprocess, items: List[Tuple[str, str]], device: torch.device, bs: int = 256, num_workers: int = 8) -> Tuple[np.ndarray, List[str]]:
    ds = _ImagePathDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    feats = []
    ids: List[str] = []
    with torch.no_grad():
        for batch_ids, imgs in dl:
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                f = model.encode_image(imgs)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.half().cpu().numpy())
            ids.extend(batch_ids)
    return np.concatenate(feats, axis=0), ids


def _encode_texts(model, tokenizer, texts: List[str], device: torch.device, bs: int = 256) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            toks = tokenizer(texts[i : i + bs]).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                f = model.encode_text(toks)
            f = F.normalize(f.float(), dim=-1)
            feats.append(f.half().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), dtype=np.float16)


def _collect_edir_items() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, List[dict]]]:
    base = REPO_ROOT / "data/EDIR"
    split_dir = base / "splits/70_10_20_seed42"
    corpus = [json.loads(l) for l in (split_dir / "corpus_full.jsonl").read_text().splitlines()]
    corpus_items = [(r["image_id"], str(base / r["image_rel"])) for r in corpus]
    splits = {}
    ref_map: Dict[str, str] = {}
    for sp in ("train", "val", "test"):
        rows = [json.loads(l) for l in (split_dir / f"{sp}.jsonl").read_text().splitlines()]
        splits[sp] = rows
        for r in rows:
            ref_map[r["ref_id"]] = str(base / r["ref_image_rel"])
    ref_items = sorted(ref_map.items())
    return corpus_items, ref_items, splits


def _collect_pinpoint_items() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, List[dict]]]:
    base = REPO_ROOT / "data/PinPoint"
    split_dir = base / "splits/70_10_20_seed42"
    corpus = [json.loads(l) for l in (split_dir / "corpus_full.jsonl").read_text().splitlines()]
    corpus_items = [(r["image_id"], str(base / r["image_rel"])) for r in corpus]
    corpus_set = {iid for iid, _ in corpus_items}
    splits = {}
    ref_map: Dict[str, str] = {}
    for sp in ("train", "val", "test"):
        rows = [json.loads(l) for l in (split_dir / f"{sp}.jsonl").read_text().splitlines()]
        splits[sp] = rows
        for r in rows:
            if r["ref_id"] not in corpus_set:
                ref_map[r["ref_id"]] = str(base / r["ref_image_rel"])
    ref_items = sorted(ref_map.items())
    return corpus_items, ref_items, splits


def _collect_fashioniq_items() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, List[dict]]]:
    base = REPO_ROOT / "data/FashionIQ"
    img_dir = base / "images"
    gal_map: Dict[str, str] = {}
    splits: Dict[str, List[dict]] = {"val": []}
    for cat in ("dress", "shirt", "toptee"):
        sp_ids = json.loads((base / f"image_splits/split.{cat}.val.json").read_text())
        for iid in sp_ids:
            p = img_dir / f"{iid}.png"
            if not p.is_file():
                p = img_dir / f"{iid}.jpg"
            gal_map[iid] = str(p)
        caps = json.loads((base / f"captions/cap.{cat}.val.json").read_text())
        for qi, r in enumerate(caps):
            for iid in (r["candidate"], r["target"]):
                if iid not in gal_map:
                    p = img_dir / f"{iid}.png"
                    if not p.is_file():
                        p = img_dir / f"{iid}.jpg"
                    gal_map[iid] = str(p)
            splits["val"].append({
                "query_id": f"{cat}:{qi}",
                "category": cat,
                "ref_id": r["candidate"],
                "target_id": r["target"],
                "text": f"{r['captions'][0]} and {r['captions'][1]}",
                "captions": r["captions"],
            })
    return sorted(gal_map.items()), [], splits


def run_arbiter(dataset: str, out_root: Path, num_workers: int = 8) -> None:
    import open_clip

    out_dir = out_root / "arbiter_clip" / dataset
    done_marker = out_dir / "DONE.json"
    if done_marker.is_file():
        print(f"[ARBITER SKIP] {dataset} already computed at {out_dir}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[ARBITER] Loading frozen CLIP ViT-L/14 (datacomp_xl_s13b_b90k) for {dataset}...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device).eval()

    if dataset == "edir":
        corpus_items, extra_ref_items, splits = _collect_edir_items()
    elif dataset == "pinpoint":
        corpus_items, extra_ref_items, splits = _collect_pinpoint_items()
    elif dataset == "fashioniq":
        corpus_items, extra_ref_items, splits = _collect_fashioniq_items()
    else:
        raise ValueError(dataset)

    print(f"[ARBITER] Encoding {len(corpus_items)} corpus images + {len(extra_ref_items)} extra ref images for {dataset}...", flush=True)
    c_emb, c_ids = _encode_images(model, preprocess, corpus_items, device, bs=256, num_workers=num_workers)
    np.save(out_dir / "corpus_emb.npy", c_emb)
    (out_dir / "corpus_ids.json").write_text(json.dumps(c_ids), encoding="utf-8")

    if extra_ref_items:
        r_emb, r_ids = _encode_images(model, preprocess, extra_ref_items, device, bs=256, num_workers=num_workers)
        np.save(out_dir / "extra_ref_emb.npy", r_emb)
        (out_dir / "extra_ref_ids.json").write_text(json.dumps(r_ids), encoding="utf-8")

    # Encode query texts per split
    for sp, rows in splits.items():
        qids = [r["query_id"] for r in rows]
        texts = [r["text"] for r in rows]
        t_emb = _encode_texts(model, tokenizer, texts, device, bs=256)
        np.save(out_dir / f"{sp}_text_emb.npy", t_emb)
        (out_dir / f"{sp}_query_ids.json").write_text(json.dumps(qids), encoding="utf-8")

    dt = time.time() - t0
    meta = {
        "dataset": dataset,
        "arbiter_model": "open_clip:ViT-L-14:datacomp_xl_s13b_b90k",
        "n_corpus": len(c_ids),
        "n_extra_ref": len(extra_ref_items),
        "splits": {sp: len(rows) for sp, rows in splits.items()},
        "elapsed_sec": round(dt, 2),
    }
    done_marker.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ARBITER DONE] {dataset} in {dt:.1f}s -> {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen CLIP ViT-L/14 DataComp-XL arbiter embeddings")
    parser.add_argument("--datasets", type=str, default="edir,pinpoint,fashioniq")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        run_arbiter(ds, args.out_dir, num_workers=min(args.num_workers, 8))


if __name__ == "__main__":
    main()
