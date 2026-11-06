"""Phase 4: native FashionIQ evaluation for a restored PAIR checkpoint.

Reimplements the public `test.py::test` scoring path exactly:

  * gallery = first-seen ordered union of (source, target) ids from the validation annotation
    pairs, i.e. `datasets.FashionIQ.get_test_data` with split='val-split'
  * query = (source image, "cap0 and cap1" after the dataset's own correction rule)
  * query features = model.extract_retrieval_compose, target features = extract_retrieval_target
  * both L2-normalised, cosine = dot product
  * source image score set to -10e10 before Recall (reference excluded)
  * R@1 / R@10 / R@50 = share of queries whose target is inside the top-K

Native preprocessing is rebuilt from `preprocess_cfg` stored in the checkpoint. Native caption
text ordering (cap0 and cap1) and correction_dict are used unchanged. Nothing here shrinks the
gallery or the query set.

Speed only (semantics preserved): the dataset is read with a thread pool, features are computed in
batches of `--batch-size`, and the model runs under torch.no_grad() in eval() mode.
"""
import argparse
import json
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compat_loader  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from open_clip.transform import image_transform  # noqa: E402

CATEGORIES = ("dress", "shirt", "toptee")


def native_transform(cfg):
    return image_transform(image_size=cfg["size"], is_train=False, mean=cfg["mean"],
                           std=cfg["std"], interpolation=cfg["interpolation"],
                           resize_mode=cfg["resize_mode"], fill_color=cfg["fill_color"])


def build_category(dataset_root, category):
    """Public gallery construction, exactly as datasets.py::get_test_data(split='val-split')."""
    ds = Path(dataset_root)
    images = json.loads((ds / "image_splits" / f"split.{category}.val.json").read_text())
    caps = json.loads((ds / "captions" / f"cap.{category}.val.json").read_text())
    corr = json.loads((ds / "captions" / f"correction_dict_{category}.json").read_text())
    trans = str.maketrans({k: " " for k in string.punctuation})

    def clean(text):
        toks = str(text).lower().translate(trans).strip().split()
        return " ".join(corr.get(t, t) for t in toks)

    queries = []
    for row in caps:
        queries.append({
            "source_id": images.index(row["candidate"]),
            "target_id": images.index(row["target"]),
            "source_name": row["candidate"],
            "target_name": row["target"],
            "mod": f"{clean(row['captions'][0])} and {clean(row['captions'][1])}",
        })

    gallery_ids, seen = [], set()
    for q in queries:
        for key in ("source_id", "target_id"):
            if q[key] not in seen:
                seen.add(q[key])
                gallery_ids.append(q[key])
    return {"images": images, "queries": queries, "gallery_ids": gallery_ids,
            "img_dir": ds / "resized_image" / category}


def load_all(paths, tf, workers):
    def one(p):
        with Image.open(p) as im:
            return tf(im.convert("RGB"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, paths))


@torch.no_grad()
def encode(model, tensors, device, batch_size, fn, text=None):
    out = []
    for i in range(0, len(tensors), batch_size):
        chunk = torch.stack(tensors[i:i + batch_size]).to(device)
        if text is None:
            f = fn(chunk)
        else:
            f = fn(chunk, text[i:i + batch_size])
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def evaluate_category(model, dataset_root, category, tf, device, batch_size, workers):
    data = build_category(dataset_root, category)
    img_dir = data["img_dir"]
    q_paths = [img_dir / f"{q['source_name']}.jpg" for q in data["queries"]]
    g_paths = [img_dir / f"{data['images'][i]}.jpg" for i in data["gallery_ids"]]
    missing = [str(p) for p in (q_paths + g_paths) if not p.is_file()]
    if missing:
        raise SystemExit(f"MISSING_ASSETS {category}: {len(missing)} e.g. {missing[:3]}")

    t0 = time.time()
    q_imgs = load_all(q_paths, tf, workers)
    g_imgs = load_all(g_paths, tf, workers)
    mods = [q["mod"] for q in data["queries"]]

    qf = encode(model, q_imgs, device, batch_size, model.extract_retrieval_compose, text=mods)
    gf = encode(model, g_imgs, device, batch_size, model.extract_retrieval_target)

    # test.py normalises again after the methods already normalised; keep both steps.
    qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
    gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)

    sims = qf @ gf.T
    pos = {gid: i for i, gid in enumerate(data["gallery_ids"])}
    n_excluded = 0
    for i, q in enumerate(data["queries"]):
        if q["source_id"] in pos:
            sims[i, pos[q["source_id"]]] = -10e10
            n_excluded += 1

    order = np.argsort(-sims, axis=1)[:, :50]
    recalls = {}
    for k in (1, 10, 50):
        hit = sum(1 for i, q in enumerate(data["queries"]) if pos[q["target_id"]] in order[i][:k])
        recalls[f"{category}_r{k}"] = 100.0 * hit / len(data["queries"])
    return {
        "category": category,
        "n_queries": len(data["queries"]),
        "n_gallery": len(data["gallery_ids"]),
        "n_source_excluded": n_excluded,
        "n_missing_assets": 0,
        "coverage": 1.0,
        "recalls": recalls,
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, choices=sorted(compat_loader.EXPECTED_SHA256))
    ap.add_argument("--hypothesis", required=True, choices=sorted(compat_loader.ATTENTION_HYPOTHESES))
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj, meta = compat_loader.load(args.checkpoint, args.tag, args.hypothesis)
    tf = native_transform(obj.backbone.clip.visual.preprocess_cfg)
    obj = obj.to(device).eval()

    per_cat = []
    for cat in CATEGORIES:
        res = evaluate_category(obj, args.dataset_root, cat, tf, device, args.batch_size, args.workers)
        per_cat.append(res)
        print(json.dumps(res["recalls"]), flush=True)

    macro = {}
    for k in ("r1", "r10", "r50"):
        macro[f"macro_{k}"] = sum(c["recalls"][f"{c['category']}_{k}"] for c in per_cat) / len(per_cat)
    macro["macro_mean"] = (macro["macro_r10"] + macro["macro_r50"]) / 2

    out = {
        "model": "pair",
        "candidate": f"pair_{args.tag.lower()}",
        "hypothesis": args.hypothesis,
        "mapping_status": "EXPLORATORY_PUBLIC_MAPPING",
        "checkpoint": {"path": meta["path"], "sha256": meta["sha256"]},
        "class_mapping": meta["mapping"],
        "protocol": "fashioniq_val_split",
        "protocol_note": ("gallery = first-seen ordered union of val source/target ids; source "
                          "image score masked to -10e10 before Recall"),
        "device": str(device),
        "batch_size": args.batch_size,
        "mode": "eval (all restored Dropout modules have p=0.0)",
        "precision": "float32",
        "per_category": per_cat,
        "macro": macro,
        "paper_reference": {"macro_r10": 52.76, "macro_r50": 74.51,
                            "note": "registry reference row; eligibility recorded in the report"},
        "torch": torch.__version__,
        "complete": True,
    }
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(macro, indent=1))


if __name__ == "__main__":
    main()
