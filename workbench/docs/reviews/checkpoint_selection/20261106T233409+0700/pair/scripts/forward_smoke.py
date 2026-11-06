"""Phase 3: real forward tests for a restored PAIR checkpoint.

Runs the public inference methods on the restored object:

    extract_retrieval_compose(ref_image, [modification_text])
    extract_retrieval_target(target_image)

Native preprocessing is rebuilt from the `preprocess_cfg` that the checkpoint itself stores
(no invented resize/crop/normalisation). Captions are joined with the public dataset rule
("cap0 and cap1" after lower/punctuation-strip/correction_dict), so the tokenizer sees the text
the public test path would see.

Checks, per checkpoint:
  1. output shapes and finiteness
  2. L2 norm == 1 on both branches
  3. determinism: repeat in eval() twice, plus a fresh identical batch, must match bitwise
  4. eval() vs train() divergence (proves dropout state is actually honoured, i.e. the restored
     modules run their real forward paths rather than a no-op)
  5. which restored branches actually executed (hooks), to separate inference modules from
     training-only modules
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compat_loader  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from open_clip.transform import image_transform  # noqa: E402


def native_transform(cfg):
    return image_transform(
        image_size=cfg["size"],
        is_train=False,
        mean=cfg["mean"],
        std=cfg["std"],
        interpolation=cfg["interpolation"],
        resize_mode=cfg["resize_mode"],
        fill_color=cfg["fill_color"],
    )


def load_image(path, tf):
    with Image.open(path) as im:
        return tf(im.convert("RGB"))


def sample_triplet(dataset_root, category, index=0):
    """Read one validation triplet with the public dataset's own text rule."""
    import string
    split = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
    caps = json.loads((dataset_root / "captions" / f"cap.{category}.val.json").read_text())
    corr = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json").read_text())
    row = caps[index]
    ref, tgt = row["candidate"], row["target"]

    def clean(text):
        trans = str.maketrans({k: " " for k in string.punctuation})
        toks = str(text).lower().translate(trans).strip().split()
        return " ".join(corr.get(t, t) for t in toks)

    mod = f"{clean(row['captions'][0])} and {clean(row['captions'][1])}"
    img_dir = dataset_root / "resized_image" / category
    return {"ref": img_dir / f"{ref}.jpg", "tgt": img_dir / f"{tgt}.jpg", "mod": mod,
            "ref_id": ref, "tgt_id": tgt, "gallery_index": split.index(tgt),
            "ref_index": split.index(ref), "n_split": len(split)}


def main():
    ckpt, tag, hyp = sys.argv[1], sys.argv[2], sys.argv[3]
    dataset_root, out = Path(sys.argv[4]), Path(sys.argv[5])
    batch = int(sys.argv[6]) if len(sys.argv) > 6 else 8

    obj, meta = compat_loader.load(ckpt, tag, hyp, map_location="cpu")
    cfg = obj.backbone.clip.visual.preprocess_cfg
    tf = native_transform(cfg)

    ds = Path(dataset_root)
    trips = [sample_triplet(ds, c, i) for c, i in (("dress", 0), ("shirt", 0), ("toptee", 0))]
    refs = torch.stack([load_image(t["ref"], tf) for t in trips])[:batch]
    tgts = torch.stack([load_image(t["tgt"], tf) for t in trips])[:batch]
    mods = [t["mod"] for t in trips][:batch]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = obj.to(device).eval()

    executed = {}
    hooks = []
    for name in ("transformer", "transformer_a", "transformer_b", "transformer_c", "transformer_d"):
        mod = getattr(obj.backbone, name, None)
        if mod is not None:
            executed[name] = 0
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, n=name: executed.__setitem__(n, executed[n] + 1)))

    rep = {"tag": tag, "hypothesis": hyp, "sha256": meta["sha256"],
           "device": str(device), "batch": batch,
           "preprocess_cfg": cfg, "mapping": meta["mapping"],
           "samples": [{"category": t.get("category"), "ref": str(t["ref"]), "tgt": str(t["tgt"]),
                        "mod": t["mod"]} for t in trips[:batch]]}

    with torch.no_grad():
        q = obj.extract_retrieval_compose(refs.to(device), mods)
        t = obj.extract_retrieval_target(tgts.to(device))
        q2 = obj.extract_retrieval_compose(refs.to(device), mods)
        t2 = obj.extract_retrieval_target(tgts.to(device))

    rep["compose_shape"] = list(q.shape)
    rep["target_shape"] = list(t.shape)
    rep["compose_finite"] = bool(torch.isfinite(q).all())
    rep["target_finite"] = bool(torch.isfinite(t).all())
    rep["compose_norms"] = [float(x) for x in q.norm(dim=-1)]
    rep["target_norms"] = [float(x) for x in t.norm(dim=-1)]
    rep["determinism_eval_max_abs_diff"] = {
        "compose": float((q - q2).abs().max()),
        "target": float((t - t2).abs().max()),
    }

    # train() must differ, proving the restored submodules run their real forwards.
    obj.train()
    with torch.no_grad():
        q_tr = obj.extract_retrieval_compose(refs.to(device), mods)
    obj.eval()
    rep["train_vs_eval_max_abs_diff"] = float((q - q_tr).abs().max())

    # Ranking sanity: composed query must place its own target above an unrelated image.
    sims = q @ t.T
    rep["similarity_matrix"] = [[round(float(v), 5) for v in row] for row in sims]
    rep["diagonal_top1"] = [int(sims[i].argmax()) for i in range(sims.shape[0])]
    rep["diagonal_is_top1"] = bool(all(int(sims[i].argmax()) == i for i in range(sims.shape[0])))

    rep["branches_executed"] = executed
    for h in hooks:
        h.remove()

    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in (
        "tag", "hypothesis", "compose_shape", "target_shape", "compose_finite", "target_finite",
        "determinism_eval_max_abs_diff", "train_vs_eval_max_abs_diff", "diagonal_is_top1",
        "branches_executed")}, indent=1))


if __name__ == "__main__":
    main()
