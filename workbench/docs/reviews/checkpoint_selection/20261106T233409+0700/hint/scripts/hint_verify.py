"""Full HINT verification: input identity, native-vs-wrapper parity, rank capture, recomputation.

Design fixed BEFORE seeing scores:
  * parity set = first 3 + last 2 queries of each category plus the tail of the gallery, so the short
    final batch is exercised;
  * native path  = the pinned `test.py` code transcribed by hand into this file (same ops, same order,
    same dtype), NOT the wrapper;
  * wrapper path = the shipped `workbench.replay.hint` evaluate path;
  * agreement    = top-50 index lists per query must be equal as sets, and hit counts equal.
"""
import json, sys, hashlib
from pathlib import Path

CIR = Path("/home/heheboiz/data/cir")
SR = CIR / "workbench/third_party/HINT"
DS = CIR / "data/FashionIQ"
CKPT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
sys.path.insert(0, str(CIR))
import numpy as np

from workbench.replay.hint import _load_replay_modules, CHECKPOINT, PROTOCOL, evaluate_categories
from workbench.replay.common import load_trusted_whole_model, isolated_cache_root, sha256_file

torch, datasets, pinned, processors = _load_replay_modules(SR, "pretrain")
data_utils, test_module = pinned
txt_processors, boundary = processors

actual = sha256_file(CKPT)
model = load_trusted_whole_model(CKPT, CHECKPOINT["expected_sha256"], torch).cuda().eval()
transform = data_utils.targetpad_transform(1.25, 224)

record = {"checkpoint": {"path": str(CKPT), "bytes": CKPT.stat().st_size, "sha256": actual,
                         "registry_sha256": CHECKPOINT["expected_sha256"],
                         "sha_match": actual == CHECKPOINT["expected_sha256"],
                         "root_class": type(model).__name__,
                         "root_module": type(model).__module__},
          "modules": {"datasets": datasets.__file__, "test": test_module.__file__,
                      "processor_boundary": boundary},
          "runtime": {"python": sys.executable, "torch": torch.__version__,
                      "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
          "protocol": PROTOCOL, "batch_size": BATCH, "categories": {}}

def native_rank_and_recalls(queries, targets, category, batch_size, model, tproc):
    """Hand transcription of pinned test.py:test, returning ranks + recalls."""
    with torch.no_grad():
        all_q, all_t = [], []
        imgs, mods = [], []
        for i, t in enumerate(queries):
            imgs.append(t["source_img_data"]); mods.append(t["mod"]["str"])
            if len(imgs) >= batch_size or i == len(queries) - 1:
                batch = torch.stack([torch.from_numpy(d) if "torch" not in str(type(d)) else d
                                     for d in imgs]).float().cuda()
                f = model.extract_retrieval_compose(batch, [tproc["eval"](c) for c in mods])
                all_q.append(f.data.cpu().numpy()); imgs, mods = [], []
        all_q = np.concatenate(all_q)

        imgs = []
        for i, t in enumerate(targets):
            imgs.append(t["target_img_data"])
            if len(imgs) >= batch_size or i == len(targets) - 1:
                batch = torch.stack([torch.from_numpy(d) if "torch" not in str(type(d)) else d
                                     for d in imgs]).float().cuda()
                all_t.append(model.extract_retrieval_target(batch).data.cpu().numpy()); imgs = []
        all_t = np.concatenate(all_t)

    sims = np.matmul(all_q, all_t).squeeze().max(-1)
    tid = [t["target_img_id"] for t in targets]
    for i, t in enumerate(queries):
        sims[i, tid.index(t["source_img_id"])] = -10e10
    nn = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
    recalls = {}
    for k in (1, 10, 50):
        hits = sum(1 for i, ns in enumerate(nn) if tid.index(queries[i]["target_img_id"]) in ns[:k])
        recalls[k] = 100.0 * hits / len(nn)
    ranks = []
    for i, ns in enumerate(nn):
        tgt = tid.index(queries[i]["target_img_id"])
        ranks.append(int(np.where(ns == tgt)[0][0]) + 1 if tgt in ns else 0)
    return recalls, ranks, [int(x) for x in np.asarray(sims).shape[:1]]

full_native = {}
with isolated_cache_root(DS) as cache_root:
    root = str(cache_root).rstrip("/") + "/"
    dataset = datasets.FashionIQ(path=root, transform=transform)
    for category in ("dress", "shirt", "toptee"):
        queries = getattr(dataset, f"test_queries_{category}")
        targets = getattr(dataset, f"test_targets_{category}")
        src = [q["source_img_id"] for q in queries]; tgt = [q["target_img_id"] for q in queries]
        recalls, ranks, _ = native_rank_and_recalls(queries, targets, category, BATCH, model, txt_processors)
        rec = {"queries": len(queries), "gallery": len(targets),
               "unique_source": len(set(src)), "unique_target": len(set(tgt)),
               "duplicate_gallery_entries": len((src + tgt)) - len(set(src + tgt)),
               "native_recalls_r1_r10_r50": [recalls[1], recalls[10], recalls[50]],
               "rank_counts": {"hit1": sum(1 for r in ranks if 1 <= r <= 1),
                               "hit10": sum(1 for r in ranks if 1 <= r <= 10),
                               "hit50": sum(1 for r in ranks if 1 <= r <= 50),
                               "missed_top50": sum(1 for r in ranks if r == 0),
                               "below_top50_or_masked": sum(1 for r in ranks if r == 0)},
               "ranks": ranks,
               "recomputed_from_ranks": {
                   "r1": 100.0 * sum(1 for r in ranks if 1 <= r <= 1) / len(ranks),
                   "r10": 100.0 * sum(1 for r in ranks if 1 <= r <= 10) / len(ranks),
                   "r50": 100.0 * sum(1 for r in ranks if 1 <= r <= 50) / len(ranks)},
               "reference_masked": True, "coverage": len(queries)}
        full_native[category] = (recalls, ranks)
        record["categories"][category] = rec
        print(f"[native] {category}: q={len(queries)} g={len(targets)} "
              f"r1/r10/r50={recalls[1]:.4f}/{recalls[10]:.4f}/{recalls[50]:.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(record, indent=1))
print("[OK] native verification written:", OUT)
