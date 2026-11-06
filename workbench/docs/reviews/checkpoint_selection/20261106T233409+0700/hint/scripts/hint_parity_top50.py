"""HINT parity: wrapper path vs hand-transcribed native path, per-query TOP-50 INDEX SETS.

Two independent paths over the SAME restored model, SAME dataset object, SAME batch size:

  wrapper path = the pinned `test.py:test()` that `workbench.replay.hint` calls, instrumented by
                 intercepting `np.argsort` so the exact `nn_result` lists it produced are captured
                 (the function only returns recalls, so interception is the only faithful capture).
  native path   = the hand transcription in hint_verify.py (already stored and matching to 1e-9).

Agreement criterion, declared before running: for every query, the top-50 index SET from the wrapper
equals the top-50 index SET from the transcription, and hit counts at 1/10/50 are identical.
Ties in scores are reported rather than hidden.
"""
import json, sys
from pathlib import Path

CIR = Path("/home/heheboiz/data/cir"); SR = CIR / "workbench/third_party/HINT"
DS = CIR / "data/FashionIQ"; CKPT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
sys.path.insert(0, str(CIR))
import numpy as np

from workbench.replay.hint import _load_replay_modules, CHECKPOINT, PROTOCOL, DATASET_LABEL
from workbench.replay.common import load_trusted_whole_model, isolated_cache_root
from types import SimpleNamespace

torch, datasets, pinned, processors = _load_replay_modules(SR, "pretrain")
data_utils, test_module = pinned
txt_processors, _boundary = processors

model = load_trusted_whole_model(CKPT, CHECKPOINT["expected_sha256"], torch).cuda().eval()
transform = data_utils.targetpad_transform(1.25, 224)
params = SimpleNamespace(batch_size=BATCH, dataset=DATASET_LABEL)

record = {"protocol": PROTOCOL, "batch_size": BATCH,
          "criterion": "top-50 index set equality per query; hit@1/10/50 identical",
          "categories": {}}

with isolated_cache_root(DS) as cache_root:
    dataset = datasets.FashionIQ(path=str(cache_root).rstrip("/") + "/", transform=transform)
    for category in ("dress", "shirt", "toptee"):
        queries = getattr(dataset, f"test_queries_{category}")
        targets = getattr(dataset, f"test_targets_{category}")
        tid = [t["target_img_id"] for t in targets]

        # ---- wrapper path, instrumented ----
        captured = []
        real_argsort = np.argsort
        def spy(a, *args, **kwargs):
            out = real_argsort(a, *args, **kwargs)
            captured.append(np.asarray(out))
            return out
        np.argsort = spy
        try:
            recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
        finally:
            np.argsort = real_argsort
        # test.py calls argsort twice per query (the 50-slice list comp, then the k-loop reuses it);
        # keep the widest captures, which are the [:50] lists
        wrapper_nn = [c[:50].tolist() for c in captured if c.ndim == 1 and c.size >= 50][: len(queries)]
        assert len(wrapper_nn) == len(queries), (len(wrapper_nn), len(queries))

        # ---- native transcription path (same ops, written out) ----
        with torch.no_grad():
            all_q, imgs, mods = [], [], []
            for i, t in enumerate(queries):
                imgs.append(t["source_img_data"]); mods.append(t["mod"]["str"])
                if len(imgs) >= BATCH or i == len(queries) - 1:
                    b = torch.stack([torch.from_numpy(d) if "torch" not in str(type(d)) else d
                                     for d in imgs]).float().cuda()
                    all_q.append(model.extract_retrieval_compose(
                        b, [txt_processors["eval"](c) for c in mods]).data.cpu().numpy())
                    imgs, mods = [], []
            all_q = np.concatenate(all_q)
            all_t, imgs = [], []
            for i, t in enumerate(targets):
                imgs.append(t["target_img_data"])
                if len(imgs) >= BATCH or i == len(targets) - 1:
                    b = torch.stack([torch.from_numpy(d) if "torch" not in str(type(d)) else d
                                     for d in imgs]).float().cuda()
                    all_t.append(model.extract_retrieval_target(b).data.cpu().numpy()); imgs = []
            all_t = np.concatenate(all_t)
        sims = np.matmul(all_q, all_t).squeeze().max(-1)
        for i, t in enumerate(queries):
            sims[i, tid.index(t["source_img_id"])] = -10e10
        native_nn = [np.argsort(-sims[i, :])[:50].tolist() for i in range(sims.shape[0])]

        set_equal = sum(1 for a, b in zip(wrapper_nn, native_nn) if set(a) == set(b))
        order_equal = sum(1 for a, b in zip(wrapper_nn, native_nn) if a == b)
        hit = lambda nn, k: sum(1 for i, ns in enumerate(nn) if tid.index(queries[i]["target_img_id"]) in ns[:k])
        # tie density inside the captured window, so equality is not claimed over hidden ties
        tie_rows = int(np.sum(np.diff(np.sort(sims, axis=1), axis=1) == 0))

        record["categories"][category] = {
            "queries": len(queries), "gallery": len(targets),
            "wrapper_recalls": {k: recalls[f"{category}_r{k}"] for k in (1, 10, 50)},
            "wrapper_hits": {f"hit{k}": hit(wrapper_nn, k) for k in (1, 10, 50)},
            "native_hits": {f"hit{k}": hit(native_nn, k) for k in (1, 10, 50)},
            "top50_set_equal_queries": set_equal,
            "top50_order_identical_queries": order_equal,
            "top50_set_mismatch_queries": len(queries) - set_equal,
            "score_tie_pairs_in_row": tie_rows,
            "first_mismatch": next(({"query": i, "wrapper": a[:10], "native": b[:10]}
                                    for i, (a, b) in enumerate(zip(wrapper_nn, native_nn))
                                    if set(a) != set(b)), None),
        }
        print(f"[parity] {category}: set_equal={set_equal}/{len(queries)} "
              f"order_identical={order_equal} hits_wrapper={record['categories'][category]['wrapper_hits']} "
              f"hits_native={record['categories'][category]['native_hits']}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(record, indent=1))
print("[OK] parity written:", OUT)
