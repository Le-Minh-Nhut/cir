"""Phase 1/2: restore both PAIR checkpoints under an EXPLORATORY_PUBLIC_MAPPING and audit the
restored object graph against the inference call graph of the public methods.

Reports, per file:
  * which classes actually ended up bound to each restored node
  * restored state vs the attributes the public inference path reads
  * which restored submodules the public inference path never touches
  * exact missing-attribute errors, if any

No forward pass is executed here; this is the load/audit stage.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compat_loader  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Everything public PAIR inference reads, by owning class.
PAIR_ROOT_ATTRS = ["backbone", "affine", "loss_T", "local_weight", "local_weight_dis", "t"]
BACKBONE_ATTRS = ["clip", "tokenizer", "hidden_dim", "fc", "text_fc", "transformer",
                  "transformer_a", "transformer_b", "mlp", "mlp_img", "mlp_text", "local_token_num"]
BACKBONE_METHODS = ["visual_out", "text_out", "extract_img_fea", "extract_img_fea_patch_selection"]
PAIR_METHODS = ["compose_feature", "target_fea", "extract_retrieval_compose", "extract_retrieval_target"]
VISUAL_ATTRS = ["conv1", "class_embedding", "positional_embedding", "patch_dropout", "ln_pre",
                "transformer", "ln_post", "proj", "pool_type"]
CLIP_ATTRS = ["token_embedding", "positional_embedding", "transformer", "ln_final",
              "text_projection", "text_pool_type", "attn_mask"]

# Public inference touches transformer (extract_img_fea), transformer_a and transformer_b
# (extract_img_fea_patch_selection) only.
INFERENCE_MODULES = ("transformer", "transformer_a", "transformer_b")
BYPASSED_MODULES = ("transformer_c", "transformer_d", "weight_chanelImg", "weight_chanelImg_t",
                    "cross_t2i", "mlp_weighted", "weighted")


def audit(obj, tag, hypothesis, meta):
    rep = {"tag": tag, "hypothesis": hypothesis, "sha256": meta["sha256"],
           "root_class": f"{type(obj).__module__}.{type(obj).__name__}",
           "mapping": meta["mapping"], "problems": []}

    rep["root_attrs_present"] = {a: hasattr(obj, a) for a in PAIR_ROOT_ATTRS}
    rep["root_methods_present"] = {m: callable(getattr(obj, m, None)) for m in PAIR_METHODS}
    bb = getattr(obj, "backbone", None)
    rep["has_backbone"] = bb is not None
    if bb is None:
        rep["problems"].append("no .backbone")
        return rep

    rep["backbone_class"] = f"{type(bb).__module__}.{type(bb).__name__}"
    rep["backbone_attrs_present"] = {a: hasattr(bb, a) for a in BACKBONE_ATTRS}
    rep["backbone_methods_present"] = {m: callable(getattr(bb, m, None)) for m in BACKBONE_METHODS}
    rep["backbone_scalars"] = {k: getattr(bb, k) for k in ("hidden_dim", "local_token_num")
                               if hasattr(bb, k)}

    clip = getattr(bb, "clip", None)
    rep["clip_class"] = f"{type(clip).__module__}.{type(clip).__name__}" if clip is not None else None
    rep["clip_attrs_present"] = {a: hasattr(clip, a) for a in CLIP_ATTRS} if clip is not None else {}
    vis = getattr(clip, "visual", None) if clip is not None else None
    rep["visual_attrs_present"] = {a: hasattr(vis, a) for a in VISUAL_ATTRS} if vis is not None else {}
    rep["tokenizer_class"] = (f"{type(bb.tokenizer).__module__}.{type(bb.tokenizer).__name__}"
                              if hasattr(bb, "tokenizer") else None)

    # Restored vs required attention class per branch, and reverse-vs-normal softmax provenance.
    branch_report = {}
    for name in INFERENCE_MODULES + ("transformer_c", "transformer_d"):
        mod = getattr(bb, name, None)
        if mod is None:
            branch_report[name] = {"present": False, "used_by_public_inference": name in INFERENCE_MODULES}
            continue
        layers = getattr(mod, "layers", [])
        attns = []
        for layer in layers:
            attn = getattr(layer, "attn", None)
            attns.append(f"{type(attn).__module__}.{type(attn).__name__}" if attn is not None else None)
        branch_report[name] = {
            "present": True,
            "used_by_public_inference": name in INFERENCE_MODULES,
            "class": f"{type(mod).__module__}.{type(mod).__name__}",
            "n_layers": len(layers),
            "attn_classes": attns,
            "num_heads": (layers[0].attn.num_heads if layers and hasattr(layers[0].attn, "num_heads") else None),
        }
    rep["branches"] = branch_report

    # Parameters reachable from the public inference path.
    used, bypassed, unreachable = set(), set(), set()
    for k in obj.state_dict().keys():
        head = k.split(".")[1] if k.startswith("backbone.") else None
        if head is None:
            used.add(k)
        elif head in INFERENCE_MODULES or head in ("clip", "fc", "text_fc", "mlp", "mlp_img", "mlp_text"):
            used.add(k)
        elif head in ("transformer_c", "transformer_d", "weight_chanelImg", "weight_chanelImg_t", "cross_t2i"):
            bypassed.add(k)
        else:
            unreachable.add(k)
    sd = obj.state_dict()
    rep["param_counts"] = {
        "total": int(sum(v.numel() for v in sd.values())),
        "reached_by_public_inference": int(sum(sd[k].numel() for k in used)),
        "bypassed_by_public_inference": int(sum(sd[k].numel() for k in bypassed)),
        "unclassified": int(sum(sd[k].numel() for k in unreachable)),
    }
    rep["bypassed_modules"] = sorted({k.split(".")[1] for k in bypassed})
    rep["unclassified_keys"] = sorted(unreachable)[:20]

    # Attribute gaps that would break the public call graph.
    for a, ok in rep["root_attrs_present"].items():
        if not ok:
            rep["problems"].append(f"PAIR.{a} missing")
    for m, ok in rep["root_methods_present"].items():
        if not ok:
            rep["problems"].append(f"PAIR.{m} missing")
    for a, ok in rep["backbone_attrs_present"].items():
        if not ok:
            rep["problems"].append(f"Backbone.{a} missing")
    for m, ok in rep["backbone_methods_present"].items():
        if not ok:
            rep["problems"].append(f"Backbone.{m} missing")
    for a, ok in rep["clip_attrs_present"].items():
        if not ok:
            rep["problems"].append(f"CLIP.{a} missing")
    for a, ok in rep["visual_attrs_present"].items():
        if not ok:
            rep["problems"].append(f"VisualTransformer.{a} missing")
    return rep


def main():
    ckpt_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    results = {}
    for tag in ("B1", "B2"):
        for hyp in compat_loader.ATTENTION_HYPOTHESES:
            key = f"{tag}/{hyp}"
            try:
                obj, meta = compat_loader.load(str(ckpt_dir / f"pair-{tag}.pt"), tag, hyp)
                results[key] = audit(obj, tag, hyp, meta)
                print(f"[OK] {key}: problems={results[key]['problems']}")
            except Exception as exc:  # noqa: BLE001 - each attempt must be recorded verbatim
                results[key] = {"tag": tag, "hypothesis": hyp, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
