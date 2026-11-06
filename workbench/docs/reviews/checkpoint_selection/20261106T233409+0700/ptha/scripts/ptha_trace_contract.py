"""PTHA: real tensor contract for the failing CIRR->FashionIQ prediction path.

Contract as the pinned author code actually builds it (validate_blip.py:169,194):
  index_features = (image_features, image_embeds_frozen)      # utils.py:80
  name_to_feat   = dict(zip(names, index_features[-1]))        # -> RAW vision embeds (1408)
  feature_curr   = index_features[0]                           # -> Q-Former states (768, unprojected)
  inference(reference_image_features, feature_curr, captions)
"""
import sys, json
from pathlib import Path
CIR = Path("/home/heheboiz/data/cir"); CKPT = Path(sys.argv[1]); DS = CIR / "data/FashionIQ"
SR = CIR / "workbench/third_party/MTST_PTHA/MTST_PTHA"
sys.path.insert(0, str(CIR)); sys.path.insert(0, str(SR / "src"))
import torch, torch.nn.functional as F
from omegaconf import OmegaConf
from lavis.common.registry import registry
from workbench.replay.ptha_mtst import _fashioniq_dataset_binding
from workbench.replay.common import sha256_file

rec = {"checkpoint_sha256": sha256_file(CKPT)}
state = torch.load(CKPT, map_location="cpu", weights_only=True)["Blip2QformerCirCat"]
cls = registry.get_model_class("blip2_cir_cat")
cfg = OmegaConf.load(cls.default_config_path("pretrain")); cfg.model.load_pretrained = False
model = cls.from_config(cfg.model); model.load_state_dict(state, strict=True); model = model.cuda().eval()
rec.update({"class": f"{type(model).__module__}.{type(model).__name__}",
            "qformer_hidden": model.Qformer.config.hidden_size,
            "vision_encoder_width": model.visual_encoder.num_features,
            "vision_proj": list(model.vision_proj.weight.shape),
            "text_proj": list(model.text_proj.weight.shape),
            "vision_proj_params": sum(p.numel() for p in model.vision_proj.parameters())})
print(json.dumps(rec, indent=1))

with _fashioniq_dataset_binding(DS) as du:
    from data_utils import FashionIQDataset
    tf = du.targetpad_transform(1.25, 224)
    classic = FashionIQDataset("val", ["dress"], "classic", tf)
    relative = FashionIQDataset("val", ["dress"], "relative", tf)
    names = list(classic.image_names)
    ref_name, tgt_name, caps = relative[0]
    print("relative[0]:", ref_name, "->", tgt_name, caps)

    want = [ref_name] + [n for n in names if n != ref_name][:2]
    idx = [names.index(n) for n in want]
    imgs = torch.stack([classic[i][1] for i in idx]).cuda()
    with torch.no_grad():
        image_features, image_embeds_frozen = model.extract_target_features(imgs, mode="mean")
    print("extract_target_features -> image_features:", tuple(image_features.shape),
          "| image_embeds_frozen:", tuple(image_embeds_frozen.shape))

    name_to_feat = dict(zip(want, image_embeds_frozen))     # RAW vision embeds, per validate_blip.py:169
    ref = name_to_feat[ref_name].unsqueeze(0)
    feats = image_features                                   # unprojected Q-Former states
    text = [f"{caps[0].strip('.?, ').capitalize()} and {caps[1].strip('.?, ')}"]
    print("reference_image_features:", tuple(ref.shape), "| target_feats:", tuple(feats.shape))

    print("--- A) author code as written ---")
    try:
        with torch.no_grad():
            simA = model.inference(ref, feats, text)
        print("OK:", tuple(simA.shape))
    except RuntimeError as e:
        print("FAILED:", e)
        print("  fusion_feats side: [B,1,1,%d]" % model.text_proj.out_features,
              "| target side after permute:", tuple(feats.permute(0, 2, 1).shape))

    print("--- B) targets through the checkpoint's OWN trained vision_proj ---")
    with torch.no_grad():
        projected = model.inference_target(feats)
    print("inference_target ->", tuple(projected.shape),
          "| norms:", [round(float(x), 6) for x in projected.norm(dim=-1)[0][:3]])
    with torch.no_grad():
        manual = F.normalize(model.vision_proj(feats.cuda()), dim=-1)
    print("boundary equivalence vs inference_target, max_abs_diff:",
          float((manual.cpu() - projected.cpu()).abs().max()))
    with torch.no_grad():
        simB = model.inference(ref, projected, text)
    print("OK:", tuple(simB.shape), "finite:", bool(torch.isfinite(simB).all()))
    print("  sim min %.6f max %.6f" % (simB.min(), simB.max()))
    with torch.no_grad():
        twice = model.inference_target(projected.cuda())
    print("double projection would change values, max_abs_diff:",
          float((twice.cpu() - projected.cpu()).abs().max()))
    rec.update({"before_shapes": {"reference": list(ref.shape), "target": list(feats.shape)},
                "after_shapes": {"target": list(projected.shape)},
                "boundary_equivalence_max_abs_diff": float((manual.cpu() - projected.cpu()).abs().max())})
