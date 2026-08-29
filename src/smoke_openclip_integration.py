from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from torch.optim import SGD

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import (
    BackboneBuildSpec,
    IAGSRME,
    IAGSRMEConfig,
    IAGSRMECore,
    build_backbone,
)
from training.engine import trainable_parameters


MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
LIBRARY_VERSION = "3.3.0"
WEIGHTS_REPOSITORY = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
WEIGHTS_REVISION = "7288da5a0d6f0b51c4a2b27c624837a9236d0112"


def _gradient_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().norm())


def main() -> None:
    parser = argparse.ArgumentParser(description="Real OpenCLIP/IAG-SRME integration smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(401)
    spec = BackboneBuildSpec(
        backbone_type="openclip",
        checkpoint=MODEL_NAME,
        revision=PRETRAINED,
        library_version=LIBRARY_VERSION,
        weights_repository=WEIGHTS_REPOSITORY,
        weights_revision=WEIGHTS_REVISION,
        train_vision=True,
        train_text=True,
        train_text_projection=False,
    )
    backbone, tokenizer, processor = build_backbone(spec, internal_width=256)
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=256,
            num_candidates=4,
            max_steps=3,
            num_heads=8,
            retrieval_dim=backbone.retrieval_dim,
            lambda_z=0.10,
            query_cap=0.50,
            selector_temperature=1.0,
            selector_gumbel_noise=False,
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
    model = IAGSRME(backbone, core).to(device).train()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=256).to(device)
    optimizer = SGD(trainable_parameters(model, objective), lr=1e-4)

    reference_images = [Image.new("RGB", (256, 256), color) for color in ("red", "blue")]
    target_images = [Image.new("RGB", (256, 256), color) for color in ("green", "yellow")]
    reference_pixels = processor.preprocess(reference_images, return_tensors="pt")[
        "pixel_values"
    ].to(device)
    target_pixels = processor.preprocess(target_images, return_tensors="pt")[
        "pixel_values"
    ].to(device)
    tokenized = tokenizer(
        ["Make it green and add sleeves", "Make it yellow and remove the collar"],
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    content_mask = attention_mask.clone()
    content_mask[:, 0] = False
    final_positions = attention_mask.sum(dim=1).sub(1)
    content_mask.scatter_(1, final_positions[:, None], False)

    model.eval()
    with torch.no_grad():
        _, reference_global_from_intermediates = model.backbone.encode_reference_images(
            reference_pixels
        )
        global_only_from_encode_image = model.encode_global_images(reference_pixels)
    torch.testing.assert_close(
        reference_global_from_intermediates,
        global_only_from_encode_image,
        atol=1e-6,
        rtol=1e-5,
    )
    global_equivalence_max_error = float(
        (reference_global_from_intermediates - global_only_from_encode_image).abs().max()
    )
    global_equivalence_cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_global_from_intermediates.float(),
            global_only_from_encode_image.float(),
            dim=-1,
        ).mean()
    )
    model.train()

    tracked = {
        "openclip_vision": next(model.backbone.vision_encoder_parameters()),
        "openclip_visual_projection": model.backbone.model.visual.proj,
        "openclip_text": next(model.backbone.text_encoder_parameters()),
        "anchor_adapter": model.backbone.anchor_projection[0].weight,
        "text_adapter": model.backbone.text_projection[0].weight,
        "intent": model.core.intent_encoder.query_bank,
        "grounder": model.core.grounder.intent_projection.weight,
        "context": model.core.context_fuser.fusion[0].weight,
        "editor": model.core.editor.direction.weight,
        "readout": model.core.readout.output_projection.weight,
        "scorer": model.core.scorer.score_head[-1].weight,
    }
    before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    optimizer.zero_grad(set_to_none=True)
    output = model(reference_pixels, input_ids, attention_mask, content_mask)
    target_embeddings = model.encode_global_images(target_pixels)
    losses = objective(output, target_embeddings, torch.eye(2, dtype=torch.bool, device=device))
    losses["total"].backward()
    if model.backbone.model.text_projection.grad is not None:
        raise AssertionError("frozen OpenCLIP text retrieval projection received a gradient")
    if model.backbone.model.logit_scale.grad is not None:
        raise AssertionError("frozen OpenCLIP logit scale received a gradient")
    gradient_norms = {name: _gradient_norm(parameter) for name, parameter in tracked.items()}
    if not all(value > 0.0 for value in gradient_norms.values()):
        raise AssertionError(f"expected nonzero OpenCLIP smoke gradients: {gradient_norms}")
    optimizer.step()
    parameter_deltas = {
        name: float((parameter.detach() - before[name]).abs().max())
        for name, parameter in tracked.items()
    }
    if not all(value > 0.0 for value in parameter_deltas.values()):
        raise AssertionError(f"expected OpenCLIP smoke parameter updates: {parameter_deltas}")
    finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (output.anchor, output.supports, output.final_query, *losses.values())
    )
    if not finite:
        raise FloatingPointError("non-finite real OpenCLIP integration smoke output")
    print(
        json.dumps(
            {
                "model": MODEL_NAME,
                "pretrained": PRETRAINED,
                "open_clip_torch": LIBRARY_VERSION,
                "weights_repository": WEIGHTS_REPOSITORY,
                "weights_revision": WEIGHTS_REVISION,
                "device": str(device),
                "shapes": {
                    "pixels": list(reference_pixels.shape),
                    "anchor": list(output.anchor.shape),
                    "reference_global": list(output.reference_global.shape),
                    "text_tokens": list(output.text_tokens.shape),
                    "intents": list(output.intents.shape),
                    "supports": list(output.supports.shape),
                    "final_query": list(output.final_query.shape),
                    "contextual_patches_before_visual_projection": [
                        reference_pixels.shape[0],
                        model.backbone.patch_tokens,
                        int(model.backbone.model.visual.proj.shape[0]),
                    ],
                },
                "reference_global_vs_global_only_max_abs_error": (
                    global_equivalence_max_error
                ),
                "reference_global_vs_global_only_cosine": global_equivalence_cosine,
                "losses": {name: float(value.detach()) for name, value in losses.items()},
                "gradient_norms": gradient_norms,
                "parameter_max_abs_deltas": parameter_deltas,
                "finite": finite,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
