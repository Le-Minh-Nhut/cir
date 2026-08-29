from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.optim import SGD

from losses.objective import IAGSRMEObjective, ObjectiveConfig
from models.iag_srme import FGCLIPBackbone, FGCLIPRegime, IAGSRME, IAGSRMEConfig, IAGSRMECore
from training.engine import assert_training_setup, trainable_parameters


def _parameter_change(parameter: torch.nn.Parameter, before: torch.Tensor) -> float:
    return float((parameter.detach() - before).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Real pinned FG-CLIP/IAG-SRME smoke update")
    parser.add_argument("--checkpoint", default="qihoo360/fg-clip-base")
    parser.add_argument(
        "--revision", default="454d76372c2cf5eb48fa0d871fd0534481484d97"
    )
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(20260829)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regime = FGCLIPRegime(
        checkpoint=args.checkpoint,
        revision=args.revision,
        train_vision=True,
        train_text=True,
    )
    backbone = FGCLIPBackbone.from_pretrained(regime, internal_width=256)
    tokenizer, processor = FGCLIPBackbone.load_processor(
        regime.checkpoint, regime.revision, regime.trust_remote_code
    )
    core = IAGSRMECore(
        IAGSRMEConfig(
            width=256,
            num_candidates=4,
            max_steps=args.max_steps,
            num_heads=8,
            retrieval_dim=backbone.retrieval_dim,
            selector_gumbel_noise=False,
        )
    )
    with torch.no_grad():
        core.scorer.score_head[-1].weight.zero_()
        core.scorer.score_head[-1].bias.fill_(0.5)
    model = IAGSRME(backbone, core).to(device).train()
    objective = IAGSRMEObjective(ObjectiveConfig(), width=256).to(device).train()
    optimizer = SGD(trainable_parameters(model, objective), lr=1e-3)
    assert_training_setup(model, objective, optimizer, device)

    arrays = [
        np.full((224, 224, 3), (48, 96, 160), dtype=np.uint8),
        np.full((224, 224, 3), (176, 72, 40), dtype=np.uint8),
        np.full((224, 224, 3), (32, 152, 88), dtype=np.uint8),
        np.full((224, 224, 3), (144, 128, 56), dtype=np.uint8),
    ]
    images = [Image.fromarray(array) for array in arrays]
    reference_pixels = processor(images=images[:2], return_tensors="pt")["pixel_values"].to(
        device
    )
    target_pixels = processor(images=images[2:], return_tensors="pt")["pixel_values"].to(
        device
    )
    tokenized = tokenizer(
        ["make it red and add long sleeves", "make it green and remove the collar"],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    content_mask = attention_mask.bool()

    with torch.no_grad():
        official_dense = model.backbone.model.get_image_dense_features(
            pixel_values=reference_pixels[:1]
        )
        official_global = model.backbone.model.get_image_features(
            pixel_values=reference_pixels[:1]
        )
        official_text = model.backbone.model.text_model(
            input_ids=input_ids[:1],
            attention_mask=attention_mask[:1],
            return_dict=True,
            walk_short_pos=True,
        )
        official_text_projected = model.backbone.model.text_projection(
            official_text.pooler_output
        )

    tracked = {
        "intent_query_bank": model.core.intent_encoder.query_bank,
        "grounding_projection": model.core.grounder.intent_projection.weight,
        "editor_direction": model.core.editor.direction.weight,
        "readout_projection": model.core.readout.output_projection.weight,
        "score_head": model.core.scorer.score_head[-1].weight,
        "text_encoder": next(model.backbone.model.text_model.parameters()),
        "vision_encoder": next(model.backbone.model.vision_model.parameters()),
    }
    before = {name: parameter.detach().clone() for name, parameter in tracked.items()}
    optimizer.zero_grad(set_to_none=True)
    encoded = model.backbone(reference_pixels, input_ids, attention_mask, content_mask)
    output = model.core(encoded)
    target_embeddings = model.encode_global_images(target_pixels)
    positives = torch.eye(2, dtype=torch.bool, device=device)
    losses = objective(output, target_embeddings, positives)
    losses["total"].backward()
    gradient_norms = {
        name: 0.0 if parameter.grad is None else float(parameter.grad.norm())
        for name, parameter in tracked.items()
    }
    optimizer.step()
    parameter_deltas = {
        name: _parameter_change(parameter, before[name]) for name, parameter in tracked.items()
    }
    finite_tensors = all(
        torch.isfinite(tensor).all()
        for tensor in (
            encoded.anchor,
            encoded.reference_global,
            encoded.text_tokens,
            output.final_query,
            target_embeddings,
            losses["terminal"],
            losses["marginal"],
        )
    )
    if not finite_tensors:
        raise AssertionError("real FG-CLIP smoke produced non-finite values")
    if not all(value > 0 for value in gradient_norms.values()):
        raise AssertionError(f"expected nonzero real gradients: {gradient_norms}")
    if not all(value > 0 for value in parameter_deltas.values()):
        raise AssertionError(f"expected real parameter changes: {parameter_deltas}")
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "revision": args.revision,
                "device": str(device),
                "official_shapes": {
                    "get_image_dense_features": list(official_dense.shape),
                    "get_image_features": list(official_global.shape),
                    "text_model_last_hidden_state": list(official_text.last_hidden_state.shape),
                    "text_projection": list(official_text_projected.shape),
                },
                "iag_srme_shapes": {
                    "anchor": list(encoded.anchor.shape),
                    "text_tokens": list(encoded.text_tokens.shape),
                    "intents": list(output.intents.shape),
                    "supports": list(output.supports.shape),
                    "candidate_states": list(output.trace[0].candidate_states.shape),
                    "candidate_queries": list(output.trace[0].candidate_queries.shape),
                    "scores": list(output.trace[0].scores.shape),
                    "final_query": list(output.final_query.shape),
                    "target_global": list(target_embeddings.shape),
                },
                "losses": {
                    "terminal": float(losses["terminal"].detach()),
                    "marginal": float(losses["marginal"].detach()),
                    "total": float(losses["total"].detach()),
                },
                "gradient_norms": gradient_norms,
                "parameter_max_abs_deltas": parameter_deltas,
                "finite": finite_tensors,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
