"""Pure PIL input interventions for the CIRCUS composition conditions."""
from __future__ import annotations

from PIL import Image

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
#: Adapter-declared ABLATION conditions. Deliberately NOT part of `CONDITIONS`: that tuple is the
#: protocol's three conditions, and every model's `mode_support`/`condition_status` block is keyed on
#: it, so widening it would add a fourth `ABLATION_UNSUPPORTED` row to every existing model's run and
#: change every run id for nothing. A caller opts in by naming one of these explicitly.
ABLATION_CONDITIONS = ("clvc_image_encoder_only",)
CONDITION_LABELS = {
    "multimodal": "Image + text",
    "text_only_black_image": "Text only",
    "image_only_empty_text": "Image only",
    # The row is `test.py:116`'s target expression applied to the reference image: no text tower at
    # all. Named separately from `image_only_empty_text`, which drives the text tower with an empty
    # string and which CLVC-Net cannot encode at all.
    "clvc_image_encoder_only": "Image only (image encoder)",
}

#: Which adapter method encodes a condition's query rows. The default is the shared
#: `encode_queries(paths, texts, categories)`. An entry here means the adapter has a dedicated
#: method with a different signature, and `runner._encode_queries` calls THAT, so a condition that
#: transports no text cannot accidentally be handed a caption.
CONDITION_ENCODERS = {"clvc_image_encoder_only": "encode_image_only_queries"}
DEFAULT_CONDITION_ENCODER = "encode_queries"

def encoder_method(condition: str) -> str:
    """The adapter method that encodes `condition`'s query rows."""
    return CONDITION_ENCODERS.get(condition, DEFAULT_CONDITION_ENCODER)


def black_image_like(image: Image.Image) -> Image.Image:
    """Return a fresh, zero-valued image with exactly the input's size and mode."""
    return Image.new(image.mode, image.size, 0)


def adapt_query_inputs(image: Image.Image, text: str, condition: str) -> tuple[Image.Image, str]:
    """Return untransformed PIL pixels and text for one condition."""
    if condition == "multimodal":
        return image, text
    if condition == "text_only_black_image":
        return black_image_like(image), text
    if condition == "image_only_empty_text":
        return image, ""
    if condition in ABLATION_CONDITIONS:
        # Reference image UNCHANGED plus an empty text. The encoder this condition maps to takes no
        # text at all (`CONDITION_ENCODERS`), so the empty string is never encoded; the untouched
        # image is what makes the row equal the reference image's own gallery row up to batch
        # composition.
        return image, ""
    raise ValueError(f"unknown composition condition: {condition!r}")


def condition_label(condition: str) -> str:
    """Return concise display text for a known condition."""
    try:
        return CONDITION_LABELS[condition]
    except KeyError:
        raise ValueError(f"unknown composition condition: {condition!r}") from None
