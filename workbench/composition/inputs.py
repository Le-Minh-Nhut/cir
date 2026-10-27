"""Pure PIL input interventions for the CIRCUS composition conditions."""
from __future__ import annotations

from PIL import Image

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CONDITION_LABELS = {
    "multimodal": "Image + text",
    "text_only_black_image": "Text only",
    "image_only_empty_text": "Image only",
}


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
    raise ValueError(f"unknown composition condition: {condition!r}")


def condition_label(condition: str) -> str:
    """Return concise display text for a known condition."""
    try:
        return CONDITION_LABELS[condition]
    except KeyError:
        raise ValueError(f"unknown composition condition: {condition!r}") from None
