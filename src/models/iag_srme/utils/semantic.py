from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


PARSER_VERSION = "fashioniq_concepts_v2"

# FashionIQ captions are short attribute edits. Removing only grammatical/action
# boilerplate keeps the parser deterministic without pretending to be a full NLP system.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "change",
        "changes",
        "changing",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "make",
        "makes",
        "making",
        "of",
        "on",
        "than",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "with",
    }
)

# These tokens carry meaning inside a phrase ("more colorful", "no sleeves",
# "v neck") but are too ambiguous to become useful standalone concepts.
PHRASE_ONLY_WORDS = frozenset({"less", "more", "no", "not", "t", "v", "without"})


def parse_instruction_concepts(text: str) -> tuple[str, ...]:
    """Return versioned unigram/bigram concepts from instruction text only."""

    normalized = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    content = [token for token in tokens if token not in STOP_WORDS]
    concepts = {token for token in content if token not in PHRASE_ONLY_WORDS}
    # Phrase boundaries remain faithful to the instruction: do not synthesize a
    # bigram across removed grammar (for example, "shirt is black").
    concepts.update(
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:], strict=False)
        if left not in STOP_WORDS and right not in STOP_WORDS
    )
    return tuple(sorted(concepts))


@dataclass(frozen=True, slots=True)
class ConceptVocabulary:
    concepts: tuple[str, ...]
    parser_version: str = PARSER_VERSION

    @classmethod
    def build(
        cls,
        training_instructions: Sequence[str],
        *,
        min_frequency: int = 2,
        max_size: int = 2048,
    ) -> "ConceptVocabulary":
        counts: Counter[str] = Counter()
        for instruction in training_instructions:
            counts.update(parse_instruction_concepts(instruction))
        ordered = sorted(
            (item for item in counts.items() if item[1] >= min_frequency),
            key=lambda item: (-item[1], item[0]),
        )
        return cls(tuple(concept for concept, _ in ordered[:max_size]))

    @property
    def fingerprint(self) -> str:
        payload = f"{self.parser_version}\n" + "\n".join(self.concepts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def labels(self, instructions: Sequence[str], device: torch.device) -> Tensor:
        index = {concept: position for position, concept in enumerate(self.concepts)}
        labels = torch.zeros(len(instructions), len(self.concepts), dtype=torch.bool)
        for row, instruction in enumerate(instructions):
            for concept in parse_instruction_concepts(instruction):
                if concept in index:
                    labels[row, index[concept]] = True
        return labels.to(device)
