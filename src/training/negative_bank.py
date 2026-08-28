from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from models.taper_mag.contracts import SupervisionBatch


@dataclass(frozen=True, slots=True)
class CommonNegativeSet:
    embeddings: Tensor
    ids: tuple[tuple[str, ...], ...]

    def validate(self, batch_size: int, retrieval_dim: int) -> None:
        if self.embeddings.ndim != 3 or self.embeddings.shape[0] != batch_size:
            raise ValueError("negative embeddings must be [B,H,D]")
        if self.embeddings.shape[-1] != retrieval_dim:
            raise ValueError("negative embedding dimension mismatch")
        if len(self.ids) != batch_size:
            raise ValueError("negative IDs must align with the batch")
        if any(len(row) != self.embeddings.shape[1] for row in self.ids):
            raise ValueError("negative ID count must match H")


class NegativeBank:
    """Detached official-training image bank with ID-aware common mining."""

    def __init__(
        self,
        embeddings: Tensor | None = None,
        ids: tuple[str, ...] = (),
        *,
        hard_negatives: int = 64,
    ) -> None:
        if hard_negatives <= 0:
            raise ValueError("hard_negatives must be positive")
        if embeddings is not None and (embeddings.ndim != 2 or embeddings.shape[0] != len(ids)):
            raise ValueError("Bank embeddings/IDs mismatch")
        self.embeddings = embeddings
        self.ids = ids
        self.hard_negatives = hard_negatives

    @torch.no_grad()
    def mine_once(
        self,
        current_query: Tensor,
        supervision: SupervisionBatch,
    ) -> CommonNegativeSet:
        batch, retrieval_dim = current_query.shape
        supervision.validate(batch, retrieval_dim)
        pools = [supervision.target_embedding.detach().float()]
        pool_ids = list(supervision.target_ids)
        if self.embeddings is not None:
            if self.embeddings.shape[1] != retrieval_dim:
                raise ValueError("Negative bank feature dimension mismatch")
            pools.append(self.embeddings.detach().float().to(current_query.device))
            pool_ids.extend(self.ids)
        pool = F.normalize(torch.cat(pools, dim=0), dim=-1)
        queries = F.normalize(current_query.detach().float(), dim=-1)
        all_scores = queries @ pool.T
        rows: list[Tensor] = []
        row_ids: list[tuple[str, ...]] = []
        for index in range(batch):
            positives = set(supervision.positive_ids[index]) | {supervision.target_ids[index]}
            allowed_indices = [
                candidate
                for candidate, image_id in enumerate(pool_ids)
                if image_id not in positives
            ]
            # Remove duplicate image IDs before top-k so a repeated target cannot dominate.
            deduplicated: list[int] = []
            seen: set[str] = set()
            for candidate in allowed_indices:
                image_id = pool_ids[candidate]
                if image_id not in seen:
                    seen.add(image_id)
                    deduplicated.append(candidate)
            if not deduplicated:
                raise RuntimeError("No valid negatives remain after positive filtering")
            allowed = torch.tensor(deduplicated, device=all_scores.device, dtype=torch.long)
            count = min(self.hard_negatives, allowed.numel())
            chosen_local = all_scores[index, allowed].topk(count).indices
            chosen = allowed[chosen_local]
            # Fixed H is required for vectorization; repeat the hardest valid entries if small tests
            # contain fewer unique negatives. IDs remain explicit and known positives stay absent.
            if count < self.hard_negatives:
                repeats = chosen.repeat((self.hard_negatives + count - 1) // count)
                chosen = repeats[: self.hard_negatives]
            rows.append(pool[chosen])
            row_ids.append(tuple(pool_ids[item] for item in chosen.tolist()))
        result = CommonNegativeSet(torch.stack(rows), tuple(row_ids))
        result.validate(batch, retrieval_dim)
        return result
