from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
        embeddings: Tensor | np.ndarray | None = None,
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
    def _mine_mmap(
        self,
        current_query: Tensor,
        supervision: SupervisionBatch,
    ) -> CommonNegativeSet:
        assert isinstance(self.embeddings, np.ndarray)
        batch, retrieval_dim = current_query.shape
        queries = F.normalize(current_query.detach().float(), dim=-1)
        excluded_bank_ids = set(supervision.target_ids)
        retained_scores = [torch.empty(0, device=current_query.device) for _ in range(batch)]
        retained_embeddings = [
            torch.empty(0, retrieval_dim, device=current_query.device) for _ in range(batch)
        ]
        retained_ids: list[list[str]] = [[] for _ in range(batch)]

        def merge(index: int, scores: Tensor, values: Tensor, ids: list[str]) -> None:
            if not ids:
                return
            combined_scores = torch.cat([retained_scores[index], scores])
            combined_values = torch.cat([retained_embeddings[index], values])
            combined_ids = retained_ids[index] + ids
            count = min(self.hard_negatives, combined_scores.numel())
            chosen = combined_scores.topk(count).indices
            retained_scores[index] = combined_scores[chosen]
            retained_embeddings[index] = combined_values[chosen]
            retained_ids[index] = [combined_ids[item] for item in chosen.tolist()]

        target_pool = F.normalize(supervision.target_embedding.detach().float(), dim=-1)
        for index in range(batch):
            positives = set(supervision.positive_ids[index]) | {supervision.target_ids[index]}
            unique: dict[str, int] = {}
            for candidate, image_id in enumerate(supervision.target_ids):
                if image_id not in positives and image_id not in unique:
                    unique[image_id] = candidate
            indices = list(unique.values())
            if indices:
                values = target_pool[indices]
                merge(
                    index,
                    queries[index] @ values.T,
                    values,
                    list(unique),
                )

        chunk_size = 8_192
        for start in range(0, len(self.ids), chunk_size):
            chunk_ids = self.ids[start : start + chunk_size]
            allowed_positions = [
                position
                for position, image_id in enumerate(chunk_ids)
                if image_id not in excluded_bank_ids
            ]
            if not allowed_positions:
                continue
            raw = np.array(self.embeddings[start : start + len(chunk_ids)], copy=True)
            block = F.normalize(torch.from_numpy(raw).to(current_query.device).float(), dim=-1)
            allowed = torch.tensor(allowed_positions, device=current_query.device)
            values = block[allowed]
            ids = [chunk_ids[position] for position in allowed_positions]
            scores = queries @ values.T
            for index in range(batch):
                positives = set(supervision.positive_ids[index]) | {supervision.target_ids[index]}
                valid_positions = [
                    position for position, image_id in enumerate(ids) if image_id not in positives
                ]
                if not valid_positions:
                    continue
                valid = torch.tensor(valid_positions, device=current_query.device)
                count = min(self.hard_negatives, len(valid_positions))
                chosen_local = scores[index, valid].topk(count).indices
                chosen = valid[chosen_local]
                merge(
                    index,
                    scores[index, chosen],
                    values[chosen],
                    [ids[item] for item in chosen.tolist()],
                )

        rows: list[Tensor] = []
        row_ids: list[tuple[str, ...]] = []
        for index in range(batch):
            if not retained_ids[index]:
                raise RuntimeError("No valid negatives remain after positive filtering")
            if len(retained_ids[index]) < self.hard_negatives:
                repeats = (
                    self.hard_negatives + len(retained_ids[index]) - 1
                ) // len(retained_ids[index])
                retained_embeddings[index] = retained_embeddings[index].repeat(repeats, 1)[
                    : self.hard_negatives
                ]
                retained_ids[index] = (retained_ids[index] * repeats)[: self.hard_negatives]
            rows.append(retained_embeddings[index])
            row_ids.append(tuple(retained_ids[index]))
        result = CommonNegativeSet(torch.stack(rows), tuple(row_ids))
        result.validate(batch, retrieval_dim)
        return result

    @torch.no_grad()
    def mine_once(
        self,
        current_query: Tensor,
        supervision: SupervisionBatch,
    ) -> CommonNegativeSet:
        batch, retrieval_dim = current_query.shape
        supervision.validate(batch, retrieval_dim)
        if isinstance(self.embeddings, np.ndarray):
            return self._mine_mmap(current_query, supervision)
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
