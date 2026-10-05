from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id


@dataclass(frozen=True)
class EvalRequest:
    model_id: str
    checkpoint_id: str
    protocol_id: str
    dataset_root: Path
    output_path: Path
    top_k: int = 200


class ModelAdapter(ABC):
    model_id: str

    def __init__(self, registry: dict[str, Any] | None = None) -> None:
        self.model = model_by_id(self.model_id, registry)

    def validate_request(self, request: EvalRequest) -> dict[str, Any]:
        if request.protocol_id not in self.model["supported_protocols"]:
            raise ValueError(f"{self.model_id} does not support {request.protocol_id}")
        checkpoint = checkpoint_by_id(self.model, request.checkpoint_id)
        path = checkpoint_path(self.model_id, checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {path}")
        if checkpoint["checkpoint_mapping_status"] == "UNVERIFIED":
            raise ValueError(f"checkpoint mapping unresolved: {request.checkpoint_id}")
        return checkpoint

    @abstractmethod
    def build_command(self, request: EvalRequest) -> list[str]:
        """Build official evaluation command. Caller executes only on approved GPU host."""

    def normalize_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload
