from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    details: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}
