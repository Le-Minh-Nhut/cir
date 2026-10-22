"""Publish ENCODER's staged observation through the canonical ResultRun schema."""
from workbench.replay.limn_publish import main, publish_payload

__all__ = ["publish_payload", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
