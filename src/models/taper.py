"""Compatibility import for the clean TAPER-MAG V4 implementation.

The historical slot-ownership/QASA/primitive-router implementation was intentionally removed on
this branch. New code should import from :mod:`models.taper_mag` directly.
"""

from models.taper_mag import TaperMAG, TaperMAGConfig

TAPER = TaperMAG

__all__ = ["TAPER", "TaperMAG", "TaperMAGConfig"]
