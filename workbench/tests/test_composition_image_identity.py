"""The composition dataset identity must follow image bytes, not file metadata."""
from __future__ import annotations

import os
from pathlib import Path

from workbench.composition import runner


def test_image_identity_follows_content_not_mtime(tmp_path: Path) -> None:
    """Same path, same size, restored mtime, different bytes: the identity must change."""
    image = tmp_path / "gallery.jpg"
    image.write_bytes(b"AAAA")
    stat = image.stat()
    atime_ns, mtime_ns = stat.st_atime_ns, stat.st_mtime_ns
    original = runner._image_identity([image])

    image.write_bytes(b"BBBB")
    os.utime(image, ns=(atime_ns, mtime_ns))
    assert image.stat().st_mtime_ns == mtime_ns
    assert image.stat().st_size == stat.st_size
    assert runner._image_identity([image]) != original, "changed bytes must change the identity"

    image.write_bytes(b"AAAA")
    os.utime(image, ns=(atime_ns, mtime_ns))
    assert runner._image_identity([image]) == original, "identical bytes must reproduce the identity"


def test_image_identity_rolls_up_per_file_digests(tmp_path: Path) -> None:
    """The identity carries one rolled-up digest, not a raw 64-char digest per file."""
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    first.write_bytes(b"AAAA")
    second.write_bytes(b"BBBB")
    identity = runner._image_identity([first, second])

    assert isinstance(identity["content_digest"], str)
    assert not any(isinstance(value, list) and any(isinstance(item, str) and len(item) == 64 for item in value)
                   for value in identity.values()), "raw per-file digests must not be stored"
    assert not any(isinstance(value, list) and len(value) == 2 for value in identity.values())

    first.write_bytes(b"CCCC")
    assert runner._image_identity([first, second])["content_digest"] != identity["content_digest"]
