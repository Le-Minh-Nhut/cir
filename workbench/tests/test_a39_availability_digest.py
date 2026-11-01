import os
import time
from pathlib import Path

from workbench.backend import registry


def test_availability_digest_cache_tracks_member_stats_and_required_files(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = bundle / "model.pt"
    other = bundle / "other.pt"
    model.write_text("initial weights", encoding="utf-8")
    other.write_text("other", encoding="utf-8")
    registry._AVAILABILITY_DIGEST_CACHE.clear()

    first = registry._cached_availability_digest(bundle, ["model.pt"])
    model.write_text("changed weights", encoding="utf-8")
    os.utime(model, ns=(model.stat().st_atime_ns, model.stat().st_mtime_ns + 1_000_000))
    changed = registry._cached_availability_digest(bundle, ["model.pt"])
    different_members = registry._cached_availability_digest(bundle, ["other.pt"])

    assert first != changed, "member-file edit must invalidate cached digest"
    assert first != different_members, "different required_files must not collide"
