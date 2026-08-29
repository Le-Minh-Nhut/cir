import pytest

from models.iag_srme.backbone import assert_cache_legal


def test_full_vision_rejects_image_cache() -> None:
    with pytest.raises(ValueError, match="illegal"):
        assert_cache_legal(True, "features/fgclip/reference")
    assert_cache_legal(False, "features/fgclip/reference")
    assert_cache_legal(True, None)
