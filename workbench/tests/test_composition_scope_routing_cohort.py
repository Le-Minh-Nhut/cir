"""The published CLVC gap over two cohorts: one whose routing read the edit caption, one that did not.

The defect this pins: CLVC-Net's routing uses the release loader's assignment for 5 884 queries and
`(reference stem, edit caption)` for the 117 the loader left unscoped. `clvc_image_encoder_only`
removes the edit from the EMBEDDING, so those 117 still had their category chosen by text the
condition claims to have removed. The size of that on the gap is a measurement, not an opinion:

* cohort A — all 6 001 scored queries: must reproduce the published `0.33061712371309204` exactly.
  A derivation that drifts from the artifact is a bug in the derivation, not a new number.
* cohort B — the 5 884 loader-scoped queries: caption-free by construction, gap `0.33400015944039774`.

Every number here is read from the rank table already on disk (`runs/da0a2162.../ranks`) and
recomputed with `metrics.compose`; nothing is re-encoded or re-scored, and no published file is
written. Skips cleanly when the artefacts or the release are absent.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "da0a2162be0e159a1a456cec"
COMPOSITION = REPO_ROOT / "workbench/artifacts/composition"
RELEASE = REPO_ROOT / "workbench/artifacts/circus_release"
FASHIONIQ = REPO_ROOT / "data/FashionIQ"

#: The two published numbers, quoted so a drift is a failure and not a silent redefinition.
COHORT_A_GAP = 0.33061712371309204
COHORT_A_COUNT = 6001
COHORT_B_GAP = 0.33400015944039774
COHORT_B_COUNT = 5884
#: A qid the loader left unscoped and the caption annotation resolved: it MUST be absent from B.
CAPTION_ROUTED_QID = "7:2179"


def _script_module():
    path = REPO_ROOT / "workbench/scripts/derive_routing_cohort_gap.py"
    spec = importlib.util.spec_from_file_location("derive_routing_cohort_gap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_artifacts() -> Path:
    ranks = COMPOSITION / "runs" / RUN_ID / "ranks" / "clvc_net_author_drive_unresolved.json"
    if not ranks.is_file():
        pytest.skip(f"run {RUN_ID} is not on this host (external artifact)")
    if not any(COMPOSITION.glob(f"summary/*_{RUN_ID}/full.json")):
        pytest.skip(f"run {RUN_ID} has no summary on this host")
    if not (FASHIONIQ / "captions/cap.shirt.val.json").is_file():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    if not (RELEASE / "mbeir_fashioniq_task7_test.jsonl").is_file():
        pytest.skip("CIRCUS release is not present on this host (external artifact)")
    return ranks


def _derive() -> dict:
    """One call, with the roots pinned: the suite clears operator settings on purpose."""
    return _script_module().derive(COMPOSITION, "clvc_net", RUN_ID,
                                  release_root=RELEASE, fashioniq_root=FASHIONIQ)


def test_cohort_a_reproduces_the_published_gap() -> None:
    """The derivation is only usable if it lands on the artifact's own number to 1e-12."""
    _require_artifacts()
    record = _derive()
    cohort = record["cohort_a_all_scored"]
    assert cohort["reproduces_published"] is True
    assert cohort["composition_gap_ndcg"] == pytest.approx(COHORT_A_GAP, abs=1e-12)
    assert cohort["published_gap_ndcg"] == pytest.approx(COHORT_A_GAP, abs=1e-12)
    assert cohort["gap_cohort_count"] == COHORT_A_COUNT
    assert cohort["caption_free"] is False


def test_cohort_b_excludes_exactly_the_caption_routed_queries() -> None:
    """5 884, and a named caption-routed qid is not in it: the exclusion is not an aggregate guess."""
    _require_artifacts()
    module = _script_module()
    record = _derive()
    routing = record["routing"]
    assert routing["replay_agrees_with_artifact"] is True
    assert routing["loader_scoped_count"] == COHORT_B_COUNT
    assert routing["resolver_routed_count"] == 117
    assert routing["still_unscoped_count"] == 2
    assert routing["resolver_key"] == "reference_stem+edit_caption"
    assert CAPTION_ROUTED_QID in routing["resolver_routed_qids"]
    # Same 6 003 queries, partitioned: 5 884 + 117 + 2, nothing dropped and nothing double counted.
    assert (routing["loader_scoped_count"] + routing["resolver_routed_count"]
            + routing["still_unscoped_count"]) == 6003
    cohort = record["cohort_b_loader_scoped"]
    assert cohort["query_count"] == COHORT_B_COUNT
    assert cohort["gap_cohort_count"] == COHORT_B_COUNT
    assert cohort["composition_gap_ndcg"] == pytest.approx(COHORT_B_GAP, abs=1e-12)
    assert cohort["caption_free"] is True


def test_the_two_cohorts_differ_and_both_are_reported() -> None:
    """A gap that did not move would mean the 117 were not caption-routed; state both, plainly."""
    _require_artifacts()
    record = _derive()
    a = record["cohort_a_all_scored"]["composition_gap_ndcg"]
    b = record["cohort_b_loader_scoped"]["composition_gap_ndcg"]
    assert a != b
    assert abs(a - b) < 0.01, "a large move would mean the 117 dominate the cohort"
    assert "117" in record["claim_narrowed"]
    assert "capton" not in record["claim_narrowed"]
    assert "edit caption" in record["claim_narrowed"]


def test_the_derivation_records_the_digests_it_read() -> None:
    """A derived number nobody can trace back to a rank table is not evidence."""
    import hashlib

    ranks = _require_artifacts()
    record = _derive()
    recorded = record["inputs"]["ranks"]
    assert recorded["sha256"] == hashlib.sha256(ranks.read_bytes()).hexdigest()
    assert Path(recorded["path"]).name == ranks.name
    payload = json.loads(ranks.read_text(encoding="utf-8"))
    assert payload["identity"]["query_scope_resolution"] == (
        "3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87")


def test_the_derivation_refuses_to_rewrite_a_published_summary(tmp_path: Path) -> None:
    """Published artifacts are immutable: the writer refuses an existing destination by name."""
    _require_artifacts()
    module = _script_module()
    destination = tmp_path / "derived.json"
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to rewrite"), \
            _pinned_roots(module, RELEASE, FASHIONIQ):
        module.main(["--run", f"clvc_net:{RUN_ID}", "--output", str(destination)])


class _pinned_roots:
    """Point `main`'s routing replay at the staged release without an operator config file."""

    def __init__(self, module, release: Path, fashioniq: Path) -> None:
        self.module, self.release, self.fashioniq = module, release, fashioniq

    def __enter__(self):
        self.original = self.module.routing_reasons
        self.module.routing_reasons = (
            lambda model_id, release_root=None, fashioniq_root=None:
            self.original(model_id, self.release, self.fashioniq))
        return self

    def __exit__(self, *_exc):
        self.module.routing_reasons = self.original
        return False
