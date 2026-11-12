"""`GET /api/models` must reflect the assets on disk in the SAME process.

The defect: `main._MODELS_CACHE` was a module-level list with no key, so the first request of a
process pinned the document for that process's whole life. After the pinned ResNet-50 file was
acquired and verified, the live API still reported it missing while `POST /api/jobs` -- which
recomputes `preflight_blockers` per request -- had already dropped the blocker
(`raw/verify/95-external-asset-unblock.md` §"Did the API pick up the registry edit without a restart?").

The fixture clones the REAL registry documents and repoints `clvc_net`'s ResNet-50 asset at a
throwaway file, so the mechanism is exercised on the real readiness path while nothing outside the
test's own tmp directory is read or written. The two directions are asserted separately: a cached
document that ignores an ADDED file is the defect, and one that ignores a REMOVED file is the same
defect wearing the other mask.

The launch gates are deliberately NOT relaxed: `POST /api/jobs` (`runtime.require_launch_ready`) and
`_job_launch_guard` at dispatch must still refuse a blocked model, and that is asserted here too, so
"the badge updates" cannot be satisfied by weakening the refusal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO / "workbench" / "registry" / "models.yaml"
CONTRACTS_YAML = REPO / "workbench" / "registry" / "preparation_contracts.yaml"
AUX_YAML = REPO / "workbench" / "registry" / "auxiliary_assets.yaml"


@pytest.fixture()
def probe_registry(tmp_path, monkeypatch):
    """The real registry, with `clvc_net`'s ResNet-50 asset pointed at a throwaway file.

    Returns `(asset_path, models_doc)` where `models_doc()` calls the route handler in this process.
    """
    from workbench.backend import main, registry as registry_module

    asset = tmp_path / "resnet50-probe.pth"
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    contracts = yaml.safe_load(CONTRACTS_YAML.read_text(encoding="utf-8"))
    contract = next(item for item in contracts["contracts"]
                    if item["preparation_id"] == "clvc_net_fashioniq")
    target = next(item for item in contract["external_assets"]
                  if "ResNet-50" in item["name"])
    target["expected_local_location"] = str(asset)
    # The cache-shaped fields would relocate the watched path away from the fixture.
    for key in ("cache_path_suffix", "cache_root_env_var"):
        target.pop(key, None)
    # Keep only the probed asset required for evaluation, so this fixture does not depend on any
    # other external file being present on the host.
    for item in contract["external_assets"]:
        if item is not target:
            item["required_for_evaluation"] = False

    (tmp_path / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
    (tmp_path / "preparation_contracts.yaml").write_text(yaml.safe_dump(contracts), encoding="utf-8")

    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(tmp_path / "models.yaml"))
    # `preparation_contract_for_model` reads through `load_preparation_contracts()`, whose path is a
    # DEFAULT ARGUMENT bound at import, so repointing the module constant would not reach it. Patch
    # the resolver itself, which is the one function every readiness caller goes through.
    probe = next(item for item in contracts["contracts"]
                 if item["preparation_id"] == "clvc_net_fashioniq")
    monkeypatch.setattr(registry_module, "preparation_contract_for_model",
                        lambda model: probe if model.get("preparation_id") == "clvc_net_fashioniq"
                        else None)
    monkeypatch.setattr(registry_module, "_VALIDATED_CONFIG_CACHE", {})
    monkeypatch.setattr(registry_module, "_YAML_PARSE_CACHE", {})
    monkeypatch.setattr(main, "_MODELS_CACHE", None)
    monkeypatch.setattr(main, "_MODELS_CACHE_KEY", None)

    def clvc_blockers() -> list[str]:
        row = next(item for item in main.models() if item["model_id"] == "clvc_net")
        return list(row["checkpoints"][0]["blocking_reasons"])

    return asset, clvc_blockers


def test_a_newly_present_asset_changes_the_badge_in_the_same_process(probe_registry):
    """The defect, in the direction that was measured live: the file appears, the badge must follow.
    """
    asset, blockers = probe_registry
    before = blockers()
    assert any("ResNet-50" in reason for reason in before), before

    asset.write_bytes(b"weights" * 64)
    after = blockers()
    # The status must MOVE: it was "file missing", and now the file exists but carries no verifiable
    # digest. Asserting the new status (not merely "no ResNet line") is what distinguishes "the cache
    # noticed the file" from "the blocker changed word for word".
    assert any("checksum mismatch" in reason for reason in after), (
        "the asset appeared on disk but the cached document still describes its absence: "
        f"{after}")
    assert before != after

    # And back again: removing the file must be visible too, for the same reason.
    asset.unlink()
    removed = blockers()
    assert any("file missing" in reason for reason in removed), (
        f"the asset was removed but the cached document did not notice: {removed}")


def test_the_launch_gate_still_refuses_a_blocked_model(probe_registry):
    """Caching the DOCUMENT must not cache away the refusal.

    `POST /api/jobs` recomputes readiness per request and `_job_launch_guard` re-checks immediately
    before dispatch. Both are asserted, because "the badge updates" would be worthless if it were
    achieved by no longer refusing a blocked model.
    """
    from fastapi.testclient import TestClient

    from workbench.backend import main

    asset, blockers = probe_registry
    asset.write_bytes(b"weights" * 64)  # the file is present; the model is still not launchable
    assert not any("file missing" in reason for reason in blockers())

    client = TestClient(main.app)
    response = client.post("/api/jobs", json={
        "model_id": "clvc_net", "checkpoint_id": "author_drive_unresolved",
        "protocol_id": "fashioniq_val_split", "top_k": 10})
    assert response.status_code == 409, response.text
    assert "blockers" in response.text, response.text

    class _Job:
        model_id = "clvc_net"

    # The dispatch-time guard is the same rule, called directly: whatever names the model
    # unrunnable, the job must be refused rather than launched. On a host with no model interpreter
    # the environment blocker comes first; on one where the pinned entrypoint is the only blocker
    # the message names `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. Both are the same refusal.
    with pytest.raises(Exception) as raised:
        main._job_launch_guard(_Job())
    assert str(raised.value).strip(), raised.value


def test_a_run_with_results_stays_openable_while_a_new_run_is_refused(probe_registry, tmp_path):
    """A blocked model with an existing run: the run must stay readable, the new launch refused.

    Two separate questions, and the cache fix must not conflate them. The run-detail route reads
    artifacts and never consults readiness; the POST route consults readiness and never reads the
    run. Asserted in one process so a shared cache cannot make one answer depend on the other.
    """
    from fastapi.testclient import TestClient

    from workbench.backend import composition as composition_module
    from workbench.backend import main

    asset, blockers = probe_registry
    assert any("ResNet-50" in reason for reason in blockers())

    artifacts = tmp_path / "artifacts"
    summary = artifacts / "composition" / "summary" / "clvc_net_probe_done1" / "full.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "schema_version": 1, "model_id": "clvc_net", "checkpoint_id": "author_drive_unresolved",
        "run_id": "done1", "subset": "full", "protocol_id": "circus_fashioniq_release_v1",
        "conditions": ["multimodal", "text_only_black_image", "image_only_empty_text"],
        "query_count": 2, "gallery_count": 2, "execution_status": "COMPLETED",
        "metrics": {"coverage": 1.0, "gap_cohort_count": 2,
                    "multimodal": {"ndcg_full": 0.5, "recall_at_1": 50.0, "recall_at_10": 50.0,
                                   "recall_at_50": 50.0, "mrr": 0.5, "query_count": 2,
                                   "scored_count": 2, "coverage": 1.0}},
    }), encoding="utf-8")
    (artifacts / "composition" / "runs" / "done1" / "ranks").mkdir(parents=True)

    # The run that already has numbers stays readable even though the model cannot be launched now.
    detail = composition_module.get_composition_run_detail("done1", "full", artifacts)
    assert detail is not None
    assert detail["execution_status"] == "COMPLETED"
    assert detail["metrics"]["multimodal"]["recall_at_10"] == 50.0

    client = TestClient(main.app)
    refused = client.post("/api/jobs", json={
        "model_id": "clvc_net", "checkpoint_id": "author_drive_unresolved",
        "protocol_id": "fashioniq_val_split", "top_k": 10})
    assert refused.status_code == 409, refused.text
    # ... and the refused launch changed nothing about the readable run.
    assert composition_module.get_composition_run_detail("done1", "full", artifacts) is not None
