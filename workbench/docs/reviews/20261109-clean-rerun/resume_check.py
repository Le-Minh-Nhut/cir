"""Cheap --resume check: does a SECOND invocation on an existing run find its feature cache?

    resume_check.py <run_id> <model> <checkpoint>

Reads the run's recorded identity, recomputes every feature-memmap fingerprint the way
`_cached_features` does, and compares against the sidecars on disk. It does not score anything and
does not start a model interpreter. Reports which memmaps a resumed run would reuse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("model")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", type=Path, default=Path.home() / "cir_rerun_20261109.yaml")
    args = parser.parse_args()

    from workbench.backend import autopilot
    from workbench.composition import runner

    config = autopilot._workbench_config(autopilot.load_config(args.config))
    artifacts = Path(config.WORKBENCH_ARTIFACTS_ROOT)
    run_dir = artifacts / "composition" / "runs" / args.run_id
    rank_path = run_dir / "ranks" / f"{args.model}_{args.checkpoint}.json"
    if not rank_path.is_file():
        print(json.dumps({"run_id": args.run_id, "status": "NO_RANK_ARTIFACT"}))
        return 1
    payload = json.loads(rank_path.read_text())
    identity = payload["identity"]
    fingerprint = runner._digest(identity)
    stored = payload["fingerprint"]
    work_root = run_dir / "work" / args.model
    scopes = tuple(identity.get("category_scopes") or (None,))
    conditions = tuple(identity.get("conditions") or ())

    found = {}
    for scope in scopes:
        name = runner._scope_name(scope)
        gallery = work_root / "features" / f"gallery_{name}.npy"
        expected = runner._digest({"run": fingerprint, "scope": scope,
                                   "kind": "full-gallery-features"})
        actual = runner._memmap_metadata_shape(gallery)
        sidecar = json.loads(gallery.with_suffix(gallery.suffix + ".json").read_text()) \
            if gallery.with_suffix(gallery.suffix + ".json").is_file() else {}
        found[f"gallery_{name}"] = {"on_disk": gallery.is_file(),
                                    "sidecar_fingerprint_matches": sidecar.get("fingerprint") == expected,
                                    "shape_matches": actual is not None and
                                    sidecar.get("shape") == list(actual),
                                    "has_content_digest": bool(sidecar.get("content_digest"))}
        for condition in conditions:
            query = work_root / "features" / f"query_{condition}_{name}.npy"
            q_expected = runner._digest({"run": fingerprint, "condition": condition,
                                         "scope": scope, "kind": "query-features"})
            q_sidecar = {}
            if query.with_suffix(query.suffix + ".json").is_file():
                q_sidecar = json.loads(query.with_suffix(query.suffix + ".json").read_text())
            found[f"query_{condition}_{name}"] = {
                "on_disk": query.is_file(),
                "sidecar_fingerprint_matches": q_sidecar.get("fingerprint") == q_expected,
                "has_content_digest": bool(q_sidecar.get("content_digest"))}

    loads = {}
    for key, value in found.items():
        path = work_root / "features" / f"{key}.npy"
        scope_part = key.split("_")[-1]
        scope = None if scope_part == "global" else scope_part
        if key.startswith("gallery_"):
            loads[key] = runner._load_memmap(
                path, runner._digest({"run": fingerprint, "scope": scope,
                                      "kind": "full-gallery-features"})) is not None
        else:
            condition = key[len("query_"):-len("_" + scope_part)]
            loads[key] = runner._load_memmap(
                path, runner._digest({"run": fingerprint, "condition": condition,
                                      "scope": scope, "kind": "query-features"})) is not None
    reusable = bool(found) and all(loads.values())
    print(json.dumps({"run_id": args.run_id, "model": args.model,
                      "identity_fingerprint_matches_stored": fingerprint == stored,
                      "cache_verify_mode": identity.get("cache_verify_mode"),
                      "files": found, "loads_under_current_fingerprint": loads,
                      "resume_would_reuse_cache": reusable}, indent=1))
    return 0 if reusable else 2


if __name__ == "__main__":
    raise SystemExit(main())
