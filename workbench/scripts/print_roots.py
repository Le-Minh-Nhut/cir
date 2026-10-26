#!/usr/bin/env python3
"""Print and check that autopilot YAML and standalone roots agree."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.autopilot import load_config
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import REGISTRY_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / "cir_pc.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    yaml_config = load_config(args.config)
    standalone_config = resolve_config()
    yaml_paths = {key: str(value) for key, value in yaml_config.paths.items()}
    standalone = {
        "dataset_root": standalone_config.FASHIONIQ_ROOT,
        "artifacts": standalone_config.WORKBENCH_ARTIFACTS_ROOT,
        "checkpoints": standalone_config.WORKBENCH_CHECKPOINT_ROOT,
        "results": standalone_config.WORKBENCH_RESULTS_ROOT,
        "third_party": standalone_config.WORKBENCH_THIRD_PARTY_ROOT,
    }
    standalone = {key: str(value) for key, value in standalone.items()}
    registry = Path(os.environ.get("WORKBENCH_REGISTRY_PATH") or REGISTRY_PATH).resolve()
    mismatches = [f"paths.{key}" for key, value in standalone.items()
                  if yaml_paths[key] != value]
    if standalone_config.CIR_REPO_ROOT != Path.cwd().resolve():
        mismatches.append("CIR_REPO_ROOT")
    if registry != yaml_config.paths["registry"]:
        mismatches.append("paths.registry (WORKBENCH_REGISTRY_PATH/REGISTRY_PATH)")
    verdict = "MATCH" if not mismatches else "MISMATCH: " + ", ".join(mismatches)
    report = {"autopilot_yaml": yaml_paths, "standalone": standalone,
              "CIR_REPO_ROOT": str(standalone_config.CIR_REPO_ROOT),
              "resolved_registry": str(registry), "verdict": verdict}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for key, value in yaml_paths.items():
            print(f"autopilot_yaml.paths.{key}={value}")
        for key, value in standalone.items():
            print(f"standalone.{key}={value}")
        print(f"CIR_REPO_ROOT={report['CIR_REPO_ROOT']}")
        print(f"resolved_registry={registry}")
        print(f"verdict={verdict}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
