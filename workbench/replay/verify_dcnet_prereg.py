"""Assert sealed DCNet preregistration source digests before launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    aliases = {
        "model/model.py": root / "workbench/third_party/DCNet/model/model.py",
        "model/metric.py": root / "workbench/third_party/DCNet/model/metric.py",
        "model/composition.py": root / "workbench/third_party/DCNet/model/composition.py",
        "trainer/trainer_joint.py": root / "workbench/third_party/DCNet/trainer/trainer_joint.py",
        "adapter/workbench.composition.model_dcnet.py": root / "workbench/composition/model_dcnet.py",
        "runner/workbench.composition.runner.py": root / "workbench/composition/runner.py",
        "composition/workbench.composition.adapters.py": root / "workbench/composition/adapters.py",
        "data/workbench.composition.data.py": root / "workbench/composition/data.py",
        "cli/workbench.scripts.composition_eval.py": root / "workbench/scripts/composition_eval.py",
        "native_capture/workbench.replay.dcnet_native_circus.py": root / "workbench/replay/dcnet_native_circus.py",
    }
    expected = prereg["source"]["files_sha256"]
    actual = {name: digest(path) for name, path in aliases.items()}
    mismatch = {name: (expected.get(name), value) for name, value in actual.items()
                if expected.get(name) != value}
    if mismatch:
        raise SystemExit(json.dumps({"status": "SOURCE_DIGEST_MISMATCH", "mismatch": mismatch}, sort_keys=True))
    print(json.dumps({"status": "SOURCE_DIGESTS_MATCH", "prereg": str(args.prereg), "files": actual}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
