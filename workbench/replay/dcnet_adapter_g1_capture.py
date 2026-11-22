"""Capture production DCNet Adapter features and score components for native G1 rows."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.composition.model_dcnet import Adapter

ROWS = (657, 933, 1362, 1980)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config()
    native_root = args.native_capture.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_rows = []
    for row_id in ROWS:
        with np.load(native_root / f"row-{row_id}.npz", allow_pickle=False) as row:
            source_rows.append({"candidate": str(row["candidate"].item()),
                                "target": str(row["target"].item())})
    gallery_ids = np.load(native_root / "gallery_ids.npy", allow_pickle=False).astype(str).tolist()
    with (Path(config.FASHIONIQ_ROOT) / "captions" / "cap.dress.glove.val.pkl").open("rb") as stream:
        rows = pickle.load(stream)
    query_rows = [rows[row_id] for row_id in ROWS]
    if [(row["candidate"], row["target"]) for row in query_rows] != [
            (row["candidate"], row["target"]) for row in source_rows]:
        raise ValueError("native G1 rows differ from FashionIQ author source rows")
    image_root = Path(config.FASHIONIQ_ROOT) / "resized_images"
    query_paths = [image_root / f"{row['candidate']}.jpg" for row in query_rows]
    gallery_paths = [image_root / f"{image_id}.jpg" for image_id in gallery_ids]
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    references = [gallery_index[row["candidate"]] for row in query_rows]
    targets = np.asarray([gallery_index[row["target"]] for row in query_rows], dtype=np.int64)
    adapter = Adapter(config, checkpoint_id="fashioniq_run_directory")
    try:
        adapter.prepare_identity()
        interpreter = adapter.require_verified_interpreter()
        adapter.work_root = args.output / "adapter-work"
        adapter.batch_size = 4
        query_texts = [" and ".join(row["captions"]) for row in query_rows]
        queries = adapter.encode_queries(query_paths, query_texts, [None] * len(query_rows))
        adapter.batch_size = 100
        gallery = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        adapter.batch_size = 4
        scores = adapter.score(queries, gallery, reference_indices=references,
                               capture_components_path=args.output / "components")
        np.save(args.output / "query_context.npy", queries, allow_pickle=False)
        np.save(args.output / "gallery_context.npy", gallery, allow_pickle=False)
        np.save(args.output / "scores.npy", scores, allow_pickle=False)
        np.save(args.output / "gallery_ids.npy", np.asarray(gallery_ids), allow_pickle=False)
        np.save(args.output / "query_ids.npy", np.asarray([row["candidate"] for row in query_rows]),
                allow_pickle=False)
        np.save(args.output / "target_indices.npy", targets, allow_pickle=False)
        (args.output / "receipt.json").write_text(json.dumps({
            "status": "adapter_selected_g1_complete", "rows": list(ROWS),
            "gallery_count": len(gallery_ids), "controller": __import__("sys").executable,
            "worker_interpreter": interpreter, "worker_identity": adapter.worker_identity,
            "checkpoint_sha256": adapter._checkpoint["sha256"],
            "score_components": ["components.composition.npy", "components.correction.npy"],
            "query_features": "query_context.npy", "gallery_features": "gallery_context.npy",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
