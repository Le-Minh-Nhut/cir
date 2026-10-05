from __future__ import annotations

from workbench.backend.schemas.results import ResultRun


def build_mock_runs() -> list[ResultRun]:
    runs = []
    for protocol, suffix, checkpoint, noise, shift in (("fashioniq_original_split", "a", "fiq_n02", 20, 0), ("fashioniq_original_split", "b", "fiq_n05", 50, 1), ("fashioniq_val_split", "a", "pair_b1", None, 0), ("fashioniq_val_split", "b", "pair_b2", None, 1)):
        queries = []
        for index in range(24):
            category = ("dress", "shirt", "toptee")[index % 3]
            reference, target = f"mock-ref-{index:02d}", f"mock-target-{index:02d}"
            rank = 1 if index == 0 else 230 if index == 1 else (3 if index == 2 and shift == 0 else 180 if index == 2 else 180 if index == 3 and shift == 0 else 3 if index == 3 else min(250, 5 + (index * 17 + shift * 23) % 220))
            top = [{"rank": position, "image_id": target if position == rank else "common-top-1" if position == 1 else "common-top-5" if position == 5 else f"mock-distractor-{index:02d}-{position:03d}", "score": round(1 - position / 300 - shift / 10000, 6)} for position in range(1, 201)]
            queries.append({"query_id": f"{category}:{index}:{reference}:{target}", "category": category, "annotation_index": index, "reference_id": reference, "target_id": target, "raw_captions": [f"mock change {index}", f"mock detail {index}"], "model_input_text": f"Mock composed request {index}, variant {suffix}", "target_rank": rank, "top_results": top})
        ranks = [query["target_rank"] for query in queries]
        r10, r50 = (100 * sum(rank <= k for rank in ranks) / len(ranks) for k in (10, 50))
        runs.append(ResultRun.model_validate({"schema_version": 2, "run": {"run_id": f"mock-{'habit' if protocol.endswith('original_split') else 'pair'}-{checkpoint}", "dataset": "FashionIQ", "split": "val", "protocol_id": protocol, "evaluation_noise_pct": 0, "model_id": "mock_habit" if protocol.endswith("original_split") else "mock_pair", "method_name": "Mock HABIT" if protocol.endswith("original_split") else "Mock PAIR", "checkpoint_id": checkpoint, "checkpoint_training_noise_pct": noise, "top_k_saved": 200, "gallery_size": 260, "timestamp": "2026-10-05T00:00:00Z", "data_kind": "mock", "reported_paper_metrics": {}, "reproduced_metrics": {"r10": r10, "r50": r50, "mean": (r10 + r50) / 2}}, "queries": queries}))
    return runs
