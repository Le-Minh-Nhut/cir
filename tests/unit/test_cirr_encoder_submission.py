from __future__ import annotations

import torch

from evaluation.cirr_encoder import build_cirr_test_submissions


def test_cirr_test1_submission_uses_reference_excluded_global_order() -> None:
    fillers = [f"filler_{index:02d}" for index in range(49)]
    gallery = ["ref", "target", "outside", "g1", "g2", "g3", *fillers]
    desired_order = ["ref", "outside", "g2", "target", "g1", "g3", *fillers]
    score_by_id = {
        image_id: float(len(desired_order) - rank)
        for rank, image_id in enumerate(desired_order)
    }
    scores = torch.tensor([[score_by_id[image_id] for image_id in gallery]])

    global_submission, subset_submission = build_cirr_test_submissions(
        scores,
        pair_ids=["123"],
        reference_ids=["ref"],
        group_members=[["ref", "g1", "target", "g2", "g3"]],
        gallery_ids=gallery,
    )

    assert global_submission["version"] == "rc2"
    assert global_submission["metric"] == "recall"
    assert subset_submission["version"] == "rc2"
    assert subset_submission["metric"] == "recall_subset"
    assert set(global_submission) == {"version", "metric", "123"}
    assert set(subset_submission) == {"version", "metric", "123"}

    global_ranking = global_submission["123"]
    assert isinstance(global_ranking, list)
    assert len(global_ranking) == 50
    assert "ref" not in global_ranking
    assert global_ranking[:5] == ["outside", "g2", "target", "g1", "g3"]
    assert global_ranking == desired_order[1:51]

    # This must be the global order filtered to group members, not a rerank.
    assert subset_submission["123"] == ["g2", "target", "g1"]
