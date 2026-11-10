"""Cross-check every number quoted in PARITY.vi.md against a measurement file."""
import json, re
from pathlib import Path
B = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair")
r2 = json.loads((B/"raw/r2_transform_tensors.json").read_text())
r3 = json.loads((B/"raw/r3_tie_harness.json").read_text())
r3real = json.loads((B/"raw/r3_real_case.txt").read_text().split("R3REAL:")[1])
r4 = json.loads((B/"raw/r4_relation_p1.json").read_text()) if (B/"raw/r4_relation_p1.json").is_file() else None
r4d = json.loads((B/"raw/r4_determinism_p1.json").read_text()) if (B/"raw/r4_determinism_p1.json").is_file() else None
print("R2", {k: r2[k] for k in ("preprocess_val_repeatable","train_vs_val_same_image","control_val_vs_val")})
print("R3 rank", r3["rank_move"], "gaps", r3["probe_gap_unmasked_matrix"], r3["probe_gap_masked_matrix"], "crossed", r3["crossed_positions"])
d = r3real["dress"]["native_train_vs_native_val"]
print("R3real", [x for x in d if "moved" in x])
print("R4", None if r4 is None else {"keys": r4["checkpoint"]["keys"], "rel": r4["checkpoint"]["relation_keys"],
      "sd": r4["constructions"][0]["state_dict_entries"]})
print("R4det", None if r4d is None else {k: r4d[k] for k in ("unexpected_keys","unseeded_a_vs_b_max_abs","seeded_same_seed_max_abs","seeded_diff_seed_max_abs","registered_parameters","is_parameter")})
