"""G-02: does the R3 harness's mask actually EXCLUDE the reference, as its docstring says?

Both scripts do `row[mask] = -10e10` and then `np.argsort(-row)`. Negating after setting a large
NEGATIVE similarity makes the masked entry the LARGEST value of `-row`, so it sorts FIRST. The
reference is therefore promoted to rank 1 rather than excluded. This script measures that on the
harness's own expression instead of arguing about it.
"""
import numpy as np

row = np.array([0.9, 0.8, 0.1], dtype=np.float32)  # 0 = reference, 1 = target, 2 = other
TARGET, REFERENCE = 1, 0


def harness_rank(row, target, mask_reference):
    r = row.copy()
    if mask_reference is not None:
        r[mask_reference] = -10e10
    order = np.argsort(-r, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1, order


def excluded_rank(row, target, mask_reference):
    """What 'exclude the reference' means: drop it, then rank. No sentinel involved."""
    keep = [i for i in range(len(row)) if i != mask_reference]
    order = np.argsort(-row[keep], kind="stable")
    return int(np.flatnonzero(np.asarray(keep)[order] == target)[0]) + 1, np.asarray(keep)[order]


h_all, order_all = harness_rank(row, TARGET, None)
h_masked, order_masked = harness_rank(row, TARGET, REFERENCE)
e_masked, order_excluded = excluded_rank(row, TARGET, REFERENCE)
print("scores:", row.tolist())
print("harness unmasked rank:", h_all, "order:", order_all.tolist())
print("harness masked   rank:", h_masked, "order:", order_masked.tolist())
print("true 'exclude reference' rank:", e_masked, "order:", order_excluded.tolist())
print("masked_order_puts_reference_first:", int(order_masked[0]) == REFERENCE)
print("harness_mask_equals_exclusion:", h_masked == e_masked)
print("conclusion:",
      "the mask PROMOTES the reference to rank 1 instead of excluding it"
      if int(order_masked[0]) == REFERENCE and h_masked != e_masked
      else "the mask behaves as an exclusion here")
