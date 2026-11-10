"""Phase A: hash the checkpoints each protocol row cites, resolved through the registry.

Never a `<root>/<model_id>/<filename>` join: `checkpoint_path()` owns where an artifact lives.
"""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend.registry import model_by_id, checkpoint_path

CASES = [
    ("encoder", "fashioniq"), ("hint", "fashioniq"),
    ("limn", "base_iter0_dress"), ("limn", "base_iter0_shirt"), ("limn", "base_iter0_toptee"),
    ("clip4cir_rn50x4_fullft", "rn50x4_fullft_unresolved"),
    ("combiner_rn50x4_noft", "rn50x4_noft_unresolved"),
    ("dcnet", "fashioniq_run_directory"), ("tme", "fashioniq_0"),
    ("habit", "fiq_n02"), ("intent", "fiq_n02"), ("conesep", "fiq_n02"),
    ("airknow", "fiq_n05"), ("ptha_mtst", "cirr_transfer"),
]

def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()

rows = ["model\tcheckpoint_id\tpath\tbytes\tsha256\tregistry_expected_sha256\tmatch"]
for model_id, checkpoint_id in CASES:
    model = model_by_id(model_id)
    variant = next((v for v in model["checkpoint_variants"] if v["checkpoint_id"] == checkpoint_id), None)
    if variant is None:
        rows.append(f"{model_id}\t{checkpoint_id}\tNO_SUCH_VARIANT\t\t\t\t")
        continue
    path = checkpoint_path(model_id, variant)
    if variant.get("artifact_type") == "bundle_directory" or variant.get("bundle_members"):
        members = variant.get("bundle_members") or []
        if members:
            for member in members:
                member_path = path / member["filename"]
                actual = sha256(member_path) if member_path.is_file() else "MISSING"
                expected = member.get("expected_sha256") or ""
                rows.append(f"{model_id}/{member['filename']}\t{checkpoint_id}\t{member_path}\t"
                            f"{member_path.stat().st_size if member_path.is_file() else ''}\t{actual}\t{expected}\t"
                            f"{'YES' if actual == expected else 'NO'}")
            continue
    actual = sha256(path) if path.is_file() else ("BUNDLE_DIR" if path.is_dir() else "MISSING")
    expected = variant.get("expected_sha256") or ""
    size = path.stat().st_size if path.is_file() else ""
    rows.append(f"{model_id}\t{checkpoint_id}\t{path}\t{size}\t{actual}\t{expected}\t"
                f"{'YES' if actual == expected else ('N/A' if not expected else 'NO')}")
out = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair/raw/checkpoint_hashes.tsv")
out.write_text("\n".join(rows) + "\n")
print("\n".join(rows))
