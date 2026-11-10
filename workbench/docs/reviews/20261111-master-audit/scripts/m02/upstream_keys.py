"""M-02: confirm the upstream UniIR FashionIQ preprocessor never wrote a garment-category key.

Reads the pinned blob from GitHub (fallback: a local copy path passed as argv[1]) and reports every
dict key the module builds, plus the two functions that assemble the release rows.

Usage: python3 scripts/m02/upstream_keys.py [local_copy]
Writes raw/m02/m02_upstream.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]          # workbench/docs/reviews/<audit>
REPO = HERE.parents[3]                              # repository root
OUT = HERE / "raw/m02/m02_upstream.json"
COMMIT = "98dc570c2645fec8e1fd42c7d4ac9853f46051fd"
BLOB_SHA = "d815957ccff7fb75ca56a34ea9822d2a9065f634"
URL = f"https://raw.githubusercontent.com/TIGER-AI-Lab/UniIR/{COMMIT}/src/data/preprocessing/fashioniq_data_preprocessor.py"
GITHUB = f"https://github.com/TIGER-AI-Lab/UniIR/blob/{COMMIT}/src/data/preprocessing/fashioniq_data_preprocessor.py"


def source(argv: list[str]) -> tuple[str, str]:
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8"), f"local:{argv[1]}"
    with urllib.request.urlopen(URL) as response:
        return response.read().decode("utf-8"), URL


def body(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    rest = text[start:]
    end = re.search(r"\ndef ", rest[1:])
    return rest if end is None else rest[: end.start() + 1]


def main(argv: list[str]) -> int:
    text, origin = source(argv)
    keys = sorted(set(re.findall(r'"([a-z_]+)":', text)))
    entry_keys = sorted(set(re.findall(r'"([a-z_]+)":', body(text, "fashioniq_to_mbeir_entry"))))
    pool_keys = sorted(set(re.findall(r'"([a-z_]+)":', body(text, "generate_fashioniq_candidate_pool"))))
    forbidden = [key for key in ("category", "garment", "garment_category", "subtask", "source_file")
                 if any(key in found for found in keys)]
    payload = {
        "upstream_repo": "TIGER-AI-Lab/UniIR",
        "path": "src/data/preprocessing/fashioniq_data_preprocessor.py",
        "commit": COMMIT,
        "blob_sha1_at_commit": BLOB_SHA,
        "url": GITHUB,
        "raw_url": URL,
        "origin_read": origin,
        "sha256_of_read_bytes": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "line_count": text.count("\n") + 1,
        "row_builder": "fashioniq_to_mbeir_entry(fashioniq_entry, candidate_pool, mbeir_data_dir, "
                       "include_src_content=True, concatenate_captions=True)",
        "row_builder_keys": entry_keys,
        "query_src_content_keys": sorted(set(re.findall(r'"([a-z_]+)":', body(text, "fashioniq_to_mbeir_entry")
                                                    .split("query_src_content = {")[1].split("}")[0]))),
        "candidate_pool_builder": "generate_fashioniq_candidate_pool(fashioniq_images_dir, "
                                  "fashioniq_candidate_pool_path, mbeir_data_dir, include_src_content=True)",
        "candidate_pool_keys": pool_keys,
        "all_dict_keys_in_module": keys,
        "garment_or_category_keys_found": forbidden,
        "verdict": ("UPSTREAM WRITES NO CATEGORY KEY: every query row lacks category/garment/"
                    "subtask/source_file, so the release's garment category can only come from a "
                    "downstream loader, never from the generator.") if not forbidden else
                   f"REFUTED: keys {forbidden} present",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("commit", "row_builder_keys", "query_src_content_keys", "candidate_pool_keys",
                       "all_dict_keys_in_module", "garment_or_category_keys_found", "verdict")}, indent=2))
    return 0 if not forbidden else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
