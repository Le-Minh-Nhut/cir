"""R6 / R8: probe the diagnostic endpoints for malformed input acceptance, on HEAD.

Read-only. Writes its own JSON so the shell cannot garble it.
Usage: workbench/envs/ctl312h/bin/python .../raw/guards/r6_probe.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).with_suffix(".json")

CASES = [
    ("baseline", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=multimodal&depth=100"),
    ("empty_payload_summary", "/api/composition/diagnostics/limn-dual-scope"),
    ("wrong_qid", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:9999&scope=shirt&condition=multimodal"),
    ("wrong_scope", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=dress&condition=multimodal"),
    ("wrong_condition", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=audio_only"),
    ("depth_zero", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=multimodal&depth=0"),
    ("depth_negative", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=multimodal&depth=-5"),
    ("depth_beyond_stored", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=multimodal&depth=1000"),
    ("depth_stored_100_but_ask_200", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3038&scope=shirt&condition=multimodal&depth=200"),
    ("rank_beyond_window", "/api/composition/diagnostics/limn-dual-scope/candidates?qid=7:3790&scope=shirt&condition=text_only_black_image&depth=100"),
]


def get(path: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"raw": body[:400]}


def main() -> int:
    report: dict = {"artifact_type": "r6_r8_before_evidence", "cases": {}}
    for name, path in CASES:
        code, body = get(path)
        entry = {"path": path, "http": code}
        if "candidates" in body:
            entry.update({
                "candidate_count": len(body["candidates"]),
                "depth_stored": body.get("depth_stored"),
                "depth_reported_in_response": body.get("depth"),
                "target_rank": body.get("target_rank"),
                "target_in_window": any(c.get("is_target") for c in body["candidates"]),
                "clamp_reported": "clamp" in json.dumps(body).lower(),
                "candidate_keys": sorted(body["candidates"][0]) if body["candidates"] else [],
                "top_level_keys": sorted(body),
                "target_object_present": "target" in body,
                "category_present": any("category" in c for c in body["candidates"]),
                "image_url_present": any("image_url" in c for c in body["candidates"]),
                "duplicate_gallery_ids": (len(body["candidates"])
                                          - len({c["gallery_id"] for c in body["candidates"]})),
            })
        else:
            entry["top_level_keys"] = sorted(body)
            entry["detail"] = body.get("detail") or body.get("status")
        report["cases"][name] = entry

    # cross-endpoint agreement: summary's per-scope claim vs what the candidates endpoint serves
    summary_code, summary = get("/api/composition/diagnostics/limn-dual-scope")
    mismatches = []
    if summary_code == 200:
        for qid in ("7:3038", "7:3790"):
            for scope in ("shirt", "toptee"):
                for condition in ("multimodal", "text_only_black_image", "image_only_empty_text"):
                    declared = summary["queries"][qid]["scopes"][scope][condition]
                    code, body = get(f"/api/composition/diagnostics/limn-dual-scope/candidates"
                                     f"?qid={qid}&scope={scope}&condition={condition}&depth=100")
                    if code != 200:
                        mismatches.append({"qid": qid, "scope": scope, "condition": condition,
                                           "problem": f"candidates HTTP {code}"})
                        continue
                    row = {"qid": qid, "scope": scope, "condition": condition}
                    problems = []
                    if declared["target_rank"] != body["target_rank"]:
                        problems.append(f"summary rank {declared['target_rank']} != candidates rank {body['target_rank']}")
                    if abs(declared["target_score"] - body["target_score"]) > 0:
                        problems.append(f"summary score {declared['target_score']} != candidates score {body['target_score']}")
                    if declared["export"] != body.get("export"):
                        problems.append(f"summary export {declared['export']} != candidates export {body.get('export')}")
                    if problems:
                        mismatches.append(row | {"problems": problems})
    report["summary_vs_candidates"] = {"checked": 12, "mismatches": mismatches}

    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(json.dumps(report["cases"], indent=1))
    print("summary_vs_candidates:", json.dumps(report["summary_vs_candidates"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
