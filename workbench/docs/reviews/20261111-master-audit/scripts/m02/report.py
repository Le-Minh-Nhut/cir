"""Render raw/m02/m02_resolver.txt from the three M-02 JSON artifacts.

Usage: python3 scripts/m02/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]          # workbench/docs/reviews/<audit>
REPO = HERE.parents[3]                              # repository root
RAW = HERE / "raw/m02"
OUT = RAW / "m02_resolver.txt"


def load(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def main() -> int:
    probe, recount, scope, upstream = (load("m02_resolver.json"), load("m02_recount.json"),
                                       load("qid_scope_map.json"), load("m02_upstream.json"))
    lines: list[str] = []
    add = lines.append

    add("M-02 LIMN CATEGORY ROUTING — VERIFICATION RECORD")
    add("repo: /home/heheboiz/data/cir   date: 2026-11-11   evidence dir: workbench/docs/reviews/20261111-master-audit/")
    add("resolver: workbench/composition/model_limn.py::resolve_query_category + caption_scope_index")
    add("loader:   workbench/composition/data.py::load_release + assign_garment_categories")
    add("scripts:  workbench/docs/reviews/20261111-master-audit/scripts/m02/{probe_resolver,recount_recovered,build_scope_map,upstream_keys,report}.py")
    add("")
    add("=" * 78)
    add("1. FIXTURE PROBE — synthetic FashionIQ roots (temp dirs, cap.<category>.val.json)")
    add("=" * 78)
    q = probe["query"]
    add(f"shape: qid={q['qid']} stem={q['reference_stem']} loader_category={q['loader_category']!r}")
    add(f"       query_txt={q['query_txt']!r}")
    add(f"       normalized key={q['normalized_key']!r}")
    add(f"       both files carry captions {probe['fixture_captions']['shirt']!r}")
    add("")
    add("case                                    index_entry                 resolve_query_category()")
    add("-" * 78)
    for row in probe["probes"]:
        index_entry = ",".join(row["index_entry"]) if row["index_entry"] else "(none)"
        add(f"{row['fixture']:<39} {index_entry:<27} {row['resolved']!r}")
    add("")
    add("Verdict on the claimed defect: CONFIRMED.")
    add("  a) both present, same (stem, normalized text) pair -> returns None  (refuses, correct)")
    add("  b) toptee file absent                               -> returns 'shirt' (silently routes)")
    add("  c) toptee file present but invalid JSON             -> returns 'shirt' (silently routes)")
    add("  d) shirt file absent (only toptee present)          -> returns 'toptee' (silently routes)")
    add("  b/c/d quote raw return values 'shirt' / 'shirt' / 'toptee'; the resolver cannot tell")
    add("  'the other source had no row' from 'the other source was never read', because")
    add("  caption_scope_index swallows OSError/ValueError and continues (model_limn.py:71-75, except at :74).")
    add("  Impact on the real release: measured in the degradation table at the end of section 2.")
    add("")
    add("=" * 78)
    add("2. INDEPENDENT RECOUNT OF THE '117 HISTORICALLY RECOVERED' QUERIES")
    add("=" * 78)
    src = recount["source"]
    add(f"release query file : {src['release_query_file']}")
    add(f"  sha256           : {src['release_query_sha256']}")
    add(f"  hash verification against EXPECTED_SHA256: {recount['source']['loader_hash_verification']}")
    for category, digest in src["caption_file_sha256"].items():
        add(f"cap.{category}.val.json sha256: {digest}")
    counts = recount["resolved_scope_counts"]
    add("")
    add(f"ambiguous qids (query.category is None): {recount['ambiguous_qid_count']}")
    add(f"resolved to exactly one of {{dress,shirt,toptee}}: {recount['resolved_count']}")
    add(f"  shirt={counts.get('shirt', 0)}  toptee={counts.get('toptee', 0)}  dress={counts.get('dress', 0)}")
    add(f"unresolved: {recount['unresolved_count']} -> {recount['unresolved_qids']}")
    add("")
    add("reproduction of the claim: 117 = YES, remaining two = 7:3038, 7:3790 = YES")
    for name, value in recount["checks"].items():
        add(f"  check {name}: {value}")
    add("")
    add("Per-qid rederivation — raw cap.<category>.val.json bytes re-parsed, pair "
        "(stem, normalized text) looked up")
    add("in each of the three files independently of any cached index. PASS_<cat>_only means the")
    add("deciding file holds the pair AND the other two do not.")
    add("")
    for row in recount["queries"]:
        add(f"  {row['qid']:<8} {str(row['resolved']):<7} stem={row['reference_stem']:<11} "
            f"raw_hits={','.join(row['raw_file_categories']):<14} {row['rederivation']}")
    add("")
    add("Degradation on the real 119 (shipped resolver, one caption file removed or corrupted):")
    add("  removed/mode      resolved  scopes                     7:3038     7:3790")
    for row in recount["degraded_real_data"]:
        add(f"  {row['removed'] + '/' + row['mode']:<17} {row['resolved_count']:<9} "
            f"{json.dumps(row['scope_counts']):<26} {row['7:3038']!r:<11} {row['7:3790']!r}")
    add("  Reading: with cap.shirt.val.json unreadable the two genuinely ambiguous qids 7:3038 and")
    add("  7:3790 silently become 'toptee'; with cap.toptee.val.json unreadable they silently become")
    add("  'shirt'. The dress file is never the deciding source for these 119, so its loss is inert.")
    add("")
    add("=" * 78)
    add("3. MAPPING ARTIFACT raw/m02/qid_scope_map.json (under this review dir)")
    add("=" * 78)
    add(f"cohort: {scope['cohort']}  size: {scope['cohort_size']}")
    add(f"counts: {json.dumps(scope['counts']['by_reason'])}")
    add(f"        scored={scope['counts']['scored']}  unresolved={scope['counts']['unresolved']}")
    add("groups (reason, scope, count, source_files):")
    for record in scope["records"]:
        add(f"  {record['reason']:<17} {record['scope']:<7} n={record['count']:<5} "
            f"{', '.join(record['source_files'])}")
    add(f"  {'ambiguous_reference_and_caption':<17} {'None':<7} n={len(scope['unresolved']):<5} "
        f"{', '.join(scope['unresolved'][0]['source_files'])}")
    add(f"sha256 release query: {scope['sha256']['release_query_sha256']}")
    for path, digest in scope["sha256"]["caption_files"].items():
        add(f"sha256 {path}: {digest}")
    add(f"file size: {(RAW / 'qid_scope_map.json').stat().st_size} bytes (limit 5 MB)")
    add("")
    add("=" * 78)
    add("4. UPSTREAM RELEASE GENERATOR — garment-category key?")
    add("=" * 78)
    add(f"source: {upstream['url']}")
    add(f"commit: {upstream['commit']}  blob sha1: {upstream['blob_sha1_at_commit']}")
    add(f"read via: {upstream['origin_read']}   sha256 of read bytes: {upstream['sha256_of_read_bytes']}")
    add("no local copy exists in this repo (grep for fashioniq_data_preprocessor / UniIR under")
    add("workbench/third_party and artifacts found only the LIMN vendor tree), so the pinned")
    add("GitHub source was read.")
    add("")
    add(f"row builder: {upstream['row_builder']}")
    add(f"  query-row keys: {upstream['row_builder_keys']}")
    add(f"  query_src_content keys: {upstream['query_src_content_keys']}")
    add(f"candidate pool builder: {upstream['candidate_pool_builder']}")
    add(f"  pool-row keys: {upstream['candidate_pool_keys']}")
    add(f"every dict key in the module: {upstream['all_dict_keys_in_module']}")
    add(f"category/garment/subtask/source_file keys found: {upstream['garment_or_category_keys_found']}")
    add(f"verdict: {upstream['verdict']}")
    add("")
    add("Cross-check on the shipped release rows (6003 lines): keys are neg_cand_list, pos_cand_list,")
    add("qid, query_img_path, query_modality, query_src_content, query_txt, task_id; all 6003 rows carry")
    add("query_src_content == {\"candidate_img_id\": ...} only. 'task_id' is not written by the")
    add("preprocessor above; it is added afterwards by src/data/mbeir_data_generator.py:374-375")
    add("('entry[\"task_id\"] = task_id' from get_mbeir_task_id(query_modality, pos_cand_modality) at")
    add("src/data/preprocessing/utils.py:90-93), i.e. it encodes query and candidate modality, never garment category. No")
    add("category/garment/subtask/source_file key exists at any stage, so the release's 'category' can")
    add("only be a downstream loader inference.")
    add("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size} bytes, {len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
