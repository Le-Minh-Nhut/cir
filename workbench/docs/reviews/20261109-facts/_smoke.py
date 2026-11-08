import json, glob, os
os.chdir('/home/heheboiz/data/cir')
for d in sorted(glob.glob('workbench/artifacts/composition/summary/*')):
    p = os.path.join(d, 'full.json')
    j = json.load(open(p, encoding='utf-8'))
    print("%-58s smoke=%-70s ref_elig=%s" % (
        os.path.basename(d),
        json.dumps(j.get('smoke'), ensure_ascii=False),
        json.dumps(j.get('reference_eligible'), ensure_ascii=False)))
    if 'reference_eligibility' in j:
        print("      reference_eligibility=", json.dumps(j['reference_eligibility'], ensure_ascii=False))
    print("      reproduction_status=", j.get('reproduction_status'),
          "| checkpoint_mapping_status=", j.get('checkpoint_mapping_status'),
          "| checkpoint_training_dataset=", j.get('checkpoint_training_dataset'),
          "| checkpoint_transfer=", j.get('checkpoint_transfer'))
    print("      category_scopes=", json.dumps(j.get('category_scopes'), ensure_ascii=False),
          "| category_resolution.ambiguous_count=", (j.get('category_resolution') or {}).get('ambiguous_count'),
          "| resolved_count=", (j.get('category_resolution') or {}).get('resolved_count'),
          "| unresolved_count=", (j.get('category_resolution') or {}).get('unresolved_count'))
    print("      query_count_per_subset=", json.dumps(j.get('query_count_per_subset'), ensure_ascii=False))
    print("      gallery_count=", j.get('gallery_count'), "| gallery_identity_sha256=", j.get('gallery_identity_sha256'))
    print("      reused_query_result_count=", j.get('reused_query_result_count'))
    print("      release_artifact_hashes=", json.dumps(j.get('release_artifact_hashes'), ensure_ascii=False)[:300])
