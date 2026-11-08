import json, glob, os, sys

os.chdir('/home/heheboiz/data/cir')
CONDS = ["multimodal", "text_only_black_image", "image_only_empty_text"]
ALIAS = {"multimodal": "multimodal", "text_only_black_image": "text_only",
         "image_only_empty_text": "image_only"}

for d in sorted(glob.glob('workbench/artifacts/composition/summary/*')):
    p = os.path.join(d, 'full.json')
    j = json.load(open(p, encoding='utf-8'))
    m = j.get('metrics', {})
    print("###", os.path.basename(d))
    print("   path=%s bytes=%d" % (p, os.path.getsize(p)))
    print("   model_id=%r run_id=%r checkpoint_id=%r" % (j.get('model_id'), j.get('run_id'), j.get('checkpoint_id')))
    print("   query_count=%r subset=%r execution_status=%r protocol_id=%r" % (
        j.get('query_count'), j.get('subset'), j.get('execution_status'), j.get('protocol_id')))
    print("   is_synthetic=%r reproduction_status=%r" % (j.get('is_synthetic'), j.get('reproduction_status')))
    print("   seed=%r seed_applied=%r" % (j.get('seed'), j.get('seed_applied')))
    print("   ckpt_noise=%r eval_noise=%r" % (j.get('checkpoint_training_noise_pct'), j.get('evaluation_noise_pct')))
    print("   gap=%r gap_cohort_count=%r" % (m.get('composition_gap_ndcg'), m.get('gap_cohort_count')))
    print("   topk_export=%s" % json.dumps(j.get('topk_export'), ensure_ascii=False))
    print("   reference_eligibility=%s reference_eligible=%r" % (
        json.dumps(j.get('reference_eligibility'), ensure_ascii=False), j.get('reference_eligible')))
    print("   variant_provenance_keys=%s" % (
        list(j['variant_provenance'].keys()) if 'variant_provenance' in j else None))
    for c in CONDS:
        blk = m.get(ALIAS[c])
        if blk is None:
            print("   %-22s ABSENT from metrics" % c)
            continue
        print("   %-22s recall_at_10=%r recall_at_50=%r ndcg_full=%r coverage=%r scored_count=%r query_count=%r mrr=%r" % (
            c, blk.get('recall_at_10'), blk.get('recall_at_50'), blk.get('ndcg_full'),
            blk.get('coverage'), blk.get('scored_count'), blk.get('query_count'), blk.get('mrr')))
    print("   top-level coverage=%r common_cohort_count=%r" % (m.get('coverage'), m.get('common_cohort_count')))
    print("   condition_status=%s" % json.dumps(j.get('condition_status'), ensure_ascii=False))
    print()
