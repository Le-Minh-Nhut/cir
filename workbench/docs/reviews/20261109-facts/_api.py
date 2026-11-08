import json, subprocess

raw = subprocess.run(
    ["curl", "-s", "http://127.0.0.1:8000/api/composition/runs?subset=full"],
    capture_output=True, text=True).stdout
rows = json.loads(raw)
print("n_rows:", len(rows))
print("top-level keys of row0:", sorted(rows[0].keys()))
print()
for r in rows:
    print("### model_id=%r run_id=%r" % (r.get('model_id'), r.get('run_id')))
    print("   execution_status=%r" % (r.get('execution_status'),))
    caps = r.get('capabilities') or {}
    meas = r.get('measurement') or {}
    si = r.get('scoring_impl') or (meas.get('scoring_impl') if isinstance(meas, dict) else None)
    print("   capabilities.candidate_topk=%s" % json.dumps(caps.get('candidate_topk'), ensure_ascii=False))
    print("   capabilities keys=%s" % sorted(caps.keys()))
    print("   measurement.coverage=%r" % (meas.get('coverage'),))
    print("   measurement keys=%s" % sorted(meas.keys()))
    print("   scoring_impl.impl=%r" % (si.get('impl') if isinstance(si, dict) else si,))
    print("   duplicate_status=%r" % (r.get('duplicate_status'),))
    print("   measurement_identity_confidence=%r" % (r.get('measurement_identity_confidence'),))
    print("   other top-level: %s" % json.dumps({k: r[k] for k in r if k not in (
        'model_id','run_id','execution_status','capabilities','measurement','scoring_impl',
        'duplicate_status','measurement_identity_confidence','adapter','category_resolution')}, ensure_ascii=False)[:600])
    print()

print("variants list:")
for r in rows:
    print("  ", r.get('model_id'), r.get('run_id'), r.get('execution_status'), r.get('duplicate_status'))
