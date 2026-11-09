import yaml, json
d=yaml.safe_load(open('workbench/registry/models.yaml'))
for m in d['models']:
    print("="*80)
    print("MODEL:", m['model_id'])
    print("KEYS:", list(m.keys()))
    cvs=m.get('checkpoint_variants') or []
    for cv in cvs:
        print("  CV:", json.dumps(cv, indent=None)[:800])
    print("  ENV:", json.dumps(m.get('environment'), indent=None)[:600])
