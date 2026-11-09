import yaml, json, sys
p='workbench/registry/models.yaml'
d=yaml.safe_load(open(p))
models=d['models']
print("COUNT", len(models))
print("IDS", [m['model_id'] for m in models])
for m in models:
    print("="*70)
    print("MODEL", m['model_id'])
    for k in ['composition_adapter','source_dir','upstream_commit_sha','native_protocol','native_entrypoint','checkpoint_variants','environment','status','deferred','notes']:
        if k in m:
            v=m[k]
            s=json.dumps(v, default=str)
            print(f"  {k}: {s[:1200]}")
