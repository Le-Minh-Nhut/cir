import yaml, os, json, glob

os.chdir('/home/heheboiz/data/cir')
d = yaml.safe_load(open('workbench/registry/models.yaml', encoding='utf-8'))
models = d['models']
print("schema_version:", d.get('schema_version'))
print("n_models:", len(models))
urls = [m.get('upstream_repo_url') for m in models]
print("n_distinct_upstream_repo_url:", len(set(urls)))
print("distinct urls:")
for u in sorted(set(urls)):
    print("   ", u, [m['model_id'] for m in models if m.get('upstream_repo_url') == u])
total_cv = 0
print("\nper-model:")
print("%-14s %-28s %-30s %-6s %-12s %-18s %s" % (
    "model_id", "native_protocol", "supported_protocols[0]", "adapter", "n_ckpt_var", "reproduction_status", "checkpoint_variants(ids)"))
for m in models:
    sp = m.get('supported_protocols') or []
    cv = m.get('checkpoint_variants') or []
    total_cv += len(cv)
    ids = []
    for c in cv:
        ids.append(c.get('variant_id') or c.get('checkpoint_id') or list(c)[:2])
    print("%-14s %-28s %-30s %-6s %-12d %-18s %s" % (
        m['model_id'], str(m.get('native_protocol')), str(sp[0] if sp else None),
        'YES' if m.get('composition_adapter') else 'no', len(cv),
        str(m.get('reproduction_status')), ids))
print("\ntotal_checkpoint_variants:", total_cv)
print("keys of one checkpoint_variant sample:")
print(json.dumps(models[0]['checkpoint_variants'][0] if models[0]['checkpoint_variants'] else None, indent=1))
print("source_available count:", sum(1 for m in models if m.get('source_available')))

root = None
for line in open('workbench/config/workbench.env', encoding='utf-8'):
    if line.startswith('WORKBENCH_CHECKPOINT_ROOT='):
        root = line.split('=', 1)[1].strip()
print("\nWORKBENCH_CHECKPOINT_ROOT:", root)
print("exists:", os.path.isdir(root))
# which variants staged
staged = {}
if os.path.isdir(root):
    for m in models:
        for c in (m.get('checkpoint_variants') or []):
            rel = c.get('relative_path') or c.get('path') or c.get('checkpoint_path')
            staged[(m['model_id'], c.get('variant_id'), rel)] = (os.path.exists(os.path.join(root, rel)) if rel else None)
print("\nvariant staging (relative_path -> exists):")
for k, v in staged.items():
    print("  ", k[0], k[1], k[2], "->", v)
print("staged true:", sum(1 for v in staged.values() if v is True), "/", len(staged))
