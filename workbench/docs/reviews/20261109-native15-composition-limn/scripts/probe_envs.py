import subprocess, json, os, sys
envs = {
 'airknow': 'workbench/envs/airknow-adapted/bin/python',
 'conesep': 'workbench/envs/ilearn39-adapted/bin/python',
 'ptha_mtst': 'workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python',
 'habit': 'workbench/envs/ilearn39-adapted/bin/python',
 'intent': 'workbench/envs/ilearn39-adapted/bin/python',
 'hint': 'workbench/envs/ilearn39-adapted/bin/python',
 'encoder': 'workbench/envs/encoder310-adapted/bin/python',
 'pair': 'workbench/envs/pair-adapted/bin/python',
 'clvc_net': None,
 'dcnet': 'workbench/artifacts/environments/dcnet-377/bin/python',
 'combiner_rn50x4_noft': 'workbench/artifacts/environments/clip4cir-38/bin/python',
 'clip4cir_rn50x4_fullft': 'workbench/artifacts/environments/clip4cir-38/bin/python',
 'tgcir': None,
 'limn': 'workbench/envs/limn310-adapted/bin/python',
 'tme': 'workbench/envs/tme-adapted/bin/python',
 'csmcir': 'workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python',
 'sprc': 'workbench/artifacts/environments/sprc-39/bin/python',
}
code = "import sys;print(sys.version.split()[0]);\ntry:\n import torch;print(torch.__version__);print(torch.cuda.is_available())\nexcept Exception as e:\n print('TORCH_ERR',repr(e));print('NA');print('NA')"
out={}
for m,p in envs.items():
    rec={'interpreter':p}
    if p is None:
        rec.update({'exists':False,'python':None,'torch':None,'cuda':None,'note':'no interpreter path declared'})
        out[m]=rec; continue
    full=os.path.join('/home/heheboiz/data/cir',p)
    rec['exists']=os.path.exists(full)
    if not rec['exists']:
        rec.update({'python':None,'torch':None,'cuda':None,'note':'path missing'}); out[m]=rec; continue
    try:
        r=subprocess.run([full,'-c',code],capture_output=True,text=True,timeout=120)
        lines=[l for l in r.stdout.strip().splitlines()]
        rec['python']=lines[0] if len(lines)>0 else None
        rec['torch']=lines[1] if len(lines)>1 else None
        rec['cuda']=lines[2] if len(lines)>2 else None
        rec['stderr_tail']=r.stderr.strip().splitlines()[-3:] if r.stderr else []
    except Exception as e:
        rec.update({'python':None,'torch':None,'cuda':None,'error':repr(e)})
    out[m]=rec
json.dump(out,open('workbench/docs/reviews/20261109-native15-composition-limn/scripts/env_probe.json','w'),indent=1)
print(json.dumps(out,indent=1))
