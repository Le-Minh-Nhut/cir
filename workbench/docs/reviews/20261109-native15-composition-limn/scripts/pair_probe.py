import sys, os, torch, pickle, zipfile, io
LOG='/tmp/pair_probe.out'
log=open(LOG,'w')
class T:
    def __init__(s,*f): s.f=f
    def write(s,x):
        for q in s.f: q.write(x)
    def flush(s):
        for q in s.f: q.flush()
sys.stdout=T(sys.__stdout__,log); sys.stderr=T(sys.__stderr__,log)
CKPT='/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/pair/pair-B1.pt'
sys.path.insert(0,'/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/scripts')
import compat_loader
obj,meta=compat_loader.load(CKPT,'B1','H_rev',map_location='cpu')

# 1. identify buffers and persistence
print("--- named_buffers ---")
for name,b in obj.named_buffers():
    # persistence: is it in state_dict?
    sd=obj.state_dict()
    print("BUFFER", name, "shape", tuple(b.shape), "in_state_dict", name in sd)
print("--- non-persistent buffer search ---")
for mod_name,mod in obj.named_modules():
    for bname,buf in mod._buffers.items():
        if buf is not None:
            persist = mod._non_persistent_buffers_set
            print("MOD", mod_name or '<root>', "BUF", bname, "non_persistent", bname in persist)

# 2. raw pickle root class name (unpickle globals) WITHOUT executing the alias
print("--- raw pickle class names ---")
import pickletools
# torch.save zip: find data.pkl
try:
    z=zipfile.ZipFile(CKPT)
    names=[n for n in z.namelist() if n.endswith('data.pkl')]
    print("data.pkl entries:", names[:3])
    data=z.read(names[0])
    import re
    # extract GLOBAL references
    globs=set()
    for op,arg,pos in pickletools.genops(data):
        if op.name in ('GLOBAL','STACK_GLOBAL'):
            pass
    # simpler: regex for module names in the pickle stream
    txt=data.decode('latin-1')
    mods=set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', txt))
    for want in ['model_vssp','MHTransformer_dis','MHTransformer','Encoder','Backbone','PAIR']:
        print("  token present:", want, want in mods)
except Exception as e:
    print("raw inspect failed", repr(e))

# 3. state_dict vs named_parameters+buffers
sd=obj.state_dict()
print("STATE_DICT_ENTRIES", len(sd))
print("NAMED_PARAMS", sum(1 for _ in obj.named_parameters()))
print("NAMED_BUFFERS", sum(1 for _ in obj.named_buffers()))
log.flush()
