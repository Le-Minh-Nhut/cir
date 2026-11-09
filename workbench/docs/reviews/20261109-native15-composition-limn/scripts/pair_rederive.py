"""Re-derive the PAIR checkpoint inventory. Logs from inside Python (shell redirect is unreliable here)."""
import sys, os, json, torch, torch.nn as nn
import importlib.util

LOG = '/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn/scripts/pair_rederive.out'
log = open(LOG, 'w')
class Tee:
    def __init__(self,*f): self.f=f
    def write(self,s):
        for x in self.f: x.write(s)
    def flush(self):
        for x in self.f: x.flush()
sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

CKPT = '/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/pair/pair-B1.pt'
PAIR_MODELS = '/home/heheboiz/data/cir/workbench/third_party/PAIR/models'

# --- 0. Is model_vssp present in the pinned repo? ---
import glob
vssp_files = glob.glob('/home/heheboiz/data/cir/workbench/third_party/PAIR/**/model_vssp*', recursive=True)
dis_files  = glob.glob('/home/heheboiz/data/cir/workbench/third_party/PAIR/**/MHTransformer_dis*', recursive=True)
print("model_vssp files in pinned repo:", vssp_files or "ABSENT")
print("MHTransformer_dis files in pinned repo:", dis_files or "ABSENT")

# --- 1. Load the pickle with the exploratory alias loader (only way to unpickle at all) ---
sys.path.insert(0, '/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/scripts')
import compat_loader
obj, meta = compat_loader.load(CKPT, 'B1', 'H_rev', map_location='cpu')
print("ROOT_CLASS:", type(obj).__module__ + '.' + type(obj).__name__)

sd = obj.state_dict()
print("STATE_DICT_ENTRIES:", len(sd))

# --- 2. Count raw tensors in the pickle's module tree vs state_dict ---
# state_dict() may collapse tied/shared tensors; count unique tensor objects too.
unique_tensor_ids = len({id(v) for v in sd.values()})
print("UNIQUE_TENSOR_OBJECTS_IN_STATE_DICT:", unique_tensor_ids)

# Walk modules: every named_parameter + named_buffer, to see the raw registration count.
nparams = sum(1 for _ in obj.named_parameters())
nbufs = sum(1 for _ in obj.named_buffers())
print("NAMED_PARAMETERS:", nparams, "NAMED_BUFFERS:", nbufs, "SUM:", nparams+nbufs)

# prefix breakdown of state_dict
from collections import Counter
pref = Counter(k.split('.')[0] for k in sd)
print("PREFIX_COUNTS:", dict(pref), "SUM:", sum(pref.values()))
print("PREFIX_COUNTS_TOP2:", {k:pref[k] for k in ('backbone','affine') if k in pref})

# --- 3. Fresh public model ---
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
import open_clip
_orig = open_clip.create_model_and_transforms
open_clip.create_model_and_transforms = lambda name, **kw: _orig(name, **{**kw, 'pretrained': None})
open_clip.get_tokenizer = lambda *a, **k: (lambda s: s)
model_mod = load_mod('model_public', os.path.join(PAIR_MODELS,'model.py'))
fresh = model_mod.PAIR()
fsd = fresh.state_dict()
print("FRESH_MODEL_CLASS:", type(fresh).__module__ + '.' + type(fresh).__name__)
print("FRESH_MODEL_KEY_COUNT:", len(fsd))

ck=set(sd); fr=set(fsd)
missing=sorted(fr-ck); unexpected=sorted(ck-fr); common=sorted(ck&fr)
shp=[k for k in common if tuple(sd[k].shape)!=tuple(fsd[k].shape)]
print("MISSING_IN_CKPT:", len(missing), missing[:8])
print("UNEXPECTED_IN_CKPT:", len(unexpected))
print("SHAPE_MISMATCH:", len(shp))

# --- 4. reachability ---
INF=('transformer','transformer_a','transformer_b','clip','fc','text_fc','mlp','mlp_img','mlp_text')
BYP=('transformer_c','transformer_d','weight_chanelImg','weight_chanelImg_t','cross_t2i','mlp_weighted','weighted')
used=byp=unc=0; total=sum(v.numel() for v in sd.values())
for k,v in sd.items():
    p=k.split('.'); head=p[1] if p[0]=='backbone' and len(p)>1 else None
    if head is None or head in INF: used+=v.numel()
    elif head in BYP: byp+=v.numel()
    else: unc+=v.numel()
print("PARAMS_TOTAL:", total)
print("PARAMS_BYPASSED:", byp)
print("PARAMS_UNCLASSIFIED:", unc)
print("PCT_BYPASSED_PLUS_UNCLASS:", round(100.0*(byp+unc)/total,4))
log.flush()
