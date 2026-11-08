import sys, torch
sys.path.insert(0,'/home/heheboiz/data/cir')
sys.path.insert(0,'/home/heheboiz/data/cir/workbench/third_party/LIMN/LIMN')
from pathlib import Path
from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
rec=model_by_id('limn'); ck=checkpoint_by_id(rec,'base_iter0_dress')
p=checkpoint_path('limn',ck,Path('workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads'))
sd=torch.load(p,map_location='cpu',weights_only=False)
sd=sd if isinstance(sd,dict) else sd.state_dict()
keys=[k for k in sd if 'image_mean' in k or 'image_std' in k]
print('norm keys',keys)
for k in keys: print(' ',k,[round(float(x),8) for x in sd[k].flatten().tolist()])
print('expected mean',[0.48145466,0.4578275,0.40821073])
print('expected std',[0.26862954,0.26130258,0.27577711])
