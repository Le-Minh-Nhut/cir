import sys
sys.path.insert(0,'/home/heheboiz/data/cir')
from PIL import Image
import numpy as np, open_clip, torch
from workbench.composition.inputs import black_image_like, adapt_query_inputs
# 1. does the raw black image survive the ENCODER val transform to a constant?
tf=(lambda: __import__('open_clip').image_transform(224,is_train=False))()
im=Image.new('RGB',(342,444),(0,0,0))
t=tf(im)
print('black->tensor shape',tuple(t.shape),'unique values',torch.unique(t).tolist()[:6])
# 2. compare black-before-transform vs zero-after-normalization
print('mean/std of transformed black:',float(t.mean()),float(t.std()))
# 3. is the ablation applied BEFORE transform in the adapter?
src=open('workbench/composition/runner.py').read()
i=src.find('def _stage_query_path')
print()
print(src[i:i+700])
