import sys, random, json
from pathlib import Path
from collections import Counter
sys.path.insert(0,'/home/heheboiz/data/cir')
from PIL import Image
from workbench.composition.inputs import black_image_like
random.seed(0)
files=sorted(Path('data/FashionIQ/images').glob('*.png'))
sample=random.sample(files, 4000)
modes=Counter()
bad=[]
for p in sample:
    with Image.open(p) as im:
        modes[im.mode]+=1
        if im.mode!='RGB':
            b=black_image_like(im.copy())
            # what RGB does the transform see?
            rgb=b.convert('RGB')
            px=rgb.getpixel((0,0))
            if px!=(0,0,0):
                bad.append((p.name, im.mode, px))
print('modes over 4000 sample:',dict(modes))
print('non-RGB files whose "black" is NOT black after convert(RGB):',len(bad))
for rec in bad[:10]: print('  ',rec)
