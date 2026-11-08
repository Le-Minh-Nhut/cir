import sys, json
sys.path.insert(0,'/home/heheboiz/data/cir')
sys.path.insert(0,'/home/heheboiz/data/cir/workbench/third_party/HABIT')
from pathlib import Path
from workbench.replay.lavis_processors import text_processors, reset
reset()
src=Path('workbench/third_party/HABIT')
tp,boundary=text_processors(src,'configs/models/blip2/blip2_pretrain.yaml')
proc=tp['eval']
print('boundary',boundary)
print('processor class',type(proc).__name__,'max_words',getattr(proc,'max_words',None),'prompt',repr(getattr(proc,'prompt',None)))
for cap in ['is long and has a floral print','IS LONG AND HAS A FLORAL PRINT','', '   ', 'is long & red!']:
    print(repr(cap),'->',repr(proc(cap)))
