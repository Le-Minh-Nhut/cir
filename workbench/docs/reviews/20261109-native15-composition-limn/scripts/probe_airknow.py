"""Probe whether the AirKnow adapted env can build and forward the pinned model.
Read-only: imports only, no checkpoint load, no GPU eval of the full set."""
import sys, os
SRC='/home/heheboiz/data/cir/workbench/third_party/AirKnow'
sys.path.insert(0, SRC)
results={}
# 1. import the model module the replay uses
for mod in ['model.blip2_models.blip2','model.blip2_models.blip2_retriever','lavis.models','torch','transformers','timm','fairscale','peft']:
    try:
        __import__(mod); results[mod]='OK'
    except Exception as e:
        results[mod]='FAIL '+repr(e)[:160]
for k,v in results.items():
    print(k,'=',v)
