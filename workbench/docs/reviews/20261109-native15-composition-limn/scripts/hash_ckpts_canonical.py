"""Compute checkpoint digests using the workbench's own canonical hasher
(workbench/backend/registry.py::sha256_file) so single files and directories are
identified exactly as the evaluation pipeline identifies them."""
import sys, json, os
sys.path.insert(0, '/home/heheboiz/data/cir')
from pathlib import Path
from workbench.backend.registry import sha256_file

ROOT = Path('/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700')

CASES = {
  'csmcir': dict(path=ROOT/'cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt'),
  'airknow_fiq_n05': dict(path=ROOT/'checkpoints/airknow/airknow_fiq_0.5.pt'),
  'airknow_fiq_n08': dict(path=ROOT/'checkpoints/airknow/airknow_fiq_0.8.pt'),
  'conesep_fiq_n02': dict(path=ROOT/'cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt'),
  'ptha_mtst_cirr_transfer': dict(path=ROOT/'cohort/downloads/ptha_mtst/tuned_mtst_cirr.pt'),
  'habit_fiq_n02': dict(path=ROOT/'cohort/downloads/habit/HABIT-FIQ_N0.2.pt'),
  'intent_fiq_n02': dict(path=ROOT/'cohort/downloads/intent/intent_fiq_0.2.pt'),
  'hint_fashioniq': dict(path=ROOT/'cohort/downloads/hint/fashioniq.pt'),
  'encoder_fashioniq': dict(path=ROOT/'cohort/downloads/encoder/fashioniq.pt'),
  'pair_b1': dict(path=ROOT/'checkpoints/pair/pair-B1.pt'),
  'pair_b2': dict(path=ROOT/'checkpoints/pair/pair-B2.pt'),
  'clvc_bundle': dict(path=ROOT/'checkpoints/clvc',
                      required=['dress_local_model.pt','dress_global_model.pt','shirt_local_model.pt',
                                'shirt_global_model.pt','toptee_local_model.pt','toptee_global_model.pt']),
  'dcnet_dir': dict(path=ROOT/'checkpoints/dcnet', required=['config.json','trained_model.pth']),
  'combiner_noft': dict(path=ROOT/'checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt'),
  'clip4cir_fullft_dir': dict(path=ROOT/'checkpoints/clip4cir/FashionIQ/RN50x4_fullft',
                      required=['fiq_comb_RN50x4_fullft.pt','fiq_clip_RN50x4_fullft.pt']),
  'tgcir': dict(path=ROOT/'downloads/tgcir/x/checkpoints/Fashioniq.pt'),
  'limn_bundle': dict(path=ROOT/'cohort/downloads/limn',
                      required=['0_dress_best_model.pt','0_shirt_best_model.pt','0_toptee_best_model.pt']),
  'tme_fashioniq_0': dict(path=ROOT/'cohort/downloads/tme/fashioniq_0_best_model.pth'),
  'sprc_vitl': dict(path=ROOT/'checkpoints/sprc/sprc_fiq_vitl.pt'),
}

res={}
for key, kw in CASES.items():
    p=kw['path']; req=kw.get('required')
    if not p.exists():
        res[key]={'path':str(p),'exists':False}; print(key,'MISSING',flush=True); continue
    try:
        d=sha256_file(p, required_files=req)
        res[key]={'path':str(p),'exists':True,'sha256':d,'kind':'directory' if p.is_dir() else 'file',
                  'required':req}
        print(key, d, flush=True)
    except Exception as e:
        res[key]={'path':str(p),'exists':True,'error':repr(e)}
        print(key,'ERROR',repr(e),flush=True)

json.dump(res, open('/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn/scripts/checkpoint_hashes_canonical.json','w'), indent=1)
print('DONE')
