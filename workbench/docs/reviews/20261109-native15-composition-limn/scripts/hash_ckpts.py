import hashlib, os, json, sys

ROOT='/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700'
files = {
 'csmcir': [f'{ROOT}/cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt'],
 'airknow_fiq_n05': [f'{ROOT}/checkpoints/airknow/airknow_fiq_0.5.pt'],
 'airknow_fiq_n08': [f'{ROOT}/checkpoints/airknow/airknow_fiq_0.8.pt'],
 'conesep_fiq_n02': [f'{ROOT}/cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt'],
 'ptha_mtst_cirr_transfer': [f'{ROOT}/cohort/downloads/ptha_mtst/tuned_mtst_cirr.pt'],
 'habit_fiq_n02': [f'{ROOT}/cohort/downloads/habit/HABIT-FIQ_N0.2.pt'],
 'intent_fiq_n02': [f'{ROOT}/cohort/downloads/intent/intent_fiq_0.2.pt'],
 'hint_fashioniq': [f'{ROOT}/cohort/downloads/hint/fashioniq.pt'],
 'encoder_fashioniq': [f'{ROOT}/cohort/downloads/encoder/fashioniq.pt'],
 'pair_b1': [f'{ROOT}/checkpoints/pair/pair-B1.pt'],
 'pair_b2': [f'{ROOT}/checkpoints/pair/pair-B2.pt'],
 'clvc_dress_local': [f'{ROOT}/checkpoints/clvc/dress_local_model.pt'],
 'clvc_dress_global': [f'{ROOT}/checkpoints/clvc/dress_global_model.pt'],
 'clvc_shirt_local': [f'{ROOT}/checkpoints/clvc/shirt_local_model.pt'],
 'clvc_shirt_global': [f'{ROOT}/checkpoints/clvc/shirt_global_model.pt'],
 'clvc_toptee_local': [f'{ROOT}/checkpoints/clvc/toptee_local_model.pt'],
 'clvc_toptee_global': [f'{ROOT}/checkpoints/clvc/toptee_global_model.pt'],
 'dcnet_dir': [f'{ROOT}/checkpoints/dcnet/config.json', f'{ROOT}/checkpoints/dcnet/trained_model.pth'],
 'combiner_noft': [f'{ROOT}/checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt'],
 'clip4cir_fullft_dir': [f'{ROOT}/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt', f'{ROOT}/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt'],
 'tgcir': [f'{ROOT}/downloads/tgcir/x/checkpoints/Fashioniq.pt'],
 'limn_dress': [f'{ROOT}/cohort/downloads/limn/0_dress_best_model.pt'],
 'limn_shirt': [f'{ROOT}/cohort/downloads/limn/0_shirt_best_model.pt'],
 'limn_toptee': [f'{ROOT}/cohort/downloads/limn/0_toptee_best_model.pt'],
 'tme_fashioniq_0': [f'{ROOT}/cohort/downloads/tme/fashioniq_0_best_model.pth'],
 'sprc_vitl': [f'{ROOT}/checkpoints/sprc/sprc_fiq_vitl.pt'],
}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<22), b''):
            h.update(chunk)
    return h.hexdigest()

res={}
for key, paths in files.items():
    entry={'members':[]}
    for p in paths:
        if not os.path.exists(p):
            entry['members'].append({'path':p,'exists':False})
            continue
        e={'path':p,'exists':True,'size':os.path.getsize(p),'sha256':sha(p)}
        entry['members'].append(e)
    # if single file, surface directly; if dir, compute concat digest over "relpath\0sha\0"
    if len(entry['members'])==1 and entry['members'][0].get('exists'):
        entry['sha256']=entry['members'][0]['sha256']
        entry['size']=entry['members'][0]['size']
    else:
        h=hashlib.sha256()
        for e in sorted(entry['members'], key=lambda x:x['path']):
            if e.get('exists'):
                h.update(os.path.basename(e['path']).encode()); h.update(b'\0')
                h.update(e['sha256'].encode()); h.update(b'\0')
        entry['sha256']=h.hexdigest()
        entry['note']='concat of sorted (basename, member-sha256)'
    res[key]=entry
    print(key, entry.get('sha256'), flush=True)

json.dump(res, open('/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn/scripts/checkpoint_hashes.json','w'), indent=1)
print("DONE")
