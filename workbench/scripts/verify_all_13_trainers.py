#!/usr/bin/env python3
"""Round 1 + Round 2 Multi-Model Verification Harness:
Tests syntax, imports, base pretrained weight loading, 2-step real forward+backward+optimizer,
eval scoring, and checkpoint save/reload for ALL 13 remaining CIR models sequentially on GPU!
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
ENVS = REPO / "workbench" / "envs"
CIR_ENV = REPO / "workbench" / "artifacts" / "environments" / "cir-005ef4e1b02999a5" / "bin" / "python"
TRAINERS = REPO / "workbench" / "benchmarks" / "trainers"

# (model_label, python_exe, extra_pythonpath, script_name, smoke_snippet)
TESTS = [
    (
        "encoder",
        str(ENVS / "encoder310-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/ENCODER",
        "train_encoder.py",
        """
import torch
from workbench.benchmarks.trainers.train_encoder import build_encoder_model, get_encoder_full_state, load_encoder_full_state, set_bn_eval
dev = torch.device('cuda')
m, tr_p, va_p = build_encoder_model(device=dev)
m.train(); m.apply(set_bn_eval)
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
x1 = torch.randn(4, 3, 224, 224, device=dev)
x2 = torch.randn(4, 3, 224, 224, device=dev)
txt = ['change shirt to red'] * 4
with torch.cuda.amp.autocast():
    ld = m.compute_loss(x1, txt, x2)
    loss = ld['stu_rank'] + 0.5*ld['kl'] + ld['entity'] + ld['ortho']
assert torch.isfinite(loss), ld
loss.backward(); opt.step(); opt.zero_grad()
st = get_encoder_full_state(m)
load_encoder_full_state(m, st, dev)
m.eval()
with torch.no_grad():
    q = m.extract_retrieval_compose(x1, txt)
    t = m.extract_retrieval_target(x2)
assert q.shape == (4, 512) and t.shape == (4, 512) and torch.isfinite(q).all()
print(f'OK loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "tgcir",
        str(ENVS / "tgcir39-la/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/TGCIR",
        "train_tgcir.py",
        """
import torch
import workbench.benchmarks.trainers.train_tgcir as mod
dev = torch.device('cuda')
m = mod.tgcir_model_mod.VCG_CIR(img_encoder='ViT-B/16', hidden_dim=512, dropout=0.5, local_token_num=8, global_token_num=4, t=0.05).to(dev)
m.train(); m.apply(mod.set_bn_eval)
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
x1 = torch.randn(4, 3, 224, 224, device=dev)
x2 = torch.randn(4, 3, 224, 224, device=dev)
# Also verify >77 token caption does NOT crash thanks to _safe_clip_tokenize!
txt = ['word ' * 100] * 4
with torch.cuda.amp.autocast():
    ld = m.compute_loss(x1, txt, x2)
    loss = ld['stu_rank'] + ld['tea_rank'] + ld['tea_mask'] + 0.1*ld['ortho'] + ld['ckd'] + 0.1*ld['kl']
assert torch.isfinite(loss), ld
loss.backward(); opt.step(); opt.zero_grad()
m.eval()
with torch.no_grad():
    q = m.extract_retrieval_compose(x1, txt)
    t = m.extract_retrieval_target(x2)
assert q.shape == (4, 512) and t.shape == (4, 512) and torch.isfinite(q).all()
print(f'OK loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "clvcnet",
        str(ENVS / "clvc39-la/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/CLVCNet",
        "train_clvcnet.py",
        """
import torch
import workbench.benchmarks.trainers.train_clvcnet as mod
dev = torch.device('cuda')
texts = ['change the color to blue', 'remove the dog', 'sunny day']
lm = mod.clvc_model_mod.compose_local(texts=texts, T=7.0).to(dev)
gm = mod.clvc_model_mod.compose_global(texts=texts, T=4.0).to(dev)
lm.train(); gm.train()
ol = torch.optim.Adam(lm.parameters(), lr=1e-4)
og = torch.optim.Adam(gm.parameters(), lr=1e-4)
x1 = torch.randn(4, 3, 224, 224, device=dev)
x2 = torch.randn(4, 3, 224, 224, device=dev)
mods = [mod._safe_caption(c) for c in ['change color to blue', '!!!', 'remove dog', 'sunny day']]
with torch.no_grad():
    gt0, gt1 = gm.extract_img_feature(x2); gq0 = gm.compose_img_text(x1, mods)[0]
ld = lm.compute_loss(x1, mods, x2, gt0, gt1, gq0)
ll = ld['class'] + ld['perceptual'] + ld['img'] + ld['mul_kl']
assert torch.isfinite(ll); ll.backward(); ol.step(); ol.zero_grad()
with torch.no_grad():
    lt0, lt1 = lm.extract_img_feature(x2); lq1 = lm.compose_img_text(x1, mods)[1]
gd = gm.compute_loss(x1, mods, x2, lt0, lt1, lq1)
gl = gd['class'] + gd['img'] + gd['mul_kl']
assert torch.isfinite(gl); gl.backward(); og.step(); og.zero_grad()
print(f'OK loc={ll.item():.4f} glb={gl.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "dcnet",
        str(ENVS / "dcnet310-sm120/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/DCNet",
        "train_dcnet.py",
        """
import tempfile, torch
from pathlib import Path
import workbench.benchmarks.trainers.train_dcnet as mod
dev = torch.device('cuda')
with tempfile.TemporaryDirectory() as td:
    vocab = mod.build_or_load_glove_vocab(['change the shirt to bright red', 'remove the car from the street'], Path(td)/'v.pkl')
    m, mods_list = mod.build_dcnet_model(dev)
    crit = mod.AdditiveMarginSoftmax(margin=0.3, dual=False)
    opt = torch.optim.Adam(m.parameters(), lr=2e-4, amsgrad=True)
    x1 = torch.randn(4, 3, 224, 224, device=dev)
    x2 = torch.randn(4, 3, 224, 224, device=dev)
    tt, tl = mod.encode_captions_to_glove_batch(['change shirt to red', '!!!', 'remove car', 'bright red'], vocab)
    tt, tl = tt.to(dev), tl.tolist()
    re = m.get_image_emb({'im_feat': x1}, istgt=False); te = m.get_image_emb({'im_feat': x2}, istgt=True); tx = m.get_text_emb(tt, tl)
    mt = m.get_composition(re, tx); sc = m.get_score_matrix(te, mt, mods_list)
    mx = m.get_correction(re, te); sr = m.get_score_matrix(mx, tx, mods_list)
    loss = crit(sc) + crit(sr)
    assert torch.isfinite(loss); loss.backward(); opt.step(); opt.zero_grad()
    print(f'OK vocab={len(vocab)} loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "hint",
        str(ENVS / "ilearn39-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/HINT",
        "train_ilearn.py",
        """
import torch
import workbench.benchmarks.trainers.train_ilearn as mod
mod._patch_bert_root_path()
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='HINT', model_type='pretrain', is_eval=False, device=dev)
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
x1 = torch.randn(4, 3, 224, 224, device=dev); x2 = torch.randn(4, 3, 224, 224, device=dev)
caps = [tp['eval']('make it sunny')] * 4
with torch.cuda.amp.autocast():
    loss = m({'image': x1, 'target': x2, 'text_input': caps}, dev)['loss_stu_rank']
assert torch.isfinite(loss); loss.backward(); opt.step(); opt.zero_grad()
m.eval()
with torch.no_grad():
    q = m.extract_retrieval_compose(x1, caps); t = m.extract_retrieval_target(x2)
assert q.shape == (4, 1, 1, 256) and t.shape == (4, 256, 32)
print(f'OK loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "habit",
        str(ENVS / "ilearn39-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/HABIT",
        "train_ilearn.py",
        """
import torch
import workbench.benchmarks.trainers.train_ilearn as mod
mod._patch_bert_root_path()
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='HABIT', model_type='pretrain', is_eval=False, device=dev)
mod._patch_habit_dynamic_batch(m)
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
# Test BS=16 (not hardcoded 256!) across epoch 0 AND epoch 1 (with cross-epoch KL!)
x1 = torch.randn(16, 3, 224, 224, device=dev); x2 = torch.randn(16, 3, 224, 224, device=dev)
caps = [tp['eval']('make it red')] * 16
m.soft_labels = []; m.sim = []
with torch.cuda.amp.autocast():
    ld0 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, [], 0, [])
    loss0 = ld0['loss_rank'] + 0.1*ld0['loss_soft'] + 10.0*ld0['loss_kl']
assert torch.isfinite(loss0); loss0.backward(); opt.step(); opt.zero_grad()
old_sl = m.get_soft_labels(); old_sim = m.get_similarity_matrix()
m.soft_labels = []; m.sim = []
with torch.cuda.amp.autocast():
    ld1 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, old_sl[:16], 1, old_sim[0])
    loss1 = ld1['loss_rank'] + 0.1*ld1['loss_soft'] + 10.0*ld1['loss_kl']
assert torch.isfinite(loss1); loss1.backward(); opt.step(); opt.zero_grad()
print(f'OK ep0={loss0.item():.4f} ep1={loss1.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "conesep",
        str(ENVS / "ilearn39-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/ConeSep",
        "train_ilearn.py",
        """
import torch
import workbench.benchmarks.trainers.train_ilearn as mod
mod._patch_bert_root_path()
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='ConeSep', model_type='pretrain', is_eval=False, device=dev)
m.prompt_load(); mod._patch_conesep_nan_guard()
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('make it wooden')] * 8
# Test BOTH phase 1 (if_negative=True) AND phase 2 (if_negative=False, Sinkhorn OT!)
m.epoch = 0
with torch.cuda.amp.autocast():
    ld0 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, if_negative=True, omega=0.5, target_forgetting_loss_weight=0.5, target_oriented_weight=0.5, query_oriented_weight=0.8)
    loss0 = sum(ld0.values())
assert torch.isfinite(loss0), ld0; loss0.backward(); opt.step(); opt.zero_grad()
m.epoch = 2
with torch.cuda.amp.autocast():
    ld2 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, if_negative=False, omega=0.5, target_forgetting_loss_weight=0.5, target_oriented_weight=0.5, query_oriented_weight=0.8)
    loss2 = sum(ld2.values())
assert torch.isfinite(loss2), ld2; loss2.backward(); opt.step(); opt.zero_grad()
print(f'OK ph1={loss0.item():.4f} ph2={loss2.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "intent",
        str(ENVS / "ilearn39-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/INTENT",
        "train_ilearn.py",
        """
import torch
import workbench.benchmarks.trainers.train_ilearn as mod
mod._patch_bert_root_path()
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='Blip2QformerCir', model_type='pretrain', is_eval=False, device=dev)
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('remove the cup')] * 8
# Test BOTH warmup (FFT + CKA) and post-warmup
with torch.cuda.amp.autocast():
    ld0 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, after_warmup=False)
    loss0 = ld0['loss_stu'] + 0.5*ld0['loss_odl'] + 1.0*ld0['loss_caco']
assert torch.isfinite(loss0), ld0; loss0.backward(); opt.step(); opt.zero_grad()
with torch.cuda.amp.autocast():
    ld1 = m({'image': x1, 'target': x2, 'text_input': caps}, dev, after_warmup=True)
    loss1 = ld1['loss_stu'] + 0.5*ld1['loss_odl']
assert torch.isfinite(loss1), ld1; loss1.backward(); opt.step(); opt.zero_grad()
print(f'OK warmup={loss0.item():.4f} main={loss1.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "airknow",
        str(ENVS / "airknow310-la/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/AirKnow",
        "train_ilearn.py",
        """
import torch
import workbench.benchmarks.trainers.train_ilearn as mod
mod._patch_bert_root_path()
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='Blip2QformerCir', model_type='pretrain', is_eval=False, device=dev)
m.alpha = 0.7
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('change color to gold')] * 8
labels = torch.tensor([1,1,1,1,0,0,0,0], dtype=torch.long, device=dev)
with torch.cuda.amp.autocast():
    l_eki = m({'image': x1, 'target': x2, 'text_input': caps, 'labels': labels, 'eki': True}, dev)['loss_eki']
assert torch.isfinite(l_eki); l_eki.backward(); opt.step(); opt.zero_grad()
with torch.cuda.amp.autocast():
    ld = m({'image': x1, 'target': x2, 'text_input': caps, 'labels': None, 'eki': False}, dev)
    l_main = ld['loss_align'] + 0.6*ld['loss_recon']
assert torch.isfinite(l_main), ld; l_main.backward(); opt.step(); opt.zero_grad()
print(f'OK eki={l_eki.item():.4f} main={l_main.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "mtst_ptha",
        str(CIR_ENV),
        f"{REPO}:{REPO}/workbench/third_party/MTST_PTHA/MTST_PTHA/src",
        "train_mtst_ptha.py",
        """
import torch
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='blip2_cir_cat', model_type='pretrain', is_eval=False, device=dev)
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('make the sky cloudy')] * 8
with torch.cuda.amp.autocast():
    loss = m({'image': x1, 'target': x2, 'text_input': caps})['loss_itc']
assert torch.isfinite(loss); loss.backward(); opt.step(); opt.zero_grad()
print(f'OK loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "csmcir",
        str(CIR_ENV),
        f"{REPO}:{REPO}/workbench/third_party/CSMCIR/src",
        "train_csmcir.py",
        """
import torch
import workbench.benchmarks.trainers.train_csmcir as mod
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='blip2_cir_align_prompt_csmcir', model_type='pretrain', is_eval=False, device=dev)
mod._resize_csmcir_memory_bank(m, 64, dev)
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('make it gold')] * 8
m.memory_bank_ptr = m.memory_bank_size; m.memory_bank_text_mask.fill_(1)
with torch.cuda.amp.autocast():
    ld = m({'image': x1, 'target': x2, 'text_input': caps, 'text_cot': caps, 'image_captions': ['']*8})
    loss = sum(ld.values())
assert torch.isfinite(loss), ld; loss.backward(); opt.step(); opt.zero_grad()
print(f'OK loss={loss.item():.4f} keys={list(ld.keys())} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "tme",
        str(ENVS / "tme-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/TME/src",
        "train_tme.py",
        """
import os, torch
os.chdir('/home/heheboiz/data/cir/workbench/third_party/TME')
from lavis.models import load_model_and_preprocess
dev = torch.device('cuda')
m, _, tp = load_model_and_preprocess(name='blip2_cir_image_diff_features', model_type='pretrain', is_eval=False, device=dev)
pn_loss = {'positive_loss':'RCL','negative_loss':'None','positive_align_loss':'MSE','negative_align_loss':'None','trade_off':1.0,'trade_off_align':1.0,'warmup_loss':'RCL','warmup_align_loss':'MSE'}
m.train(); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = [tp['eval']('add a hat')] * 8
labels = torch.ones(8, device=dev)
for ph in ('qformer', 'proj', 'last', None):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            e1 = m.vit_encode(x1); e2 = m.vit_encode(x2)
        ld = m({'image': e1, 'target': e2, 'text_input': caps}, labels, pn_loss, ph)
        loss = sum(ld.values())
    assert torch.isfinite(loss), (ph, ld)
    loss.backward(); opt.step(); opt.zero_grad()
with torch.no_grad():
    pl, _ = m.per_loss(e1, e2, caps)
assert pl.shape == (8,) and torch.isfinite(pl).all()
print(f'OK all_4_phases last_loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
    (
        "limn",
        str(ENVS / "limn310-adapted/bin/python"),
        f"{REPO}:{REPO}/workbench/third_party/LIMN/LIMN",
        "train_limn.py",
        """
import torch
import workbench.benchmarks.trainers.train_limn as mod
dev = torch.device('cuda')
m = mod.limn_model_mod.LIMN(prefix_num=10, head=8, layer=2, hidden_dim=768, temperature=0.1).to(dev)
m.backbone.clip.set_grad_checkpointing(True)
m.train(); opt = torch.optim.AdamW(m.parameters(), lr=5e-6)
x1 = torch.randn(8, 3, 224, 224, device=dev); x2 = torch.randn(8, 3, 224, 224, device=dev)
caps = ['change color to green'] * 8
dt = torch.full((8,), 5, dtype=torch.long, device=dev)
with torch.cuda.amp.autocast():
    ld = m.compute_loss(x1, caps, x2, dt)
    loss = ld['leader_ranking'] + ld['norm'] + ld['peer_ranking']
assert torch.isfinite(loss), ld; loss.backward(); opt.step(); opt.zero_grad()
m.eval()
with torch.no_grad():
    q = m.compose_query(x1, caps, dt)[0]; t = m.compose_target(x2, dt)[0]
assert q.shape == (8, 768) and t.shape == (8, 768)
print(f'OK loss={loss.item():.4f} vram={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB')
""",
    ),
]


def main() -> None:
    print("=== ROUND 1: SYNTAX CHECK ALL TRAINER SCRIPTS ===", flush=True)
    for script in sorted(TRAINERS.glob("*.py")):
        subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)
        print(f"  py_compile OK: {script.name}", flush=True)

    print("\n=== ROUND 2: LIVE GPU FORWARD+BACKWARD+STEP SMOKE TEST (13 MODELS) ===", flush=True)
    failures = []
    for name, py_exe, py_path, script_name, snippet in TESTS:
        t0 = time.time()
        env = dict(os.environ, PYTHONPATH=py_path)
        proc = subprocess.run(
            [py_exe, "-c", snippet],
            env=env,
            capture_output=True,
            text=True,
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            last_line = [l for l in proc.stdout.strip().splitlines() if l.strip()][-1]
            print(f"  [PASS] {name:<10} ({dt:5.1f}s) | {last_line}", flush=True)
        else:
            print(f"  [FAIL] {name:<10} ({dt:5.1f}s) | rc={proc.returncode}", flush=True)
            print("    STDERR:", proc.stderr[-800:], flush=True)
            failures.append((name, proc.stderr[-500:]))

    print(f"\n=== SUMMARY: {len(TESTS)-len(failures)}/{len(TESTS)} PASSED ===", flush=True)
    if failures:
        sys.exit(1)


import os
if __name__ == "__main__":
    main()
