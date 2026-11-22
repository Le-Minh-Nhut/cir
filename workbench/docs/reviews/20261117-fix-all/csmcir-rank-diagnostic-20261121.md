# CSMCIR dress-r14 rank diagnostic — 2026-11-21

## Finding

Full Q=2017/G=3817 arrays were read directly; no model execution or evaluator rerun. Query IDs, gallery IDs, and positive label indices match exactly between roles. All scores finite. Score max absolute difference `5.960464477539062e-7`; `5,553,185 / 7,700,189` score cells differ bitwise. Exact complete sorted-index rows differ for `1952/2017` queries. Positive target ranks differ for `1/2017` queries.

First sorted-row divergence: query 0, `dress:0000:B005X4PL1G:B0084Y8XIU`, first differing position/rank `1056`. Native puts `B00B5R9XQ4` before `B00B2JI3K2`; native scores are both `0.6384462118148804` (tie). Adapter scores: `B00B5R9XQ4=0.6384461522102356`, `B00B2JI3K2=0.6384462118148804`; adapter favors second by `5.960464477539063e-8`. Query 0 positive target rank is 4 on both sides.

Only positive-rank divergence: query 228, `dress:0228:B0087PLHB6:B004HX2DYW`, target `B004HX2DYW`: native rank 2, adapter rank 3; both hit R@10/R@50. At rank 2 native orders target `B004HX2DYW` (score `0.7318334579467773`) above `B004HX0CP4` (same score). Adapter orders `B004HX0CP4` (`0.7318333387374878`) above target (`0.7318333387374878`); stored score gap is exactly zero there. Thus positive-rank failure is a `tie_atol=1e-6` near-tie outcome, not evidence of a non-tie score crossing. Aggregate integer hits equality does not erase row differences.

Observed mechanism: exact score ties and sub-`1e-6` gaps interact with source-faithful `torch.argsort(1-scores)` ranking. Source evaluator uses this ordering (`workbench/third_party/CSMCIR/src/validate_blip_csmcir.py:43-46`); producer stores same `torch.argsort(1-scores)` semantics (`workbench/replay/run_fashioniq_v3.py:120-132`). Adapter scoring is max over 32 normalized target-token cosine dot products (`workbench/composition/model_csmcir.py:114-127`); source inference is max over 32 target-token similarities (`workbench/third_party/CSMCIR/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:439-451`). Observed floating-point score differences can change ordering within near-ties. No evidence supports capture misalignment: query/gallery ordered IDs and labels match exactly; plan order digests agree; bounded G1 captures had exact shared inputs per receipt. Source/runtime arithmetic is not proven causal beyond these measured near-ties; arrays alone cannot isolate reduction/precision effects versus tiesort details.

## Identity and policy

Plan `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.multimodal-q256-g8192.r14.v3-plan.json`, SHA-256 `17f82bc1612805673b4df637fa27f1dc8cc1c2a806da3570ee51ae3c1e973d6f`. Native artifact `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.multimodal-q256-g8192.r14.v3-plan.native_adapter_native.native.npz`, SHA-256 `ca9ae2127ecfe3da7474cb6299903453ca9369e0a75a7c82610d9a84ccc203c9`. Adapter artifact `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.multimodal-q256-g8192.r14.v3-plan.native_adapter_adapter.adapter.npz`, SHA-256 `d3663d99b67f2041ec2be29ce9023290ff698c170af7a67a26868087286f383c`. Checkpoint SHA-256 `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`; native source SHA-256 `5f654386cb353646831d59b62d370c0691ae5f97a10062650158242622ec80fb`; Adapter source SHA-256 `47df9b3637f534be0ac084134ac4d4ab5df99d629aa272c513d0e7609cffbbd7`.

Plan identity: native Torch 2.7.1+cu128/Python 3.10.22; adapter controller `workbench/envs/ctl312h/bin/python`; checkpoint/config/cohort identity `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e` / `0162a071d19cae1e834f77744e33338c0000eed88cdace4450f94ba7e2aedf0b` / `fashioniq-val-full-csmcir-dress-multimodal-q256-g8192-v3`. Query-order SHA-256 `363ce25bbc400ec44ab8757634d4a2e4b1b75725da360563b31a08755bb2a229`; gallery-order SHA-256 `98400ef5b45bd759e9b969330da9973d9697fff615947857f22df5c3598c0fc5`; reference-exclusions SHA-256 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`. Protocol: FashionIQ native relative/classic; target category `Final_Caption`; source strict=False reports `missing=[]`, `unexpected=token_importance`; native maybe_autocast; float32 captures; TF32 unchanged. Source and adapter roles use same full ordered population; reference exclusion hash is empty-list digest, consistent with all references remaining eligible per native FashionIQ policy.

The plan's `controls.native_adapter.tolerance` field is `atol=1e-5, rtol=1e-4`; aggregator applies this control tolerance when comparing saved score arrays, and it is also the forward-component control tolerance. This plan field is not the campaign's fixed scientific score criterion. Campaign score criterion remains absolute `1e-6`; observed max abs `5.960464477539062e-7` passes it. Separate fixed tie tolerance remains `1e-6`; forward FP32 component tolerance `atol=1e-5/rtol=1e-4`; batch control `atol=1e-4/rtol=1e-3`. No tolerance, sort, exclusion, or population change made. Existing custom component report records `native_adapter.ranks: exact sorted-index rows differ` and `native_adapter.positive_ranks: per-query target ranks differ`. This is not canonical gate verdict: existing receipt says canonical G2 remains pending; no parity PASS or scientific acceptance inferred. The existing preregistered policy requires exact ranks; this diagnostic does not waive that criterion.

## Reproducibility record

CPU-only analysis used required interpreter, CPU lock, one-thread environment, no model code, no blobs printed. Exact numerical comparison command:
```bash
flock -x /tmp/cir-fix-all-20261120-cpu.lock -c 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 workbench/envs/ctl312h/bin/python - <<'"'"'PY'"'"'
import numpy as np
b="workbench/artifacts/parity/v3/g1_g2.csmcir.dress.multimodal-q256-g8192.r14.v3-plan"
n=np.load(b+".native_adapter_native.native.npz",allow_pickle=False); a=np.load(b+".native_adapter_adapter.adapter.npz",allow_pickle=False)
ns=n["scores"]; ass=a["scores"]; nr=n["ranks"]; ar=a["ranks"]; nl=n["labels"]; al=a["labels"]
print("max_abs",float(np.max(np.abs(ns-ass))),"score_neq",int(np.count_nonzero(ns!=ass)))
print("rank_rows",int(np.count_nonzero(np.any(nr!=ar,axis=1))))
np_=np.argmax(nr==nl[:,None],axis=1); ap_=np.argmax(ar==al[:,None],axis=1)
print("positive_rank_rows",int(np.count_nonzero(np_!=ap_)))
print("IDs",bool(np.array_equal(n["query_ids"],a["query_ids"])),bool(np.array_equal(n["gallery_ids"],a["gallery_ids"])),"labels",bool(np.array_equal(nl,al)))
q=0; pos=int(np.flatnonzero(nr[q]!=ar[q])[0]); ni=int(nr[q,pos]); ai=int(ar[q,pos]); print("first",q,str(n["query_ids"][q]),"rankpos",pos+1,"native_item",str(n["gallery_ids"][ni]),"adapter_item",str(a["gallery_ids"][ai]),"native_scores",repr(float(ns[q,ni])),repr(float(ns[q,ai])),"adapter_scores",repr(float(ass[q,ni])),repr(float(ass[q,ai])),"native_gap",repr(float(ns[q,ni]-ns[q,ai])),"adapter_gap",repr(float(ass[q,ai]-ass[q,ni])))
print("target_ranks_q0",int(np_[0]+1),int(ap_[0]+1))
PY'
```
Exit 0. Output: `max_abs 5.960464477539062e-07 score_neq 5553185`; `rank_rows 1952`; `positive_rank_rows 1`; `IDs True True labels True`; first divergence/ranks/scores as recorded above; query-0 target ranks `4 4`. Separate finite-score check reported both sides `True`; positive-rank query 228 output: `target B004HX2DYW ranks 2 3 hit@10 True True hit@50 True True`. First attempt timed out at 120s while initializing/loading arrays and produced no result; no deadline rerun completed. Analysis output is from actual saved arrays, not a model or evaluator rerun.
