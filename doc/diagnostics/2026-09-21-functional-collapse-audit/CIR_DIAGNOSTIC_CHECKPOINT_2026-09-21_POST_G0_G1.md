# CIR / IAG-SRME — POST-G0/G1 DIAGNOSTIC CHECKPOINT

**Ngày chốt:** 21/09/2026  
**Repo:** `https://github.com/Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-iag-srme-v2-r0-functional-collapse-audit`  
**Mục đích:** checkpoint chuyên biệt cho toàn bộ chuỗi chẩn đoán thực nghiệm từ G0 → G1 → held-out selector audit → frozen ScoreNet stream-refit. File này tập trung vào **bằng chứng đã đo**, diễn giải nhân quả ở mức được dữ liệu cho phép, failure modes đã tách được, các giả thuyết đã bị loại/giảm ưu tiên, và experiment kế tiếp cần chạy. Một cuộc trò chuyện mới có thể đọc riêng file này để tiếp tục mà không cần hỏi lại lịch sử chẩn đoán.

---

## 0. Executive summary — trạng thái nghiên cứu hiện tại

Bài toán CIR hiện được xây dựng như một planner tuần tự: từ reference image + text instruction, tại mỗi bước model sinh `K=4` candidate edits/actions, preview hệ quả của từng candidate, ScoreNet dự đoán marginal utility so với KEEP, chọn một candidate hoặc STOP, commit candidate được chọn vào internal state, rồi lặp lại tối đa `T=3` bước. Không có external observation mới sau mỗi action; toàn bộ recurrent state về sau là state do model tự xây dựng.

Failure ban đầu được mô tả là **winner-index / candidate collapse**: một index mạnh kéo dài và ScoreNet gần như luôn chọn cùng index. Sau chuỗi audit hiện tại, failure này không còn được xem là một cơ chế duy nhất. Dữ liệu đã tách nó thành ít nhất bốn thành phần:

1. **Proposal / functional representation collapse.** Với `proposal_mode=attention`, các proposal khác index gần như đồng nhất, action downstream gần như đồng nhất, `delta_q` cũng gần như đồng hướng. G0 localize contraction đầu tiên vào output của ProposalNet cross-attention.
2. **Missing identity path trong ProposalNet là nguyên nhân quan trọng của representation collapse.** Residual bypass `q + att` phục hồi sibling identity rất mạnh mặc dù raw attention output vẫn collapse. `residual_ln` giữ diversity ổn định hơn `residual` trong one-epoch test.
3. **Selector / ScoreNet collapse là một failure độc lập.** Sau khi `residual_ln` làm candidates khác nhau rõ rệt, ScoreNet vẫn monopolize C3 trên held-out VAL-160: `98.1%` executed edits chọn C3, trong khi oracle thường thích C0. Đây không còn có thể giải thích chỉ bằng proposal geometry collapse.
4. **Candidate semantic competence vẫn yếu.** `residual_ln` tạo slot-different actions/effects, nhưng correct-vs-shuffled-caption comparisons cho thấy cùng một slot gần như không thay đổi theo caption (`proposal cosine ≈ 0.995`, `action cosine ≈ 0.993`, `delta_q cosine ≈ 0.9997`). Mean useful candidate count vẫn thấp. Vì vậy **diversity đã tăng nhưng diversity đó chưa được chứng minh là instruction-grounded/useful diversity**.

Frozen upstream + gain-only ScoreNet stream-refit cho thấy ScoreNet **có partial learnability**: held-out Pearson tăng từ khoảng `-0.003` lên `0.108`, selected utility từ `-0.0080` lên `+0.0032`, regret giảm từ `0.04635` xuống `0.03477`, harmful-execution rate giảm từ `69.4%` xuống `47.9%`. Nhưng policy vẫn không được cứu: exact oracle accuracy gần như không đổi (`22.2% → 23.0%`), slot monopoly chuyển từ C3 sang C0 (`78.5%` của executed edits), và STOP bùng lên `52.9%` trong khi oracle STOP chỉ `12.1%`. Gain-only regression làm scores bị shrink quanh/ dưới zero, nên zero-threshold STOP trở thành failure mới.

**Kết luận hiện tại:**

- `residual_ln` đã giải quyết mạnh **representation / functional sibling collapse**, nhưng chưa chứng minh cải thiện retrieval hay semantic grounding.
- ScoreNet hiện là bottleneck thật, nhưng dữ liệu **chưa đủ để kết luận architecture/features của ScoreNet không đủ**. Frozen gain-only refit chứng minh nó học được một phần signal.
- Gain-only Huber không phải rescue đúng cho decision policy hiện tại vì decision boundary tại utility `0` rất quan trọng cho STOP.
- Candidate competence vẫn thấp và correct-vs-shuffled sensitivity cho thấy slot diversity có thể phần lớn là **slot-identity-driven** hơn là instruction-driven.
- DPP numerical catastrophe không được quan sát trong distributed one-epoch audit; không tăng `lambda_DPP` hoặc sửa DPP lúc này.
- Experiment kế tiếp sạch nhất: **freeze upstream residual_ln, refit ScoreNet bằng đúng pair+gain objective của main training**, giữ source hyperparameters. Mục tiêu là tách `online moving-target/co-training` khỏi `objective mismatch`.

---

# 1. Provenance và code state

## 1.1 Branch / commits

Toàn bộ work thuộc branch:

```text
exp/e2e-iag-srme-v2-r0-functional-collapse-audit
```

Các commit quan trọng đã xác minh trước chuỗi run hiện tại:

```text
c2e24572014f8c7666af9224a7c853304bf5a395  historical base trước G0/G1
48a51e3                                      Improve functional collapse diagnostics
3d7bee5                                      Add ProposalNet residual normalization ablations
fd582002e3621819c83cb7e14ce37aa550a0650a  Add DPP zero-effect gradient audit
7971dea2dd8d871fb3bfe6cccfa51a905ee2fc1f  Fix DPP gradient audit run accounting
```

Remote branch HEAD được kiểm tra trực tiếp trước experiment và là `7971dea2...`.

## 1.2 Local patch sau commit `7971...`

Để chạy scorer-refit trên checkpoint full-finetune, local `src/refit_score_net.py` đã bỏ một historical guard hard-code chỉ cho phép policy:

```text
finetune_policy=text_only
train_vision=false
train_text=true
train_text_projection=false
```

Checkpoint `residual_ln` hiện tại là:

```text
backbone=fgclip_base_full
finetune_policy=full
train_vision=true
train_text=true
train_text_projection=false
global_readout_mode=learned_qg
```

Guard này chỉ là restriction lịch sử của scorer rescue cũ. Compatibility thực sự vẫn được kiểm tra sau đó bằng `_validate_source_checkpoint(...)`. Patch không đổi ScoreNet objective, candidate generator, checkpoint weights hoặc inference policy; nó chỉ cho phép refit tooling hoạt động với backbone policy của source checkpoint hiện tại.

**Reproducibility note:** patch này nên được review bằng `git diff -- src/refit_score_net.py` và commit cùng diagnostic artifacts nếu muốn người khác reproduce stream-refit trên full backbone.

---

# 2. Canonical model path và invariants quan trọng

Mỗi live timestep về cơ bản chạy:

```text
current recurrent state
    ↓
ProposalNet → K edits/proposals
    ↓
Grounder → read/write support
    ↓
ActionFusion → K actions
    ↓
Executor → K candidate states
    ↓
retrieval readout → K candidate queries
    ↓
ScoreNet → K predicted marginal utilities
    ↓
argmax + zero STOP rule
    ↓
commit one candidate OR STOP
```

### 2.1 Proposal modes trong G1

Bốn modes được test:

```text
attention      : out = att
attention_ln   : out = LN_out(att)
residual       : out = q + att
residual_ln    : out = LN_out(q + att)
```

`q` ở đây là actual `query_post_norm`. Learned `output_norm` chỉ tồn tại ở `_ln` modes.

### 2.2 ScoreNet

ScoreNet là shared independent predictor của marginal utility so với KEEP. Features gồm:

```text
context(current_global, text_global)
action
local_effect from delta
candidate_global - current_global
compatibility = action * global_effect
```

Trong main model, các tensors upstream truyền vào ScoreNet đều `.detach()`. Vì vậy score losses train ScoreNet projections/MLP nhưng **không train upstream proposal/grounder/action/executor** qua scorer loss.

### 2.3 Teacher utility và STOP semantics

Teacher utility:

\[
u_k = L_{retrieval}(q_t) - L_{retrieval}(q_t^{(k)})
\]

- `u_k > 0`: candidate cải thiện retrieval so với parent/KEEP.
- `u_k < 0`: candidate harmful.
- runtime STOP anchor hiện tại là `0` (`epsilon_stop=0.0`).
- execute nếu ít nhất một predicted score > 0; nếu không thì STOP.

Main objective ScoreNet supervision có cả:

```text
lambda_pair = 0.5
lambda_gain = 0.5
```

Pairwise term học relative sibling ranking; absolute gain Huber học numerical marginal utility.

Điều này rất quan trọng khi diễn giải gain-only stream-refit: intervention đó vừa freeze upstream **vừa bỏ pairwise term**, nên nó chưa phải phép thử thuần túy của moving-target.

---

# 3. G0 — precision / metric / DPP audit trước khi sửa architecture

Mục tiêu G0 là tránh sửa model dựa trên metric artifact hoặc DPP numerical pathology chưa được chứng minh.

## 3.1 Diagnostic robustness fixes

Functional-collapse diagnostics được sửa để tránh các kết luận giả:

- cosine diversity mask low-energy vectors thay vì coi tiny vectors là directions hợp lệ;
- effective rank dùng scale-normalized Gram trong double precision;
- log cả centered/un-centered statistics;
- log norm quantiles / valid fractions;
- phân biệt energy collapse với directional collapse;
- với `K=4`, centered sibling rank tối đa chỉ là `K-1=3` nên interpretation phải đúng bound này;
- selection audit dùng KEEP-aware oracle/regret;
- score margin-to-STOP sử dụng `epsilon_stop` đúng semantics.

## 3.2 DPP zero-effect gradient concern

Một phản ví dụ lý thuyết cho thấy DPP có thể tạo gradient lớn tại near-zero effect slot nếu các slot khác mở quality gate. Vì vậy G0 thêm gradient audit thay vì giả định DPP vô hại.

Audit sử dụng production `functional_dpp_loss`, production history semantics, production teacher quality gate. FP32 reference là **DPP arithmetic/backward từ live `delta_q`**, không phải full-model FP32 rerun. Diagnostic autograd xảy ra trước main backward và không ghi `.grad` của optimizer path.

Run accounting sau patch `7971...`:

- cumulative global batch index xuyên epoch;
- interval/max update cap tính global;
- `max_updates=0` disable;
- log `sample_ids` và row indices;
- so sánh objective DPP valid rows vs audit valid rows;
- test interval continuity / cross-epoch / max0 / production count.

Focused tests: `15 passed`. Full suite sau patch: `168 passed, 2 skipped`, Ruff/compile passed.

## 3.3 DPP focused early run

One epoch, `proposal_mode=attention`, 563 batches.

Summary:

```text
total loss     = 1.5243
mean_recall    = 15.131
audit_updates  = 16
objective/audit valid rows = 15 / 15
count mismatch = 0
eligible rows  = 449
valid rate     = 0.033408
nonfinite live = 0
```

Effect norms:

```text
p00  = 0.00919462
p50  = 0.0112703
p100 = 0.012766
```

Live DPP gradient norms:

```text
p00  = 0.251398
p50  = 1.57596
p100 = 2.80202
```

FP32 reference gần như khớp:

```text
p00  = 0.251381
p50  = 1.57597
p100 = 2.80262
```

Audit này chỉ sample đầu trajectory nên chưa đủ kết luận late-training.

## 3.4 DPP distributed run

Sampling phân bố khoảng batch `0, 35, ..., 525`, 16 updates xuyên epoch.

```text
audit_updates             = 16
objective/audit valid     = 168 / 168
count mismatch            = 0
eligible                  = 1124
valid_rate                = 0.149466
nonfinite live            = 0
```

Effect norm distribution:

```text
p00  = 0.0128562
p10  = 0.023425
p50  = 0.0385555
p90  = 0.054038
p99  = 0.0602937
p100 = 0.0614443
```

Live gradient distribution:

```text
p00  = 0.00148857
p10  = 0.00509115
p50  = 0.0497531
p90  = 0.211432
p99  = 1.34276
p100 = 1.49681
```

FP32 reference rất gần live:

```text
p00  = 0.00148859
p10  = 0.00509109
p50  = 0.0496835
p90  = 0.21105
p99  = 1.34252
p100 = 1.49738
```

Có 1 AMP skip/scale-decrease overlap, nhưng overlap không đủ để suy ra causal relation.

### G0-DPP conclusion

Trong one-epoch trajectory và sampled updates đã đo:

- không thấy nonfinite DPP gradients;
- FP32 DPP-only reference khớp live AMP rất tốt;
- low-effect rows đôi khi gradient lớn hơn nhưng max observed ~`1.5` ở distributed run, không phải catastrophe;
- không có bằng chứng hiện tại rằng DPP numerical pathology là primary cause của collapse.

**Không tăng `lambda_DPP`. Không sửa DPP trước.** Đây là kết luận theo dữ liệu đã đo, không phải claim DPP luôn an toàn ở mọi checkpoint/training regime.

---

# 4. G0 attention baseline — localization của representation collapse

Baseline attention-only one-epoch run:

```text
checkpoint/run: outputs/2026-09-21/18-26-04
mean_recall ≈ 15.131
```

## 4.1 Full-epoch functional geometry

```text
proposal cosine       ≈ 0.9994
proposal eff. rank    ≈ 1.0604
proposal spread       ≈ 0.0175

alpha_read cosine     ≈ 1.0000
exec mask cosine      ≈ 1.0000
exec spread           ≈ 0.0034

entity cosine         ≈ 1.0000
entity eff. rank      ≈ 1.0022
entity spread         ≈ 0.0024

action cosine         ≈ 0.9998
action eff. rank      ≈ 1.0335
action spread         ≈ 0.0099

delta cosine          ≈ 0.9958
delta eff. rank       ≈ 1.0044
delta spread          ≈ 0.0030
valid delta fraction  ≈ 0.996
low-energy fraction   ≈ 0.004

delta_q cosine        ≈ 0.9988
delta_q eff. rank     ≈ 1.0781
delta_q spread        ≈ 0.0224
```

K candidates về mặt effect gần như cùng một hướng.

## 4.2 Internal ProposalNet localization

Full-epoch internal metrics:

```text
base_query cosine                   ≈ -0.0059
base_query rank                     ≈ 3.9955

conditioned_residual cosine         ≈ 0.9937
conditioned_residual rank           ≈ 1.2460

query_pre_norm cosine               ≈ 0.6266
query_pre_norm rank                 ≈ 3.1128

query_post_norm cosine              ≈ 0.6265
query_post_norm rank                ≈ 3.1139

raw_attention_output cosine         ≈ 0.9994
raw_attention_output rank           ≈ 1.0604

proposal_output cosine              ≈ 0.9994
proposal_output rank                ≈ 1.0604
```

Retention estimates:

```text
conditioner/base norm ratio         ≈ 1.278412
base_pre_cosine                     ≈ 0.629162
mean_relative_displacement          ≈ 1.278412
attention_diversity_retention       ≈ 0.032722
attention_rank_retention            ≈ 0.340451
```

First 10 vs last 10 updates cũng cho cùng pattern:

```text
FIRST 10:
q_post cosine/rank    ≈ 0.5613 / 3.2845
raw_att cosine/rank   ≈ 0.9908 / 1.2869

LAST 10:
q_post cosine/rank    ≈ 0.6402 / 3.0760
raw_att cosine/rank   ≈ 0.9998 / 1.0387
```

### G0 representation conclusion

Ngay trước attention, candidate queries vẫn giữ substantial sibling rank (~3.1/4). Ngay sau cross-attention, rank rơi về ~1.06 và cosine lên ~0.9994. Vì vậy **first major representation contraction được localize rất mạnh tại ProposalNet cross-attention output**.

Đây ban đầu là localization evidence, chưa phải causal proof. G1 residual intervention sau đó cung cấp bằng chứng can thiệp mạnh hơn.

---

# 5. G1 — ProposalNet residual × output-LN 2×2 causal intervention

Bốn modes test cùng one-epoch configuration:

```text
attention
attention_ln
residual
residual_ln
```

Baseline `attention` được dùng làm anchor. Ba run mới:

```text
attention_ln : outputs/2026-09-21/19-44-18
residual     : outputs/2026-09-21/19-53-29
residual_ln  : outputs/2026-09-21/20-02-45
```

## 5.1 Headline comparison

| mode | mean recall | proposal rank | delta_q rank | delta_q cosine |
|---|---:|---:|---:|---:|
| attention | 15.131 | 1.0604 | 1.0781 | 0.9988 |
| attention_ln | 15.220 | 1.0600 | 1.0792 | 0.9987 |
| residual | 14.854 | 2.8211 | 1.7215 | 0.2147 |
| residual_ln | 14.447 | 3.0255 | 2.2744 | 0.0926 |

**Không được dùng table này để tuyên bố `residual_ln` tốt nhất về retrieval.** Tất cả chỉ là one-epoch runs; mean recall của residual variants thậm chí thấp hơn baseline ở epoch này. Giá trị của G1 là causal/diagnostic geometry.

## 5.2 `attention_ln`

Epoch summary:

```text
total       = 1.5170
mean_recall = 15.220
```

Full functional:

```text
proposal cosine/rank    ≈ 0.9994 / 1.0600
action cosine/rank      ≈ 0.9995 / 1.0545
delta cosine/rank       ≈ 0.9958 / 1.0036
delta_q cosine/rank     ≈ 0.9987 / 1.0792
```

Internal:

```text
q_post cosine/rank      ≈ 0.6287 / 3.1077
raw_att cosine/rank     ≈ 0.9994 / 1.0600
proposal cosine/rank    ≈ 0.9994 / 1.0600
```

**Conclusion:** LayerNorm sau một attention output đã collapse không phục hồi candidate identity. `attention_ln` gần như baseline.

## 5.3 `residual`

Epoch summary:

```text
total       = 1.8625
mean_recall = 14.854
```

Full functional:

```text
proposal cosine/rank    ≈ 0.7216 / 2.8211
alpha_read cosine       ≈ 0.8705
exec mask cosine        ≈ 0.8553
entity cosine/rank      ≈ 0.9833 / 1.2091
action cosine/rank      ≈ 0.6939 / 2.8863
delta cosine/rank       ≈ 0.7623 / 1.8613
delta_q cosine/rank     ≈ 0.2147 / 1.7215
```

Internal:

```text
q_post cosine/rank      ≈ 0.6560 / 3.0283
raw_att cosine/rank     ≈ 0.9987 / 1.0955
proposal cosine/rank    ≈ 0.7216 / 2.8211
```

Raw attention vẫn collapse, nhưng identity residual bypass phục hồi proposal identity sau attention. Diversity propagates đến actions/effects.

Tuy nhiên proposal diversity bị degrade phần nào về cuối epoch:

```text
FIRST 10 proposal cosine/rank ≈ 0.6147 / 3.1462
LAST 10  proposal cosine/rank ≈ 0.7581 / 2.6969
```

## 5.4 `residual_ln`

Epoch summary:

```text
total       = 1.6195
mean_recall = 14.447
```

Full functional:

```text
proposal cosine/rank    ≈ 0.6564 / 3.0255
alpha_read cosine       ≈ 0.8171
exec mask cosine        ≈ 0.7833
entity cosine/rank      ≈ 0.9814 / 1.2284
action cosine/rank      ≈ 0.5136 / 3.2530
delta cosine/rank       ≈ 0.5367 / 2.3009
delta_q cosine/rank     ≈ 0.0926 / 2.2744
```

Internal:

```text
q_post cosine/rank      ≈ 0.5877 / 3.2152
raw_att cosine/rank     ≈ 0.9973 / 1.1402
proposal cosine/rank    ≈ 0.6564 / 3.0255
```

Stability qua epoch:

```text
FIRST 10 proposal cosine/rank ≈ 0.6146 / 3.1466
LAST 10  proposal cosine/rank ≈ 0.6156 / 3.1450
```

Khác `residual`, `residual_ln` giữ proposal diversity gần như không suy giảm cuối epoch.

### G1 causal conclusion

- Post-attention LayerNorm một mình không cứu collapse.
- Raw cross-attention outputs vẫn collapse ngay cả ở residual modes.
- Identity residual bypass làm final proposals và downstream effects khác nhau rõ rệt.
- `residual_ln` giữ diversity ổn định hơn `residual` trong one-epoch trajectory.

Paper-safe wording:

> G0 strongly localizes the first major representation contraction to ProposalNet cross-attention. Learned/conditioned queries retain substantial sibling rank immediately before attention, whereas raw attention outputs collapse. G1 intervention shows that adding an identity residual bypass restores proposal and downstream action/effect diversity despite the raw attention output remaining highly collapsed; post-attention normalization alone does not. This supports a causal role for the missing identity path in representation collapse under the tested one-epoch configuration, while usefulness and retrieval benefit remain unresolved.

---

# 6. Online TRAIN selector audit sau G1

`selection_quality_audit.overall` được aggregate qua toàn bộ 563 train updates cho ba G1 variants.

## 6.1 Full-epoch comparison

### attention_ln

```text
executed_fraction               = 0.852800
stop_fraction                   = 0.147200
oracle_keep_fraction            = 0.588552
useful_candidate_count          = 1.618175
selected_utility                = -0.000689
oracle_utility                  = 0.007489
one_step_regret                 = 0.008178
teacher_oracle_utility_margin   = 0.008176
teacher_utility_std             = 0.021495
score_margin_mean               = 0.013717

selected occupancy              = [0.2150, 0.2108, 0.2127, 0.2143]
raw_best occupancy              = [0.2515, 0.2490, 0.2494, 0.2501]
oracle occupancy                = [0.1091, 0.0999, 0.1009, 0.1015]
```

### residual

```text
executed_fraction               = 0.905118
stop_fraction                   = 0.094882
oracle_keep_fraction            = 0.148448
useful_candidate_count          = 2.072458
selected_utility                = 0.000085
oracle_utility                  = 0.006468
one_step_regret                 = 0.006383
teacher_oracle_utility_margin   = 0.003825
teacher_utility_std             = 0.009693
score_margin_mean               = 0.043224

selected occupancy              = [0.1625, 0.1811, 0.3834, 0.1781]
raw_best occupancy              = [0.1849, 0.2052, 0.4066, 0.2034]
oracle occupancy                = [0.3765, 0.0971, 0.3366, 0.0414]
```

### residual_ln

```text
executed_fraction               = 0.892677
stop_fraction                   = 0.107323
oracle_keep_fraction            = 0.139444
useful_candidate_count          = 1.950126
selected_utility                = -0.000526
oracle_utility                  = 0.030395
one_step_regret                 = 0.030921
teacher_oracle_utility_margin   = 0.019688
teacher_utility_std             = 0.037934
score_margin_mean               = 0.023049

selected occupancy              = [0.1773, 0.1093, 0.1423, 0.4638]
raw_best occupancy              = [0.2017, 0.1286, 0.1642, 0.5055]
oracle occupancy                = [0.2743, 0.1911, 0.2353, 0.1599]
```

`residual_ln` đặc biệt thú vị vì oracle utility lớn hơn rất nhiều so với selected utility. Candidate set chứa tốt hơn nhưng selector không khai thác được.

## 6.2 `residual_ln` temporal dynamics trong epoch

### First 50 updates

```text
C0 score=+0.019850 teacher=+0.001217 raw_best=.301 selected=.264 oracle=.443
C1 score=+0.013753 teacher=+0.001128 raw_best=.207 selected=.173 oracle=.083
C2 score=+0.015970 teacher=+0.001108 raw_best=.245 selected=.206 oracle=.044
C3 score=+0.016498 teacher=+0.001179 raw_best=.247 selected=.215 oracle=.073

selected utility       = +0.001019
oracle utility         = +0.004842
regret                 = 0.003824
teacher utility std    = 0.009743
STOP fraction          = 0.141782
oracle KEEP fraction   = 0.356833
```

### Middle 50 updates

```text
C0 score=-0.009835 teacher=+0.000271 raw_best=.129 selected=.112 oracle=.236
C1 score=-0.000498 teacher=+0.000329 raw_best=.222 selected=.192 oracle=.232
C2 score=-0.009558 teacher=-0.001447 raw_best=.112 selected=.094 oracle=.328
C3 score=+0.017994 teacher=-0.000155 raw_best=.538 selected=.485 oracle=.106

selected utility       = -0.000145
oracle utility         = +0.016649
regret                 = 0.016794
teacher utility std    = 0.020648
STOP fraction          = 0.117729
oracle KEEP fraction   = 0.097992
```

### Last 50 updates

```text
C0 score=+0.017246 teacher=-0.004905 raw_best=.422 selected=.389 oracle=.214
C1 score=-0.031692 teacher=-0.005271 raw_best=.018 selected=.014 oracle=.225
C2 score=-0.020510 teacher=-0.007646 raw_best=.027 selected=.023 oracle=.173
C3 score=+0.021403 teacher=-0.002175 raw_best=.533 selected=.503 oracle=.313

selected utility       = -0.002575
oracle utility         = +0.069451
regret                 = 0.072026
teacher utility std    = 0.085700
STOP fraction          = 0.071232
oracle KEEP fraction   = 0.074924
```

### Online interpretation

Trong cùng epoch:

```text
oracle utility: 0.00484 → 0.01665 → 0.06945
regret:         0.00382 → 0.01679 → 0.07203
selected util: +0.00102 → -0.00015 → -0.00258
```

Upstream candidate opportunity tăng rất mạnh trong train trajectory, nhưng ScoreNet khai thác càng tệ. C3 bias xuất hiện rõ giữa/cuối epoch mặc dù C3 không tương ứng với oracle occupancy.

Tuy vậy đây chỉ là online TRAIN telemetry. Vì policy/state distribution thay đổi cùng training, cần held-out cohort để xác nhận selector failure generalizes.

---

# 7. Held-out VAL-160 paired diagnostic — attention vs residual_ln

Cùng một persistent manifest:

```text
outputs/diagnostics/2026-09-21/shared_val160.json
```

20 batches × 8 samples = 160 starting examples, cùng order/batching. Diagnostic uses targets chỉ để tạo teacher/oracle, không đưa target vào inference features.

## 7.1 Attention baseline held-out

Checkpoint:

```text
outputs/2026-09-21/18-26-04/best.pt
epoch = 1
checkpoint metric = 15.13077262789011
```

Automatic flags:

```text
HARMFUL_EXECUTIONS                  61.7% executed actions teacher-negative
LOW_EXACT_ORACLE_ACCURACY           25.4%
DPP_STARVED_FOR_GOOD_CANDIDATES     mean useful = 0.00
FUNCTIONAL_COLLAPSE                 mean sibling delta_q cosine = 0.9926
```

Selector summary:

```text
exact oracle accuracy       = 0.2543
stop/execute accuracy       = 0.4914
execute rate                = 0.6735
harmful / executions        = 0.6173
harmful / all decisions     = 0.4158
missed-opportunity STOP     = 0.1203
selected utility            = 0.00018
oracle utility              = 0.00084
regret                      = 0.00066
ScoreNet/teacher Pearson    = -0.0334
STOP precision              = 0.6316
STOP recall                 = 0.3468
STOP F1                     = 0.4478
```

Sibling geometry:

```text
proposal off-diagonal cosine ≈ 0.9997–0.9998
action off-diagonal cosine   = 1.0000
delta_q off-diagonal cosine  ≈ 0.9925–0.9928
candidate-state L2           ≈ 0.0022–0.0036
```

Attention baseline therefore has genuine held-out functional collapse.

## 7.2 Residual_ln held-out before scorer refit

Checkpoint:

```text
outputs/2026-09-21/20-02-45/best.pt
epoch = 1
checkpoint metric = 14.446642498175304
```

Automatic flags:

```text
SELECTOR_SLOT_COLLAPSE              C3 selected in 98.1% executed edits
SELECTOR_BIAS_NOT_ORACLE_BIAS       selector C3, oracle most often C0
HARMFUL_EXECUTIONS                  69.4%
LOW_EXACT_ORACLE_ACCURACY           22.2%
DPP_STARVED_FOR_GOOD_CANDIDATES     mean useful = 0.63
```

### Per-slot behavior

```text
C0: selected/execute  1.869% | oracle/oracle-execute 40.351%
    mean score +0.00685 | mean teacher utility +0.00241

C1: selected/execute  0.000% | oracle/oracle-execute 18.797%
    mean score -0.03110 | mean teacher utility -0.00378

C2: selected/execute  0.000% | oracle/oracle-execute 14.787%
    mean score -0.02789 | mean teacher utility -0.00493

C3: selected/execute 98.131% | oracle/oracle-execute 26.065%
    mean score +0.02411 | mean teacher utility -0.00599

STOP selected/all 4.036% | oracle STOP 10.538%
```

C3 có **mean predicted score cao nhất** nhưng **mean teacher utility tệ nhất** trong bốn slots.

### Selector summary

```text
exact oracle accuracy       = 0.2220
stop/execute accuracy       = 0.8587
execute rate                = 0.9596
harmful / executions        = 0.6939
harmful / decisions         = 0.6659
missed-opportunity STOP     = 0.0381
selected utility            = -0.00801
oracle utility              = +0.03834
regret                      = 0.04635
ScoreNet/teacher Pearson    = -0.003085
STOP precision              = 0.0556
STOP recall                 = 0.0213
STOP F1                     = 0.0308
```

Đây là bằng chứng held-out rất mạnh rằng selector bottleneck là thật. Candidate set có oracle utility `+0.03834`, nhưng live selector biến nó thành `-0.00801`.

### Functional geometry đã được phục hồi

Proposal sibling cosine matrix off-diagonal:

```text
0.6036 – 0.6250
```

Action sibling cosine matrix off-diagonal:

```text
C0-C1 0.4878
C0-C2 0.4312
C0-C3 0.0752
C1-C2 0.5348
C1-C3 0.0878
C2-C3 0.1224
```

`delta_q` sibling cosine:

```text
C0-C1 +0.3401
C0-C2 +0.1350
C0-C3 -0.6242
C1-C2 +0.1312
C1-C3 -0.3768
C2-C3 -0.2026
```

Candidate-state pairwise L2:

```text
~15.1 – 26.7
```

Đây là qualitative regime hoàn toàn khác attention baseline; candidates không còn gần-identical effect.

## 7.3 Candidate competence vẫn yếu

Held-out residual_ln:

```text
mean useful candidate count        = 0.6300
rows with >=2 useful candidates    = 0.1659
positive candidate fraction        = 0.4170
```

Tức geometry diversity mạnh nhưng average row vẫn có ít useful alternatives theo DPP quality definition.

---

# 8. Correct-vs-shuffled-caption sensitivity — finding rất quan trọng

Đây là điểm không được bỏ sót khi handoff.

Diagnostic ngoài sibling geometry còn rerun cùng starting cohort với **correct caption** và **category-shuffled caption** rồi so sánh cùng slot/effect.

## 8.1 Attention baseline

```text
actions cosine(correct, shuffled)      = 0.996924
proposals cosine(correct, shuffled)    = 0.983802
delta_q cosine(correct, shuffled)      = 0.989337

best candidate utility correct-shuffle = +0.000006
candidate utility correct-shuffle      = +0.000017
oracle policy utility correct-shuffle  = +0.000012
terminal retrieval correct advantage   = +0.000248
```

## 8.2 Residual_ln before scorer refit

```text
actions cosine(correct, shuffled)      = 0.993173
proposals cosine(correct, shuffled)    = 0.995140
delta_q cosine(correct, shuffled)      = 0.999687

best candidate utility correct-shuffle = +0.000108
candidate utility correct-shuffle      = +0.000098
oracle policy utility correct-shuffle  = +0.000128

selected utility correct               = -0.007238
selected utility shuffled              = -0.006696
selected utility correct-shuffle       = -0.000542
terminal retrieval correct advantage   = -0.004196
```

### Critical interpretation

Hai loại cosine **không được nhầm**:

- sibling cosine: so C0/C1/C2/C3 với nhau dưới cùng caption;
- caption-sensitivity cosine: so cùng slot dưới correct caption vs shuffled caption.

`residual_ln` làm sibling actions rất khác nhau, nhưng **cùng slot hầu như không đổi khi caption bị shuffle**. Ví dụ sibling C0-C3 action cosine chỉ `0.0752`, trong khi correct-vs-shuffled action cosine trung bình `0.9932`.

Vì vậy một interpretation rất quan trọng là:

> Residual identity bypass phục hồi slot identity và tạo slot-specific functional trajectories, nhưng hiện chưa có bằng chứng các trajectories đó được điều khiển mạnh bởi instruction semantics. Diversity có thể phần lớn đến từ learned slot/query priors hơn là caption-dependent specialization.

Đây là lý do không được viết “candidate problem đã được giải”. Chính xác hơn:

```text
representation diversity:    đã cải thiện mạnh
functional sibling diversity: đã cải thiện mạnh
instruction sensitivity:      vẫn rất yếu
candidate competence:         vẫn yếu
```

Đây có thể là bottleneck upstream kế tiếp sau khi selector objective được cô lập.

---

# 9. Frozen ScoreNet gain-only stream-refit

## 9.1 Vì sao chuyển từ disk cache sang stream-refit

Attempt đầu tiên dùng `collect → refit` fixed cache.

Partial collection tạo:

```text
270 shard files
~55 GB
shard_000000.pt ... shard_000269.pt
```

Không có:

```text
manifest.json
collection_report.json
```

Full TRAIN cần khoảng 563 batches; extrapolation cho thấy full raw cache có thể vượt ~110 GB. Process dừng giữa collect và partial cache không dùng được bởi `_load_cache_manifest(...)` vì không có complete manifest.

Vì vậy bỏ đường full disk cache và dùng `stream_refit` có sẵn trong repo.

`stream_refit` giữ một `behavior_model` frozen từ source checkpoint, tạo rollouts/candidates online từ model này, còn một deep-copied ScoreNet riêng được train. Code fingerprint-check trước/sau để bảo đảm behavior model không đổi.

## 9.2 Source / configuration

```text
source checkpoint:
outputs/2026-09-21/20-02-45/best.pt

source epoch              = 1
source optimizer_step     = 560
source batch_step         = 563
source git SHA            = 7971dea2dd8d871fb3bfe6cccfa51a905ee2fc1f

proposal_mode             = residual_ln
K                         = 4
T                         = 3
epsilon_stop              = 0.0
backbone                  = fgclip_base_full
precision                 = fp16 behavior rollout
refit ScoreNet arithmetic = fp32
```

Gain-only refit:

```text
stream_epochs             = 3
TRAIN samples             = 18,000
total sample visits       = 54,000
optimizer steps           = 5,067
valid decisions           = 149,595
teacher invalid rows      = 0
learning rate             = 1e-5
weight decay              = 0.01
lambda_pair               = 0.0
loss                      = absolute_gain_huber_only
trainable params          = 1,643,777 (ScoreNet only)
```

## 9.3 Intervention validity / fingerprints

Behavior model:

```text
before = 8bef450ee14cf4c2414f5d06706001747b5ff1df2d0db8057e0ec323fb07894b
after  = 8bef450ee14cf4c2414f5d06706001747b5ff1df2d0db8057e0ec323fb07894b
unchanged = true
```

Behavior ScoreNet:

```text
before = 73aff157aae0d2ddb5d572726a4260afa35d4fe42599ea61f5c8b14cde5dacb2
after  = 73aff157aae0d2ddb5d572726a4260afa35d4fe42599ea61f5c8b14cde5dacb2
unchanged = true
```

Trainable copied ScoreNet:

```text
before = 73aff157aae0d2ddb5d572726a4260afa35d4fe42599ea61f5c8b14cde5dacb2
after  = ad6e67a658d613147a8c30a544ce4f1689bca04040df8a88aef0cc0ce19b64d1
changed = true
```

Do đó experiment thực sự là scorer-only intervention trên frozen behavior/candidate policy.

## 9.4 Stream training calibration telemetry

Across 598,380 candidate labels:

```text
Pearson                    = 0.108470
bias                       = +0.000102
MAE                        = 0.052083
RMSE                       = 0.080354
sign agreement at zero     = 0.576961
```

Teacher utility decile means vs predicted score means:

```text
teacher -0.15079 → score -0.00930
teacher -0.06675 → score -0.00795
teacher -0.03788 → score -0.00772
teacher -0.01996 → score -0.00711
teacher -0.00808 → score -0.00659
teacher -0.00100 → score -0.00392
teacher +0.00656 → score -0.00226
teacher +0.02299 → score -0.00280
teacher +0.05303 → score -0.00274
teacher +0.14729 → score -0.00318
```

Đây là strong shrinkage toward a narrow negative band. Even large positive teacher utilities vẫn nhận predicted negative score trung bình.

---

# 10. Held-out VAL-160 sau gain-only stream-refit

Checkpoint:

```text
outputs/2026-09-21/residual_ln_score_stream/score_stream.pt
```

Diagnostic:

```text
outputs/diagnostics/2026-09-21/residual_ln_stream_val160/
```

Automatic flags:

```text
SELECTOR_SLOT_COLLAPSE              C0 selected in 78.5% executed edits
SELECTOR_BIAS_NOT_ORACLE_BIAS       selector C0, oracle most often C3
HARMFUL_EXECUTIONS                  47.9%
LOW_EXACT_ORACLE_ACCURACY           23.0%
DPP_STARVED_FOR_GOOD_CANDIDATES     mean useful = 0.56
```

## 10.1 Per-slot

```text
C0:
selected/all                   = 36.965%
selected/execute               = 78.512%
oracle/all                     = 26.848%
oracle/oracle-execute          = 30.531%
mean score                     = -0.00137
mean teacher utility           = -0.00034
positive utility rate          = 39.300%
harmful when selected          = 54.737%

C1:
selected/execute               = 0.826%
oracle/oracle-execute          = 17.699%
mean score                     = -0.01361
mean teacher utility           = -0.00472

C2:
selected/execute               = 1.653%
oracle/oracle-execute          = 18.142%
mean score                     = -0.02408
mean teacher utility           = -0.00869

C3:
selected/execute               = 19.008%
oracle/oracle-execute          = 33.628%
mean score                     = -0.00879
mean teacher utility           = -0.00448

STOP selected/all              = 52.918%
oracle STOP                    = 12.062%
```

Slot monopoly đổi từ C3 trước refit sang C0 sau refit. Upstream không đổi, nên winner-index monopoly rõ ràng có component do scorer itself, không chỉ ProposalNet slot geometry.

## 10.2 Selector summary

```text
exact oracle accuracy       = 0.2296
stop/execute accuracy       = 0.4825
execute rate                = 0.4708
harmful / executions        = 0.4793
harmful / all decisions     = 0.2257
missed-opportunity STOP     = 0.4630
selected utility            = +0.00317
oracle utility              = +0.03794
regret                      = 0.03477
ScoreNet/teacher Pearson    = +0.1077
STOP precision              = 0.1250
STOP recall                 = 0.5484
STOP F1                     = 0.2036
```

## 10.3 Calibration bins trên held-out

```text
Pearson                    = 0.107671
bias                       = -0.007408
MAE                        = 0.034699
RMSE                       = 0.069721
sign agreement at zero     = 0.634241
```

Selected bins:

```text
teacher -0.117453 → score -0.015791
teacher -0.040377 → score -0.018270
...
teacher +0.003816 → score -0.008875
teacher +0.018565 → score -0.011757
teacher +0.115913 → score -0.011089
```

Positive high-utility candidates vẫn bị predict negative. Với `epsilon_stop=0`, policy naturally STOP nếu mọi candidate score âm.

## 10.4 Before vs after frozen gain-only refit

| Metric | residual_ln source | gain-only stream-refit | interpretation |
|---|---:|---:|---|
| Pearson | -0.0031 | **+0.1077** | learns some utility signal |
| selected utility | -0.00801 | **+0.00317** | materially better |
| oracle utility | +0.03834 | +0.03794 | similar opportunity scale |
| regret | 0.04635 | **0.03477** | ~25% lower |
| harmful / executions | 69.4% | **47.9%** | large improvement |
| exact oracle accuracy | 22.2% | 23.0% | essentially unchanged |
| dominant execute slot | C3 98.1% | C0 78.5% | monopoly changes, not solved |
| STOP rate | 4.0% | **52.9%** | severe over-STOP |
| oracle STOP rate | 10.5% | 12.1% | nowhere near 52.9% |
| missed-opportunity STOP | 3.8% | **46.3%** | severe |

### Interpretation

Frozen gain-only refit **falsifies** hypothesis:

> “ScoreNet/features have zero learnability.”

Nó học được signal đủ để giảm harmful edits và regret.

Nhưng nó **không cứu decision policy**. Numeric regression shrinkage khiến positive utilities không vượt zero; zero STOP boundary tạo massive premature STOP. Exact oracle agreement gần như không cải thiện, và slot monopoly chỉ đổi slot.

Do đó chưa được kết luận:

```text
ScoreNet architecture/features fundamentally insufficient
```

vì objective intervention không còn giống main objective.

---

# 11. Correct-vs-shuffled after stream-refit

Upstream frozen nên raw candidate sensitivity statistics gần như giữ nguyên:

```text
actions cosine(correct, shuffled)      = 0.993173
proposals cosine(correct, shuffled)    = 0.995140
delta_q cosine(correct, shuffled)      = 0.999687
best candidate utility advantage       = +0.000108
candidate utility advantage            = +0.000098
oracle policy utility advantage        = +0.000128
```

Nhưng selector exploitation thay đổi:

```text
selected utility correct               = +0.004960
selected utility shuffled              = +0.001848
selected utility correct-shuffle       = +0.003112
terminal retrieval correct advantage   = +0.002672
```

ScoreNet refit giúp policy tận dụng correct caption tốt hơn một chút, nhưng candidate generator bản thân vẫn cực kỳ insensitive giữa correct và shuffled captions.

**Research implication:** sau khi selector objective được cô lập, một priority upstream rất mạnh là tăng **instruction-conditioned candidate specialization**, không chỉ slot diversity.

---

# 12. Những giả thuyết đã được hỗ trợ / giảm ưu tiên / chưa phân giải

## 12.1 Được hỗ trợ mạnh

### H1 — attention-only proposal mất sibling identity

Được hỗ trợ bởi pre/post attention rank/cosine và residual intervention.

### H2 — missing identity bypass là một causal contributor của functional collapse

Được hỗ trợ vì raw attention vẫn collapse nhưng `q+att` phục hồi final proposals/effects.

### H3 — selector collapse là failure độc lập với candidate representation collapse

Được hỗ trợ trên held-out residual_ln: candidates diverse nhưng ScoreNet C3 monopoly 98.1%, near-zero Pearson, negative selected utility despite positive oracle utility.

### H4 — winner-index monopoly có scorer component trực tiếp

Được hỗ trợ vì chỉ thay ScoreNet qua frozen refit làm monopoly đổi từ C3 sang C0 mà upstream weights/fingerprints không đổi.

### H5 — gain-only Huber không đủ để optimize zero-threshold policy

Được hỗ trợ bởi regression/utility improvements cùng massive over-STOP và calibration bins bị shrink below zero.

### H6 — slot diversity hiện tại không đồng nghĩa semantic/instruction diversity

Được hỗ trợ bởi low sibling cosine nhưng correct-vs-shuffled same-slot cosine ~0.99+ và tiny utility advantages.

## 12.2 Giảm ưu tiên

### DPP numerical catastrophe là primary cause

Không thấy trong distributed audit. Không bị “bác bỏ cho mọi regime”, nhưng không phải priority hiện tại.

### Chỉ cần output LayerNorm sau attention

`attention_ln` gần như identical baseline geometry; không giải quyết collapse.

### Chỉ cần train ScoreNet lâu hơn online

Online residual_ln regret tăng mạnh theo epoch trong khi oracle opportunity tăng. Held-out source checkpoint cũng xác nhận mismatch. Không còn hợp lý giả định đơn giản “thêm epoch là tự hết”.

### Gain-only scorer rescue là đủ

Đã falsified trong current residual_ln stream-refit: có improvement nhưng decision policy vẫn fail nặng.

## 12.3 Chưa phân giải

### Moving-target/co-training vs main ScoreNet objective mismatch

Frozen gain-only test thay cả stationarity và loss, nên chưa isolate.

### ScoreNet feature/architecture capacity

Partial learnability cho thấy không thể gọi architecture incapable, nhưng Pearson 0.108 vẫn thấp. Chỉ redesign sau frozen pair+gain control.

### Candidate competence loss / safe loss / aWTA-like credit

Useful candidates còn ít. Có thể cần sau selector control, nhưng chưa nên bundle vào cùng experiment kế tiếp.

### Grounder/entity collapse

Entity cosine vẫn ~0.98 trong residual variants dù actions/effects diverse. Có thể là semantic bottleneck, nhưng chưa được causal-isolated.

### Long-horizon scorer / timestep features

Chưa nên thêm chỉ vì slot bias. Chỉ relevant nếu one-step selector được cải thiện nhưng terminal H-step oracle gap vẫn còn và metrics chỉ ra horizon mismatch.

---

# 13. Experiment kế tiếp — clean frozen pair+gain ScoreNet control

## 13.1 Research question

> Nếu giữ upstream residual_ln behavior hoàn toàn frozen nhưng train ScoreNet bằng **đúng main-training score objective** (`pair + gain`) thay vì gain-only, selector có học được ranking/STOP tốt hơn không?

## 13.2 Vì sao experiment này cần trước redesign

Current source main training:

```text
lambda_pair = 0.5
lambda_gain = 0.5
```

Frozen stream rescue vừa chạy:

```text
lambda_pair = 0.0
gain-only Huber
```

Do đó current comparison confounds:

```text
online moving targets vs frozen behavior
AND
pair+gain vs gain-only
```

Frozen pair+gain giữ only one major factor changed: upstream stationarity. Đây là experiment đúng để isolate co-training/moving-target effect.

## 13.3 Metrics bắt buộc

Không chỉ nhìn Pearson. Cần ít nhất:

```text
ScoreNet/teacher Pearson
pairwise ranking accuracy / exact oracle agreement
selected utility
oracle utility
regret
harmful execution fraction
STOP precision / recall / F1
missed-opportunity STOP
selected slot occupancy vs oracle occupancy
score sign agreement at zero
calibration bins around zero
by-timestep selector metrics
```

## 13.4 Decision tree

### Case A — pair+gain frozen cải thiện mạnh

Nếu Pearson/ranking/selected utility tăng, regret giảm, monopoly và over-STOP giảm:

```text
=> online co-training / moving-target is a major bottleneck
```

Sau đó nghiên cứu stabilization: delayed scorer updates, target scorer/EMA, alternating training, staged training, replay/cache, slower LR, etc. Không cần redesign feature ngay.

### Case B — ranking cải thiện nhưng STOP vẫn tệ

```text
=> zero-boundary / KEEP-vs-execute calibration is the next bottleneck
```

Khi đó mới thêm explicit decision-boundary objective hoặc STOP classification/calibration control, không tune epsilon để che lỗi.

### Case C — frozen pair+gain vẫn Pearson/ranking rất thấp và monopoly persist

```text
=> stronger evidence that current ScoreNet features/architecture are insufficient
```

Khi đó redesign scorer hợp lý: candidate-set-aware competition, relational sibling features, normalized utility head, context-candidate interactions, possibly timestep/budget only if justified.

### Case D — scorer tốt nhưng oracle utility/useful count vẫn thấp

```text
=> candidate competence / semantic candidate generation becomes primary bottleneck
```

Sau đó ưu tiên direct candidate-quality/safety credit và instruction-conditioned specialization.

---

# 14. Candidate-generation track sau selector control

Ngay cả nếu ScoreNet được fix, current upstream vẫn có hai warning lớn:

## 14.1 Useful candidate scarcity

Held-out source residual_ln:

```text
mean useful candidates       = 0.6300
>=2 useful rows              = 16.59%
positive fraction            = 41.70%
```

Held-out after changed policy trajectory:

```text
mean useful candidates       = 0.5564
>=2 useful rows              = 14.79%
positive fraction            = 37.16%
```

Policy trajectory khác nhau nên hai numbers không phải apples-to-apples state comparison, nhưng absolute scarcity là rõ.

## 14.2 Weak instruction sensitivity

Correct-vs-shuffled same-slot effects gần như identical. Đây là evidence rằng slot identity may dominate text conditioning.

Candidate track sau scorer isolation nên đo/đẩy:

- correct-vs-shuffled caption action/effect separation;
- oracle utility advantage correct vs shuffled;
- concept/edit evidence per slot;
- per-slot semantic specialization, không chỉ occupancy balancing;
- candidate safety/positive utility rate;
- number of useful candidates per parent;
- target-free training proxy nào thực sự correlates với teacher utility trên diagnostics.

Possible later interventions từ prior research plan:

```text
candidate outcome/safety credit
edit-evidence competition
anti-repeat / usage balancing only if semantics remain distinct
ModeSeq / structured edit slots if arbitrary K hypotheses remain redundant
```

Không bundle nhiều interventions cùng lúc.

---

# 15. Reproducibility caveats

1. One-epoch architecture ablations không đủ để rank retrieval quality. G1 chỉ dùng để causal-diagnose geometry.
2. Memory-efficient attention backward từng báo nondeterministic warning. Close benchmark comparisons cần multiple seeds / deterministic configuration khi chuyển từ diagnosis sang result claim.
3. `_ln` modes có learned output LayerNorm params; đây là architecture ablation thực sự, không parameter-identical mode switch.
4. `stream_pre_update_calibration` là telemetry của predictions trước mỗi online scorer update trên TRAIN stream; held-out VAL-160 report mới là generalization evidence.
5. VAL-160 là diagnostic cohort, không phải official full FashionIQ validation metric. Không dùng nó để claim benchmark performance.
6. Correct-vs-shuffled metrics đo dependence vào caption; chúng không chứng minh model hoàn toàn ignore text, nhưng current effect magnitudes rất nhỏ và cần được xem là warning nghiêm trọng.
7. Oracle utility dùng target-derived teacher only cho diagnostics/training supervision, không phải inference input.
8. DPP audit FP32 reference là DPP-only backward từ live `delta_q`; không phải full-model FP32 reference.
9. Partial raw cache 270 shards / 55 GB không có manifest nên không phải valid fixed-cache experiment artifact; không archive shards vào Git.
10. Source checkpoint residual_ln là epoch-1 fresh run. Không so trực tiếp với mature historical 20-epoch models như thể cùng training maturity.

---

# 16. Paper-safe conclusions hiện tại

Có thể viết:

> The initial attention-only proposal path exhibits severe sibling contraction: candidate queries retain substantial diversity immediately before cross-attention, but raw attention outputs and downstream effects become nearly rank-one. A controlled residual intervention restores candidate identity and downstream functional diversity even though the raw attention output itself remains collapsed, supporting a causal role for the missing identity path in this representation-collapse failure mode.

Có thể viết:

> Restoring sibling diversity does not by itself solve the sequential CIR policy. On a held-out paired diagnostic cohort, the residual-LN variant produces strongly differentiated candidate effects, yet the selector collapses to one slot and exhibits near-zero score–teacher correlation, while a target-derived oracle can achieve substantially higher utility from the same candidate sets.

Có thể viết:

> A frozen-behavior, ScoreNet-only gain-regression intervention partially improves utility prediction and reduces harmful executions, but it does not recover the oracle policy. Instead, scores shrink toward negative values around the zero STOP boundary, producing severe premature stopping and a different slot monopoly. This indicates partial scorer learnability but a mismatch between numeric gain regression and the actual decision policy.

Có thể viết:

> Functional sibling diversity is not yet equivalent to instruction-grounded diversity: correct-vs-shuffled-caption comparisons remain highly similar within each slot, suggesting that a substantial component of the recovered diversity may be slot-identity-driven rather than strongly conditioned on the edit instruction.

Không được viết:

- `residual_ln` đã tăng benchmark Recall;
- ScoreNet architecture chắc chắn không đủ;
- DPP chắc chắn vô hại;
- candidate generation đã được giải;
- C3/C0 là inherent bad slots;
- slot histogram cân bằng = specialization;
- gain-only refit chứng minh moving target là nguyên nhân;
- shuffled-caption evidence chứng minh text hoàn toàn bị ignore.

---

# 17. Artifact map cần archive vào repository

Recommended destination:

```text
doc/diagnostics/2026-09-21-functional-collapse-audit/
```

Nên archive **text / JSON / YAML / JSONL / log artifacts**, không archive `.pt` checkpoints hoặc raw scorer cache.

Recommended structure:

```text
doc/diagnostics/2026-09-21-functional-collapse-audit/
├── CIR_DIAGNOSTIC_CHECKPOINT_2026-09-21_POST_G0_G1.md
├── attention_epoch1/
│   ├── metrics.jsonl
│   ├── resolved_config.yaml
│   └── run_metadata.json                 # nếu tồn tại
├── attention_ln_epoch1/
│   ├── metrics.jsonl
│   ├── resolved_config.yaml
│   └── run_metadata.json                 # nếu tồn tại
├── residual_epoch1/
│   ├── metrics.jsonl
│   ├── resolved_config.yaml
│   └── run_metadata.json                 # nếu tồn tại
├── residual_ln_epoch1/
│   ├── metrics.jsonl
│   ├── resolved_config.yaml
│   └── run_metadata.json                 # nếu tồn tại
├── attention_val160/
│   ├── candidate_selector_diagnostic.json
│   └── candidate_selector_diagnostic.md
├── residual_ln_val160/
│   ├── candidate_selector_diagnostic.json
│   └── candidate_selector_diagnostic.md
├── residual_ln_score_stream/
│   ├── stream_refit_report.json
│   ├── metrics.jsonl
│   └── resolved_config.yaml
└── residual_ln_stream_val160/
    ├── candidate_selector_diagnostic.json
    └── candidate_selector_diagnostic.md
```

Nếu DPP analyzer reports hiện còn trên disk, thêm thư mục:

```text
dpp_gradient_audit/
```

và copy report JSON/MD/text summary, nhưng không đưa checkpoint hoặc tensor dumps lớn vào Git.

---

# 18. Current handoff state

Nếu cuộc trò chuyện mới chỉ đọc file này, thứ tự tiếp tục nên là:

```text
1. Verify branch + local diff, nhất là src/refit_score_net.py.
2. Archive/push current diagnostics và checkpoint markdown.
3. Không sửa ProposalNet thêm.
4. Implement controlled frozen ScoreNet pair+gain stream-refit mode.
5. Chạy same VAL-160 manifest.
6. So sánh source residual_ln vs gain-only frozen vs pair+gain frozen.
7. Quyết định moving-target vs objective-boundary vs architecture.
8. Sau đó quay lại candidate competence / caption sensitivity nếu scorer được cô lập.
```

**State of evidence at handoff:** representation collapse mechanism đã được localize và can thiệp thành công về geometry; selector collapse đã được xác nhận trên held-out; gain-only frozen scorer rescue chỉ partially successful và tạo over-STOP; candidate instruction sensitivity vẫn rất yếu; DPP không phải primary numerical suspect trong measured trajectory. Đây là baseline chẩn đoán cần giữ cố định để mọi experiment tiếp theo có causal meaning.