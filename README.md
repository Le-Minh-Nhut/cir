# TAPER-MAG V4 — Branch Diagnostic / Research Handoff

**Branch:** `exp/taper-mag-v4-fgclip2-base-text-ft`
**Repository:** `Le-Minh-Nhut/cir`
**Snapshot HEAD:** `ecedff8c583dcee3dde73fe1605558d6d69dc18a`
**Canonical reference:** `CIR_TAPER_CANONICAL_END_TO_END_METHOD_SPEC_V4_2026-08-28.md`
**Date:** 2026-08-29

---

# 0. Executive Verdict

Branch này hiện tại:

```text
IMPLEMENTATION COMPLETE
RUNTIME PASS
ACTOR USEFUL
ACTOR HEALTH GATE NOT PASSED
CANDIDATE SPACE LOW-RANK / NEAR-CLONE
DO NOT CONTINUE CANONICAL CRITIC TRAINING
```

Điểm quan trọng nhất:

> Actor không thất bại trong việc học edit.

Actor thực sự học được transformation có ích cho CIR.

Nhưng:

> Bốn candidate/action được sinh ra chưa trở thành bốn lựa chọn chức năng đủ khác nhau.

Kết quả teacher-shadow trên 256 samples:

```text
oracle_action_realized_gain   = 0.94985
random_action_realized_gain   = 0.89484
uniform_mean_realized_gain    = 0.89159
learned_policy_realized_gain  = 0.89299

oracle_vs_random_gap          = 0.05502
oracle_vs_uniform_gap         = 0.05827

candidate_outcome_variance    = 2.923e-06
mean_effective_rank           = 1.16866
oracle_positive_gain_rate     = 0.80469
```

Vì vậy diagnosis chính là:

```text
NOT:

actor cannot learn useful edits

BUT:

actor learns a strong shared edit family
and produces several weak variants of that family
instead of genuinely distinct functional actions
```

Health verdict:

```text
Numerical health                 PASS
Target firewall                  PASS
FG-CLIP2 text finetuning         PASS
Causal intervention runtime      PASS
FashionIQ VAL protocol           PASS
Actor usefulness                 PASS

Candidate functional diversity   FAIL / HIGH RISK
Candidate effective rank         FAIL / HIGH RISK

Critic quality                   NOT YET APPLICABLE
Dynamic-vs-frozen                NOT APPLICABLE AT T=1

actor_warmup_passed              DO NOT APPROVE
```

---

# 1. Research Idea Tested By This Branch

TAPER-MAG V4 thử nhìn CIR dưới dạng một quá trình editing động:

```text
reference image
      +
modification text
      ↓
shared query extraction
      ↓
edit-conditioned grounding
      ↓
K candidate edit operators
      ↓
execute candidates on visual state
      ↓
candidate future states
      ↓
predict marginal utility
      ↓
choose action / STOP
      ↓
update state
      ↓
repeat
      ↓
final retrieval query
```

Thay vì:

```text
image + text
    ↓
one-shot fusion
    ↓
query
```

mục tiêu của TAPER là:

```text
image state
   ↓
make one useful edit
   ↓
new image state
   ↓
make another useful edit
   ↓
...
   ↓
retrieval
```

Branch này đặc biệt tránh semantic ownership cứng của các slot cũ.

Không ép:

```text
slot 1 = color
slot 2 = sleeve
slot 3 = texture
slot 4 = shape
```

Mà cho cả K query đọc toàn bộ text và kỳ vọng chúng tự sinh ra các functional edit alternatives khác nhau.

---

# 2. Final Architecture Snapshot

Core settings:

```text
d_model       = 256
num_queries   = 4

Tmin          = 1
Tmax          = 4

STOP anchor   = 0
repeat        = allowed
```

Backbone:

```text
FG-CLIP2-Base
```

Vision:

```text
fully frozen
```

Text:

```text
last 4 transformer blocks trainable
final norm trainable
projection frozen
```

Cụ thể:

```text
trainable:
    text_model.encoder.layers.8
    text_model.encoder.layers.9
    text_model.encoder.layers.10
    text_model.encoder.layers.11
```

Parameter counts:

```text
total FG-CLIP2 params       383,803,394

trainable FG-CLIP2 params    28,353,024
trainable text params        28,353,024
trainable vision params               0

trainable TAPER params        5,864,470
```

---

# 3. Training Objective

Phase-1 objective:

```math
L = L_ret_terminal + L_util
```

Trong đó:

```text
L_ret_terminal
    = terminal CIR retrieval loss

L_util
    = utility prediction loss
```

Retrieval:

```text
bidirectional multi-positive InfoNCE
```

Utility teacher:

```text
detached
target-aware
common-negative
marginal ΔInfoNCE
```

STOP:

```text
fixed utility = 0
```

Không dùng mặc định:

```text
orthogonality loss
slot diversity loss
semantic slot loss
coverage loss
explicit anti-collapse loss
```

---

# 4. Target Firewall

Một invariant quan trọng của branch:

```text
target image
```

được phép dùng trong:

```text
training teacher
terminal retrieval loss
scientific diagnostics
```

nhưng tuyệt đối không được đi vào:

```text
actor
query generator
operator generator
executor
recurrent state
policy history
inference
```

Teacher-shadow audit xác nhận:

```text
policy_forward_target_argument_absent = true

inference_without_supervision_succeeded = true

target_entered_policy_or_history = false

target_shuffle_changed_teacher = true
```

Kết luận:

```text
TARGET FIREWALL = PASS
```

Không có evidence rằng actor và teacher “thông đồng” bằng cách leak target.

---

# 5. Important Runtime / Evaluation Fixes

Trước khi có kết quả scientific cuối cùng, branch đã phải sửa một số lỗi quan trọng.

## 5.1 FashionIQ VAL protocol

VAL gallery hiện dùng:

```text
ordered unique(
    reference IDs
    ∪
    target IDs
)
```

Sau đó với từng query:

```text
mask own reference image score
```

chứ không globally loại mọi reference image khỏi gallery.

Current reports:

```text
validation_protocol = fashioniq_val
reference_exclusion = true
```

---

## 5.2 Functional intervention phải đi qua real executor

Không dùng surrogate kiểu:

```text
q + delta
q + 2 delta
```

để kết luận causal behavior.

Current intervention path:

```text
operator
   ↓
executor
   ↓
local visual state
   ↓
readout
   ↓
retrieval query
```

Report xác nhận:

```text
execution_contract =
operator_to_executor_to_state_to_readout

query_delta_arithmetic_used = false
```

---

## 5.3 TextDriftMonitor bug

Ban đầu drift monitor forward text encoder hai lần:

```text
encode_text_tokens()
+
get_text_features()
```

gây CUDA assertion.

Đã sửa thành:

```text
single text-transformer forward
          ↓
hidden states
     ├─ token drift
     └─ short-walk pooled representation
```

Không còn redundant text forward.

---

## 5.4 BF16 audit mismatch

FG-CLIP2 chạy BF16 trong khi TAPER weights giữ FP32.

Audit ban đầu thiếu autocast nên gặp:

```text
BFloat16 × Float
```

Current runtime đã centralize:

```text
CUDA + bf16
→ torch.autocast(cuda, bfloat16)
```

và dùng cho:

```text
training
validation
functional audit
causal interventions
teacher shadow
dynamic/frozen audit
profiler
```

Scientific epoch 1–8 đã chạy hết mà không còn dtype crash.

---

# 6. Actor Warmup Training Result

Canonical actor warmup:

```text
epoch 1 → 8
T = 1
```

Kết quả:

| Epoch | Train Loss | MeanR |  R@10 |  R@50 | Oracle Best Gain | Positive Rate |
| ----: | ---------: | ----: | ----: | ----: | ---------------: | ------------: |
|     1 |     2.9238 | 20.13 | 13.61 | 26.66 |           0.0015 |         0.632 |
|     2 |     2.9212 | 20.38 | 13.84 | 26.91 |          -0.0119 |         0.549 |
|     3 |     2.8947 | 20.97 | 14.29 | 27.64 |          -0.0865 |         0.255 |
|     4 |     2.8110 | 21.82 | 15.04 | 28.61 |          -0.1544 |         0.288 |
|     5 |     2.6789 | 23.76 | 16.51 | 31.02 |          -0.1413 |         0.349 |
|     6 |     2.5309 | 26.23 | 18.22 | 34.24 |           0.2496 |         0.636 |
|     7 |     2.4271 | 27.63 | 19.10 | 36.15 |           0.5366 |         0.755 |
|     8 |     2.3291 | 30.16 | 21.03 | 39.30 |           0.7455 |         0.813 |

Training trajectory:

```text
loss:
2.9238
→
2.3291
```

Retrieval:

```text
MeanR:
20.13
→
30.16
```

```text
R@10:
13.61
→
21.03
```

```text
R@50:
26.66
→
39.30
```

Vì vậy không thể nói:

```text
actor failed to learn
```

Ngược lại:

```text
ACTOR LEARNED USEFUL EDIT BEHAVIOR
```

Epoch 8 đạt:

```text
MeanR = 30.16
R@10  = 21.03
R@50  = 39.30
```

trước khi health gate chủ động dừng run.

---

# 7. Text Encoder Health

Frozen blocks:

```text
block 0–7 grad = 0
```

Trainable blocks:

```text
block 8–11 grad > 0
```

Epoch 8:

```text
text_parameter_relative_change
≈ 1.52e-4

text_pooled_cosine
≈ 0.999984

text_token_cosine
≈ 0.999982
```

Interpretation:

```text
text encoder đang update
nhưng pretrained semantic space không bị phá mạnh
```

Không có evidence text encoder là nguyên nhân chính của failure.

---

# 8. Candidate Diversity During Training

Candidate outcome variance:

```text
epoch 1
4.81e-09

epoch 2
5.41e-08

epoch 3
2.65e-07

epoch 4
5.77e-07

epoch 5
2.05e-06

epoch 6
4.57e-06

epoch 7
6.29e-06

epoch 8
7.93e-06
```

Điều này cho thấy:

```text
candidates không hoàn toàn identical
```

Nhưng epoch 8 vẫn có:

```text
query_query_cosine_offdiag
≈ 0.876

operator_operator_cosine_offdiag
≈ 0.938

response_effective_rank
≈ 1.31
```

Teacher-shadow audit lớn hơn sau đó còn cho:

```text
mean_effective_rank
= 1.1687
```

Vì có K=4 candidates mà effective rank chỉ khoảng:

```text
1.17
```

nên representation thực tế gần:

```text
candidate 1 ─┐
candidate 2 ─┼── shared dominant function
candidate 3 ─┤
candidate 4 ─┘
```

hơn là:

```text
candidate 1 → function A
candidate 2 → function B
candidate 3 → function C
candidate 4 → function D
```

Đây là:

```text
FUNCTIONAL LOW-RANK COLLAPSE
```

không nhất thiết là exact vector cloning.

---

# 9. Teacher-Shadow Actor Audit

256 samples.

Actor usefulness:

```text
oracle_best_gain
= 0.94985

oracle_positive_gain_rate
= 0.80469
```

Điều này rất quan trọng:

```text
~80.5% samples
có ít nhất một candidate hữu ích
```

Actor thực sự tạo được edit tốt.

Nhưng candidate discrimination rất yếu:

```text
oracle_action_realized_gain
= 0.94985

random_action_realized_gain
= 0.89484

uniform_mean_realized_gain
= 0.89159

learned_policy_realized_gain
= 0.89299
```

Khoảng cách:

```text
oracle - random
≈ 0.055

oracle - uniform
≈ 0.058
```

Nghĩa là:

```text
chọn đúng action
```

chỉ tốt hơn:

```text
chọn ngẫu nhiên một action
```

một lượng khá nhỏ.

Đây là evidence mạnh nhất rằng:

```text
candidate identity không đủ quan trọng
```

---

# 10. Candidate Geometry Diagnosis

Có thể mô hình hóa failure như:

```text
reference + text
       ↓
shared edit direction d
       ↓
q1 = d + ε1
q2 = d + ε2
q3 = d + ε3
q4 = d + ε4
```

Trong đó:

```text
ε1, ε2, ε3, ε4 ≠ 0
```

nhưng quá nhỏ về mặt functional outcome.

Do đó:

```text
q1 ≠ q2 ≠ q3 ≠ q4
```

về numerical tensor,

nhưng:

```text
Effect(q1)
≈
Effect(q2)
≈
Effect(q3)
≈
Effect(q4)
```

đây mới là collapse thực sự cần quan tâm.

---

# 11. Functional Retrieval Controls

Functional retrieval audit hiện dùng:

```text
32 validation samples
```

nên chỉ nên dùng như evidence bổ sung, không phải benchmark cuối.

Aggregate:

```text
reference_only
MeanR = 23.48

full_dynamic
MeanR = 28.18

clone_all_best
MeanR = 33.03

clone_all_mean
MeanR = 29.85

operator_mean
MeanR = 29.85

repeat_best
MeanR = 26.67

mean_repeat
MeanR = 25.15

operator_zero
MeanR = 23.48
```

---

# 12. Zero Operator Sanity Check

Một kết quả rất tốt:

```text
reference_only
MeanR = 23.48485

operator_zero
MeanR = 23.48485
```

Điều này chứng minh audit causal intervention đang hoạt động hợp lý:

```text
zero edit
≈
no edit
```

---

# 13. Full Actor Is Useful

```text
reference_only
23.48

full_dynamic
28.18
```

Difference:

```text
≈ +4.70 MeanR
```

Do đó:

```text
actor edit computation
```

thực sự cải thiện retrieval.

Đây là reason quan trọng để không gọi branch này là “actor failure”.

---

# 14. Clone-All-Best Control

```text
full_dynamic
MeanR = 28.18

clone_all_best
MeanR = 33.03
```

Difference:

```text
+4.85 MeanR
```

Nhưng:

```text
clone_all_best
```

dùng:

```text
detached teacher argmax
```

để chọn best candidate.

Vì critic chưa được train hoàn chỉnh:

```text
clone_all_best > learned
```

không tự nó chứng minh collapse.

Nhưng kết hợp với:

```text
effective rank ≈ 1.17
oracle-random gap ≈ 0.055
operator cosine ≈ 0.94
```

thì nó củng cố diagnosis rằng candidate bank chưa đủ differentiated.

---

# 15. Mean Operator Is Too Competitive

```text
operator_mean
MeanR = 29.85

full_dynamic
MeanR = 28.18
```

Đây là warning lớn.

Nếu bốn candidate thực sự đại diện bốn functional choices khác nhau, thì:

```text
mean(q1,q2,q3,q4)
```

không nên dễ dàng giữ hoặc vượt phần lớn performance.

Việc mean operator mạnh cho thấy:

```text
shared component
>>
candidate-specific component
```

---

# 16. Repeat Is Not The Main Shortcut

```text
repeat_best
MeanR = 26.67

mean_repeat
MeanR = 25.15

full_dynamic
MeanR = 28.18
```

Do đó failure không đơn giản là:

```text
one good operator
+
repeat it multiple times
```

Repeat có functional effect, nhưng không tự recover full retrieval behavior.

---

# 17. Support Saturation

Support metrics tăng mạnh trong warmup.

Gần cuối:

```text
support_mass
≈ 0.96

support_saturation
≈ 0.98
```

Điều này gợi ý operator ngày càng tác động trên phần lớn visual support.

Conceptually:

```text
desired:
candidate A focuses one relevant transformation
candidate B focuses another one
```

nhưng observed behavior có thể gần:

```text
all candidates broadly touch similar visual state
```

Support saturation một mình chưa chứng minh collapse.

Nhưng khi kết hợp:

```text
effective rank ≈ 1.17
operator cosine ≈ 0.94
small oracle-random gap
```

thì nó trở thành evidence supporting the same diagnosis.

---

# 18. Critic Diagnosis

Teacher-shadow:

```text
top1_agreement
≈ 0.2695

pairwise_accuracy
≈ 0.533

mean_regret
≈ 0.0569
```

Calibration cũng cho thấy critic prediction chưa match teacher.

Nhưng:

```text
THIS IS NOT THE REASON
TO FAIL ACTOR WARMUP
```

Vì critic chưa tới dedicated training phase.

Do đó:

```text
critic quality
=
NOT YET APPLICABLE
```

---

# 19. Dynamic-vs-Frozen Is Not Applicable Yet

Current horizon:

```text
T = 1
```

Report ghi đúng:

```text
status = not_applicable_horizon_1
```

Vì vậy không được dùng:

```text
dynamic == frozen
```

để kết luận multi-step mechanism fail.

Multi-step chưa thực sự được test.

---

# 20. Why The Health Gate Should Fail

Actor gate cần trả lời:

```text
Actor có tạo được một action space đủ tốt
để critic học cách lựa chọn hay không?
```

Current answer:

```text
actor usefulness:
YES

action-space differentiation:
NO / HIGH RISK
```

Nếu continue critic training ngay:

```text
critic
↓
học phân biệt
q1 q2 q3 q4
```

trong khi:

```text
q1 ≈ q2 ≈ q3 ≈ q4
```

thì về sau nếu critic fail sẽ không biết:

```text
critic yếu
```

hay:

```text
action space vốn đã không có gì để phân biệt
```

Health gate tồn tại chính để tránh confound này.

---

# 21. Why This Failure Is Different From Earlier Slot Collapse

Các TAPER branch trước từng có:

```text
slot monopoly
dead slots
hard ownership collapse
clone slots
mean-slot × K recovery
```

Branch này bỏ semantic competitive ownership.

Nhưng vẫn collapse.

Điều này cho thấy một lesson sâu hơn:

```text
removing hard slot ownership
does NOT automatically solve
functional redundancy
```

Hay:

```text
K latent vectors
≠
K functional actions
```

Một model hoàn toàn có thể tạo:

```text
4 different tensors
```

nhưng cả 4 cùng thực hiện:

```text
almost the same useful edit
```

---

# 22. What This Branch Successfully Proved

## 22.1 Shared actor can learn useful CIR transformations

Yes.

## 22.2 Frozen vision + partially tuned FG-CLIP2 text works technically

Yes.

## 22.3 Local executor can causally improve retrieval

Yes.

## 22.4 Zero intervention behaves approximately like no intervention

Yes.

## 22.5 Target-free inference firewall works

Yes.

## 22.6 Multiple candidate vectors automatically become different actions

No.

Đây là hypothesis quan trọng đã bị phản chứng.

---

# 23. What Must NOT Be Concluded

Không được kết luận:

```text
multi-step CIR không hoạt động
```

vì chưa bao giờ train qua T>1.

Không được kết luận:

```text
critic approach sai
```

vì critic chưa được fully trained.

Không được kết luận:

```text
shared query sai
```

vì shared query đã học useful edits.

Không được kết luận:

```text
repeat vô dụng
```

vì mature multi-step policy chưa được test.

Không được kết luận:

```text
MeanR 30 là ceiling
```

vì training bị intentional stop ở epoch 8.

Conclusion đúng chỉ là:

> Actor parameterization hiện tại chưa tạo ra một action space đủ differentiated trước khi critic training bắt đầu.

---

# 24. Health Gate Decision

Giữ:

```yaml
approved_health_gates: []
```

Không thêm:

```yaml
approved_health_gates:
  - actor_warmup_passed
```

Status:

```text
actor_warmup_passed = REJECTED
```

cho scientific continuation của architecture hiện tại.

---

# 25. Diagnostics Worth Keeping For Any Future Network

Dù bỏ architecture này, nên giữ lại diagnostic framework.

## Target firewall

```text
target never enters inference actor
```

## Oracle vs random

```text
oracle gain
vs
random action gain
```

Nếu gap rất nhỏ:

```text
candidate identity probably does not matter
```

## Oracle vs uniform mean

Nếu:

```text
uniform ≈ oracle
```

candidate set có thể đang encode cùng một function.

## Effective rank

Không chỉ đo cosine.

Đo:

```text
functional response rank
```

## Clone-all-best

Để test:

```text
one functional primitive
```

có đang dominate không.

## Clone-all-mean

Để test shared component.

## Operator mean

Để kiểm tra:

```text
mean(candidate set)
```

có giữ gần hết useful effect không.

## Zero operator

Phải gần:

```text
reference-only
```

## Repeat control

Test repeated action staleness.

## Dynamic-vs-frozen

Chỉ dùng khi:

```text
T > 1
```

## Exact FashionIQ retrieval protocol

Luôn giữ benchmark evaluation nhất quán.

---

# 26. Questions A Future Network Must Answer

Một architecture mới có K candidate actions nên phải chứng minh được:

```text
1. K candidate có tạo ra K future states khác nhau thật không?

2. Oracle selection có tốt hơn random rõ rệt không?

3. Candidate effective rank có thực sự > 1 đáng kể không?

4. Mean candidate có làm mất useful information không?

5. Clone one candidate across all slots có thay đổi behavior mạnh không?

6. Zero candidate có trở về reference-only không?

7. Candidate differences có sống sót sau real executor không?

8. State thay đổi có làm action ordering thay đổi ở bước sau không?

9. Repeat một action có trở nên stale sau khi action đó đã hoàn thành không?

10. Target-free policy có recover được phần đáng kể oracle gain không?
```

Nếu chưa trả lời được những câu này thì:

```text
multiple-action claim
```

chưa thật sự được chứng minh.

---

# 27. Important Artifacts To Preserve

Run directory:

```text
runs/taper-mag-v4-fgclip2-base-last4/
```

Nên giữ:

```text
last.ckpt

teacher_shadow_report.json
functional_health.json
functional_retrieval.json

firewall_report.json

metrics_val.jsonl
policy_trace_sampled.jsonl

config_resolved.yaml
run_manifest.json
backbone_manifest.json
```

Đặc biệt:

```text
epoch-8 last.ckpt
```

là một negative-result checkpoint rất hữu ích.

Sau này architecture mới có thể compare candidate geometry trực tiếp với checkpoint này.

---

# 28. Relevant Final Commits

```text
cf640a9
Complete TAPER-MAG Phase-1 audit runtime

0a87c5e
Correct FashionIQ validation and causal audits

5205c43
Fix FashionIQ VAL reference exclusion

41085b7
Fix redundant text drift forward

a1aee1c
Fix BF16 audit runtime consistency

ecedff8
Add TAPER training progress display
```

---

# 29. Reproduce The Actor Warmup

```bash
cd ~/data/cir
conda activate fpclip2-cir

PYTHONPATH=src python src/train.py \
  --config conf/taper_mag_v4_base.yaml
```

Expected:

```text
epoch 1 → 8
```

sau đó stop:

```text
actor_warmup health gate
```

---

# 30. Reproduce Teacher-Shadow

```bash
PYTHONPATH=src python src/train.py \
  --config conf/taper_mag_v4_base.yaml \
  --resume runs/taper-mag-v4-fgclip2-base-last4/last.ckpt \
  --teacher-shadow-audit \
  --audit-samples 256
```

---

# 31. Final Research Takeaway

Kết quả quan trọng nhất của branch:

```text
A USEFUL ACTOR IS NOT ENOUGH.
```

Model có thể:

```text
improve retrieval
```

và đồng thời:

```text
fail to learn a meaningful action space
```

Benchmark curve nhìn khá tốt:

```text
MeanR:

20.13
→
30.16
```

nhưng diagnostic cho thấy:

```text
effective rank ≈ 1.17

random
≈
uniform
≈
learned

oracle chỉ tốt hơn modestly

mean operator vẫn rất mạnh
```

Do đó câu hỏi quan trọng cho network tiếp theo không nên là:

```text
How do I generate K vectors?
```

Mà phải là:

```text
Why should these K candidates
represent genuinely different
future interventions?
```

Và sự khác biệt đó nên xuất hiện từ:

```text
architecture
interaction
state transition structure
information structure
```

chứ không chỉ nhờ:

```text
orthogonality loss
cosine diversity
arbitrary semantic slot labels
```

---

# 32. Final Branch Status

```text
TAPER-MAG V4

IMPLEMENTATION:
    COMPLETE

RUNTIME:
    PASS

BACKBONE:
    PASS

TEXT FINETUNING:
    PASS

FASHIONIQ PROTOCOL:
    PASS

TARGET FIREWALL:
    PASS

CAUSAL EXECUTOR:
    PASS

ACTOR USEFULNESS:
    PASS

CANDIDATE FUNCTIONAL DIVERSITY:
    FAIL / HIGH RISK

EFFECTIVE RANK:
    FAIL / HIGH RISK

CRITIC:
    NOT YET JUDGED

MULTI-STEP:
    NOT YET JUDGED

ACTOR HEALTH GATE:
    NOT PASSED

RECOMMENDATION:
    ARCHIVE THIS BRANCH
    KEEP IT AS A NEGATIVE-RESULT / DIAGNOSTIC BASELINE
    DESIGN THE NEXT NETWORK FROM FIRST PRINCIPLES
```

Branch này không phải codebase vô dụng.

Giá trị lớn nhất của nó là đã isolate được vấn đề tiếp theo rất rõ:

> **Trước khi học cách chọn action, model phải có lý do cấu trúc đủ mạnh để sinh ra những action thật sự khác nhau về chức năng.**
