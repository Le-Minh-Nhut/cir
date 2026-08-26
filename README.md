# A3.2 Contextual-Key / Local-Value — Failure Diagnosis README

**Date:** 2026-08-26  
**Branch:** `exp/e2e-a3.2-contextual-key-local-value`  
**Parent reference:** `exp/e2e-a3.1-qasa-slot-filter-eval-winner`

---

## 1. Mục đích

Tài liệu này là checkpoint chẩn đoán cho nhánh A3.2 sau khi thử hai chế độ:

- `soft_shared`
- `hard_st_exclusive`

Hypothesis ban đầu của A3.2:

> Edit Slots collapse vì VALUE mang quá nhiều thông tin global/contextual và vì cùng một token có thể bị copy mềm sang nhiều slots.

A3.2 cô lập information path như sau:

- **KEY / ownership / QASA:** vẫn dùng contextual Q-Former text states.
- **VALUE:** dùng raw CSMCIR word embeddings `teacher_text_states`.
- **`slot_effects = q_full - q_minus`:** vẫn được tính cho diagnostic nhưng không còn feed vào Edit Slot latent.
- Assignment có toggle giữa `soft_shared` và `hard_st_exclusive`.

Mục tiêu của tài liệu này là ghi lại chính xác cái gì đã được chứng minh, cái gì bị falsify, và failure mode còn lại là gì.

---

## 2. Kiến trúc đang test

### 2.1 Contextual KEY

Ownership vẫn được tính từ contextual text states:

```python
ownership_logits, soft_slot_masks = self._competitive_ownership(
    text_states,
    slot_valid,
)
```

`text_states` là Q-Former text states đã contextualized và reference-conditioned.

Ý nghĩa:

```text
whole caption + reference context
             |
             v
       contextual KEY
             |
             v
      token-slot scores
```

KEY được phép global vì nhiệm vụ của nó là quyết định token nên thuộc slot nào.

### 2.2 Raw / local VALUE

VALUE không còn lấy từ contextual Q-Former states. Nó lấy từ:

```text
teacher_text_states
```

được cache từ raw word embedding lookup trước Q-Former contextualization.

Sau assignment:

```python
slot_semantics = pool(
    teacher_text_states,
    value_slot_masks,
)

raw_edit_slots = self.slot_mlp(slot_semantics)
```

Không concatenate `slot_effects`.

### 2.3 `soft_shared`

```text
contextual KEY
      |
      v
soft ownership
      |
      v
raw token VALUE
```

Một token có thể contribute vào nhiều slots.

Ví dụ:

```text
red:
S0 = 0.60
S1 = 0.20
S2 = 0.10
S3 = 0.10
```

### 2.4 `hard_st_exclusive`

Forward:

```text
contextual KEY
      |
      v
soft scores
      |
      v
argmax/token
      |
      v
one hard owner
      |
      v
raw token VALUE
```

Mỗi valid token chỉ contribute vào đúng một VALUE slot.

Backward dùng Straight-Through:

```python
value_slot_masks = (
    hard_slot_masks
    + soft_slot_masks
    - soft_slot_masks.detach()
)
```

Forward hard, backward vẫn có gradient qua soft ownership.

---

## 3. Baseline A3.1 trước A3.2

P0 audit reference checkpoint A3.1:

| Metric | A3.1 |
|---|---:|
| Hard partition mean K | 1.175 |
| Dominant hard token share | 0.976 |
| Gradient error-mode rank | 2.837 |
| Functional Phi effective rank | 1.018 |
| Functionally useful slots | 3.903 |
| QASA functional precision | 0.974 |
| QASA functional recall | 0.497 |
| Median best SINGLE/FULL | 0.790 |
| Median best REPEAT/FULL | 1.037 |
| Median MEANxK/FULL | 1.000 |
| Mean K95 | 2.514 |
| Mean K99 | 2.959 |
| qasa_full Mean Recall | 58.58 |
| all_slots_full Mean Recall | 58.88 |
| reference_only Mean Recall | 8.10 |

Interpretation baseline:

```text
gradient error-mode rank ≈ 2.84
```

Task có nhiều error directions, nhưng:

```text
Phi rank ≈ 1.02
```

learned slot effects gần như chỉ nằm trên một functional direction.

Ngoài ra:

```text
REPEAT/FULL ≈ 1.04
MEANxK/FULL ≈ 1.00
```

cho thấy slot identity/content không thực sự cần thiết; một representation gần-global cộng với nhiều executor tickets có thể tái tạo gần toàn bộ gain.

---

## 4. Training results A3.2

### 4.1 HARD — `hard_st_exclusive`

Checkpoint:

```text
outputs/2026-08-26/20-48-08/best.pt
```

10-epoch training:

```text
best Mean Recall ≈ 43.27
```

Cuối training:

```text
value_k          ≈ 2.06
value_dominant   ≈ 0.792
value_empty      ≈ 0.486
```

Performance giảm mạnh so với A3.1 reference checkpoint.

### 4.2 SOFT — `soft_shared`

Checkpoint:

```text
outputs/2026-08-26/21-22-35/best.pt
```

10-epoch training:

```text
best Mean Recall ≈ 42.68
```

Cuối training hard-argmax diagnostic:

```text
value_k          ≈ 1.03
value_dominant   ≈ 0.994
value_empty      ≈ 0.743
```

Soft mode học tới gần monopoly tuyệt đối theo argmax ownership.

---

## 5. Local-value failure diagnosis — HARD

Checkpoint:

```text
/home/heheboiz/data/cir/outputs/2026-08-26/20-48-08/best.pt
```

Full validation: 6016 queries.

### 5.1 Retrieval interventions

| Variant | Mean Recall |
|---|---:|
| deployed_qasa | **43.25** |
| raw_hard_qasa | 43.25 |
| raw_hard_qasa_nonempty | 43.26 |
| raw_hard_all_nonempty | 43.35 |
| raw_soft_qasa | 42.76 |
| raw_soft_all | 36.54 |
| contextual_hard_qasa_nonempty | 21.88 |
| contextual_soft_qasa | 23.02 |
| teacher_full | **64.94** |
| reference_only | 13.81 |
| empty_selected_only | 13.78 |

### 5.2 Routing / ownership

```text
hard_value_effective_k         = 1.9456
hard_value_dominant_share      = 0.8267
hard_value_empty_slot_fraction = 0.5136
```

Hard privacy không còn cho token bị copy sang nhiều VALUE slots, nhưng model thích nghi bằng cách tạo một **giant slot**:

- chỉ khoảng 1.95 / 4 slots có token,
- slot lớn nhất ăn ~82.7% token,
- ~51.4% slot positions empty.

### 5.3 QASA mismatch

```text
qasa_selected_empty_fraction                 = 0.0427
qasa_selected_nonempty_precision             = 0.9573
qasa_hard_nonempty_recall                    = 0.9083
hard_owned_token_fraction_unselected_by_qasa = 0.0214
```

Mismatch có thật nhưng nhỏ.

```text
deployed_qasa          = 43.254
raw_hard_qasa_nonempty = 43.263
```

Chặn QASA-selected empty slots chỉ tăng khoảng `+0.008 Mean Recall`.

**Kết luận:** QASA/value-support mismatch không phải root cause của performance failure.

### 5.4 Empty-slot executor contract issue

```text
executor/empty_selected_actual_change_norm_sum = 0.0524
```

Một slot có hard VALUE = zero nhưng nếu QASA chọn nó, Executor vẫn có thể thay đổi state.

Đây là behavior không sạch về contract:

```text
empty VALUE
   nhưng
selected by QASA
   ->
non-zero state transition
```

Nên sửa về lâu dài thành exact no-op cho empty VALUE slot. Tuy nhiên retrieval intervention cho thấy issue này **không giải thích performance drop**.

### 5.5 Raw vs contextual token geometry

```text
raw word embedding pairwise cosine   = 0.2374
contextual embedding pairwise cosine = 0.7247
```

Đây là evidence mạnh rằng Q-Former contextual tokens đã bị global/contextualized rất nhiều.

Hypothesis:

> final contextual text tokens quá global

được hỗ trợ bởi token geometry.

Nhưng hypothesis:

> chỉ cần bỏ contextual VALUE là slots sẽ tự specialization

bị bác bỏ bởi A3.2.

---

## 6. Local-value failure diagnosis — SOFT

Checkpoint:

```text
/home/heheboiz/data/cir/outputs/2026-08-26/21-22-35/best.pt
```

Full validation: 6016 queries.

### 6.1 Retrieval interventions

| Variant | Mean Recall |
|---|---:|
| deployed_qasa | **42.67** |
| raw_hard_qasa | 42.59 |
| raw_hard_qasa_nonempty | 42.59 |
| raw_hard_all_nonempty | 42.59 |
| raw_soft_qasa | **42.67** |
| raw_soft_all | 37.20 |
| contextual_hard_qasa_nonempty | 30.37 |
| contextual_soft_qasa | 30.50 |
| teacher_full | **64.94** |
| reference_only | 12.86 |
| empty_selected_only | 12.86 |

### 6.2 Ownership collapse

Hard-argmax diagnostic của soft ownership:

```text
hard_value_effective_k         = 1.000
hard_value_dominant_share      = 1.000
hard_value_empty_slot_fraction = 0.750
```

Đây là absolute winner monopoly:

```text
mọi valid token
    ->
cùng một winning slot
```

Soft VALUE vẫn cho các slot khác fractional mass, nhưng ownership ranking đã collapse hoàn toàn.

### 6.3 Hard-vs-soft frozen intervention

```text
raw_soft_qasa = 42.67
raw_hard_qasa = 42.59
```

Difference chỉ ~`0.08 Mean Recall`.

Không có evidence rằng chỉ riêng hard assignment là nguyên nhân gây tụt performance.

Quan trọng hơn, hai model đã được train from scratch riêng:

```text
HARD best ≈ 43.27
SOFT best ≈ 42.68
```

=> `soft_shared` không cứu được performance hoặc collapse.

---

## 7. P0 Functional Audit — HARD

Checkpoint:

```text
outputs/2026-08-26/20-48-08/best.pt
```

### 7.1 Results

| Metric | HARD A3.2 |
|---|---:|
| Hard partition mean K | 1.946 |
| Dominant hard token share | 0.827 |
| Gradient error-mode rank | **2.672** |
| Functional Phi effective rank | **1.059** |
| Functionally useful slots | 3.673 |
| QASA functional precision | 0.911 |
| QASA functional recall | 0.471 |
| Median best SINGLE/FULL | **1.010** |
| Median best REPEAT/FULL | **1.668** |
| Median MEANxK/FULL | **0.276** |
| Mean K95 | **1.286** |
| Mean K99 | **1.611** |
| qasa_full Mean Recall | 43.25 |
| all_slots_full Mean Recall | 43.18 |
| reference_only Mean Recall | 13.81 |

### 7.2 Functional interpretation

Task vẫn có nhiều error modes:

```text
gradient rank = 2.672
```

nhưng learned functional effects vẫn gần rank-1:

```text
Phi rank = 1.059
```

Hard privacy **không tạo functional decomposition**.

#### SINGLE/FULL

```text
median best SINGLE/FULL = 1.010
```

Một slot đơn lẻ trung vị đã đạt hoặc vượt forced-all FULL gain.

#### REPEAT/FULL

```text
median best REPEAT/FULL = 1.668
```

Copy một slot vào tất cả executor tickets có thể tạo gain ~1.67× forced-all FULL.

Interpretation:

> Executor/recurrent compute vẫn có thể amplify một single functional direction qua nhiều execution tickets.

Hard information isolation không giải quyết compute-ticket degeneracy.

#### MEANxK/FULL

```text
baseline A3.1 = 1.000
A3.2 HARD     = 0.276
```

Đây là thay đổi quan trọng.

Hard private VALUE đã phá được **mean-slot clone symmetry** ở representation level.

Nhưng vì:

```text
Phi rank       ≈ 1.06
SINGLE/FULL    ≈ 1.01
REPEAT/FULL    ≈ 1.67
```

nên representation khác nhau **không đồng nghĩa** với function khác nhau.

Failure mode chuyển từ:

```text
global/mean clone collapse
```

sang:

```text
giant functional slot + weak/empty auxiliary slots
```

#### K95 / K99

```text
K95 = 1.286
K99 = 1.611
```

Phần lớn full functional gain chỉ cần rất ít slots.

---

## 8. P0 Functional Audit — SOFT

Checkpoint:

```text
outputs/2026-08-26/21-22-35/best.pt
```

### 8.1 Results

| Metric | SOFT A3.2 |
|---|---:|
| Hard partition mean K | **1.000** |
| Dominant hard token share | **1.000** |
| Gradient error-mode rank | **2.654** |
| Functional Phi effective rank | **1.166** |
| Functionally useful slots | 3.888 |
| QASA functional precision | 0.921 |
| QASA functional recall | 0.236 |
| Median best SINGLE/FULL | **1.041** |
| Median best REPEAT/FULL | **1.659** |
| Median MEANxK/FULL | **0.910** |
| Mean K95 | 1.761 |
| Mean K99 | 1.982 |
| qasa_full Mean Recall | 42.67 |
| all_slots_full Mean Recall | 37.20 |
| reference_only Mean Recall | 12.86 |

### 8.2 Interpretation

Ownership collapse hoàn toàn:

```text
K = 1
dominant share = 1
```

Task error structure vẫn:

```text
gradient rank = 2.654
```

nhưng functional Phi:

```text
Phi rank = 1.166
```

vẫn thấp.

```text
SINGLE/FULL = 1.041
REPEAT/FULL = 1.659
MEANxK/FULL = 0.910
```

Một single slot vẫn đủ hoặc tốt hơn full coalition, và repeating một slot qua các execution tickets vẫn cực mạnh.

---

## 9. So sánh A3.1 vs A3.2 HARD vs A3.2 SOFT

| Metric | A3.1 | A3.2 HARD | A3.2 SOFT |
|---|---:|---:|---:|
| Mean Recall qasa | **58.58** | 43.25 | 42.67 |
| Hard K | 1.175 | 1.946 | **1.000** |
| Dominant token share | 0.976 | 0.827 | **1.000** |
| Gradient rank | 2.837 | 2.672 | 2.654 |
| Phi rank | 1.018 | 1.059 | 1.166 |
| SINGLE/FULL | 0.790 | **1.010** | **1.041** |
| REPEAT/FULL | 1.037 | **1.668** | **1.659** |
| MEANxK/FULL | 1.000 | **0.276** | 0.910 |
| K95 | 2.514 | **1.286** | 1.761 |
| K99 | 2.959 | **1.611** | 1.982 |

---

## 10. Hypothesis audit

### H1 — Contextual Q-Former tokens quá global

**Status: SUPPORTED**

Evidence:

```text
raw token pairwise cosine        = 0.237
contextual token pairwise cosine = 0.725
```

Contextual representations đồng dạng hóa token rất mạnh.

### H2 — Soft token sharing là nguyên nhân chính của collapse

**Status: REJECTED AS PRIMARY CAUSE**

Soft A3.2 collapse:

```text
K = 1.0
dominant = 1.0
```

Hard A3.2:

```text
K ≈ 1.95
dominant ≈ 0.83
```

Hard làm token partition bớt monopoly hơn nhưng functional decomposition vẫn gần rank-1 và performance không hồi phục.

### H3 — Hard exclusivity tự nó đủ để tạo specialization

**Status: REJECTED**

Evidence HARD:

```text
Phi rank    = 1.059
SINGLE/FULL = 1.010
REPEAT/FULL = 1.668
K95         = 1.286
```

Slots có thể khác raw content nhưng vẫn không phân chia functional responsibility.

### H4 — Raw/local VALUE tự nó đủ để giải quyết collapse

**Status: REJECTED**

Cả soft và hard raw VALUE đều:

- Recall thấp ~42–43.
- Functional Phi gần rank-1.
- Một slot đơn đủ hoặc tốt hơn full.
- Repeat một slot qua nhiều executor steps cực mạnh.

### H5 — Retrieval task thực sự chỉ cần một edit direction

**Status: NOT SUPPORTED**

Gradient error-mode rank vẫn khoảng:

```text
2.65 – 2.67
```

Task local retrieval error geometry có nhiều independent directions.

Vấn đề là learned slots không ownership các directions đó.

### H6 — Retrieval-only objective không tạo đủ pressure functional specialization

**Status: STRONGLY SUPPORTED BY CURRENT EVIDENCE**

Current objective chỉ yêu cầu:

```text
final query -> target
```

Nó không yêu cầu:

```text
slot 0 owns error mode A
slot 1 owns error mode B
slot 2 owns residual C
...
```

Vì vậy model có nghiệm rẻ hơn:

```text
one giant useful slot
+
unused / weak / redundant slots
+
repeated executor compute
```

---

## 11. Kết luận khoa học chính

### 11.1 Information isolation có tác dụng nhưng không đủ

A3.2 HARD đã phá được một phần clone symmetry:

```text
MEANxK/FULL
1.000 -> 0.276
```

và hard token K tăng:

```text
1.175 -> 1.946
```

Do đó hard/private VALUE **có thay đổi representation structure thật**.

Nhưng functional Phi:

```text
1.018 -> 1.059
```

gần như không cải thiện.

Kết luận:

> **Representation diversity != functional specialization.**

### 11.2 Collapse đã đổi hình thức

A3.1:

```text
near-global slot representations
+
mean/repeat clone behavior
```

A3.2 HARD:

```text
private token supports
+
giant functional slot
+
many empty/weak slots
+
single slot amplified by executor tickets
```

A3.2 SOFT:

```text
absolute winner monopoly
+
soft leakage to other slots
+
global/mean-like functional behavior
```

### 11.3 Core failure hiện tại

```text
TOKEN OWNERSHIP
    !=
FUNCTIONAL ERROR OWNERSHIP
```

A3.2 chỉ cưỡng chế hoặc thay đổi token-information ownership.

Nó chưa cưỡng chế mỗi slot phải giải quyết một residual/error direction khác mà slot trước chưa giải quyết.

---

## 12. Warning về contextual-V intervention

Các frozen probes:

```text
contextual_hard_qasa_nonempty
contextual_soft_qasa
```

rất thấp.

Không được kết luận đơn giản rằng contextual VALUE intrinsically tệ hơn raw VALUE.

`slot_mlp` của checkpoint A3.2 được train trên distribution của raw word embeddings. Thay trực tiếp contextual Q-Former states vào frozen `slot_mlp` tạo distribution shift lớn.

Probe này chỉ nói:

> frozen raw-trained slot pipeline không chịu được representation swap.

Muốn so raw-vs-contextual VALUE công bằng phải train riêng từ scratch.

---

## 13. Những kết luận KHÔNG được phép rút ra

Không được kết luận:

1. `hard assignment` là nguyên nhân duy nhất làm Recall giảm.
2. `raw embedding` intrinsically tốt hơn contextual embedding.
3. `QASA` là nguyên nhân chính của collapse.
4. `K=1` nghĩa task chỉ có một edit factor.
5. `functionally useful slots ≈ 4` nghĩa bốn slots đã specialization.
6. representation cosine thấp nghĩa functional decomposition đã thành công.

---

## 14. Contract issue cần ghi nhớ

Trong HARD run, empty VALUE slot có thể execute:

```text
empty selected state-change norm ≈ 0.0524
```

Nên sửa contract sau:

```text
effective_selected
=
qasa_selected
&
value_nonempty
```

ít nhất cho hard-exclusive mode.

Tuy nhiên đây là cleanup/correctness issue, **không phải root cause** theo retrieval intervention.

---

## 15. Decision after A3.2

Điều đã học được:

```text
Global contextual VALUE leakage
    -> có thật

Soft sharing leakage
    -> có thật

Nhưng cắt cả hai
    -> vẫn không tạo functional specialization
```

Do đó không nên tiếp tục chỉ chỉnh:

- temperature,
- hard/soft,
- QASA threshold,
- token balance,
- slot count,
- Entmax/Sinkhorn/OT,

với kỳ vọng chúng tự sinh functional decomposition.

Các cơ chế này chủ yếu thay assignment geometry, trong khi failure còn lại là functional ownership.

---

## 16. Hướng experiment tiếp theo

Next experiment phải đánh trực tiếp vào:

```text
FUNCTIONAL ERROR OWNERSHIP
```

thay vì chỉ:

```text
TOKEN OWNERSHIP
```

Candidate direction:

1. Xây per-negative retrieval error directions.
2. Slot đầu claim một subset/error direction.
3. Project/remove phần error direction đã được slot đó giải quyết.
4. Slot tiếp theo chỉ được reward cho residual error chưa được giải quyết.
5. Clone direction phải có marginal gain gần zero.
6. Không ép semantic labels.
7. Không ép balanced token count.
8. Cho phép `K_eff = 1` khi query thật sự chỉ cần một factor.
9. Audit bằng exact coalitions, SINGLE, DROP, REPEAT, MEAN, K95/K99.

Core principle:

```text
token ownership
      ↓
không đủ

functional residual ownership
      ↓
pressure cần test tiếp
```

---

## 17. Commands / reports đã dùng

### HARD local-value diagnosis

```bash
python src/diagnose_taper_local_value_failure.py \
  --checkpoint /home/heheboiz/data/cir/outputs/2026-08-26/20-48-08/best.pt \
  --slot-value-assignment hard_st_exclusive \
  --max-queries-per-category 0 \
  --json-output reports/taper_local_value_failure_full.json
```

### SOFT local-value diagnosis

```bash
python src/diagnose_taper_local_value_failure.py \
  --checkpoint /home/heheboiz/data/cir/outputs/2026-08-26/21-22-35/best.pt \
  --slot-value-assignment soft_shared \
  --max-queries-per-category 0 \
  --json-output reports/a3_2_soft_local_value_failure_full.json
```

### HARD P0

```bash
python src/audit_taper_merit_p0.py \
  --checkpoint /home/heheboiz/data/cir/outputs/2026-08-26/20-48-08/best.pt \
  --slot-value-assignment hard_st_exclusive \
  --max-queries-per-category 0 \
  --hard-negatives 16 \
  --json-output reports/a3_2_hard_merit_p0_full.json
```

### SOFT P0

```bash
python src/audit_taper_merit_p0.py \
  --checkpoint /home/heheboiz/data/cir/outputs/2026-08-26/21-22-35/best.pt \
  --slot-value-assignment soft_shared \
  --max-queries-per-category 0 \
  --hard-negatives 16 \
  --json-output reports/a3_2_soft_merit_p0_full.json
```

---

## 18. Final diagnosis

A3.2 không thất bại vô ích.

Nó đã tách được hai vấn đề trước đây bị trộn lẫn:

```text
INFORMATION COLLAPSE
vs
FUNCTIONAL COLLAPSE
```

Kết quả hiện tại cho thấy:

```text
Contextual/global information leakage
    là một vấn đề thật.

Nhưng:
cắt leakage không tự tạo slot specialization.

Model chuyển sang:
giant-slot / single-direction / executor-ticket solution.
```

Kết luận hiện tại:

> **Edit-slot collapse không còn có thể giải thích chỉ bằng token globalization hay soft ownership leakage. Failure còn lại là thiếu pressure để các slots ownership các functional retrieval errors khác nhau.**

Đây là checkpoint để tránh quay lại lặp lại các experiment routing-only mà A3.2 đã falsify.

## A3.2 Executor Shortcut Forensic — Result

To test whether the downstream Executor was the main cause of functional slot collapse, we ran a frozen-checkpoint forensic analysis on all 6,016 FashionIQ validation queries.

The test explicitly isolated two hypotheses:

1. **Slot-content ignoring:** whether the Executor produces nearly the same transition when the real slot is replaced by a zero or shuffled slot.
2. **Compute-ticket shortcut:** whether repeatedly executing the same dominant slot can substitute for using multiple distinct slots.

### Results

Slot-content dependence was clearly present:

- median real-vs-zero relative transition difference: **0.985**
- median real-vs-zero transition cosine: **0.427**
- median real-vs-shuffled relative transition difference: **1.190**
- median real-vs-shuffled transition cosine: **0.286**

Therefore, the Executor is **not simply ignoring Edit Slot content**. Replacing the actual slot with zero or with another sample's slot substantially changes both the magnitude and direction of the state update.

Repeated execution of the same dominant slot does amplify the amount of state/query movement:

- repeated ×K_eff / single query movement: **1.338×**
- repeated ×4 / single query movement: **1.598×**

However, this extra recurrent computation does **not** improve retrieval:

| Intervention | Mean Recall |
|---|---:|
| Original hard-nonempty slots | **43.346** |
| Dominant slot ×1 | **42.296** |
| Dominant slot ×K_eff | **41.455** |
| Dominant slot ×4 | **35.826** |

Performance decreases as the same slot is repeated. Therefore, recurrent execution depth cannot simply replace genuine multi-slot information.

### Updated Diagnosis

These results substantially weaken the hypothesis that the Executor is the primary source of A3.2 functional collapse.

The current evidence instead indicates:

> **The Edit Slots are representationally different, and the Executor is sensitive to those differences, but training does not provide sufficient pressure for the slots to acquire distinct functional responsibilities.**

The failure is therefore more likely located in the **slot-learning / routing dynamics** rather than in the Executor itself.

A plausible training shortcut is:

```text
one slot becomes slightly more useful
        ↓
selected / reinforced more often
        ↓
receives more useful retrieval signal
        ↓
becomes the giant slot
        ↓
remaining slots stay weak, auxiliary, or empty