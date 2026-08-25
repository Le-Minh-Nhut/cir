# Competitive NULL × QASA Slot Filtering

## 1. Mục tiêu của nhánh

Nhánh này thử nghiệm việc thay thế cơ chế **learned sigmoid slot gate** hiện tại của TAPER bằng một cơ chế **QASA-inspired slot selector**.

Mục tiêu không phải là sao chép toàn bộ kiến trúc QASA, mà là lấy phần:

```text
slot attention
→ quality estimation
→ coverage + novelty based selection
→ adaptive number of active slots
```

và áp dụng nó lên hệ thống Edit Slots hiện tại của TAPER.

Điểm quan trọng là TAPER hiện tại đã có một cơ chế khác mà QASA gốc không có:

```text
Competitive NULL Ownership
```

Do đó việc ghép QASA vào TAPER không thể đơn giản là lấy attention của các Edit Slots rồi chạy Algorithm 1 của QASA.

Ta phải quyết định rõ:

> NULL ảnh hưởng như thế nào đến attention mà QASA nhìn thấy?

Đây là một trong những design decisions quan trọng nhất của experiment này.

---

# 2. Hai cơ chế giải quyết hai câu hỏi khác nhau

Competitive NULL và QASA không nên được coi là cùng một loại selector.

Chúng giải quyết hai câu hỏi khác nhau.

## Competitive NULL

Competitive NULL trả lời:

```text
Token này có thực sự thuộc về một modification/edit factor nào không?
```

Ở mỗi token \(t\), ta có:

$$
P(\varnothing\mid t)
+
\sum_{i=1}^{K}
P(S_i\mid t)
=
1.
$$

Trong đó:

```text
NULL / ∅
    token không nên thuộc Edit Slot nào

S0 ... SK
    các candidate Edit Slots
```

Ví dụ:

```text
"red"

NULL = 0.03
S0   = 0.80
S1   = 0.10
S2   = 0.04
S3   = 0.03
```

ở đây một Edit Slot rõ ràng đang claim token.

Ngược lại:

```text
"the"

NULL = 0.95
S0   = 0.02
S1   = 0.01
S2   = 0.01
S3   = 0.01
```

thì ý nghĩa tự nhiên là:

```text
token này gần như không phải edit evidence
```

---

## QASA

QASA lại trả lời:

```text
Trong các candidate slots hiện tại,
slot nào có chất lượng đủ tốt và cung cấp evidence đủ khác biệt
để cần được giữ lại?
```

QASA gốc không có NULL slot.

Nó nhận attention:

$$
A\in[0,1]^{N\times K}
$$

với:

$$
\sum_{i=1}^{K} A_{t,i}=1
$$

và tính quality của từng slot dựa trên việc:

```text
slot có attention mass lớn ở những token mà chính slot đó thắng hay không?
```

Sau đó QASA dùng:

```text
Quality
+
Coverage
+
Novelty
```

để quyết định một subset slots.

Do đó về mặt logic, pipeline mong muốn là:

```text
Competitive NULL
        ↓
xác định edit evidence
        ↓
QASA
        ↓
xác định Edit Slots nào đáng tồn tại
        ↓
Router
        ↓
Executor
```

chứ không phải:

```text
QASA thay luôn vai trò của NULL
```

---

# 3. Vấn đề khi bỏ NULL rồi renormalize trực tiếp

Một cách triển khai rất dễ nghĩ tới là:

```python
edit_probs = slot_probs / slot_probs.sum(dim=slot_axis)
```

Tức là:

```text
bỏ NULL
→ renormalize S0...S3 về tổng bằng 1
→ đưa vào QASA
```

Về mặt toán học, đây tương đương với:

$$
P(S_i\mid t,\text{EDIT})
=
\frac{P(S_i\mid t)}
{\sum_jP(S_j\mid t)}.
$$

Điều này tạo ra đúng một distribution trên các Edit Slots:

$$
\sum_iP(S_i\mid t,\text{EDIT})=1.
$$

Vấn đề là:

> Sau khi condition trên EDIT, toàn bộ thông tin về việc NULL mạnh đến mức nào bị mất.

---

# 4. Ví dụ đơn giản

Giả sử token `"the"` có:

```text
NULL = 0.95
S0   = 0.02
S1   = 0.01
S2   = 0.01
S3   = 0.01
```

Tổng Edit mass:

$$
P(\text{EDIT}\mid t)
=
1-P(NULL\mid t)
=
0.05.
$$

Nếu bỏ NULL và renormalize:

$$
\frac{
[0.02,0.01,0.01,0.01]
}{
0.05
}
$$

ta nhận:

$$
[0.40,0.20,0.20,0.20].
$$

QASA lúc này chỉ nhìn thấy:

```text
S0 = 40%
S1 = 20%
S2 = 20%
S3 = 20%
```

Nó không còn biết rằng:

```text
95% probability ban đầu thuộc NULL
```

Tức một token mà Competitive NULL gần như đã reject hoàn toàn lại trở thành một token có apparent Edit Slot winner khá rõ.

---

# 5. Trường hợp nguy hiểm hơn: NULL thắng toàn bộ câu

Ví dụ:

```text
"make"

NULL = .90
S0   = .04
S1   = .03
S2   = .02
S3   = .01
```

```text
"it"

NULL = .92
S0   = .02
S1   = .02
S2   = .02
S3   = .02
```

```text
"different"

NULL = .88
S0   = .05
S1   = .03
S2   = .02
S3   = .02
```

Competitive NULL đang biểu diễn:

```text
không Edit Slot nào thực sự sở hữu evidence mạnh
```

Nhưng sau khi bỏ NULL:

```text
"make"
→ S0 .40
   S1 .30
   S2 .20
   S3 .10
```

```text
"it"
→ S0 .25
   S1 .25
   S2 .25
   S3 .25
```

```text
"different"
→ S0 .42
   S1 .25
   S2 .17
   S3 .17
```

QASA sẽ luôn nhìn thấy một distribution hợp lệ giữa các slots.

Do đó có thể xảy ra:

```text
Competitive NULL:
    không có edit evidence đáng kể

nhưng

QASA sau renormalization:
    S0 thắng nhiều token
    → S0 quality cao
    → S0 được selected
    → Executor vẫn chạy
```

Đây là một inconsistency giữa hai tầng.

---

# 6. Điều chắc chắn cần tránh

Điểm cần khóa là:

$$
\boxed{
\text{Không được renormalize Edit Slots rồi hoàn toàn quên mất NULL mass.}
}
$$

Nói cách khác:

```text
QASA không được vô tình undo quyết định của Competitive NULL.
```

Nếu NULL đang đóng vai trò:

```text
reject / no-edit sink
```

thì thông tin reject này phải được bảo toàn khi đưa attention sang QASA.

---

# 7. Phương án hard NULL-winner rejection

Một phương án đơn giản là:

```python
ownership_winner = ownership_logits.argmax(dim=1)

qasa_valid = (
    slot_valid
    & ownership_winner.ne(0)
)
```

Trong đó:

```text
0 = NULL
1 = S0
2 = S1
...
```

Ta chỉ cho token tham gia QASA nếu:

$$
\arg\max
\{
P(NULL\mid t),
P(S_0\mid t),
\dots,
P(S_K\mid t)
\}
\neq NULL.
$$

Ý nghĩa:

```text
NULL thắng token
→ token không được coi là edit evidence
→ QASA bỏ token đó
```

Ví dụ:

```text
token A:
NULL thắng
→ qasa_valid = false

token B:
S1 thắng
→ qasa_valid = true

token C:
S3 thắng
→ qasa_valid = true
```

QASA chỉ chấm B và C.

---

# 8. Ý nghĩa của hard NULL gate

Pipeline trở thành:

```text
Token
  ↓
Competitive NULL Ownership
  ↓
NULL thắng?
  ├── YES
  │      ↓
  │   reject token
  │
  └── NO
         ↓
      Edit evidence
         ↓
      QASA
```

Ta có thể hiểu thành hai conditional questions.

## Stage A

```text
Token này có phải edit evidence hay không?
```

Competitive NULL trả lời.

## Stage B

```text
Nếu đây là edit evidence,
slot nào sở hữu nó tốt nhất
và slot nào cần được giữ lại?
```

QASA trả lời.

---

# 9. All-NULL invariant

Nếu NULL thắng toàn bộ valid content tokens:

$$
qasa\_valid_t=0
\quad\forall t
$$

thì:

```text
QASA không có evidence token nào để chấm.
```

Expected behavior:

$$
K_{\text{selected}}=0.
$$

Do đó:

```text
0 active Edit Slots
→ Router không có candidate
→ Executor không thực hiện transition
→ state được giữ nguyên
```

Tức:

$$
\boxed{
\text{all NULL}
\Rightarrow
\text{zero selected Edit Slots}
\Rightarrow
\text{exact no-op}
}
$$

Đây là một useful invariant cho TAPER.

---

# 10. Tuy nhiên hard NULL winner không phải QASA gốc

Cần ghi rõ:

> QASA paper không có NULL competitor.

Do đó:

```python
ownership_winner != NULL
```

không phải một phần của thuật toán QASA gốc.

Nó là:

$$
\boxed{
\text{TAPER-specific adaptation}
}
$$

được thêm vào để nối:

```text
Competitive NULL
```

với:

```text
QASA slot selection.
```

Vì vậy nếu experiment thành công, không được mô tả:

```text
we reproduce QASA exactly
```

Mà phải mô tả gần hơn với:

```text
We apply a QASA-derived quality/coverage/novelty
selection rule on top of TAPER's Competitive-NULL
Edit-Slot ownership.
```

---

# 11. Vấn đề của hard argmax NULL rejection

Hard winner rule cũng không hoàn hảo.

Ví dụ:

```text
NULL = 0.35

S0 = 0.30
S1 = 0.20
S2 = 0.10
S3 = 0.05
```

NULL là class lớn nhất riêng lẻ:

$$
0.35>0.30.
$$

Do đó hard winner rule nói:

```text
NULL thắng
→ bỏ token
```

Nhưng tổng Edit mass là:

$$
P(EDIT)
=
0.30+0.20+0.10+0.05
=
0.65.
$$

Tức model tổng thể đang nói:

```text
65% probability cho rằng token thuộc một Edit Slot nào đó.
```

Chỉ vì evidence này phân tán giữa nhiều slots nên NULL thắng individual argmax.

Do đó hard NULL winner có thể reject token hơi quá mạnh.

---

# 12. Ba cách bảo toàn thông tin NULL

Hiện có ba lựa chọn hợp lý.

---

## Option A — Hard NULL winner rejection

Rule:

$$
valid_t
=
\mathbf 1[
\arg\max
(NULL,S_1,\dots,S_K)
\neq NULL
].
$$

Ưu điểm:

```text
rất rõ ràng
dễ audit
dễ smoke-test
NULL semantics mạnh
all-NULL → exact no-op
```

Nhược điểm:

```text
hard discontinuity
có thể reject token dù tổng Edit mass > NULL
```

Đây là phương án đơn giản nhất cho experiment đầu tiên.

---

## Option B — Threshold theo total Edit mass

Định nghĩa:

$$
p^{edit}_t
=
1-P(NULL\mid t).
$$

Sau đó token được QASA xem nếu:

$$
p^{edit}_t\ge\eta.
$$

Ví dụ:

```text
NULL=.95
→ Edit=.05
→ reject

NULL=.35
→ Edit=.65
→ keep
```

Ưu điểm:

```text
không mắc lỗi "NULL thắng individual argmax
nhưng tổng Edit mass lớn"
```

Nhược điểm:

```text
thêm một hyperparameter η
phải chọn/tune threshold
```

Điều này làm experiment ít paper-clean hơn.

---

## Option C — Soft Edit evidence weight

Không hard reject token.

Thay vào đó giữ:

$$
p^{edit}_t
=
1-P(NULL\mid t)
$$

như một evidence weight.

Ví dụ Edit-only conditional attention:

$$
\tilde A_{t,i}
=
P(S_i\mid t,EDIT).
$$

Sau đó effective attention có thể được xây từ:

$$
A^{eff}_{t,i}
=
p^{edit}_t
\cdot
\tilde A_{t,i}.
$$

Vì:

$$
p^{edit}_t
=
1-P(NULL\mid t),
$$

ta có:

$$
A^{eff}_{t,i}
=
P(S_i\mid t).
$$

Điều này bảo toàn luôn Edit mass.

Ví dụ:

```text
NULL=.95
```

thì tổng effective Edit attention chỉ còn:

$$
0.05.
$$

Token gần như không đóng góp vào Quality/Coverage.

Ngược lại:

```text
NULL=.10
```

thì:

$$
P(EDIT)=0.90
$$

và token có ảnh hưởng mạnh.

Ưu điểm:

```text
không mất NULL confidence
không hard reject sớm
giữ nhiều thông tin nhất
```

Nhược điểm quan trọng:

```text
QASA gốc giả định attention normalized
over candidate slots cho mỗi token.

Sau khi multiply bởi P(EDIT),
tổng attention trên Edit Slots không còn bằng 1.
```

Do đó Option C không còn là direct application của QASA equations.

Nó là một deeper adaptation cần audit lại Quality, Coverage và Novelty definitions.

---

# 13. Vì sao Option C không thể được thêm tùy tiện

QASA gốc giả định:

$$
\sum_iA_{t,i}=1.
$$

Nếu dùng:

$$
A^{eff}_{t,i}
=
P(EDIT\mid t)
P(S_i\mid t,EDIT),
$$

thì:

$$
\sum_iA^{eff}_{t,i}
=
P(EDIT\mid t),
$$

không nhất thiết bằng 1.

Ví dụ:

```text
NULL = .95
```

thì:

$$
\sum_i A^{eff}_{t,i}=.05.
$$

Điều này có thể hợp lý về mặt TAPER, nhưng QASA coverage:

$$
\mathbf 1
\left[
\sum_{i\in S}A_{t,i}\ge\tau
\right]
$$

sẽ thay đổi interpretation.

Trong paper:

```text
tau
```

được hiểu trong một normalized slot distribution.

Trong soft-NULL version:

```text
tau
```

đồng thời trở thành một implicit Edit-confidence threshold.

Ví dụ:

$$
P(EDIT)=0.3
$$

thì dù chọn tất cả Edit Slots:

$$
\sum_iA^{eff}_{t,i}=0.3.
$$

Nếu:

$$
\tau=0.5,
$$

token đó sẽ **không bao giờ được covered**.

Điều này có thể là desirable, nhưng nó là một thuật toán khác.

Do đó Option C cần một experiment riêng, không nên silently merge vào initial QASA test.

---

# 14. Quyết định cho experiment đầu tiên

Để giữ experiment đầu tiên đơn giản, interpretable và dễ audit, lựa chọn ban đầu là:

$$
\boxed{
\text{Option A: hard NULL-winner rejection}
}
$$

Pipeline:

```text
Competitive ownership
        ↓
NULL winner filtering
        ↓
Edit-only conditional attention
        ↓
QASA Algorithm 1
        ↓
selected Edit Slots
        ↓
Router
        ↓
Executor
```

Với các Edit tokens được giữ lại:

$$
A^{QASA}_{t,i}
=
P(S_i\mid t,EDIT).
$$

Do đó trên mỗi token QASA-valid:

$$
\sum_iA^{QASA}_{t,i}=1.
$$

Như vậy Algorithm 1 của QASA vẫn được chạy trên đúng loại normalized attention mà nó mong đợi.

---

# 15. NULL không được chấm Quality

Một invariant khác cần khóa:

$$
\boxed{
NULL \notin QASA
}
$$

Tức không có:

```text
Q_NULL
```

Không:

```text
sort NULL by quality
```

Không:

```text
select NULL
```

Không:

```text
NULL enters Router
```

Không:

```text
NULL enters Executor
```

NULL chỉ tham gia tầng Competitive Ownership.

Sau đó nó đóng vai trò:

```text
reject / sink / no-edit option
```

chứ không trở thành một executable slot.

---

# 16. Quality score của QASA chỉ được tính cho Edit Slots

Sau NULL filtering, với attention:

$$
A_{t,i}
$$

QASA tính:

$$
w_t
=
\arg\max_iA_{t,i}.
$$

Sau đó:

$$
W_i^{win}
=
\sum_{t:w_t=i}A_{t,i},
$$

$$
W_i
=
\sum_tA_{t,i},
$$

và:

$$
Q_i
=
\frac{
W_i^{win}
}{
W_i+\epsilon
}.
$$

Interpretation:

```text
Q_i cao
→ phần lớn attention mass của slot i
  nằm ở những token mà chính slot i thắng

Q_i thấp
→ slot có nhiều diffuse attention
  ở những token thực ra slot khác sở hữu tốt hơn
```

Quality không phải:

```text
slot importance probability
```

và cũng không phải:

```text
Executor strength.
```

---

# 17. QASA không đơn giản threshold Quality

Một lỗi implementation khác cần tránh:

```python
selected = quality > threshold
```

QASA không làm vậy.

Actual selection logic là:

```text
1. tính quality của tất cả Edit Slots
2. sort giảm dần theo Quality
3. xét từng slot theo thứ tự đó
4. tính Novelty(candidate | already selected)
5. nếu novelty thấp → skip
6. nếu novelty đủ → add
7. recompute Coverage
8. nếu coverage >= rho → stop
```

Do đó số slot active được quyết định bởi:

```text
Quality ordering
+
Novelty
+
Coverage stopping
```

chứ không phải một learned sigmoid hay một single threshold.

---

# 18. Learned slot gate cũ phải được loại bỏ

Base TAPER hiện tại có:

```text
edit_slot
→ MLP
→ sigmoid
→ slot_gate
```

và gate đó hiện ảnh hưởng nhiều chỗ:

```text
candidate eligibility
Router score bias
transition magnitude
STE recovery
```

Trong QASA experiment, nếu giữ lại gate cũ rồi đặt QASA phía trước hoặc phía sau:

```text
QASA
→ sigmoid gate
```

thì ta không còn biết kết quả đến từ:

```text
QASA
```

hay:

```text
learned gate.
```

Do đó experiment phải thay thế gate cũ hoàn toàn.

Architecture mong muốn:

```text
Competitive NULL
        ↓
QASA
        ↓
binary slot existence mask
        ↓
Router
        ↓
Executor
```

Không còn:

```text
slot_gate_threshold
hard_slot_gating_during_training
gate_mode
st_gate_recovery
log(slot_gate) Router bias
alpha * slot_gate
```

---

# 19. QASA chỉ quyết định existence, không quyết định strength

Một selected slot:

```text
selected = true
```

chỉ có nghĩa:

```text
slot này được phép tham gia candidate set của Router.
```

Nó không có nghĩa:

```text
Quality cao → execute mạnh hơn.
```

Không nên dùng:

$$
\alpha'=\alpha Q_i.
$$

Không nên dùng:

```python
router_score += log(Q_i)
```

trong experiment đầu tiên.

Nếu làm vậy, QASA Quality đồng thời trở thành:

```text
selection signal
+
routing prior
+
action magnitude
```

và experiment không còn isolate đúng vai trò của QASA.

Do đó:

$$
\boxed{
QASA = existence selector only
}
$$

cho experiment này.

---

# 20. Exact no-op khi không có slot nào được chọn

Nếu:

$$
K_{selected}=0,
$$

Executor phải:

```text
không chọn primitive
không apply transition
không LayerNorm lại state
không drift state
```

và:

$$
z_{final}=z_0.
$$

Đây là exact no-op contract.

Nó rất quan trọng vì nếu:

```text
NULL wins everything
```

nhưng state vẫn thay đổi do một normalization/update side effect thì NULL semantics bị phá.

---

# 21. Đây không phải reproduction toàn bộ QASA

Cần phân biệt rất rõ.

QASA paper:

```text
canonical Slot Attention
→ iterative slot refinement
→ QASA quality selection
```

TAPER hiện tại:

```text
learned Edit Slot queries
+
Competitive NULL token ownership
+
mass-aware slot pooling
+
counterfactual slot effect
→ Edit Slots
```

Sau đó mới:

```text
QASA-derived selection
```

Do đó nhánh này phải được gọi là:

```text
TAPER + QASA-derived slot filtering
```

hoặc:

```text
Competitive-NULL TAPER with QASA slot selection
```

Không nên gọi:

```text
exact QASA reproduction
```

---

# 22. Training vs inference cũng là một adaptation

QASA paper sử dụng quality selection trong training, nhưng inference behavior của paper không hoàn toàn giống việc giữ hard QASA selector ở deployment.

TAPER experiment này ban đầu muốn QASA trở thành actual slot filter.

Do đó default experiment có thể dùng:

```text
train:
    QASA enabled

eval:
    QASA enabled
```

Nhưng nên giữ một ablation:

```text
train:
    QASA enabled

eval:
    QASA disabled
```

để phân biệt:

```text
QASA có giúp tạo training pressure hay không?
```

với:

```text
QASA có thực sự cần làm inference-time pruning hay không?
```

---

# 23. Các invariants bắt buộc

## QASA-NULL-1

NULL tham gia Competitive Ownership:

$$
P(NULL|t)+\sum_iP(S_i|t)=1.
$$

---

## QASA-NULL-2

NULL không có Quality score.

---

## QASA-NULL-3

NULL không thể được selected.

---

## QASA-NULL-4

NULL không vào Router hoặc Executor.

---

## QASA-NULL-5

Token mà NULL thắng bị loại khỏi QASA trong initial hard-reject design.

---

## QASA-NULL-6

Nếu NULL thắng toàn bộ valid content tokens:

$$
K_{selected}=0.
$$

---

## QASA-NULL-7

Nếu:

$$
K_{selected}=0,
$$

thì:

$$
z_{final}=z_0
$$

exactly.

---

## QASA-A-1

Với mọi QASA-valid token:

$$
\sum_iA^{QASA}_{t,i}=1.
$$

---

## QASA-A-2

Special/padding tokens không ảnh hưởng:

```text
Quality
Novelty
Coverage
```

---

## QASA-A-3

Quality chỉ được tính trên Edit Slots.

---

## QASA-SEL-1

Slots được xét theo descending Quality.

---

## QASA-SEL-2

Novelty thấp hơn \(\mu\) phải skip candidate.

---

## QASA-SEL-3

Selection dừng khi:

$$
Coverage\ge\rho.
$$

---

## QASA-SEL-4

Không có sigmoid gate hoặc STE selector chen sau QASA.

---

## QASA-EXEC-1

QASA mask chỉ ảnh hưởng candidate existence.

---

## QASA-EXEC-2

Quality không trực tiếp scale transition strength.

---

# 24. Metrics cần log

Experiment không nên chỉ log retrieval.

Cần log:

```text
null_ownership_rate
null_argmax_fraction

qasa_selected_slot_count
qasa_quality_mean
qasa_final_coverage
qasa_novelty_skip_count

per-slot quality

slot mass
dominant slot share
near-monopoly fraction

execution active slot count
```

Đặc biệt cần nhìn quan hệ:

```text
NULL rate
vs
QASA selected count
```

---

# 25. Các failure pattern quan trọng

## Pattern A

```text
NULL rất cao
QASA selected count vẫn gần 4
```

Có thể có bug:

```text
NULL information bị mất trước QASA.
```

---

## Pattern B

```text
NULL rất cao
QASA selected count → 0
```

Điều này phù hợp với hard NULL-reject semantics.

Nhưng vẫn phải xem:

```text
retrieval có collapse hay không?
```

---

## Pattern C

```text
QASA selected count ≈ 4 trên gần mọi sample
```

QASA thực tế không pruning.

Khả năng:

```text
coverage threshold khó đạt
attention quá diffuse
novelty gần như luôn cao
```

---

## Pattern D

```text
QASA selected count ≈ 1 trên gần mọi sample
```

Có thể là:

```text
một slot đủ coverage gần như toàn bộ tokens
```

hoặc:

```text
ownership collapse
```

Cần phân biệt bằng:

```text
quality
coverage
dominant slot share
token ownership maps
```

---

## Pattern E

```text
selected K đa dạng
nhưng tất cả Edit Slots vẫn semantic giống nhau
```

Điều này nhắc rằng:

$$
\boxed{
QASA selection
\neq
slot specialization proof
}
$$

QASA giúp selection/pruning.

Nó không tự động tạo causal pressure đủ để các slots học semantic factors khác nhau.

---

# 26. Scientific interpretation

Nếu experiment thành công, safe conclusion ban đầu là:

```text
QASA-style adaptive slot selection
is more effective than the previous learned sigmoid gate
for deciding which TAPER Edit Slots participate in execution.
```

Không được nhảy thẳng sang:

```text
QASA proves TAPER discovered compositional semantic edit factors.
```

QASA giải quyết:

```text
slot quality
slot redundancy
adaptive K
```

nhưng không chứng minh:

```text
semantic disentanglement
causal factorization
functional necessity of multiple Edit Slots
```

---

# 27. Current experiment decision

Initial implementation sử dụng:

```text
Competitive NULL
        ↓
hard NULL-winner evidence rejection
        ↓
Edit-only normalized attention
        ↓
QASA Quality
        ↓
QASA Novelty
        ↓
QASA Coverage
        ↓
hard selected Edit Slots
        ↓
Slot × Primitive Router
        ↓
Executor
```

NULL:

```text
participates in token competition
does not participate in QASA ranking
does not receive quality
does not execute
```

---

# 28. Open research question

Hard NULL-winner rejection được xem là **initial controlled design**, không phải final theorem.

Một follow-up experiment có thể so sánh:

```text
A. hard NULL winner rejection

B. threshold on total Edit probability:
   1 - P(NULL)

C. soft NULL-aware evidence weighting
```

Nhưng không nên trộn các variant đó vào run đầu tiên.

Mục tiêu của run đầu tiên là trả lời một câu hỏi sạch:

$$
\boxed{
\text{Nếu thay learned sigmoid slot gate bằng
QASA-derived quality/coverage/novelty selection,
TAPER có cải thiện slot selection hay không?}
}
$$

với Competitive NULL semantics vẫn được giữ rõ ràng.
