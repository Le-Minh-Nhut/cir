# CIR V2 — Kế hoạch sửa lỗi và phục hồi benchmark với vision frozen

Ngày tổng hợp: 2026-09-07.

Repo: `Le-Minh-Nhut/cir`, branch `exp/e2e-iag-srme-v2-r0`.

Snapshot làm căn cứ: `c00147863c34c5871f65d62b76a04a1aa985954a`.

**Trạng thái:** đây là kế hoạch triển khai dựa trên audit source, raw diagnostics và các kiểm chứng toán/CPU đã thực hiện. Những sửa đổi và thí nghiệm dưới đây chưa được thực hiện trong repo, trừ khi ghi rõ “đã xác nhận”. Chưa có benchmark mới sau sửa. File này đủ để tiếp tục công việc khi mất ngữ cảnh; không cần đọc toàn bộ nhật ký điều tra trước để bắt đầu.

**Ràng buộc:** giữ vision encoder và native image projection frozen; fine-tune text encoder. Các module IAG-SRME mới vẫn được train như hiện tại. “Text-only fine-tuning” ở đây nói về backbone, không có nghĩa đóng băng cả Proposer/Executor/ScoreNet.

## 1. Hiện tại cần làm gì trước?

Thực hiện theo thứ tự sau; chưa chạy thêm một run strong dài với cấu hình hiện tại.

| Thứ tự | Việc cần làm | Kết quả phải thu được |
|---|---|---|
| 1 | Cố định config/checkpoint, sửa logging và diagnostic | Số liệu OLD/STRONG so sánh đúng representation, cùng input và cùng gallery |
| 2 | Đánh giá lại hai checkpoint và bổ sung per-slot geometry | Biết pooled rank thấp do slot offsets hay do từng slot mất biến thiên theo input |
| 3 | Thử scorer gain-only, giữ nguyên bộ sinh candidate | Biết lỗi pair/gain có làm sai dấu score và STOP trong model thực tế không |
| 4 | Thêm loss chất lượng trực tiếp cho candidate chưa được chọn | Các slot yếu bớt gây hại; theo dõi cả no-op và best-of-K improvement |
| 5 | So full-K DPP với useful-subset DPP | Đa dạng có đi cùng chất lượng không, thay vì đẩy candidate xấu đi xa hơn |
| 6 | Chạy baseline native image+text và editor K1/T1 | Biết độ phức tạp IAG-SRME mang lại lợi ích gì |
| 7 | Trên cấu hình tốt nhất, tách LR và mở rộng TRAIN negative bank | Tăng chất lượng tối ưu và độ khó retrieval mà không fine-tune vision |
| 8 | Sau khi one-step ổn, kiểm tra K4 và T2/T3 | Xác nhận nhiều candidate/nhiều bước có lợi ích thực tế |

Nếu tài nguyên ít, hoàn thành bước 1–4 và một baseline đơn giản trước. Chỉ đưa những cấu hình có tín hiệu tốt vào training đủ ngân sách. Không cần chạy tất cả tổ hợp trong file này.

**Ba việc chưa nên làm:** ép các slot được chọn đều 25%; thêm loss tăng rank chỉ vì thấy PR≈3; tăng đồng thời nhiều auxiliary weights. Chúng không trực tiếp giải quyết candidate quality hoặc score calibration.

## 2. Chúng ta đã biết gì, còn chưa biết gì?

### 2.1 Những phát hiện đã có bằng chứng

| Phát hiện | Bằng chứng | Ý nghĩa hành động |
|---|---|---|
| OLD có candidates rất giống nhau; STRONG khác nhau hơn nhưng metric thấp hơn | Metric checkpoint 30.824426 → 28.244677; diagnostic diversity/utility | Diversity tự nó chưa đủ |
| C3 được ưu tiên và thực sự tốt hơn trên visited states | Model chọn C3 72.73%; oracle 63.21%; chỉ C3 có mean utility dương | Cần nâng chất lượng các slot còn lại, không chỉ cân bằng selector |
| STOP hoạt động kém trên diagnostic hiện tại | Chỉ 1 true-positive trong 45 oracle STOP: recall 2.22% | Ưu tiên score calibration và đo STOP đúng |
| Pair loss có thể xung đột với mốc score=0 của gain loss | Phản ví dụ tối ưu bằng các hàm loss thật của repo | Ablate gain-only trước |
| Scorer teacher được detach; task gradient qua hard-selected path không trực tiếp giám sát mọi preview | Source/autograd audit | Thêm live candidate-quality loss |
| Full-K DPP vẫn tác động candidate xấu khi parent qua useful gate | Autograd probe | Thử useful-subset DPP, cùng quality supervision |
| STRONG global rank tập trung mạnh sau edit đầu rồi phục hồi | PR global 29.67 → 9.53 → 22.24; terminal query PR 24.31 | Điều tra transient concentration; chưa gọi terminal collapse |
| Diagnostic geometry gộp input và slot | Reshape `[B,K,D]` thành `[BK,D]` | Cần per-slot và slot-centered spectra |

Scorer diagnostic dùng 473 decisions từ 160 TRAIN inputs, utility từ batch 8. Đây không phải 473 input độc lập hoặc full-gallery validation. Geometry dùng 160 TRAIN inputs nhưng seed lấy mẫu khác selector; chưa thể ghép từng quyết định C3 với từng hiện tượng geometry.

### 2.2 Các giả thuyết chưa được xác nhận

- Proposer bỏ qua text/input do collapse kiểu JEPA.
- DPP một mình gây giảm benchmark: bốn hệ số đã cùng tăng.
- Mean-pooled state rank thấp nghĩa toàn bộ patch state mất thông tin.
- Positive cosine giảm nghĩa target rank xấu hơn.
- Tăng effective rank sẽ tăng Recall: terminal STRONG có rank cao hơn OLD nhưng Recall thấp hơn.
- Cần fine-tune vision mới có thể cải thiện.

Phân biệt “code cho phép một nghiệm xấu” với “checkpoint hiện tại đã rơi đúng nghiệm đó”. Các phản ví dụ toán chỉ chứng minh điều thứ nhất; diagnostic/intervention mới kiểm tra điều thứ hai.

## 3. P0 — Cố định thí nghiệm và sửa khả năng quan sát

### 3.1 Giữ hai mốc OLD và STRONG

Giữ nguyên checkpoint và artifact cũ. Tạo config experiment rõ ràng cho từng run, không sửa `core.yaml` rồi coi mọi run cũ dùng config mới.

| Hệ số | OLD | STRONG |
|---|---:|---:|
| `lambda_c` | 0.01 | 0.6 |
| `lambda_bind` | 0.01 | 1.0 |
| `lambda_rel` | 0.001 | 0.6 |
| `lambda_dpp` | 0.01 | 0.6 |

OLD là mốc benchmark tốt hơn để recovery. STRONG là checkpoint cần giải thích và là nhánh thử rescue. Không coi chúng là thí nghiệm chỉ khác DPP hoặc coi epoch 15 và 18 là matched training budget.

Mỗi run phải lưu: git SHA, resolved config, seed, dataset/split/category, caption policy, image preprocessing, backbone revision, readout mode, trainable parameter groups, K, T, STOP rule, batch/accumulation, precision, LR/scheduler, số optimizer updates thật và tiêu chí chọn best checkpoint.

### 3.2 Sửa evaluation/config và logging

File cần xem: `src/evaluate.py`, `src/training/engine.py`, `conf/protocol/fashioniq_val.yaml`.

- Khi load checkpoint, restore hoặc kiểm tra các config ảnh hưởng hành vi; không chỉ load weights rồi dùng config hiện hành. Mọi override T/STOP/readout phải được ghi rõ là counterfactual evaluation.
- `fashioniq_val.yaml` rỗng trong snapshot đã audit: điền theo schema protocol evaluator thực sự yêu cầu trước khi dùng lựa chọn này. Không quy lỗi đó cho một run cũ vốn đã evaluate thành công qua protocol khác.
- Ghi JSONL các loss thành phần, metric từng category, R@10/R@50 và mean recall; không chỉ in total loss.
- Log AMP skipped updates/scale, finite gradients, optimizer-step count; probe parameter delta định kỳ. Loss hữu hạn không đảm bảo optimizer đã update.
- Khi cần tiếp tục training chính xác, lưu optimizer/scheduler/scaler/RNG/global step/best metric. Nạp weights và tạo optimizer mới phải gọi là warm-start, không gọi exact resume.

**Điều kiện hoàn thành:** hai checkpoint evaluate bằng cùng protocol; config mismatch được phát hiện; log đủ để biết model đang học, STOP khi nào và candidate nào có ích. Sai lệch khi replay phải được giải thích trước khi dùng chênh lệch nhỏ để chọn run.

## 4. P0 — Sửa diagnostic trước khi kết luận collapse

File chính: `src/diagnose_latent_geometry.py`, `src/diagnose_candidate_selector.py`, `src/diagnose_iag_srme.py`.

### 4.1 Dùng cùng cohort và đúng mẫu số

Xuất một manifest sample IDs/captions/category cố định, dùng cho cả geometry lẫn selector và cho OLD/STRONG. Tách hai mục đích: TRAIN diagnostic để quan sát trạng thái đã học; validation diagnostic để kiểm tra khả năng tổng quát. Không trộn thành một con số.

Với mỗi timestep, lưu `sample_id`, live mask, execute mask, selected slot và oracle action. So trước/sau transition trên **cùng execute IDs**. Ngoài bảng live trajectory, giữ terminal đủ cohort gồm cả những mẫu STOP sớm.

Ví dụ: current có 160 mẫu, execute chỉ 153. Không lấy mean loss của 153 mẫu sau edit trừ mean của 160 mẫu trước edit. Phải so 153 cặp trước/sau tương ứng.

Sửa nhãn harmful execution: hiện 119 harmful / 473 decisions = 25.16%; nếu mẫu số là executed actions thì 119 / 469 = **25.37%**. Xuất cả count và denominator.

### 4.2 Đo đúng loại rank

Với feature matrix X, center theo mẫu rồi tính covariance eigenvalues λ. Participation ratio là:

\[
PR=\frac{(\sum_j\lambda_j)^2}{\sum_j\lambda_j^2}.
\]

Nếu phương sai chia đều trên r hướng thì PR=r; nếu một hướng chi phối thì PR tiến về 1. Nó đo độ tập trung phương sai, không trực tiếp đo thông tin ngữ nghĩa hay số chiều khác zero. Với N mẫu, centered empirical rank tối đa `min(N−1,D)`; N=160 không thể yêu cầu rank 768.

Cho mỗi `proposal`, `action`, `delta_q`, lưu:

| Phép đo | Câu hỏi được trả lời |
|---|---|
| Sibling diversity, từng input qua K slot | Một input có nhiều phương án khác nhau không? |
| Per-slot spectrum, từng slot qua B input | Slot đó còn biến thiên theo input không? |
| Pooled spectrum `[BK,D]` | Tổng phương sai sau trộn input và slot như thế nào? |
| Pooled spectrum sau trừ mean riêng từng slot | Khi bỏ offset cố định của slot, còn bao nhiêu biến thiên? |
| Between-slot variance / total variance | Bao nhiêu phương sai chỉ do danh tính slot? |
| Raw và L2-normalized feature spectra | Rank thấp có chủ yếu do norm lớn/nhỏ không? |

Phân rã cần dùng:

\[
\operatorname{Cov}(X)=\mathbb E_k[\operatorname{Cov}(X\mid k)]
+\operatorname{Cov}_k(\mathbb E[X\mid k]).
\]

Ví dụ bốn slot có bốn mean cách xa nhau: phần giữa-slot sau center có tối đa ba chiều. PR gộp quanh 3 vẫn có thể đi cùng per-slot rank cao. Phản ví dụ CPU đã có: pooled PR 3.56 nhưng từng slot PR 45–46. Vì vậy **không thêm anti-collapse loss chỉ từ pooled PR**.

Report `N`, `D`, PR, entropy effective rank, top-PC variance, norm/std, zero-variance count; không trộn PR với entropy rank dưới cùng một tên. Zero effect/zero variance phải có nhãn riêng, không giả làm một representation rank-one bình thường.

### 4.3 Đo đúng representation và toàn bộ các bước

- So initial retrieval query với terminal retrieval query cùng projection/normalization. Không dùng `V0_pooled` 768D làm baseline tương ứng cho query 512D.
- Đo trước/sau từng edit, kể cả committed state cuối; flag hiện chỉ so current đầu/cuối có thể bỏ sót intermediate dip.
- Mean-pool patch-state là một thống kê riêng. Nếu muốn kết luận về full state, đo thêm biến thiên token/patch với cách lấy mẫu và centering ghi rõ.
- Dùng cùng readout và precision policy cho live forward và feature recomputation.
- Drift t2/current là trạng thái vào bước 3, không phải terminal sau edit 3. Drift lớn chỉ là warning; chưa chứng minh off-manifold.

### 4.4 Gắn geometry với retrieval utility

Trên cùng fixed gallery hoặc negative bank, đo paired:

- Loss query→image trước/sau; target rank; Recall tại các k đang dùng.
- Positive similarity và hard-negative margin.
- Utility của chosen, oracle và từng candidate.
- Norm `delta_q`, thay đổi logits trên gallery, overlap/complementarity top-k.

Ví dụ positive similarity 0.60→0.50 nhưng strongest negative 0.59→0.10 thì margin cải thiện 0.01→0.40. Vì vậy không dùng positive similarity đơn lẻ để kết luận edit phá retrieval.

Thử tráo caption giữa samples cùng category, giữ ảnh; so với caption đúng bằng utility/Recall. Chỉ thấy feature thay đổi theo caption chưa chứng minh model hiểu đúng caption. Entity-shuffle là kiểm tra riêng cho binding.

**Điều kiện hoàn thành:** phân biệt được slot offsets, input dependence, transient global concentration và retrieval damage. Nếu chỉ có JSON aggregate hiện tại thì đánh dấu câu hỏi chưa trả lời; không suy ra per-slot covariance từ dữ liệu đã mất trục slot.

## 5. P1 — Sửa ScoreNet/gain/STOP trước

### 5.1 Vì sao ưu tiên?

Gain teacher quy ước:

\[
u_k=\ell(q_t,Y)-\ell(\hat q_k,Y).
\]

`u>0`: edit tốt hơn parent; `u<0`: edit làm hại. STOP có gain=0. Score dùng để STOP nên phải giữ được mốc 0, ngoài việc xếp hạng candidate.

Pair loss hiện muốn giãn cách score theo thứ tự, trong khi gain loss muốn score khớp utility. Phản ví dụ đã kiểm chứng: utilities `[-0.06,-0.06,-0.04,-0.04]` đều xấu, nhưng nghiệm objective hiện có thể cho scores khoảng `[-0.577,-0.577,+0.477,+0.477]`, dẫn tới EXECUTE. Đây là lý do cụ thể để ablate pair loss.

### 5.2 Thực hiện

1. Trên TRAIN, thu scorer inputs đã detach và teacher utilities từ checkpoint cố định; đóng băng toàn bộ phần còn lại, train/refit riêng scorer với `lambda_pair=0` và gain regression giữ nguyên.
2. Cache tạo từ rollout gốc chỉ là chẩn đoán offline. Sau refit, phải chạy live rollout lại vì selector mới làm thay đổi những states được ghé thăm.
3. Đánh giá trên dữ liệu held-out bằng cùng gallery/teacher definition. Không fit scorer trên validation rồi dùng chính validation đó làm bằng chứng tăng benchmark không thiên lệch.
4. Đo STOP precision/recall, confusion matrix, harmful execution, selected utility, regret so oracle, score–utility bias theo bins và từng timestep. Không chỉ đo action accuracy.
5. Nếu cần threshold khác 0, chọn trên calibration/dev split cố định; báo riêng threshold tuning và weight training. Tránh batch-center utility rồi tiếp tục xem 0 là STOP.

Scorer-only refit cần workflow mới; `lambda_pair=0` trong trainer hiện tại chỉ là thay objective của một run training, không tự đóng băng generator.

### 5.3 Nếu vẫn cần pair objective

Dùng soft target phản ánh khoảng cách utility:

\[
p^*_{ij}=\operatorname{sg}\!\left[\sigma\left(\frac{u_i-u_j}{T_p}\right)\right],
\qquad
L_{softpair}=\operatorname{BCEWithLogits}\left(\frac{s_i-s_j}{T_p},p^*_{ij}\right).
\]

Khi `s=u`, soft-pair và gain regression có thể cùng đạt nghiệm phù hợp; gain regression neo offset và mốc 0. Không tự bảo đảm generalization/calibration, vẫn phải đo held-out. `sg` nghĩa dừng gradient qua teacher.

Dropout=0 của scorer là ablation riêng, không bật cùng lần thay pair nếu muốn biết cái gì tạo cải thiện. Scorer đa dạng quyết định do dropout không đồng nghĩa candidates thực sự khác nhau.

**Điều kiện chuyển tiếp:** kiểm chứng synthetic all-negative không còn bị objective khuyến khích score dương; trên held-out, so calibration/utility/Recall với mốc gốc. STOP nhiều hơn tự nó không phải thành công nếu dừng quá sớm.

## 6. P1 — Cho candidate chưa được chọn tín hiệu học chất lượng

### 6.1 Vấn đề bản chất

Hard rollout chọn một candidate để tiếp tục; terminal task loss chủ yếu dạy đường đã chọn. ScoreNet học chấm preview bằng teacher detached. DPP khuyến khích khác biệt nhưng quality cũng detached. Vì vậy “chấm điểm cả bốn” không tương đương “dạy cả bốn tạo edit hữu ích”.

Giữ hard preview–score–commit hiện tại; chưa cần ST/Gumbel/RL. Thêm loss trực tiếp trên live candidate queries ở objective.

### 6.2 Phương án đầu: loss hạn chế candidate gây hại

Với cùng positives/negatives/masks cho parent và mọi sibling:

\[
\ell_k=\ell_{ret}(\hat q_k,Y),\quad b=\operatorname{sg}[\ell_{ret}(q_t,Y)],
\qquad L_{safe}=\mathbb E_{i,t,k}\max(0,\ell_k-b).
\]

Ví dụ parent CE=3.0; candidate CE `[2.8,3.1,3.3,2.7]` nhận penalty `[0,0.1,0.3,0]`. Hai candidate gây hại có gradient để giảm CE; hai candidate đã tốt hơn không bị ép đạt target mạnh thêm bằng loss này.

Yêu cầu implementation trong `src/losses/objective.py` và helper retrieval:

- Candidate query phải giữ graph tới Proposer/Fusion/Executor/text. Không lấy utility từ hàm teacher `no_grad()` rồi dùng làm upstream loss.
- Parent baseline detach; bank targets frozen. Candidate vẫn phụ thuộc live parent/shared weights, nên baseline detach không bảo đảm parent bất biến sau optimizer update.
- Lọc rows không có positive hoặc negative hợp lệ trước CE/logsumexp. Không tính NaN rồi nhân mask=0. Khi không có valid row, trả differentiable zero.
- Tính trên tất cả valid live parent rows và K previews, gồm rows scorer muốn STOP; chọn denominator rõ ràng. Terminal loss vẫn giữ.
- Log selected/unselected query gradient norms, candidate utility distribution và zero-effect rate.

**Giới hạn:** no-op có `L_safe=0`; loss này không tự buộc sinh candidate hữu ích. Nếu harmful giảm nhưng mọi `delta_q` tiến về 0 và best-of-K gain cũng mất, không gọi đó là thành công.

No-harm còn tạo thiên hướng tham lam: một bước tạm xấu nhưng mở đường tốt về sau sẽ bị phạt. Khởi đầu kiểm tra ở T1; khi nâng T phải so terminal trade-off. Không áp margin dương cứng cho mọi state, nhất là khi nên STOP.

### 6.3 Phương án thứ hai, có điều kiện: candidate CE bootstrap

Nếu candidate starvation/no-op vẫn nặng, thử tại T1:

\[
w_k=\operatorname{sg}[\operatorname{softmax}_k(-\ell_k/T_a)],
\qquad L_{cand}=\mathbb E_{i,t}\sum_k w_k\ell_k.
\]

Temperature cao cho nhiều slot học, hạ dần tập trung vào candidate tốt. Weight/temperature schedule là hyperparameter mới, chưa có giá trị tối ưu xác nhận. Không bật đồng thời L_safe, L_cand và thay DPP trong lần đầu.

Loss này hướng nhiều candidate về final target; có thể làm chúng giống nhau và đổi ý nghĩa “atomic partial edit”. Chỉ dùng như nhánh bootstrap/benchmark được ghi rõ, không tuyên bố tự học decomposition đúng.

**Điều kiện hoàn thành:** upstream gradient thật đi qua candidate chưa chọn; harmful giảm mà effect không biến mất; oracle/best-of-K utility hoặc Recall có cải thiện. Tiêu chí cuối vẫn là retrieval held-out bằng selector thật.

## 7. P1 — Sửa diversity sau khi có quality signal

### 7.1 Useful-subset DPP

Hiện tại một row qua gate khi có ít nhất hai useful candidates, rồi full K vẫn đi vào DPP. Thay bằng tập:

\[
\mathcal U_{i,t}=\{k:q(u_{i,t,k})>q_{min}\}.
\]

Chỉ xây candidate kernel từ indices trong tập này; nếu dưới hai candidate thì sibling-diversity contribution bằng 0. Giữ history detached và công thức conditional history nhất quán với ablation; không đổi history policy trong cùng lần thay subset.

Với tham số hiện tại, useful gate tương ứng utility lớn hơn khoảng **0.020067**, không đồng nghĩa utility>0. STRONG khoảng 0.99 useful/parent nhưng khoảng 1.81 positive/parent; phải report hai số riêng.

Yêu cầu:

- Giữ gate/quality detached, không cho model tăng teacher utility qua đường giả.
- Ghi cách chuẩn hóa: hiện DPP average theo valid rows; valid fraction thấp không tự làm loss nhỏ tương ứng. Subset size thay đổi cũng làm logdet scale thay đổi; log loss theo subset size. Nếu đổi normalization thì đặt thành ablation riêng.
- Linalg/logdet trong autocast-disabled FP32; `.float()` đơn lẻ không đủ vô hiệu autocast cho mọi phép toán.
- Theo dõi nonpositive determinant/nonfinite và numerical-invalid count riêng. Không báo row “valid” như một diversity update thành công nếu số học fail rồi trả zero.
- Kiểm tra gradient trực tiếp theo candidate effect: harmful effect bị loại không nhận DPP effect gradient; shared weights vẫn có thể thay đổi nó gián tiếp.

Useful-subset DPP không dạy candidate ngoài tập trở nên tốt; phải đi cùng quality learning ở bước trước. Exact clones cũng có thể có diversity gradient bằng 0; không chữa được bằng việc chỉ tăng lambda.

### 7.2 Auxiliary ablation

Giữ concept/bind/rel cố định khi đổi DPP. Sau khi có mốc tốt, thêm hoặc bỏ từng auxiliary riêng. Kiểm tra gradient norm và gradient cosine với terminal loss trên cùng batch/module, không suy “loss lớn nhất = gradient chi phối”.

Relation assignment entropy cao có thể là mọi input nhận phân phối uniform. Đo cả conditional entropy, entropy của marginal và chênh lệch hai đại lượng; entity-shuffle/retrieval kiểm tra tiếp xem binding có ích không. Concept MIL loss thấp cũng không chứng minh slots khác nhau.

**Điều kiện giữ auxiliary:** cải thiện chất lượng hoặc khả năng grounding có kiểm chứng, với benchmark không bị đánh đổi ngoài mục tiêu đã chấp nhận. Không giữ chỉ vì diversity/entropy đẹp.

## 8. P2 — Baseline bắt buộc và tối ưu phù hợp text-only

### 8.1 Native image + native text

\[
v=\operatorname{Norm}(E_v(x_r)),\quad
t=\operatorname{Norm}(E_t^{native}(x_m)),\quad
q=\operatorname{Norm}(v+t).
\]

Trước tiên frozen baseline image-only, text-only, sum. Sau đó fine-tune text với vision frozen; dùng đúng native pooling/projection/token-position policy của FG-CLIP, không thay bằng mean của random adapter hiện tại. So cùng dataset, preprocessing và gallery.

Đây là baseline ngoài đường forward canonical IAG-SRME. Nếu thêm direct text shortcut vào model chính, phải đặt experiment/spec variant rõ ràng. Không âm thầm thay kiến trúc rồi coi kết quả chứng minh pipeline cũ.

### 8.2 Editor tối giản K1/T1

Chạy một candidate, một bước, STOP off, retrieval-only. Mục đích: Proposer/Grounder/Executor/readout có học được retrieval không khi bỏ bài toán lựa chọn và nhiều auxiliaries? Đây là structural control, không phải ablation một biến.

Với cold start, kiểm tra zero-init: update đầu có thể chỉ mở output projection, gradient action đến sau. Đừng kết luận toàn model chết từ một backward đầu; cũng đừng để STOP t0 chặn toàn đường học. Hai checkpoint đã train hiện tại có effects, nên cold-start risk không thay thế nguyên nhân đang quan sát.

### 8.3 Tách learning rate

Pretrained text và module mới có nhu cầu cập nhật khác nhau. Đề xuất điểm bắt đầu để thử, không phải cấu hình tối ưu đã xác nhận:

| Parameter group | LR thử ban đầu |
|---|---:|
| Pretrained text | 5e-6 hoặc 1e-5 |
| Module mới / objective-owned trainable parameters | 1e-4 |
| Module mới nếu còn underfit và ổn định | 3e-4, ở run riêng |

Warmup/schedule theo optimizer updates. Kiểm tra mỗi trainable parameter có đúng một group; objective-owned parameters không bị bỏ sót; frozen vision không vào optimizer. Log update-to-weight ratio. Warm-start checkpoint đã học có thể cần LR nhỏ hơn fresh modules.

### 8.4 Frozen TRAIN image bank

Cache native final image features vì vision/projection frozen. Cache phải ghi image IDs, backbone revision, preprocessing, normalization, dtype và split. Đây là target/negative bank, không phải cache thay thế graph của edited-state readout.

Thứ tự thử: in-batch → random negatives từ TRAIN bank → thêm semi-hard cùng category. Một thí nghiệm minh họa có thể dùng 1,024 negatives/query rồi tăng nếu VRAM cho phép; đây là điểm thử, không phải yêu cầu hay tối ưu đã biết.

Các điều kiện bắt buộc:

- Parent và K siblings dùng đúng cùng positive set, negative set và masks trong một decision.
- Giữ multi-positive theo target IDs; loại known positives khỏi negatives. Khác ID không bảo đảm không phải false negative.
- Hard-negative mining không dùng validation/test target labels để train.
- Khi teacher bank thay đổi, utility scale và STOP calibration cũng thay đổi; cần refit/recalibrate scorer tương ứng. Không trộn teacher từ hai bank rồi xem cùng thang điểm.
- Memory bank chỉ là phần features. Logits/gradient có thể lớn hơn; chunk hoặc sampled bank phải giữ đúng global logsumexp/CE nếu muốn tương đương full bank, không cộng trung bình CE từng chunk như thể tương đương.
- Gradient accumulation không mở rộng negative pool nếu mỗi microbatch tính loss riêng.

Ví dụ 50,000 ảnh × 512 chiều × FP16 ≈ 48.8 MiB cho features, chưa tính logits và graph. Đây là tính dung lượng minh họa, không phải số ảnh đã xác nhận của run.

## 9. P3 — Chỉ mở rộng recurrence sau khi one-step tốt

So T1/T2/T3 cùng các điều kiện còn lại; phân biệt hai loại thí nghiệm:

1. Evaluate cùng checkpoint với horizon khác: chẩn đoán tác động policy, có thể lệch phân phối so training.
2. Train matched experiments với từng horizon: đánh giá giá trị kiến trúc, tốn hơn nhưng kết luận mạnh hơn.

Đo lợi ích biên từng bước bằng paired retrieval, số steps thực thi, latency/VRAM và terminal Recall. Ablate history-DPP độc lập với sibling DPP. Lặp cùng hướng đôi khi hữu ích; khác history tối đa không luôn là điều tốt.

CIR hiện có reference + một instruction + target cuối, không có ground-truth intermediate states/actions. Vì vậy nhiều bước nội bộ không tự trở thành robot planning có ý nghĩa. Ví dụ “đổi màu đỏ và bỏ tay áo” có thể giải một bước hoặc hai bước; terminal retrieval loss chấp nhận cả hai nếu target đúng.

Chỉ xem xét continuation value/lookahead khi đo được nhiều trường hợp one-step gain âm nhưng continuation cải thiện terminal. Oracle/lookahead có dùng target chỉ là diagnostic trong train/eval phân tích, không được đưa target vào inference hoặc báo oracle Recall như benchmark thực tế.

Nếu baseline native sum tốt nhưng editor chưa đóng góp, có thể thử `Norm(q_base + η Δq_local)` với `Δq_local = q_edited − q_initial` nhất quán. Đây là variant mới; bắt buộc so base-only/local-only/combined. Không dùng kết quả combined để che việc local branch chưa có ích.

## 10. Ma trận chạy tối thiểu và quy tắc ra quyết định

| ID đề xuất | Thí nghiệm | So với | Quyết định sau run |
|---|---|---|---|
| D0 | Replay OLD/STRONG, diagnostic đã sửa | Artifacts gốc | Chốt chất lượng số liệu và loại concentration |
| S1 | Scorer-only gain refit | Cùng generator checkpoint | Nếu calibration tốt hơn, evaluate live rollout; nếu Recall không tăng, đo quality/headroom |
| Q1 | Thêm L_safe | Cùng parent run, chỉ thêm loss này | Giữ nếu bớt harmful và không thành no-op |
| Q2, có điều kiện | Candidate CE bootstrap T1 | T1 control tương ứng | Chỉ thử nếu starvation/no-op còn nặng |
| D1 | Useful-subset DPP | Q1 với full-K DPP | Giữ nếu quality/diversity trade-off tốt hơn |
| B0/B1 | Frozen native sum / text-finetuned sum | Cùng evaluator | Xác định baseline text-only thực dụng |
| B2 | Editor K1/T1 retrieval-only | Full architecture và B1 | Nếu đã yếu, ưu tiên đường text→edit→readout |
| O1 | Parameter-group LR | Cấu hình tốt nhất cố định | Chọn bằng validation, không bằng tốc độ giảm total loss |
| N1 | Random TRAIN bank | O1/in-batch | Giữ nếu retrieval gain lặp lại; refit teacher/scorer đúng bank |
| H1 | Horizon và history ablation | Mốc one-step tốt | Chỉ giữ recurrence nếu có lợi ích biên |

Không bắt buộc chạy Q1 từ STRONG và OLD đầy đủ cùng lúc. Có thể dùng STRONG để đo rescue ngắn, còn OLD làm mốc recovery; báo rõ khác biệt và chỉ so causal khi warm-start/budget/config khớp.

Đối với lựa chọn mô hình: cố định screening budget trước khi chạy, ví dụ một số optimizer updates chung. Run ngắn chỉ sàng lọc lỗi/xu hướng, không đủ kết luận cấu hình nào đạt best cuối. Train đủ ngân sách cho ứng viên tốt; nếu có điều kiện chạy ít nhất 3 seed và báo mean/std. Không diễn giải chênh lệch nhỏ một seed là cải thiện chắc chắn.

Mọi benchmark chính dùng selector thật, cùng protocol/checkpoint-selection rule, report R@10/R@50 từng category và mean recall. Ghi thêm số steps/compute. Không chọn cấu hình theo test rồi báo test như phép đánh giá độc lập.

## 11. Phần nào đã bật được, phần nào cần viết code?

### 11.1 Lệnh config hiện có

Mẫu sau dựa trên schema snapshot đã audit. Chạy từ repo và cấu hình dataset/runtime đúng máy. Đây là **fresh-run**, chưa được chạy trong tài liệu này; không phải scorer-only refit hoặc exact resume. Có thể thêm `--cfg job --resolve` để xem config trước khi train.

Gain-only so với bộ auxiliary OLD:

```bash
python src/train.py \
  backbone=fgclip_base_text_native_cls \
  objective.lambda_c=0.01 \
  objective.lambda_bind=0.01 \
  objective.lambda_rel=0.001 \
  objective.lambda_dpp=0.01 \
  objective.lambda_pair=0 \
  hydra.run.dir=outputs/action_plan_gain_only_old_aux
```

Editor structural control:

```bash
python src/train.py \
  backbone=fgclip_base_text_native_cls \
  model.num_candidates=1 \
  model.max_steps=1 \
  model.stop_enabled=false \
  model.score_dropout=0 \
  objective.lambda_pair=0 \
  objective.lambda_gain=0 \
  objective.concept_enabled=false \
  objective.bind_enabled=false \
  objective.rel_ortho_enabled=false \
  objective.dpp_enabled=false \
  hydra.run.dir=outputs/action_plan_k1_t1_retrieval_only
```

### 11.2 Cần implementation mới

| Hạng mục | Nơi bắt đầu xem | Tiêu chí xác nhận |
|---|---|---|
| Per-slot/matched diagnostic | `src/diagnose_latent_geometry.py` và selector diagnostic | Synthetic slot-offset case phân biệt được pooled/per-slot; sample IDs khớp |
| Config-aware evaluation | `src/evaluate.py`, model construction | Reject/ghi rõ mismatch hành vi |
| Scorer-only refit / soft pair | `src/losses/objective.py`, workflow train mới | Generator frozen; chỉ scorer thay đổi; soft-pair khớp utility |
| L_safe / L_cand | `src/losses/objective.py`, `src/losses/retrieval.py` | Unselected preview có task gradient; teacher/target detached |
| Useful-subset DPP | Hàm DPP trong `src/losses/objective.py` | Excluded effect không nhận direct DPP gradient; invalid numerical được log |
| LR groups, scheduler, JSONL, resume | `src/training/engine.py`, `src/train.py` | Parameter groups đầy đủ, actual updates và resume được kiểm tra |
| Native sum/Combiner baseline | Backbone utility và experiment riêng | Native text projection thực sự dùng; vision frozen |
| TRAIN negative bank | Data/cache + retrieval loss + teacher | Shared IDs/masks; không false-positive-as-negative đã biết |

Không thêm tên flag vào YAML rồi giả định code đã xử lý. Tách helper tính loss, teacher, bank và diagnostic theo trách nhiệm; tái dùng retrieval/mask logic để tránh teacher và live loss lệch nhau.

## 12. Kiểm chứng tối thiểu trước run tốn GPU

Các test cần thiết tập trung vào rủi ro toán/gradient, không cần tạo bộ test lớn cho việc đổi tên/log đơn giản.

1. **Score sign:** utilities đều âm; objective mới không khuyến khích crossing-zero để chỉ cải thiện pair ranking. Với soft pair, kiểm tra gradient tại `s=u`.
2. **Candidate gradient:** một unselected candidate gây hại nhận live gradient; detached parent baseline không nhận gradient trực tiếp từ baseline term; teacher và frozen target không bị train.
3. **Mask correctness:** duplicate positives, không có negative hợp lệ, batch/subset rỗng không tạo NaN.
4. **DPP subset:** một useful candidate skip đúng; hai useful + harmful siblings chỉ useful effects nhận direct diversity gradient; FP32/AMP parity trong sai số đã chấp nhận.
5. **Geometry:** Gaussian features cộng slot offsets tạo pooled PR thấp nhưng per-slot PR cao; flags không kết luận input collapse chỉ từ pooled PR.
6. **Smoke update:** vài optimizer updates có parameter delta thực, text và module mới có đường học như mong đợi; vision giữ frozen nhưng live readout vẫn truyền gradient vào edited states.

Sau đó chạy matched diagnostics. Chỉ mở full training khi đã biết loss mới tác động đúng tensor và metric đang đo đúng câu hỏi.

## 13. Cách đọc kết quả để biết sửa tiếp ở đâu

| Kết quả quan sát | Ưu tiên tiếp |
|---|---|
| Pooled PR thấp nhưng slot-centered/per-slot rank tốt | Bỏ kết luận input collapse; tập trung quality và selection |
| Per-slot variation thấp, caption đúng không hơn caption tráo | Probe text→proposal→effect, giảm auxiliary pressure, candidate-quality training |
| Per-slot rank tốt nhưng paired first-edit utility xấu | Sửa hướng/độ lớn edit; thử residual-step scale ở ablation riêng |
| Candidate oracle tốt, selected utility thấp | Scorer calibration/feature sufficiency; kiểm tra on-policy distribution |
| Oracle cũng thấp | Generator/Executor/representation, không chỉ selector |
| L_safe giảm, effects gần zero, best-of-K không tăng | No-op escape; cần bootstrap hoặc terminal quality path mạnh hơn |
| T3 kém T1 | Horizon/history/STOP; giữ T1 làm mốc, chưa thêm RL |
| Sum baseline vượt editor | Kiểm tra text route/readout/optimization; độ phức tạp chưa có lợi ích |
| Rank/diversity tăng nhưng Recall giảm | Không giữ chỉ vì diagnostic đẹp; giảm hoặc bỏ thay đổi đó |

Không có ngưỡng PR, cosine hoặc occupancy duy nhất xác nhận “khỏe”. Cần đồng thời xem geometry, input sensitivity, candidate quality, selector regret và benchmark.

## 14. Checklist checkpoint để tiếp tục công việc

- [ ] Đã lưu resolved configs và giữ nguyên OLD/STRONG checkpoints.
- [ ] Replay cùng evaluator/protocol, metric lệch được giải thích.
- [ ] Diagnostic dùng cùng sample IDs, cùng gallery, paired before/after. **Code + manifest đã triển khai; OLD/STRONG replay chưa chạy vì checkpoint không có trong workspace hiện tại.**
- [x] Có per-slot và slot-centered spectra; sửa representation mismatch/flags. **Đã kiểm chứng bằng synthetic slot-offset/collapse/covariance tests.**
- [x] Có JSONL loss/quality/STOP/optimizer updates đầy đủ. **Đã kiểm chứng một CPU optimizer update; CUDA AMP smoke còn pending.**
- [ ] Đã kiểm tra gain-only scorer và live rollout sau refit.
- [ ] Đã kiểm chứng live candidate-quality gradient, không dùng detached teacher làm loss upstream.
- [ ] Đã thử L_safe và kiểm tra no-op escape.
- [ ] Đã so useful-subset DPP với full-K trong điều kiện matched.
- [ ] Đã chạy native sum và K1/T1 controls.
- [ ] Đã thử LR groups và TRAIN bank trên cấu hình tốt nhất.
- [ ] Chỉ mở K/T/history/auxiliary tiếp khi thấy lợi ích.
- [ ] Benchmark được báo bằng selector thật, validation/test protocol rõ, không hứa mức tăng chưa đo.

Mỗi lần hoàn thành một mục, cập nhật ngay trong file này: run ID, commit, config, seed, checkpoint, dữ liệu, kết quả chính, quyết định giữ/bỏ và câu hỏi còn lại. Trạng thái hiện tại: **P0 instrumentation đã được triển khai trên base `c0014786`; unit suite 68 passed/1 CUDA skip. Chưa replay OLD/STRONG và chưa có benchmark mới.**

Mẫu ghi một thí nghiệm:

```text
Run ID:
Giả thuyết:
Commit / parent checkpoint:
Thay đổi duy nhất hoặc nhóm thay đổi có chủ đích:
Split / sample manifest / gallery:
Seed / optimizer updates / compute budget:
Config artifact:
Benchmark R10/R50/category/mean:
Candidate utility / STOP confusion / geometry:
So với control:
Kết luận: giữ / bỏ / chưa đủ bằng chứng
Việc tiếp theo:
```

## 15. Nguồn và phạm vi kế thừa

- [Snapshot repo đã audit](https://github.com/Le-Minh-Nhut/cir/tree/c00147863c34c5871f65d62b76a04a1aa985954a).
- [Geometry diagnostic source](https://github.com/Le-Minh-Nhut/cir/blob/c00147863c34c5871f65d62b76a04a1aa985954a/src/diagnose_latent_geometry.py).
- [OLD geometry JSON](https://github.com/Le-Minh-Nhut/cir/blob/c00147863c34c5871f65d62b76a04a1aa985954a/doc/diagnostics/v2-r0/latent_geometry_old.json).
- [STRONG geometry JSON](https://github.com/Le-Minh-Nhut/cir/blob/c00147863c34c5871f65d62b76a04a1aa985954a/doc/diagnostics/v2-r0/latent_geometry_strong.json).
- [Selector JSON](https://github.com/Le-Minh-Nhut/cir/blob/c00147863c34c5871f65d62b76a04a1aa985954a/doc/diagnostics/v2-r0/candidate_selector_diagnostic.json).
- [Training objective](https://github.com/Le-Minh-Nhut/cir/blob/c00147863c34c5871f65d62b76a04a1aa985954a/src/losses/objective.py).
- Audit chi tiết đã lưu: `CIR_V2_FAILURE_AUDIT_CHECKPOINT_2026-09-07.md`, bản cập nhật gồm phản ví dụ CPU, review source và giới hạn bằng chứng. File hiện tại chuyển các kết luận đó thành thứ tự công việc, không thay đặc tả canonical V2.

Các đường dẫn source trong tài liệu là đường dẫn tương đối từ repo root. Nếu branch có push mới sau snapshot trên, kiểm tra diff trước khi áp dụng; không ghi đè sửa đổi mới của người dùng dựa trên số dòng hoặc conf cũ.