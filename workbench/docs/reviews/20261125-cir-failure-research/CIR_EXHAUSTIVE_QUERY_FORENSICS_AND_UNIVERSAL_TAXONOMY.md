# Báo Cáo Đợt 5 (Tổng Hợp Chuyên Sâu): Phân Rã 4 Góc Phần Tư `(ΔV, ΔT)`, Vực Thẳm Khoảng Cách Chỉnh Sửa, Điểm Mù Kiến Trúc BLIP-2 & Hệ Thống Phân Loại 12 Lỗi Chung Của CIR Hiện Đại

> [!IMPORTANT]
> **Chuẩn kiểm chứng không sai sót (Zero-Error Protocol):**
> - Toàn bộ chỉ số phân rã đơn phương thức trong báo cáo này được tính độc lập bằng bộ mã hóa trung lập đóng băng **`CLIP ViT-L/14 (DataComp-XL-13B)`** trên **`9,107` ảnh duy nhất** và **`3,630` câu truy vấn duy nhất** trải đều cả 3 benchmark (`1,003` EDIR, `1,337` PinPoint, `1,304` FashionIQ `ALL13 Fail@10`).
> - Kết hợp **kiểm toán hình ảnh trực tiếp bằng mắt thêm `69` ca mới hoàn toàn** (`24` EDIR + `21` PinPoint + `24` FashionIQ, **`0` ca trùng lặp** với `66` ca ở Đợt 3 → **tổng cộng `135` truy vấn thất bại độc lập đã được soi trực tiếp từng ảnh**: `42` EDIR, `39` PinPoint, `54` FashionIQ).
> - Mọi số liệu, phần trăm, đếm số lượng và ID truy vấn đều đã được đối chiếu tự động bằng script với kết quả thực nghiệm gốc.

---

## 1. Phân Rã 4 Góc Phần Tư `(ΔV, ΔT)` — Chẩn Đoán Toán Học Cho Mọi Cú Trượt CIR

Để biết chính xác **TẠI SAO** mô hình CIR lại xếp một ảnh sai (`Wrong`) cao hơn ảnh đúng (`Target`), ta dùng một trọng tài độc lập không tham gia huấn luyện CIR (**frozen `CLIP ViT-L/14 DataComp-XL`**) đo 2 độ lệch đơn phương thức cho từng truy vấn thất bại:
- **Độ lệch Thị giác:** $\Delta V = \cos(I_{\text{ref}}, I_{\text{wrong}}) - \cos(I_{\text{ref}}, I_{\text{target}})$ — nếu $\Delta V > 0$, ảnh sai giống ảnh gốc hơn ảnh đích.
- **Độ lệch Văn bản:** $\Delta T = \cos(T, I_{\text{wrong}}) - \cos(T, I_{\text{target}})$ — nếu $\Delta T > 0$, ảnh sai khớp văn bản thô hơn ảnh đích.

Từ dấu của $(\Delta V, \Delta T)$, **mọi lỗi truy hồi CIR trên đời đều rơi vào đúng 1 trong 4 cơ chế trực giao**:
1. **`Q1` — Bẫy Neo Thị Giác (`Visual-Anchor Trap`: $\Delta V > 0, \Delta T \le 0$):** Ảnh sai thắng chỉ vì giống ảnh gốc hơn, dù khớp văn bản kém hơn ảnh đích.
2. **`Q2` — Chiếm Quyền Văn Bản / Xóa Ảnh Gốc (`Text-Hijack / Visual-Erasure`: $\Delta V \le 0, \Delta T > 0$):** Ảnh sai khớp từ khóa văn bản hơn nhưng rời bỏ bản sắc ảnh gốc.
3. **`Q3` — Bẫy Kép (`Double-Trap`: $\Delta V > 0, \Delta T > 0$):** Ảnh sai thắng ảnh đích trên **CẢ HAI** trục đơn phương thức (đặc trưng của *Hiệu ứng Con Voi Hồng* trong phủ định — ảnh sai vừa giống ảnh gốc vừa chứa đúng danh từ bị cấm — hoặc *False Negative / Nhãn GT lỗi*).
4. **`Q4` — Sụp Đổ Bộ Trộn Thuần Túy (`Pure Fusion Collapse`: $\Delta V \le 0, \Delta T \le 0$):** Ảnh đích vốn đã thắng ảnh sai trên **CẢ HAI** trục đơn phương thức của CLIP, nhưng chính tầng fusion của mô hình CIR lại làm nhiễu và xếp hạng ngược!

### 1.1. Kết Quả 4 Góc Phần Tư Trên Cả 3 Benchmark

| Tập truy vấn thất bại | Số ca (`n`) | **`Q1` Visual-Anchor** ($\Delta V>0, \Delta T\le 0$) | **`Q2` Text-Hijack** ($\Delta V\le 0, \Delta T>0$) | **`Q3` Double-Trap** ($\Delta V>0, \Delta T>0$) | **`Q4` Pure Fusion** ($\Delta V\le 0, \Delta T\le 0$) | Cơ chế chi phối |
|---|---:|---:|---:|---:|---:|---|
| **EDIR Majority-Fail** (BLIP-2 đúng `≤4/10`) | 717 | **`51.6%`** (`370`) | `13.4%` (`96`) | `21.5%` (`154`) | **`13.5%` (`97`)** | **Q1 thống trị (>50%)**, kèm `13.5%` hỏng thuần tại tầng fusion Q-Former! |
| **EDIR `All-42-Failed`** (0/42 runs đúng R@1) | 95 | **`44.2%`** (`42`) | `6.3%` (`6`) | **`49.5%`** (`47`) | `0.0%` (`0`) | **`93.7%` (`89/95`) đều có $\Delta V > 0$** (ảnh sai gần `Ref` hơn GT!) |
| **PinPoint Majority-Fail** (BLIP-2 Hit@10 `≤2/10`, xét *Non-Ref Wrong* vs *GT Pos*) | 288 | `24.0%` (`69`) | `18.4%` (`53`) | **`50.0%`** (`144`) | `7.6%` (`22`) | **Q3 chiếm 50%** (cao nhất **`69.6%` ở `Negation`**!); **`Context Fit`** nặng Q1 (`37.9%`) |
| **FashionIQ `ALL13 Fail@10`** (tất cả `13` modern runs trượt R@10) | 1,304 | `25.1%` | `15.8%` | **`54.5%`** | `4.6%` | **Q3 chiếm `54.5%`** (phản ánh lượng lớn FN & nhãn nhiễu), Q1 chiếm `25.1%` |

### 1.2. Phân Rã 4 Góc Phần Tư Chi Tiết Theo Từng Phép Sửa (EDIR) & Nhóm Truy Vấn (PinPoint)

- **EDIR (`15` phép sửa, tập Majority-Fail `n=717`):**
  - Nhóm bị **`Q1 Visual-Anchor Trap` thống trị `>60%`**: `addition` (`62.2%`), `time` (`62.2%`), `color` (`61.8%`), `weather` (`61.1%`), `viewpoint` (`60.9%`), `replace` (`59.5%`), `cardinality` (`57.1%`), `texture` (`56.8%`).
  - Nhóm duy nhất bị **`Q3 Double-Trap` thống trị**: **`remove` (`41.2%` Q3 + `27.5%` Q2)** — khẳng định toán học rằng khi nhắc vật thể cần xóa, ảnh sai (vẫn giữ vật thể đó) được thưởng điểm dương trên cả trục thị giác lẫn trục văn bản thô!
  - Nhóm có tỷ lệ **`Q4 Pure Fusion Collapse` cao nhất**: `addition` (`27.0%`), `style` (`22.0%`), `material` (`18.0%`), `action` (`17.6%`), `complex` (`16.4%`), `shape` (`15.2%`). Ví dụ soi trực tiếp trên ảnh (`r5_edir_4`): **`qid=1201`** (*"thin, decorative metal ring around the edge"*, `dV=-0.01, dT=-0.04` — CLIP đơn phương thức chọn đúng GT có viền kim loại, nhưng **`10/10` BLIP-2** đều chọn đèn trắng trơn có 4 kẹp đen!); **`qid=617`** (*"phone with different shape"*, `dV=-0.12, dT=-0.03` — `7/10` BLIP-2 chọn ảnh phóng to cắt cụt thay vì điện thoại viền bo tròn!); **`qid=63`** (*"don't want trees and foliage to be green"*, `dV=-0.03, dT=-0.03` — `6/10` BLIP-2 vẫn chọn hàng cây xanh!).
- **PinPoint (`5` nhóm truy vấn, tập Majority-Fail `n=288`, Non-Ref Wrong vs GT Pos):**
  - `Negation` (`n=23`, `V(Ref,Pos)=0.626`): **`Q3 = 69.6%`**, `Q4 = 21.7%`, `Q1 = 8.7%`, `Q2 = 0.0%`.
  - `Complement` (`n=23`, `V(Ref,Pos)=0.521`): **`Q3 = 56.5%`**, `Q2 = 26.1%`, `Q1 = 17.4%`.
  - `Swap` (`n=128`, `V(Ref,Pos)=0.656`): **`Q3 = 55.5%`**, `Q2 = 22.7%`, `Q1 = 17.2%`, `Q4 = 4.7%`.
  - `Explore` (`n=11`, `V(Ref,Pos)=0.598`): **`Q3 = 54.5%`**, `Q1 = 18.2%`, `Q4 = 18.2%`.
  - `Context Fit` (`n=103`, `V(Ref,Pos)=0.518` — thấp nhất!): **`Q1 = 37.9%`** (cao gấp đôi mọi nhóm khác!), `Q3 = 36.9%`, `Q2 = 16.5%`.
- **FashionIQ (`5` dải hạng trên `1,304` ca `ALL13 Fail@10`):**

| Dải hạng trung vị (`med_rk`) | `n` | `V(Ref,Tgt)` | `T(Txt,Tgt)` | `HasCol%` | `CmpNoCol%` | `Neg%` | `Short ≤6w%` | `Antonym%` | `Q1` | `Q2` | `Q3` | `Q4` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`11–25` (Near-Miss)** | 10 | `0.721` | `0.180` | `100.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `10.0%` | **`90.0%`** | `0.0%` |
| **`26–50` (Close-Miss)** | 129 | `0.635` | `0.179` | `73.6%` | `24.8%` | `7.0%` | `7.8%` | `1.6%` | `23.3%` | `18.6%` | `53.5%` | `4.7%` |
| **`51–150` (Mid-Miss)** | 443 | `0.620` | `0.167` | `68.4%` | `27.3%` | `13.5%` | `9.0%` | `1.6%` | `26.9%` | `13.5%` | `53.3%` | `6.3%` |
| **`151–500` (Far-Miss)** | 450 | `0.592` | `0.154` | `55.3%` | `39.8%` | `13.8%` | `12.4%` | `3.1%` | `26.7%` | `19.1%` | `49.3%` | `4.9%` |
| **`>500` (Extreme-Miss)** | 272 | **`0.557`** | **`0.128`** | **`42.6%`** | **`53.7%`** | **`14.7%`** | **`14.3%`** | **`4.4%`** | `21.3%` | `12.9%` | **`64.3%`** | `1.5%` |

→ Nhìn ngang toàn bảng thấy **quy luật đơn điệu hoàn hảo**: càng trôi về đuôi `>500`, `V(Ref, Tgt)` rơi thẳng từ `0.721 → 0.557`, `T(Txt, Tgt)` rơi từ `0.180 → 0.128` (chứng minh caption và GT lệch nhau dữ dội do nhãn lỗi/đảo ngược), trong khi `CmpNoCol%` vọt từ `0.0% → 53.7%` và `Q3` vọt lên `64.3%`!

---

## 2. Định Luật "Vực Thẳm Khoảng Cách Chỉnh Sửa Thị Giác" (`Monotonic Visual Edit-Distance Cliff`)

Đo khoảng cách biến đổi thị giác thuần túy $d_{\text{edit}} = 1 - \cos_{\text{CLIP}}(I_{\text{ref}}, I_{\text{target}})$ trên toàn bộ **`1,003` truy vấn EDIR** và **`1,337` truy vấn PinPoint** hé lộ một **quy luật sụp đổ đơn điệu tuyệt đối** trên mọi kiến trúc:

| **EDIR (`1,003` queries)** | `n` | BLIP-2 ID `R@1` | CLIP ID `R@1` | **`All-42-Failed`** | **PinPoint (`1,337` queries)** | `n` | BLIP-2 `Hit@10` | CLIP `Hit@10` | **`All-31-Zero@10`** |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `1_TinyEdit` ($d < 0.12$) | 161 | **`36.96%`** | `31.21%` | **`3.1%`** | `1_CloseEdit` ($d < 0.25$) | 200 | **`82.0%`** | `80.8%` | **`0.0%` (`0/200`!)** |
| `2_SmallEdit` ($0.12–0.22$) | 307 | `35.54%` | `24.84%` | `5.5%` | `2_ModEdit` ($0.25–0.35$) | 256 | `76.5%` | `70.0%` | `4.3%` |
| `3_MediumEdit` ($0.22–0.35$) | 319 | `30.97%` | `19.67%` | `8.2%` | `3_LargeEdit` ($0.35–0.45$) | 342 | `73.0%` | `63.6%` | `5.3%` |
| `4_LargeEdit` ($0.35–0.50$) | 161 | **`16.15%`** | `10.56%` | `16.8%` | `4_FarLeap` ($0.45–0.55$) | 329 | `60.2%` | `51.4%` | `11.9%` |
| **`5_DrasticEdit` ($d \ge 0.50$)** | 55 | **`12.00%`** | **`7.27%`** | **`36.4%` (`11.7×`!)** | **`5_CrossManifold` ($d \ge 0.55$)** | 210 | **`47.0%`** | **`34.3%`** | **`19.5%`** |

> [!WARNING]
> **Quy luật bản chất của CIR hiện đại:**
> - Trên **PinPoint**, ở vùng chỉnh sửa gần ($d < 0.25$, `200` truy vấn), **KHÔNG CÓ BẤT KỲ CA THẤT BẠI PHỔ QUÁT NÀO (`0 / 200 = 0.0%`)**! Toàn bộ `100%` lỗi phổ quát của PinPoint xảy ra khi ảnh đích vượt khỏi bán kính thị giác $d \ge 0.25$, và bùng phát tới `19.5%` ở vùng $d \ge 0.55$.
> - Trên **EDIR**, ngay khi phép chỉnh sửa kéo theo độ dịch chuyển thị giác lớn ($d \ge 0.50$, `55` ca), **`R@1` của BLIP-2 rơi xuống `12.00%` (dưới cả đoán mò `16.67%`!)** và tỷ lệ trượt cả 42 runs tăng gấp **`11.7×`** (`3.1% → 36.4%`).
> - Kiểm toán ảnh trên `r5_edir_1` & `r5_edir_2` (nhóm `0/16` trượt cả `R@5`!) phơi bày đúng cơ chế này: **`qid=1982`** (*"Replace the bottle with a metal can"*, `V=0.21`, `dV=+0.55` — nền đổi sang phòng tắm xám → `10/10` ID dính chặt chai xịt cũ); **`qid=1929`** (*"Replace man with businesswoman"*, `V=0.28`, `dV=+0.48` — banner chữ cam đổi thành tablet → `9/10` ID dính chặt người đàn ông + chữ cam); **`qid=127`** (*"pastel pink background"*, `V=0.49`, `dV=+0.44` — mất banner *"DIY MAKEUP REMOVER"* → `8/10` ID dính chặt banner chữ); **`qid=1254`** (*"baggage cart with landing gear"*, `V=0.63` — zoom ra toàn thân máy bay ban đêm → `6/10` ID + `3/7` CIRR dính mũi máy bay ban ngày!).

---

## 3. Điểm Mù Kiến Trúc Của Riêng BLIP-2 Q-Former (`BLIP-2 = 0` Nhưng `CLIP` Giải Được!)

Phân tích giản đồ Venn giữa 3 thế hệ kiến trúc cho thấy một kết quả bất ngờ: **BLIP-2 không hoàn toàn bao trùm (strict superset) CLIP!** Vẫn tồn tại một lát cắt rõ rệt các truy vấn mà **CẢ 10 model BLIP-2 (EVA-ViT-g + ViT-L) đều trượt, nhưng họ CLIP nhỏ hơn lại giải đúng**:

| Benchmark | Cả 2 cùng đúng | Chỉ BLIP-2 đúng | **BLIP-2 Blind Spot (0 BLIP-2 đúng, ≥1 CLIP đúng!)** | Cả 2 cùng trượt | Đặc điểm các ca CLIP thắng BLIP-2 |
|---|---:|---:|---:|---:|---|
| **EDIR In-Domain (`R@1`)** | `395` (`39.4%`) | `289` (`28.8%`) | **`66` (`6.6%`)** *(16 ca `≥2` CLIP thắng!)* | `253` (`25.2%`) | Tập trung ở `complex` (12), `shape` (6), `style` (6), `texture` (5), `cardinality` (5), `viewpoint` (5); **`CLIP4Cir` thắng 31 ca**, `TG-CIR` 22, `LIMN` 19! |
| **PinPoint In-Domain (`Hit@10`)** | `1,014` (`75.8%`) | `152` (`11.4%`) | **`40` (`3.0%`)** | `131` (`9.8%`) | Tập trung ở `Entertainment` (13), `Women's Fashion` (8), `Swap` (15), `Context Fit` (12), `Negation` (8); `TG-CIR` 16, `LIMN` 15! |
| **FashionIQ (`R@10`)** | `3,887` (`64.6%`) | `451` (`7.5%`) | **`374` (`6.2%`)** | `1,304` (`21.7%`) | Chiếm `6.22%` toàn bộ FashionIQ (gần ngang phần chỉ BLIP-2 đúng `7.50%`)! |

- **Bằng chứng định lượng & hình ảnh về nút thắt nén 32-token của Q-Former:**
  - Trên `66` ca `BLIP-2 Blind Spot` của EDIR, trọng tài CLIP đo được **`mean ΔV = -0.0011, mean ΔT = -0.0026`** (GT đã nhỉnh hơn Wrong trong không gian CLIP trực tiếp), ngược hẳn với `253` ca cả 2 cùng trượt (`mean ΔV = +0.1400`).
  - Soi ảnh trực tiếp trên `r5_edir_3` và `r5_pp_4` chứng minh rõ **Q-Former làm mất chi tiết thị giác tần số cao (high-frequency spatial/color/pose fidelity)**:
    - **`EDIR qid=4455`** (**`4/4` CLIP thắng, `0/10` BLIP-2!**): *"watercolor style, rounded phone case corners, subtle vignette"* — `8/10` ID & `5/7` CIRR BLIP-2 nén mất sắc xanh/vàng watercolor và làm bệt ảnh thành bản phác thảo đen trắng!
    - **`EDIR qid=1743`** (**`3/4` CLIP thắng, `0/10` BLIP-2!**): *"shadows on the grass removed"* — `8/10` BLIP-2 chọn ảnh biến **con bò đen-trắng thành bò nâu** trong khi bóng cỏ vẫn còn!
    - **`EDIR qid=832`** & **`3453`** (**`2/4` CLIP thắng, `0/10` BLIP-2!**): `9/10` BLIP-2 rơi vào đúng thuộc tính cấm sau `"except"` (*curved label* trên băng cassette, *watercolor* trên váy tím), trong khi CLIP tránh được!
    - **`PinPoint query_07835`** (**`2/4` CLIP thắng, `0/10` BLIP-2!**): *"Show the flamingo standing on both legs."* — ảnh REF là hồng hạc **đứng 1 chân**, GT là hồng hạc **đứng 2 chân**. `5/10` Top-1 & `6/10` Non-Ref #1 BLIP-2 đều chọn **bản sao cận trùng của hồng hạc đứng 1 chân (`EXPLICIT-NEG!`)**, trong khi `2/4` CLIP nhận ra tư thế 2 chân!

---

## 4. Khám Phá 5 Cơ Chế Thất Bại Mới & Kiểm Toán Trung Thực Độ Sạch Đuôi Cực Đại Của Cả 3 Benchmark

Từ phân tích toàn bộ `95` ca `All-42-Failed` EDIR, `41` cảnh độc lập `All-31-Failed` PinPoint, và soi bằng mắt **`69` ca mới hoàn toàn**, chúng tôi phát hiện thêm **5 cơ chế lỗi mới chưa từng nêu ở Đợt 1–4**:

### 4.1. Cơ Chế Mới #1 — Lỗi "Khác Đi Không Nêu Đích" (`Open-Ended Negative Difference: "different X"`)
- Trên EDIR, **`96` truy vấn** chỉ yêu cầu *"a different color / shape / finish / style / weather / time"* mà **không gọi tên trạng thái đích cụ thể**. Kết quả: **sụp thẳng xuống `17.50%` ID (`-13.72%`) và `20.09%` CIRR (`-18.56%`)** — **ngang bằng đúng sàn ngẫu nhiên của `remove` (`17.6%`)**!
- Ngược lại, nhóm `explicit_named_concrete_target` (`n=168`, *"to/into a <concrete noun>"*) đạt **`36.49%` ID (`+7.90%`)** và **`48.89%` CIRR (`+14.44%`)**.
- **Giải mã trọn vẹn vì sao các nhóm "dễ" trên EDIR vẫn có ca `All-42-Failed`:**
  - `color` (`9` ca): **`6/9`** là loại trừ ẩn hoặc `"different color"` (`qid=67, 207, 221, 231, 280, 294`).
  - `time` (`7` ca): **`6/7`** là `"different time of day, except for..."` (`qid=3602, 3608, 3659, 3664, 3668, 3844`).
  - `weather` (`7` ca): **`6/7`** là `"different weather, except for..."` hoặc `"don't want darker clouds"` (`qid=3932, 3968, 4053, 4085, 4123, 4199`).

### 4.2. Cơ Chế Mới #2 — Lực Hút Định Dạng Ảnh Chụp (`Photo-Presentation Gravity`) & Rò Rỉ Vật Thể Đi Kèm (`Co-Occurring Garment Leakage`)
Soi `18` ca **Near-Miss sạch (`med_rk = 12–65`)** trên FashionIQ (`r5_fiq_1..3`) phát hiện hai thiên lệch thị giác cực mạnh không liên quan gì đến sản phẩm mục tiêu:
- **Bám kiểu chụp ảnh catalog (`Photo-Format Gravity`):** Khi ảnh REF chụp dạng **áo gấp gọn (`shirt:1114`)**, **mannequin tàng hình không người (`toptee:1621`)**, hay **trải phẳng flat-lay (`toptee:198`)**, cả **`7/8` đến `8/8` BLIP-2** đều ưu tiên trả về ảnh **cùng kiểu chụp (gấp / không người / flat-lay)** và đẩy ảnh GT (có người mẫu mặc!) ra ngoài Top-10!
- **Rò rỉ trang phục không liên quan (`Co-Occurring Garment Leakage`):** Trên **`toptee:392`** (*"is black with long sleeves, less revealing"*), ảnh REF là cô gái mặc áo croptop nâu **phối với váy da báo**. **`7/8` BLIP-2** bị chiếc váy da báo hút chặt nên chọn cô gái mặc **áo croptop đen (vẫn hở bụng, vi phạm `"less revealing"`!) phối quần soóc da báo y hệt**!

### 4.3. Cơ Chế Mới #3 — Hướng Nhìn (`Gaze`), Tư Thế Chi (`Limb Pose`), Đồng Âm (`Homophone`) & Mất Bản Sắc Giới Tính (`Gender Drop`)
- **Mù hướng nhìn mắt/đầu (`Gaze Direction`):** Đúng **`2 / 2`** ca `All-42-Failed` của nhóm `action` trên EDIR đều là hướng nhìn: **`qid=2750`** (*"man looking to the left"* — `7/10` ID chọn nhìn sang phải, `4/7` CIRR chọn góc nghiêng ngược chiều!) và **`qid=2820`** (*"player look towards the camera"* — `0/16@R5`!).
- **Mù tư thế/số chi (`Pose & Limb Count`):** **`PinPoint query_07835`** (*"flamingo standing on both legs"* → `0/10` BLIP-2, kẹt ảnh 1 chân).
- **Lạc nghĩa do lỗi đồng âm (`Homophone Hijack`):** **`FashionIQ dress:1193`** (*"is shorter and is **sea through**"* — gõ nhầm từ `"see-through"`!) khiến toàn bộ model đọc `"sea"` thành **màu xanh nước biển/ngọc lam** thay vì váy ren đen xuyên thấu!
- **Mất bản sắc giới tính dưới biến đổi phong cách (`Gender-Identity Drop`):** **`PinPoint query_00470`** (*"Make this outfit more formal"* trên ảnh **phụ nữ** mặc sơ mi caro + quần soóc bò) → mô hình lại trả về **đàn ông mặc tuxedo đen**!

### 4.4. Cơ Chế Mới #4 — Kiểm Toán Đuôi Cực Đại PinPoint (`41` Cảnh `All-31-Failed` & `17` Cảnh `Zero-Pos@50`): Phát Hiện Nhãn Hỏng Ẩn!
Kiểm toán mắt trên `21` cảnh PinPoint mới (`r5_pp_1..4`) mang lại một phát hiện khoa học cực kỳ quan trọng về chất lượng dữ liệu:
- **PinPoint sạch ở vùng trung bình (`~83%` sạch ở Đợt 3), NHƯNG ở đuôi cực đại `0/31@10` và `0/16@50` (`r5_pp_1` & `r5_pp_3`), có tới `~40%–45%` ca thực chất là LỖI GHÉP NHÃN / GT HỎNG CỦA PINPOINT hoặc FALSE NEGATIVE HỢP LÝ!**
  - `query_00340` (`0/16@50`, *"Create an outfit for this look"*): Ảnh REF là **tóc nam giới**, nhưng GT Pos #1 & #2 là **váy đầm phụ nữ**; còn model trả về **nam giới cùng kiểu tóc mặc vest đen**!
  - `query_04079` (`0/16@50`, *"flip-flops at the beach"*): GT Pos #1 & #2 là **hình xăm chữ Hoa/Ả Rập trên cánh tay** (không dép, không biển!).
  - `query_07661` (`0/16@50`, *"Make the lip color a simple nude matte"* trên phụ nữ da màu): GT Pos #1 & #2 là mặt chia đôi da trắng và **môi đỏ hút thuốc** (trùng với `query_03084`!), còn model trả về đúng **phụ nữ da màu môi nude matte**!
  - `query_03084` (`0/31@10`, *"warm autumn shade"* cho môi): GT Pos #2 là **môi màu xanh dương chói**!
  - `query_00259` (`0/16@50`, *"turn this artwork into a dress using its colors"*): GT Pos #2 là **bóng bay trái tim cầu vồng**, còn model chọn đúng **chiếc váy chuyển màu đỏ-cam-tím khớp 100% bức tranh**!
  - `query_03102` (`CLIP=2/4, BLIP2=0/10`, *"Korean night routine way simpler"*): GT Pos #1 là **ngôi chùa cổ Hàn Quốc trong rừng**!
  - `query_03002` (`0/31@10`, *"house with facade redone in concrete"*): GT là nhà gạch & nhà di động nhựa xanh, còn `8/10` model tìm ra đúng **ngôi nhà mặt tiền bê tông giữa sa mạc có xương rồng**!
  - `query_01306` (`0/31@10`, *"amber-toned tablescape"*): `8/10` model tìm ra đúng bàn tiệc nến màu hổ phách nhưng lại bị gán nhãn `EXPLICIT-NEG`!
  - Ngoài ra, **`5 / 16` cảnh `Swap` `0/31`** thực chất là truy vấn **Xóa/Phủ định ẩn** bị phân loại vào `Swap` (`"Remove all text"`, `"remove peach socks"`, `"Remove finger wave"`, `"taking off statement ring"`, `"Remove tie-waist belt"`).

### 4.5. Tổng Kết So Sánh Chất Lượng Nhãn Trên Toàn Bộ `135` Ca Đã Soi Bằng Mắt (`Đợt 3 + Đợt 5`)

| Benchmark & Phân khúc soi bằng mắt | Số ca soi (`Tổng = 135`) | Tỷ lệ GT sai/hỏng | Tỷ lệ False Negative hợp lệ | **Tỷ lệ Lỗi Model THẬT** | Kết luận dùng cho Research |
|---|---:|---:|---:|---:|---|
| **EDIR** (`All-42` + `All-16@R5` + `BLIP2-BS` + `Q4`) | **42 ca** (`18` R3 + `24` R5) | `~10–14%` *(chủ yếu drift nền mạnh ở `0/16@R5`)* | `0%` | **`~86%`** | **Sạch nhất để kiểm định thuật toán** (nên tách riêng lát cắt $d_{\text{edit}} < 0.45$ để tránh collateral drift) |
| **PinPoint — Vùng fail chuẩn (`Zero-Pos@10`)** | **21 ca** *(R3 + `r5_pp_2,4`)* | `~10%` | `~14%` | **`~76%`** | **Rất tốt cho bài toán đời thực** (`Negation`, `Complement`, `Context Fit`, `Explore`, `Paraphrase`) |
| **PinPoint — Đuôi cực đại (`0/31@10` `Swap` & `0/16@50`)** | **18 ca** *(`r5_pp_1,3`)* | **`~39%`** (`7/18`) | **`~17%`** (`3/18`) | `~44%` | **Cảnh giác trần tuyệt đối (~3–4% không thể về 0 do nhiễu pin Pinterest)** |
| **FashionIQ — Near-Miss sạch (`med_rk 12–65`)** | **18 ca** *(`r5_fiq_1..3`)* | `~11%` (`2/18`) | **`~61%`** (`11/18`) | `~28%` (`5/18`) | **Nhãn sạch hơn hẳn đuôi `>500`, phơi bày rõ `Photo-Format Gravity` & `Co-Occurring Leakage`** |
| **FashionIQ — Đuôi cực đại (`ALL13 Fail@50` & Q4)** | **36 ca** *(`30` R3 + `6` R5)* | **`~36%`** | **`~39%`** | `~25%` | **Nhiễu nhãn + FN rất nặng (~75–80%); KHÔNG dùng đơn lẻ để chứng minh đột phá** |

---

## 5. Bảng Tổng Hợp Tối Hậu: 12 Lỗi Chung Phổ Quát Của Mọi Mô Hình CIR Hiện Đại (Master Failure Taxonomy)

Đây là **bản tổng kết toàn bộ 5 đợt nghiên cứu** — gom toàn bộ lỗi thật đã được kiểm chứng (loại bỏ hoàn toàn nhiễu nhãn benchmark) thành **12 lỗi chung có hệ thống** chia làm 4 nhóm lớn:

| Nhóm | # | Tên Lỗi Chung Phổ Quát | Cơ chế gốc rễ & Góc phần tư `(ΔV, ΔT)` | Chứng cứ định lượng bit-exact | Chứng cứ hình ảnh thực tế (`query_id`) | Kháng quy mô? | Benchmark sạch nhất để nghiên cứu |
|---|---|---|---|:---:|---|:---:|---|
| **I. Logic & Ngữ nghĩa Query** | **1** | **Đảo ngược Phủ định, Xóa & Loại trừ (`Pink Elephant Inversion`)** | Biểu diễn đơn cực dương kéo query về đúng vật bị cấm (**`Q3 Double-Trap`**: `41.2%` EDIR `remove`, `69.6%` PinPoint `Negation`) | EDIR `remove` `17.6%`, hidden excl `Rk 3.596 (p=8.3e-7)`; PinPoint `Negation Pos@1=4.3%`, `NegRec=0.350` | EDIR `#67`, `#1503`, `#1530`, `#3318`; FIQ `shirt:803`; PP `query_06179`, `06830` | **★★★★★** (`-1.5%` khi scale!) | EDIR (`remove`+excl), PinPoint (`Negation`), MA-CIR |
| | **2** | **Sụp đổ "Khác Đi Không Nêu Đích" & So Sánh Tương Đối Giả (`Open-Ended Difference & Pseudo-Relative`)** | Không đo trạng thái hiện tại của `I_ref` để tính độ lệch tương đối; chỉ dựa vào từ khóa tuyệt đối | EDIR `"different X"` **`17.5%`** (`-13.7%`); FIQ `CmpNoCol` **`38.1%`** vs `64.3%` (**`p=2.3e-70`**); giải mã `dress < toptee` | EDIR `#670`, `#207`, `#371`; FIQ `dress:1385`, `shirt:1229`; PP `query_00470` | **★★★★☆** | EDIR (`different`), FIQ lát cắt `CmpNoCol`, PinPoint |
| | **3** | **Giòn Ngôn Ngữ, Tham Chiếu Ngầm (`Deixis`) & Hiểu Nghĩa Đen / Đồng Âm** | Khớp bề mặt từ khóa cụ thể (`+0.063` màu `p=0.002`), phạt từ trừu tượng (`-0.022 p=0.024`), không giải quy chiếu `"this/here"` | PinPoint `ling_sens` **`71–92%`** tổng mAP, `20–29%` lật `0↔hit`; EDIR `Deixis -5.4%` | PP `query_07355` (*snake chain → con rắn*); FIQ `dress:1193` (*sea through → váy xanh biển*) | **★★★★☆** | PinPoint (`258` nhóm paraphrase + `Deixis`) |
| **II. Bảo Toàn, Neo Thị Giác & Trộn** | **4** | **Vực Thẳm Khoảng Cách Chỉnh Sửa, `Same-Scene Lock` & `Ref-Copy Gravity`** | Đường tắt InfoNCE bám `I_ref` (**`Q1 Visual-Anchor`**: **`51.6%`** toàn bộ lỗi EDIR, **`37.9%`** PinPoint `Context Fit`) | EDIR $d\ge 0.50 \to 12.0\%$ (`11.7×` fail); `CondR@1` tụt `15/15` phép sửa (`replace -46.4%`); PinPoint `RefCopy@1 = 29.7%` (CLIP `68.4%`); **`ρ = -0.758`** | EDIR `#1982`, `#1929`, `#1254`, `#3030`; PP `query_03562`, `02949`, `07626` | **★★★★☆** | EDIR (`ID` vs `CIRR`), PinPoint (`Raw` vs `NoRef`) |
| | **5** | **Định Dạng Ảnh Chụp, Vật Thể Đi Kèm & Lệch Cân Bằng 2 Chiều** | Quá bám studio format/IP/đồ phụ (`Q1`), hoặc xóa sạch màu/bản sắc gốc (`Q2 Text-Hijack: 13–26%`) | `Q2 = 13.4%` EDIR (`spatial 29.5%`), `18.4%` PinPoint (`Complement 26.1%`), `15.8%` FIQ | FIQ `shirt:1114`, `toptee:1621, 198` (format), `toptee:392` (váy da báo), `shirt:982`; PP `query_01866` (bếp cam→trắng) | **★★★★☆** | FashionIQ Near-Miss + PinPoint + EDIR |
| | **6** | **Bỏ Sót Thuộc Tính Kém Nổi Bật, Tráo Buộc, Thiên Vị Vế Đầu & Sụp Đổ Tầng Fusion (`Q4`)** | Màu/vật thể rời nuốt hình học/chất liệu; nén Q-Former 32-token gây **`Q4 Pure Fusion Collapse`** & **`BLIP-2 Blind Spot`** | EDIR **`Q4 = 13.5%`** (`97` ca), **`66` ca CLIP thắng BLIP-2**; FIQ **`C1 > C2` (`-3.21%`)**, `formality 42.1%`, `fit 46.0%`; PinPoint `Long < Short` (`p=2.7e-3`) | EDIR `#4336`, `#4501`, `#4549`, `#1201`, `#4455`, `#1743`; FIQ `shirt:1811` (*black writing → áo đen*); PP `query_07835` | **★★★★★** | EDIR (`complex`, `Q4`), PinPoint (`Long`), FIQ |
| **III. Nhận Thức Thị Giác & Phi Đẳng Cấu** | **7** | **Mù Hình Học 3D, Hình Dạng, Đếm Số Lượng, Hướng Nhìn & Tư Thế** | Tiền huấn luyện ViT bất biến hình học/góc nhìn/số lượng | EDIR `shape` **`+0.23%`** khi scale 10×!, `viewpoint 25.0%`, `cardinality 25.6%`; `2/2` fail `action` là hướng nhìn | EDIR `#670`, `#727`, `#2135`, `#2750`, `#3030`, `#923`; PP `query_07835` (hồng hạc 2 chân) | **★★★★★** (`+0.2%` khi scale!) | EDIR (`shape`, `viewpoint`, `cardinality`, `texture`) |
| | **8** | **Mù Chữ Trong Ảnh & Chi Tiết Độ Phân Giải Cao (`OCR / Scene-Text Blindness`)** | Nén ảnh `224×224` (`16×16` patch) làm mất chữ nhỏ, watermark, phông chữ, vị trí tiêu đề | EDIR OCR/Text (`n=66`): **`-11.67%`** trên CIRR (`25.97%` vs `37.64%`), `2/2` fail `spatial` là vị trí chữ | EDIR `#1892` (`7+ → 10+`), `#2051` (font), `#2629` (`PENCIL TOPPERS`), `#2412`, `#1709`, `#1731` | **★★★★☆** | EDIR lát cắt OCR/Text (`n=66`) |
| | **9** | **Bẫy Chủng Loại & Ý Định Thực Dụng (`Cross-Category & Pragmatic Blindness`)** | Đích không đẳng cấu pixel với `I_ref` ($V \approx 0.52$), đòi hỏi suy luận phối hợp/chức năng/tri thức thế giới | PinPoint `Complement` **`0.074`**, `Explore` **`0.080`** (kể cả xóa sạch negative vẫn kẹt `0.09`!) | PP `query_02301/02` (makeup→outfit), `00753` (váy→bảng màu), `02072` (bát→quán), `00452` (ngồi bệt), `03318` | **★★★★☆** (`+0.016` khi scale!) | PinPoint (`Complement`, `Explore`, `Context Fit`) |
| **IV. Hệ Thống Quy Mô, Bất Định & Công Bằng** | **10** | **Vực Thẳm Hiếm Đích (`1–2` Pos), Sập Quy Mô `24×` & Hubness Gallery Đặc** | Tắc nghẽn distractor cục bộ trên kho lớn (`109k`) + ảnh hub FP trên kho đặc | PinPoint `1–2 pos`: **`P@1 = 8.57%`**, **`ZeroPos@10 = 60.3%`**; `SPRC` giữ `42.5%` vs `ConeSep 55.5%`; FIQ hub **`98%` FP** | Toàn tập PinPoint `109.6k` & FashionIQ `val_split` | **★★★★☆** | PinPoint `109.6k` (`UltraRare`), FashionIQ |
| | **11** | **Mù Bất Định & Tự Tin Thái Quá (`Calibration & Abstention Blindness`)** | Điểm cosine không phân biệt được truy vấn trúng (`≤10`) vs sai nặng (`>50`) | FIQ `AUROC(Top-1)`: `CLIP4Cir 0.513`, `CSMCIR 0.570`, `QuRe 0.581` (`Margin 0.549`) | Toàn bộ `6,016` queries FashionIQ × `13` runs | **★★★★★** | FashionIQ scores + PinPoint + AmbiCIR |
| | **12** | **Chênh Lệch Công Bằng, Chuyển Giao Âm Đa Miền & Hạn Chế Đơn Ảnh** | Nhóm da sẫm dính `NegRec` cao hơn; train gộp phá hỏng miền phụ; 16/16 model không nhận được `≥2` ảnh gốc | PinPoint `Swap-only`: `DARKEST/2ND` `NegRec` gấp **`3.65–4.21×`**; `Animals -0.054`, `Entertainment -0.050`; `Women's` thua `Men's -40%` dù train `7.4×`; `Multi-Image -40..70%` | PinPoint `skin_tone_bucket`, `20` domains & full Multi-Image split | **★★★★★** | PinPoint (`skin_tone_bucket`, `20` domains, Multi-Image) |

---

## 6. Bộ Tiêu Chí Lọc Hướng Đi Sạch Nhất Cho Bạn (Tổng Kết Sau 5 Đợt Kiểm Định)

Nếu đặt toàn bộ 12 lỗi chung trên lên bàn cân theo **3 điều kiện vàng của một hướng nghiên cứu chất lượng cao** (*Dữ liệu đánh giá sạch ít nhiễu* + *Kháng lại việc scale backbone 5–10×* + *Có dư địa cải thiện định lượng rõ ràng*), nhóm **5 hướng lõi xuất sắc nhất** lộ diện cực kỳ rõ ràng:

1. **Lỗi #1 + #2 (Phủ định/Xóa + Mở-Đầu `"different X"` / Tương đối):** Kháng quy mô tuyệt đối (`-1.5%` khi scale!), góc phần tư `Q3` chứng minh rõ cơ chế toán học, benchmark cực sạch (`EDIR` + `PinPoint Negation` + `MA-CIR`).
2. **Lỗi #4 + #10 (Vực thẳm khoảng cách chỉnh sửa $d_{\text{edit}}$, `Same-Scene Lock` vs `24× Scale-Retention`, `ρ = -0.758` & Đích hiếm `1–2 pos`):** Giải thích trực tiếp >50% toàn bộ cú ngã trên EDIR (`Q1 = 51.6%`) và `0% → 19.5%` cú sập theo khoảng cách thị giác trên PinPoint.
3. **Lỗi #6 + #7 + #8 (Điểm mù nén Q-Former `Q4` & `BLIP-2 Blind Spot`, Hình học 3D/Shape/Gaze/Pose/Counting, OCR/Scene-Text):** Kháng quy mô tuyệt đối (`shape +0.23%`), trực tiếp chứng minh bằng `66` ca CLIP thắng cả 10 BLIP-2 và `97` ca `Q4 Pure Fusion Collapse` trên EDIR.
4. **Lỗi #11 (Hiệu chuẩn Bất định, Từ chối & Tương tác Chủ động):** `AUROC = 0.51–0.58` là bằng chứng thực nghiệm cực sốc trên 13 model hiện đại, ít đối thủ cạnh tranh hơn mảng fusion truyền thống.
5. **Lỗi #9 + #12 (Phi đẳng cấu Cross-Category/Pragmatic/Multi-Image & Công bằng/Chống Chuyển Giao Âm Đa Miền):** Rất giàu tác động thực tế, đặc biệt mảng **Fairness (`3.65–4.21×` NegRec disparity)** và **Anti-Negative-Transfer Multi-Domain CIR** còn rất nhiều khoảng trống mở.

---

## 7. Liên Kết Trọn Bộ 5 Báo Cáo & Bản Tổng Hợp Kiểm Chứng Tối Hậu Trong Repo

1. **Đợt 1** — Tổng quan 6 điểm gãy phổ quát trên 16 model (`86` runs): [CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md](CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md)
2. **Đợt 2** — Kiểm định thống kê (`p-values`), động học `history.jsonl`, phản thực (`NoRef`/`RRF`) & SOTA 2024–2026: [CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md](CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md)
3. **Đợt 3** — Kiểm toán hình ảnh trực tiếp `66` truy vấn đầu tiên & Bản đồ 8 lỗi composition: [CIR_VISUAL_FORENSICS_AND_COMPOSITION_FAILURE_ATLAS.md](CIR_VISUAL_FORENSICS_AND_COMPOSITION_FAILURE_ATLAS.md)
4. **Đợt 4** — Bản đồ hệ thống ngoài composition (Scaling-Resistance, Calibration, OCR, Hubness/Rarity, Fairness, Long-Tail): [CIR_FRONTIER_SYSTEM_FAILURE_LANDSCAPE.md](CIR_FRONTIER_SYSTEM_FAILURE_LANDSCAPE.md)
5. **Đợt 5 (Tài liệu này)** — Phân rã 4 góc phần tư `(ΔV, ΔT)` (`9,107` ảnh), Vực thẳm khoảng cách chỉnh sửa, Điểm mù Q-Former (`BLIP-2 vs CLIP`), kiểm toán thêm `69` ảnh mới (`135` ảnh tích lũy) & **Hệ thống phân loại 12 lỗi chung tối hậu**: [CIR_EXHAUSTIVE_QUERY_FORENSICS_AND_UNIVERSAL_TAXONOMY.md](CIR_EXHAUSTIVE_QUERY_FORENSICS_AND_UNIVERSAL_TAXONOMY.md)
6. **Bản Tổng Hợp Kiểm Chứng Tối Hậu (Master Verified Synthesis — Zero-Error Audit toàn bộ 5 đợt):** [CIR_MASTER_VERIFIED_FAILURE_SYNTHESIS.md](CIR_MASTER_VERIFIED_FAILURE_SYNTHESIS.md)
