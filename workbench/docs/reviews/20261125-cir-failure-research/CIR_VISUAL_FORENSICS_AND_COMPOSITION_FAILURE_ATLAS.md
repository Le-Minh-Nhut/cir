# Báo Cáo Đợt 3: Kiểm Toán Hình Ảnh Trực Tiếp (66 Truy Vấn) & Bản Đồ Các Lỗi Composition Của Mô Hình CIR Hiện Tại

> [!IMPORTANT]
> Khác với Đợt 1 ([CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md](CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md)) và Đợt 2 ([CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md](CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md)) vốn tập trung vào thống kê định lượng và thiết kế giải pháp, báo cáo Đợt 3 này thực hiện **kiểm toán hình ảnh trực tiếp bằng mắt (visual forensics) trên 66 ca thất bại đồng loạt** (`30` FashionIQ + `18` EDIR + `18` PinPoint) kết hợp phân tích bỏ sót thuộc tính khi kết hợp đa phương thức (composition drop).
>
> **Mục tiêu cốt lõi:** Trả lời chính xác hai câu hỏi trước khi bắt tay nghiên cứu cách sửa:
> 1. **Trong những ca "tất cả model đều sai", bao nhiêu phần trăm thực sự là lỗi của model và bao nhiêu phần trăm là lỗi target/caption của chính benchmark?** *(Đặc biệt kiểm chứng nghi ngờ thực tế về chất lượng nhãn FashionIQ.)*
> 2. **Khi kết hợp `(I_ref, T)`, các model CIR hiện tại thực sự đang bỏ sót gì, bóp méo gì, tráo buộc gì và kẹt ở đâu?**

---

## 1. Sự Thật Về Chất Lượng Ground-Truth Của 3 Benchmark (Kiểm Chứng Bằng Mắt Trên 66 Ca Fail Đồng Loạt)

Soi trực tiếp từng bộ tứ/bộ năm ảnh **`[Reference] | [Ground-Truth Target] | [Consensus Top-1] | [Consensus Top-2]`** của 66 truy vấn khó nhất (nơi **tất cả** mô hình BLIP-2/CLIP đều trượt) cho thấy sự chênh lệch chất lượng nhãn **cực kỳ lớn** giữa 3 bộ dữ liệu:

| Benchmark | Mẫu soi trực tiếp | **Lỗi nhãn / Target sai / Hallucination** | **False Negative hợp lệ** *(Model chọn đúng bằng/hơn GT!)* | **Lỗi nhận thức & Composition THẬT của model** | Nhận xét định hướng nghiên cứu |
|---|---:|---:|---:|---:|---|
| **FashionIQ** (`ALL13 Fail@50`, chia đều 5 nhóm & 3 ngành) | **30 ca** | **`~40%`** (`12/30`) | **`~40%`** (`12/30`) | **`~20%`** (`6/30`) | **Cảnh báo đỏ:** ~80% ca trượt sâu nhất (`>50`) là do caption đảo ngược Ref↔Target, bịa thuộc tính không có trên GT, rác MTurk, hoặc GT đơn lẻ quá tùy ý! |
| **EDIR** (`ALL-42-failed`, trải trên 9 phép sửa khó nhất) | **18 ca** | **`~11%`** (`2/18`, nhẹ: 1 mơ hồ, 1 edit màu chưa trọn) | `0%` (`0/18`) | **`~89%`** (`16/18`) | **Benchmark sạch nhất** để mổ xẻ lỗi composition tinh vi (`remove`, `viewpoint`, `shape`, `cardinality`, `complex`, `exclusion`). |
| **PinPoint** (`ALL-10-BLIP2 Zero-Pos@10`, 18 cảnh độc lập) | **18 ca** | **`~6%`** (`1/18` nhầm ảnh REF `query_06689`) | **`~11%`** (`2/18` FN hợp lý sau khi bỏ Ref-Copy) | **`~83%`** (`15/18`) | **Benchmark thực tế nhất** để nghiên cứu `Negation`, `Complement`, `Explore`, `Context Fit`, và `Ref-Copy Gravity`. |

---

### 1.1. Bằng Chứng Hình Ảnh Trực Tiếp Về Lỗi Nhãn Nghiêm Trọng Trên FashionIQ (`30` Ca Soi Bằng Mắt)

Trực giác ban đầu rằng *"khá nhiều case fail trên FashionIQ có target sai sai"* là **hoàn toàn chính xác**. Bốn dạng hỏng dữ liệu xuất hiện lặp đi lặp lại trong các ca `ALL13 Fail@50`:

#### Dạng 1 — Annotator Đảo Ngược `Ref ↔ Target` (Mô tả nhầm ảnh gốc thay vì ảnh đích!)
- **`shirt:1617`** (`B00A4LC2II → B006H6TYSK`, `med_rk = 500`): `C1 = "is black with yellow letters"`. **Ảnh REF** là áo đen in hình con ong vàng + chữ vàng `00892-B`, còn **ảnh GT** là áo đen trắng ban nhạc Bathory hoàn toàn không có chữ vàng! Cả `8/8` BLIP-2 trả về đúng áo con ong chữ vàng giống REF.
- **`shirt:698`** (`B007UNXLB0 → B003674EUQ`, `med_rk = 5,634`): `C1 = "is button-up and plaid"` (mô tả **đúng ảnh REF** sơ mi caro có cúc!) trong khi `C2 = "is t shirts with no buttons"` (mô tả GT áo phông đen). `6/8` BLIP-2 tin theo `C1` và trả về sơ mi caro.
- **`toptee:1562`** (`B00CD5NHMI → B004CPWN1I`, `med_rk = 523`): `C2 = "a solid darker color and has shorter sleeves"` mô tả **đúng ảnh REF** (áo phông đen ngắn tay Jack Daniels), trái ngược hoàn toàn với GT sơ mi dài tay họa tiết kẻ! `6/8` BLIP-2 trả về một áo Jack Daniels đen ngắn tay khác.
- **`dress:1400`** (`B000BL6VJM → B007OZSTY8`, `med_rk = 2,441`) & **`dress:1468`** (`B008GQ3H50 → B00CFD6ZBI`, `med_rk = 2,222`): Một annotator mô tả GT (ngắn, sát nách), annotator còn lại đảo ngược mô tả REF (dài, dài tay).

#### Dạng 2 — Annotator Bịa Thuộc Tính Không Hề Tồn Tại Trên Ảnh Target (Hallucinated Target Attributes)
- **`dress:826`** (`B007HZAFSI → B008MLXYYS`, `med_rk = 3,057`): `C1 = "has summery straps and wild print"`, `C2 = "patterned and much darker"`. **Ảnh GT là váy len đen trơn dài tay cổ lọ** (không dây mùa hè, không họa tiết!). `7/8` BLIP-2 trả về đúng váy tối màu hai dây họa tiết sặc sỡ!
- **`dress:846`** (`B008E6CKVY → B00A9T9N6O`, `med_rk = 981`): `C2 = "has more of a red chinese design and up top has short sleeves in a v shape"`. **Ảnh GT là váy ngắn màu đen trơn sát nách** (không một chấm đỏ hay họa tiết Trung Hoa nào!). Model trả về đúng váy đỏ cổ V họa tiết!
- **`toptee:1486`** (`B0061M50AG → B003I7POCG`, `med_rk = 1,243`): `C1 = "The tank top is tight fitting and black and red in color"`. **Ảnh GT là áo phông nâu in hình xanh**, còn `7/8` BLIP-2 trả về đúng áo ba lỗ bó màu đen-đỏ!
- **`dress:472`** (`B008R8BFFG → B00C9OICDM`, `med_rk = 204`, `J = 0.91`): `C1 = "with a black matching belt"`, `C2 = "is more blue"`. **Ảnh GT là váy quây xanh không thắt lưng**; còn Top-1 & Top-2 đồng thuận của model là đúng người mẫu dáng đứng của REF mặc **váy xanh có thắt lưng đen**!
- Các ca bịa nhãn rõ rệt khác: **`dress:900`** (GT màu tím không thắt lưng nhưng caption ghi *"blue with belt"*), **`dress:700`** (GT váy bò denim nhưng caption ghi *"teal stripes"*), **`toptee:264`** (GT áo đen in chữ *"I HAIR"* nhưng `C1` ghi *"button up and solid pink"*), **`toptee:776`** (GT áo xanh sát nách nhưng `C1` ghi *"short sleeved shirt is white"*), **`dress:1383`** (GT váy đỏ rực nhưng `C2` ghi *"light color"*), **`toptee:1775`** (GT trắng-vàng nhưng `C2` ghi *"more pink"*), và **`toptee:1146`** chứa rác MTurk nguyên văn: `C2 = "and that's reason enough not to take it home"`.

#### Dạng 3 — False Negative Hợp Lệ (Caption Quá Thiếu Thông Tin → Model Còn Hợp Lý Hơn GT!)
- **`shirt:923`** (`med_rk = 3,975`): REF = áo xám dài tay; Caption chỉ ghi *"more casual with shorter sleeves"*. GT lại là **áo phông tím in đàn chim cánh cụt** (caption không hề nhắc tím hay chim cánh cụt!). Top-1 & Top-2 của model đều là **áo phông xám ngắn tay** — giữ đúng màu REF + thỏa mãn đúng caption!
- **`toptee:438`** (`med_rk = 2,072`): REF = áo đỏ phối ren đen dài tay; Caption ghi *"more casual, darker and shorter sleeves"*. GT là áo rock in bộ trống, còn `7/8` BLIP-2 trả về **áo đỏ thẫm phối ren đen ngắn tay**!
- **`shirt:430`** (`med_rk = 1,033`): REF = sơ mi kaki dài tay; Caption ghi *"shorter sleeves, more casual, less formal"*. GT là áo phông rêu in hình người, còn `7/8` BLIP-2 trả về **sơ mi kaki ngắn tay**!
- **`shirt:1229`** (`med_rk = 2,351`), **`shirt:1789`** (`med_rk = 2,144`), **`toptee:722`** (`med_rk = 261`), **`toptee:873`** (`med_rk = 481`), **`toptee:1138`** (`med_rk = 1,835`): đều là những ca mà Top-1/Top-2 của model khớp 100% văn bản **và** bảo toàn nhận diện thị giác của REF tốt hơn hẳn GT đơn lẻ.

> [!WARNING]
> **Hệ quả cực quan trọng cho hướng đi research của bạn:**
> Khi đọc báo cáo hoặc tự phân tích lỗi trên FashionIQ, **tuyệt đối không lấy các ca rank cực tệ của FashionIQ làm động lực thiết kế mô hình nếu chưa soi ảnh**, vì ~80% trong số đó là nhiễu nhãn / false negative. Thay vào đó, **6 ca lỗi thật trên FashionIQ** cộng với **31/36 ca lỗi thật cực sạch trên EDIR & PinPoint** mới chính là mỏ vàng phơi bày trọn vẹn các điểm mù composition của CIR hiện đại!

*(Lưu ý nhỏ trên PinPoint từ kiểm toán hình ảnh: trong 18 ca soi bằng mắt, phát hiện 1 ca lỗi ghép ảnh REF là `query_06689` — câu hỏi nói "remove print from the durag" nhưng ảnh REF lại là cô gái mặc áo sơ mi hoa, trong khi câu anh em `query_06692` mới đúng ảnh nam giới đội durag. Các ca còn lại trên PinPoint và 16/18 ca trên EDIR đều có ảnh sạch và minh họa lỗi model cực kỳ rõ nét.)*

---

## 2. Bản Đồ Toàn Diện 8 Cơ Chế Lỗi Composition Của Các Mô Hình CIR Hiện Tại

Tổng hợp từ **66 ca soi ảnh trực tiếp** + **kiểm định định lượng trên 8,356 truy vấn** (`6,016` FashionIQ + `1,003` EDIR + `1,337` PinPoint), chúng ta rút ra được **8 hiện tượng hỏng hóc lặp đi lặp lại trong quá trình kết hợp `(I_ref, T)`** — đây chính là 8 "vết nứt" cụ thể để bạn chọn làm đối tượng nghiên cứu:

---

### Hiện Tượng #1 — Bỏ Sót & Đảo Ngược Phép Xóa / Phủ Định (`Deletion & Exclusion Inversion`)
*Đây là lỗi rõ ràng nhất khi nhìn tận mắt vào ảnh trên cả 3 benchmark.*

- **Điều gì xảy ra thực tế trên ảnh?**
  - **EDIR `qid=1503`** (*"Remove the burnt vehicle from the scene"*): GT đã xóa sạch xe cháy, nhưng **`10/10` BLIP-2** đều chọn ảnh vẫn còn nguyên chiếc xe cháy to tướng ở tiền cảnh.
  - **EDIR `qid=1530`** (*"without the image of the US Constitution"*): GT đã xóa bản Hiến pháp Mỹ, nhưng **`8/10` BLIP-2** chọn 2 ảnh vẫn trưng nguyên tờ *"We the People / CONSTITUTION"* giữa màn hình.
  - **EDIR `qid=67`** (*[color] "I don't want the graphic to be pink"*): REF là áo tím hình in hồng; GT là áo xanh hình in **cam**. Cả **`5/10` ID BLIP-2** lẫn **`4/7` CIRR BLIP-2** lại chọn đúng ảnh áo xanh có **hình in HỒNG chói**!
  - **EDIR `qid=3318`** (*[style] "different style, except for an antique look or a line drawing"*): ID BLIP-2 (`5/10`) chọn **đồng xu cổ (antique)**, CIRR BLIP-2 (`3/7`) chọn **tranh vẽ nét (line drawing)** — rơi trúng **cả 2 phương án bị loại trừ tường minh**!
  - **FashionIQ `shirt:803`** (*"darker and not a horror movie character"* trên áo *Frankenweenie*): GT là áo *SEAL Team VI*, nhưng Consensus #1 (`4/8`) & #2 là áo **Jack Skellington (*The Nightmare Before Christmas*)** — đúng nhân vật phim kinh dị Tim Burton!
  - **FashionIQ `dress:82`** (*"does not cover the sholders"*): `6/8` BLIP-2 chọn váy **có tay con che vai** (bỏ qua mệnh đề phủ định + nhạy cảm lỗi chính tả).
  - **PinPoint `query_06179`** (*"Remove the tie-dye pattern from the cardigan, making it a solid blue"*): `7/10` copy ảnh gốc, **`8/10` Non-Ref #1 chọn một áo cardigan tie-dye xanh khác**!
  - **PinPoint `query_06830`** (*"without the denim jacket"*): `9/10` Non-Ref #1 trả về ảnh gần trùng khớp của **đúng cô gái đó vẫn đang mặc áo khoác bò**!
- **Bản chất composition bị hỏng:** Các cơ chế cross-attention / pooling hiện tại hoạt động theo nguyên lý **cộng dồn dương (additive bag-of-concepts)**: nhắc đến thực thể `X` trong ngữ cảnh phủ định (`no X`, `remove X`, `except X`, `not X`) lại làm tăng trọng số attention vào `X`, kéo vector truy vấn về đúng phía các ảnh chứa `X`.

---

### Hiện Tượng #2 — Chiếm Quyền Của Thuộc Tính Nổi Bật & Bỏ Sót Hình Học / Cấu Trúc / Số Lượng Trong Tổ Hợp Đa Thuộc Tính (`Salient-Attribute Dominance & Geometric/Structural Drop`)
*Khi một truy vấn yêu cầu kết hợp nhiều thuộc tính cùng lúc (`A + B + C`), các model KHÔNG sai ngẫu nhiên — chúng có một **thứ bậc ưu tiên cứng nhắc**: giữ thuộc tính dễ (màu sắc, vật thể rời) và **vứt bỏ** thuộc tính khó (hình dạng, chất liệu bề mặt, góc nhìn 3D, số lượng, phom dáng).*

#### Chứng cứ từ soi ảnh tổ hợp 3 thuộc tính (`EDIR complex`, `shape`, `viewpoint`, `cardinality`, `texture`):
- **`qid=4336`** (*"gold-plated tuning pegs + wood grain texture on neck + small wrench"*): Model khớp đúng **2 chi tiết rời rạc** (`khóa vàng ✓` + `cờ lê nhỏ ✓`), nhưng **BỎ SÓT hoàn toàn `vân gỗ`** (giữ nguyên đàn ukulele màu trắng trơn của REF!).
- **`qid=4501`** (*"pale peach tone + hexagonal shape + translucent porcelain"*): Model khớp đúng **`màu đào ✓` + `sứ ✓`**, nhưng **BỎ SÓT `hình lục giác`** (trả về cốc trụ tròn!).
- **`qid=4549`** (*"view shifted left + lighter green shade + slight curvature as if wrapped around a globe"*): Model khớp đúng **`màu xanh nhạt ✓`**, nhưng **BỎ SÓT cả `độ cong 3D quả cầu` lẫn `dịch góc nhìn`** (trả về bản đồ phẳng hình chữ nhật!).
- **`qid=727`** (*"book with a slightly rounded cover"*): `10/10` BLIP-2 giữ nguyên chi tiết **kẹp sách dọc** nổi bật nhưng **BỎ SÓT hoàn toàn `góc bìa bo tròn`** (chọn quyển sổ góc vuông sắc cạnh!).
- **`qid=670`** (*"tabletop with a different shape"* trên bàn tròn): `8/10` ID + `5/7` CIRR vẫn chọn **bàn tròn** thay vì bàn vuông/chữ nhật!
- **`qid=2135`** (*"Reduce the number of Corsair Dominator RAM modules to one"*): Model không phân biệt được **2 thanh RAM** vs **1 thanh RAM**!
- **`qid=923`** (*"duckling covered in water droplets"*): Model chọn ảnh có vòng nước xung quanh nhưng **BỎ SÓT `micro-texture giọt nước trên thân vịt`**!

#### Chứng cứ định lượng — Tháp Chính Xác 13 Thuộc Tính Thị Giác Trên FashionIQ (`6,016` queries × `13` modern runs):
Sắp xếp từ **dễ bị bỏ sót / fail nhất** đến **được giữ tốt nhất** (Overall `R@10 = 53.63%`):

| Hạng khó | Nhóm thuộc tính thị giác / ngôn ngữ | `n` | `Mean R@10` | `Δ vs Overall` | `ALL13 Fail@50` | Mức độ hay bị bỏ sót |
|---:|---|---:|---:|---:|---:|---|
| **#1 (Kém nhất)** | **Phong cách / Trang trọng (`casual/formal/dressy/sexy/plain`)** | 780 | **`42.13%`** | **`-11.50%`** | `12.18%` | Rất hay bị bỏ sót / hiểu mơ hồ |
| **#2** | **Phom dáng / Ôm-rộng (`tight/loose/fitted/flared/a-line`)** | 482 | **`46.03%`** | **`-7.61%`** | `11.62%` | Hay bị bỏ sót khi đi kèm màu |
| **#3** | **Phủ định tường minh (`no/not/without/don't`)** | 664 | **`47.22%`** | **`-6.41%`** | `10.99%` | Hay bị đảo ngược / bỏ qua |
| **#4** | **Độ dài trang phục (`longer/shorter/mini/midi/maxi/crop`)** | 2,206 | **`48.50%`** | **`-5.13%`** | `9.93%` | Hay bị bỏ sót khi không có màu |
| **#5** | **So sánh tương đối (`more/less/-er`)** | 4,133 | **`49.16%`** | **`-4.48%`** | `9.97%` | Sụp mạnh (`38.1%`) nếu thiếu màu |
| **#6** | **Sắc độ sáng/tối (`darker/lighter/brighter/pastel`)** | 2,176 | `50.72%` | `-2.92%` | `9.38%` | Trung bình yếu |
| **#7** | **Kiểu tay áo (`sleeveless/short/long sleeve/strapless`)** | 2,279 | `51.68%` | `-1.96%` | `8.73%` | Dễ bị bỏ sót khi mẫu bám dáng REF |
| #8–#10 | Cổ áo (`54.9%`), Vị trí bộ phận (`55.5%`), Khuy/nút/thắt lưng (`57.0%`) | 573–654 | `54.9%–57.0%` | `+1.3%..+3.4%` | `6.5%–8.9%` | Tương đối ổn |
| #11–#13 *(Tốt nhất)* | **Họa tiết (`57.3%`) → Màu tuyệt đối (`58.4%`) → Chất liệu (`61.1%`)** | 227–4,335 | **`57.3%–61.1%`** | **`+3.6%..+7.5%`** | `5.8%–7.5%` | **Được ưu tiên cao nhất trong embedding** |

---

### Hiện Tượng #3 — Tráo Buộc Thuộc Tính — Vật Thể & Hiểu Nghĩa Đen (`Attribute-Object Misbinding & Homonym Hallucination`)
*Khi câu lệnh gán thuộc tính `A` cho bộ phận/vật thể con `B` (chứ không phải toàn bộ đối tượng chính), biểu diễn gộp toàn cục làm **rơi mối liên kết `A → B`** và gán nhầm `A` sang toàn ảnh!*

- **FashionIQ `shirt:1811`** (*"is short sleeves with **black writing**"* trên nền áo sơ mi **trắng**): Ảnh GT là áo phông **trắng** có chữ **đen** (`BUNGIE`). Consensus #1 của các model lại là **áo phông ĐEN có chữ TRẮNG (`POLO JEANS`)**! Model nhận đủ 2 khái niệm `"black"` + `"writing"` nhưng **tráo ngược quan hệ gắn kết (binding)** giữa màu áo và màu chữ!
- **PinPoint `query_07355`** (*"Simplify the look to just the top **snake chain**"* — kiểu dây chuyền mắt dẹt trơn trên cổ): `5/10` BLIP-2 trả về dây chuyền có **mặt hình con rắn (reptile pendant)**! Model không nối từ `"snake chain"` với sợi dây đang hiện diện trong ảnh REF mà giải nghĩa văn bản độc lập theo nghĩa đen phổ biến nhất trong tập tiền huấn luyện!

---

### Hiện Tượng #4 — Lệch Cân Bằng Phương Thức Hai Chiều (`Two-Sided Modality Imbalance`)
*Composition trong CIR đòi hỏi giữ đúng **phần cần giữ** của `I_ref` và đổi đúng **phần cần đổi** của `T`. Trên ảnh thực tế, các model liên tục ngã về **một trong hai cực đoan**:*

1. **Cực đoan A — Quá bám Ảnh Gốc, Bỏ Sót Văn Bản (`Visual-Anchor / IP / Silhouette Dominance`):**
   - **Bám bản sao / cận bản sao (`Ref-Copy`):** `29.7%` Rank-1 BLIP-2 & `68.4%` Rank-1 CLIP trên PinPoint (`query_03562`, `query_06830`, `query_07626`, `query_02949`).
   - **Bám nhân vật / thương hiệu / text phủ trên ảnh gốc (`IP & Overlay-Text Sticky Trap`):**
     - `shirt:982`: REF là áo hoạt hình *Adventure Time* xanh; yêu cầu *"darker and less graphics"*. Model đổi màu sang đen nhưng **cố sống cố chết bám nhân vật *Adventure Time*** (`ZOMBIE TIME`, `PUNCHA YO BUNS`) — hình in còn to hơn cả REF!
     - `toptee:1146`: Model bám chặt hình **bốt điện thoại Doctor Who TARDIS** thay vì đổi sang áo xanh nhạt.
     - `edir qid=3030`: Yêu cầu *"wider angle, revealing more of the background"*, nhưng model bám chặt **banner chữ *"Learn Easy Punjabi Dance Steps"*** trên REF nên không chọn được ảnh góc rộng sạch chữ!
   - **Bám chất liệu / dáng gốc đè bẹp yêu cầu đổi (`Material/Silhouette Inertia`):**
     - `shirt:794` (*"graphic design + navy blue + shorter sleeves"* trên áo thể thao dài tay khóa kéo): Model giữ chặt dáng áo thể thao/khóa kéo và **bỏ sót cả `"graphic design"` lẫn `"shorter sleeves"`**!
     - `pinpoint query_02949` (*"...same silhouette... converting entire exterior to wood"* trên tòa nhà bê tông kính): `9/10` copy ảnh gốc, `6/10` Non-Ref #1 trả về **tòa nhà bê tông kính khác** (giữ nguyên bê tông, bỏ sót hẳn *"wood"*!).
2. **Cực đoan B — Quá bám Văn Bản, Xóa Sạch Bản Sắc Ảnh Gốc (`Text Dominance / Visual-Identity Erasure`):**
   - **`pinpoint query_01866`** (*"Add subtle blue accents here"* trên **căn bếp màu cam rực**): GT là căn bếp **vẫn màu cam** có thêm điểm nhấn xanh lam. Nhưng `5/10` BLIP-2 lại trả về **căn bếp màu TRẮNG ốp gạch xanh** (explicit negative!) — chỉ vì *"bếp trắng + gạch xanh"* là cặp khái niệm phổ biến nhất trong CLIP/BLIP-2 cho từ khóa `"blue + kitchen"`, nên nó **xóa sổ luôn màu cam chủ đạo của ảnh REF**!

---

### Hiện Tượng #5 — Thiên Vị Thứ Tự Vế Văn Bản (`Caption Primacy Bias` / Bỏ Sót Vế Sau)
Kiểm chứng định lượng trên `6,016` truy vấn FashionIQ (mỗi query gồm 2 câu độc lập `C1` và `C2`, đa số model ghép chuỗi `"C1 and C2"`):

| Điều kiện xuất hiện thuộc tính màu tuyệt đối | `n` | `Mean R@10` (`13` runs) | `Mean R@50` | `ALL13 Fail@50` |
|---|---:|---:|---:|---:|
| **Cả `C1` và `C2` đều có màu** | 1,638 | **`64.59%`** | `84.82%` | `3.79%` |
| **Chỉ `C1` (vế đầu) có màu** | 1,383 | **`56.27%`** | `77.59%` | `6.80%` |
| **Chỉ `C2` (vế sau) có màu** | 1,314 | **`53.06%`** (`-3.21%`!) | `74.98%` | `7.38%` |
| **Cả `C1` và `C2` đều KHÔNG có màu** | 1,681 | **`41.24%`** | `62.00%` | **`15.76%`** |

→ Dù vai trò ngữ nghĩa của `C1` và `C2` là **hoàn toàn đối xứng** trong dữ liệu FashionIQ, các mô hình lại cho kết quả **cao hơn `+3.21%` khi thông tin quan trọng nằm ở `C1` so với `C2`** (xem thêm `shirt:698` nơi model bám `C1` và bỏ sạch `C2`). Điều này hé lộ **thiên vị vị trí token (positional decay / primacy bias)** trong bộ mã hóa văn bản & Q-Former khi xử lý nhiều mệnh đề ghép nối!

---

### Hiện Tượng #6 — Kẹt Trong Chủng Loại Vật Thể Gốc, Không Thể "Nhảy Không Gian" (`Cross-Category & Pragmatic Manifold Lock`)
*Hình ảnh trên `pp_2_complement_explore.png` và `pp_3_swap_contextfit.png` minh họa hiện tượng này cực kỳ sắc nét — đây là ranh giới lớn nhất giữa CIR học thuật cổ điển và CIR ứng dụng đời thực (như tìm kiếm trên Pinterest/thương mại điện tử):*

- **Ảnh chân dung makeup → Phối trang phục (`query_02301`, `query_02302`):** REF là cận mặt trang điểm mắt khói; GT là ảnh **toàn thân mặc đồ dự tiệc tối**. Model trả về **`REF-Copy`** và **một khuôn mặt cận cảnh mắt khói khác** — hoàn toàn không thể bước ra khỏi chủng loại *"khuôn mặt"* sang *"trang phục"*!
- **Chiếc váy → Bảng màu (`query_00753`, *"color palette from this dress"*):** REF là váy lolita hồng pastel; GT là **thẻ bảng màu pastel**. Model trả về **một chiếc váy lolita hồng khác** — không thể tách thuộc tính trừu tượng (`color palette`) ra khỏi vật thể mang nó (`dress`)!
- **Bát smoothie → Cửa hàng bán nó (`query_02072`, *"Where can I buy this smoothie bowl?"*):** GT là ảnh **quầy quán nước / cửa tiệm**. Model trả về **một bát smoothie cận cảnh khác**!
- **Phòng ăn ghế cao → Bố trí ngồi bệt (`query_00452`, *"Redesign this dining area for floor seating"*):** GT là **bàn thấp kiểu Nhật ngồi đệm sàn**. Model trả về **bàn ăn ghế cao khác** — không hiểu phép biến đổi **chức năng / không gian (functional & structural layout shift)**!

---

### Hiện Tượng #7 — Bỏ Sót Suy Luận Tương Đối Có Điều Kiện Ảnh Gốc (`Pseudo-Relative Composition`)
- Như đã chứng minh định lượng ở Đợt 1 & 2 (`p = 2.32e-70` trên FashionIQ, và lý giải trực tiếp vì sao `dress` `48.2%` luôn thua `toptee` `58.0%`): các model chủ yếu hoạt động như **bộ lọc thuộc tính tuyệt đối từ văn bản** cộng neo mờ từ ảnh gốc, thay vì thực sự **đọc trạng thái hiện tại của `I_ref`** (ví dụ: váy hiện tại dài đến đâu, màu sáng cỡ nào, họa tiết rậm cỡ nào) rồi tính phép dịch chuyển có hướng (`darker`, `shorter`, `less formal`, `wider flare`).

---

### Hiện Tượng #8 — Bỏ Sót Ngữ Nghĩa Khi Diễn Đạt Tự Nhiên / Trừu Tượng (`Lexical Concreteness Bias & Paraphrase Fragility`)
- Từ thí nghiệm tự nhiên trên `258` nhóm ý định trùng khớp của PinPoint (Đợt 2): giữ nguyên 100% ảnh gốc và ảnh đích, nhưng **chỉ thay đổi ngôn từ** đã khiến `20%–29%` nhóm ý định lật từ đúng sang sai hoàn toàn (`mAP` dao động `71%–92%` điểm trung bình của chính model!).
- Model thưởng lớn cho **từ khóa cụ thể ngắn gọn** (`+0.0634` khi nêu tên màu, `p=0.0023`; `+0.0164` khi `≤6` từ, `p=0.0042`) và **phạt nặng ngôn ngữ tự nhiên trừu tượng** (`-0.0224` khi xuất hiện các từ *aesthetic/approach/idiom/elements/vibe/rendered*, `p=0.0242`).

---

## 3. Bản Đồ Các "Bài Toán Mở" Đáng Nghiên Cứu (Để Bạn Tự Chọn Hướng & Tự Nghiên Cứu Cách Sửa)

Thay vì áp đặt sẵn một kiến trúc hay công thức cụ thể, bảng dưới đây đóng gói **6 bài toán mở sạch nhất, có chứng cứ hình ảnh & thống kê rõ ràng nhất** để bạn tự cân nhắc xem bản thân hứng thú với khía cạnh bản chất nào nhất của CIR:

| # | Bài toán mở (Problem Space) | Mô hình đang hỏng cụ thể ở đâu? | Bộ dữ liệu **SẠCH** nhất để nghiên cứu & đo kiểm | Câu hỏi khoa học mở để bạn tự đào sâu tìm hướng sửa |
|---|---|---|---|---|
| **1** | **Hiểu Phủ Định, Xóa Vật Thể & Loại Trừ (`Negation & Deletion in CIR`)** | Đảo ngược thành "Con Voi Hồng" (chọn đúng vật bị cấm: *pink graphic*, *US Constitution*, *tie-dye*, *horror movie*, *antique/line drawing*). | **EDIR** (`remove` + `105` câu exclusion ẩn) + **PinPoint** (`Negation`, tách riêng `NonRef-ExplicitNeg`) + **MA-CIR** (ICCV 2025). | *Làm thế nào để biểu diễn truy hồi phân biệt rạch ròi giữa khái niệm "được nhắc đến để GIỮ/THÊM" và khái niệm "được nhắc đến để XÓA/LOẠI TRỪ"?* |
| **2** | **Độ Trung Thực Tổ Hợp Đa Thuộc Tính & Chống Bỏ Sót Hình Học (`Multi-Attribute & Geometric Fidelity`)** | Khi sửa 2–3 thuộc tính cùng lúc, màu/vật thể rời nuốt chửng **hình dạng (`shape`), góc nhìn 3D (`viewpoint`), số lượng (`cardinality`), micro-texture (`texture`), phom dáng (`fit`)**, và bị **tráo buộc thuộc tính (`black writing → black shirt`)** + **thiên vị vế đầu (`C1 > C2`)**. | **EDIR** (`complex`, `viewpoint`, `shape`, `cardinality`, `texture`) + lát cắt sạch **FashionIQ** (`fit`, `length`, `attribute binding`). | *Tại sao nén toàn bộ tổ hợp về 1 vector toàn cục làm rơi mất hình học, số lượng, vị trí và mối liên kết thuộc tính–vật thể? Cần cấu trúc biểu diễn hay cơ chế đối chiếu nào để bảo toàn đủ mọi vế?* |
| **3** | **Cân Bằng Giữa Bảo Toàn Ảnh Gốc & Độ Đàn Hồi Chỉnh Sửa (`Preservation–Plasticity & Shortcut Dilemma`)** | Hai thái cực: (a) **Quá bám `I_ref`** → `Same-Scene Lock` trên EDIR (`CondR@1` tụt `-46%` khi fine-tune), `Ref-Copy` `30–68%` trên PinPoint, bám IP/nhân vật/banner chữ; (b) **Mất `I_ref`** → bếp cam biến thành bếp trắng (`query_01866`). Tạo nghịch tương quan **`ρ = -0.758`** giữa EDIR và PinPoint. | **EDIR** (`In-Domain` vs `CIRR Transfer`, chỉ số `CondR@1` vs `Scene@1`) + **PinPoint** (`RefCopy@1` + `24× Scale Retention`). | *Làm sao một hệ thống biết chính xác phần nào của `I_ref` cần giữ cứng, phần nào cần buông bỏ, và tránh đường tắt nhận diện ảnh gốc trong quá trình huấn luyện contrastive?* |
| **4** | **Truy Hồi Phi Đẳng Cấu / Nhảy Chủng Loại & Ý Định Thực Dụng (`Cross-Category & Pragmatic CIR`)** | Kẹt cứng trong chủng loại ảnh gốc: *makeup → mặt makeup khác* (không ra outfit), *váy → váy khác* (không ra bảng màu), *bát smoothie → bát khác* (không ra cửa hàng), *bàn ăn → bàn cao khác* (không ra ngồi bệt). | **PinPoint** (`Complement`, `Explore`, `Context Fit`). *(Lưu ý: ngay cả xóa sạch negative điểm vẫn kẹt `0.09`, nên đây là bài toán suy luận ý định & truy hồi thô!)* | *Khi ảnh đích không hề giống ảnh gốc về hình thái pixel (chỉ chia sẻ phong cách, bảng màu, hoặc quan hệ chức năng/bổ trợ), cơ chế suy luận trung gian nào có thể bắc cầu giữa hai không gian thị giác khác nhau?* |
| **5** | **Suy Luận Tương Đối Có Điều Kiện Ảnh Gốc & Kháng Paraphrase (`Grounded Relative Reasoning & Linguistic Robustness`)** | Gãy `-26.2%` khi thiếu màu tuyệt đối (`comparative-only`), lệch ngôn ngữ tự nhiên/trừu tượng (`-0.0224`), dao động `71–92%` điểm mAP chỉ vì đổi cách diễn đạt cùng một ý định. | **PinPoint** (`258` nhóm paraphrase, chỉ số `ling_sens_range`) + **FashionIQ** lát cắt sạch `comparative_WITHOUT_color`. | *Làm sao để mô hình thực sự "đọc" trạng thái hiện tại của `I_ref` để neo hóa các từ tương đối (`darker`, `shorter`, `wider flare`) và đưa mọi cách diễn đạt đồng nghĩa về cùng một không gian ý định bất biến?* |
| **6** | **Kiểm Toán & Làm Sạch Chuẩn Đánh Giá CIR (`Debiased Evaluation & Noise-Aware CIR`)** | ~80% ca fail sâu nhất của FashionIQ là lỗi đảo ngược Ref↔Target, bịa thuộc tính, rác MTurk hoặc false negative; giao thức đánh giá đang ẩn `RefCopy` trên `val_split/full_gallery` nhưng lộ `22–50%` trên `original_split`. | Toàn bộ hệ thống 3 benchmark (**FashionIQ** + **EDIR** + **PinPoint**). | *Đánh giá các mô hình CIR hiện tại bị méo bao nhiêu bởi nhiễu nhãn đơn đích & giao thức ẩn ảnh gốc? Một bộ tiêu chuẩn chẩn đoán đa trục sạch cần đo những gì?* |

---

## 4. Tóm Tắt Gợi Ý Cách Chọn Hướng Đi Cho Cá Nhân Bạn

Nếu bạn muốn chọn **1 hướng tập trung nhất** để bắt đầu đọc sâu paper và tự nghiên cứu cách khắc phục:
- **Muốn một vấn đề sắc nét, dễ thấy tận mắt, ai đọc cũng gật gù ngay, benchmark cực sạch (`EDIR` + `PinPoint` + `MA-CIR`):** Bắt đầu từ **Bài toán #1 (Negation & Deletion)** hoặc **Bài toán #2 (Multi-Attribute / Geometric Drop & Attribute Binding)**.
- **Muốn một vấn đề mang tính bản chất sâu về cơ chế học biểu diễn (Representation Learning & Contrastive Dynamics):** Bắt đầu từ **Bài toán #3 (Preservation–Plasticity & Same-Scene Shortcut, giải thích trực tiếp `ρ = -0.758`)**.
- **Muốn một hướng rất gần ứng dụng thực tế & làn sóng VLM/Reasoning 2025–2026:** Bắt đầu từ **Bài toán #4 (Cross-Category / Pragmatic CIR)** kết hợp **Bài toán #5 (Grounded Relative Reasoning & Paraphrase Invariance)**.

---

## 5. Tài Liệu Liên Quan Trong Cùng Bộ Nghiên Cứu

- **Đợt 1** — Tổng quan 6 điểm gãy phổ quát trên 16 model (`86` runs): [CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md](CIR_UNIVERSAL_FAILURE_MODES_AND_RESEARCH_DIRECTIONS.md)
- **Đợt 2** — Kiểm định thống kê (`p-values`), động học `history.jsonl`, can thiệp phản thực (`NoRef` / `RRF`) & bản đồ SOTA 2024–2026: [CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md](CIR_DEEP_RESEARCH_FORENSICS_AND_ROADMAP.md)
- **Đợt 4** — Bản đồ toàn diện mọi trục thất bại hệ thống vượt ngoài composition (Scaling-Resistance, Calibration/Uncertainty, OCR, Hubness/Rarity, Fairness, Long-Tail/Negative Transfer): [CIR_FRONTIER_SYSTEM_FAILURE_LANDSCAPE.md](CIR_FRONTIER_SYSTEM_FAILURE_LANDSCAPE.md)
- **Bảng tổng hợp số liệu gốc 16 model (`73/73` runs bit-exact):** [MASTER_BENCHMARK_SUMMARY.md](../20261123-edir-pinpoint-splits/MASTER_BENCHMARK_SUMMARY.md)
