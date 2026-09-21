# H7 — DAIR Emotion — bộ duy nhất có số Jev đo bởi bên thứ ba

1 dataset, 2.000 dòng. Rebuild: `.venv-data/bin/python data/build/phase3_build_h7_emotion.py`
(tái lập **byte-identical**, đã kiểm: sha256 `8650ddca…b9a8` hai lần chạy).

**Phê duyệt:** user 2026-09-21 — "OK NC, chỉ Emotion". Review: Biscuit (peer-34e) cùng ngày.
AG News nằm trong cùng đề xuất nhưng user **chưa duyệt** — không tải.

## Câu hỏi mở mà bộ này trả lời (§7 policy: data mới phải nêu câu hỏi)

Trước bộ này, "ta đứng đâu so với Jev và các bản open (laya/kev/openjev/simple-jev)"
chỉ trả lời được trên 4 case Cloudflare, **direction-only**. DAIR Emotion là chỗ duy nhất
trong cụm dùng chung mà Jev **được đo là hiệu chuẩn kém** — đúng đại lượng Phase 3 tồn tại để đo:

| nguồn | Jev | ghi chú |
|---|---|---|
| AbdelStark/jev-benchmarks (BTZSC pilot-v1, n=100) | acc **0.480**, Brier 0.846, NLL 5.588, **16% ví dụ p=0 ở nhãn đúng** | `jev-1.13.0`, pre-registered, bootstrap ghép đôi |
| laya BENCHMARKS.md | Jev 0.480 (trích lại AbdelStark) · laya 0.595–0.600 | |
| kev `transfer-v4` | Emotion nằm trong suite OOD | kev tự đo Jev qua gateway |

## Tiêu chí ĐĂNG KÝ TRƯỚC — chốt khi CHƯA có số nào

Biscuit chốt 2026-09-21, đã vào brief §1 (commit `2ee4e00`) thành luật chung cho **bộ nhãn nhiễu**.
Ghi ở đây để không ai đổi tiêu chí sau khi nhìn thấy kết quả:

| | h7/emotion |
|---|---|
| **ECE** | **BÁO, KHÔNG GATE.** Nhãn nhiễu thì ECE là ECE so với nhãn nhiễu — không đủ tư cách làm cổng đậu/rớt |
| **Cổng 1** | **Selective**: bỏ 20% kém tin nhất thì accuracy tăng **≥ 5 điểm** |
| **Cổng 2** | **Hướng** so với Jev trên subset `btzsc_sample` |

Tức h7 **không** dùng ngưỡng ECE < 0.05 của brief §7 — đó là ngưỡng cho bộ nhãn người.

## emotion.jsonl — 2.000 dòng = TOÀN BỘ test split

- Source: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) config `split`, split `test`.
- License: card ghi **`other`** (dùng cho nghiên cứu/giáo dục, không có quyền tái phân phối)
  → mọi dòng mang `license_nc: true`. **User chấp nhận NC 2026-09-21** cho mục đích đo + công bố bảng số;
  ta **không** tái phân phối text (file nằm trong repo nội bộ; nếu publish thì publish số, không publish data).
- `qtype: choice`, 6 option = đúng tên nhãn nguồn `["sadness","joy","love","anger","fear","surprise"]`.
  `known_benchmark: true`, `split_hint: test` (split của NGUỒN; FIT/SELECT/TEST của ta vẫn
  là `hash(group_id)` lúc fit).
- **`label_source: derived` — KHÔNG phải `human`.** Card khai `annotations_creators: machine-generated`;
  dữ liệu đã được pipeline CARER (Saravia et al., EMNLP 2018) tiền xử lý, nhãn do máy sinh từ khâu
  thu thập chứ không do người gán. Cùng lớp với NVD CVSS→severity ở h2.
  **Không vi phạm** luật "không nhận nhãn LLM-sinh" (2018, pattern/graph, không có LLM).
  *(Bản commit đầu b8bc569 ghi nhầm `human` — Mario tự phát hiện khi đọc card, sửa trước khi thu GPU.)*
- ⚠️ **NHÃN NHIỄU — đọc số thấp cho đúng.** Mọi bên đều thấp ở đây: Jev 0.480, laya 0.595–0.600,
  và kev liệt kê "noisy-label emotion" là một trong bốn khoảng cách còn lại của họ.
  Hệ quả: accuracy thấp là **trần của bộ dữ liệu**, không chỉ là model kém; và ECE đo trên nhãn nhiễu
  là ECE **so với nhãn nhiễu** — đừng đọc thành "model hiệu chuẩn kém" mà không nói kèm điều này.
- `id` = `group_id` = `emotion-test-<sha1(state)[:7]>` — một câu hỏi mỗi text, không có text trùng (2.000 unique).
- **`stratified: false` CÓ CHỦ Ý — đây là MỤC ĐÍCH của bộ này, không phải chi tiết kỹ thuật:
  h7 là bộ calib ĐẦU TIÊN đo ECE trên prior thật của dữ liệu.** Mọi bộ trước (h2/goemotions
  50/lớp, h3/sst5 80/mức, banking77) đều ép cân bằng, nên ECE của chúng đọc trên một thế giới
  không tồn tại. Prior nguồn giữ nguyên:

  | joy | sadness | anger | fear | love | surprise |
  |---:|---:|---:|---:|---:|---:|
  | 695 | 581 | 275 | 224 | 159 | 66 |

  ECE phải đọc trên đúng prior dữ liệu có. Đây là chỗ khác `goemotions_sentiment` (h2) vốn ép 50/lớp.
  **Hệ quả bắt buộc (Biscuit 2026-09-21): KHÔNG đọc metric per-class.** `surprise` chỉ có 66 dòng,
  qua TEST 20% còn **≈13 dòng** — chỉ đọc số tổng, đúng luật đã áp cho cwe_category ở h2 (<10 mẫu/lớp).
- `short_state: true` **toàn file** (tweet ngắn: token min 3 / trung bình 19.9 / max 62).
  Đặt theo tính chất file như goemotions/banking77, không đặt theo ngưỡng từng dòng
  (audit yêu cầu cờ có-tất-cả hoặc không-có-dòng-nào).
- **31 dòng < 5 token** (ngắn nhất 3) — **giữ có chủ ý để đồng nhất item với BTZSC**
  (`31 rows < 5 tokens kept for item parity`). Luật của h2 là 5–800 token; đây là chỗ h7
  **cố ý lệch**, Biscuit gật 2026-09-21. Audit không chặn.
  ⚠️ Cờ `short_state: true` đặt toàn file nên **nó che 31 dòng này** — đừng tưởng đã lọc sót.

## Công bố: MỨC A — số ra, text ở lại (user chốt 2026-09-21)

License card: *"The dataset should be used for educational and research purposes only."*
→ cho phép **dùng**, không cho phép **phát lại**. User duyệt hai bước riêng:
**NC để ĐO** (2026-09-21) rồi **NC để CÔNG BỐ ở mức A** (2026-09-21).

| | ra ngoài được | KHÔNG ra ngoài |
|---|---|---|
| **Mức A (đã duyệt)** | bảng số; `data/build/phase3_build_h7_emotion.py`; `id`/`group_id` (sha1 của state); sha256 file | 2.000 dòng `state` (text tweet) |

Không mất gì về khoa học: builder tái lập **byte-identical**, ai cũng dựng lại đúng file từ nguồn gốc.
Đây đúng cách nibzard/decision-model-benchmark xử lý SMS Spam ("that license does not cover republication":
công bố số + script, bôi text khỏi archive).

**CHECKLIST TRƯỚC KHI PUSH CÔNG KHAI — chưa làm, làm lúc publish:**
`emotion.jsonl` **đang nằm trong git** (tracked, không ignore) và là file `license_nc` **duy nhất** như vậy.
Repo hiện **không có remote** nên chưa có gì rò. Nhưng `git push` lên repo công khai một lần là
2.000 dòng text đi theo, và **lịch sử git vẫn giữ kể cả khi xoá sau** (commit b8bc569, d24849c, d8672d4).
**Biscuit chốt 2026-09-21:** text ĐÃ nằm trong lịch sử, nên mức A lúc publish nghĩa là
**repo TÁCH hoặc `git filter-repo`** — **không phải** "xoá file rồi push". `git rm --cached` một mình
KHÔNG cứu được, vì blob cũ vẫn trong lịch sử và đi theo lần push đầu tiên.
Chưa làm gì bây giờ; đang treo chờ user chọn cách khi thực sự publish. Ai làm thì sửa luôn dòng này.

## Dòng đối chiếu Jev: cờ `btzsc_sample` trên 100 dòng

**Sự thật đã kiểm** (assert trong builder, chạy lại mỗi lần rebuild):
`btzsc/btzsc` config `emotiondair` @ `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` **chính là**
2.000 text này ở dạng NLI (2.000 × 6 hypothesis, đúng 1 entailed mỗi text):
**2.000/2.000 text trùng, 0/2.000 lệch nhãn**. Nghĩa là các dòng AbdelStark chấm Jev
là **tập con của file này**.

**Thứ KHÔNG tái lập được:** mẫu 100 dòng của họ do code của họ rút và manifest **không công bố**
(`results/runs/` bị gitignore trong repo họ). 100 dòng gắn cờ `btzsc_sample` ở đây là
**bản rút của TA** theo **seed của họ** (`20260917`, từ `configs/pilot-v1.yaml`), cân bằng lớp
17/17/17/17/16/16 (joy, sadness, anger, fear, love, surprise).

→ **Cùng nguồn, cùng revision, cùng cỡ, cùng thế cân bằng — KHÔNG đảm bảo cùng item.**
Báo dòng này là **n=100 → tín hiệu** (§5: n<50 là tín hiệu; n=100 vẫn chưa là kết quả),
không bao giờ báo là head-to-head trên item đồng nhất.

**Hai bẫy khi đọc:**
1. Subset-100 **cân bằng lớp**, file đầy đủ theo prior tự nhiên → accuracy hai bên **không so được với nhau**.
   Số của Jev (0.480) là trên bản cân bằng, nên dòng subset mới là dòng đối chiếu.
2. 100 dòng đó rơi vào cả FIT/SELECT/TEST theo `hash(group_id)`. Dòng nào rơi FIT/SELECT thì
   **T đã fit có thấy nó** → trên dòng đối chiếu chỉ báo **accuracy**; ECE/Brier chỉ báo trên TEST của ta.

## Câu hỏi/prompt

`question_desc: "Emotion the author of this tweet expresses"` — bám sát template hypothesis của BTZSC
("This example tweet expresses the emotion: X") để prompt so được, trong khi catalog vẫn là catalog của ta
(bất biến §4.1: nhãn xuất hiện verbatim — 6 tên nhãn viết thường đúng như nguồn, không đổi chữ).
Câu Jev mà AbdelStark gửi là câu chung chung của họ ("Which single label best describes the input text?"),
**khác câu của ta** — một lý do nữa để dòng đối chiếu là tín hiệu, không phải phán quyết.

## Đã kiểm

- Rebuild 2 lần → sha256 trùng khít (byte-identical).
- `data/build/phase3_audit_calib.py`: h7/emotion **0 hard fail, 0 warn**
  (5 hard fail còn lại của repo là h5/h6 có sẵn từ trước, không liên quan).
- Nhãn ∈ options mọi dòng; 2.000 id duy nhất; tương đương BTZSC assert trong builder.
