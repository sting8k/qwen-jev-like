# H3 — Score / ordinal — calibration data

1 dataset mới + 1 dataset mở rộng (nằm ở h2). Rebuild: `.venv-data/bin/python data/build/phase3_build_h3h4.py` (seed 42). Engine cap 10 mức 1 chữ số → mọi scale nằm 0..9 (chốt Biscuit/Dax 18/09).

## amazon_reviews_en.jsonl — 500 dòng
- Source: [SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) split `test` (5.000) · License: **Apache-2.0** (card) · Amazon Reviews Multi (McAuley 2020), language en.
- State = review **body only** (title bị LOẠI — title thường lộ sao kiểu "Terrible"); lọc 30–400 từ; dedup text.
- `qtype: score`, `question_key: stars`, `question_desc`: "Star rating the reviewer gave, 1 = worst, 5 = best" · `scale {min:1, max:5}` · label = số sao thật người đánh giá (`label_source: human`).
- **Stratified 100/sao** (`stratified: true`; pool sau lọc: 1★377 / 2★462 / 3★450 / 4★415 / 5★351 — gần đều tự nhiên; sample cân bằng tuyệt đối). `split_hint: test`, `known_benchmark: true`, `group_id = id`. `short_state: true` cho 198 dòng <50 token.
- Bản gốc Amazon Reviews Multi có train 200k nếu cần mở rộng.
- **Nhiễu nhãn nguồn (không sửa data)**: `amazon-en-en_0941552` có state rõ ràng tích cực ("Fantastic belt ,love it . ... this one is perfect") nhưng label = 1. Kiểm phân bố: nhãn 1 đều tiêu cực, nhãn 5 đều tích cực → 1/500 dòng nhiễu của bộ gốc, không phải lệch hệ thống. Giữ nguyên để số công bố tái lập được.
- **Lưu ý khi thu lại**: engine đánh số mức 0-based (`0: 1`, `1: 2`, ...) trong khi `question_desc` nói "1 = worst, 5 = best" → model trả lời bằng số sao, token `'5'` rơi ngoài tập, 103/500 dòng mất ~70% mass (xem docs/results/PHASE3_RESULTS.md §5, backlog P1).

## NVD CVSS base score (file h2/nvd2023_v2.jsonl, không phải ở đây)
- Cùng 300 CVE với NVD H2, thêm dòng thứ 4 `qtype: score`, `question_key: cvss_base_score`, `scale {min:0, max:9}`, **label = min(round(CVSS v3.1 baseScore), 9)** — "9 = dải 8.5–10.0". `label_source: derived`, `group_id` = CVE-ID (4 dòng/CVE cùng split).
- Phân bố label score (300 CVE): 2★:1 · 3:20 · 4:28 · 5:45 · 6:54 · 7:15 · 8:57 · 9:80 — 0 và 1 không có trong mẫu (CVE có CVSS <3.0 hiếm trong pool đã lọc). Đuôi ordinal mỏng → không dùng per-bin metric cho bin <10 mẫu.

## STS-B — BỎ (license "other"/SemEval murky; quota đã đạt).

## sst5.jsonl — 400 dòng (80/mức) — thêm 2026-09-19

- Source: [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) split `test` (2.210) · License: **không ghi trên card** (SST-5 = Stanford Sentiment Treebank, Socher et al. 2013) · `label_source: human` · `known_benchmark: true`.
- `qtype: score`, `scale {min:0, max:4}`, nhãn **đã 0-based sẵn trong nguồn** (không remap gì).
- Lệnh tái lập: `.venv-data/bin/python data/build/phase3_build_score.py` (seed 42, lọc 5–400 từ, lấy đều 80 dòng/mức rồi xáo).
  `id`/`group_id` dùng **sha1(state)[:7]** — bản đầu tiên (19/09, trước commit 443c359) lỡ dùng `hash()` của Python vốn bị muối ngẫu nhiên mỗi process nên id KHÔNG tái lập được; đã dựng lại, chỉ id đổi, 400 dòng (state,label) y nguyên.

**Mục đích: phép A/B tách bug P1 (remap thang) khỏi "model kém score".**
Cùng họ bài toán với `amazon_reviews_en.jsonl` (chấm cảm xúc một bài đánh giá), cùng thiết kế cân bằng,
`question_desc` cùng dạng ("… 0 = worst, 4 = best" vs "… 1 = worst, 5 = best") — **khác đúng một biến: gốc thang**.
- amazon tệ + sst5 ổn ⇒ lỗi nằm ở remap 1-based.
- cả hai tệ ⇒ model yếu score thật.

**Cảnh báo khi đọc kết quả — độ dài state KHÔNG khớp** (SST-5 là câu đơn, amazon là đoạn):

| | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| sst5 | 400 | 5 | 14 | 21 | 30 | 63 |
| amazon | 500 | 32 | 44 | 56 | 80 | 298 |

→ Nếu hai bộ chênh nhau, **độ dài là cách giải thích cạnh tranh**. Để loại nó, so trong dải chồng lấn
30–63 token: sst5 có 107 dòng, amazon có 294 dòng.

**Không dùng (có chủ ý)**: cột `label_text` của nguồn (`very negative / negative / neutral / positive /
very positive`). Đưa tên mức vào `question_desc` sẽ đổi hai biến cùng lúc và làm hỏng phép A/B. Giữ lại
cho phép thử sau: "mức mô tả bằng chữ" vs "mức là chữ số" — xem `reports/candidates_jev_oracle.md` §Phát hiện 1.

## sst5_desc.jsonl — 400 dòng, CÙNG 400 dòng của sst5.jsonl + `scale_labels` — thêm 2026-09-19

- Khác `sst5.jsonl` **đúng 2 trường**: `id` (thêm hậu tố `-desc`) và `scale_labels`
  (`["very negative","negative","neutral","positive","very positive"]`, index khớp `scale` 0..4, đúng brief §1).
  `group_id` **giữ y hệt** → cả hai bộ rơi **cùng split** (đã kiểm 400/400: FIT 163 / SELECT 77 / TEST 160) và pair được theo dòng.
- `question_desc` cố ý để **nguyên như sst5** — nếu đổi luôn nó thì lại nhích 2 biến cùng lúc.

### Ba biến tách được khi có đủ 3 bộ

Catalog mà engine thực sự dựng (`core/jev_engine.py:222` render `chỉ_mục: mô_tả`,
`run_calib_collect.py:182` đặt mô tả = chính chữ số):

| bộ | gốc thang | mức hiện trong catalog | lệch chỉ mục↔nghĩa |
|---|---|---|---|
| `amazon_reviews_en` | 1 | `0: 1` `1: 2` … `4: 5` | **có** — trả `0` để nói "1 sao" |
| `sst5` | 0 | `0: 0` `1: 1` … `4: 4` | không (nhưng mức vẫn là số trần) |
| `sst5_desc` | 0 | `0: very negative` … `4: very positive` | không, và mức **có nghĩa** |

- amazon vs sst5 → tách **lệch off-by-one / gốc thang**.
- sst5 vs sst5_desc → tách **mức-số-trần vs mức-có-mô-tả** (docs TypeSafe: cùng bug report, mức chữ cho
  score 0.0/conf 1.0, mức `["0","1","2"]` cho score 0.57/conf 0.35 — xem `data/jev_examples/typesafe_docs/score.json`,
  example `score_numeric_levels_docs_bugreport`).
- Vẫn giữ cảnh báo độ dài state ở mục trên khi so với amazon.

> **ĐÃ NỐI** (c6219b3, 19/09 00:28): collector đọc `scale_labels` khi dòng có trường này
> (`run_calib_collect.py:182`). Kiểm chứng bằng chạy thật, không phải đọc code: `test_jev_cpu.py` mục **4c** dựng
> prompt thật qua `_shared_prompt` và khẳng định catalog chứa `0: very negative` cho `sst5_desc`, còn `amazon` vẫn ra
> `0: 1` — ALL PASS. Collector cũng chặn cứng hai kiểu hỏng: số nhãn ≠ số mức, và nhãn lại đúng bằng chữ số (bộ có
> nhãn mà render y như bộ không nhãn thì sau này không ai phân biệt nổi hai lần collect).
>
> Nhắc lại phần **chưa** sửa: `amazon_reviews_en.jsonl` không có `scale_labels` nên vẫn render `0: 1 … 4: 5` —
> lệch off-by-one đó là bug P1 của engine, chưa đóng.

## amazon_desc.jsonl — 500 dòng, CÙNG 500 dòng của amazon_reviews_en.jsonl + `scale_labels` — thêm 2026-09-19

Duyệt: Biscuit 19/09 18:43 (đang chờ user xác nhận lại). Khác `amazon_reviews_en.jsonl` **đúng 2 trường**:
`id` (+`-desc`) và `scale_labels = ["1 star","2 stars","3 stars","4 stars","5 stars"]`. `group_id` giữ y hệt →
đã kiểm **500/500 cặp cùng split** (FIT 198 / SELECT 93 / TEST 209). Dựng bằng cách **đọc file đã có trên đĩa**,
không tải gì mới, **không đụng** `amazon_reviews_en.jsonl` (đã diff xác nhận nguồn nguyên vẹn).

### Tại sao có hai file amazon — ô 2×2

`amazon_reviews_en` một mình **trộn 2 biến**: mức là chữ số trần, VÀ chỉ mục lệch nghĩa (`scale 1..5` nhưng catalog
luôn `enumerate` từ 0 → `0: 1 … 4: 5`). Nếu nó tệ hơn `sst5` thì không tách được do *thiếu nghĩa* hay do *lệch chỉ mục*.

| | mức = **chữ số trần** | mức = **có nghĩa** |
|---|---|---|
| **0-based**, chỉ mục khớp nghĩa | `sst5` → `0: 0 … 4: 4` | `sst5_desc` → `0: very negative …` |
| **1-based**, chỉ mục lệch nghĩa | `amazon_reviews_en` → `0: 1 … 4: 5` | `amazon_desc` → `0: 1 star … 4: 5 stars` |

- **Hàng ngang** (cùng dataset, pair theo `group_id`, cùng split, cùng độ dài state) = phép so **sạch**: gắn nghĩa
  vào mức đáng giá bao nhiêu.
- **Hàng dọc** = có gỡ được tác hại của lệch chỉ mục không.
- So **chéo** `sst5` với `amazon` vẫn phải giới hạn dải chồng lấn **30–63 token** (sst5 107 dòng / amazon 294),
  không thì độ dài state (median 21 vs 56) là biến nhiễu.

Catalog thật của `amazon_desc` (dựng bằng `_shared_prompt`, không phải suy từ code):

```
- stars (score): Star rating the reviewer gave, 1 = worst, 5 = best
    0: 1 star
    1: 2 stars
    2: 3 stars
    3: 4 stars
    4: 5 stars
```

> **Điều kiện thu** (Biscuit + Dax 19/09): cả 4 bộ phải thu trong **một lần gọi collector duy nhất**, `JEV_APC=0`.
> Không chỉ vì nhiễu: với APC bật, cache mang trạng thái xuyên dataset nên bộ thứ hai chạy trong trạng thái cache
> khác bộ thứ nhất — **thứ tự liệt kê `--datasets` trở thành một biến**. Tắt APC thì tất định tuyệt đối (Dax đo
> 307/307 giống hệt bit qua 2 tiến trình). Thu cùng một lần còn để đóng băng chung phiên bản collector, prompt
> template và tokenizer.

### Bất biến bắt buộc khi thu bộ có cặp biến thể: hai arm phải ở HAI FILE RIÊNG

`group_states()` gom dòng theo **đúng văn bản state** (`run_calib_collect.py:160`), mà cặp biến thể của tao
(`sst5`↔`sst5_desc`, `amazon_reviews_en`↔`amazon_desc`) **dùng chung 100% state** — đó là điều kiện để pair theo
dòng. Nhưng collector gọi `group_states` **bên trong vòng lặp từng file** (`:528-541`: `for path in files: rows =
load_dataset(path) ... groups = group_states(rows)`), nên dòng của hai file khác nhau **không bao giờ** rơi chung
một nhóm — bất kể `question_key` đặt thế nào.

| tình huống | gom nhóm | hậu quả |
|---|---|---|
| hai arm ở **hai file** (hiện tại) | mỗi file gom riêng | ✅ an toàn, `question_key` đặt sao cũng được |
| hai arm bị **trộn vào một file**, `question_key` khác nhau | cùng state → **một nhóm cỡ 2** | ❌ **một call chứa cả hai arm**, model thấy cả `0: 0…4: 4` lẫn `0: very negative…` |
| hai arm trộn một file, `question_key` giống nhau | trùng qid → tách thành singleton | ✅ thoát nhờ chốt trùng qid |

Tức bất biến thật là **giữ hai arm ở hai file**, không phải "giữ `question_key` giống nhau" — bản ghi trước của
mục này nói sai điều đó, vì tao dựng phản chứng bằng cách tự gộp rows hai file rồi mới gọi `group_states`, một
đường chạy collector không hề đi (Dax chỉ ra, đã kiểm lại code).

Cái đáng sợ vẫn nguyên: nhánh hỏng **không báo lỗi gì** — vẫn đủ dòng, vẫn ra số hợp lý, chỉ là hai arm đã nhìn
thấy catalog của nhau. Cùng họ với bug khoá gom nhóm mù thứ tự option (`0c7634b`): **khoá gom nhóm không nhìn
thấy thứ phân biệt các biến thể**. Guard lúc thu kiểu "một nhóm chứa nhiều dataset" **không chặn được** ca này
(`_ds` lấy từ tên file nên luôn giống nhau trong một nhóm) — nên chốt chặn phải nằm sau khi thu.

**Phép kiểm sau khi thu** (bắt buộc, chạy trên chính file kết quả):

1. Mỗi cặp cùng `group_id`: hai arm phải **KHÁC nhau** — và so **cả thứ tự khoá** của dict logits, không chỉ giá
   trị. Thứ tự khoá mang thứ tự catalog thật đem chấm; đó là thứ lộ bug `0c7634b` ở 300/300 dòng trong khi so giá
   trị chỉ bắt được 291/300.
2. Mỗi arm: **số dòng kết quả = số call**, mỗi dòng đúng một `question_key` của arm mình. Nếu hai arm lọt chung
   một file thì một state sinh hai câu hỏi trong một call, và số call sẽ lệch so với dự tính — bắt được đúng ca
   mà guard lúc thu bỏ sót.
