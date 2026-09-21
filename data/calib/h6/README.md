# H6 — Bộ probe thiên vị (diagnostics, KHÔNG dùng cho FIT/SELECT/TEST)

Mọi dòng có `probe: true` — Dax loại khỏi chia split calib, chỉ dùng chẩn đoán tật model. Sinh deterministic seed 42: `.venv-data/bin/python data/build/phase3_build_h5h6.py` (build content_free/permutation/label_name; vi cần `vi_translations.json` đã có).

## content_free.jsonl — 150 dòng (50 câu × 3 biến thể state)
- 50 câu choice lấy từ banking77.jsonl (seed 42) — options + label giữ nguyên.
- State bị thay bằng: `N/A` / chuỗi rỗng / đoạn lorem ipsum (`variant: na|empty|lorem`).
- Mục đích: đo **prior bẩm sinh** cho mỗi option khi không có bằng chứng — nền để "trừ prior" cho kết quả thật. `group_id` = group của câu gốc.

## permutation_arc.jsonl — 400 dòng (100 câu ARC-Challenge × 4 hoán vị)
- Source: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) config ARC-Challenge split test · License: **CC-BY-SA-4.0** (chốt Biscuit 19:39 — thay MMLU vì license murky).
- 100 câu 4-option (lọc sẵn), `perm_id: 0..3` theo 4 hoán vị cố định [(0,1,2,3), (3,2,1,0), (1,3,0,2), (2,0,3,1)].
- **Options giữ NGUYÊN VĂN text đáp án** (không A/B/C/D) — đo position bias thuần: cùng câu, đúng-sai chỉ đổi chỗ.
- `group_id = arc-XXX` (câu gốc) — so 4 dòng cùng group. `known_benchmark: true`.

## label_name_arc_v2.jsonl — 100 dòng (cùng 100 câu ARC, options A/B/C/D + MAPPING) — [THAY label_name_arc.jsonl]
- **Bản v1 (label_name_arc.jsonl) LỖI THIẾT KẾ, bỏ dùng** (Dax 20:22 + Biscuit 20:23): options là chữ cái nhưng state không có bảng ánh xạ A→nội dung → model không thể biết chữ cái nghĩa gì, accuracy 0.300 ≈ chance, probe không đo được letter bias. File v1 giữ trên đĩa (frozen) chỉ làm bằng chứng.
- **v2**: state = câu hỏi + bảng ánh xạ MMLU chuẩn `A. <text đáp án>` đủ 4 dòng, **thứ tự A–D khớp đúng permutation perm_id=0** (đã assert: group_ids/questions/option-order/label-mapping khớp 100%) → so "chấm text đáp án" vs "chấm chữ cái trên cùng nội dung" = đo chi phí của lớp indirection thuần.
- Validate mới (rule chung từ lỗi này): qtype choice với options trừu tượng (A/B/C/D) → state PHẢI chứa mapping đầy đủ cho mọi option; check chạy trong builder.
- `group_id` trùng permutation (arc-XXX).

## vi_boolq.jsonl — 50 dòng (stretch tiếng Việt)
- 50 câu BoolQ có passage ngắn nhất (50–73 token) — state + question **dịch máy sang tiếng Việt** (`machine_translated: true`, dịch bởi agent Gizmo, lưu `vi_translations.json` để audit).
- **Nhãn giữ nguyên 100% từ bản tiếng Anh** (đã assert khớp 0 mismatch) — probe đo sự sập xác suất khi đổi ngôn ngữ, không đo calib vs nhãn mới.
- `group_id` = id BoolQ gốc (link về bản English). `lang: vi`.

## Cách đọc kết quả (ghi chú cho Dax)
- Content-free: argmax(options) với state trống = prior; kỳ vọng model tốt cho prior gần đều; prior lệch lớn → khử trước khi tin probability thấp.
- Permutation: label đúng phải giữ xác suất cao ở mọi perm_id; drop xác suất khi đáp án đổi vị trí = position bias.
- Label-name: so phân phối giữa text-option và letter-option cùng câu → letter bias.
- VI: so noul probability với bản EN cùng group_id → language shift degradation.
