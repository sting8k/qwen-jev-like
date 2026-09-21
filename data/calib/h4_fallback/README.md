# H4 — Choice lớn — fallback + catalog

Bối cảnh (báo cáo theo brief mục H4): CROSS rulings CBP không có API/bulk công khai hợp lệ (chỉ UI search + scraper bên thứ 3); dataset CROSS trên HF (flexifyai) gated/private không kiểm được; bộ HS PDDL 141k là synthetic (text ≈ copy mô tả HS). → H4 giao theo fallback: **Banking77 (77 option) + CLINC-150 (150 option)** — cho 2 điểm cỡ catalog để đo calib theo N-option (đường tới ceiling 255 của preset tariff).

## banking77.jsonl — 400 dòng (đã nộp từ H2, không đổi)
- CC-BY-4.0, test split, 77 options, human. Xem README h2.

## banking77_catalog.json — 77 intent (REGENERATED: thêm `desc`, additive)
- Mỗi dòng: `{"intent": "<tên nguyên văn>", "desc": "<desc deterministic>"}`.
- **Rule desc deterministic**: `desc = intent.replace("_", " ")` — ví dụ `card_arrival` → "card arrival". Không thêm/bớt nghĩa, không LLM, không đổi tên intent. Rule chốt Biscuit 18/09.
- **Invariant bắt buộc (không đổi)**: chuỗi `options` trong banking77.jsonl giữ NGUYÊN VĂN tên gốc (kể cả `Refund_not_showing_up`, `reverted_card_payment?` — dấu hoa/thường và dấu hỏi giữ y hệt) vì token chấm phải xuất hiện đúng trong catalog. `desc` chỉ là chú thích cạnh option khi dựng prompt.

## clinc150.jsonl — 300 dòng + clinc150_catalog.json — 150 intent
- Source: [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) config `imbalanced` split `test` (5.500) · License: **CC-BY-3.0** (card, không gated).
- Lọc: drop lớp `oos`; drop utterance <5 token; random seed 42 lấy 300 (phân bố intent trong test ~đều, mỗi intent ~3 câu) · state = utterance · `qtype: choice`, options = 150 tên intent nguyên văn, label = intent thật (`label_source: human`) · `short_state: true` (utterance ngắn) · `split_hint: test`, `known_benchmark: true`, `group_id = id`.
- Catalog 150 intent + desc theo cùng rule snake_case→spaces (CLINC không có mô tả chính thức per-intent).

Rebuild cả hai: `.venv-data/bin/python data/build/phase3_build_h3h4.py`.
