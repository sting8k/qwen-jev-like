# H2 — Choice nhỏ (3–77 option), multi-field — calibration data

4 dataset, 1.712 dòng + catalog. Rebuild: `.venv-data/bin/python data/build/phase3_build_h2.py` (seed 42; NVD đọc `data/raw/nvdcve-2.0-2023.json.gz` đã tải, banking77 đọc CSV từ `data/raw/banking77_test.csv`).

**`group_id`** (mọi dòng): NVD = CVE-ID → 2–3 field của cùng CVE luôn cùng split khi Dax chia FIT/SELECT/TEST (chống leak fit-severity/test-attack_surface cùng CVE); dataset 1-field = id.

**UPDATE 19:37 UTC:** bản hiện hành là **`nvd2023_v2.jsonl`** (1.012 dòng = 712 choice + 300 dòng `cvss_base_score` qtype=score scale 0..9, label=min(round(baseScore),9), "9 = dải 8.5–10.0", cùng group_id CVE-ID). Bản `nvd2023.jsonl` cũ (712 dòng) FROZEN trên đĩa cho đợt collect trước của Dax — không ghi đè; mọi đợt sau đọc bản v2. Builder: `data/build/phase3_build_h3h4.py`.

## nvd2023.jsonl — 712 dòng (300 CVE × 2–3 field) — [SUPERSEDED by nvd2023_v2.jsonl] — bản THẬT của preset `code_security`
- Source: NVD CVE 2.0 feed 2023 (public domain, US Gov/NIST) · `label_source: derived` từ CVSS v3.1 vector do NVD analyst gán · `split_hint: null` (Dax chia).
- state = description + CVE-ID + published date. **KHÔNG có CVSS vector/score/severity string trong state** (chống leak nhãn — Biscuit chốt).
- Fallback: 8 CWE readable names chọn theo tần suất pool (CWE-79 2.584, CWE-89 1.104, CWE-352 953, CWE-862 939, CWE-787 880, CWE-22 567, CWE-125 531, CWE-416 440); generic CWE-16/200/... bị loại.
- Lọc pipeline: 28.722 CVE 2023 → bỏ 1.414 không có CVSS v3.1 → bỏ 766 AV:Adjacent → **bỏ 309 desc chứa CVSS tường minh** (regex `CVSS[\s:v]*\d|AV:[NALP]/|base score` case-insensitive — chống leak nhãn vào state; từ "critical"/"high" đơn lẻ được GIỮ vì là tín hiệu thật; Biscuit chốt 18/09) → dedup template (normalize version + prefix-60, giữ ≤2/template; bỏ 8.197) → bỏ 1.802 desc <25 token, 14 >800 → pool 20.756 → chọn 300.
- **Vendor bias do anti-leak filter:** 309 CVE bị drop chủ yếu là Microsoft/Oracle (desc có chữ CVSS/AV tường minh) → pool thiên về CVE plugin/OSS nhỏ; nếu severity dist của bộ này khác NVD toàn cục thì đây là lý do.
- Stratify: LOW ép 30/300 (10%); 270 còn lại phân bố tự nhiên → CRITICAL 43 / HIGH 105 / MEDIUM 122.
- Phân bố label:
  - severity (300): P2_MEDIUM 122, P1_HIGH 105, P0_CRITICAL 43, P3_LOW 30 — min 10% ✓
  - attack_surface (300): REMOTE_UNAUTH 128, REMOTE_AUTH 95, LOCAL 77 — min 26% ✓; AV:A đã drop, options chỉ 3 nhãn derive được (không SUPPLY_CHAIN/NONE)
  - cwe_category (chỉ CVE có CWE trong top-8): **Biscuit chốt 18/09 — giữ top-8; ECE là số toàn cục nên đuôi 5–6 mẫu/lớp không phá; KHÔNG báo per-class metric cho lớp <10 mẫu**
- Ghi chú: state lặp 2–3 dòng/CVE là multi-field theo thiết kế (không phải dedup miss).

## tickets — **KHÔNG CONVERT — dừng sau khi phát hiện synthetic** (2026-09-18)
- Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0) card tự khai: "Synthetic IT Ticket Generator — Custom Dataset", quảng bá dịch vụ sinh data (open-ticket-ai.com). Nhãn queue/priority/type do máy sinh → vi phạm quy tắc "không nhận nhãn LLM-sinh" của brief. Bảng ứng viên của Gizmo đã ghi sai `label_source: human` — lỗi của Gizmo, đã báo Biscuit. Đang chờ quyết định thay thế (xem reports/candidates_h1_h2.md mục tickets).

## goemotions_sentiment.jsonl — 200 dòng (50/lớp) — field `sentiment` của `support_triage`
- Source: [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) config `simplified` split `train` · License: **Apache-2.0** (mirror không gated; bản google/goemotions bị gated).
- Rule gộp 27→4 (label_source: derived, rule deterministic trên nhãn human gốc): anger|disgust→ANGRY; annoyance|disappointment→FRUSTRATED; neutral→NEUTRAL; joy|gratitude|approval|admiration→SATISFIED. Strict: drop dòng có emotion ngoài rule-set hoặc multi-label xung đột sau gộp (bucket còn: ANGRY 1.561 / FRUSTRATED 2.239 / NEUTRAL 12.823 / SATISFIED 7.959).
- Lọc: text ≥10 token; **mỗi lớp lấy đúng 50 (stratified)** — `stratified: true` trong từng dòng; phân bố pool gốc rất lệch (ANGRY 1.561 / FRUSTRATED 2.239 / NEUTRAL 12.823 / SATISFIED 7.959) → sample cân bằng 4 lớp KHÔNG phản ánh tần suất tự nhiên, người đọc ECE cần biết. `short_state: true` toàn bộ (Reddit comment ngắn — Biscuit chốt). `known_benchmark: true`.

## banking77.jsonl — 400 dòng (77 option) — topic/intent + H4 fallback
- Source: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) test split · License: **CC-BY-4.0**. Data gốc tải từ PolyAI-LDN/task-specific-datasets (GitHub) vì repo HF chỉ có loader script; đã assert tên intent khớp dataset_infos.json.
- 77 options = full intent catalog (lưu riêng `../h4_fallback/banking77_catalog.json`). `label_source: human`, `split_hint: test`, `short_state: true` (query ngắn là bản chất dữ liệu), query ≥5 token. `known_benchmark: true`.

## Validation đã chạy
Label ∈ options mọi dòng choice; state 5–800 token (short_state flag riêng); không state trùng trong từng dataset; mọi dataset đơn ≥100 dòng.
