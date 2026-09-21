# H1 — Noul (true/false) với passage — calibration data

3 dataset, 1.100 dòng. Format: xem `docs/briefs/PHASE3_DATA_BRIEF.md` mục 1. Rebuild tất cả: `.venv-data/bin/python data/build/phase3_build_h1.py` (seed 42, deterministic; HF datasets cache trong `~/.cache/huggingface`). Token đếm bằng Qwen3.5-9B-AWQ tokenizer local (`state_tokens` ghi trong từng dòng).

**`group_id`** (mọi dataset): định danh state để chia FIT/SELECT/TEST theo nhóm — dataset 1-field thì `group_id = id`; không bao giờ 2 dòng cùng state rơi vào 2 split khác nhau.

## boolq.jsonl — 300 dòng
- Source: [google/boolq](https://huggingface.co/datasets/google/boolq) · License: **CC-BY-SA-3.0** · split: `validation` (giữ nguyên làm `split_hint`).
- Lọc: state 50–800 token (bỏ 306 ngắn, 1 dài); dedup state trùng (2.963→2.658); **stratified theo answer** (`stratified: true` trong từng dòng — Biscuit chốt 18/09): phân bố pool gốc sau lọc True 1.648 / False 1.010 (62/38) → sample True 186 / False 114. Đọc ECE cần biết sample hơi cân hơn tự nhiên.
- `question_desc` = câu hỏi yes/no gốc verbatim; `label` bool; `label_source: human`; `known_benchmark: true` (contamination có thể — model có thể đã thấy khi pretrain).

## pubmedqa.jsonl — 400 dòng
- Source: [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) config `pqa_labeled` · License: **MIT** · split_hint: `train` (HF chỉ có 1 split; pqa_labeled là bộ expert-labeled 1.000).
- Lọc: drop `maybe` (110 dòng); state = abstract (`context.contexts` join), 50–800 token (0 dropped).
- `question_desc` = câu hỏi PICO gốc; `label` bool từ `final_decision`; `label_source: human`; `known_benchmark: true`.

## scitail.jsonl — 400 dòng
- Source: [allenai/scitail](https://huggingface.co/datasets/allenai/scitail) config `tsv_format` split `validation` · License: **Apache-2.0** (LICENSE trong github allenai/scitail).
- Lọc: bỏ premise <10 token (84); `short_state: true` cho mọi dòng (premise 1 câu, bản chất dữ liệu — Biscuit chốt 2026-09-18).
- state = premise; `question_desc` = `Does the passage entail: "<hypothesis>"`; `label` = entails→true / neutral→false; `label_source: human`; `known_benchmark: true`.

## Validation đã chạy
Mọi dòng: label ∈ domain (bool/enum), 10 ≤ state_tokens ≤ 800 (trừ nơi có short_state flag ≥10), không state trùng trong dataset, mọi dataset đơn ≥100 dòng ≥ quota.
