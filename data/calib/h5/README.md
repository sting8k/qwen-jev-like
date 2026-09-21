# H5 — Fraud textualized — calibration data

## paysim_textualized.jsonl — 400 dòng
- Source: Kaggle [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) (PaySim mobile-money simulator, Lopez-Rojas et al., EMSS 2016) · License: **CC BY-SA 4.0** · tải bằng Kaggle API (token user cấp; token không lưu trong repo).
- 6.363.620 giao dịch (fraud 8.213 = 0.129% base rate) → **stratify 200 fraud / 200 legit** (`stratified: true` — sample KHÔNG phản ánh base rate; base rate gốc ghi ở đây để tham chiếu). Legit lấy qua reservoir 5.000 (seed 42) rồi sample 200.
- **Template deterministic (nguyên văn trong `data/build/phase3_build_h5h6.py`, hằng số `PAYSIM_TEMPLATE`):**

```
Transaction Review: Mobile money {type} of {amount:.2f}. Origin account balance
{oldbalanceOrg:.2f} -> {newbalanceOrig:.2f}. Destination account balance
{oldbalanceDest:.2f} -> {newbalanceDest:.2f}. Time: hour {step%24:02d}:00 of day {step//24+1}.
```

- Chỉ 7 field trên (chốt Biscuit) — KHÔNG suy field khác: nameOrig/nameDest bỏ, isFlaggedFraud bỏ.
- `qtype: noul`, `question_key: is_fraud`, question_desc "Was this transaction fraudulent (simulator ground truth)", `label` = isFraud gốc, `split_hint: null` (nguồn không có split), `group_id = id`, `lang: en`.
- **`label_source` đọc đúng là `simulated`, KHÔNG phải `textualized`** (Biscuit 20:20): PaySim là mô phỏng agent-based từ log thật, isFraud do **luật simulator** gán — "textualized" chỉ mô tả bước biến bảng→văn bản của mình. File JSONL đã freeze nên trường ghi `textualized` — **đọc là `simulated`**; bản rebuild kế tiếp sẽ ghi đúng.
- Lưu ý phương pháp: PaySim là simulator (agent-based) — nhãn "fraud" là ground truth của mô phỏng do luật simulator gán, KHÔNG phải giao dịch thật; dùng được vì nhãn không do LLM sinh và quy trình reproducible. Ghi rõ khi công bố (`label_source: simulated`).
- Kết quả đo (Dax, 18/09): model đoán is_fraud ở mức ngẫu nhiên (0.506) trên bộ này — hợp lý: luật simulator không suy ra được từ văn bản template. Bộ này vẫn dùng được cho đo calibration theo nhóm xác suất, không dùng cho claim độ chính xác.

Rebuild: `.venv-data/bin/python data/build/phase3_build_h5h6.py` (đọc `data/raw/paysim1/paysim1.zip` đã tải).

## Đã bác (lý do trong reports/candidates_h5_h6.md)
- ULB credit card: V1–V28 PCA ẩn danh → template chỉ còn Time/Amount, không đủ chất liệu ngữ nghĩa.
- paysim_mini (GitHub): transactions.csv không có cột isFraud.
- IEEE-CIS: competition, phải accept rules bằng web UI — bỏ.
