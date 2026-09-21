# data/jev_examples — sanity oracle: ví dụ Jev CHÍNH THỨC (input + output số thật)

Mỗi file JSON: `_source`, `_fetched`, `_model_version`, **`_tier`**, `_note`, `examples[]`.
Mỗi example: `input {state, questions}` + `published_response {answers}`.
Tên trường trong `examples` **đã chuẩn hóa về contract của mình** (noul/choice/score) — tên gốc
của nguồn ghi trong `_note` (ví dụ Vercel gọi noul là `boolean`, trường số là `probability`).

Tái lập (trừ cloudflare, lấy tay từ trước):
```sh
python3 -m tools.phase3_fetch_jev_examples  # --dry để chỉ đếm, không ghi (chạy từ gốc repo)
```
Chạy 2 lần ra **byte y hệt** (sort_keys + indent cố định); nội dung chỉ đổi khi trang nguồn đổi.

## Tier — mức tin cậy khi so số

| tier | nghĩa | file |
|---|---|---|
| `pinned` | response ghi rõ version model | `cloudflare/jev_examples.json` (jev-1.13.0) |
| `unpinned` | tài liệu chính thức nhưng model là `jev-latest` (đổi khi TypeSafe ra bản mới) | `typesafe_docs/*.json`, `vercel/ai_gateway_guide.json` |
| `weak` | output in ở dạng không phải JSON (python repr), ngữ nghĩa trường khác | `pydantic/typesafe_model_page.json` |

→ Chỉ `pinned` mới nên so trị tuyệt đối. `unpinned` dùng cho **direction check**. `weak` chỉ tham khảo:
pydantic-ai định nghĩa `confidence` là **margin so với ngưỡng** (p=0.01 → báo 0.98), khác hẳn
confidence "độ tản của phân phối" của docs.typesafe.ai — đã lưu thành `confidence_margin` để khỏi so nhầm.

## Số câu theo tier (2026-09-19)

| file | examples | câu | noul/choice/score |
|---|---|---|---|
| cloudflare/jev_examples.json | 4 | 8 | 4/2/2 |
| typesafe_docs/choice.json | 3 | 7 | 0/7/0 |
| typesafe_docs/score.json | 5 | 11 | 0/0/11 |
| typesafe_docs/noul.json | 1 | 2 | 2/0/0 |
| typesafe_docs/quickstart.json | 1 | 3 | 1/1/1 |
| typesafe_docs/api.json | 4 | 4 | 2/1/1 |
| vercel/ai_gateway_guide.json | 2 | 4 | 2/1/1 |
| pydantic/typesafe_model_page.json | 2 | 2 | 1/0/1 |
| **tổng** | **22** | **41** | **12/12/17** |

## Hai example đặc biệt trong `typesafe_docs/score.json`

- **`score_severity_monotonic_table`** (`_monotonic_check: true`): 1 câu hỏi, 5 state xếp theo mức
  nghiêm trọng tăng dần → Jev trả 0.0 / 1.0 / 1.12 / 1.3 / 2.0. **Thứ tự mới là oracle**, không phải trị số.
  Dùng `states[]` thay cho `input.state`.
- **`score_numeric_levels_docs_bugreport`**: cùng bug report, nhưng mức score là chữ số `["0","1","2"]`
  thay cho mô tả → Jev **tản** 0.43/0.57 (score 0.57, conf 0.35) trong khi mức mô tả bằng chữ cho
  score 0.0 / conf 1.0. Đây là bằng chứng Jev cũng chấm digit 0..N-1 và mức-chỉ-là-số yếu theo thiết kế
  (liên quan backlog P1 + `@event.paths-fix-verification-score-levels`).

## Nguồn đã kiểm nhưng KHÔNG lấy

- Vercel: khối `clearAnswers` / `mockJev` là **mock fixture của unit test**, không phải output model.
- `docs/concepts/how-to-build-with-system-one` (7 request), `primitives/advanced` (5), `patterns/*` (4):
  chỉ có input, không có số.
- langchain / pypi langchain-typesafe / llama-index / apidog / dev.to / aihubmix / samelogic: không có response số.

Các file `.md` cùng thư mục (`integrations_snapshot.md`, `mchromiak_snapshot.md`, `novcog/`, `langchain/`,
`typesafe/`, `hn_thread_49717558.json`) là snapshot hệ sinh thái cho `reports/jev_ecosystem.md`, không phải oracle.
