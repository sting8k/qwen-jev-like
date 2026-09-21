# h6 — diagnostic probes

Rows built to expose specific failure modes: answering with no content at all, the same question under permuted option order, and letter-only labels.

| | |
|---|---|
| Source | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) config ARC-Challenge split `test`; the Vietnamese rows are machine-translated BoolQ |
| Licence | CC-BY-SA-4.0 |
| Label source | human (ARC, BoolQ); the translation is machine-produced and recorded as such |
| Rows | 150 content_free · 400 permutation_arc · 100 label_name_arc · 100 label_name_arc_v2 · 50 vi_boolq |
| Split | **none** — every row is `probe: true` and is excluded from FIT, SELECT and TEST |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h5h6.py
```

SHA-256 of the rebuilt files:

```
d93021800bc333f0ec8e49b02480f03da6d74405d5c274501f4cbf4f583ceb5f  content_free.jsonl
ff6305e91cbaa120cd9c309956b64507197e20e2c506ab6b1920022c6ff166b4  permutation_arc.jsonl
081a270d37c0eb2b742d016ca13ab8da27adbfeed6ded9ba57bd0448e600ebd7  label_name_arc.jsonl
d1bda67f93cfd20caf21e65a2462bf5de54e3c371eba6b70f352c2d93f0d24cf  label_name_arc_v2.jsonl
e722bf2ea2e7aebba6e8be6882b679516224fcd077ecc7aacd6df923a4532282  vi_boolq.jsonl
```

**Reading the numbers.** These rows never enter a fit or a test set; they diagnose the model, they do not score it. `label_name_arc.jsonl` presents options as bare letters with no mapping in the state, so nothing can identify what a letter means and its accuracy sits at chance — `_v2` carries the mapping and is the one to use. The first file is kept because published numbers refer to it.
