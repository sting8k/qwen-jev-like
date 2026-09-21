# h4_fallback, intent classification, 150 options

A wide option set: the catalogue dominates the prompt, which is the regime the engine was built for.

| | |
|---|---|
| Source | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) config `imbalanced` split `test` |
| Licence | CC-BY-3.0 |
| Label source | human |
| Rows | 300 clinc150 |
| Split | assigned by the collector; `split_hint: test` |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h3h4.py
```

SHA-256 of the rebuilt file:

```
0d501bdf5f713503a9d97dc084cf8e7d22a9a6710b51ad3815cfc86c3d6036c3  clinc150.jsonl
```

**Reading the numbers.** Rows are `short_state: true` (one utterance). Every option label must appear verbatim in the catalogue, so intent strings are used unchanged and a readable description is derived deterministically from the intent name, no label was invented.
