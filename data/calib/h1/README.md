# h1 — yes/no questions over a passage

Three sets where the answer is true or false and the evidence is in the state itself, so a wrong answer is a reading failure rather than missing knowledge.

| | |
|---|---|
| Source | [google/boolq](https://huggingface.co/datasets/google/boolq) `validation` · [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) config `pqa_labeled` · [allenai/scitail](https://huggingface.co/datasets/allenai/scitail) config `tsv_format` split `validation` |
| Licence | CC-BY-SA-3.0 (BoolQ) · MIT (PubMedQA) · CC-BY-4.0 (SciTail) |
| Label source | human |
| Rows | 300 boolq · 400 pubmedqa · 400 scitail |
| Split | assigned by the collector; `split_hint` carries the source split |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h1.py
```

SHA-256 of the rebuilt files:

```
44569984ae7855c17006c7a543133308e498d8b67ee32d8a591132769b70ff95  boolq.jsonl
aee89aeaabb0b093aecf84cc33ed2e1b86e30a5f5526a72eff217908574f8d74  pubmedqa.jsonl
343460b4c7cd5c6273ff55e81c24e2cb0fb88cd06cfcde9edf5f629bc4993850  scitail.jsonl
```

**Reading the numbers.** BoolQ is sampled stratified by answer, so its true/false balance is more even than the source pool's 62/38 — a prior fitted on it does not see the natural rate. All three are `known_benchmark: true`: the model may have seen them in training, which inflates accuracy but says nothing either way about calibration.
