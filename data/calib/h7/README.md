# h7, emotion labels

Single sentences with one emotion label each; the largest single set here, and the one used to fit the choice temperature.

| | |
|---|---|
| Source | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) config `split`, split `test` |
| Licence | the card states `other` — research and educational use, no redistribution |
| Label source | **derived**, not human — the labels come from hashtags, not from annotators |
| Rows | 2000 emotion |
| Split | assigned by the collector; `split_hint: test` |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h7_emotion.py
```

SHA-256 of the rebuilt file:

```
a962e83c4ea8967858e6d71eef893985e6cac15fc044a3d144019b05c4663d95  emotion.jsonl
```

**Reading the numbers.** The labels are known to be noisy, so an ECE measured here is calibration against noisy labels and is reported as such rather than as a pass criterion. `known_benchmark: true`.
