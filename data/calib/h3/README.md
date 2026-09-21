# h3, rubric scores on an ordered scale

Sentiment on an ordered scale, which is the only question type here where being wrong by one level differs from being wrong by four.

| | |
|---|---|
| Source | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) `test` · [SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) `test` |
| Licence | Apache-2.0 (card) · SST-5 card records no licence; treated as measure-only |
| Label source | human (SST-5) · derived from the star rating (Amazon) |
| Rows | 400 sst5 · 400 sst5_desc · 500 amazon_reviews_en · 500 amazon_desc |
| Split | assigned by the collector; `split_hint: test` |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_score.py
```

SHA-256 of the rebuilt files:

```
97c7f9cc915e4c18acd223b4b4415efbc250b98bca104b2882f513a7edf86969  sst5.jsonl
e4681ea8bde57c3eef87ae201b5d50f4e9bce3991bc508df40d6076d80833cfc  sst5_desc.jsonl
34db0d2f13977dc82902e73c9189c10182f3bc3099101d6eaf68a716ccc06651  amazon_reviews_en.jsonl
82e1a093614a3b8f94c5719445d9299aaa4b755383825dde7555e1eb3f31fd75  amazon_desc.jsonl
```

**Reading the numbers.** The `_desc` files are the same rows with named levels instead of bare digits, and they exist because a bare digit `'5'` fell outside the scored token set on 103 of 500 rows, losing mass. Score accuracy has two definitions in this project, within one level, and exact, and they differ by roughly 0.35 on the same rows, so every score number states which it used.
