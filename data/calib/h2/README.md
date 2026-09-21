# h2, multi-field classification

Several independent choice questions asked of one state, which is where a per-question temperature has to hold up across fields that differ in difficulty.

| | |
|---|---|
| Source | NVD CVE 2.0 feed 2023 · [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) config `simplified` · [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) `test` |
| Licence | public domain, US Gov/NIST (NVD) · Apache-2.0 (GoEmotions) · CC-BY-4.0 (banking77) |
| Label source | derived from the CVSS v3.1 vector recorded by NVD (nvd2023) · human (GoEmotions, banking77) |
| Rows | 712 nvd2023, superseded by 1012 nvd2023_v2 · 200 goemotions_sentiment · 400 banking77 |
| Split | assigned by the collector; `split_hint: null` for NVD |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h2.py
```

SHA-256 of the rebuilt files:

```
f17d1c91abeb142f939d3f45bffe733d56685906639f233e067356c5ad8cd44c  nvd2023.jsonl
bbd0777f95125cc9e2cd15648d027faec72ea302e3fb0432f0dce0cac731f109  nvd2023_v2.jsonl
5edb5dca52ad538e893171920cae319831075ad6bac68b506a78a3cd5d217777  goemotions_sentiment.jsonl
5a29583c6c0b990d5c42889461dfeb868dba3d8203b63e854eb60e764ada9d6e  banking77.jsonl
```

**Reading the numbers.** `nvd2023.jsonl` is superseded by `nvd2023_v2.jsonl`; the old file stays on disk so earlier numbers remain checkable, and results state which one they used. GoEmotions rows are `short_state: true`, single Reddit comments, so a long-context claim cannot be made from them.
