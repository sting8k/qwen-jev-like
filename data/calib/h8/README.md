# h8, CVSS v3.1 metric prediction

Eight independent choice questions per vulnerability description. The severity score itself is a formula over those eight answers and is never asked.

| | |
|---|---|
| Source | [AI4Sec/cti-bench](https://huggingface.co/datasets/AI4Sec/cti-bench) config `cti-vsp` |
| Licence | **CC-BY-NC-SA-4.0** — non-commercial; every row carries `license_nc: true` and no row is redistributed |
| Label source | derived from the CVSS vector published with each CVE |
| Rows | 2400 (300 descriptions × 8 metrics) |
| Split | assigned by the collector |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_h8_ctivsp.py
```

SHA-256 of the rebuilt file:

```
86bbcd742b28235a8ed1c5a350ad0da4caf818cb522d45569d57b88fd4ebba16  cti_vsp_300.jsonl
```

**Reading the numbers.** Two of the eight metrics are so imbalanced that every system measured on them lost to predicting the majority class, so a per-metric table is the only honest one, an average over the eight hides it.
