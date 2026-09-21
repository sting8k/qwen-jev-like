# bugreport_cls, bug-report classification

Disclosed security reports with the decision a human triager actually recorded, and the weakness family they assigned.

| | |
|---|---|
| Source | HackerOne Hacktivity, disclosed reports only, fetched through the public API |
| Licence | each report belongs to its finder and the program that received it; nothing is redistributed here |
| Label source | human — the recorded state and weakness of the report |
| Rows | 991 rows over 300 reports (is_valid, outcome, weakness, severity) |
| Split | none — this set is evaluated whole, not fitted on |

Rebuild, from the repository root:

```sh
python data/build/phase3_build_bugreport_cls.py
```

SHA-256 of the rebuilt file:

```
8d2b725a8d16520a4406e1bd3ead4d852833050d0612848344891e21799b7e71  bugreport_cls_300.jsonl
```

**The digest above is derived, not measured.** It is the digest of the file the current builder produces, arrived at by transforming the measured corpus rather than by running the builder. The measured corpus hashes to `ecbbb7609229dd9e80637be644bedd6084318e2679725a8d0e3b6fe52d934ab1`; the two files differ in their row ids, 991 of them and 3964 bytes, and in nothing else. The 300 reports, the 991 rows and every label are identical.

**Reading the numbers.** The report text contains live credentials, which is the reason no row is committed anywhere. 140 of 300 reports have an attachment the model never sees, and 155 name their own program in the text; `build/bugreport_cls_manifest.json` carries both flags per report so a table can say which subset its `n` refers to.
