# data/

**No dataset rows live here.** Nothing in this directory is a corpus — it holds
recipes and provenance, and the count of `.jsonl` files under `data/` is zero by
design.

```
build/          the builders. Running one downloads from the original source and
                writes the exact file that was measured, into data/raw/ (ignored).
calib/<set>/    per-set README: licence, source, how labels were produced,
                the split, and the hash to check a rebuild against.
```

The reason is licensing, not tidiness. Several of these sets are published
under terms that permit use and measurement but not redistribution — one is
`CC-BY-NC-SA-4.0`, one card says only "educational and research purposes" — and
one contains live credentials inside the report text it was built from. A
number measured on data you cannot ship is still checkable if the recipe, the
provenance and the hash are shipped instead, so that is what this is.

To reproduce a set: read `calib/<set>/README.md` for what it is and where it
comes from, run the matching builder in `build/`, and compare the hash.

## What is recorded, and what is not

| set | what it is | licence | per-set README |
|---|---|---|---|
| h1 | yes/no questions over a passage | CC-BY-SA-3.0 | yes |
| h2 | CVE attribute classification | Apache-2.0 | yes |
| h3 | review sentiment, rubric scores | Apache-2.0 | yes |
| h4_fallback | intent classification | CC-BY-3.0 | yes |
| h5 | transaction labels | CC-BY-SA-4.0 | yes |
| h6 | multi-field choice | CC-BY-SA-4.0 | yes |
| h7 | emotion labels | card says `other` | yes |
| h8 | CVSS v3.1 metric prediction | CC-BY-NC-SA-4.0 | yes |
| bugreport_cls | bug-report classification, 12 weakness families | reports belong to their finders | yes |

One digest in that set is **derived rather than measured**, and its README says
so: the bug-report builder was renamed during publication and its row ids
changed with it, so the recorded digest is what applying exactly that rename to
the measured corpus produces, not the output of a builder run. The two files
differ in 991 ids and 3964 bytes and in nothing else — same 300 reports, same
991 rows, same labels.

`build/bugreport_cls_manifest.json` is committed and carries, per report, the
labels, the weakness family, whether an attachment was present, and whether the
text names its own program under each of two rules — ids and flags, no text.
That manifest is what a table means when it cites an `n`.

## The bench presets are not here either

`build/fetch_bench_presets.py` pulls the four latency presets from their
upstream author at a pinned revision and verifies each against a recorded
SHA-256. Same posture: linked, hash-checked, not vendored.
