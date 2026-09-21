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
| h8 | CVSS v3.1 metric prediction | CC-BY-NC-SA-4.0 | **no — see below** |
| bugreport_cls | bug-report classification, 12 weakness families | reports belong to their finders | **no — see below** |

The last two ship a builder whose docstring carries the licence and the
provenance, but not the per-set README the others have. That is a gap, and it is
written here rather than smoothed over: the table above would otherwise read as
if every set were documented to the same standard.

`bugreport_cls` has a second wrinkle worth stating. The builder was renamed
during publication and its row ids changed with it (`bugreport-<id>-<question>`),
so **a rebuild today does not hash to the file the published numbers were
measured on** — the contents differ by exactly that prefix. The counts are
unaffected: 300 reports, 991 rows. No digest is recorded for it here, because
the only digest that has actually been measured belongs to the file with the
old ids, and recording that one would invite a comparison that must fail.

`build/bugreport_cls_manifest.json` is committed and carries, per report, the
labels, the weakness family, whether an attachment was present, and whether the
text names its own program under each of two rules — ids and flags, no text.
That manifest is what a table means when it cites an `n`.

## The bench presets are not here either

`build/fetch_bench_presets.py` pulls the four latency presets from their
upstream author at a pinned revision and verifies each against a recorded
SHA-256. Same posture: linked, hash-checked, not vendored.
