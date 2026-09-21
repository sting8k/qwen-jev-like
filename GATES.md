# Pre-publication gates

Three greps run before anything is pushed. Two must come back empty; the third
returns matches by design, and every one of them is accounted for here. A gate
whose exceptions are re-argued from memory each time stops being a gate, so the
reasons live in this file and are reviewed when they change.

## 1. Private scenario material — must be 0

```sh
rg -in 'gorren|hessa|apothecary|airlock|dose|injection_note|\bgame ?[12]\b|\bplayer\b' -g '!.git' .
```

The two interactive scenario benches are the author's own design. This
repository carries their **numbers and criteria** and nothing else: no world, no
mechanics, no action names. `games/` and the scenario briefs were never copied.

Expected: **0**. Known false positives if the pattern is widened: `proc.kill()`
in `core/bonsai_llm.py` and `tests/test_cpu.py`, and `kill` in `gpu.sh` — process
teardown, not scenario vocabulary. Read matches; a count alone cannot tell a
leak from a method call.

Three comments in the engine and the fork worker used to name that harness. They
were reworded. Their technical content and every measurement in them
(930 ms against 274 ms for single-slot eviction; LCP 1058 within a call versus
535 across calls) are unchanged — the naming leaked, the evidence did not.

## 2. Live credentials — must be 0

```sh
rg -n 'sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY' -g '!.git' .
```

The bug-report corpus contains live secrets in its report text, which is why no
row of it is committed anywhere: `data/build/` rebuilds it into `data/raw/`,
which is ignored. This gate is the backstop for that rule, not the rule itself.

Expected: **0**.

## 3. Application framing — matches allowed, each with a reason

```sh
rg -in '\bsoc\b|\btier\b|analyst|replace|deploy' --type py --type md .
```

The register of this repository is descriptive: tasks are classification
problems with human labels, and measurements are properties of a system. There
is no claim about what the numbers qualify anything to do, in either direction —
an argument that a system is *unfit* for some role is still an argument about
that role.

Current matches and why they stay:

| Match | Where | Why it is not framing |
|---|---|---|
| `.replace(` | `core/bonsai_llm.py`, `run_*.py`, `data/build/*` | Python string method. |
| `support_triage` | `presets.py`, `run_bench*.py` | Name of a published upstream latency preset; renaming it would break the comparison it exists for. |
| `Tier 1 Enterprise`, `deploy`, `analyst` | `presets.py` fixture text | Inside the customer-support and fraud fixtures themselves — the content being classified, not a claim about the classifier. |
| `deploy` | `examples/langchain/*.md` | Verbatim snapshot of third-party documentation, kept as an upstream record. |
| `analyst` | `data/build/phase3_build_h8_ctivsp.py`, `data/calib/h2/README.md` | Names who produced the labels (NVD analysts; the CVSS vector author). Label provenance is a property of the data and has to be stated. |
| `triager` | `data/build/phase3_build_bugreport_cls.py` | Same reason: the `substate` label *is* a decision a human triager recorded. Dropping the word would obscure where the label came from. |

Dataset naming follows the same rule: the bug-report set is
`bugreport_cls` throughout — builder, manifest, row ids and results — and is
described by what it is, "bug-report classification, 12 weakness families,
human-labelled".

## 4. The CPU gate must pass in a fresh clone

```sh
python -m tests.test_cpu
```

Not on a machine that happens to have model weights. A clone has no `models/`
directory, so the tokenizer falls back to the upstream vocabulary
(`Qwen/Qwen3.5-9B`, Apache-2.0, ~22 MB of vocabulary files, no weights), and the
run prints which tokenizer it loaded.

Because every assertion in that file is about where token boundaries fall, the
vocabulary is pinned by fingerprint rather than by trust. The upstream
vocabulary and the local AWQ checkout were compared and carry the same
fingerprint `4ba4dbcd1fab5671` over 248077 tokens, so the fallback tests the
same question. A different tokenizer fails the first check by name instead of
quietly changing what the other 63 checks mean.

Current: **64 checks, 64 ok, 1 skipped**. The skip is the score-level check,
whose corpus is not shipped; it prints the reason and the builder that recreates
it. Skips are counted separately and never absorbed into the pass count.
