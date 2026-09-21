# Methods

*(Protocol, reporting rules and the engine invariants go here. This file is
being written; the section below is complete.)*

## Pre-publication gates

Three greps run before anything is published. Two must come back empty.

```sh
rg -in 'gorren|hessa|apothecary|airlock|dose|injection_note|\bgame ?[12]\b' -g '!.git' .
rg -n  'sk-[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}' -g '!.git' .
rg -in '\bsoc\b|\btier\b|analyst|replace|deploy' --type py --type md .
```

The first keeps the two interactive scenario benches out: this repository
carries their numbers and criteria, never their design. Three engine comments
once named that harness and were reworded — their measurements are unchanged.
The second is a backstop for the rule that no row of the bug-report corpus is
committed anywhere, because its report text contains live secrets.

The third is expected to return matches, and each is a false positive with a
reason: `.replace(` is a string method; `support_triage` is an upstream
latency preset whose name is the comparison; `TIER1`–`TIER3`, `deploy` and
`analyst` appear inside fixture and documentation text that is *being
classified*; `analyst` and `triager` also name **who produced a label** in the
NVD and bug-report sets, which is provenance and has to be stated. The
register of this repository is descriptive — a task is a classification problem
with human labels, and a measurement is a property of a system. There is no
claim about what any number qualifies a system to do, in either direction.

A fourth check is not a grep: `python -m tests.test_cpu` must pass in a fresh
clone, on a machine with no model weights. Currently 64 checks, 64 ok, 1
skipped, the skip being a corpus this repository does not ship. Running it
costs a ~2.5 GB vLLM install even though it is CPU-only, because
`core/jev_engine.py` imports `SamplingParams` at module scope and so the engine
cannot be imported without vLLM even when the backend is llama.cpp. Moving that
construction behind the backend boundary would remove the dependency; it is a
change to the engine that produced the published numbers, so it was not made
during packaging.

## Committed artifacts must be what their generator writes

The bug-report manifest once carried a block the builder never wrote — accurate,
but added by hand, so a rebuild silently produced a file missing the counts that
every table citing an `n` depends on. The builder emits it now, summed from the
rows. The rule generalises: if a committed artifact says something its generator
does not, fix the generator.

A related trap from the same day: the rename that found it had missed that file,
because `§` is stored escaped as `\u00a7` and a literal search for `§4` matched
nothing while the string sat in plain sight. Structured files get edited through
a parser, not by text replacement.
