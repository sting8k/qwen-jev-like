# Sources & Method (novcog) — H7 snapshot

> Source: https://jev.novcog.us.com/sources — fetched 2026-09-18. Methodology + link list for the novcog analysis.

## Primary sources (novcog's list)

- TypeSafe AI blog: "Introducing System One Models & Jev", Diogo Almeida, 15 Sep 2026.
- typesafe.ai homepage — 193.6x / 444.6x / $42 per 1B input / 238x vs Claude Fable 5.1 / "Zero Hallucinations" / 0.114s vs 8.566s.
- **evals.typesafe.ai** — workflow evals: four workflows, per-model accuracy/cost/time, reference-label statement.
- **docs.typesafe.ai + llms.txt** — Choice/Score/Noul primitives, confidence and calibration.
- Diogo Almeida launch thread X (@CompleteSkeptic) 15 Sep 18:17 UTC — "20-200x faster", "40-400x cheaper".
- **typesafe-ai/system-one-adapter-python** — wrapper constraining LLMs for the comparison harness.
- HN thread **49717558** (15–16 Sep, ~1500 pts, 426 comments).

## Independent measurements

- Jon Reed, Near Here, 16 Sep — "TypeSafe Jev vs Mistral Small and Gemini Flash-Lite for local event validation", jev-1.13.0: 96% / 0.59s / $0.043 per 1k vs Mistral Small 4 (84% / 2.90s / $0.370), Gemini 3.5 Flash-Lite (86% / 3.40s / $2.496).
- Alex Duffy, Good Start Labs, 15 Sep — "Verification is the bottleneck": 6,003 checks, 91.5% agreement with Fable 5.1, ~$160/M graded vs $33,000 (Fable 5.1) / $260 (DeepSeek V4.1 Flash), ~0.5s/call. Duffy states agreement "is not accuracy".
- Mike Taylor, Every, 15 Sep — "Jev judged everything I've written in 0.7 seconds": 777 judgments <0.7s, 0.35s median vs 8.83s Fable, 6/7 planted defects (Fable 7/7), ~580x cheaper.
- typesafeainate/dspy-typesafeify — DSPy fork by TypeSafe employee: 15.9% faster, 30.1% cheaper over 3 test cases.

## Community

- Latent Space / AINews "Jev: a System One model", 16 Sep — TypeSafe previewed at an AI Engineer event.
- Reddit r/singularity, r/accelerate, r/codex.
- **Qwen-2.5-1B-RLCD — a same-day reproduction attempt** (harshatheg) — untested by novcog. ← this is the upstream of our jev-rlcd reference/ code.

## What could NOT be established (novcog, 16 Sep)

- Funding/investors/team size ($40M seed reported by one low-authority site, unconfirmed).
- Diogo Almeida "co-inventor of ChatGPT" — self-description, unverified.
- X replies/quotes — not enumerable without account.
- **Any calibration data: "No reliability diagram, expected calibration error or Brier score has been published for Jev."** ← the gap Phase 3 fills.
- Architecture: "close to the chest", no paper.
- Early-access commenters report 32k context window + a 10-option cap in doc examples vs stated 255-option cardinality — unconfirmed by TypeSafe material.

## What would move claims vendor-stated → measured (novcog)

"A published calibration curve, an architecture paper, open benchmarking by people who did not receive vendor-granted access, or independent timing at scale."
