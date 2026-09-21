# mchromiak: Jev Typed Decisions for Enterprise AI — H7 snapshot

> Source: https://mchromiak.github.io/articles/2026/Sep/17/Jev-Typed-Decisions-for-Enterprise-AI/ (17 Sep 2026) — fetched 2026-09-18.

## Eval numbers (TypeSafe workflow evals, accessed 17 Sep — most granular public readout)

- Jev aggregate: **67.8% mean agreement, $0.0004/case, 0.4s**. GPT-5.6 Terra: 67.9% / $0.0304 / 10.1s (Δ0.1pp, 76× cost, 25× time). GPT-5.6 Sol: 74.1% / $0.0836 / 23.3s. (Vertical axis = agreement with reference = avg GPT-6 Astra + Claude Fable 5.1 high-thinking; workflows written by TypeSafe's own team; latency measured from US West Coast laptops.)
- Per-workflow (Jev): security incidents 61.7% ($0.0001, 0.3s) · agent-trace observability 71.6% ($0.0003, 0.5s) · invoice processing 61.8% ($0.0011, 0.5s) · customer service 76.0% ($0.0001, 0.4s). Invoice: several LLM workflows materially higher than Jev → aggregate ≠ SLA per domain.
- NOTE: direct fetch of evals.typesafe.ai on 18 Sep rendered workflow #1 as "Expense claims" — site content differs from mchromiak's 17-Sep readout (renamed/updated). Cite both with dates.

## Enterprise placement (mirrors our bench shape)

Decision layer at agent boundaries: (1) before LLM — intent classify + specialist routing; (2) after retrieval — passage relevance/contradiction/prompt-injection scoring; (3) before tool call — policy match + review-needed; (4) after generation — response-addresses-request / citation support; (5) after run — trace triage for silent failures.

## Probability semantics (four confusable quantities — quote-worthy)

- Probability: outcome-level, e.g. P(refund ok)=0.83. Confidence: scalar summar of distribution concentration. Correctness: only vs reference outcome. Calibration: do probability estimates match frequencies over many cases.
- **"TypeSafe documents these intended semantics but has not published reliability diagrams, expected calibration error, Brier scores, or an independent calibration study. Treat the returned probabilities as measurements to validate on representative labelled traffic, especially after changing the model version or deployment domain."**

## Production boundaries

- Score = probability-weighted position, 2–10 levels; same score from different distributions → inspect distribution. Not a reconstructed amount.
- Choice option list incomplete → include "none of the above" (model must pick least-wrong otherwise).
- 13-question batched call: 12.2× cheaper, 10× faster than 13 sequential (repeated-input cost remains even if calls made concurrent).
- Failure-mode list: exact work in code; focused untrusted state (prompt injection moves answers); leave room for "none"; validate + version decision boundary (precision/recall/coverage/calibration/business cost, pin model version per threshold); engineer the service around it.
