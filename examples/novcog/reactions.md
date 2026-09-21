# Reactions: 1,500 Points, 17 Million Views, and Three Independent Tests

> Source: https://jev.novcog.us.com/reactions — fetched 2026-09-18 by Gizmo (H7).
> Novel Cognition (novcog) is NOT affiliated with TypeSafe AI, no early access. Snapshot below is the fetched page content, lightly reformatted. Numbers are quoted as published.

## The scale of it

- Founder's launch thread: 17.1M views, 51,400 likes, 2,576 replies by midday 16 Sep.
- HN thread: ~1,500 points, 426 comments, led front page for the day.
- HN title changed after the fact. WhitneyLand recorded the original: "The original title before it changed less than an hour ago was: 'Jev: New frontier model 40-400x cheaper and 20-200x faster'. I'm going to agree that was misleading."

## What the room objected to

Main criticism: the phrase "can't hallucinate".

- jacobgold: "Also 'can't hallucinate' seems wrong? Sure, it can't emit an invalid type, but it can still emit a completely wrong valid value."
- WhitneyLand: "Type safety is not factual correctness."
- StevenWaterman: "Yeah saying it can't hallucinate is crazy. It can still forward a billing query to the dev department incorrectly."
- Founder's counter: category argument — "I don't think it's fair to say a random forest 'hallucinates' in the way LLMs do", "Would you say a linear classifier hallucinates?" — and conceded: "because these models are probabilistic, it's also possible to be confidently wrong."
- Sharpest rebuttal (8note): "a linear classifier that classifies between red and yellow balls will hallucinate on blue."
- elil17 supplied the testable claim: "If the value is 0.9 for 1000 different answers, then approximately 900 of those answers should be correct." ← **This is exactly the Phase 3 calibration test.**

## The three independent tests

| Test | Setup | Jev result | Baseline |
|---|---|---|---|
| **Near Here** (Jon Reed, 16 Sep) | 50 real event-listing decisions | 96% accuracy (vs 84% and 86%), 0.59s avg, $0.043 per 1,000 decisions | Mistral Small 4 (2.90s, $0.370); Gemini 3.5 Flash-Lite (3.40s, $2.496) |
| **Good Start Labs** (Alex Duffy, 15 Sep) | 6,003 rubric checks | 91.5% agreement with Fable 5.1, ~0.5s per call, $160 per million graded answers | Fable 5.1 ~$33,000 per million; DeepSeek V4.1 Flash ~$260 |
| **Every** (Mike Taylor, 15 Sep) | writing + code-defect checks | 777 judgments in <0.7s total; 0.35s median per passage vs 8.83s; ~580× cheaper | Fable 5.1 — caught all 7 planted defects vs Jev's 6 |

Novcog's read: against real production alternatives, Near Here measured ~5× faster and 8.6× cheaper than Mistral Small 4 — NOT 193×/444×. Good Start Labs: 1.6× cost efficiency of DeepSeek V4.1 Flash. The enormous multiples hold only against expensive frontier reasoning models doing the same narrow job.

Caveats: Jon Reed calls his "a use-case study, not a general model ranking", "not an independently administered blind test" — and his accuracy result is the strongest pro-Jev finding anywhere. Mike Taylor: "want a more thorough accuracy check before putting it into production." Access gating: TypeSafe moved people up the waitlist during launch; Near Here appears to be the only test from someone who simply bought API access → tester sample selected by vendor.

## TypeSafe employee published the smallest number

zenlikethat (TypeSafe employee) released a DSPy fork routing decision steps through Jev: over 3 test cases, Jev-decorated path averaged 1.958s vs 2.329s plain DSPy (15.9% faster); modelled cost $0.000377 → $0.000263 per ticket (30.1% reduction). HN (snthpy): "I'm surprised the cost saving is so little though." README honest: 3 test cases, estimated price input, other model calls still dominate. Lesson: replacing one step with a 400× cheaper component does not make the workflow 400× cheaper.

## Who is impressed, and why

- lubujackson (HN): "After much fumbling around with prompts and evals, this is exactly how I am using LLMs in production... If this does at all what it claims, I think this is going to quickly become the new standard approach for agentic systems."
- janalsncm (HN): "Frontier LLMs are expensive jack of all trades. You can absolutely compare them to purpose-built tools on any domain they touch."
- Dan Shipper (Every, early access): "we almost never test new foundation models but we've been testing this for ~a week @every and it's pretty wild" — reporting 25× faster and 600× cheaper than a Fable-level judge.

## Reddit

Discussion almost entirely in r/singularity, r/accelerate, r/codex; as of 16 Sep no thread in r/LocalLLaMA, r/MachineLearning, r/programming, r/ExperiencedDevs. Top comment was a question: "if it doesn't output text, what can you do with it, exactly?" Verdicts: "this is not an LLM" (playpoxpax); "glorified encoder + classifier... useful for computer use, not so for coding or reasoning" (DivideHorror3217); "Output tokens are free because there are no output tokens" (Brainlag). Evidence complaints: launch shows hallucination rates but "0 intelligence benchmarks" (Charuru); eval chart "one of the most impressively vague i've seen yet" (Dangerous-Sport-2347). Biggest thread (478 pts, 115 comments) removed by moderators, no reason given. **Nobody on Reddit raised the reference-label question: eval accuracy numbers come from averaging GPT-6 Astra + Fable 5.1 — LLM labels, not ground truth. That detail is on the evals page.**

## Verdict most people landed on

flyinglizard (HN): "the claim of not hallucinating is made technically true... The latency and cost - yes, those are super interesting."

Novcog closing: product is real; speed and price large enough to change classification-shaped workloads; framing loose enough that the company's own title had to be rewritten. "The numbers that support it are on the evals, and what remains unproven is on sources & method."
