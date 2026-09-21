# novcog Prior Art + The Guarantee — H7 snapshots

> Sources: https://jev.novcog.us.com/prior-art + /the-guarantee — fetched 2026-09-18.

## Prior Art: "Basically a Zero-Shot Classifier" — and the Founder Agreed

- petesergeant (HN, 10 min into thread): "basically a zero-shot classifier that can accept raw text... classify as accurately (they claim) as a frontier-level LLM." CEO CompleteSkeptic: **"exactly right!"**
- Community prior-art map: encoder classifiers BERT/DeBERTa-family ("no hallucinations and faster inference for free"); GLiNER2/GLiClass ("encoder-based, multiple tasks in single forward pass, deterministic"); constrained/grammar decoding ("read the logits for confidence"); text diffusion models (parallel generation signature); conformal prediction ("productionized conformal prediction" — Wazzymandias); DSPy typed signatures.
- Same-night repro: Harsha Gundala, Qwen-2.5-1B-RLCD on HF — "They were building in stealth for 2 years, I was building in stealth for 2 hours". Claim: "every LLM has the ability to efficiently batch inference every key of a JSON at the same time and generate probabilities from a set of possible categories. No new training required." Untested by anyone publicly. **(This is the upstream of our reference/harshatheg-rlcd code.)**
- CEO's real counter-argument (strongest technical point, from HN): "constrained decoding (OpenAI-style structured outputs) make models dumber... simply masking logits is insufficient because if ever a model was assigning probability to an invalid token, the model is by definition confused. you'd be better off erroring IMO." → Whether Jev's training removes that failure mode "is exactly what a calibration curve would show, and none is published."
- Unanswered questions (the two that would settle novelty): tidewave — "What specifically changes in the training objective with RLCD? Are its benefits isolated from Jev's new architecture/parallelism?" Mentlo — "Is anything published on how it maintains calibration? Or... 'as calibrated as frontier LLM models, just cheaper' — which is a different claim; as LLMs aren't particularly well calibrated." CEO on architecture: "close to the chest for now, but we have talked about writing a paper."
- Fair statement of the unmatched combination: runtime-defined schemas + trained calibration (not raw logits) + frontier-adjacent agreement at ~1/1000th price. padolsey: "I did similar things for zero-shot criterion-based classification using a 4B Qwen model but could not reach the level of intelligence they've got here. Tho speed/cheapness was similar."

## The Guarantee: "Can't Hallucinate" Covers Shape, Not Truth

- TypeSafe: "never makes type errors", "mathematically impossible", "easy to falsify with a single counter-example" — strong, narrow, honest.
- Chart nuance note (their own): **"Our number is not empirical. Schema matching is guaranteed, thus we can confidently add 0% into the plots."** LLM bars: "from OpenRouter i.e., there almost certainly is bias here." → chart places measured quantities next to an asserted one; homepage "Zero Hallucinations" headline doesn't carry the caveat.
- Calibration is "the claim that would matter": TypeSafe argues an uncalibrated model cannot automate a task even at 95% accuracy "if it doesn't say when it's in the 5%" — correct, and it IS the product. **No reliability diagram, no ECE, no Brier, no paper; waitlisted access. "The company that says confidence is the point has not yet published the plot that would show it."**
- Production advice: keep an oracle where you have one (sample decisions vs held ground truth, keep measuring post-launch); treat confidence as routing input, not proof; watch the schema itself (a guarantee over an incomplete option set is worth nothing).
