# The Evals: Measured Against Two Models, Not Against Truth

> Source: https://jev.novcog.us.com/the-evals — fetched 2026-09-18 (H7 snapshot).
> Key takeaway for jev-rlcd Phase 3: TypeSafe's published benchmark measures AGREEMENT WITH TWO FRONTIER LLMS, not correctness vs human labels. No calibration methodology, no reliability curve, no human-labeled eval exists publicly.

## What the benchmark does

- TypeSafe's own eval: evals.typesafe.ai. Their words: "we assume there is a correct compute graph (a 'workflow' represented in code) and use the predictions of the largest, smartest, and most expensive external models as reference probabilities."
- Design virtue they claim: harness fixed for all models — "we don't ... allow the harness and model to change (potentially allowing for overfitting via harness engineering)".

## What the reference answer actually is

| Element | TypeSafe's own description |
|---|---|
| Reference answer | "the average of GPT-6 Astra and Fable 5.1" |
| Acknowledged bias | "biases answers towards OpenAI and Anthropic's models. We likely underestimate the relative performance of our model and DeepSeek's models." |
| Who built the workflows | "made by individuals on our model capabilities team, so some bias could exist" |
| How the LLMs are called | Through TypeSafe's own System One adapter, which "constrains LLMs to output structured decisions compatible with our API" |
| Headline figures | "This is where the claims of 193.6x faster, 444.6x cheaper on our home page comes from" |

Consequence (novcog): a score on this benchmark = agreement with two frontier LLMs, not correctness. Perfect tracking scores perfectly even where they are jointly wrong; being right where they are wrong gets marked down. Defensible for "as good as the expensive model, 2 OOM cheaper" business framing — not what "off the charts" implies.

## The arithmetic on the home page

- Home page: 0.114s vs 8.566s latency pair = 75.1×; headline says 193.6× — both on one screen, different measurements, unmarked.
- Blog body: "40x-200x faster ... for System One shaped queries". Launch thread: "20-200x faster ... 40-400x cheaper". Title changed on HN from "40-400x cheaper and 20-200x faster".
- TypeSafe caveat: expects gains "on the higher end of real world gains."

## Demos with footnotes

- **Doom**: ~10 queries/s bot, ~$7/hour. TypeSafe's own notes: runs "on structured state as a data structure with text, not on images"; "a non-AI doom bot could play better".
- **Wikiracing**: choosing among thousands of links per step. Note: comparison "against the non-reasoning modes of the models (except Astra which was set to the lowest reasoning setting)" — "The LLMs look much worse at this task than with reasoning enabled."

## FAQ facts (novcog)

- Is a high score correctness? No.
- Independent benchmark of Jev? "None had been published as of 16 September 2026." Access is waitlisted — main obstacle to outside measurement. (Near Here = only tester who just bought API access.)
