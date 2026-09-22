# qwen-jev-like

A for-fun project. I wanted to know what happens if you take a Jev-shaped contract (`state` in, typed answers out, no
text generation) and run it against open models on one consumer GPU. This repo is the engine, the pipeline, and the
numbers that came out of a few weeks of poking at it.

Most of the code and measurements were produced by AI coding agents; I set the scope and decided what got published.
Everything here can be checked without taking my word for it.

This is not Jev. It is an attempt to see how close open models get to the same contract on one GPU. Jev appears as
one column in the tables, measured through OpenRouter on the dates noted, the same way any other model does.

## Where the idea came from

Jev (TypeSafe) is closed. [harshatheg](https://github.com/harshatheg) reproduced the idea on a 1B Qwen the same
night: score the options from the logits, never decode. This started as a port of that; the tree became a trie.
Calibration protocol from [jevlike](https://github.com/vinnylarouge/jevlike). Full list under References.

## What it does

You hand the engine a `state` (a piece of text) and a list of typed questions:

```
noul    yes / no
choice  pick one option from a list
score   a level on a rubric
```

It returns one answer per question with a probability on every option. No text is generated.

## How a chat model becomes Jev-like

Nothing in the weights changes. The engine changes how the model is asked and how the answer is read.

1. One prompt: catalogue of questions and allowed values first, state last, "answer as JSON".
2. Chat template with thinking off, then `{` appended, so the next token is the first answer key.
3. No generation. `max_tokens=1`, temperature 0; read the log-probability of each allowed value at its position.
   Multi-token values follow their token path; shared prefixes share the work (a trie).
4. Every allowed value must appear verbatim in the catalogue. The model echoes what it was shown.
5. Renormalise over the allowed values. `in_set_mass` says how much probability landed on them at all; under 0.5
   means the prompt is broken, not that the question is hard.
6. One forward pass per state carries all its questions. The catalogue's KV state is reused across calls.

Probabilities are T=1 unless a table says otherwise; temperature is fitted offline and reported as a column.

```
[catalogue][pad][state][pad][requests]  ->  one forward pass  ->  log-prob per allowed value  ->  typed answers
```

## Models

| model, as named in every table | source | quant | file size | backend |
|---|---|---|---|---|
| Ternary-Bonsai-2-27B PQ2_0 | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), Qwen3.8-27B base | ternary, ~1.7 bpw | 6.7 GB | llama.cpp fork |
| Qwen3.8-27B UD-Q2_K_XL | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q2_K_XL, ~2 bpw | 9.2 GB | llama.cpp fork |
| Qwen3.5-9B-AWQ | [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) | AWQ 4-bit | 12 GB | vLLM 0.29 (optional) |
| Qwen3.5-4B-AWQ | [cyankiwi/Qwen3.5-4B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) | AWQ 4-bit | 3.8 GB | vLLM 0.29 (optional) |
| Jev 1.13 | `typesafe/jev-1.13-20260917` via OpenRouter | closed | | remote, dates in each table |

Each file was checked against the sha256 the Hub publishes; `docs/MODELS.md` says which ones matched and which were
not checked.

## What I tried

A fair spread of tasks: intent classification (Banking77), emotion labels, CVE attribute classification, CVSS v3.1
metric prediction, bug-report classification, the public `typesafe-ai-benchmark`, the four Cloudflare example cases,
and two small interactive scenarios I wrote myself. The security-flavoured sets are there because I find that area
interesting; they are used as classification tasks with human labels, nothing more is claimed.

## The short version

Everything below ran on one desktop:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, driver 610.88 |
| CPU | AMD Ryzen 7 5700X, 8 cores / 16 threads |
| RAM | 32 GB, of which WSL2 is given 25 GB |
| OS | Windows host, Ubuntu 22.04 under WSL2 (kernel 5.15) |
| Stack | Python 3.12, vLLM 0.29, torch 2.13 + CUDA 13.0; llama.cpp fork built against CUDA 13.4 |

| | Ternary-Bonsai-2-27B PQ2_0 | Qwen3.8-27B UD-Q2_K_XL |
|---|---|---|
| file size | 6.7 GB | 9.2 GB |
| repeat call, same catalogue, new state | ~290 ms median | ~560 ms median |
| deterministic across two processes | yes, byte-identical | yes, byte-identical |

On Ternary-Bonsai-2-27B with `n_ubatch=4096` and 12 sequences the process holds about 15.7 GB of VRAM in total, loads in 11 s, and
the first call on a new catalogue takes 1.3 s. The Qwen3.8-27B UD-Q2_K_XL column was measured at a smaller batch shape, so its resident
VRAM and cold-call numbers are not comparable and live in `docs/RESULTS.md` with their configuration.

Both run alongside a desktop session.

Same model, two ways of asking. Qwen3.5-9B-AWQ on vLLM, warm, same full prompt for both sides:

| preset | fields | typed, warm | vs generating the JSON as text | schema-valid |
|---|---|---|---|---|
| fintech_fraud | 4 | 419 ms | 9.7x faster | 24/24 vs 0/24 |
| code_security | 4 | 449 ms | 9.3x | |
| support_triage | 28 | 446 ms | 9.1x | |
| tariff_255 | 1 field, 255 options | 543 ms | 1.7x | |

The typed path costs about the same whether a state has 4 questions or 28, because every option is scored in one
forward pass. The text path grows with the length of the answer it has to write. The one place the gap closes is a
single question with 255 options, where the text path only has to write one short label. The text baseline saw the
same full prompt and took about 4 s on Qwen3.5-9B-AWQ; its output failed schema validation on every one of 24 attempts.
The typed path cannot produce an invalid answer.

What worked:

- On Banking77 (77 intents, human labels) Ternary-Bonsai-2-27B lands at 0.760, Qwen3.5-9B-AWQ at 0.731, and Jev 1.13 at 0.760 to 0.803
  depending on who measured it. Same band.
- On the public `typesafe-ai-benchmark` the local models agree with Jev's direction on 92 to 93 % of items.
- On a 6-way emotion set Ternary-Bonsai-2-27B reaches 0.60 (n=2000). The only published Jev number on the same set is 0.48, from a
  100-item sample; the labels are noisy, so both numbers sit under a low ceiling.
- On bug-report classification (12 weakness families) all three columns sit at 0.64 to 0.67 against a 0.15 majority
  class, and all three know when they are unsure: dropping the 20 % least confident answers adds 6 to 10 points.
- On the two scenario benches, Ternary-Bonsai-2-27B and Qwen3.8-27B UD-Q2_K_XL are the first models to pass all three pre-registered
  criteria; Qwen3.5-9B-AWQ and Qwen3.5-4B-AWQ did not.

What did not:

- Neither quantisation wins. One measurement says the ternary QAT model is better, one says the 2-bit post-training
  file is, and every other confidence interval contains zero. Any single one of them, run on its own, would have told
  a different story.
- On CVSS metric prediction the column closest to Jev was also the column furthest from NIST. Agreeing with Jev is
  not the same as being right.
- How you phrase the question moves results more than which model you use. On `typesafe-ai-benchmark` the scoring
  task went from 70 to 94 by changing wording and how the state was packed, with the model unchanged.
- The `confidence` a model reports is a property of its output distribution, not the probability it is right. All
  three columns are worst in the 0.6 to 0.8 confidence bin, where they claim 0.70 and are right 0.35 to 0.44 of the time.

## Quickstart

Three steps, no GPU needed for the first one.

### 1. Check the engine on CPU

```sh
git clone <this repo> qwen-jev-like && cd qwen-jev-like
uv venv --python 3.12 && uv pip install -r requirements.txt   # Python >= 3.10
python -m tests.test_cpu
```

This runs the contract, routing, and window-encoding tests against a deterministic fake model. It needs no weights
and no GPU; the tokenizer is the only download (~22 MB of vocabulary). It ends with a line like

```
ran 64 checks: 64 ok, 0 failed, 1 skipped
ALL PASS
```

The skip is a check whose dataset this repo does not ship; it prints the reason and the builder that recreates it.
Skips are counted separately and never folded into the pass count. vLLM is not needed for any of this.

### 2. Get a model

Pick one. Both fit on a 24 GB card next to a desktop session.

| model | file | size | backend |
|---|---|---|---|
| Ternary-Bonsai-2-27B PQ2_0 (ternary QAT) | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 6.7 GB | llama.cpp fork (bundled, `backends/llamacpp/`) |
| Qwen3.8-27B UD-Q2_K_XL (Unsloth) | `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.2 GB | same |
| Qwen3.5-9B-AWQ | QuantTrio HF checkpoint | 12 GB | vLLM 0.29 |

`docs/MODELS.md` has the download source and sha256 for each. Point the engine at the GGUF with
`JEV_BONSAI_GGUF=/path/to/file.gguf` and set `JEV_BACKEND=bonsai`. Building the fork worker takes one `make` in
`backends/llamacpp/`; the README there covers the CUDA toolkit trap on WSL2.

### 3. Ask a question

```python
from transformers import AutoTokenizer
from core.jev_engine import JevEngine
from core import bonsai_llm

llm = bonsai_llm.from_env()                       # reads JEV_BONSAI_GGUF
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")   # tokenizer only, shared by the Qwen family
eng = JevEngine(llm, tok)

answer = eng.run(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                      "criteria": {"true": "Explicitly time-sensitive",
                                   "false": "No urgency expressed"}},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"payments": "Payouts, transfers, settlement",
                              "account": "Login, profile, verification",
                              "fraud": "Suspicious or unauthorised activity",
                              "general": "Anything else"}},
        "severity": {"type": "score", "instructions": "How severe is the issue?",
                     "criteria": ["low", "medium", "high", "critical"]},
    },
)
print(answer)
```

What comes back, with the shapes the three question types actually return:

```python
{"model": "...",
 "answers": {
   "is_urgent": {"type": "noul",   "noul": 0.574},
   "team":      {"type": "choice", "choice": "payments", "confidence": 0.371,
                 "probabilities": {"payments": 0.371, "account": 0.275, ...}},
   "severity":  {"type": "score",  "score": 1.13, "confidence": 0.282,
                 "legend": {"0": "low", "1": "medium", "2": "high", "3": "critical"},
                 "probabilities": {"0": 0.371, "1": 0.275, ...}}},
 "usage": {"input_tokens": 1056, "output_tokens": 3},
 "_engine": {"elapsed_ms": ..., "per_question": {...}}}
```

A `noul` answer is a single probability of true; there is no separate label to pick. `choice` and `score` carry a
probability for every option; `score` also returns the expected level and the legend it was computed over, so a
`score` of 1.13 means "just past medium, leaning high" rather than naming a class.

The diagnostics live under `_engine.per_question`, not next to the answers: `in_set_mass` is how much of the model's
probability landed on the options you gave, and `low_evidence` is set when that drops under 0.5, which in practice
means the prompt or the formatting is broken, not that the question was hard.

The first call on a new set of questions costs about 1.3 s while the catalogue is encoded. Every later call with the
same questions and a new state reuses that work and comes back in roughly 300 ms on Ternary-Bonsai-2-27B.

### Running the benches

```sh
./gpu.sh sanity  python run_sanity.py                       # 4 Cloudflare cases, direction vs Jev
./gpu.sh bench   python run_bench.py --preset fintech_fraud  # speed, needs the fetched presets below
./gpu.sh collect python run_collect.py --datasets banking77 --batch 8
python run_fit.py runs/calib_<label>/                        # temperature fit + ECE, offline

python data/build/fetch_bench_presets.py   # four upstream presets the speed bench reads; not redistributed here
```

`gpu.sh` is a lock so two processes never load a model on the same card at once; it also writes the log the results
files cite. Datasets are rebuilt with the scripts in `data/build/`; each prints the sha256 of what it produced so you
can check it against `docs/RESULTS.md`.

## The long version

### Method

Protocol was written before any number was taken: FIT 60 / SELECT 20 / TEST 20 splits, temperature scaling only
(no fine-tuning, no LLM-generated labels), pass and fail thresholds fixed in advance. Every table states the backend,
`PAD`, context, `n_seq_max`, batch size, and T that produced it, because changing any of these changes the logits.
Accuracy is always printed next to the majority-class rate. Score accuracy comes in two flavours, within one level
and exact, and every table names which. Anything under n=50 is called a signal, never a result. Details in
`docs/METHODS.md`.

### Results

Full tables with n, bootstrap CI95, configuration rows, and the raw run filenames are in `docs/RESULTS.md`, which
prints the corrected value wherever a number was revised.

Sections there:

1. Speed: cold and warm per preset, Qwen3.5-9B-AWQ on vLLM, against an autoregressive baseline that saw the same prompt.
2. Accuracy per dataset: all five models above, with majority-class rate.
3. Calibration: ECE, Brier, selective accuracy, reliability diagrams, T per question type.
4. QAT vs PTQ: per-set differences with CIs.
5. Direction agreement with Jev on `typesafe-ai-benchmark` and the Cloudflare cases.
6. Scenario benches: criteria and per-model verdicts (the scenarios themselves are not published), plus a
   5-step browser task from an external MIT fixture, where a sequence has to compose rather than one answer
   be right.
7. Determinism: byte-identity across processes on the llama.cpp fork, and the batch-shape effect on vLLM.
8. Known limits: what is not shown, not calibrated, or waived.

### Engine

`core/jev_engine.py`. Eight invariants, each with an assert or a test: scored tokens appear verbatim in the
catalogue; prompt order is fixed so the catalogue block hashes identically across states; one `generate()` per state;
options are scored through a shared trie so the state is encoded once; window encoding never re-encodes the full
prompt per option. `docs/METHODS.md` walks through them and the bug each one was added after.

### Model files

Not included. `docs/MODELS.md` lists the exact checkpoints, their digests, and how far each file was actually
checked, which is not the same for all of them. The `n_seq_max` and `ctx` a bench ran at belong to the bench, so
they are in `docs/RESULTS.md` next to the numbers they produced.

## Data

No dataset rows are in this repository at all: the count of `.jsonl` files under `data/` is zero, and that is
deliberate rather than an oversight about which sets were redistributable. Every set ships as a builder script plus
the sha256 of the file it produces, so anyone can rebuild the exact rows and check they got the same ones. Some of
these licences permit measurement but not redistribution (the emotion set, the CVSS set at CC-BY-NC-SA), one corpus
belongs to the people who wrote it (the bug reports), and that corpus contains live credentials in its text.
`data/calib/*/README.md` gives each set's source, licence, label provenance, rebuild command, digest, and the caveat
that changes how its numbers should be read. `data/README.md` is the index.

## References

Models: Ternary-Bonsai-2-27B (prism-ml), Qwen3.8-27B UD-Q2_K_XL (Unsloth), Qwen3.5-9B-AWQ (QuantTrio) and
Qwen3.5-4B-AWQ (cyankiwi), both from Alibaba's Qwen3.5; Jev 1.13 (TypeSafe, via OpenRouter).

Data: Banking77 (PolyAI, CC-BY-4.0), Emotion (dair-ai), CTI-Bench (Alam et al., CC-BY-NC-SA), NVD/CVE, HackerOne
disclosed reports, `typesafe-ai-benchmark`, Cloudflare Jev examples.

Prior work this leans on: harshatheg's Qwen-2.5-1B-RLCD (the presets and the first port), vinnylarouge/jevlike (the
ECE protocol), AbdelStark/jev-benchmarks (the pre-registered Emotion number), nibzard/decision-model-benchmark (the
"builder, not text" posture for non-commercial data), Aitejiu/jev-harness-lab and onlyoneaman/jev-eval (independent
Jev numbers on Banking77), KaLM-Embedding/KaLM-Jev (a reranker take on the same contract), and Bernoulli's write-up
of Jev's confidence formula.

MIT. See `CITATION.cff`.
