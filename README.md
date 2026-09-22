# qwen-jev-like

A for-fun project: take a Jev-shaped contract (`state` in, typed answers out, no text generation) and run it
against open models on one consumer GPU.

- **Not Jev.** Jev 1.13 is one column in the tables, measured through OpenRouter like any other model.
- **Built by agents.** Most code and measurements came from AI coding agents; I set scope and published.
- **Checkable.** Every number states its configuration and the run it came from.

## Where the idea came from

Jev (TypeSafe) is closed. [harshatheg](https://github.com/harshatheg) reproduced the idea on a 1B Qwen the same
night: score the options from the logits, never decode.

This is a port of that with the tree replaced by a trie, plus the calibration protocol from
[jevlike](https://github.com/vinnylarouge/jevlike).

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

| task | source |
|---|---|
| intent classification | Banking77, 77 intents |
| emotion labels | 6-way, n=2000 |
| CVE attribute classification | NVD 2023 |
| CVSS v3.1 metric prediction | CTI-Bench, 300 CVEs |
| bug-report classification | 12 weakness families, 300 reports |
| typed-decision replay | `typesafe-ai-benchmark`, 1257 questions |
| direction check | 4 Cloudflare example cases |
| interactive scenarios | 3 benches, two of them mine |

The security sets are there because I find that area interesting. They are classification tasks with human labels.

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
| weights on GPU | 6540 MiB | 8631 MiB |
| KV cache | 884 MiB | 884 MiB |
| recurrent state | 1945 MiB | 1945 MiB |
| compute buffer | 1054 MiB | 1014 MiB |
| **GPU buffers, total** | **10.2 GB** | **12.2 GB** |
| repeat call, same catalogue, new state | 88 ms median | 102 ms median |
| 2000-row collect, wall clock | 187 s | 218 s |
| byte-identical across two processes | yes | yes |

Fork backend, `PAD=0`, `ctx/seq=2048`, `n_seq_max=13`, 4 catalogue slots, `n_ubatch=1024`, T=1, one state per call.

`nvidia-smi` shows more than the total above, because that total is only what the allocator reserves: the CUDA
context and the desktop session are on the same card. The KV cache also scales with context, reaching 1768 MiB at
`ctx/seq=4096`, which is the shape the scenario benches use.

Same model, two ways of asking. Qwen3.5-9B-AWQ on vLLM, warm, same full prompt for both sides:

| preset | fields | typed, warm | vs generating the JSON as text | schema-valid |
|---|---|---|---|---|
| fintech_fraud | 4 | 419 ms | 9.7x faster | 24/24 vs 0/24 |
| code_security | 4 | 449 ms | 9.3x | |
| support_triage | 28 | 446 ms | 9.1x | |
| tariff_255 | 1 field, 255 options | 543 ms | 1.7x | |

Cost is flat in the number of questions because all options score in one pass. The text baseline saw the same
prompt, took about 4 s, and failed schema validation 24/24.

What worked:

- **0.760 / 0.731 / 0.760-0.803** on Banking77: ternary, Qwen3.5-9B-AWQ, Jev. Same band.
- **92-93 %** direction agreement with Jev on `typesafe-ai-benchmark`.
- **0.60** on 6-way emotion, n=2000. The one published Jev number there is 0.48 at n=100, noisy labels.
- **0.64-0.67** on bug-report classification against a 0.15 majority, all three columns.
- **+6 to +10 points** from dropping the least-confident 20 %, in every column.
- **3/3** pre-registered criteria on the scenario benches for both 27B files; the AWQ columns did not.

What did not:

- **1 for, 1 against, rest contain zero**: no quantisation wins across the comparisons.
- **Closest to Jev was furthest from NIST** on CVSS. Agreeing with Jev is not being right.
- **70 to 94** on one task from wording and state packing alone, model unchanged.
- **0.70 claimed, 0.35-0.44 actual** in the 0.6-0.8 confidence bin, all three columns.
- **0/3** for Qwen3.8-27B on a 5-step browser task that Jev completes 3/3.

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

The skip is a check whose dataset this repo does not ship, counted separately and never folded into the pass count.
vLLM is not needed here.

### 2. Get a model

Pick one. Both fit on a 24 GB card next to a desktop session.

| model | file | size | backend |
|---|---|---|---|
| Ternary-Bonsai-2-27B PQ2_0 (ternary QAT) | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 6.7 GB | llama.cpp fork (bundled, `backends/llamacpp/`) |
| Qwen3.8-27B UD-Q2_K_XL (Unsloth) | `Qwen3.8-27B-UD-Q2_K_XL.gguf` | 9.2 GB | same |
| Qwen3.5-9B-AWQ | QuantTrio HF checkpoint | 12 GB | vLLM 0.29 |

Download source and sha256 per file: [docs/MODELS.md](docs/MODELS.md).

- Point the engine at a GGUF: `JEV_BONSAI_GGUF=/path/to/file.gguf`, `JEV_BACKEND=bonsai`.
- Build the fork worker: one `make` in `backends/llamacpp/`, whose README covers the CUDA trap on WSL2.

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

- `noul`: one probability of true, no label to pick.
- `choice` and `score`: a probability for every option.
- `score` also returns the expected level and its legend, so 1.13 is a position, not a class id.
- Diagnostics sit under `_engine.per_question`: `in_set_mass`, and `low_evidence` when it drops under 0.5.

The first call on a question set pays to encode its catalogue; later calls reusing it come back in about
100 ms on either 27B file. Measured cost depends on how many options the catalogue holds.

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

Written before any number was taken:

- Splits FIT 60 / SELECT 20 / TEST 20, grouped so one state cannot straddle two splits.
- Temperature scaling only. No fine-tuning, no model-generated labels.
- Pass and fail thresholds fixed in advance.
- Every table states backend, `PAD`, context, `n_seq_max`, batch size and T.
- Accuracy always printed next to the majority-class rate.
- Score accuracy has two measures, within one level and exact; each table names its own.
- Under n=50 is a signal, never a result.

Full protocol in [docs/METHODS.md](docs/METHODS.md).

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

`core/jev_engine.py`, eight invariants, each with an assert or a test:

- Scored tokens appear verbatim in the catalogue.
- Prompt order is fixed, so the catalogue hashes identically across states.
- One `generate()` per state.
- Options score through a shared trie, so the state is encoded once.
- Window encoding never re-encodes the full prompt per option.

All eight, and the bug behind each, in [docs/METHODS.md](docs/METHODS.md).

### Model files

Not included. [docs/MODELS.md](docs/MODELS.md) has the checkpoints, digests, and how far each file was checked,
which differs per file.

## Data

No dataset rows are here: zero `.jsonl` files under `data/`. Every set ships as a builder plus the sha256 of what it
produces.

| set | licence | why it is not shipped |
|---|---|---|
| Banking77, CLINC150, BoolQ, SciTail, PubMedQA | CC-BY / Apache-2.0 / MIT | rebuilt for consistency, not licence |
| emotion | card says `other`, research use | no redistribution |
| CVSS metrics (CTI-Bench) | CC-BY-NC-SA-4.0 | non-commercial |
| bug reports | belong to their finders | that, and live credentials in the text |
| PaySim transactions | CC-BY-SA-4.0 | needs a Kaggle token |

Per-set source, label provenance, rebuild command, digest and caveat: `data/calib/*/README.md`, indexed by
[data/README.md](data/README.md).

## References

**Models**

- [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), the ternary QAT file.
- [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), the 2-bit post-training comparison.
- [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ), the vLLM column and speed baseline.
- [cyankiwi/Qwen3.5-4B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit), the small-model check.
- Jev 1.13 (TypeSafe), reached through [OpenRouter](https://openrouter.ai), the hosted reference.

**Data**

- [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), intents with human labels.
- [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), the largest single set here.
- [AI4Sec/cti-bench](https://huggingface.co/datasets/AI4Sec/cti-bench), CVSS metric prediction.
- [NVD](https://nvd.nist.gov), the CVE feed behind the attribute and CVSS labels.
- [HackerOne Hacktivity](https://hackerone.com/hacktivity), disclosed reports for the bug-report set.
- [iammrduncan/typesafe-ai-benchmark](https://github.com/iammrduncan/typesafe-ai-benchmark), the replay harness.
- Cloudflare's published Jev request and response examples, the direction oracle.

**Prior work**

- [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), the presets and the first port.
- [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike), the ECE protocol.
- [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks), the pre-registered emotion number.
- [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark), builder-not-text for NC data.
- [Aitejiu/jev-harness-lab](https://github.com/Aitejiu/jev-harness-lab) and
  [onlyoneaman/jev-eval](https://github.com/onlyoneaman/jev-eval), independent Jev numbers on Banking77.
- [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast), the browser fixture in section 6b.
- KaLM-Embedding/KaLM-Jev, a reranker take on the same contract.
- [Bernoulli](https://bernoulli.app)'s write-up of Jev's confidence formula.

MIT. See `CITATION.cff`.
