# Models

Five checkpoints appear in the results. Four are downloadable and are named by
their exact repository; one is not, and says so rather than pointing at
something that resembles it.

| short name | checkpoint | quantisation | on disk | backend |
|---|---|---|---|---|
| **Bonsai** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` — local file, not a published artifact | ternary, ~1.7 bits/weight | 6.71 GB | llama.cpp fork |
| **PTQ** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) → `Qwen3.8-27B-UD-Q2_K_XL.gguf` | Q2_K_XL, ~2 bits/weight | 9.15 GB | llama.cpp fork |
| **9B** | [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) | AWQ 4-bit | 12 GB | vLLM 0.29 |
| **4B** | [cyankiwi/Qwen3.5-4B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) | AWQ 4-bit | 3.8 GB | vLLM 0.29 |
| **Ornith** | [cyankiwi/Ornith-1.5-9B-AWQ-INT4](https://huggingface.co/cyankiwi/Ornith-1.5-9B-AWQ-INT4) | INT4, compressed-tensors W4 group-32 asymmetric | 8.5 GB | vLLM 0.29 |

## Digests, and how far each was actually checked

```
3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1  Ternary-Bonsai-2-27B-PQ2_0.gguf
fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0  Qwen3.8-27B-UD-Q2_K_XL.gguf
996e60e31539f0d7108e0d1dd3b82ce476ccefa9a0746f5fcb29707cfc14c4dd  Qwen3.5-9B-AWQ/model-00005-of-00005.safetensors
```

- **PTQ** was checked against the upstream repository: the local digest equals
  the LFS SHA-256 the Hub records for that file, so the file measured here is
  the file that repository serves.
- **9B** was fetched through a ModelScope mirror (`tclf90/Qwen3.5-9B-AWQ` — the
  checkout's own `.msc` metadata records it), so the Hub link above needed
  checking rather than assuming. Shard 5 of 5 hashes identically to the LFS
  digest `QuantTrio/Qwen3.5-9B-AWQ` publishes, which is why that link is here.
  The other four shards were not checked.
- **Bonsai** has no upstream to check against. Its GGUF header carries no
  repository, only `general.name = "Hf"` and `general.basename = "folded"`: it
  is a local conversion, and the digest above identifies the exact file the
  numbers came from. The published ternary pack in that family is
  [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit),
  in MLX format rather than GGUF; **this file is not that download**, and no
  claim is made here about how one relates to the other.
- **4B** and **Ornith** were not hashed. Their repositories are named above and
  were confirmed to exist; that is all.

Two consequences worth stating plainly. The Bonsai column **cannot be
reproduced from a download** — everything else here can, and that is a real
asymmetry between the two fork-backend columns. And a digest that has not been
computed is not evidence of anything, which is why the last line above says
which files were skipped instead of leaving the section looking complete.

## Licences

| model | licence |
|---|---|
| Qwen3.5-9B / 4B and their quantisations | Apache-2.0 |
| Qwen3.8-27B (Unsloth GGUF) | Apache-2.0 |
| Ornith-1.5-9B | MIT ([licence](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE)) |
| Ternary Bonsai 2 27B | upstream pack's terms; the local file is not redistributed |

## Pointing the engine at one

```sh
JEV_BACKEND=bonsai JEV_BONSAI_GGUF=/path/to/model.gguf   # llama.cpp fork, the default path
JEV_MODEL=models/Qwen3.5-9B-AWQ                          # vLLM, needs requirements-vllm.txt
```

The tokenizer is a separate, much smaller thing: the CPU tests fetch vocabulary
only (~22 MB) from `Qwen/Qwen3.5-9B` and never need weights. Ornith's
`tokenizer.json` is byte-identical to Qwen3.5-9B's, which is why the same
window-encoding assertions hold for both.
