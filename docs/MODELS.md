# Models

Five checkpoints appear in the results. Four are downloadable and are named by
their exact repository; one is not, and says so rather than pointing at
something that resembles it.

| model | source | quantisation | on disk (GiB, binary) | backend |
|---|---|---|---|---|
| **Ternary-Bonsai-2-27B PQ2_0** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) → `Ternary-Bonsai-2-27B-PQ2_0.gguf` | ternary, ~1.7 bits/weight | 6.71 GiB | llama.cpp fork |
| **Qwen3.8-27B UD-Q2_K_XL** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Q2_K_XL, ~2 bits/weight | 9.15 GiB | llama.cpp fork |
| **Qwen3.5-9B-AWQ** | [QuantTrio/Qwen3.5-9B-AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) | AWQ 4-bit | 12 GiB | vLLM 0.29 |
| **Qwen3.5-4B-AWQ** | [cyankiwi/Qwen3.5-4B-AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) | AWQ 4-bit | 3.8 GiB | vLLM 0.29 |
| **Ornith-1.5-9B-AWQ-INT4** | [cyankiwi/Ornith-1.5-9B-AWQ-INT4](https://huggingface.co/cyankiwi/Ornith-1.5-9B-AWQ-INT4) | INT4, compressed-tensors W4 group-32 asymmetric | 8.5 GiB | vLLM 0.29 |

## Digests, and how far each was actually checked

```
3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1  Ternary-Bonsai-2-27B-PQ2_0.gguf
fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0  Qwen3.8-27B-UD-Q2_K_XL.gguf
996e60e31539f0d7108e0d1dd3b82ce476ccefa9a0746f5fcb29707cfc14c4dd  Qwen3.5-9B-AWQ/model-00005-of-00005.safetensors
```

- **Qwen3.8-27B UD-Q2_K_XL** was checked against the upstream repository: the local digest equals
  the LFS SHA-256 the Hub records for that file, so the file measured here is
  the file that repository serves.
- **Qwen3.5-9B-AWQ** was fetched through a ModelScope mirror (`tclf90/Qwen3.5-9B-AWQ`, the
  checkout's own `.msc` metadata records it), so the Hub link above needed
  checking rather than assuming. All five shards hash identically to the LFS
  digests `QuantTrio/Qwen3.5-9B-AWQ` publishes, which is why that link is here.
  Four of them were checked later than shard five, on 2026-09-22.
- **Ternary-Bonsai-2-27B PQ2_0** was checked the same way, against
  `prism-ml/Ternary-Bonsai-2-27B-gguf`: 7,206,168,928 bytes and the digest
  above, equal to what the Hub records for that file. Note the **2** in the
  name. `prism-ml/Ternary-Bonsai-27B-gguf` is a different model on a different
  base and also publishes a `PQ2_0.gguf`, and comparing against that one makes
  this byte-identical file look 41 MB wrong.
- **Qwen3.5-4B-AWQ** and **Ornith-1.5-9B-AWQ-INT4** were hashed on 2026-09-22, later than the rest. All four
  weight files across the two repositories match `cyankiwi/Qwen3.5-4B-AWQ-4bit` and
  `cyankiwi/Ornith-1.5-9B-AWQ-INT4` by size and by LFS sha256.

Every column here now reproduces from a download of the exact file that was
measured. That was not true when this section was first written, and the two
unchecked entries were named at the time rather than left out: a digest that has
not been computed is not evidence of anything, and a section that lists only its
successes reads as though it covered everything.

## Licences

| model | licence |
|---|---|
| Qwen3.5-9B / 4B and their quantisations | Apache-2.0 |
| Qwen3.8-27B (Unsloth GGUF) | Apache-2.0 |
| Ornith-1.5-9B | MIT ([licence](https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE)) |
| Ternary-Bonsai-2-27B | Apache-2.0 |

## Pointing the engine at one

```sh
JEV_BACKEND=bonsai JEV_BONSAI_GGUF=/path/to/model.gguf   # llama.cpp fork, the default path
JEV_MODEL=models/Qwen3.5-9B-AWQ                          # vLLM, needs requirements-vllm.txt
```

The tokenizer is a separate, much smaller thing: the CPU tests fetch vocabulary
only (~22 MB) from `Qwen/Qwen3.5-9B` and never need weights. The Ornith
`tokenizer.json` is byte-identical to Qwen3.5-9B's, which is why the same
window-encoding assertions hold for both.
