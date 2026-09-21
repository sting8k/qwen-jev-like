# backends/llamacpp — gate only, not a backend yet

Nothing here is wired into the engine. This exists to answer the three questions in
`reports/llamacpp_recon.md` §f before anyone writes a second backend. The user approved
a 2-hour gate; `core/` is untouched.

The checkout and the CUDA shadow root are gitignored — only this README, the two
sources below, and the measured numbers are tracked.

## What is pinned

| thing | value |
|---|---|
| llama.cpp commit | `e613ef2c81bae98d59850d061ac29e6e3e88cb00` (shallow clone of master, 2026-09-20) |
| model | `unsloth/Qwen3.5-4B-GGUF` → `Qwen3.5-4B-Q4_K_M.gguf`, 2.74 GB |
| model sha256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` (verified after download) |
| model path | `models/Qwen3.5-4B-GGUF/` (gitignored, like every other model) |

4B is the *cheap mechanism probe*: same `qwen3_5` hybrid GDN family as the 9B, so prefix
reuse and the GDN prefill path behave the same way. It is not a quality comparison.
Do not download the 9B Q8_0 (9.53 GB) until the gate passes.

## Building (three traps, all paid for already)

1. **The `nvcc` on PATH is 11.7** and `/usr/local/cuda-12.0` ships the cuBLAS runtime
   *without* headers — configure dies at `CUDA::cublas` / `cublas_v2.h: No such file`.
   The only complete toolkit is CUDA 13.4 inside the venv.
2. **`CUDA_HOME` is exported as `/usr/local/cuda-11.7`** in this shell. nvcc honours it and
   pre-includes 11.7's `cuda_runtime.h`, so CCCL aborts with *"CUDA compiler and CUDA
   toolkit headers are incompatible"*. It must be overridden, not just ignored.
3. The pip toolkit lays libraries out as `lib/`, with versioned `.so.13` only. nvcc looks
   for `../lib64` relative to its own *real* path, so `-lcudadevrt` / `-lcudart_static`
   are not found. Hence the shadow root below, plus `LIBRARY_PATH`.

```sh
V=$PWD/../../.venv/lib/python3.12/site-packages/nvidia/cu13
S=$PWD/cuda13                      # shadow toolkit root (symlinks only, venv untouched)
mkdir -p "$S/lib64"
ln -s "$V/bin" "$S/bin"; ln -s "$V/include" "$S/include"; ln -s "$V/nvvm" "$S/nvvm"
for f in "$V"/lib/*.so.13* "$V"/lib/*.a; do ln -sf "$f" "$S/lib64/$(basename $f)"; done
for n in cublas cublasLt cudart nvrtc; do
  ln -sf "$(ls $S/lib64/lib$n.so.* | head -1)" "$S/lib64/lib$n.so"; done

export CUDA_HOME=$S CUDA_PATH=$S LIBRARY_PATH=$S/lib64 LD_LIBRARY_PATH=$S/lib64
unset CPATH                        # setting it reorders the CCCL headers and breaks the build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=$S/bin/nvcc -DCUDAToolkit_ROOT=$S \
      -DCMAKE_LIBRARY_PATH=$S/lib64 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12 --target llama-server llama-bench
```

**Caveat on `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`**: the venv ships nvcc 13.4 with CUDA
runtime headers 13.0 (two different pip packages), and CCCL refuses that pairing by
default. The flag turns the refusal off. Minor-version skew between compiler and CTK
headers is normally harmless, but it is a real caveat: if a number from this build looks
wrong, suspect this before suspecting the model.

## The PrismML fork (Bonsai-27B path)

`llama.cpp-prism/` is a second checkout: PrismML-Eng/llama.cpp at tag
`prism-b10685-7dffb15` (= `7dffb158de30ebb8ef9d64f33c6b0b2d7c1e6313`, the tag the
shipped binaries in `../Bonsai/bin` came from). Gitignored like the upstream one.

It exists because **stock llama.cpp cannot load Bonsai**: the fork adds the low-bit
types `PQ2_0` (ggml id 142, group 128) and `PTQ1_0` with their own CUDA/Vulkan
kernels. It also carries `kv-mean-center` (a K-cache bias for Q4_0 KV) and
`examples/rs-rollback` (a correctness harness for the recurrent-state snapshot ring).

**The source had to be cloned.** The release tarball is binaries only — no headers,
no source — and the `.so` files are stripped. Which matters because:

> **Never compile against upstream's `llama.h` and link the fork's `libllama.so`.**
> The symbol sets are one function apart (240 vs 241 `llama_*`), so it links and
> runs. But the fork inserts `const char * path_kv_mean_center` into the *middle*
> of `llama_context_params`, and upstream adds `lazy_mode` to `llama_model_params`
> — both structs are passed **by value**. Every field after the insertion point
> would read from the wrong offset and the probe would print plausible numbers.
> The fork's own README says the same thing: do not mix its `ggml-*` libraries
> with a stock build.

Build (same shadow root and the same three traps as above; `configure` took 4 s and
the build 6m34s on 16 cores):

```sh
cd backends/llamacpp/llama.cpp-prism
S=$PWD/../cuda13
export CUDA_HOME=$S CUDA_PATH=$S LIBRARY_PATH=$S/lib64 LD_LIBRARY_PATH=$S/lib64
unset CPATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_CUDA_COMPILER=$S/bin/nvcc -DCUDAToolkit_ROOT=$S \
      -DCMAKE_LIBRARY_PATH=$S/lib64 -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama llama-bench
```

Compiling `fork_probe.cpp` against the fork (this command was never written down for
the 4B build — that gap cost a rebuild):

```sh
cd backends/llamacpp
S=$PWD/cuda13; B=$PWD/llama.cpp-prism/build/bin
g++ -O2 -std=c++17 fork_probe.cpp -o fork_probe_bonsai \
    -I llama.cpp-prism/include -I llama.cpp-prism/ggml/include \
    -L "$B" -lllama -lggml -lggml-base -L "$S/lib64" \
    -Wl,-rpath,"$B" -Wl,-rpath,"$S/lib64" -Wl,-rpath-link,"$S/lib64"
ldd fork_probe_bonsai | grep llama   # must resolve to llama.cpp-prism, not llama.cpp
```

`fork_worker.cpp` (the resident worker behind `core/bonsai_llm.py`) builds the same
way, just swap the two file names. It is a separate file from `fork_probe.cpp` on
purpose: the probe produced the published gate numbers, so it stays runnable and
untouched.

**`-rpath` on the executable is not enough at run time.** The linker writes
`DT_RUNPATH`, which glibc does *not* use for transitive dependencies:
`libggml-cuda.so` is the one that needs `libcudart.so.13` and its own RUNPATH does
not list the shadow root, so the probe dies with `error while loading shared
libraries` (exit 127) until you export it:

```sh
export LD_LIBRARY_PATH=$PWD/backends/llamacpp/cuda13/lib64
```

### Models run through the fork

| | Bonsai-2-27B | Qwen3.8-27B (PTQ column) |
|---|---|---|
| file | `Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7.21 GB | `Qwen3.8-27B-UD-Q2_K_XL.gguf`, 9.83 GB |
| repo | local (`../Bonsai/models/`) | `unsloth/Qwen3.8-27B-GGUF` |
| path | `../Bonsai/models/` (outside this repo) | `models/Qwen3.8-27B-GGUF/` (gitignored) |
| sha256 | — | `fd4730dd8aad070517978752b63d530aeb1740d2283cab9fa24f1e404032ddb0` (verified after download) |
| quant | ternary QAT, `PQ2_0` — needs the fork | PTQ ~2 bpw `Q2_K_XL` — a stock type |
| weights on CUDA0 | 6 540 MiB | 8 631 MiB |

The second one is here to answer **"is QAT necessary for Jev-mode"**, not "which model
is better": same `qwen35` arch and same vocabulary, quantized the ordinary way. Note it
is the *larger* of the two on the card — the ternary file is 7.21 GB against 9.83 GB, so
QAT is not being compared against a cheaper option here.

Weights are the one figure that does not move with the run: the compute buffer does, so
it belongs to a measured run and its sequence budget, not to this table (§5 of AGENTS).

**Check the vocabulary before spending GPU on a new GGUF.** The engine tokenizes with a
HF tokenizer and hands raw ids to the worker; if the ids mean different tokens on the
other side you still get a full probability table, and it is nonsense. Both files above
agree with `models/Qwen3.5-9B-AWQ` on all 248 077 ids that tokenizer can emit (each GGUF
carries 243 more at the tail, which the engine never emits).

**Never use 4-bit KV cache** (`-ctk q4_1` etc.) on this architecture: upstream #27109
measured prefill collapsing from ~1000 t/s to ~34 t/s on an RTX 3090.

The acceptance run that produced the numbers above lived in a harness that is not
part of this repository; what it concluded is in `docs/RESULTS.md`, and
`fork_probe.cpp` below is the probe it was built around.
