"""BonsaiLLM: a `vllm.LLM.generate`-shaped backend on top of the llama.cpp fork.

The engine only ever touches `generate(prompts, sps)` and reads
`out.outputs[0].logprobs[0][token_id].logprob` -- `core/fake_llm.FakeLLM` is the
proof of how little that is. This class provides the same shape over a resident
C++ worker (`backends/llamacpp/fork_worker.cpp`) that holds the model.

Division of labour: the worker is dumb (decode these tokens, give me the full
log-softmax at the last position). Everything here is policy.

Prefix policy -- two levels, nothing is ever truncated:

    seq 0  catalog+pad   learned, kept across calls
    seq 1  + state       rebuilt every call
    seq 2+ + node tail   one batch

The catalog boundary is not told to us by the engine (the seam stays at
`generate`), it is *learned*: within one call the longest common prefix contains
the state, but the prefix shared by two consecutive calls is exactly the catalog
(measured: LCP 1058 within a call, 535 across calls). So the catalog is
known from the second call on; call 1 is cold by construction and is reported as
such rather than hidden.

Log-probabilities are log-softmax over the WHOLE vocabulary. Do not renormalize
over the requested ids here -- the engine computes `in_set_mass` from exactly
these numbers, and renormalizing would pin it at 1.0 and silently disable the
format-bug check (AGENTS 4.7).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "backends", "llamacpp", "fork_worker")
CUDA_LIB = os.path.join(REPO, "backends", "llamacpp", "cuda13", "lib64")
FORK_LIB = os.path.join(REPO, "backends", "llamacpp", "llama.cpp-prism", "build", "bin")

# catalogue slots held resident; callers that alternate question sets need >1
SLOTS = 4


DEFAULT_GGUF = "/home/bean/Develope/Bonsai/models/Ternary-Bonsai-2-27B-PQ2_0.gguf"


def from_env(**overrides):
    """Build the backend from JEV_BONSAI_* so every caller agrees on the knobs.

    JEV_BONSAI_CTX is per SEQUENCE, which is the budget that binds (llama.cpp
    divides n_ctx by n_seq_max). JEV_BONSAI_SEQS must cover the widest trie the
    engine builds plus two -- clinc150's 150-label catalogue reaches 29 nodes --
    and costs about 150 MiB of recurrent state each on this hybrid model, so it
    is sized per workload rather than set high once.
    """
    kw = dict(ctx_per_seq=int(os.environ.get("JEV_BONSAI_CTX", "2048")),
              n_seq_max=int(os.environ.get("JEV_BONSAI_SEQS", "13")),
              n_slots=int(os.environ.get("JEV_BONSAI_SLOTS", str(SLOTS))),
              n_ubatch=int(os.environ.get("JEV_BONSAI_UBATCH", "0")),
              worker=os.environ.get("JEV_BONSAI_WORKER", WORKER))
    kw.update(overrides)
    return BonsaiLLM(os.environ.get("JEV_BONSAI_GGUF", DEFAULT_GGUF), **kw)


class _Logprob:
    __slots__ = ("logprob",)

    def __init__(self, logprob: float):
        self.logprob = logprob


class _Output:
    def __init__(self, table: dict):
        self.token_ids = [0]
        self.logprobs = [table]
        self.text = ""


class _RequestOutput:
    def __init__(self, table: dict):
        self.outputs = [_Output(table)]


def _lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _lcp_all(seqs) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        t = seqs[0][i]
        if any(s[i] != t for s in seqs):
            return i
    return n


def slot_key(lcp_ids, shared_len: int) -> str:
    """Identity of a catalogue: the engine's own boundary, hashed.

    Exact from the first call, because the engine tells us `shared_len` through
    SamplingParams.extra_args instead of leaving us to infer it. The three
    heuristics this replaces -- a four-call window, a pairwise comparison and an
    8-token margin -- were all patches for having to guess, and two of them were
    wrong in ways that cost nothing but speed, which is the hardest kind to see.
    """
    h = hashlib.sha1()
    for t in lcp_ids[:shared_len]:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()[:16]


def plan_slot(slots, keys, order, lcp_ids, shared_len, key):
    """Pick the slot and what it should hold: exactly the catalogue, always.

    The key is a hash of precisely those `shared_len` tokens, so a slot carrying
    the key already holds them; there is nothing to estimate and nothing to grow.

    An earlier version extended the resident prefix past the catalogue, taking
    the extra from the previous call with the same key. It reused more when warm
    -- a game state is a dict whose leading fields are often constant -- but game
    2's state head changes every attempt, so the extension was invalidated and
    rebuilt on nearly every call: 51 loads against 21, and the median warm call
    got slower (227 ms against 214). Worse, the resident length then depended on
    the ORDER calls arrived in, and so did the last digits of the answers. A
    fixed cut costs about 391 tokens of reuse on the games and buys a result that
    does not depend on call history. That trade was the user's call.
    """
    want = list(lcp_ids[:shared_len])
    for i, k in enumerate(keys):
        if k == key:
            return (i, None) if slots[i] == want else (i, want)
    return _victim(slots, order), want


def _victim(slots, order):
    """An empty slot if there is one, otherwise the least recently used."""
    for i, tok in enumerate(slots):
        if not tok:
            return i
    for i in reversed(order):
        return i
    return 0


class BonsaiWorkerError(RuntimeError):
    pass


class BonsaiLLM:
    """Resident llama.cpp-fork worker behind vLLM's `generate` shape."""

    # This backend has NO block-aligned cache -- it reuses an exact token prefix
    # with seq_cp, so padding buys it nothing (measured: 121 tokens of emotion
    # prompt padded to 1056, 88 % of the decode). 528 is vLLM's block, declared
    # here on purpose so a run CAN be made byte-identical to the published
    # collects that every cross-model comparison rests on -- set PAD=1 for that.
    # It is not the default any more: pad_shared_default() leaves this backend
    # unpadded, because padding for a cache it does not have cost 4.9x on the
    # full h7 collect and changed no accuracy.
    block_size = 528

    def __init__(self, model_path: str, *, ctx_per_seq: int = 2048,
                 n_seq_max: int = 13, ngl: int = 99, n_slots: int = SLOTS,
                 n_ubatch: int = 0, worker: str = WORKER,
                 log_path: str = "runs/bonsai_worker.log",
                 calls_path: str = "runs/bonsai_calls.jsonl"):
        """`ctx_per_seq` is the budget one sequence gets, which is the number
        that actually binds: llama.cpp splits n_ctx across n_seq_max sequences,
        so asking for n_ctx=4096 with 16 sequences gives 256 cells each and a
        1583-token game prompt dies with "find_slot: n_tokens = 1583 > size =
        256". The multiplication lives here so no caller has to remember it."""
        n_ctx = ctx_per_seq * n_seq_max
        # one decode can be a whole sequence's prompt, so the batch budget is
        # not an independent knob -- setting it lower aborts the worker on a
        # GGML_ASSERT rather than failing the call
        n_batch = ctx_per_seq
        # n_ubatch only sizes the compute buffer; llama.cpp micro-batches for us,
        # so it can be well under n_batch (4.2 GiB at 4096 against ~1 at 1024)
        if not os.path.exists(worker):
            raise FileNotFoundError(
                f"{worker} not built -- see backends/llamacpp/README.md")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        # llama.cpp is chatty on stderr; a pipe nobody drains fills at 64 KB and
        # wedges the worker mid-game, so it goes to a file (which doubles as the
        # load-time / VRAM evidence for the resource table).
        self._errfile = open(log_path, "ab", buffering=0)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [CUDA_LIB, FORK_LIB, env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [worker, model_path, str(n_ctx), str(n_seq_max), str(ngl), str(n_batch),
             str(n_slots), str(n_ubatch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errfile,
            text=True, bufsize=1, env=env)
        ready = self._readline()
        if not ready.startswith("READY"):
            raise BonsaiWorkerError(f"worker did not start: {ready!r}{self._tail()}")
        _, n_vocab, ctx, seqmax, per_seq, batch, slots, ubatch = ready.split()
        self.n_vocab, self.n_ctx = int(n_vocab), int(ctx)
        self.n_seq_max, self.ctx_per_seq = int(seqmax), int(per_seq)
        self.n_batch, self.n_slots = int(batch), int(slots)
        self.n_ubatch = int(ubatch)
        self.load_s = time.perf_counter() - t0
        self.calls = 0
        self.catalog_calls = 0          # how many times a catalogue was decoded
        self.slot_hits = 0              # calls that reused a resident catalogue
        self._slots: list[list[int]] = [[] for _ in range(self.n_slots)]
        self._keys: list[str | None] = [None] * self.n_slots
        self._order: list[int] = []          # most recently used first
        self._calls_path = calls_path

    # ---------------------------------------------------------------- io ----
    def _tail(self, n: int = 20) -> str:
        try:
            with open(self._errfile.name, "r", errors="replace") as f:
                lines = f.read().splitlines()[-n:]
            return "\n  worker log tail:\n    " + "\n    ".join(lines)
        except Exception:
            return ""

    def _readline(self) -> str:
        line = self.proc.stdout.readline()
        if line == "":
            raise BonsaiWorkerError(
                f"worker died (exit {self.proc.poll()}){self._tail()}")
        line = line.strip()
        if line.startswith("ERR "):
            raise BonsaiWorkerError(f"worker: {line[4:]}{self._tail()}")
        return line

    def _send(self, text: str) -> None:
        try:
            self.proc.stdin.write(text)
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise BonsaiWorkerError(f"worker closed its input{self._tail()}")

    # ------------------------------------------------------------ catalog ----
    def _set_catalog(self, slot: int, ids: list[int]) -> float:
        self._send("CAT %d %d %s\n" % (slot, len(ids), " ".join(map(str, ids))))
        rep = self._readline().split()
        if rep[0] != "CATOK" or int(rep[1]) != slot or int(rep[2]) != len(ids):
            raise BonsaiWorkerError(f"bad CAT reply: {rep}")
        self._slots[slot] = list(ids)
        self.catalog_calls += 1
        return float(rep[3])

    def _plan_catalog(self, lcp_ids: list[int],
                      shared_len: int) -> tuple[int, float, bool]:
        """Choose the slot; returns (slot, ms spent loading it, was it loaded).

        The decisions are `slot_key` and `plan_slot`, kept pure so they can be
        tested without a worker; this method is only the IO and the bookkeeping.
        """
        key = slot_key(lcp_ids, shared_len)
        slot, install = plan_slot(self._slots, self._keys, self._order, lcp_ids,
                                  shared_len, key)
        if slot in self._order:
            self._order.remove(slot)
        self._order.insert(0, slot)
        if install is None:
            self.slot_hits += 1
            return slot, 0.0, False
        self._keys[slot] = key
        return slot, self._set_catalog(slot, install), True

    # ----------------------------------------------------------- generate ----
    def generate(self, prompts, sps, use_tqdm: bool = False):
        t_wall = time.perf_counter()
        prompts = [prompts] if isinstance(prompts, dict) else list(prompts)
        sps = sps if isinstance(sps, list) else [sps] * len(prompts)
        seqs = [list(p["prompt_token_ids"]) for p in prompts]
        cands = [list(getattr(sp, "logprob_token_ids", None) or []) for sp in sps]
        # the engine hands us the catalogue boundary; without it we would be back
        # to inferring it, so refuse rather than guess
        shared = {(getattr(sp, "extra_args", None) or {}).get("shared_len")
                  for sp in sps}
        if len(shared) != 1 or None in shared:
            raise BonsaiWorkerError(
                "every request in a call must carry the same extra_args"
                f"['shared_len']; got {sorted(x for x in shared if x is not None)}"
                " -- is the engine current? (core/jev_engine._shared_ids)")
        shared_len = int(shared.pop())

        if len(seqs) + self.n_slots + 1 > self.n_seq_max:
            raise BonsaiWorkerError(
                f"{len(seqs)} sequences with {self.n_slots} catalogue slots need "
                f"n_seq_max {len(seqs) + self.n_slots + 1}, worker was started "
                f"with {self.n_seq_max}")
        longest = max(len(s) for s in seqs)
        if longest > self.ctx_per_seq:
            raise BonsaiWorkerError(
                f"prompt of {longest} tokens over the per-sequence capacity "
                f"{self.ctx_per_seq} (n_ctx {self.n_ctx} / n_seq_max "
                f"{self.n_seq_max}); raise ctx_per_seq")

        # every scored position must live in a tail, so the shared part stops one
        # token short of the shortest prompt
        share = min(_lcp_all(seqs), min(len(s) for s in seqs) - 1)
        if share < 0:
            raise BonsaiWorkerError("a prompt is empty")
        lcp_ids = seqs[0][:share]

        shared_len = min(shared_len, share)   # a warm call can be shorter
        slot, cat_ms, installed = self._plan_catalog(lcp_ids, shared_len)
        catalog = self._slots[slot]
        work = lcp_ids[len(catalog):]

        parts = ["RUN %d %d %s %d" % (slot, len(work), " ".join(map(str, work)),
                                      len(seqs))]
        for s, c in zip(seqs, cands):
            tail = s[share:]
            # the worker refuses an empty tail; `share` above guarantees one
            parts.append("%d %s %d %s" % (len(tail), " ".join(map(str, tail)),
                                          len(c), " ".join(map(str, c))))
        self._send(" ".join(parts) + "\n")

        head = self._readline().split()
        if head[0] != "RES" or int(head[1]) != len(seqs):
            raise BonsaiWorkerError(f"bad RUN reply: {head}")
        outs = []
        for i in range(len(seqs)):
            f = self._readline().split()
            if f[0] != "LP" or int(f[1]) != i or int(f[2]) != len(cands[i]):
                raise BonsaiWorkerError(f"bad LP line {i}: {f[:3]}")
            vals = [float(x) for x in f[3:]]
            outs.append(_RequestOutput(
                {tid: _Logprob(v) for tid, v in zip(cands[i], vals)}))
        done = self._readline().split()
        if done[0] != "DONE":
            raise BonsaiWorkerError(f"bad DONE line: {done}")

        self.calls += 1
        self._log_call(n_seq=len(seqs), reused=int(done[1]), decoded=int(done[2]),
                       worker_ms=float(done[3]), catalog_ms=cat_ms, slot=slot,
                       installed=installed,
                       wall_ms=(time.perf_counter() - t_wall) * 1000)
        return outs

    def _log_call(self, **row) -> None:
        row["call"] = self.calls
        row["catalog_len"] = len(self._slots[row["slot"]])
        try:
            os.makedirs(os.path.dirname(self._calls_path) or ".", exist_ok=True)
            with open(self._calls_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass  # a diagnostics file must never take the run down

    # --------------------------------------------------------------- life ----
    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send("QUIT\n")
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        try:
            self._errfile.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
