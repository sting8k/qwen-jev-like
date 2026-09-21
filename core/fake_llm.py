"""CPU stand-ins for vLLM's `LLM.generate`, so a script can run the engine
without a GPU and without importing a test module.

There are TWO of them and that is deliberate — they model different things,
and merging them would silently drop one guard:

`FakeLLM` seeds every logprob with the WHOLE prompt. Two states that share a
question set therefore produce different numbers, which is what makes a routing
bug visible: with a candidate-only fake, a batch that hands state A the logits
of state B compares equal and ships (it did — see 0c7634b, where the batch
bucket key was blind to option order and 300/300 rows were scored with another
state's catalogue). Used by the JevEngine tests, `run_server.py --fake` and
`tools/run_seg_diag.py`.

`CacheSimFakeLLM` models the prefix cache instead: the first call reports
nothing cached, every later call reports a hit. That is what the old
`ParallelScorer` dry run asserts on. Its logprobs do NOT depend on the prompt,
so it cannot see a routing bug — it was never meant to.

Extracted from `test_jev_cpu.py` and `test_cpu_sim.py`; both still re-export
their original name, so existing callers are unaffected.
"""


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


# --- prompt-seeded: for JevEngine, sees cross-state routing bugs ---


def _prompt_seed(prompt) -> int:
    # the WHOLE prompt: the tail of a scored request is the same JSON suffix
    # for every state, so hashing a suffix window makes states indistinguishable
    ids = prompt.get("prompt_token_ids", ()) if isinstance(prompt, dict) else ()
    return sum((i + 1) * t for i, t in enumerate(ids))


class FakeOutput:
    def __init__(self, ids, seed=0):
        self.token_ids = [0]
        # the value depends on the PROMPT as well as the candidate: with a
        # candidate-only fake, two states sharing a question set produce the
        # same numbers, so a batch that hands state A the logits of state B
        # still compares equal and the routing bug ships
        self.logprobs = [{tid: FakeLogprob(-0.5 - (i % 5) * 0.3 - (seed % 89) * 0.001)
                          for i, tid in enumerate(ids)}]
        self.text = ""


class FakeRequestOutput:
    def __init__(self, ids, seed=0):
        self.outputs = [FakeOutput(ids, seed)]
        self.num_cached_tokens = 0


class FakeLLM:
    # these stand in for vLLM, so they carry vLLM's cache block: the engine now
    # refuses to guess one (JevEngine._block_size)
    block_size = 528

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def generate(self, prompts, sps, use_tqdm=False):
        self.calls += 1
        prompts = [prompts] if isinstance(prompts, dict) else list(prompts)
        sps = sps if isinstance(sps, list) else [sps] * len(prompts)
        self.batch_sizes.append(len(prompts))
        return [
            FakeRequestOutput(sp.logprob_token_ids or [0], _prompt_seed(p))
            for p, sp in zip(prompts, sps)
        ]


# --- cache-simulating: for the old ParallelScorer dry run ---


class CacheSimOutput:
    def __init__(self, requested_ids, cached):
        self.token_ids = [0]
        self.logprobs = [{tid: FakeLogprob(-1.0 - (i % 7) * 0.1)
                          for i, tid in enumerate(requested_ids)}]
        # v.logprob must never be exactly 0.0


class CacheSimRequestOutput:
    def __init__(self, requested_ids, cached):
        self.outputs = [CacheSimOutput(requested_ids, cached)]
        self.num_cached_tokens = cached
        self.num_cache_creation_tokens = 0


class CacheSimFakeLLM:
    block_size = 528

    def __init__(self):
        self.calls = 0
        self.last_ids = []
        self.prompts = []

    def generate(self, prompts, sps, use_tqdm=False):
        import itertools

        if isinstance(prompts, dict):
            prompts = [prompts]
        sps = sps if isinstance(sps, list) else [sps] * len(prompts)
        self.calls += 1
        outs = []
        for p, sp in zip(prompts, sps):
            ids = sp.logprob_token_ids or [0]
            self.prompts.append(p["prompt_token_ids"])
            self.last_ids.append(list(ids))
            outs.append(CacheSimRequestOutput(ids, cached=0 if self.calls == 1 else 999))
        return outs
