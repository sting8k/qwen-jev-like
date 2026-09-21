"""jev-rlcd engine: Jev-contract parallel constrained scoring on vLLM (CUDA).

Contract (verbatim from typesafe/jev 1.13 docs):
  run(state, questions) -> {model, answers, usage}
  - state: str | JSON-serializable object
  - questions: {id: {type: noul|choice|score, instructions, criteria}}
      noul   criteria: optional {true: desc, false: desc}
      choice criteria: {option_key: description_per_option}
      score  criteria: [level0_desc, level1_desc, ...] (ordered)
  - answers:
      noul   -> {type, noul: P(true) in 0..1}
      choice -> {type, choice: key, confidence, probabilities}
      score  -> {type, score: E[level], confidence, legend, probabilities}

Mechanism:
  - one shared prompt (state + question catalog incl. criteria descriptions)
  - P(option) = product of conditional logprobs along the option-token TRIE,
    MINIMAL EXPANSION: only nodes with >=2 distinct child tokens are scored
    (single-child chains contribute a common ~constant and are dropped --
    "the decision is made at the divergence point"; changelog'd trade-off)
  - every scored node is known in advance -> ALL requests in ONE llm.generate
  - renormalize within option set; in_set_mass at each question's first scored
    node is reported as the format-adherence diagnostic
  - documented deviations: confidence = max(probs) for choice and
    1 - std/((n-1)/2) for score; temperature applies inside the softmax
  - score supports <=10 levels (digits are single tokens; '10' is not)
  - node budget (default 64) counts scored nodes only
"""

import json
import os
import math
import time

from vllm import LLM, SamplingParams

MAX_IDS_PER_REQUEST = 128
NODE_BUDGET = 64
# Which checkpoint the whole harness runs. JEV_MODEL swaps it for a POC; the
# label is derived from the path so provenance in every output follows the
# weights automatically (default reproduces "jev-rlcd-qwen3.5-9b-awq" exactly).
MODEL_PATH = os.environ.get("JEV_MODEL", "models/Qwen3.5-9B-AWQ")


def require_model() -> str:
    """Fail on a missing checkout here, with the instruction, rather than in the
    hub client. A relative path that does not exist is indistinguishable from a
    repository id, so the loaders report it as a 404 against a name nobody typed.
    Entry points call this before building anything."""
    if not os.path.isdir(MODEL_PATH):
        raise SystemExit(
            f"no model at {MODEL_PATH!r}.\n"
            f"Set JEV_MODEL to a local checkout, or download weights into "
            f"models/. The CPU tests need no weights: python -m tests.test_cpu")
    return MODEL_PATH
MODEL_LABEL = "jev-rlcd-" + os.path.basename(MODEL_PATH.rstrip("/")).lower()
# Phase-3 collection ran the 9B on the awq_marlin kernel; keep that the default
# so those numbers stay reproducible. A checkpoint quantized another way (e.g.
# compressed-tensors) needs JEV_QUANT="" -> None -> vLLM reads config.json.
MODEL_QUANT = os.environ.get("JEV_QUANT", "awq_marlin") or None


class QSpec:
    """Question spec; accepts `type=` (Jev wire format) as alias for qtype."""

    def __init__(self, qid: str, qtype: str | None = None, instructions: str = "",
                 criteria=None, **aliases):
        if qtype is None:
            qtype = aliases.get("type")
        assert qtype in ("noul", "choice", "score"), f"bad question type: {qtype!r}"
        self.qid = qid
        self.qtype = qtype
        self.instructions = instructions or ""
        self.criteria = criteria
        self.options: list[str] = []
        self._resolve()

    def _resolve(self):
        if self.qtype == "noul":
            self.options = ["true", "false"]
        elif self.qtype == "score":
            assert isinstance(self.criteria, (list, tuple)), "score criteria must be a list"
            assert 2 <= len(self.criteria) <= 10, (
                f"score supports 2..10 levels (digits are single tokens); got {len(self.criteria)}"
            )
            self.options = [str(i) for i in range(len(self.criteria))]
        else:  # choice
            assert isinstance(self.criteria, dict) and len(self.criteria) >= 2, (
                "choice criteria must be a map of >=2 options"
            )
            self.options = list(self.criteria.keys())


def pad_shared_default() -> bool:
    """Whether the shared prefix is padded to a cache block, when nothing says.

    Padding belongs to the BACKEND, not to the run. On vLLM the 528-token block
    is real: padding makes the catalogue block set hash-identical across states
    and buys the cross-call prefix hit (809 ms -> 42 ms, §4.3 of AGENTS). The
    llama.cpp fork has no such cache, so there the same padding is pure cost --
    it stretched a 121-token emotion prompt to 1056 and made the h7 collect 4.9x
    slower for accuracy unchanged inside +-1.2 points.

    `PAD` still overrides, because the padded prompt is what the vLLM and Jev
    numbers were made with and comparisons against them have to reproduce it.
    """
    override = os.environ.get("PAD", "").strip()
    if override:
        return override == "1"
    return os.environ.get("JEV_BACKEND", "").strip() != "bonsai"


class JevEngine:
    def __init__(self, llm: LLM, tokenizer, temperature: float = 1.0,
                 pad_shared: bool | None = None):
        self.llm = llm
        self.tok = tokenizer
        self.temperature = temperature
        self.pad_shared = pad_shared_default() if pad_shared is None else pad_shared
        self._jit_warmed = False

    # ---------- tokenization ----------

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _diff(self, base_ids: list[int], continuation: str) -> list[int]:
        """Tokens of (base+continuation) minus base. Never trust standalone
        tokenization (space merge / BPE boundary).
        Window-encoded: a full re-encode is O(len(base)) and the catalogs
        re-encode 1.5-8.5K tokens per question (0.2-2s of pure CPU)."""
        win, win_text = self._stable_window(base_ids)
        fw = self._ids(win_text + continuation)
        if fw[: len(win)] != win:
            # merge spans the window -- full-encode fallback
            base_text = self.tok.decode(base_ids)
            full = self._ids(base_text + continuation)
            assert full[: len(base_ids)] == base_ids, f"tokenization drift: {continuation!r}"
            d = full[len(base_ids):]
        else:
            d = fw[len(win):]
        assert d, f"empty diff: {continuation!r}"
        return d

    def _stable_window(self, base_ids: list[int], K: int = 16):
        """Last K tokens of base that re-encode to themselves standalone.
        A window starting mid-BPE-run (e.g. inside a long '\n' run) parses
        differently -- widen by 8 until stable."""
        while K <= len(base_ids):
            cand = base_ids[-K:]
            cand_text = self.tok.decode(cand)
            if self._ids(cand_text) == cand:
                return cand, cand_text
            K += 8
        b = list(base_ids)
        return b, self.tok.decode(b)

    def _paths_for(self, base_ids: list[int], conts: dict):
        """Token path per continuation, tolerating boundary merges by shrinking
        the base (a trailing space/quote token re-merges into the continuation).

        The prompt TEXT is invariant under that shrink: characters dropped from
        the base ride on the continuation, so a scored path always spells
        decode(base_ids) + cont. Encoding only `decode(base) + cont` instead
        silently drops those characters -- with a '.'-style option the base
        shrinks past '  "field": "' and the trie ends up asking for 'a' right
        after the colon, where the model wants the quote (measured: p(' "')
        0.987, every option then scored at ~e^-15 and in_set_mass 0.0).

        Returns (aligned_base_ids, {option: [tokens]})."""
        full_text = self.tok.decode(base_ids)
        base = list(base_ids)
        for _ in range(4):
            text = self.tok.decode(base)
            assert full_text.startswith(text), (
                "base decode is not a text prefix of the original base")
            tail = full_text[len(text):]  # dropped chars belong to the continuation
            if len(conts) <= 4:
                enc = {o: self._ids(full_text + c) for o, c in conts.items()}
            else:
                # window encoding: full re-encode per option is O(len(text))
                # -- 255 options x 8ms = 2.1s on the tariff catalog (the
                # entire warm anomaly was this, CPU-side). Window must be
                # standalone-stable; widen if it starts mid-BPE-run.
                win, win_text = self._stable_window(base)
                enc = {}
                oracle_done = False
                for o, c in conts.items():
                    fw = self._ids(win_text + tail + c)
                    if fw[: len(win)] != win:
                        fw = None  # boundary merge spans the window
                    if (not oracle_done) or fw is None:
                        full = self._ids(full_text + c)  # oracle / fallback
                        if full[: len(base)] != base:
                            enc = None
                            break
                        if fw is not None:
                            assert full[len(base):] == fw[len(win):], (
                                f"window encoding diverged for {o!r}")
                        enc[o] = full
                        oracle_done = True
                    else:
                        enc[o] = base + fw[len(win):]
            paths, ok = {}, True
            if enc is None:
                ok = False
            else:
                for o, e in enc.items():
                    if e[: len(base)] != base:
                        ok = False
                        break
                    paths[o] = e[len(base):]
            if ok and paths:
                return base, paths
            if len(base) <= 1:
                break
            base = base[:-1]
        raise AssertionError(f"cannot align continuation tokenization: {list(conts.values())[:2]}")

    @staticmethod
    def _prompt(ids: list[int]) -> dict:
        return {"prompt_token_ids": ids}

    def _block_size(self) -> int:
        """Cache block the shared prefix is padded to.

        A backend may declare it as `.block_size`; vLLM is read from its own
        config. Anything else is an error rather than a default, because the
        silent 528 was being inherited by a backend that has no block structure
        at all: on the llama.cpp fork it padded a 121-token emotion prompt to
        1056, so 88 % of every decode was padding bought for a cache the fork
        does not use.
        """
        declared = getattr(self.llm, "block_size", None)
        if declared is not None:
            return int(declared)
        try:
            return self.llm.llm_engine.vllm_config.cache_config.block_size
        except Exception as exc:
            raise RuntimeError(
                f"{type(self.llm).__name__} declares no block_size and is not a "
                "vLLM engine: padding cannot be sized. Set `.block_size` on the "
                "backend (see core/bonsai_llm.py) or run with PAD=0."
            ) from exc

    # ---------- prompt building ----------

    def _shared_prompt(self, state, questions: list[QSpec]) -> str:
        state_text = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
        q_lines = []
        for q in questions:
            if q.qtype == "noul":
                crit = q.criteria or {}
                t_desc = crit.get("true", "yes")
                f_desc = crit.get("false", "no")
                # label "(true/false)" + quoted path: the model echoes the
                # catalog's vocabulary. "(noul)" -> digits (0.89 mass,
                # diag_noul.log); "(yes/no)" -> 'yes' strings (0.80-0.97,
                # diag_noul2.log); "(true/false)" + `  "qid": "` suffix ->
                # 'true'/'false' at 0.999-1.000 combined mass (diag_noul2).
                q_lines.append(
                    f"- {q.qid} (true/false): {q.instructions}\n"
                    f"    true: {t_desc}\n    false: {f_desc} "
                    "(answer \"true\" or \"false\")"
                )
            elif q.qtype == "choice":
                pairs = "\n".join(f"    {k}: {v}" for k, v in q.criteria.items())
                q_lines.append(f"- {q.qid} (choice): {q.instructions}\n{pairs}")
            else:
                lv = "\n".join(f"    {i}: {d}" for i, d in enumerate(q.criteria))
                q_lines.append(f"- {q.qid} (score): {q.instructions}\n{lv}")
        catalog = "\n".join(q_lines)
        # invariant: every scored continuation must appear verbatim in the
        # catalog -- the model echoes catalog vocabulary (noul saga: "(noul)"
        # label -> digits 0.89 mass; "(yes/no)" -> 'yes' strings; only
        # "(true/false)" put mass on the scored tokens). A continuation the
        # catalog never shows fights the format the model expects.
        for q in questions:
            for o in q.options:
                assert o in catalog, f"{q.qid}: option {o!r} not in catalog text"
        msgs = [
            {
                "role": "user",
                "content": (
                    "You are a calibrated decision engine. Answer every question "
                    "with exactly one allowed value based on the STATE, as a JSON "
                    "object mapping each question id to its answer.\n\n"
                    # catalog FIRST, state LAST: the catalog is identical across
                    # requests in production, so its cache blocks must hash
                    # identically -- a varying state at the top would invalidate
                    # every block after it.
                    f"QUESTIONS:\n{catalog}\n\nSTATE:\n{state_text}"
                ),
            }
        ]
        try:
            p = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:
            p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return p + "{\n"

    def _pad_align(self, head: str, tail: str) -> tuple[list[int], str]:
        """ids of head+filler+tail with len % block_size == 0. The filler is a
        '#\n' wall inserted between head and tail: Qwen BPE packs "\n" runs
        densely (a 32-newline run is ONE token) so blank-line padding cannot
        be steered, while '#\n' is exactly 2 tokens at any count, canonically
        stable, and never merges with the trailing "{\n". Odd counts append a
        lone 'x' ('#'/'.'/'-' merge into #{ .{ -{ tokens and add 0).
        Returns (ids, padded_text) or raises."""
        bs = self._block_size()
        base = self._ids(head + tail)
        e0 = (-len(base)) % bs
        if e0 == 0:
            return base, head + tail
        for delta in (0, 1, -1, 2, -2, 3, -3, 4):
            e = e0 + delta
            if e < 0:
                continue
            filler = "#\n" * (e // 2) + ("x" if e % 2 else "")
            text = head + filler + tail
            cand = self._ids(text)
            if len(cand) % bs == 0 and self._ids(self.tok.decode(cand)) == cand:
                return cand, text
        raise AssertionError("cannot align with '#\\n' filler")

    # ---------- trie plan (single source of truth for build + routing) ----------

    def _plan(self, shared_ids: list[int], questions: list[QSpec]) -> dict:
        """Build the trie plan. For each question:
        nodes = [(node_prefix_tokens, {child_token: [options]})] -- SCORED nodes
        only (>=2 distinct children); single-child chains are traversed but not
        scored (their factor is common to the surviving options and cancels).
        Also returns the flat request list across all questions."""
        requests: list[tuple[list[int], list[int]]] = []  # (prompt_ids, candidate ids)
        plan: dict[str, dict] = {}

        for q in questions:
            # suffix ends ': ' / ': "'; continuations carry NO leading space.
            # _paths_for resolves per-question whether the space token merges.
            if q.qtype in ("choice", "noul"):
                # noul rides the quoted-string path: at the bare-value position
                # the model wants digits ('1' 0.49 / '0' 0.30) or ' "' -- even
                # with a yes/no label the true/false literals keep a ~0.1%
                # sliver (noul_fix_sanity). After the opening quote the model
                # completes JSON strings naturally.
                suffix = f'  "{q.qid}": "'
                conts = {o: o + '"' for o in q.options}
            else:  # score: bare JSON value
                suffix = f'  "{q.qid}": '
                conts = {o: o for o in q.options}

            suffix_ids = self._diff(shared_ids, suffix)
            base_ids = shared_ids + suffix_ids
            base_ids, paths = self._paths_for(base_ids, conts)
            assert len({tuple(p) for p in paths.values()}) == len(paths), (
                f"ambiguous paths in {q.qid}"
            )

            nodes = []
            active = {(): list(q.options)}
            while active:
                new_active = {}
                for node, opts in sorted(active.items()):
                    child: dict[int, list[str]] = {}
                    for o in opts:
                        tid = paths[o][len(node)]
                        child.setdefault(tid, []).append(o)
                    if len(child) >= 2:  # scored node only
                        nodes.append((list(node), dict(child)))
                        requests.append((base_ids + list(node), sorted(child)))
                    for tid, opts_here in child.items():
                        deeper = [o for o in opts_here
                                  if len(paths[o]) > len(node) + 1]
                        if deeper:
                            new_active[tuple(list(node) + [tid])] = deeper
                active = new_active
            assert nodes, f"no scored node for {q.qid} (paths not distinct?)"
            plan[q.qid] = {"q": q, "base_ids": base_ids, "paths": paths, "nodes": nodes}

        assert len(requests) <= NODE_BUDGET, (
            f"trie scored {len(requests)} nodes > budget {NODE_BUDGET}; "
            "shorten option keys or use a code-mapped fallback (not in v1)"
        )
        return {"requests": requests, "plan": plan}

    # ---------- main ----------

    @staticmethod
    def _token_lcp(a: list[int], b: list[int]) -> int:
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _shared_ids(self, state, questions: list[QSpec]) -> tuple[list[int], int]:
        """The prompt ids every scored request is built on. THE only way to
        build them: a diagnostic that re-derives the prompt without the pad
        walls reads a different distribution than the one it is explaining
        (leak_diag did exactly that -- its top-K disagreed with the scored
        row's in_set_mass because the two prompts were not the same prompt).

        Padding aligns the shared prefix to whole cache blocks: the remainder
        tokens below one block are re-prefilled per request (~60-80ms each --
        verified: sawtooth remainder x nreq == fields_ms). Padding is applied
        at STRING level before the trailing "{\n"; one canonical tokenization
        of the whole padded string keeps re-encoding stable (repeating raw
        token ids broke it: re-encoding merges "\n" runs differently and the
        _diff drift assert fires).

        Two stages for the [catalog][STATE] layout:
          1) the catalog prefix (incl. the "STATE:\n" marker) ends exactly on
             a block boundary -> catalog blocks hash identically across
             requests with different states (cross-request cache hits);
          2) state + "{\n" tail also ends on a boundary -> per-request
             uncached tokens stay ~suffix only (sawtooth fix).
        """
        shared = self._shared_prompt(state, questions)
        shared_ids = self._ids(shared)
        st_text = state if isinstance(state, str) else json.dumps(
            state, indent=2, ensure_ascii=False)
        # the chat template may strip the state's surrounding whitespace, so
        # anchor on the stripped form (data states are not always clean)
        probe = st_text.strip() or st_text
        cut = shared.rindex(probe)
        if not self.pad_shared:
            return shared_ids, self._token_lcp(self._ids(shared[:cut]), shared_ids)
        prefix, rest = shared[:cut], shared[cut:]
        _, t1 = self._pad_align(prefix, "")
        shared_ids, _ = self._pad_align(t1 + rest[:-2], "{\n")
        # How many leading tokens are catalogue rather than state -- everything
        # before the "STATE:" marker, padding included. Measured as a token LCP
        # and not as len(ids(prefix)): BPE can merge across the boundary, so the
        # two tokenizations need not agree on the last unit. A backend that can
        # hold a prefix resident (the llama.cpp fork) is handed this through
        # SamplingParams.extra_args instead of guessing it from call history.
        return shared_ids, self._token_lcp(self._ids(t1), shared_ids)

    def _decode(self, qspecs: list[QSpec], plan: dict, outs, T: float):
        """Route ONE state's logprob outputs -- given in plan order -- into
        answers. Shared by run() and run_many() so the two paths cannot drift:
        batching changes which generate call produced a logprob, nothing about
        how it is read."""
        answers: dict[str, dict] = {}
        engine_diag: dict[str, dict] = {}
        zero_logprobs = 0
        out_iter = iter(outs)
        for q in qspecs:
            pq = plan[q.qid]
            cond = {o: 0.0 for o in q.options}
            in_set_mass = None
            for node, child in pq["nodes"]:
                out = next(out_iter)
                lp = out.outputs[0].logprobs[0]
                lps = {}
                for tid, opts_here in child.items():
                    v = lp.get(tid)
                    assert v is not None, f"missing logprob {tid} in {q.qid}"
                    if v.logprob == 0.0:
                        zero_logprobs += 1  # legit only under extreme confidence; warn
                    lps[tid] = v.logprob
                    for o in opts_here:
                        cond[o] += v.logprob
                if in_set_mass is None:  # first scored node = format adherence
                    m = max(lps.values())
                    in_set_mass = sum(math.exp(v - m) for v in lps.values()) * math.exp(m)

            mx = max(cond.values())
            exp = {o: math.exp((v - mx) / T) for o, v in cond.items()}
            Z = sum(exp.values())
            probs = {o: e / Z for o, e in exp.items()}
            top = sorted(probs.items(), key=lambda kv: -kv[1])

            if q.qtype == "noul":
                answers[q.qid] = {"type": "noul", "noul": round(probs["true"], 4)}
            elif q.qtype == "choice":
                w, wp = top[0]
                answers[q.qid] = {
                    "type": "choice",
                    "choice": w,
                    "confidence": round(wp, 4),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()},
                }
            else:  # score
                n = len(q.options)
                ev = sum(i * probs[str(i)] for i in range(n))
                var = sum(((i - ev) ** 2) * probs[str(i)] for i in range(n))
                sm = (n - 1) / 2.0  # max std on support 0..n-1 (bimodal extremes)
                conf = 1.0 - (math.sqrt(var) / sm if sm > 0 else 0.0)
                answers[q.qid] = {
                    "type": "score",
                    "score": round(ev, 2),
                    "confidence": round(max(0.0, conf), 4),
                    "legend": {str(i): d for i, d in enumerate(q.criteria)},
                    "probabilities": {str(i): round(probs[str(i)], 4) for i in range(n)},
                }
            mass = round(in_set_mass, 4) if in_set_mass is not None else None
            engine_diag[q.qid] = {
                "in_set_mass": mass,
                # raw per-option log-mass BEFORE any T/softmax -- Phase 3
                # calibration fits T offline from these (brief §7). No rounding
                # games: 6 decimals is float-safe for sum-of-logprobs.
                "log_mass": {o: round(v, 6) for o, v in cond.items()},
                "n_scored_nodes": len(pq["nodes"]),
                # answers renormalized from a tiny in-set sliver are dominated
                # by measurement noise (observed: noul masses 0.000-0.009 on
                # all sanity cases; escalate 0.59 vs 0.38 across pad A/B)
                "low_evidence": mass is not None and mass < 0.5,
            }
        return answers, engine_diag, zero_logprobs

    def run(self, state, questions: dict, temperature: float | None = None) -> dict:
        return self.run_many([state], questions, temperature)[0]

    def run_many(self, states: list, questions,
                 temperature: float | None = None) -> list[dict]:
        """Score several states in ONE generate call.

        `questions` is either one question set for every state, or a list of
        one set per state. Per-state sets matter because a dataset whose
        question text lives in the row (boolq, pubmedqa) has no two states
        sharing a set, and under a single-set API those states can never batch
        at all -- 771 of 3900 states, 19.8 % of the data but 74.9 % of the
        generate calls.

        Planning stays per state -- each state has its own padded shared prefix
        and its own trie (invariants #2, #3) -- only the GPU call is shared.
        Requests are concatenated in plan order and routed back by count, so
        vLLM returning outputs in input order is the one added assumption; it
        is asserted on the count here and checked against --batch 1 logits
        offline (runs/batch_equiv_4b.log) before any batched number is used.

        run(state) is run_many([state])[0] and keeps every number it had: with
        one state the prefill warm still fires and the timings mean what they
        meant. With several states there is no single shared prefix to warm
        (in-batch block sharing is the point, invariant #4) and no honest
        per-state GPU time, so elapsed_ms/prefill_ms/fields_ms are None and
        plan_ms/batch_ms/batch_size carry what is still true.
        """
        T = temperature if temperature is not None else self.temperature
        T = max(T, 1e-4)
        assert states, "run_many needs at least one state"
        qsets = [questions] * len(states) if isinstance(questions, dict) else list(questions)
        assert len(qsets) == len(states), (
            f"{len(qsets)} question sets for {len(states)} states")
        t0 = time.perf_counter()

        planned = []
        for state, qset in zip(states, qsets):
            tp = time.perf_counter()
            qspecs = [QSpec(qid=qid, **spec) for qid, spec in qset.items()]
            shared_ids, catalog_len = self._shared_ids(state, qspecs)
            built = self._plan(shared_ids, qspecs)
            planned.append({
                "qspecs": qspecs,
                "shared_ids": shared_ids,
                "catalog_len": catalog_len,
                "requests": built["requests"],
                "plan": built["plan"],
                "plan_ms": (time.perf_counter() - tp) * 1000,
            })
        batched = len(planned) > 1

        # optional prefill warm (only if the shared prefix can hit the cache).
        # JEV_SKIP_WARM=1: rely on in-batch block sharing instead (experiment --
        # if the scheduler dedups the shared prefix across requests of the
        # same batched call, the separate warm forward is pure overhead).
        # With several states there is no one prefix to warm: skip it.
        prefill_ms = 0.0
        n_extra_calls = 0
        if (not batched and len(planned[0]["shared_ids"]) >= self._block_size()
                and os.environ.get("JEV_SKIP_WARM") != "1"):
            t1 = time.perf_counter()
            self.llm.generate(
                [self._prompt(planned[0]["shared_ids"])],
                SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                               extra_args={"shared_len": planned[0]["catalog_len"]}),
                use_tqdm=False,
            )
            prefill_ms = (time.perf_counter() - t1) * 1000
            n_extra_calls += 1

        requests = [r for p in planned for r in p["requests"]]

        # JIT warm once per engine lifetime (real logprob request)
        if not self._jit_warmed:
            self.llm.generate(
                [self._prompt(requests[0][0])],
                SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                               logprob_token_ids=requests[0][1][:8],
                               extra_args={"shared_len": planned[0]["catalog_len"]}),
                use_tqdm=False,
            )
            self._jit_warmed = True
            n_extra_calls += 1

        prompts = [self._prompt(ids) for ids, _ in requests]
        # extra_args is vLLM's metadata channel: the stock sampler ignores it, so
        # a backend that can keep the catalogue resident gets the boundary told
        # to it instead of inferring it (AGENTS §4.8)
        cat_lens = [p["catalog_len"] for p in planned for _ in p["requests"]]
        sps = [
            SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                           logprob_token_ids=ids,
                           extra_args={"shared_len": cl})
            for (_, ids), cl in zip(requests, cat_lens)
        ]
        t2 = time.perf_counter()
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        batch_ms = (time.perf_counter() - t2) * 1000
        assert len(outs) == len(requests), (
            f"vLLM returned {len(outs)} outputs for {len(requests)} requests"
        )

        results = []
        cursor = 0
        for p in planned:
            n_nodes = len(p["requests"])
            answers, engine_diag, zero_logprobs = self._decode(
                p["qspecs"], p["plan"], outs[cursor:cursor + n_nodes], T)
            cursor += n_nodes
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({
                "model": MODEL_LABEL,
                "answers": answers,
                "usage": {
                    "input_tokens": len(p["shared_ids"]),
                    "output_tokens": n_nodes,  # decision positions scored (our metering)
                },
                "_engine": {
                    "elapsed_ms": None if batched else round(elapsed, 2),
                    "prefill_ms": None if batched else round(prefill_ms, 2),
                    "fields_ms": None if batched else round(batch_ms, 2),
                    # per-state CPU plan and the shared GPU call: the two times
                    # that stay true under batching (measure them separately, S5)
                    "plan_ms": round(p["plan_ms"], 2),
                    "batch_ms": round(batch_ms, 2),
                    "batch_size": len(planned),
                    "n_requests": n_nodes,
                    "n_generate_calls": 1 + n_extra_calls,
                    "shared_prompt_tokens": len(p["shared_ids"]),
                    "zero_logprob_events": zero_logprobs,
                    "per_question": engine_diag,
                    "note": "confidence=deviation (choice:max-prob, score:1-std/((n-1)/2)); "
                        "probabilities renormalized within option set; trie minimal "
                        "expansion (single-child tails dropped, changelog'd)",
                },
            })
        return results
