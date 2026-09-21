"""CPU-only tests for JevEngine: tokenizer behaviour, trie routing, contract shapes,
and the llama.cpp backend's slot planner. No GPU and no model weights.

    python -m tests.test_cpu

Run it as a module from the repository root. By file path, sys.path[0] becomes
tests/ and `core` stops being importable.

It does need a tokenizer, which is the one thing here that can touch the
network: with no local checkout and no cache, `AutoTokenizer` fetches ~22 MB of
vocabulary from TOKENIZER_ID -- no weights. Set HF_HUB_OFFLINE=1 to forbid that
and use only what is already on disk.

Every result below is tokenizer-specific: these are assertions about where token
boundaries fall. So the tokenizer is fingerprinted and the fingerprint is
checked, and the run prints which tokenizer it loaded. A different one fails
loudly instead of quietly testing a different question.

Checks that need a dataset which this repository does not ship report SKIP and
say why; a skip is counted and printed separately, never as a pass.
"""

import hashlib
import json
import os
import sys


from core.fake_llm import FakeLLM  # prompt-seeded: sees cross-state routing bugs

# Upstream vocabulary for the checkpoint this harness was measured on. Apache-2.0,
# and only the tokenizer files are fetched -- no weights.
TOKENIZER_ID = "Qwen/Qwen3.5-9B"

# Token boundaries are the subject of these tests, so the vocabulary is pinned by
# fingerprint rather than by trust: vocab size plus the ids of probes that every
# assertion below depends on (the noul suffix, digits, colliding option keys).
FINGERPRINT = "4ba4dbcd1fab5671"
FP_PROBES = ['  "q": "', ' true', ' false', '0123456789', 'CWE_327_BROKEN_CRYPTO',
             '{\n  ', 'BLOCK_IMMEDIATELY', '"\n', 'very negative', '  "risk_level": ']


def tokenizer_fingerprint(tok) -> str:
    ids = [tok.encode(p, add_special_tokens=False) for p in FP_PROBES]
    blob = json.dumps({"v": len(tok), "ids": ids}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_tokenizer():
    """Local checkout if there is one, else the upstream vocabulary.

    Returns (tokenizer, source). A clone has no models/ directory, so falling
    back is the normal path, not an error case.
    """
    from transformers import AutoTokenizer

    from core.jev_engine import MODEL_PATH

    src = os.environ.get("JEV_TOKENIZER") or (
        MODEL_PATH if os.path.isdir(MODEL_PATH) else TOKENIZER_ID)
    return AutoTokenizer.from_pretrained(src), src


# --- Cloudflare docs example question sets (contract fixtures) ---

CF_SUPPORT = {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
}

CF_RISK = {
    "risk_level": {"type": "score", "instructions": "How risky does this account activity appear?",
                   "criteria": ["Low risk: activity is consistent with the account history",
                                "Moderate risk: some unusual activity needs monitoring",
                                "High risk: multiple strong indicators of account compromise"]},
    "escalate": {"type": "noul", "instructions": "Should this account be escalated for manual security review?",
                 "criteria": {"true": "The activity warrants immediate human review",
                              "false": "The activity can be handled with normal automated controls"}},
}

CF_COLLISIONS = {
    "cwe": {"type": "choice", "instructions": "Primary CWE classification",
            "criteria": {"CWE_327_BROKEN_CRYPTO": "Broken crypto",
                         "CWE_798_HARDCODED": "Hardcoded creds",
                         "CWE_20_VALIDATION": "Missing input validation"}},
    "action": {"type": "choice", "instructions": "Pipeline action",
               "criteria": {"BLOCK_IMMEDIATELY": "Block now", "BLOCK_LATER": "Block at deploy",
                            "ALLOW": "Pass through"}},
}

CF_ROUTE = {
    "department": {"type": "choice", "instructions": "Which team should handle this support request?",
                   "criteria": {"account": "Login, password, profile, or security issues",
                                "billing": "Charges, invoices, refunds, or subscriptions",
                                "technical": "Product bugs, outages, or integrations",
                                "other": "Requests that do not fit the other departments"}},
}


def main() -> int:
    from core.jev_engine import JevEngine

    tok, tok_src = load_tokenizer()
    failures = []
    skipped = []
    n_ok = 0

    def check(name, cond, detail=""):
        nonlocal n_ok
        status = "OK  " if cond else "FAIL"
        print(f"  [{status}] {name} {detail}")
        if cond:
            n_ok += 1
        else:
            failures.append(name)

    def skip(name, why):
        print(f"  [SKIP] {name} -- {why}")
        skipped.append(name)

    print(f"== 0. tokenizer: {tok_src} ==")
    fp = tokenizer_fingerprint(tok)
    check(f"vocabulary is the one these assertions were written against "
          f"({len(tok)} tokens, fp {fp})", fp == FINGERPRINT,
          "" if fp == FINGERPRINT else
          f"expected {FINGERPRINT}; token boundaries differ, so every result "
          f"below would be about a different vocabulary")

    print("== 1. Biscuit trap #3: noul position (space merges into ' true') ==")
    fake = FakeLLM()
    eng = JevEngine(fake, tok)
    # direct probe: suffix diff must produce single ' true'/' false' tokens
    from core.jev_engine import QSpec
    qs = [QSpec(qid="q", **CF_SUPPORT["is_urgent"])]
    shared = eng._shared_prompt("Help! My payouts have been failing for 3 days.", qs)
    sids = eng._ids(shared)
    suffix_ids = eng._diff(sids, '  "q": ')
    base = sids + suffix_ids
    base, paths = eng._paths_for(base, {"true": "true", "false": "false"})
    t, f = paths["true"], paths["false"]
    check("noul true single token", len(t) == 1, f"={t}")
    check("noul false single token", len(f) == 1, f"={f}")
    check("true/false distinct tokens", t[0] != f[0])

    print("== 2. score digits: 0..9 single tokens, no '10' token ==")
    base_s, paths_s = eng._paths_for(base, {"0": "0", "1": "1", "9": "9"})
    lv = tuple(paths_s[k] for k in ("0", "1", "9"))
    check("digits single-token", all(len(x) == 1 for x in lv), f"={[x for x in lv]}")

    print("== 2b. base shrink keeps the prompt text (regression) ==")
    # An option like '.' encodes as one merged ' "."' token, so _paths_for has
    # to shrink the base past '  "q": "'. The characters it drops must ride on
    # the continuations: encoding only decode(base)+cont asked the model for
    # 'a' straight after the colon (where it wants the quote at p=0.987), and
    # every option scored at ~e^-15 with in_set_mass 0.0.
    base_q = sids + eng._diff(sids, '  "q": "')
    want = eng.tok.decode(base_q)
    for tag, probe, conts in [
            ("no shrink", "a", {"a": 'a"', "b": 'b"'}),
            ("shrink via '.'", "a", {"a": 'a"', "b": 'b"', ".": '."'}),
            ("shrink, windowed", "7", {str(i): f'{i}"' for i in range(30)} | {".": '."'})]:
        nb, pth = eng._paths_for(base_q, conts)
        spelled = eng.tok.decode(nb + pth[probe])
        check(f"path spells prompt+cont [{tag}]", spelled == want + probe + '"',
              f"got {spelled[-14:]!r} want {(want + probe + chr(34))[-14:]!r}")

    print("== 3. minimal trie: scored nodes only (>=2 children), collisions ==")
    qs = [QSpec(qid="cwe", **CF_COLLISIONS["cwe"])]
    shared = eng._shared_prompt("audit finding", qs)
    sids = eng._ids(shared)
    built = eng._plan(sids, qs)
    plan = built["plan"]
    # CWE_* share 'C','WE','_' (single-child chain -> pruned) then split at
    # the digit: exactly ONE scored node carries the whole decision
    check("CWE single scored node", len(built["requests"]) == 1,
          f"scored={len(built['requests'])}")
    paths = plan["cwe"]["paths"]
    check("all CWE paths distinct", len({tuple(p) for p in paths.values()}) == 3)
    check("CWE node has >=2 children", len(plan["cwe"]["nodes"][0][1]) >= 2)

    qs = [QSpec(qid="act", **CF_COLLISIONS["action"])]
    sids2 = eng._ids(eng._shared_prompt("scan result", qs))
    built2 = eng._plan(sids2, qs)
    check("BLOCK_* scored nodes minimal", 1 <= len(built2["requests"]) <= 2,
          f"scored={len(built2['requests'])}")

    print("== 4. full run on Cloudflare fixtures (FakeLLM, shape checks) ==")
    fixtures = [
        ("support", "Help! My payouts have been failing for 3 days.", CF_SUPPORT),
        ("risk", {"account_age_days": 12, "recent_events": ["Five failed login attempts"]}, CF_RISK),
        ("route", "I cannot log in after changing my password.", CF_ROUTE),
    ]
    for name, state, qs in fixtures:
        fake = FakeLLM()
        eng = JevEngine(fake, tok)
        r = eng.run(state, qs)
        ans = r["answers"]
        check(f"{name}: model field", r["model"].startswith("jev-rlcd"))
        check(f"{name}: all questions answered", set(ans) == set(qs))
        for qid, spec in qs.items():
            a = ans[qid]
            if spec["type"] == "noul":
                check(f"{name}/{qid}: noul shape", set(a) == {"type", "noul"} and 0 <= a["noul"] <= 1)
            elif spec["type"] == "choice":
                check(f"{name}/{qid}: choice shape",
                      set(a) == {"type", "choice", "confidence", "probabilities"}
                      and a["choice"] in spec["criteria"]
                      and abs(sum(a["probabilities"].values()) - 1) < 0.01)
            else:
                n = len(spec["criteria"])
                check(f"{name}/{qid}: score shape",
                      set(a) == {"type", "score", "confidence", "legend", "probabilities"}
                      and 0 <= a["score"] <= n - 1
                      and len(a["legend"]) == n
                      and abs(sum(a["probabilities"].values()) - 1) < 0.01)
        check(f"{name}: usage present", "input_tokens" in r["usage"] and "output_tokens" in r["usage"])

    print("== 4b. run_many: one generate for N states, same numbers as run ==")
    states = [
        "Help! My payouts have been failing for 3 days.",
        "Hi, could someone explain the invoice line for seat licences?",
    ]
    singles = []
    for s in states:
        fake_s = FakeLLM()
        singles.append(JevEngine(fake_s, tok).run(s, CF_SUPPORT))

    fake_b = FakeLLM()
    many = JevEngine(fake_b, tok).run_many(states, CF_SUPPORT)

    check("run_many returns one result per state", len(many) == len(states))
    # the contract: batching may move a logprob to another call, never change it
    for i, (a, b) in enumerate(zip(singles, many)):
        check(f"run_many[{i}]: answers identical to run", a["answers"] == b["answers"])
        check(f"run_many[{i}]: log_mass/in_set_mass identical",
              a["_engine"]["per_question"] == b["_engine"]["per_question"])
        check(f"run_many[{i}]: usage identical", a["usage"] == b["usage"])
    # ...and that it actually batched: 1 JIT warm + 1 scored call, no per-state warm
    check("run_many uses a single scored generate call", fake_b.calls == 2)
    check("run_many batch carries every state's requests",
          fake_b.batch_sizes[-1] == sum(r["_engine"]["n_requests"] for r in many))
    check("run_many marks per-state GPU time as unknown",
          all(r["_engine"]["fields_ms"] is None
              and r["_engine"]["batch_size"] == len(states)
              and r["_engine"]["plan_ms"] is not None for r in many))
    # single state keeps the old contract, numbers included
    check("run(state) still reports its own timings",
          all(singles[0]["_engine"][k] is not None
              for k in ("elapsed_ms", "prefill_ms", "fields_ms"))
          and singles[0]["_engine"]["batch_size"] == 1)

    print("== 4c. score levels: scale_labels reach the catalog ==")
    import os.path
    if not os.path.exists("data/calib/h3/sst5_desc.jsonl"):
        skip("score levels reach the catalog",
             "data/calib/h3/sst5_desc.jsonl is not shipped; rebuild it with "
             "data/build/phase3_build_h3h4.py to run this check")
    else:
        import run_collect as RC
        for name, path, want in (
                ("sst5_desc", "data/calib/h3/sst5_desc.jsonl", "0: very negative"),
                ("amazon", "data/calib/h3/amazon_reviews_en.jsonl", "0: 1")):
            rows = RC.load_dataset(path)
            g = RC.group_states(rows)[0]
            qs = RC.build_questions(g, RC.load_catalog_desc(path))
            specs = [QSpec(qid=k, **v) for k, v in qs.items()]
            text = JevEngine(FakeLLM(), tok)._shared_prompt(g[0]["state"], specs)
            check(f"{name}: catalog contains {want!r}", want in text,
                  text[text.find("QUESTIONS:"):][:200])
        # the whole point of the flag: a labelled run must not render like an
        # unlabelled one, or nobody can tell the two collects apart afterwards
        rows = RC.load_dataset("data/calib/h3/sst5_desc.jsonl")
        g = RC.group_states(rows)[0]
        crit = RC.build_questions(g, {})[g[0]["question_key"]]["criteria"]
        sc = g[0]["scale"]
        check("sst5_desc levels are not digits again",
              crit != [str(v) for v in range(sc["min"], sc["max"] + 1)], str(crit))

    print("== 4d. batching: every state keeps its own question set ==")
    import run_collect as RC2

    def _row(rid, opts, state, qkey="q"):
        return {"id": rid, "state": state, "qtype": "choice", "question_key": qkey,
                "options": opts, "question_desc": "pick one", "_ds": "t"}

    # a mixed chunk is allowed now -- what must hold is that the question set
    # travelling with each state is that state's own. The two option orders
    # below are the case that shipped broken: they were merged and all scored
    # with the first one's catalogue.
    mixed = [[_row("p0", ["alpha", "beta", "gamma"], "s0")],
             [_row("p1", ["gamma", "alpha", "beta"], "s1")],
             [_row("a", ["alpha", "beta"], "s2")],
             [_row("b", ["alpha", "beta"], "s3")]]
    paired = True
    for chunk, qs_list in RC2.batch_chunks(mixed, {}, 32):
        paired &= len(chunk) == len(qs_list)
        for g, qs in zip(chunk, qs_list):
            paired &= qs == RC2.build_questions(g, {})
    check("each state in a chunk carries its own question set", paired)
    check("states with identical prompts still share a chunk",
          any(len(c) > 1 for c, _ in RC2.batch_chunks(mixed, {}, 32)))

    # engine level: per-state question sets, and the mutation that swaps them
    QA = {"qa": {"type": "choice", "instructions": "A?",
                 "criteria": {"alpha": "a", "beta": "b"}}}
    QB = {"qb": {"type": "noul", "instructions": "B?",
                 "criteria": {"true": "yes", "false": "no"}}}
    st = ["state one", "state two"]
    solo = [JevEngine(FakeLLM(), tok).run(st[0], QA),
            JevEngine(FakeLLM(), tok).run(st[1], QB)]
    both = JevEngine(FakeLLM(), tok).run_many(st, [QA, QB])
    check("run_many answers each state with its own question set",
          [set(r["answers"]) for r in both] == [{"qa"}, {"qb"}]
          and all(a["answers"] == b["answers"] for a, b in zip(solo, both)))
    swapped = JevEngine(FakeLLM(), tok).run_many(st, [QB, QA])
    check("swapping the sets changes the answers (the test has teeth)",
          [set(r["answers"]) for r in swapped] == [{"qb"}, {"qa"}]
          and swapped[0]["_engine"]["per_question"] != both[0]["_engine"]["per_question"])

    print("== 5. assertions fire correctly ==")
    try:
        QSpec(qid="x", qtype="score", instructions="i", criteria=list(range(11)))
        check("score >10 levels rejected", False)
    except AssertionError:
        check("score >10 levels rejected", True)
    try:
        QSpec(qid="x", qtype="choice", instructions="i", criteria={"a": "d"})
        check("choice <2 options rejected", False)
    except AssertionError:
        check("choice <2 options rejected", True)

    # the padding size must be declared, never guessed: the silent 528 was a
    # vLLM constant that a backend without any block structure inherited, and it
    # padded a 121-token prompt to 1056
    class _NoBlockLLM(FakeLLM):
        block_size = None

    check("engine uses the block size a backend declares",
          JevEngine(FakeLLM(), tok)._block_size() == 528)
    try:
        JevEngine(_NoBlockLLM(), tok)._block_size()
        check("a backend that declares no block size is refused, not defaulted", False)
    except RuntimeError as ex:
        check("a backend that declares no block size is refused, not defaulted",
              "block_size" in str(ex))

    try:
        check("window==oracle (coverage printed above)", test_window_oracle())
    except Exception as ex:
        check(f"window==oracle: {ex}", False)

    print("== backend contract: BonsaiLLM against a stub worker (no model) ==")
    _ok, _detail = test_bonsai_contract()
    check("BonsaiLLM exposes what its callers read", _ok, _detail)

    print("== 5b. catalogue boundary handed to the backend ==")

    def _lcp_tokens(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _cat(state, qs, pad):
        eng = JevEngine(FakeLLM(), tok, pad_shared=pad)
        specs = [QSpec(qid=q, **spec) for q, spec in qs.items()]
        return eng._shared_ids(state, specs)

    Q1 = dict(CF_SUPPORT)
    # the first field is identical in both states on purpose (see the last check)
    sA, cA = _cat({"plan": "business", "message": "Open the gate."}, Q1, True)
    sB, cB = _cat({"plan": "business", "message": "The invoice is wrong again."},
                  Q1, True)
    check("catalogue length is the same for two states of one question set",
          cA == cB, f"{cA} vs {cB}")
    check("and those leading tokens are identical",
          sA[:cA] == sB[:cB])
    check("catalogue stops before the state, not after",
          cA < min(len(sA), len(sB)), f"cat {cA}, prompt {len(sA)}")
    # with padding on, the catalogue is aligned to whole cache blocks -- that is
    # what the padding is for, and it is checkable
    eng_bs = JevEngine(FakeLLM(), tok, pad_shared=True)._block_size()
    check("PAD=1 catalogue ends on a cache-block boundary",
          cA % eng_bs == 0, f"{cA} % {eng_bs} = {cA % eng_bs}")

    s0, c0 = _cat({"plan": "business", "message": "Open the gate."}, Q1, False)
    check("PAD=0 still reports a boundary before the state",
          0 < c0 < len(s0), f"cat {c0}, prompt {len(s0)}")
    check("PAD=0 catalogue is shorter than the padded one (the pad is the difference)",
          c0 < cA, f"{c0} vs {cA}")

    # The engine's boundary is the catalogue, which is NOT the same thing as the
    # longest prefix two calls happen to share: two states whose leading fields
    # are equal share more than the catalogue. Both numbers are real and they
    # are not interchangeable.
    shared_more = _lcp_tokens(sA, sB)
    check("two calls can share more than the catalogue (state prefix is constant)",
          shared_more >= cA, f"shared {shared_more}, catalogue {cA}")

    print("== 6. Bonsai backend: catalogue slots ==")
    from core.bonsai_llm import SLOTS, plan_slot, slot_key

    def replay(calls, k=SLOTS):
        """calls: (prefix_ids, shared_len). Mirrors BonsaiLLM._plan_catalog."""
        slots, keys, order, out = [[] for _ in range(k)], [None] * k, [], []
        for lcp, sl in calls:
            key = slot_key(lcp, sl)
            idx, install = plan_slot(slots, keys, order, lcp, sl, key)
            if install is not None:
                slots[idx], keys[idx] = list(install), key
            if idx in order:
                order.remove(idx)
            order.insert(0, idx)
            ok = lcp[:len(slots[idx])] == slots[idx]
            out.append({"slot": idx, "installed": install is not None,
                        "reused": len(slots[idx]), "prefix_ok": ok})
        return out

    CAT_A = list(range(1000, 1528))          # a catalogue
    CAT_B = [1000, 1001] + list(range(9000, 9500))
    FIXED = list(range(3000, 3391))          # the constant head of a game state
    def state(i):
        return [7000 + i, 7100 + i, 7200 + i]

    # the engine's number is exact, so the very first call already reuses it
    one = replay([(CAT_A + FIXED + state(i), len(CAT_A)) for i in range(6)])
    check("first call of a question set already reuses the whole catalogue",
          one[0]["reused"] == len(CAT_A), str(one[0]))
    check("one load per question set, ever",
          sum(r["installed"] for r in one) == 1,
          f"{sum(r['installed'] for r in one)} installs in {len(one)} calls")

    # THE POINT of a fixed cut: the resident length cannot depend on call order,
    # so neither can the answers. The earlier version grew past the catalogue and
    # both did.
    mixed = [(CAT_A + FIXED + state(0), len(CAT_A)),
             (CAT_A + FIXED + state(1), len(CAT_A)),
             (CAT_A + list(range(4000, 4100)) + state(2), len(CAT_A))]  # head changed
    mx = replay(mixed)
    check("a changed state head changes nothing: the cut is fixed",
          [r["reused"] for r in mx] == [len(CAT_A)] * 3, str([r["reused"] for r in mx]))
    check("call order cannot change what is resident",
          [r["reused"] for r in replay(list(reversed(mixed)))] == [len(CAT_A)] * 3)
    check("the slot is a prefix of the call every single time",
          all(r["prefix_ok"] for r in mx + one))

    # alternating question sets: the reason there is more than one slot
    alt = []
    for i in range(8):
        cat = CAT_A if i % 2 == 0 else CAT_B
        alt.append((cat + FIXED + state(i), len(cat)))
    al = replay(alt)
    check("two catalogues alternating stay resident",
          not any(r["installed"] for r in al[4:]),
          f"installs at {[i for i, r in enumerate(al) if r['installed']]}")
    check("a single slot would have thrashed",
          any(r["installed"] for r in replay(alt, k=1)[4:]))

    # LRU: a fifth catalogue takes the least recently used slot
    cats = [[500 + j] + list(range(2000 + 1000 * j, 2300 + 1000 * j))
            for j in range(SLOTS + 1)]
    warm = [(c + state(i), len(c)) for c in cats[:SLOTS] for i in (0, 1)]
    ev = replay(warm + [(cats[SLOTS] + state(0), len(cats[SLOTS]))]
                + [(cats[1] + state(5), len(cats[1]))]
                + [(cats[0] + state(5), len(cats[0]))])
    check("a new catalogue evicts the least recently used one",
          not ev[-2]["installed"], f"cats[1] reloaded: {ev[-2]}")
    check("the evicted one must be reloaded",
          ev[-1]["installed"], f"cats[0] not reloaded: {ev[-1]}")

    print(f"\n  ran {n_ok + len(failures)} checks: {n_ok} ok, "
          f"{len(failures)} failed, {len(skipped)} skipped")
    if skipped:
        print("  skipped: " + ", ".join(skipped))
    print(f"{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 0 if not failures else 1


def test_bonsai_contract() -> tuple[bool, str]:
    """Build the real BonsaiLLM against a stub worker and read what callers read.

    A `--fake` / FakeLLM pre-flight takes the OTHER branch, so it cannot see a
    name that does not exist on this one: `llm.gguf` was read in a lane-J script,
    passed every CPU check, and raised AttributeError only after the model was
    on the GPU. The stub speaks the worker's READY line and nothing else, so this
    costs no model load and still executes the real constructor.
    """
    import os
    import subprocess
    import tempfile
    import textwrap

    import core.bonsai_llm as bonsai

    stub = textwrap.dedent("""\
        #!/usr/bin/env python3
        import sys
        # READY n_vocab n_ctx n_seq_max ctx_per_seq n_batch n_slots n_ubatch
        argv = sys.argv[1:]
        n_ctx, n_seq = int(argv[1]), int(argv[2])
        print(f"READY 248077 {n_ctx} {n_seq} {n_ctx // n_seq} "
              f"{argv[4]} {argv[5]} {argv[6]}", flush=True)
        for _ in sys.stdin:
            pass
    """)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "stub_worker.py")
        with open(path, "w") as f:
            f.write(stub)
        os.chmod(path, 0o755)
        model = os.path.join(d, "not-a-real.gguf")
        open(model, "wb").close()
        llm = None
        try:
            llm = bonsai.BonsaiLLM(model, worker=path, ctx_per_seq=2048,
                                   n_seq_max=14, log_path=os.path.join(d, "w.log"),
                                   calls_path=os.path.join(d, "calls.jsonl"))
            # exactly what jev_engine and the run scripts read off a backend
            reads = ["block_size", "ctx_per_seq", "n_seq_max", "n_slots",
                     "n_vocab", "n_ctx", "generate"]
            missing = [r for r in reads if not hasattr(llm, r)]
            if missing:
                return False, f"missing: {', '.join(missing)}"
            if llm.ctx_per_seq != 2048 or llm.n_seq_max != 14:
                return False, f"handshake lost: {llm.ctx_per_seq}/{llm.n_seq_max}"
            return True, f"({len(reads)} names, ctx/seq {llm.ctx_per_seq}, "\
                         f"n_seq_max {llm.n_seq_max}, no model loaded)"
        except (subprocess.SubprocessError, OSError, ValueError,
                bonsai.BonsaiWorkerError) as e:
            return False, f"{type(e).__name__}: {e}"
        finally:
            if llm is not None and getattr(llm, "proc", None):
                llm.proc.kill()


def test_window_oracle() -> bool:
    """Oracle test: window-encoded paths must equal FULL-encode paths for EVERY
    option, across every shipped bench preset plus the sanity question sets.
    Slow is fine here -- this is a correctness oracle, not a runtime measure.
    It prints the number of options and cases it actually covered."""
    from core.jev_engine import JevEngine, QSpec

    tok, _ = load_tokenizer()

    class _NoLLM:
        # the oracle only builds prompts, but padding needs a block size and the
        # engine no longer invents one
        block_size = 528

        class llm_engine:
            vllm_config = None

    eng = JevEngine(_NoLLM(), tok)  # type: ignore[arg-type]

    def full_paths(base_ids, conts):
        base = list(base_ids)
        for _ in range(4):
            text = tok.decode(base)
            paths, ok = {}, True
            for o, c in conts.items():
                full = tok(text + c, add_special_tokens=False)["input_ids"]
                if full[: len(base)] != base:
                    ok = False
                    break
                paths[o] = full[len(base):]
            if ok and paths:
                return base, paths
            if len(base) <= 1:
                break
            base = base[:-1]
        raise AssertionError("full oracle cannot align")

    cases = {}
    from presets import PRESETS
    from run_bench_ar import map_schema_to_questions
    for name, (context, schema) in PRESETS.items():
        cases[f"preset:{name}"] = (context, map_schema_to_questions(schema))
    for extra_name, extra_qs in (("cf_support", CF_SUPPORT),
                                 ("cf_collisions", CF_COLLISIONS),
                                 ("cf_risk", CF_RISK)):
        cases[f"sanity:{extra_name}"] = (
            "Customer reports failing payouts for 3 days.", extra_qs)

    n_opts = 0
    over_budget = []
    for label, (state, qs) in cases.items():
        qspecs = [QSpec(qid=k, **v) for k, v in qs.items()]
        shared = eng._shared_prompt(state, qspecs)
        st = state if isinstance(state, str) else json.dumps(state)
        cut = shared.rindex(st)
        _, t1 = eng._pad_align(shared[:cut], "")
        ids, _ = eng._pad_align(t1 + shared[cut:][:-2], "{\n")
        try:
            built = eng._plan(ids, qspecs)
        except AssertionError as ex:
            # Option sets too wide for the scored-node budget are a known v1
            # limit, not an encoding fault. Name them so the coverage line
            # below cannot read as if they had passed.
            if "budget" not in str(ex):
                raise
            over_budget.append(f"{label} ({str(ex).split(';')[0]})")
            continue
        for q in qspecs:
            if q.qtype in ("choice", "noul"):
                suffix = f'  "{q.qid}": "'
                conts = {o: o + '"' for o in q.options}
            else:
                suffix = f'  "{q.qid}": '
                conts = {o: o for o in q.options}
            sids = eng._diff(ids, suffix)
            _, p_full = full_paths(ids + sids, conts)
            p_win = built["plan"][q.qid]["paths"]
            assert p_full == p_win, f"{label}/{q.qid}: window != oracle"
            n_opts += len(conts)
    covered = len(cases) - len(over_budget)
    print(f"  [OK  ] window==oracle for {n_opts} options across {covered} cases")
    if over_budget:
        print(f"         not covered, over the scored-node budget: "
              f"{', '.join(over_budget)}")
    return True


if __name__ == "__main__":
    sys.exit(main())
