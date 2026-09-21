"""llama.cpp gate — answers the three questions in reports/llamacpp_recon.md §f
before anyone writes a backend.

Does NOT import core/ (Dax is editing it). The prompt is a faithful REPLICA of
`JevEngine._shared_prompt`: same preamble, catalog FIRST / state LAST, same
"(true/false)" and "(choice)" labels, same trailing "{\\n". It is a replica, not
the engine's own output — so token counts are comparable, not provably identical.

Steps (stops at the first failure):
  0  build the fintech_fraud prompt, report its token count
  1  prefix reuse: same catalog, two different states. If llama.cpp re-processes
     the whole prompt on state B, the engine's 809ms->42ms trick is gone and the
     branch is closed (upstream #20643, #22384: hybrid/recurrent memory).
  2  prefill throughput on the 3090, KV f16/q8_0 only (4-bit KV collapses
     prefill 20x -- upstream #27109).
  3  candidate logprobs at the scored position via n_probs, and whether the
     option set is even reachable in top-N.

Usage (server started separately, one GPU process at a time):
  llama-server -m models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf \
      -ngl 99 -c 8192 -np 1 -fa on -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8080
  .venv/bin/python backends/llamacpp/gate.py --url http://127.0.0.1:8080
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import presets  # noqa: E402  (root module, not core/)

TOKENIZER_DIR = "models/Qwen3.5-4B-AWQ-4bit"
PREFILL_FLOOR = 1500.0   # t/s; below this llama.cpp cannot win on latency
STATE_B = (
    "Transaction Review: Card-present chip purchase of $32.40 at 'Corner Grocery', "
    "Portland, 6:02 PM local. Cardholder: account age 6 years, prior spend pattern "
    "groceries/utilities under $200, geo-location matches home ZIP. Device fingerprint "
    "known. 3D-Secure not required. Velocity: 1 transaction in 24 hours. BIN issued in US."
)


# ---------------------------------------------------------------- prompt ----
def catalog_and_prompt(state_text):
    """Replica of _shared_prompt for the fintech_fraud preset."""
    lines = []
    for qid, f in presets.FRAUD_SCHEMA.items():
        if f["type"] == "boolean":
            lines.append(
                f"- {qid} (true/false): {f['description']}\n"
                f"    true: yes\n    false: no (answer \"true\" or \"false\")")
        else:
            pairs = "\n".join(f"    {c}: {c}" for c in f["choices"])
            lines.append(f"- {qid} (choice): {f['description']}\n{pairs}")
    catalog = "\n".join(lines)
    content = (
        "You are a calibrated decision engine. Answer every question with exactly "
        "one allowed value based on the STATE, as a JSON object mapping each "
        "question id to its answer.\n\n"
        f"QUESTIONS:\n{catalog}\n\nSTATE:\n{state_text}")
    return catalog, [{"role": "user", "content": content}]


def build(tok, state_text, scored_prefix=""):
    _, msgs = catalog_and_prompt(state_text)
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + "{\n" + scored_prefix


# ------------------------------------------------------------------ http ----
def post(url, path, payload, timeout=600):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def completion(url, prompt, *, cache, n_probs=0, n_predict=1):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0,
            "cache_prompt": cache, "return_tokens": False}
    if n_probs:
        body["n_probs"] = n_probs
    return post(url, "/completion", body)


def timings(resp):
    t = resp.get("timings") or {}
    return t.get("prompt_n"), t.get("prompt_ms"), t.get("prompt_per_second")


# ----------------------------------------------------------------- steps ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tokenizer", default=TOKENIZER_DIR)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # ---- step 0 -----------------------------------------------------------
    print("=" * 66)
    print("STEP 0  prompt (replica of _shared_prompt, preset fintech_fraud)")
    pa = build(tok, presets.FRAUD_CONTEXT)
    pb = build(tok, STATE_B)
    ia, ib = tok(pa).input_ids, tok(pb).input_ids
    shared = 0
    for x, y in zip(ia, ib):
        if x != y:
            break
        shared += 1
    print(f"  state A : {len(ia)} tokens")
    print(f"  state B : {len(ib)} tokens")
    print(f"  shared prefix (HF tokenizer): {shared} tokens "
          f"({100*shared/len(ia):.1f}% of A)")
    if shared < 100:
        print("  FAIL: the two prompts barely share a prefix — replica is wrong")
        return 1

    try:
        props = json.loads(urllib.request.urlopen(args.url + "/props", timeout=30).read())
        print(f"  server model: {props.get('model_path', '?')}")
    except urllib.error.URLError as e:
        print(f"  FAIL: no llama-server at {args.url} ({e})")
        return 1

    # ---- step 1 -----------------------------------------------------------
    print("\nSTEP 1  prefix reuse across two states (the decisive one)")
    completion(args.url, pa, cache=True)                      # warm the catalog
    ra = completion(args.url, pa, cache=True)                 # A again: full hit
    rb = completion(args.url, pb, cache=True)                 # B: shares catalog
    na, _, _ = timings(ra)
    nb, _, _ = timings(rb)
    print(f"  A repeated : prompt_n = {na}  (cache hit => small)")
    print(f"  B after A  : prompt_n = {nb}  (of {len(ib)} tokens)")
    if nb is None:
        print("  FAIL: server did not report timings")
        return 1
    if nb >= 0.9 * len(ib):
        print(f"  FAIL: B re-processed {nb}/{len(ib)} tokens — no prefix reuse.")
        print("        This is upstream #20643 / #22384 (hybrid/recurrent memory).")
        print("        The engine's cross-call cache (809ms -> 42ms) does not survive. STOP.")
        return 1
    print(f"  PASS: reused ~{len(ib)-nb} tokens of the shared catalog")

    # ---- step 2 -----------------------------------------------------------
    # Biscuit: measure both. (a) the preset prompt, absolute ms, recorded only --
    # it is prefill alone, NOT comparable to run_bench_prod's 440 ms end-to-end
    # (4 questions through the trie). (b) a ~2K prompt, t/s against the 0.25
    # ms/token floor -- the stop threshold lives here.
    print("\nSTEP 2  prefill (KV q8_0; never 4-bit KV — #27109)")
    cold = completion(args.url, pb + "\n#cold\n", cache=False)
    n, ms, tps = timings(cold)
    if not tps and ms:
        tps = 1000.0 * n / ms
    print(f"  (a) preset  : {n} tok, {ms:.1f} ms, {tps:.1f} t/s   [record only]")
    print(f"      vLLM same preset, same session: 440 ms end-to-end for 4 questions")
    print(f"      -- llama.cpp number above is PREFILL ONLY, not the same quantity")

    big = STATE_B
    while len(tok(build(tok, big)).input_ids) < 2000:
        big += " " + STATE_B
    pbig = build(tok, big)
    nbig = len(tok(pbig).input_ids)
    cold2 = completion(args.url, pbig + "\n#cold2\n", cache=False)
    n2, ms2, tps2 = timings(cold2)
    if not tps2 and ms2:
        tps2 = 1000.0 * n2 / ms2
    print(f"  (b) ~2K     : {n2} tok ({nbig} by HF), {ms2:.1f} ms, {tps2:.1f} t/s")
    print(f"      floor (AGENTS §5, vLLM 0.25 ms/tok at 2K): ~4000 t/s")
    if tps2 < PREFILL_FLOOR:
        print(f"  FAIL: {tps2:.1f} < {PREFILL_FLOOR} t/s — cannot win on latency.")
        print("        Only remaining argument would be Q8 precision, which is a")
        print("        different goal and belongs to the user. STOP.")
        return 1
    print("  PASS")

    # ---- step 3 -----------------------------------------------------------
    print("\nSTEP 3  candidate logprobs at the scored position")
    opts = presets.FRAUD_SCHEMA["risk_level"]["choices"]
    scored = build(tok, presets.FRAUD_CONTEXT, '  "risk_level": "')
    firsts = {o: tok(o, add_special_tokens=False).input_ids[0] for o in opts}
    got = {}
    for n_probs in (20, 100, 500):
        r = completion(args.url, scored, cache=True, n_probs=n_probs)
        probs = (r.get("completion_probabilities") or [{}])[0]
        top = probs.get("top_logprobs") or probs.get("top_probs") or []
        table = {}
        for e in top:
            p = e.get("prob")
            if p is None and e.get("logprob") is not None:
                p = 2.718281828 ** e["logprob"]
            table[e.get("id")] = (e.get("token"), p)
        got = {o: table.get(t) for o, t in firsts.items()}
        hit = sum(1 for v in got.values() if v)
        mass = sum(v[1] for v in got.values() if v)
        print(f"  n_probs={n_probs:>3}: returned {len(top):>3} entries, "
              f"{hit}/{len(opts)} options found, in-set mass = {mass:.4f}")
        if hit == len(opts):
            if mass < 0.5:
                print(f"  WARN: in-set mass {mass:.4f} < 0.5 — that is low_evidence "
                      f"territory; compare against the vLLM run before trusting it")
            print("  PASS: all candidates reachable via n_probs")
            print("\nGATE PASSED — all three steps. Report the numbers to Biscuit; "
                  "do not write a backend before the user decides.")
            return 0
    print(f"  FAIL: only {sum(1 for v in got.values() if v)}/{len(opts)} options in "
          f"top-N even at n_probs=500.")
    print("        n_probs returns the top of the distribution, not a chosen set —")
    print("        a 150-option catalog would lose its tail. The honest path is")
    print("        llama-cpp-python llama_get_logits_ith (full f32 vector), which")
    print("        needs its vendored llama.cpp to support qwen3_5. STOP.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
