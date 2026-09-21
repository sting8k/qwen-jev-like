"""Phase-3 calibration collector (brief §7, Biscuit spec 2026-09-18).

Reads data/calib/<huong>/<dataset>.jsonl (brief §1 rows), runs the production
engine (1 generate call per state, catalog-first layout), and appends ONE line
per input row to runs/calib/<dataset>.logits.jsonl:

  {id, dataset, qtype, question_key, options|scale, label,
   logits: {option: raw log-mass},      # AFTER trie, BEFORE softmax/T
   in_set_mass, low_evidence, ms_plan, ms_gpu, split_hint, state_tokens}

plus one content-free prior row per dataset (state "N/A", id "<dataset>::prior",
label null). Resume-safe: skips ids already in the output file.

Modes:
  --dry      P0: FakeLLM over EVERY row, no GPU, no output write. Asserts
             brief §4.1 contract (label in options/scale, options in catalog
             via engine invariant #1, prompt < 8K tokens, no dup ids).
  --probe N  P1: first N state-groups per dataset on real GPU + invariants
             (in_set_mass >= 0.99, low_evidence = 0, ms_gpu window).
  default    full collect (~2100 states, ~16 min warm), dataset order stable
             (no shuffle -- catalog prefix cache across calls).

Usage:
  .venv/bin/python run_calib_collect.py --dry          # P0, CPU only
  .venv/bin/python run_calib_collect.py --probe 5      # P1 GPU smoke
  .venv/bin/python run_calib_collect.py                # full
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict

# Raw logits are model-specific and this collector APPENDS + skips ids it has
# already seen. Sharing one directory between checkpoints would silently mix
# two models' logits in one file and then skip almost every row of the second
# run. The 9B keeps the original path so Phase-3 data stays where it is.
def _pad_shared() -> bool:
    from core.jev_engine import pad_shared_default
    return pad_shared_default()


def _out_dir() -> str:
    from core.jev_engine import MODEL_LABEL
    # JEV_OUT_DIR sends a run somewhere else on purpose: the default path is
    # derived from the model, so a timing or ablation re-run lands on top of
    # the published logits for that model and the resume logic then makes it
    # look like the run did nothing (backlog P1: default paths, no warning).
    override = os.environ.get("JEV_OUT_DIR", "").strip()
    if override:
        return override
    base = ("runs/calib" if MODEL_LABEL == "jev-rlcd-qwen3.5-9b-awq"
            else "runs/calib_" + MODEL_LABEL.replace("jev-rlcd-", ""))
    return base if _pad_shared() else base + "_pad0"


OUT_DIR = _out_dir()
PROMPT_LIMIT = 8000


# ---------------------------------------------------------------- data ----
def load_dataset(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("state"), str):
                r["state"] = r["state"].strip() or "(no content)"
            r["_ds"] = os.path.basename(path)[:-6]  # strip .jsonl
            r["_ln"] = ln
            rows.append(r)
    return rows


def check_row(r):
    """brief §1 contract; returns error string or None."""
    if r["qtype"] not in ("noul", "choice", "score"):
        return f"bad qtype {r['qtype']!r}"
    if r["qtype"] == "choice":
        opts = r.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            return "choice needs options list >= 2"
        if len(set(opts)) != len(opts):
            return "duplicate options"
        bad = [o for o in opts if '"' in str(o) or "\n" in str(o) or "\r" in str(o)]
        if bad:
            return f"option breaks JSON quoting: {bad[:1]}"
        if str(r["label"]) not in map(str, opts):
            return f"label {r['label']!r} not in options"
    elif r["qtype"] == "score":
        sc = r.get("scale") or {}
        if "min" not in sc or "max" not in sc:
            return "score needs scale {min,max}"
        n = sc["max"] - sc["min"] + 1
        if not (2 <= n <= 10):
            return f"scale has {n} levels; engine supports 2..10"
        try:
            ok = sc["min"] <= int(r["label"]) <= sc["max"]
        except (TypeError, ValueError):
            return f"score label {r['label']!r} not an int"
        if not ok:
            return f"score label {r['label']!r} outside scale"
    else:  # noul
        if str(r["label"]).lower() not in ("true", "false"):
            return f"noul label {r['label']!r} not true/false"
    if not r.get("question_key"):
        return "missing question_key"
    if r.get("state_tokens", 0) > PROMPT_LIMIT:
        return f"state_tokens {r['state_tokens']} > {PROMPT_LIMIT}"
    return None


def norm_label(r):
    lb = r["label"]
    if r["qtype"] == "noul":
        return "true" if str(lb).lower() == "true" else "false"
    return str(lb)


def load_catalog_desc(path):
    """Sibling <dataset>_catalog.json (JSONL rows {intent, desc}).
    Desc is Biscuit-approved deterministic prompt material; intent strings
    stay verbatim (engine invariant #1 unaffected)."""
    cat = os.path.join(os.path.dirname(path),
                       os.path.basename(path)[:-6] + "_catalog.json")
    m = {}
    if not os.path.exists(cat):
        return m
    for line in open(cat):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if isinstance(d, dict) and d.get("intent"):
            m[str(d["intent"])] = str(d.get("desc") or d["intent"])
    return m


def naturalize(opt: str) -> str:
    """Natural-case desc for an option: split snake_case, title-case words.
    'ANGRY' -> 'Angry', 'P0_CRITICAL' -> 'P0 Critical', 'top_up' -> 'Top Up'.
    Applied only when no catalog desc exists (Biscuit 2026-09-18)."""
    words = opt.replace("-", "_").split("_")
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


def group_states(rows):
    """Rows sharing the exact same state = one production call. Order =
    first appearance (no shuffle -- catalog cache). H6 probe shapes
    (content-free 50 questions on one "N/A" state; permutation 4 orders of
    one question) repeat a question_key inside a state -> engine qids must
    be unique per call, so split those states into singleton groups."""
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(r["state"], []).append(r)
    out = []
    for rs in groups.values():
        qids = [r["question_key"] for r in rs]
        if len(set(qids)) == len(qids):
            out.append(rs)
        else:
            out.extend([r] for r in rs)
    return out


def build_questions(group, desc_map=None):
    desc_map = desc_map or {}
    qs = {}
    for r in group:
        if r["qtype"] == "noul":
            crit = {"true": "yes", "false": "no"}
        elif r["qtype"] == "choice":
            crit = {str(o): desc_map.get(str(o)) or naturalize(str(o))
                    for o in r["options"]}
        else:
            sc = r["scale"]
            n = sc["max"] - sc["min"] + 1
            labels = r.get("scale_labels")
            if labels:
                # Without labels the catalog renders "0: 1 / 1: 2 / ..." -- the
                # level description IS the digit, so the model reads a 1..5
                # scale while the engine scores 0..4 (backlog P1). scale_labels
                # gives each level something to match against.
                assert len(labels) == n, (
                    f"{r['id']}: {len(labels)} scale_labels for {n} levels")
                crit = [str(x) for x in labels]
                assert crit != [str(v) for v in range(sc["min"], sc["max"] + 1)], (
                    f"{r['id']}: scale_labels are just the digits again -- a run "
                    f"collected like this is indistinguishable from one without "
                    f"labels, and someone will read it as 'labels do not help'")
            else:
                crit = [str(v) for v in range(sc["min"], sc["max"] + 1)]
        qid = r["question_key"]
        assert qid not in qs, f"duplicate question_key {qid} in one state"
        instr = r.get("question_desc", "")
        if r["qtype"] == "choice":
            # format-adherence hint (noul's equivalent took in_set_mass
            # 0.001 -> 0.999); 77-option catalogs leak ~1.5% without it
            instr += " (answer with one option name exactly as listed)"
        qs[qid] = {"type": r["qtype"], "instructions": instr,
                   "criteria": crit}
    return qs


def filter_split(groups, want):
    """Keep only the groups in one split, using run_fit's own hash.

    The split is a property of the data, not of a run, so it is imported rather
    than re-derived: a second copy of the 60/20/20 rule that drifts would put
    different rows in TEST for different models and quietly destroy every
    cross-model comparison.
    """
    if not want:
        return groups
    import run_fit
    test_r = 0.40 if len(groups) <= 500 else 0.20
    ratios = (1 - test_r - 0.20, 0.20, test_r)
    keep = [g for g in groups
            if run_fit.split_hash(g[0]["group_id"], ratios) == want]
    print(f"[split] {want}: {len(keep)} of {len(groups)} states "
          f"(test_r={test_r})", flush=True)
    return keep


def sample_groups(groups, n):
    """The first `n` states in split-hash order, with the id set fingerprinted.

    Ordered by the hash the split already uses, not by file order, so the same
    `n` comes out whatever order the rows were written in and whichever machine
    runs it. The printed digest is what makes an A/B on a subset auditable: both
    sides must report the same one or they were not scored on the same rows.
    """
    if not n:
        return groups
    import hashlib
    keep = sorted(groups,
                  key=lambda g: hashlib.sha1(g[0]["group_id"].encode()).hexdigest()
                  )[:n]
    ids = sorted(r["id"] for g in keep for r in g)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
    print(f"[sample] {len(keep)} of {len(groups)} states, id-set sha256 {digest}",
          flush=True)
    return keep


# --------------------------------------------------------------- engine ----
def make_engine(dry):
    from core.jev_engine import JevEngine, MODEL_PATH, MODEL_QUANT
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if dry:
        # prompt-seeded: two states sharing a question set get different
        # numbers, so a state scored with another state's plan fails here
        from core.fake_llm import FakeLLM
        return JevEngine(FakeLLM(), tok, pad_shared=_pad_shared()), tok
    if os.environ.get("JEV_BACKEND") == "bonsai":
        # The tokenizer above stays the Qwen3.5 one -- Bonsai's vocabulary is
        # identical. OUT_DIR is NOT: it is derived from the tokenizer path, so
        # without JEV_OUT_DIR this run would land on top of the published 9B
        # logits. Refuse rather than overwrite.
        import core.bonsai_llm as bonsai
        if not os.environ.get("JEV_OUT_DIR", "").strip():
            sys.exit("JEV_BACKEND=bonsai needs JEV_OUT_DIR (the default path is "
                     "derived from the tokenizer and would overwrite runs/calib)")
        llm = bonsai.from_env()
        print(f"[engine] backend=bonsai ctx/seq={llm.ctx_per_seq} "
              f"n_seq_max={llm.n_seq_max} out_dir={OUT_DIR}", flush=True)
        return JevEngine(llm, tok, pad_shared=_pad_shared()), tok

    from vllm import LLM
    # JEV_APC=0 turns the prefix cache off: the batch-equivalence gate needs to
    # know whether a cached catalog block computed under one batch is what makes
    # a logit depend on batch composition at all
    apc = os.environ.get("JEV_APC", "1") != "0"
    print(f"[engine] pad_shared={_pad_shared()} apc={apc} out_dir={OUT_DIR}",
          flush=True)
    llm = LLM(model=MODEL_PATH, quantization=MODEL_QUANT,
              max_model_len=12000, gpu_memory_utilization=0.92,
              enable_prefix_caching=apc)
    return JevEngine(llm, tok, pad_shared=_pad_shared()), tok


def row_out(r, diag, eng, prompt_toks, group=None):
    e = eng["_engine"]
    if e["fields_ms"] is None:
        # batched: one generate call covered several states, so this row has no
        # GPU time of its own. Write null -- an evenly split number would read
        # like a measurement (S5: a number that fits any hypothesis proves none)
        ms_gpu, ms_plan = None, round(e["plan_ms"], 1)
    else:
        ms_gpu = round(e["fields_ms"], 1)
        ms_plan = round(e["elapsed_ms"] - e["fields_ms"] - e["prefill_ms"], 1)
    o = {
        "id": r["id"], "dataset": r["_ds"], "qtype": r["qtype"],
        # prefer the data's group_id (Gizmo contract #1: NVD = CVE-ID);
        # sha1(state) fallback for files built before the rebuild
        "group": group or (
            str(r["group_id"]) if r.get("group_id") else
            (hashlib.sha1(r["state"].encode()).hexdigest()[:12]
             if isinstance(r.get("state"), str) else None)),
        "question_key": r["question_key"],
        "label": norm_label(r),
        "logits": diag["log_mass"],
        "in_set_mass": diag["in_set_mass"],
        "low_evidence": diag["low_evidence"],
        "ms_plan": ms_plan, "ms_gpu": ms_gpu,
        # how this row was produced, so a mixed file stays readable
        "batch_size": e["batch_size"],
        "split_hint": r.get("split_hint"),
        "state_tokens": r.get("state_tokens"),
        "prompt_tokens": prompt_toks,
        "source": r.get("source"), "license": r.get("license"),
        "label_source": r.get("label_source"),
        "known_benchmark": r.get("known_benchmark", False),
        # H6 diagnostic rows must never enter FIT/SELECT/TEST (Biscuit)
        "probe": bool(r.get("probe", False)),
        "probe_kind": r.get("probe_kind"),
        "perm_id": r.get("perm_id"),
        "variant": r.get("variant"),
    }
    if r["qtype"] == "choice":
        o["options"] = [str(x) for x in r["options"]]
    elif r["qtype"] == "score":
        o["scale"] = r["scale"]
        # engine keys are indices "0".."K-1"; relabel to actual scale values
        off = r["scale"]["min"]
        o["logits"] = {str(int(k) + off): v for k, v in o["logits"].items()}
    return o


def batch_chunks(groups, desc_map, n):
    """State-groups in chunks of <=n, each chunk paired with ONE question set
    per state (run_many scores every state against its own).

    Full chunks are drawn from one signature bucket first, because states that
    share a catalogue share cache blocks inside the call. What is left over --
    including datasets where the question text lives in the row, so every state
    is its own bucket (boolq, pubmedqa, vi_boolq: 771 states, 74.9 % of the
    generate calls before this) -- is packed into mixed chunks, which is safe
    only because each state carries its own set.

    The signature is order-sensitive on purpose (see the json.dumps below).
    n=1 yields singletons in file order: the old one-state-per-call path.
    """
    if n <= 1:
        for g in groups:
            yield [g], [build_questions(g, desc_map)]
        return
    buckets = OrderedDict()
    for g in groups:
        qs = build_questions(g, desc_map)
        # NOT sort_keys: option order is part of the prompt, and a bucket is
        # the unit that shares a catalogue. With sort_keys=True the four
        # permutation_arc orders of one item hashed to one signature and p1-p3
        # were scored against p0's catalogue -- 300 of 300 rows, the exact
        # question that dataset exists to ask. Question order counts too:
        # requests are built per question, in order.
        sig = json.dumps(qs)
        buckets.setdefault(sig, (qs, []))[1].append(g)

    leftover = []
    for qs, gs in buckets.values():
        i = 0
        while len(gs) - i >= n:
            yield gs[i:i + n], [qs] * n
            i += n
        leftover.extend((g, qs) for g in gs[i:])
    for i in range(0, len(leftover), n):
        part = leftover[i:i + n]
        yield [g for g, _ in part], [qs for _, qs in part]


def run_states(engine, chunk, qs_list):
    """Score one chunk of state-groups in a single generate, each state against
    its own question set."""
    results = engine.run_many([g[0]["state"] for g in chunk], qs_list)
    out = []
    for group, res in zip(chunk, results):
        diag = res["_engine"]["per_question"]
        prompt_toks = res["usage"]["input_tokens"]
        for r in group:
            d = diag.get(r["question_key"])
            assert d is not None, f"{r['id']}: no engine diag for {r['question_key']}"
            out.append(row_out(r, d, res, prompt_toks))
    return out


def leak_diag(engine, rows, dmap, n_states=2, k=20):
    """Where does the non-catalog mass go at the first scored node?
    vLLM rejects logprobs + logprob_token_ids of different sizes, so this
    issues its own top-K request on the SAME prompt ids the row was scored
    with (engine._shared_ids -- rebuilding the prompt here without the pad
    walls measured a different distribution and made the top-K disagree with
    the row's stored in_set_mass)."""
    from vllm import SamplingParams
    from core.jev_engine import QSpec
    out = []
    for g in group_states(rows)[:n_states]:
        qs = build_questions(g, dmap)
        qspecs = [QSpec(qid=k_, **v) for k_, v in qs.items()]
        ids, _ = engine._shared_ids(g[0]["state"], qspecs)
        built = engine._plan(ids, qspecs)
        first_ids = built["requests"][0][0]
        res = engine.llm.generate(
            [engine._prompt(first_ids)],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=k),
            use_tqdm=False,
        )
        lp = res[0].outputs[0].logprobs[0]
        opts = set()
        for q in qspecs:
            opts |= {o for o in q.options}
        tops = sorted(lp.values(), key=lambda v: -v.logprob)[:k]
        first_tok = {}
        for q in qspecs:
            _, paths = engine._paths_for(ids, {o: o for o in q.options})
            for o, path in paths.items():
                first_tok.setdefault(engine.tok.decode([path[0]]), []).append(o)
        out.append({
            "id": g[0]["id"],
            "top": [(getattr(v, "decoded_token", "?"), round(math.exp(v.logprob), 4),
                     first_tok.get(getattr(v, "decoded_token", "?"), []))
                    for v in tops],
            "options_first_tokens": sorted(opts)[:6],
        })
    return out


# ----------------------------------------------------------------- prior ----
def canonical_schema(ds, rows, desc_map=None):
    """Field set + options = the most common layout of the dataset."""
    desc_map = desc_map or {}
    layouts = Counter()
    for r in rows:
        layouts[(r["question_key"], r["qtype"],
                 tuple(r["options"]) if r["qtype"] == "choice" else None)] += 1
    keys = OrderedDict()
    # deterministic: field order by global question_key frequency
    freq = Counter(r["question_key"] for r in rows)
    by_key = {}
    for (qkey, qt, opts), n in layouts.items():
        best = by_key.get(qkey)
        if best is None or n > best[2]:
            by_key[qkey] = (qt, opts, n)
    for qkey in sorted(by_key, key=lambda k: (-freq[k], k)):
        qt, opts, _ = by_key[qkey]
        if qt == "noul":
            crit = {"true": "yes", "false": "no"}
        elif qt == "choice":
            crit = {str(o): desc_map.get(str(o)) or naturalize(str(o))
                    for o in opts}
        else:  # score prior: scale from the rows of this question_key
            scales = Counter((r["scale"]["min"], r["scale"]["max"])
                             for r in rows if r["question_key"] == qkey)
            mn, mx = scales.most_common(1)[0][0]
            crit = [str(v) for v in range(mn, mx + 1)]
        keys[qkey] = {"type": qt, "instructions": "", "criteria": crit,
                      "_is_choice": qt == "choice"}
    return keys


def run_prior(engine, ds, rows, desc_map=None):
    schema = canonical_schema(ds, rows, desc_map)
    # descriptions from any real row (question_desc carries the real question)
    desc = {r["question_key"]: r.get("question_desc", "")
            for r in reversed(rows)}
    for qk, q in schema.items():
        q["instructions"] = desc.get(qk, qk.replace("_", " "))
        if q.pop("_is_choice", False):
            q["instructions"] += " (answer with one option name exactly as listed)"
    res = engine.run("N/A", schema)
    diag = res["_engine"]["per_question"]
    prompt_toks = res["usage"]["input_tokens"]
    out = []
    for qk in schema:
        d = diag.get(qk)
        assert d is not None, f"prior: no diag for {qk}"
        e = res["_engine"]
        qdef = schema[qk]
        fake = {"id": f"{ds}::prior", "_ds": ds, "qtype": qdef["type"],
                "question_key": qk, "label": None, "split_hint": None,
                "state_tokens": 0, "source": "content-free prior",
                "license": "", "label_source": None,
                "known_benchmark": False, "group_id": None}
        if qdef["type"] == "choice":
            fake["options"] = list(qdef["criteria"].keys())
        elif qdef["type"] == "score":
            fake["scale"] = {"min": 0, "max": len(qdef["criteria"]) - 1}
        o = row_out(fake, d, res, prompt_toks)
        o["label"] = None
        out.append(o)
    return out


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="P0: CPU FakeLLM, no GPU")
    ap.add_argument("--probe", type=int, default=0,
                    help="P1: only first N state-groups per dataset")
    ap.add_argument("--datasets", default="",
                    help="comma list, by name or by path: emotion,h1/boolq "
                         "(default: all)")
    ap.add_argument("--leak-diag", default="",
                    help="datasets to diagnose mass leak on (top-K at first node)")
    ap.add_argument("--leak-ids", default="",
                    help="restrict --leak-diag to these row ids (comma list). "
                         "The leak lives in a handful of rows, not in the first "
                         "two states of the file, so the diagnosis has to be "
                         "aimed at the rows that actually leaked.")
    ap.add_argument("--batch", type=int, default=None,
                    help="states per generate call (1 = the old one-state path; "
                         "only states sharing a question set are batched). "
                         "Unset means 32 for a collect and 1 for --dry: the "
                         "pre-commit gate stays one-state-per-call, and the "
                         "run_many cross-check runs only when you ask for it")
    ap.add_argument("--sample", type=int, default=0,
                    help="first N states in split-hash order; prints the id-set "
                         "digest so an A/B on a subset is auditable")
    ap.add_argument("--split", default="", choices=["", "FIT", "SELECT", "TEST"],
                    help="collect only one split (same hash as run_fit)")
    ap.add_argument("--skip-prior", action="store_true")
    ap.add_argument("--only-prior", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    # data/calib is committed, so a set whose text cannot be redistributed lives
    # outside it (data/raw/... , gitignored) and points JEV_CALIB_DIR here. The
    # rows are the same brief-§1 shape either way; only the licence differs.
    base = os.environ.get("JEV_CALIB_DIR", "data/calib")
    if not os.path.isdir(base):
        raise SystemExit(f"JEV_CALIB_DIR={base!r} is not a directory")
    files = sorted(
        os.path.join(dp, f) for dp, _, fs in os.walk(base)
        for f in sorted(fs)
        if f.endswith(".jsonl") and "_samples" not in dp and not f.startswith("_")
    )
    # versioned datasets: nvd2023.jsonl superseded by nvd2023_v2.jsonl
    # (Gizmo freeze protocol: new files, old ones stay on disk)
    files = [f for f in files if not os.path.exists(f[:-6] + "_v2.jsonl")]
    if args.datasets:
        # Accept both the bare name and the h*/name path. The datasets live in
        # per-hypothesis subdirectories, so the original "base + name + .jsonl"
        # never resolved and the flag always raised FileNotFoundError; matching
        # by name fixed that but then the path form printed in --help stopped
        # working, which cost Mario a run. Both work now.
        def keys(f):
            rel = os.path.relpath(f, base)
            return {os.path.basename(f)[:-6], rel[:-6], rel}
        want = [d.strip() for d in args.datasets.split(",") if d.strip()]
        files = [f for f in files if keys(f) & set(want)]
        missing = set(want) - {k for f in files for k in keys(f)}
        if missing:
            sys.exit(f"--datasets: no such dataset under {base}: {sorted(missing)}")

    # ---------- P0 row contract ----------
    n_bad = 0
    all_rows = []
    ids = set()
    for path in files:
        rows = load_dataset(path)
        for r in rows:
            err = check_row(r)
            if err:
                n_bad += 1
                print(f"[P0-ROW-FAIL] {r.get('id', path)}:{r['_ln']}: {err}")
            if r["id"] in ids:
                n_bad += 1
                print(f"[P0-ROW-FAIL] duplicate id {r['id']}")
            ids.add(r["id"])
            all_rows.append(r)
        print(f"[load] {os.path.relpath(path, base)}: {len(rows)} rows")
    if n_bad:
        sys.exit(f"P0 failed: {n_bad} row-contract violations (fix data first)")

    if args.dry:
        # ---------- P0 engine pass: FakeLLM, real tokenizer ----------
        engine, tok = make_engine(dry=True)
        # unset => 1: the pre-commit gate must not silently cost twice the wall
        # time because the collect's default happens to be 32
        n_batch = 1 if args.batch is None else args.batch
        t0 = time.perf_counter()
        checked, max_prompt, crosschecked = 0, 0, 0
        per_ds = {}
        for path in files:
            rows = load_dataset(path)
            ds = rows[0]["_ds"]
            dmap = load_catalog_desc(path)
            groups = sample_groups(filter_split(group_states(rows), args.split),
                                   args.sample)
            if args.probe:
                groups = groups[: args.probe]
            schema = canonical_schema(ds, rows, dmap)  # dry prior too
            # through batch_chunks even at n=1 (singletons, file order): the
            # bucket signature is the thing 0c7634b got wrong, so the gate has
            # to walk the real bucketing rather than a loop of its own
            for chunk, qs_list in batch_chunks(groups, dmap, n_batch):
                solo = []
                for g, qs in zip(chunk, qs_list):
                    res = engine.run(g[0]["state"], qs)  # asserts invariant #1
                    pt = res["usage"]["input_tokens"]
                    max_prompt = max(max_prompt, pt)
                    if pt >= PROMPT_LIMIT:
                        print(f"[P0-FAIL] prompt {pt} >= {PROMPT_LIMIT}: {g[0]['id']}")
                        sys.exit(1)
                    nans = res["answers"]
                    assert all(q in nans for q in qs), "missing answer for a field"
                    checked += len(g)
                    solo.append(res)
                if len(chunk) < 2:
                    continue
                # one generate for the whole chunk must reproduce the solo runs
                # exactly: the fake is deterministic, so any difference is the
                # engine handing a state another state's plan, not noise
                many = engine.run_many([g[0]["state"] for g in chunk], qs_list)
                for g, qs, a, b in zip(chunk, qs_list, solo, many):
                    if a["answers"] == b["answers"] and \
                            a["_engine"]["per_question"] == b["_engine"]["per_question"]:
                        crosschecked += 1
                        continue
                    print(f"[P0-FAIL] batched != solo for {g[0]['id']} "
                          f"(chunk of {len(chunk)}, --batch {n_batch})")
                    for q in qs:
                        sa, sb = a["answers"].get(q), b["answers"].get(q)
                        if sa != sb:
                            print(f"    {q}: solo={sa!r} batched={sb!r}")
                        elif a["_engine"]["per_question"].get(q) != \
                                b["_engine"]["per_question"].get(q):
                            print(f"    {q}: same answer, different mass")
                    sys.exit(1)
            if not args.skip_prior and not args.only_prior:
                engine.run("N/A", schema)  # prior prompt builds too
            per_ds[ds] = len(groups)
        dt = time.perf_counter() - t0
        print(f"[P0] dry pass: {checked} rows / {sum(per_ds.values())} states "
              f"+ {len(per_ds)} priors, max prompt {max_prompt} tok, "
              f"{dt:.1f}s CPU")
        print(f"[P0] batch cross-check: {crosschecked} states re-scored solo, "
              f"identical" if crosschecked else
              f"[P0] batch cross-check: NOT RUN at --batch {n_batch} "
              f"(needs >1; run --dry --batch 8 to exercise run_many)")
        print("[P0] PASS -- contract clean, safe to burn GPU")
        return

    # ---------- GPU ----------
    engine, tok = make_engine(dry=False)
    summary = []
    for path in files:
        rows = load_dataset(path)
        ds = rows[0]["_ds"]
        out_path = os.path.join(OUT_DIR, f"{ds}.logits.jsonl")
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["id"])
                    except json.JSONDecodeError:
                        pass
        dmap = load_catalog_desc(path)
        groups = sample_groups(filter_split(group_states(rows), args.split),
                               args.sample)
        if args.probe:
            groups = groups[: args.probe]
        todo = [g for g in groups
                if not all(r["id"] in done for r in g)]
        n_out = 0
        t0 = time.perf_counter()
        n_states = 0
        with open(out_path, "a") as wf:
            for chunk, qs in batch_chunks(todo, dmap, 32 if args.batch is None else args.batch):
                for o in run_states(engine, chunk, qs):
                    wf.write(json.dumps(o) + "\n")
                wf.flush()
                n_out += sum(len(g) for g in chunk)
                n_states += len(chunk)
                print(f"[collect] {ds}: {n_states}/{len(todo)} states, "
                      f"{n_out} rows, {time.perf_counter() - t0:.0f}s",
                      flush=True)
        dt = time.perf_counter() - t0
        # prior
        n_prior = 0
        if not args.skip_prior and not any(
                i.startswith(f"{ds}::prior") for i in done):
            for dsname, drows in ((ds, rows),):
                with open(out_path, "a") as wf:
                    for o in run_prior(engine, dsname, drows, dmap):
                        o["id"] = f"{ds}::prior::{o['question_key']}"
                        wf.write(json.dumps(o) + "\n")
                    wf.flush()
                    n_prior = 1
        summary.append((ds, len(todo), n_out, n_prior, dt))
    if args.leak_diag:
        want = set(args.leak_diag.split(","))
        want_ids = {i for i in args.leak_ids.split(",") if i}
        for path in files:
            rows = load_dataset(path)
            ds = rows[0]["_ds"]
            if ds not in want:
                continue
            sel = [r for r in rows if r["id"] in want_ids] if want_ids else rows
            if want_ids and not sel:
                continue
            print(f"[leak-diag] {ds}: {len(sel)} rows")
            for d in leak_diag(engine, sel, load_catalog_desc(path),
                               n_states=max(2, len(sel))):
                print(f"  {d['id']}: " + ", ".join(
                    f"{t!r}={p}" + (f"->{owners[0]}" if owners else "->OUT")
                    for t, p, owners in d["top"][:10]))
                print(f"    catalog options (sample): {d['options_first_tokens']}")

    for ds, ng, nr, np_, dt in summary:
        print(f"[collect] {ds}: {ng} states, {nr} rows, prior={'y' if np_ else 'n'}, "
              f"{dt:.0f}s", flush=True)
    print("[collect] done", flush=True)


if __name__ == "__main__":
    main()
