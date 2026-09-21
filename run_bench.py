"""Production-mode benchmark: same schema (catalog), DIFFERENT states per
request -- the realistic serving pattern the [catalog][STATE] reorder targets.

Per preset: 6 sequential runs with 6 state variants.
- engine: JEV_SKIP_WARM=1 (in-batch sharing); run 1 = cold (catalog prefilled),
  runs 2..6 = catalog cache-hit (state is the only new content)
- AR baseline: same 6 states, same full catalog prompt

Report: cold ms / median warm ms for both, plus schema validity.

Run: HF_HUB_OFFLINE=1 ... JEV_SKIP_WARM=1 python run_bench.py
"""

import json
import os
import time


def state_variants(context: str, n: int = 6) -> list[str]:
    out = []
    for i in range(n):
        v = context + (
            f"\nCase reference: TX-{i:03d}-KQ. Handling agent note #{i}: "
            f"prior contact {'resolved' if i % 2 else 'open'}."
        )
        out.append(v)
    return out


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM

    from core.jev_engine import JevEngine, pad_shared_default
    from run_bench_ar import ar_baseline, map_schema_to_questions

    from core.jev_engine import MODEL_PATH as MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("loading vLLM ...", flush=True)
    mnbt = int(os.environ.get("MNBT", "8192"))
    llm = LLM(model=MODEL, max_model_len=16384, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, disable_log_stats=True,
              max_num_batched_tokens=mnbt)
    pad = pad_shared_default()
    print(f"[engine] pad_shared={pad}", flush=True)
    eng = JevEngine(llm, tokenizer, pad_shared=pad)

    presets = ["fintech_fraud", "code_security", "support_triage", "high_cardinality_255"]
    report = {}
    for name in presets:
        d = json.load(open(f"reference/presets/{name}.json"))
        context, schema = d["context"], d["schema"]
        questions = map_schema_to_questions(schema)
        states = state_variants(context)

        print(f"\n=== {name}: {len(questions)} fields, {len(states)} state variants ===", flush=True)
        entry = {"n_fields": len(questions), "n_states": len(states)}

        eng_ms, ok_all = [], True
        for i, st in enumerate(states):
            t0 = time.perf_counter()
            r = eng.run(st, questions)
            ms = (time.perf_counter() - t0) * 1000
            legal = all(
                (a["noul"] is not None) if q["type"] == "noul"
                else a["choice"] in q["criteria"]
                for q, a in ((questions[qid], r["answers"][qid]) for qid in questions)
                if q["type"] != "score"
            )
            ok_all &= legal
            eng_ms.append(round(ms, 1))
            tag = "cold" if i == 0 else "warm"
            print(f"  engine[{tag}] {ms:8.1f}ms valid={legal}", flush=True)
        warm = eng_ms[1:]
        warm.sort()
        entry["engine"] = {"cold_ms": eng_ms[0], "warm_median_ms": warm[len(warm)//2],
                           "all_ms": eng_ms, "valid": ok_all}

        ar_ms, ar_ok = [], True
        for i, st in enumerate(states):
            b = ar_baseline(llm, tokenizer, st, schema)
            ar_ms.append(b["elapsed_ms"])
            ar_ok &= b["schema_match"]
        ar_warm = sorted(ar_ms[1:])
        entry["ar"] = {"cold_ms": ar_ms[0], "warm_median_ms": ar_warm[len(ar_warm)//2],
                       "all_ms": ar_ms, "valid": ar_ok}
        print(f"  AR cold {ar_ms[0]:.0f} / warm median {ar_warm[len(ar_warm)//2]:.0f} valid={ar_ok}")

        report[name] = entry

    print("\n=== PRODUCTION SUMMARY (catalog cached, state varies) ===")
    print(f"{'preset':22s} {'eng cold':>9s} {'eng warm':>9s} {'AR cold':>8s} {'AR warm':>8s}  {'warm ratio':>10s}")
    for name, e in report.items():
        ec, ew = e["engine"]["cold_ms"], e["engine"]["warm_median_ms"]
        ac, aw = e["ar"]["cold_ms"], e["ar"]["warm_median_ms"]
        print(f"{name:22s} {ec:>9.0f} {ew:>9.0f} {ac:>8.0f} {aw:>8.0f}  {aw/ew:>9.1f}x")
    from core.jev_engine import MODEL_LABEL
    out = ("runs/jev_bench_prod.json" if MODEL_LABEL == "jev-rlcd-qwen3.5-9b-awq"
           else f"runs/jev_bench_prod_{MODEL_LABEL.replace('jev-rlcd-', '')}.json")
    if not pad_shared_default():
        out = out.replace(".json", "_pad0.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
