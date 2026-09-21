"""Step I benchmark: harshatheg's 4 presets mapped to Jev questions.

Compares (1) our Jev-contract engine on 3090, (2) our local AR baseline
(same model/machine), (3) their published M4 Max numbers (reference only --
different model Qwen2.5-1.5B-4bit and hardware).

CAVEATS (from deliberation):
- harshatheg torch-path scores FIRST TOKEN only; on code_security every CWE_*
  collides at 'C' and tariff_255 collapses to a 10-way digit decision, so their
  68ms/89ms reference numbers do not solve the same problem our trie solves.
- tariff_255 is expected to exceed the scored-node budget (documented v1
  limitation: code-map fallback out of scope).

Run: HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
     VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python run_jev_bench.py
"""

import json
import os
import sys
import time


def map_schema_to_questions(schema: dict) -> dict:
    questions = {}
    for name, spec in schema.items():
        if spec.get("type") == "boolean":
            questions[name] = {"type": "noul", "instructions": spec.get("description", name)}
        else:
            questions[name] = {
                "type": "choice",
                "instructions": spec.get("description", name),
                "criteria": {c: c for c in spec.get("choices", [])},
            }
    return questions


def ar_baseline(llm, tok, context: str, schema: dict) -> dict:
    """harshatheg-style naive JSON generation baseline on the same engine."""
    from vllm import SamplingParams

    lines = ["{"]
    for name, field in schema.items():
        if field.get("type") == "boolean":
            lines.append(f'  "{name}": boolean, // {field.get("description", "")}')
        else:
            # FULL option list, no truncation: the engine reads every criteria
            # description (8448-token catalog on tariff) -- a truncated AR
            # prompt made AR look 3x faster than it really is: 842 ms is
            # impossible for a true 8448-token prefill at 4-5K tok/s.
            ch = field.get("choices", [])
            cs = " | ".join(f'"{c}"' for c in ch)
            lines.append(f'  "{name}": {cs}, // {field.get("description", "")}')
    lines.append("}")
    schema_str = "\n".join(lines)

    msgs = [
        {"role": "system", "content": (
            "You are a precise data extraction system. You must output ONLY a valid "
            "JSON object with 2-space indentation matching the schema below. "
            f"Do not output markdown.\n\nJSON Schema:\n{schema_str}")},
        {"role": "user", "content": f"Analyze the following context and generate the "
                                    f"required JSON object:\n\n{context}"},
    ]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    except Exception:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt += "{\n  "

    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(max_tokens=512, min_tokens=5,
                                                temperature=0.2, detokenize=True),
                       use_tqdm=False)
    dt = (time.perf_counter() - t0) * 1000
    text = out[0].outputs[0].text
    ntok = len(out[0].outputs[0].token_ids)

    parsed, valid = None, False
    try:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1:
            parsed = json.loads(text[i:j + 1])
            valid = True
    except Exception:
        pass
    schema_match = bool(valid and isinstance(parsed, dict)
                        and set(parsed.keys()) == set(schema.keys()))
    return {"elapsed_ms": round(dt, 1), "tokens": ntok, "valid_json": valid,
            "schema_match": schema_match}


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM

    from core.jev_engine import JevEngine

    from core.jev_engine import MODEL_PATH as MODEL, require_model

    require_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("loading vLLM ...", flush=True)
    import os
    mnbt = int(os.environ.get("MNBT", "8192"))
    llm = LLM(model=MODEL, max_model_len=16384, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, disable_log_stats=True,
              max_num_batched_tokens=mnbt)
    eng = JevEngine(llm, tokenizer)

    presets = ["fintech_fraud", "code_security", "support_triage", "high_cardinality_255"]
    mlx_ref = {"fintech_fraud": ("420", "75"), "code_security": ("380", "68"),
               "support_triage": ("1900", "270"), "high_cardinality_255": ("500", "89")}

    report = {}
    for name in presets:
        d = json.load(open(f"reference/presets/{name}.json"))
        context, schema = d["context"], d["schema"]
        questions = map_schema_to_questions(schema)
        print(f"\n=== {name}: {len(questions)} fields ===", flush=True)

        entry = {"n_fields": len(questions)}
        try:
            eng.run(context, questions)  # warm
            r = eng.run(context, questions)  # measured
            e = r["_engine"]
            # validity: every answer must be a schema-legal value
            legal = all(
                (a["noul"] is not None) if q["type"] == "noul"
                else a["choice"] in q["criteria"]
                for q, a in ((questions[qid], r["answers"][qid]) for qid in questions)
                if q["type"] != "score"
            )
            entry.update({
                "elapsed_ms": e["elapsed_ms"], "fields_ms": e["fields_ms"],
                "n_requests": e["n_requests"], "n_calls": e["n_generate_calls"],
                "schema_match": legal,
                "answers_head": {k: (v.get("noul") or v.get("choice"))
                                 for k, v in list(r["answers"].items())[:4]},
            })
            print(f"  engine: {e['elapsed_ms']}ms ({e['n_requests']} nodes, "
                  f"{e['n_generate_calls']} calls) schema_match={legal}")
        except AssertionError as ex:
            entry["error"] = f"{type(ex).__name__}: {ex}"
            print(f"  engine: OUT OF SCOPE -- {ex}")

        base = ar_baseline(llm, tokenizer, context, schema)
        entry["ar_baseline"] = base
        print(f"  AR base: {base['elapsed_ms']}ms ({base['tokens']} tok, "
              f"match={base['schema_match']})")
        if "elapsed_ms" in entry:
            entry["speedup"] = round(base["elapsed_ms"] / entry["elapsed_ms"], 2)
            print(f"  speedup vs local AR: {entry['speedup']}x")
        report[name] = entry

    print("\n=== SUMMARY (3090, Qwen3.5-9B-AWQ) vs harshatheg M4 Max (Qwen2.5-1.5B-4bit) ===")
    print(f"{'preset':22s} {'ours ms':>8s} {'AR ms':>7s} {'ratio':>6s} {'nodes':>6s}  {'mlx ref (AR/par)':>18s}")
    for name in presets:
        rr = report[name]
        if "error" in rr:
            print(f"{name:22s} {'OUT OF SCOPE (budget)':>8s} {rr['ar_baseline']['elapsed_ms']:>7.0f}"
                  f"        {rr['n_fields']:>6d}  {mlx_ref[name][0]+'/'+mlx_ref[name][1]:>18s}"
                  f"  <- first-token, not same problem")
            continue
        print(f"{name:22s} {rr['elapsed_ms']:>8.0f} {rr['ar_baseline']['elapsed_ms']:>7.0f} "
              f"{rr['speedup']:>5.1f}x {rr['n_requests']:>6d}  {mlx_ref[name][0]+'/'+mlx_ref[name][1]:>18s}")
    print("(mlx ref: different model/hw; code_security+tariff refs score first token only)")

    with open("runs/jev_bench.json", "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print("wrote runs/jev_bench.json")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
