"""GPU sanity: run the 4 Cloudflare-docs Jev examples with the real model and
compare qualitatively against their published answers (jev-1.13.0 reference).

Run: HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
     VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python run_jev_sanity.py
"""

import json
import os
import sys
import time


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM

    from core.jev_engine import JevEngine, pad_shared_default

    from core.jev_engine import MODEL_PATH as MODEL
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("loading vLLM ...", flush=True)
    llm = LLM(model=MODEL, max_model_len=16384, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, disable_log_stats=True)
    eng = JevEngine(llm, tokenizer, pad_shared=pad_shared_default())
    import torch

    from core.jev_engine import MODEL_LABEL
    print(f"model={MODEL} label={MODEL_LABEL} block_size={eng._block_size()} "
          f"vram_after_load={torch.cuda.memory_allocated()/2**30:.2f}GiB "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB", flush=True)

    cases = [
        {
            "name": "cf_support",
            "state": "Help! My payouts have been failing for 3 days.",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                              "criteria": {"true": "Explicitly time-sensitive",
                                           "false": "No urgency expressed"}},
                "department": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "Payments, invoicing, refunds",
                                            "technical": "Bugs, outages, integrations",
                                            "sales": "Pricing, upgrades, new accounts"}},
                "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                                "criteria": ["Calm", "Frustrated", "Very angry"]},
            },
            "expected": "jev: noul 0.95 | choice billing 0.87/0.13 tech | score ~1.0 (0.96 lv1)",
        },
        {
            "name": "cf_routing",
            "state": "I cannot log in after changing my password, and the reset email never arrives.",
            "questions": {
                "department": {"type": "choice",
                               "instructions": "Which team should handle this support request?",
                               "criteria": {"account": "Login, password, profile, or security issues",
                                            "billing": "Charges, invoices, refunds, or subscriptions",
                                            "technical": "Product bugs, outages, or integrations",
                                            "other": "Requests that do not fit the other departments"}},
            },
            "expected": "jev: choice account, confidence 1.0",
        },
        {
            "name": "cf_risk",
            "state": {
                "account_age_days": 12,
                "recent_events": [
                    "Five failed login attempts",
                    "Password reset requested from a new country",
                    "Successful login from the usual device",
                ],
                "account_verified": True,
            },
            "questions": {
                "risk_level": {"type": "score",
                               "instructions": "How risky does this account activity appear?",
                               "criteria": ["Low risk: activity is consistent with the account history",
                                            "Moderate risk: some unusual activity needs monitoring",
                                            "High risk: multiple strong indicators of account compromise"]},
                "escalate": {"type": "noul",
                             "instructions": "Should this account be escalated for manual security review?",
                             "criteria": {"true": "The activity warrants immediate human review",
                                          "false": "The activity can be handled with normal automated controls"}},
            },
            "expected": "jev: score 1.84 (p2=0.84), confidence 0.77 | noul 0.81",
        },
        {
            "name": "cf_refund",
            "state": {
                "ticket": {"subject": "Duplicate charge",
                           "message": "I was charged twice for order A-104. Please refund the duplicate."},
                "order": {"id": "A-104",
                          "charges": [{"amount_usd": 49, "status": "captured"},
                                      {"amount_usd": 49, "status": "captured"}]},
                "refund_policy": "Duplicate charges are eligible for a refund.",
            },
            "questions": {
                "refund_requested": {"type": "noul",
                                     "instructions": "Does `ticket.message` request a refund?"},
                "policy_supports_refund": {"type": "noul",
                                           "instructions": "Does `refund_policy` support the refund requested in `ticket.message`, given `order.charges`?"},
            },
            "expected": "jev: noul 0.99 | noul 0.98",
        },
    ]

    report = {}
    for c in cases:
        # warm run then measured run (second absorbs any residual warmup)
        eng.run(c["state"], c["questions"])
        t0 = time.perf_counter()
        r = eng.run(c["state"], c["questions"])
        wall = (time.perf_counter() - t0) * 1000
        print(f"\n=== {c['name']} (wall {wall:.0f} ms) ===")
        print(f"  expected: {c['expected']}")
        for qid, a in r["answers"].items():
            print(f"  {qid}: {json.dumps(a, ensure_ascii=False)}")
        print(f"  engine: {r['_engine']}")
        report[c["name"]] = {"expected": c["expected"], "got": r}

    from core.jev_engine import MODEL_LABEL
    out = ("runs/jev_sanity.json" if MODEL_LABEL == "jev-rlcd-qwen3.5-9b-awq"
           else f"runs/jev_sanity_{MODEL_LABEL.replace('jev-rlcd-', '')}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nwrote {out}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
