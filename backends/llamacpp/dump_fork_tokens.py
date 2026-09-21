"""Dump the token ids fork_probe.cpp needs, using the SAME prompt builder as
gate.py (so the fork test and the server test speak about the same prompts).

8 states sharing one fintech_fraud catalog; the common token prefix is the
catalog region that seq 0 will hold and seq 1..8 will fork from.

  .venv/bin/python backends/llamacpp/dump_fork_tokens.py runs/fork_tokens.txt

Format (whitespace separated, parsed by ifstream >>):
  n_prefix n_suffix n_cand
  <prefix ids ...>
  <len> <ids ...>          x n_suffix
  <candidate token ids ...>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gate  # noqa: E402  (same prompt replica, same preset)

SCORED = '  "risk_level": "'

# 8 states. Only the STATE block differs; the catalog above it is byte-identical,
# which is the whole point of the engine's prompt order (AGENTS §4.3).
STATES = [
    gate.presets.FRAUD_CONTEXT,
    gate.STATE_B,
    "Transaction Review: Card-not-present purchase of $89.99 at 'StreamCo', "
    "Dublin, 9:12 PM local. Cardholder: account age 3 months, no prior pattern, "
    "geo-location matches billing. Device fingerprint known. 3D-Secure passed. "
    "Velocity: 2 transactions in 7 days. BIN issued in IE.",
    "Transaction Review: ATM withdrawal of $900.00, Bucharest, 04:41 AM local. "
    "Cardholder: account age 11 years, prior spend pattern domestic only, "
    "geo-location 6000km from home. Device fingerprint n/a. 3D-Secure n/a. "
    "Velocity: 3 withdrawals in 20 minutes. BIN issued in US.",
    "Transaction Review: Recurring charge of $14.99 at 'CloudBackup', "
    "same merchant as prior 22 months. Cardholder: account age 8 years, "
    "geo-location matches home. Device fingerprint known. 3D-Secure not required. "
    "Velocity: 1 transaction today. BIN issued in US.",
    "Transaction Review: Card-not-present purchase of $2,300.00 at 'GiftCardHub', "
    "unknown region, 02:58 AM local. Cardholder: account age 14 days, no prior "
    "pattern, geo-location unknown (proxy). Device fingerprint new. 3D-Secure "
    "challenge failed once. Velocity: 9 transactions in 6 minutes. BIN issued in US.",
    "Transaction Review: In-store chip purchase of $212.50 at 'HomeHardware', "
    "Denver, 11:05 AM local. Cardholder: account age 5 years, prior spend pattern "
    "home improvement seasonal, geo-location matches home ZIP. Device fingerprint "
    "known. 3D-Secure not required. Velocity: 1 transaction in 3 days. BIN issued in US.",
    "Transaction Review: Card-not-present refund reversal of $640.00 at "
    "'TicketResale', Lisbon, 1:30 PM local. Cardholder: account age 2 years, "
    "prior disputes 3, geo-location matches travel history. Device fingerprint "
    "known. 3D-Secure passed. Velocity: 2 transactions in 1 hour. BIN issued in US.",
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/fork_tokens.txt"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(gate.TOKENIZER_DIR, trust_remote_code=True)

    seqs = [tok(gate.build(tok, s, SCORED)).input_ids for s in STATES]

    # longest common prefix across ALL states = the catalog region
    p = 0
    while all(p < len(s) for s in seqs) and len({s[p] for s in seqs}) == 1:
        p += 1

    opts = gate.presets.FRAUD_SCHEMA["risk_level"]["choices"]
    cands = [tok(o, add_special_tokens=False).input_ids[0] for o in opts]
    if len(set(cands)) != len(cands):
        print("FAIL: option first-tokens collide, cannot score them apart")
        return 1

    lines = [f"{p} {len(seqs)} {len(cands)}", " ".join(map(str, seqs[0][:p]))]
    for s in seqs:
        suf = s[p:]
        lines.append(f"{len(suf)} " + " ".join(map(str, suf)))
    lines.append(" ".join(map(str, cands)))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {out}")
    print(f"  shared catalog prefix : {p} tokens")
    print(f"  states                : {len(seqs)}")
    print(f"  suffix lengths        : {[len(s)-p for s in seqs]}")
    print(f"  full-prompt lengths   : {[len(s) for s in seqs]}")
    print(f"  candidates {opts} -> ids {cands}")
    print(f"  fork should process   : {p} + {sum(len(s)-p for s in seqs)} = "
          f"{p + sum(len(s)-p for s in seqs)} tokens")
    print(f"  no-sharing would cost : {sum(len(s) for s in seqs)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
