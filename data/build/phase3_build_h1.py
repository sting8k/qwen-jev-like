"""Phase 3 H1 builder — noul-with-passage datasets -> data/calib/h1/*.jsonl

Datasets:
  - google/boolq      validation, 300 rows, CC-BY-SA-3.0
  - qiaojin/PubMedQA  pqa_labeled, ~400 rows (drop 'maybe'), MIT
  - allenai/scitail   tsv_format validation, 400 rows, Apache-2.0 (short_state)

Rules: 50-800 state tokens (Qwen tokenizer), short_state flag, keep source
split as split_hint, no fit/test split of our own, seed 42 everywhere.

Run: .venv-data/bin/python phase3_build_h1.py
"""
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# repo root: this file lives in data/build/, so two levels up. Keep in sync
# with the directory depth -- ROOT is what every output path hangs off.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "calib" / "h1"
SEED = 42
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))


def ntok(s: str) -> int:
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def stratified(rows, n, key, rng):
    """Proportional allocation, every class >=1, deterministic."""
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    total = len(rows)
    picked = []
    alloc = {k: max(1, round(n * len(v) / total)) for k, v in groups.items()}
    # scale pass to hit n exactly
    while sum(alloc.values()) > n:
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = min(alloc, key=lambda k: alloc[k])
        alloc[k] += 1
    for k, c in alloc.items():
        picked.extend(groups[k][:c])
    return picked


def write_jsonl(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {p} ({len(rows)} rows)")
    return p


def main():
    rng = random.Random(SEED)
    stats = {}

    # ---- BoolQ -------------------------------------------------------
    print("[boolq] loading ...")
    ds = load_dataset("google/boolq", split="validation")
    rows, drop_short, drop_long = [], 0, 0
    for i, ex in enumerate(ds):
        t = ntok(ex["passage"])
        if t < 50:
            drop_short += 1
            continue
        if t > 800:
            drop_long += 1
            continue
        rows.append({
            "id": f"boolq-val-{i:05d}",
            "group_id": f"boolq-val-{i:05d}",
            "source": "google/boolq",
            "source_url": "https://huggingface.co/datasets/google/boolq",
            "license": "CC-BY-SA-3.0",
            "state": ex["passage"],
            "qtype": "noul",
            "question_key": "passage_question",
            "question_desc": ex["question"],
            "label": bool(ex["answer"]),
            "label_source": "human",
            "lang": "en",
            "split_hint": "validation",
            "known_benchmark": True,
            "stratified": True,
            "state_tokens": t,
        })
    seen = set()
    deduped = [r for r in rows if not (r["state"] in seen or seen.add(r["state"]))]
    print(f"  boolq dedup: {len(rows)} -> {len(deduped)}")
    picked = stratified(deduped, 300, lambda r: r["label"], rng)
    write_jsonl("boolq.jsonl", picked)
    stats["boolq"] = dict(kept=len(picked), dropped_short=drop_short,
                          dropped_long=drop_long,
                          pool_dist=dict(Counter(str(r["label"]) for r in deduped)),
                          label_dist=dict(Counter(str(r["label"]) for r in picked)))

    # ---- PubMedQA ----------------------------------------------------
    print("[pubmedqa] loading ...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows, drop_maybe, drop_short, drop_long = [], 0, 0, 0
    for ex in ds:
        if ex["final_decision"] not in ("yes", "no"):
            drop_maybe += 1
            continue
        state = " ".join(ex["context"]["contexts"])
        t = ntok(state)
        if t < 50:
            drop_short += 1
            continue
        if t > 800:
            drop_long += 1
            continue
        rows.append({
            "id": f"pubmedqa-{ex['pubid']}",
            "group_id": f"pubmedqa-{ex['pubid']}",
            "source": "qiaojin/PubMedQA",
            "source_url": "https://huggingface.co/datasets/qiaojin/PubMedQA",
            "license": "MIT",
            "state": state,
            "qtype": "noul",
            "question_key": "final_decision",
            "question_desc": ex["question"],
            "label": ex["final_decision"] == "yes",
            "label_source": "human",
            "lang": "en",
            "split_hint": "train",  # HF has one split; pqa_labeled is the expert-labeled set
            "known_benchmark": True,
            "state_tokens": t,
        })
    picked = stratified(rows, min(400, len(rows)), lambda r: r["label"], rng)
    write_jsonl("pubmedqa.jsonl", picked)
    stats["pubmedqa"] = dict(kept=len(picked), dropped_maybe=drop_maybe,
                             dropped_short=drop_short, dropped_long=drop_long,
                             label_dist=dict(Counter(str(r["label"]) for r in picked)))

    # ---- SciTail -----------------------------------------------------
    print("[scitail] loading ...")
    ds = load_dataset("allenai/scitail", "tsv_format", split="validation")
    rows, drop_short, drop_long = [], 0, 0
    for i, ex in enumerate(ds):
        t = ntok(ex["premise"])
        if t < 10:  # degenerate only; single sentences are expected short (short_state)
            drop_short += 1
            continue
        if t > 800:
            drop_long += 1
            continue
        rows.append({
            "id": f"scitail-val-{i:05d}",
            "group_id": f"scitail-val-{i:05d}",
            "source": "allenai/scitail",
            "source_url": "https://huggingface.co/datasets/allenai/scitail",
            "license": "Apache-2.0",
            "state": ex["premise"],
            "qtype": "noul",
            "question_key": "entailment",
            "question_desc": f"Does the passage entail: \"{ex['hypothesis']}\"",
            "label": ex["label"] == "entails",
            "label_source": "human",
            "lang": "en",
            "split_hint": "validation",
            "known_benchmark": True,
            "short_state": True,  # premise is a single sentence by construction
            "state_tokens": t,
        })
    picked = stratified(rows, min(400, len(rows)), lambda r: r["label"], rng)
    write_jsonl("scitail.jsonl", picked)
    stats["scitail"] = dict(kept=len(picked), dropped_short=drop_short,
                            dropped_long=drop_long,
                            label_dist=dict(Counter(str(r["label"]) for r in picked)))

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
