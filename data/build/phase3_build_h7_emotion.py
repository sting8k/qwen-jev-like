"""Phase 3 h7 builder — DAIR Emotion (user-approved 2026-09-21, Emotion only, NC accepted).

Output:
  data/calib/h7/emotion.jsonl — 2,000 rows = the FULL dair-ai/emotion test split,
                                qtype=choice over the 6 source labels.

Why this set and not another (the open question it answers): we had no dataset on
which a third party has measured Jev itself, so "where do we stand against Jev and
the open Jev-likes" could only be answered on 4 Cloudflare cases, direction-only.
Labels are NOISY (see `label_source` below): everyone scores low here -- Jev 0.480,
laya 0.595-0.600, and kev names "noisy-label emotion" as one of its four residual gaps.
Read a low accuracy on this set as the set's ceiling, not only as a model failure.

DAIR Emotion is the one shared item where Jev is published as *badly calibrated*
(AbdelStark/jev-benchmarks: accuracy 0.480, Brier 0.846, NLL 5.588, and zero
probability on the true label for 16% of examples) -- i.e. exactly the quantity
Phase 3 exists to measure. laya (0.595/0.600) and kev (transfer-v4) also report it.

VERIFIED EQUIVALENCE (the basis of the comparison line, checked in build()):
  btzsc/btzsc @ fef2a2ac62b69c58670047dddf045c53d7c3cb5e, config `emotiondair`,
  is these same 2,000 texts in NLI form (2,000 x 6 hypotheses, exactly one
  entailed each): 2,000/2,000 texts overlap and 0/2,000 label disagreements.
  So the rows AbdelStark scored Jev on are a subset of this file's rows.

  What is NOT reproducible: their sample is drawn by their own code and their
  manifest is not published (results/runs/ is gitignored in their repo). The 100
  rows flagged `btzsc_sample` here are OUR class-balanced draw at THEIR seed
  (20260917) from the same 2,000 -- same source, same size, same balance,
  NOT guaranteed the same items. Report that line as n=100 signal, never as a
  head-to-head on identical items.

Distribution is the source's own (sadness 581, joy 695, love 159, anger 275,
fear 224, surprise 66) -- NOT stratified, on purpose: ECE is read against the
prior the data actually has. The balanced-100 subset has a different prior by
construction, which is why its accuracy is not comparable to the full file's.

Run: .venv-data/bin/python data/build/phase3_build_h7_emotion.py
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# repo root: this file lives in data/build/, so two levels up. Keep in sync
# with the directory depth -- ROOT is what every output path hangs off.
ROOT = Path(__file__).resolve().parents[2]
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))

BTZSC_REV = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
BTZSC_SEED = 20260917   # AbdelStark configs/pilot-v1.yaml
BTZSC_N = 100           # samples_per_dataset


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def check_btzsc_equivalence(ds, names):
    """Assert the pinned BTZSC config is this exact split, so the comparison line
    rests on a checked fact and not on two datasets sharing a name."""
    print(f"[btzsc] loading btzsc/btzsc:emotiondair @ {BTZSC_REV[:7]} ...")
    b = load_dataset("btzsc/btzsc", "emotiondair", revision=BTZSC_REV, split="test")
    positive = {}
    for text, hyp, lab in zip(b["text"], b["hypothesis"], b["labels"]):
        if lab == 1:
            assert text not in positive, f"two entailed hypotheses for {text!r}"
            positive[text] = hyp.rsplit(": ", 1)[1]
    overlap = sum(1 for t in ds["text"] if t in positive)
    disagree = sum(1 for t, l in zip(ds["text"], ds["label"]) if positive.get(t) != names[l])
    print(f"  texts {overlap}/{len(ds)} overlap, {disagree} label disagreements")
    assert overlap == len(ds) and disagree == 0, "BTZSC is not this split any more"


def pick_btzsc_sample(rows):
    """Class-balanced 100 at their seed: the classes are ordered as the source
    orders them and the remainder goes to the first classes, so the draw is a
    pure function of (seed, source order)."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    labels = sorted(by_label, key=lambda x: [r["label"] for r in rows].index(x))
    base, extra = divmod(BTZSC_N, len(labels))
    rng = random.Random(BTZSC_SEED)
    picked = set()
    for i, lab in enumerate(labels):
        k = base + (1 if i < extra else 0)
        pool = by_label[lab]
        assert len(pool) >= k, f"{lab}: {len(pool)} rows, need {k}"
        picked |= {r["id"] for r in rng.sample(pool, k)}
    assert len(picked) == BTZSC_N
    return picked


def build():
    print("[emotion] loading dair-ai/emotion config `split`, split `test` ...")
    ds = load_dataset("dair-ai/emotion", "split", split="test")
    names = ds.features["label"].names
    print(f"  {len(ds)} rows, options = {names}")
    check_btzsc_equivalence(ds, names)

    rows = []
    for text, label in zip(ds["text"], ds["label"]):
        text = text.strip()
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:7]
        gid = f"emotion-test-{digest}"
        rows.append({
            "id": gid,
            "group_id": gid,           # one question per text: id == group_id
            "source": "dair-ai/emotion",
            "source_url": "https://huggingface.co/datasets/dair-ai/emotion",
            # card declares license `other`: educational/research use, no
            # redistribution grant. User accepted NC for measurement 2026-09-21.
            "license": "other (research/educational use; dair-ai card)",
            "license_nc": True,
            "state": text,
            "qtype": "choice",
            "question_key": "emotion",
            "question_desc": "Emotion the author of this tweet expresses",
            "options": list(names),
            "label": names[label],
            # NOT human: the card declares `annotations_creators: machine-generated`
            # and the data ships preprocessed by the CARER pipeline (Saravia et al.,
            # EMNLP 2018). Labels are machine-produced from the source collection,
            # so the brief's value is `derived`, same class as NVD CVSS -> severity.
            # Not a violation of "no LLM-generated labels" (2018, pattern/graph based),
            # but it IS noisy supervision -- see README for what that does to ECE.
            "label_source": "derived",
            "lang": "en",
            "split_hint": "test",      # source split; FIT/SELECT/TEST is hash(group_id)
            "known_benchmark": True,
            "stratified": False,       # source prior kept on purpose
            # every row is a short tweet (max 62 tokens): whole-file property,
            # flagged per file like goemotions/banking77, not per row
            "short_state": True,
            "state_tokens": ntok(text),
        })

    assert len({r["id"] for r in rows}) == len(rows), "duplicate id / duplicate state"
    assert all(r["label"] in r["options"] for r in rows), "label outside options"

    sample = pick_btzsc_sample(rows)
    for r in rows:
        if r["id"] in sample:
            r["btzsc_sample"] = True

    out = ROOT / "data/calib/h7/emotion.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out} ({len(rows)} rows)")

    dist = Counter(r["label"] for r in rows)
    sub = Counter(r["label"] for r in rows if r.get("btzsc_sample"))
    print("  label dist :", dict(sorted(dist.items(), key=lambda kv: -kv[1])))
    print("  btzsc-100  :", dict(sorted(sub.items(), key=lambda kv: -kv[1])))
    toks = [r["state_tokens"] for r in rows]
    print("  state_tokens: min %d avg %.1f max %d" % (
        min(toks), sum(toks) / len(toks), max(toks)))


if __name__ == "__main__":
    build()
