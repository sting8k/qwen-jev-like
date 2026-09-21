"""Phase 3 score builder — SST-5, S2 only.

Outputs (NEW files; previously submitted JSONL are frozen on disk):
  data/calib/h3/sst5.jsonl      — 400 rows (80/level), score scale 0..4, label_source human
  data/calib/h3/sst5_desc.jsonl — the same 400 rows plus scale_labels (brief §1):
                                  level names instead of digits
  data/calib/h3/amazon_desc.jsonl — the same 500 rows of amazon_reviews_en.jsonl plus
                                  scale_labels; read off disk,
                                  no download, the published file is not touched

Purpose: an A/B against data/calib/h3/amazon_reviews_en.jsonl for the P1 remap bug.
Same task family (sentiment of a review), same balanced design, same shape of
question_desc -- the ONLY difference is that this scale is natively 0-based while
amazon's is 1-based. So:
  amazon bad + sst5 fine  -> the 1-based remap is the problem
  both bad                -> the model is genuinely weak at score
Deliberately NOT used here: the source's level names (label_text: very negative ..
very positive). Putting them in question_desc would change two variables at once and
destroy the comparison. They are kept in the README for a later descriptions-vs-digits test.

Run: .venv-data/bin/python phase3_build_score.py
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
SEED = 42
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))

PER_LEVEL = 80
MIN_WORDS, MAX_WORDS = 5, 400


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def build_sst5():
    print("[sst5] loading SetFit/sst5 test ...")
    ds = load_dataset("SetFit/sst5", split="test")
    rng = random.Random(SEED)

    by_level, level_name = {}, {}
    for ex in ds:
        level_name.setdefault(int(ex["label"]), set()).add(ex["label_text"])
        text = ex["text"].strip()
        if not (MIN_WORDS <= len(text.split()) <= MAX_WORDS):
            continue
        by_level.setdefault(int(ex["label"]), []).append(ex)
    print("  pool per level:", {k: len(v) for k, v in sorted(by_level.items())})
    assert all(len(v) == 1 for v in level_name.values()), "a label has several names"
    labels = [level_name[i].pop() for i in range(5)]
    print("  level names:", labels)

    rows = []
    for level in sorted(by_level):
        picked = rng.sample(by_level[level], PER_LEVEL)
        for ex in picked:
            text = ex["text"].strip()
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:7]
            gid = f"sst5-test-{ex['label']}-{digest}"
            rows.append({
                "id": gid,
                "group_id": gid,
                "source": "SetFit/sst5",
                "source_url": "https://huggingface.co/datasets/SetFit/sst5",
                # no license field on the mirror's card; SST-5 = Stanford Sentiment
                # Treebank (Socher et al. 2013), redistributed widely for research
                "license": "unspecified (SST-5 / Stanford Sentiment Treebank)",
                "state": text,
                "qtype": "score",
                "question_key": "sentiment",
                # parallel in shape to amazon's "Star rating the reviewer gave,
                # 1 = worst, 5 = best" -- endpoints only, no level names
                "question_desc": "Sentiment a human annotator gave this movie-review "
                                 "sentence, 0 = worst, 4 = best",
                "scale": {"min": 0, "max": 4},
                "label": int(ex["label"]),
                "label_source": "human",
                "lang": "en",
                "split_hint": "test",
                "known_benchmark": True,
                "stratified": True,
                "state_tokens": ntok(text),
            })
            if rows[-1]["state_tokens"] < 50:
                rows[-1]["short_state"] = True

    rng.shuffle(rows)
    assert len({r["id"] for r in rows}) == len(rows), "duplicate id"
    assert Counter(r["label"] for r in rows) == {i: PER_LEVEL for i in range(5)}
    write_jsonl(ROOT / "data/calib/h3/sst5.jsonl", rows)
    return rows, labels


def build_desc_variant(rows, labels, out_name):
    """Same rows, same group_id (so both sets land in the same split and pair line
    by line); the ONLY difference is scale_labels -- descriptive level names
    instead of bare digits. question_desc is left identical on purpose."""
    scales = {(r["scale"]["min"], r["scale"]["max"]) for r in rows}
    assert len(scales) == 1, f"{out_name}: mixed scales {scales}"
    lo, hi = scales.pop()
    assert len(labels) == hi - lo + 1, f"{out_name}: {len(labels)} labels for {hi - lo + 1} levels"
    # a "labelled" set that renders exactly like the unlabelled one is
    # indistinguishable from it after collection (collector asserts this too)
    assert labels != [str(v) for v in range(lo, hi + 1)], f"{out_name}: labels are the digits"
    out = []
    for r in rows:
        d = dict(r)
        d["id"] = r["id"] + "-desc"          # group_id stays r["group_id"]
        d["scale_labels"] = labels           # index-aligned to the scale (brief §1)
        out.append(d)
    write_jsonl(ROOT / "data/calib/h3" / out_name, out)
    return out


def build_amazon_desc():
    """The 1-based arm of the 2x2: amazon's scale is 1..5 while the catalog always
    enumerates from 0, so unlabelled it renders "0: 1 ... 4: 5" -- levels that are
    bare digits AND offset from their meaning. Labelling it separates the two."""
    src = ROOT / "data/calib/h3/amazon_reviews_en.jsonl"
    rows = [json.loads(l) for l in open(src)]
    print(f"[amazon_desc] read {src.name}: {len(rows)} rows")
    return build_desc_variant(rows, ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
                              "amazon_desc.jsonl")


if __name__ == "__main__":
    rows, labels = build_sst5()
    desc = build_desc_variant(rows, labels, "sst5_desc.jsonl")
    assert [r["group_id"] for r in rows] == [d["group_id"] for d in desc]
    build_amazon_desc()
    print("  label dist:", dict(sorted(Counter(r["label"] for r in rows).items())))
    print("  state_tokens: min %d avg %.1f max %d" % (
        min(r["state_tokens"] for r in rows),
        sum(r["state_tokens"] for r in rows) / len(rows),
        max(r["state_tokens"] for r in rows)))
