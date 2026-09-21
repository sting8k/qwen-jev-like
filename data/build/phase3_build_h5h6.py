"""Phase 3 H5+H6 builder.

H5  data/calib/h5/paysim_textualized.jsonl — 400 rows (200 fraud / 200 legit,
    stratified from ~0.13% base rate), deterministic template over
    type/amount/4 balances/step(hour), label isFraud -> noul is_fraud.
    label_source: textualized.
H6  data/calib/h6/content_free.jsonl       — 50 real questions x 3 state variants
    data/calib/h6/permutation_arc.jsonl    — ARC-Challenge 100 questions x 4 option
                                              permutations, option text verbatim
    data/calib/h6/label_name_arc.jsonl     — same 100 questions, options A/B/C/D
    data/calib/h6/vi_boolq.jsonl           — 50 BoolQ state+question machine-translated
                                              (skips if vi_translations.json absent)
    All H6 rows carry probe: true (excluded from FIT/SELECT/TEST — diagnostics only).

Run: .venv-data/bin/python phase3_build_h5h6.py
"""
import csv
import io
import itertools
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# repo root: this file lives in data/build/, so two levels up. Keep in sync
# with the directory depth -- ROOT is what every output path hangs off.
ROOT = Path(__file__).resolve().parents[2]
SEED = 42
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


# ------------------------------------------------------------------ H5 PaySim
PAYSIM_TEMPLATE = (
    "Transaction Review: Mobile money {ttype} of {amount:.2f}. "
    "Origin account balance {old_org:.2f} -> {new_org:.2f}. "
    "Destination account balance {old_dst:.2f} -> {new_dst:.2f}. "
    "Time: hour {hour:02d}:00 of day {day}."
)


def paysim_state(row):
    step = int(row["step"])
    return PAYSIM_TEMPLATE.format(
        ttype=row["type"], amount=float(row["amount"]),
        old_org=float(row["oldbalanceOrg"]), new_org=float(row["newbalanceOrig"]),
        old_dst=float(row["oldbalanceDest"]), new_dst=float(row["newbalanceDest"]),
        hour=step % 24, day=step // 24 + 1)


def build_paysim():
    print("[paysim] streaming zip ...")
    zp = ROOT / "data/raw/paysim1/paysim1.zip"
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            rdr = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            fraud, legit = [], []
            n_legit_seen = 0
            rng_r = random.Random(SEED)  # reservoir for legit
            for row in rdr:
                keep = {"type": row["type"], "amount": row["amount"],
                        "oldbalanceOrg": row["oldbalanceOrg"],
                        "newbalanceOrig": row["newbalanceOrig"],
                        "oldbalanceDest": row["oldbalanceDest"],
                        "newbalanceDest": row["newbalanceDest"],
                        "step": row["step"], "isFraud": row["isFraud"]}
                if row["isFraud"] == "1":
                    fraud.append(keep)
                else:
                    n_legit_seen += 1
                    if len(legit) < 5000:
                        legit.append(keep)
                    else:
                        j = rng_r.randrange(n_legit_seen)
                        if j < 5000:
                            legit[j] = keep
    print(f"  fraud={len(fraud)} legit_reservoir=5000 (of {n_legit_seen})")
    rng = random.Random(SEED)
    picked = rng.sample(fraud, 200) + rng.sample(legit, 200)
    rng.shuffle(picked)
    rows = []
    for i, row in enumerate(picked):
        state = paysim_state(row)
        rows.append({
            "id": f"paysim-{i:04d}",
            "group_id": f"paysim-{i:04d}",
            "source": "Kaggle ealaxi/paysim1 (PaySim simulator)",
            "source_url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
            "license": "CC-BY-SA-4.0",
            "state": state,
            "qtype": "noul",
            "question_key": "is_fraud",
            "question_desc": "Was this transaction fraudulent (simulator ground truth)",
            "label": row["isFraud"] == "1",
            "label_source": "simulated",  # isFraud assigned by simulator rules
            "lang": "en",
            "split_hint": None,
            "stratified": True,
            "state_tokens": ntok(state),
        })
    write_jsonl(ROOT / "data/calib/h5/paysim_textualized.jsonl", rows)
    return {"rows": len(rows), "label_dist": dict(Counter(r["label"] for r in rows)),
            "base_rate": f"{len(fraud)}/{len(fraud)+n_legit_seen}"}


# ------------------------------------------------------------------ H6 probes
def build_content_free():
    print("[content_free] ...")
    src = [json.loads(l) for l in open(ROOT / "data/calib/h2/banking77.jsonl")]
    rng = random.Random(SEED)
    qs = rng.sample(src, 50)
    variants = {"na": "N/A", "empty": "", "lorem":
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
                "eiusmod tempor incididunt ut labore et dolore magna aliqua."}
    rows = []
    for q in qs:
        for vname, vstate in variants.items():
            rows.append({
                "id": f"cf-{q['id']}-{vname}",
                "group_id": q["group_id"],
                "source": q["source"], "source_url": q["source_url"],
                "license": q["license"],
                "state": vstate,
                "qtype": "choice",
                "question_key": "intent",
                "question_desc": "Banking customer intent",
                "options": q["options"],
                "label": q["label"],  # human label of the real question (unused for fitting)
                "label_source": q["label_source"],
                "lang": "en", "split_hint": None,
                "probe": True, "probe_kind": "content_free", "variant": vname,
                "state_tokens": ntok(vstate) if vstate else 0,
            })
    write_jsonl(ROOT / "data/calib/h6/content_free.jsonl", rows)
    return {"rows": len(rows)}


def _arc_pool():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(SEED)
    pool = [ex for ex in ds if len(ex["choices"]["text"]) == 4]
    return rng.sample(pool, 100)  # deterministic: same 100 questions for both probes


def build_arc_probes():
    print("[arc] permutation ...")
    qs = _arc_pool()
    perm_rows = []
    # fixed distinct permutations: identity, reverse, and two deterministic shuffles
    perm_sel = [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)]
    for qi, ex in enumerate(qs):
        texts = ex["choices"]["text"]
        letters = ex["choices"]["label"]
        ans_idx = letters.index(ex["answerKey"])
        ans_text = texts[ans_idx]
        gid = f"arc-{qi:03d}"
        for pid, perm in enumerate(perm_sel):
            opts = [texts[i] for i in perm]
            perm_rows.append({
                "id": f"{gid}-p{pid}",
                "group_id": gid,
                "source": "allenai/ai2_arc (ARC-Challenge test)",
                "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
                "license": "CC-BY-SA-4.0",
                "state": ex["question"],
                "qtype": "choice",
                "question_key": "answer",
                "question_desc": "Multiple-choice science question; pick the correct answer text",
                "options": opts,
                "label": ans_text,
                "label_source": "human",
                "lang": "en", "split_hint": "test",
                "known_benchmark": True,
                "probe": True, "probe_kind": "permutation", "perm_id": pid,
                "state_tokens": ntok(ex["question"]),
            })
    write_jsonl(ROOT / "data/calib/h6/permutation_arc.jsonl", perm_rows)
    return {"permutation": len(perm_rows)}


def build_label_name_v2():
    """State carries the full A->content mapping, in the same order as
    permutation perm_id=0."""
    print("[arc] label_name v2 ...")
    qs = _arc_pool()
    rows = []
    for qi, ex in enumerate(qs):
        texts = ex["choices"]["text"]
        ans_idx = ex["choices"]["label"].index(ex["answerKey"])
        mapping = "\n".join(f"{L}. {t}" for L, t in zip("ABCD", texts))
        state = f"{ex['question']}\n\n{mapping}"
        rows.append({
            "id": f"arc-{qi:03d}-letters",
            "group_id": f"arc-{qi:03d}",
            "source": "allenai/ai2_arc (ARC-Challenge test)",
            "source_url": "https://huggingface.co/datasets/allenai/ai2_arc",
            "license": "CC-BY-SA-4.0",
            "state": state,
            "qtype": "choice",
            "question_key": "answer",
            "question_desc": "Multiple-choice science question; pick the LETTER of the correct "
                             "answer using the A-D mapping given in the state",
            "options": ["A", "B", "C", "D"],
            "label": "ABCD"[ans_idx],
            "label_source": "human",
            "lang": "en", "split_hint": "test",
            "known_benchmark": True,
            "probe": True, "probe_kind": "label_name",
            "state_tokens": ntok(state),
        })
    write_jsonl(ROOT / "data/calib/h6/label_name_arc_v2.jsonl", rows)
    return {"label_name_v2": len(rows)}


def build_vi():
    tr_path = ROOT / "data/calib/h6/vi_translations.json"
    if not tr_path.exists():
        boolq = [json.loads(l) for l in open(ROOT / "data/calib/h1/boolq.jsonl")]
        boolq.sort(key=lambda r: r["state_tokens"])
        cand = boolq[:50]
        todo = [{"id": r["id"], "state": r["state"], "question_desc": r["question_desc"]}
                for r in cand]
        (ROOT / "data/calib/h6").mkdir(parents=True, exist_ok=True)
        write_jsonl(ROOT / "data/calib/h6/_vi_todo.jsonl", todo)
        return {"status": "pending_translation", "todo_file": str(ROOT / "data/calib/h6/_vi_todo.jsonl")}
    tr = {t["id"]: t for t in json.load(open(tr_path))}
    boolq = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/calib/h1/boolq.jsonl")}
    rows = []
    for tid, t in tr.items():
        src = boolq[tid]
        rows.append({
            "id": f"vi-{tid}",
            "group_id": tid,  # links back to the English BoolQ row
            "source": "google/boolq (machine-translated state+question)",
            "source_url": src["source_url"],
            "license": "CC-BY-SA-3.0",
            "state": t["state_vi"],
            "qtype": "noul",
            "question_key": "passage_question",
            "question_desc": t["question_vi"],
            "label": src["label"],
            "label_source": "human",
            "lang": "vi",
            "split_hint": None,
            "known_benchmark": True,
            "probe": True, "probe_kind": "language_shift",
            "machine_translated": True,
            "state_tokens": ntok(t["state_vi"]),
        })
    write_jsonl(ROOT / "data/calib/h6/vi_boolq.jsonl", rows)
    return {"rows": len(rows)}


if __name__ == "__main__":
    stats = {"paysim": build_paysim(), "content_free": build_content_free(),
             "arc": build_arc_probes(), "vi": build_vi()}
    print(json.dumps(stats, indent=2, default=str)[:1500])
