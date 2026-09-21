"""Phase 3 H3+H4 builder — Biscuit sign-off 2026-09-18 19:34.

Outputs (NEW files only — previously submitted JSONL are frozen on disk):
  data/calib/h2/nvd2023_v2.jsonl      — 300 CVE x 4 rows (3 choice + 1 score cvss_base_score
                                        scale 0..9, label=min(round(baseScore),9))
  data/calib/h3/amazon_reviews_en.jsonl — 500 rows (100/star), body only, 30-400 words,
                                        score scale 1..5
  data/calib/h4_fallback/clinc150.jsonl — 300 rows test split, oos dropped, 150-intent options
  data/calib/h4_fallback/banking77_catalog.json — regenerated, additive 'desc' field
                                        (deterministic snake_case->spaces; intent verbatim)

Run: .venv-data/bin/python phase3_build_h3h4.py
"""
import csv
import gzip
import io
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# repo root: this file lives in data/build/, so two levels up. Keep in sync
# with the directory depth -- ROOT is what every output path hangs off.
ROOT = Path(__file__).resolve().parents[2]
SEED = 42
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))

SEV_MAP = {"CRITICAL": "P0_CRITICAL", "HIGH": "P1_HIGH",
           "MEDIUM": "P2_MEDIUM", "LOW": "P3_LOW"}
SEV_OPTIONS = list(SEV_MAP.values())
AS_OPTIONS = ["REMOTE_UNAUTH", "REMOTE_AUTH", "LOCAL"]
CWE_NAMES = {
    "CWE-79": "CWE_79_XSS", "CWE-89": "CWE_89_SQLI",
    "CWE-787": "CWE_787_OOB_WRITE", "CWE-22": "CWE_22_PATH_TRAVERSAL",
    "CWE-352": "CWE_352_CSRF", "CWE-862": "CWE_862_MISSING_AUTHZ",
    "CWE-434": "CWE_434_FILE_UPLOAD", "CWE-94": "CWE_94_CODE_INJECTION",
    "CWE-798": "CWE_798_HARDCODED", "CWE-287": "CWE_287_AUTH_BYPASS",
    "CWE-125": "CWE_125_OOB_READ", "CWE-416": "CWE_416_UAF",
}
GENERIC_CWE = {"CWE-16", "CWE-200", "CWE-693", "CWE-1021", "CWE-1188", "CWE-668"}


def ntok(s):
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------- NVD v2
def build_nvd_v2():
    print("[nvd_v2] parsing local feed ...")
    with gzip.open(ROOT / "data" / "raw" / "nvdcve-2.0-2023.json.gz", "rt") as f:
        feed = json.load(f)
    cves, drop = [], Counter()
    cvss_leak = re.compile(r"CVSS[\s:v]*\d|AV:[NALP]/|base score", re.I)
    for item in feed["vulnerabilities"]:
        cve = item["cve"]
        desc = next((d["value"] for d in cve["descriptions"]
                     if d["lang"] == "en" and d["value"].strip()), None)
        if not desc:
            drop["no_desc"] += 1
            continue
        if cvss_leak.search(desc):
            drop["cvss_in_desc"] += 1
            continue
        mets = [m for m in cve.get("metrics", {}).get("cvssMetricV31", [])
                if m["cvssData"].get("baseSeverity")]
        if not mets:
            drop["no_cvss31"] += 1
            continue
        cvss = next((m["cvssData"] for m in mets if m.get("type") == "Primary"),
                    mets[0]["cvssData"])
        av, pr = cvss.get("attackVector"), cvss.get("privilegesRequired")
        if av == "ADJACENT_NETWORK":
            drop["av_adjacent"] += 1
            continue
        cwe = None
        for w in cve.get("weaknesses", []):
            for d in w.get("description", []):
                m = re.match(r"(CWE-\d+)$", d["value"].strip())
                if m:
                    cwe = m.group(1)
                    break
            if cwe:
                break
        cves.append({"id": cve["id"], "published": cve["published"][:10],
                     "desc": desc, "severity": cvss["baseSeverity"],
                     "av": av, "pr": pr, "cwe": cwe,
                     "base_score": float(cvss.get("baseScore"))})

    def norm_key(d):
        d = d.lower()
        d = re.sub(r"\bv?\d+(\.\d+)+\b", "VER", d)
        return d[:60]
    seen = Counter()
    deduped = []
    for c in cves:
        k = norm_key(c["desc"])
        if seen[k] >= 2:
            continue
        seen[k] += 1
        deduped.append(c)
    drop["dup_template"] = len(cves) - len(deduped)

    cwe_freq = Counter(c["cwe"] for c in deduped if c["cwe"])
    chosen = sorted([c for c in CWE_NAMES if c in cwe_freq and cwe_freq[c] >= 15],
                    key=lambda c: -cwe_freq[c])[:8]
    cwe_options = [CWE_NAMES[c] for c in chosen]

    rng = random.Random(SEED)
    pool = [c for c in deduped if 25 <= ntok(c["desc"]) <= 800]
    rng.shuffle(pool)
    lows = [c for c in pool if c["severity"] == "LOW"][:30]
    rest_n = 300 - len(lows)
    rest = [c for c in pool if c["severity"] != "LOW"]
    counts = Counter(c["severity"] for c in rest)
    total_rest = len(rest)
    alloc = {k: int(rest_n * n / total_rest) for k, n in counts.items()}
    while sum(alloc.values()) < rest_n:
        k = max(alloc, key=lambda k: rest_n * counts[k] / total_rest - alloc[k])
        alloc[k] += 1
    rng_rest = random.Random(SEED + 1)
    picked = lows[:]
    for k, n in alloc.items():
        ks = [c for c in rest if c["severity"] == k]
        rng_rest.shuffle(ks)
        picked.extend(ks[:n])
    rng.shuffle(picked)
    print(f"  severity dist: {Counter(c['severity'] for c in picked)}")

    def as_label(av, pr):
        if av == "NETWORK":
            return "REMOTE_UNAUTH" if pr == "NONE" else "REMOTE_AUTH"
        return "LOCAL"

    rows = []
    for c in picked:
        state = f"{c['desc']}\n\nCVE ID: {c['id']}\nPublished: {c['published']}"
        t = ntok(c["desc"])
        common = {
            "source": "NVD CVE 2.0 feed 2023",
            "source_url": "https://nvd.nist.gov/feeds/json/cve/2.0/",
            "license": "public-domain (U.S. Government work, NIST NVD)",
            "state": state,
            "lang": "en",
            "split_hint": None,
            "label_source": "derived",
            "group_id": c["id"],
            "state_tokens": t,
        }
        if t < 50:
            common["short_state"] = True
        rows.append({**common, "id": f"nvd2023-{c['id']}-severity",
                     "qtype": "choice", "question_key": "severity",
                     "question_desc": "Overall severity of this vulnerability",
                     "options": SEV_OPTIONS, "label": SEV_MAP[c["severity"]]})
        rows.append({**common, "id": f"nvd2023-{c['id']}-attack_surface",
                     "qtype": "choice", "question_key": "attack_surface",
                     "question_desc": "Most exposed attack surface",
                     "options": AS_OPTIONS, "label": as_label(c["av"], c["pr"])})
        if c["cwe"] in chosen:
            rows.append({**common, "id": f"nvd2023-{c['id']}-cwe_category",
                         "qtype": "choice", "question_key": "cwe_category",
                         "question_desc": "Primary CWE category",
                         "options": cwe_options, "label": CWE_NAMES[c["cwe"]]})
        rows.append({**common, "id": f"nvd2023-{c['id']}-base_score",
                     "qtype": "score", "question_key": "cvss_base_score",
                     "question_desc": "Overall CVSS v3.1 base score "
                                      "(0 = none; 9 = 8.5-10.0 critical band)",
                     "scale": {"min": 0, "max": 9},
                     "label": min(round(c["base_score"]), 9)})
    write_jsonl(ROOT / "data/calib/h2/nvd2023_v2.jsonl", rows)
    return {"cves": len(picked), "rows": len(rows),
            "score_dist": dict(Counter(min(round(c["base_score"]), 9) for c in picked)),
            "chosen_cwes": chosen}


# ---------------------------------------------------------------- Amazon H3
def build_amazon():
    print("[amazon_en] loading ...")
    ds = load_dataset("SetFit/amazon_reviews_multi_en", split="test")  # 5000
    rng = random.Random(SEED)
    buckets = {s: [] for s in range(1, 6)}
    seen = set()
    for ex in ds:
        text = (ex.get("text") or "").strip()
        label = int(ex["label"]) + 1  # 0-4 -> stars 1-5
        words = len(text.split())
        if not (30 <= words <= 400):  # Biscuit: body only, 30-400 words
            continue
        if text in seen:
            continue
        seen.add(text)
        buckets[label].append((ex["id"], text))
    rows = []
    for stars, items in buckets.items():
        rng.shuffle(items)
        for rid, text in items[:100]:
            t = ntok(text)
            r = {
                "id": f"amazon-en-{rid}",
                "group_id": f"amazon-en-{rid}",
                "source": "SetFit/amazon_reviews_multi_en",
                "source_url": "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en",
                "license": "Apache-2.0",
                "state": text,
                "qtype": "score",
                "question_key": "stars",
                "question_desc": "Star rating the reviewer gave, 1 = worst, 5 = best",
                "scale": {"min": 1, "max": 5},
                "label": stars,
                "label_source": "human",
                "lang": "en",
                "split_hint": "test",
                "known_benchmark": True,
                "stratified": True,
                "state_tokens": t,
            }
            if t < 50:
                r["short_state"] = True
            rows.append(r)
    write_jsonl(ROOT / "data/calib/h3/amazon_reviews_en.jsonl", rows)
    return {"kept": len(rows), "bucket_pool": {k: len(v) for k, v in buckets.items()}}


# ---------------------------------------------------------------- CLINC H4
def build_clinc():
    print("[clinc] loading ...")
    ds = load_dataset("clinc/clinc_oos", "imbalanced", split="test")
    names = ds.features["intent"].names
    oos_idx = names.index("oos") if "oos" in names else None
    intents = [n for n in names if n != "oos"]
    rng = random.Random(SEED)
    rows_pool = [(i, ex) for i, ex in enumerate(ds)
                 if oos_idx is None or ex["intent"] != oos_idx]
    rng.shuffle(rows_pool)
    rows = []
    for i, ex in rows_pool:
        if len(rows) >= 300:
            break
        t = ntok(ex["text"])
        if t < 5:  # degenerate 1-4 token utterances
            continue
        rows.append({
            "id": f"clinc-test-{i:05d}",
            "group_id": f"clinc-test-{i:05d}",
            "source": "clinc/clinc_oos",
            "source_url": "https://huggingface.co/datasets/clinc/clinc_oos",
            "license": "CC-BY-3.0",
            "state": ex["text"],
            "qtype": "choice",
            "question_key": "intent",
            "question_desc": "Customer intent for the utterance",
            "options": intents,
            "label": names[ex["intent"]],
            "label_source": "human",
            "lang": "en",
            "split_hint": "test",
            "known_benchmark": True,
            "short_state": True,
            "state_tokens": t,
        })
    write_jsonl(ROOT / "data/calib/h4_fallback/clinc150.jsonl", rows)
    # catalog with deterministic desc (snake_case -> spaces), intent verbatim
    cat = [{"intent": n, "desc": n.replace("_", " ")} for n in intents]
    write_jsonl(ROOT / "data/calib/h4_fallback/clinc150_catalog.json", cat)
    return {"kept": len(rows), "n_options": len(intents),
            "label_dist_top": Counter(r["label"] for r in rows).most_common(5)}


# ---------------------------------------------------------------- banking77 catalog v2
def rebuild_banking_catalog():
    raw = json.loads(urllib.request.urlopen(
        "https://huggingface.co/datasets/PolyAI/banking77/raw/main/dataset_infos.json",
        timeout=60).read())
    names = raw["default"]["features"]["label"]["names"]
    cat = [{"intent": n, "desc": n.replace("_", " ")} for n in names]
    write_jsonl(ROOT / "data/calib/h4_fallback/banking77_catalog.json", cat)
    return {"n": len(cat)}


if __name__ == "__main__":
    stats = {"nvd_v2": build_nvd_v2(), "amazon": build_amazon(),
             "clinc": build_clinc(), "banking_catalog": rebuild_banking_catalog()}
    print(json.dumps(stats, indent=2, default=str)[:2500])
