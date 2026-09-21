"""Phase 3 H2 builder — small-choice multi-field datasets -> data/calib/h2/*.jsonl

Datasets:
  - NVD CVE 2.0 feed 2023 (local data/raw/nvdcve-2.0-2023.json.gz), public domain,
    300 CVEs stratified (LOW>=30), 2-3 derived fields per CVE:
      severity (P0-P3), attack_surface (REMOTE_UNAUTH/REMOTE_AUTH/LOCAL; AV:A dropped),
      cwe_category (top distinct CWEs, readable names; out-of-top -> field dropped)
    state = description + CVE-ID + published date. NO CVSS vector/score in state.
  - Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0): language==en, cap 300,
    stratified by priority, fields queue/priority/type, license_nc=true.
  - google-research-datasets/go_emotions simplified (Apache-2.0): 27->4 rule
    (anger|disgust->ANGRY, annoyance|disappointment->FRUSTRATED, neutral->NEUTRAL,
    joy|gratitude|approval|admiration->SATISFIED), strict rows only, 50/class = 200.
  - PolyAI/banking77 (CC-BY-4.0): test split, 400 rows, 77-intent options
    (also serves as H4 fallback catalog).

Run: .venv-data/bin/python phase3_build_h2.py
"""
import gzip
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
OUT = ROOT / "data" / "calib" / "h2"
H4 = ROOT / "data" / "calib" / "h4_fallback"
SEED = 42
TOK = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))

SEV_MAP = {"CRITICAL": "P0_CRITICAL", "HIGH": "P1_HIGH",
           "MEDIUM": "P2_MEDIUM", "LOW": "P3_LOW"}
SEV_OPTIONS = list(SEV_MAP.values())
AS_OPTIONS = ["REMOTE_UNAUTH", "REMOTE_AUTH", "LOCAL"]
CWE_NAMES = {  # readable preset-style names
    "CWE-79": "CWE_79_XSS", "CWE-89": "CWE_89_SQLI",
    "CWE-787": "CWE_787_OOB_WRITE", "CWE-22": "CWE_22_PATH_TRAVERSAL",
    "CWE-352": "CWE_352_CSRF", "CWE-862": "CWE_862_MISSING_AUTHZ",
    # fallback pool if a shortlist CWE lacks support (all semantically distinct):
    "CWE-434": "CWE_434_FILE_UPLOAD", "CWE-94": "CWE_94_CODE_INJECTION",
    "CWE-798": "CWE_798_HARDCODED", "CWE-287": "CWE_287_AUTH_BYPASS",
    "CWE-125": "CWE_125_OOB_READ", "CWE-416": "CWE_416_UAF",
}
GENERIC_CWE = {"CWE-16", "CWE-200", "CWE-693", "CWE-1021", "CWE-1188", "CWE-668"}


def ntok(s: str) -> int:
    return len(TOK(s, add_special_tokens=False)["input_ids"])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


def hf_load(name, config, split):
    """load_dataset with parquet fallback (datasets 5.x dropped script datasets)."""
    try:
        return load_dataset(name, config, split=split)
    except Exception as e:
        print(f"  load_dataset failed ({type(e).__name__}: {str(e)[:120]}), parquet fallback ...")
        req = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{name}/parquet",
            headers={"User-Agent": "jev-rlcd-calib/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            files = json.load(r)
        urls = files.get(config, {}).get(split, []) if isinstance(files, dict) else []
        if not urls:
            raise
        return load_dataset("parquet", data_files=urls, split="train")


# --------------------------------------------------------------------------
def build_nvd():
    print("[nvd] parsing local feed ...")
    with gzip.open(ROOT / "data" / "raw" / "nvdcve-2.0-2023.json.gz", "rt") as f:
        feed = json.load(f)
    cves = []
    drop = Counter()
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
                     "av": av, "pr": pr, "cwe": cwe})
    # dedup: normalized 60-char prefix, keep <=2 per template
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

    # cwe support across pool
    cwe_freq = Counter(c["cwe"] for c in deduped if c["cwe"])
    chosen = [cwe for cwe in CWE_NAMES
              if cwe in cwe_freq and cwe_freq[cwe] >= 15]
    chosen.sort(key=lambda c: -cwe_freq[c])
    if len(chosen) < 5:  # widen with top-freq distinct CWEs
        for cwe, n in cwe_freq.most_common():
            if len(chosen) >= 8:
                break
            if cwe in chosen or cwe in GENERIC_CWE or cwe not in CWE_NAMES:
                continue
            chosen.append(cwe)
    chosen = chosen[:8]
    cwe_options = [CWE_NAMES[c] for c in chosen]
    print(f"  pool={len(deduped)} cwe_freq_top={cwe_freq.most_common(10)}")
    print(f"  chosen CWEs: {chosen}")

    def as_label(av, pr):
        if av == "NETWORK":
            return "REMOTE_UNAUTH" if pr == "NONE" else "REMOTE_AUTH"
        return "LOCAL"  # LOCAL | PHYSICAL

    # token filter + stratified pick of 300 CVEs (LOW >= 30)
    rng = random.Random(SEED)
    pool = []
    for c in deduped:
        t = ntok(c["desc"])
        if t < 25:
            drop["too_short"] += 1
            continue
        if t > 800:
            drop["too_long"] += 1
            continue
        pool.append({**c, "tokens": t})
    rng.shuffle(pool)
    lows = [c for c in pool if c["severity"] == "LOW"][:30]
    rest_n = 300 - len(lows)
    rest = [c for c in pool if c["severity"] != "LOW"]
    counts = Counter(c["severity"] for c in rest)
    # proportional allocation across remaining severities, largest-remainder rounding
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
    print(f"  picked severity dist: {Counter(c['severity'] for c in picked)}")
    print(f"  drop stats: {dict(drop)}")

    rows = []
    for c in picked:
        state = f"{c['desc']}\n\nCVE ID: {c['id']}\nPublished: {c['published']}"
        common = {
            "source": "NVD CVE 2.0 feed 2023",
            "source_url": "https://nvd.nist.gov/feeds/json/cve/2.0/",
            "license": "public-domain (U.S. Government work, NIST NVD)",
            "state": state,
            "lang": "en",
            "split_hint": None,
            "label_source": "derived",
            "state_tokens": c["tokens"],
        }
        if c["tokens"] < 50:
            common["short_state"] = True
        rows.append({**common, "id": f"nvd2023-{c['id']}-severity",
                     "group_id": c["id"],
                     "qtype": "choice", "question_key": "severity",
                     "question_desc": "Overall severity of this vulnerability",
                     "options": SEV_OPTIONS, "label": SEV_MAP[c["severity"]]})
        rows.append({**common, "id": f"nvd2023-{c['id']}-attack_surface",
                     "group_id": c["id"],
                     "qtype": "choice", "question_key": "attack_surface",
                     "question_desc": "Most exposed attack surface",
                     "options": AS_OPTIONS, "label": as_label(c["av"], c["pr"])})
        if c["cwe"] in chosen:
            rows.append({**common, "id": f"nvd2023-{c['id']}-cwe_category",
                         "group_id": c["id"],
                         "qtype": "choice", "question_key": "cwe_category",
                         "question_desc": "Primary CWE category",
                         "options": cwe_options, "label": CWE_NAMES[c["cwe"]]})
    write_jsonl(OUT / "nvd2023.jsonl", rows)
    return {"cves": len(picked), "rows": len(rows),
            "severity": dict(Counter(r["label"] for r in rows if r["question_key"] == "severity")),
            "attack_surface": dict(Counter(r["label"] for r in rows if r["question_key"] == "attack_surface")),
            "cwe": dict(Counter(r["label"] for r in rows if r["question_key"] == "cwe_category")),
            "chosen_cwes": chosen}


# --------------------------------------------------------------------------
def build_tickets():
    print("[tickets] loading ...")
    ds = hf_load("Tobi-Bueck/customer-support-tickets", None, "train")
    en = [ex for ex in ds if ex.get("language") == "en"
          and (ex.get("body") or "").strip() and (ex.get("subject") or "").strip()]
    print(f"  total={len(ds)} en_nonempty={len(en)}")
    for field in ("queue", "priority", "type"):
        print(f"  {field}: {Counter(ex.get(field) for ex in en).most_common(12)}")
    return en, ds


def build_goemotions():
    print("[goemotions] loading ...")
    ds = hf_load("google-research-datasets/go_emotions", "simplified", "train")
    names = ds.features["labels"].feature.names
    RULE = {}
    for i, nm in enumerate(names):
        if nm in ("anger", "disgust"):
            RULE[i] = "ANGRY"
        elif nm in ("annoyance", "disappointment"):
            RULE[i] = "FRUSTRATED"
        elif nm == "neutral":
            RULE[i] = "NEUTRAL"
        elif nm in ("joy", "gratitude", "approval", "admiration"):
            RULE[i] = "SATISFIED"
    rng = random.Random(SEED)
    buckets = {k: [] for k in ("ANGRY", "FRUSTRATED", "NEUTRAL", "SATISFIED")}
    for i, ex in enumerate(ds):
        mapped = {RULE[l] for l in ex["labels"] if l in RULE}
        if not ex["labels"] or any(l not in RULE for l in ex["labels"]) or len(mapped) != 1:
            continue  # strict: unmapped emotion present OR multi-class conflict
        buckets[mapped.pop()].append((i, ex))
    rows = []
    for cls, items in buckets.items():
        rng.shuffle(items)
        kept = 0
        for i, ex in items:
            if kept >= 50:
                break
            t = ntok(ex["text"])
            if t < 10:
                continue
            kept += 1
            rows.append({
                "id": f"goemotions-{i}",
                "group_id": f"goemotions-{i}",
                "source": "google-research-datasets/go_emotions",
                "source_url": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
                "license": "Apache-2.0",
                "state": ex["text"],
                "qtype": "choice",
                "question_key": "sentiment",
                "question_desc": "Emotional state of the author (rule-collapsed from GoEmotions)",
                "options": ["ANGRY", "FRUSTRATED", "NEUTRAL", "SATISFIED"],
                "label": cls,
                "label_source": "derived",
                "lang": "en",
                "split_hint": "train",
                "known_benchmark": True,
                "stratified": True,
                "short_state": True,  # Reddit comments, mostly <50 tokens
                "state_tokens": t,
            })
    write_jsonl(OUT / "goemotions_sentiment.jsonl", rows)
    return {"kept": len(rows), "bucket_sizes": {k: len(v) for k, v in buckets.items()}}


def build_banking77():
    print("[banking77] loading ...")
    # HF repo is script-only; load the original CSVs from PolyAI-LDN/task-specific-datasets
    import csv as _csv
    import io
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    test_csv = raw_dir / "banking77_test.csv"
    if not test_csv.exists():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            "master/banking_data/test.csv", test_csv)
    names_doc = json.loads(urllib.request.urlopen(
        "https://huggingface.co/datasets/PolyAI/banking77/raw/main/dataset_infos.json",
        timeout=60).read())
    names = names_doc["default"]["features"]["label"]["names"]
    with open(test_csv, newline="", encoding="utf-8") as f:
        data = [(r["text"], r["category"]) for r in _csv.DictReader(f)]
    assert set(l for _, l in data) <= set(names), "CSV categories differ from HF names"
    rng = random.Random(SEED)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= 400:
            break
        text, lab = data[i]
        t = ntok(text)
        if t < 5:  # degenerate 1-4 token queries
            continue
        rows.append({
            "id": f"banking77-test-{i:05d}",
            "group_id": f"banking77-test-{i:05d}",
            "source": "PolyAI/banking77",
            "source_url": "https://huggingface.co/datasets/PolyAI/banking77",
            "license": "CC-BY-4.0",
            "state": text,
            "qtype": "choice",
            "question_key": "intent",
            "question_desc": "Banking customer intent",
            "options": names,
            "label": lab,
            "label_source": "human",
            "lang": "en",
            "split_hint": "test",
            "known_benchmark": True,
            "short_state": True,  # short queries
            "state_tokens": t,
        })
    write_jsonl(OUT / "banking77.jsonl", rows)
    H4.mkdir(parents=True, exist_ok=True)
    write_jsonl(H4 / "banking77_catalog.json", [{"intent": n} for n in names])
    return {"kept": len(rows), "n_options": len(names)}


if __name__ == "__main__":
    stats = {"nvd": build_nvd(), "goemotions": build_goemotions(),
             "banking77": build_banking77()}
    # tickets: distribution printed first; finalize mapping in same run
    en, ds = build_tickets()
    json.dump({"stats": stats, "tickets_en": len(en)},
              open(OUT / "_run_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2, default=str)[:3000])
