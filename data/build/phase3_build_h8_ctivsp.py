"""Build data/calib/h8/cti_vsp_300.jsonl from AI4Sec/cti-bench, config cti-vsp.

Per docs/briefs/CTI_VSP_BRIEF.md. The task is the analyst step
*description -> 8 CVSS v3.1 base metrics*; the score itself is a formula over
those 8 answers and is never asked, so every row here is a choice question.

License is CC-BY-NC-SA-4.0: every row carries `license_nc: true`. Measure only,
never redistributed, never in the publish repo.

Run from the repo root:
    .venv/bin/python data/build/phase3_build_h8_ctivsp.py
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "data/calib/h8/cti_vsp_300.jsonl")

DATASET, CONFIG, SPLIT = "AI4Sec/cti-bench", "cti-vsp", "test"
SEED = 20260921          # recorded in the report; changing it changes the sample
N_SAMPLE = 300
SOURCE_URL = "https://huggingface.co/datasets/AI4Sec/cti-bench"
LICENSE = "CC-BY-NC-SA-4.0 (AI4Sec/cti-bench, NeurIPS 2024 D&B)"

# The description must not already contain the answer. Only a handful of rows do.
LEAK = re.compile(r"CVSS|AV:", re.IGNORECASE)

# field -> (vector key, {code: NIST option name}, one-line NIST v3.1 definition)
METRICS = [
    ("attack_vector", "AV",
     {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"},
     "The context by which vulnerability exploitation is possible: remotely over a "
     "network, from an adjacent network, locally, or by physical access."),
    ("attack_complexity", "AC",
     {"L": "Low", "H": "High"},
     "The conditions beyond the attacker's control that must exist to exploit the "
     "vulnerability. High means a successful attack depends on conditions the "
     "attacker cannot readily produce."),
    ("privileges_required", "PR",
     {"N": "None", "L": "Low", "H": "High"},
     "The level of privileges an attacker must possess before successfully "
     "exploiting the vulnerability."),
    ("user_interaction", "UI",
     {"N": "None", "R": "Required"},
     "Whether a human user other than the attacker must participate for the "
     "vulnerability to be exploited."),
    ("scope", "S",
     {"U": "Unchanged", "C": "Changed"},
     "Whether the vulnerability can affect resources beyond the security scope of "
     "the vulnerable component."),
    ("confidentiality", "C",
     {"N": "None", "L": "Low", "H": "High"},
     "The impact to the confidentiality of the information managed by the "
     "affected component."),
    ("integrity", "I",
     {"N": "None", "L": "Low", "H": "High"},
     "The impact to the integrity, that is the trustworthiness and veracity, of "
     "the information managed by the affected component."),
    ("availability", "A",
     {"N": "None", "L": "Low", "H": "High"},
     "The impact to the availability of the affected component itself."),
]


def fetch_rows() -> list[dict]:
    """The split, from the repo's own TSV -- one request, no pagination.

    The datasets-server route was tried first and is not usable here: rows carry
    a very large `Prompt` field so a full page 502s at the gateway, and the
    smaller pages that fixes it then trip a 429 partway through (at offset 600,
    reproducibly). The repo ships the source TSV, so one download replaces fifty
    requests and removes the rate limit from the picture entirely. No parquet
    reader is needed either, which keeps this builder free of new dependencies.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, f"{CONFIG}.tsv", repo_type="dataset")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    print(f"  {len(rows)} rows from {CONFIG}.tsv")
    if not rows:
        raise SystemExit(f"{path} parsed to 0 rows")
    missing = {"URL", "Description", "GT"} - set(rows[0])
    if missing:
        raise SystemExit(f"{CONFIG}.tsv is missing columns: {sorted(missing)}")
    return rows


def parse_vector(gt: str) -> dict | None:
    """'CVSS:3.1/AV:N/AC:L/...' -> {'AV': 'N', ...}, or None if incomplete."""
    parts = dict(p.split(":", 1) for p in gt.strip().split("/")
                 if ":" in p and not p.startswith("CVSS"))
    out = {}
    for _, key, codes, _ in ((m[0], m[1], m[2], m[3]) for m in METRICS):
        code = parts.get(key)
        if code not in codes:
            return None
        out[key] = code
    return out


def main() -> int:
    if os.path.basename(os.getcwd()) and not os.path.isdir("data/calib"):
        sys.exit("run me from the repo root (data/calib must exist)")
    raw = fetch_rows()
    print(f"  {len(raw)} rows in {CONFIG}/{SPLIT}")

    leaked = [r for r in raw if LEAK.search(r.get("Description") or "")]
    clean = [r for r in raw if not LEAK.search(r.get("Description") or "")]
    print(f"  dropped {len(leaked)} rows whose Description contains CVSS/AV:")

    parsed = []
    for r in clean:
        v = parse_vector(r.get("GT") or "")
        if v:
            parsed.append((r, v))
    if len(parsed) != len(clean):
        print(f"  dropped {len(clean)-len(parsed)} rows with an unparsable GT vector")

    # Distinct descriptions, not distinct CVEs. The collector groups rows by the
    # exact state text, so two CVEs sharing a description become ONE state with
    # two copies of every question -- which it then has to split into singleton
    # groups, because a call cannot carry a repeated question id. That silently
    # turns 8-questions-per-call into 16 one-question calls for those rows and
    # changes the prompt they are scored in. Four CVEs in this split do it
    # (CVE-2024-23108/23109, CVE-2024-20252/20254), so dedupe before sampling.
    by_desc = {}
    for r, v in parsed:
        by_desc.setdefault((r.get("Description") or "").strip(), (r, v))
    unique = list(by_desc.values())
    if len(unique) != len(parsed):
        print(f"  dropped {len(parsed)-len(unique)} rows whose Description "
              f"duplicates another CVE's")

    rnd = random.Random(SEED)
    sample = rnd.sample(unique, min(N_SAMPLE, len(unique)))
    print(f"  sampled {len(sample)} of {len(unique)} with seed {SEED}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for r, vec in sample:
            cve = (r.get("URL") or "").rstrip("/").split("/")[-1] or f"row{n}"
            state = (r.get("Description") or "").strip()
            for field, key, codes, desc in METRICS:
                f.write(json.dumps({
                    "source": f"{DATASET} {CONFIG} ({SPLIT})",
                    "source_url": SOURCE_URL,
                    "license": LICENSE,
                    "license_nc": True,
                    "known_benchmark": True,
                    "state": state,
                    "lang": "en",
                    "split_hint": "TEST",
                    "label_source": "derived",
                    "group_id": cve,
                    "state_tokens": len(state.split()),
                    "id": f"ctivsp-{cve}-{field}",
                    "qtype": "choice",
                    "question_key": field,
                    "question_desc": desc,
                    "options": [codes[c] for c in sorted(codes)],
                    "label": codes[vec[key]],
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"  wrote {n} rows -> {OUT}")

    # marginals, sample against the leak-free population, for the report's §0
    print("\n  label distribution (sample 300 / all "
          f"{len(unique)}), majority class first:")
    for field, key, codes, _ in METRICS:
        from collections import Counter
        cs = Counter(codes[v[key]] for _, v in sample)
        ca = Counter(codes[v[key]] for _, v in unique)
        top, cnt = cs.most_common(1)[0]
        print(f"    {field:<21} majority {top:<10} {cnt/len(sample):.3f} "
              f"(all {ca[top]/len(parsed):.3f})  n_classes={len(cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
