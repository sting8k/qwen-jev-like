#!/usr/bin/env python3
"""Read-only audit of data/calib JSONL against docs/briefs/PHASE3_DATA_BRIEF.md §1.

Usage: python3 phase3_audit_calib.py [--all]
  default: only files the collector actually reads (skips _samples/, _*, and
           <name>.jsonl superseded by <name>_v2.jsonl -- same rule as
           run_calib_collect.py:391-395)
  --all:   include superseded files too

Never writes to data/. Exit 1 if any ACTIVE file breaks a hard requirement.
"""
import json
import os
import sys
from collections import Counter

BASE = "data/calib"   # relative to the CWD: run from the repo root
assert os.path.isdir(BASE), (
    f"run this from the repo root: {BASE!r} not found from {os.getcwd()!r} "
    "(os.walk on a missing dir reports 0 files instead of failing)")
# §1 required on every row, all qtypes
REQUIRED = ["id", "source", "source_url", "license", "state", "qtype",
            "question_key", "question_desc", "label", "label_source",
            "lang", "split_hint"]
LABEL_SOURCES = {"human", "derived", "textualized", "simulated"}
ABSTRACT = {"A", "B", "C", "D", "E"}


def collect_files(include_superseded=False):
    files = []
    for dp, _, fns in os.walk(BASE):
        if "_samples" in dp:
            continue
        for f in sorted(fns):
            if not f.endswith(".jsonl") or f.startswith("_"):
                continue
            p = os.path.join(dp, f)
            if not include_superseded and os.path.exists(p[:-6] + "_v2.jsonl"):
                continue
            files.append(p)
    return sorted(files)


def audit(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    rep = {"path": path, "n": n, "hard": [], "soft": []}

    missing = Counter()
    for r in rows:
        for k in REQUIRED:
            if k not in r or r[k] in (None, ""):
                missing[k] += 1
        # qtype-conditional fields
        if r.get("qtype") == "choice" and not r.get("options"):
            missing["options"] += 1
        if r.get("qtype") == "score" and not r.get("scale"):
            missing["scale"] += 1
    for k, c in sorted(missing.items()):
        rep["hard"].append(f"missing `{k}` on {c}/{n} rows")

    # label inside options / scale
    bad_label = 0
    for r in rows:
        if r.get("qtype") == "choice" and r.get("options"):
            if r.get("label") not in r["options"]:
                bad_label += 1
        elif r.get("qtype") == "score" and r.get("scale"):
            lo, hi = r["scale"]["min"], r["scale"]["max"]
            try:
                if not (lo <= float(r["label"]) <= hi):
                    bad_label += 1
            except (TypeError, ValueError):
                bad_label += 1
    if bad_label:
        rep["hard"].append(f"label outside options/scale on {bad_label}/{n} rows")

    # semantic rule (Biscuit 2026-09-18): abstract-label choice needs the
    # mapping spelled out in the state, otherwise the question is unanswerable
    unmapped = 0
    for r in rows:
        if r.get("qtype") == "choice" and r.get("options") and set(r["options"]) <= ABSTRACT:
            if any(f"{o}. " not in r.get("state", "") for o in r["options"]):
                unmapped += 1
    if unmapped:
        rep["hard"].append(f"abstract-label options without A->text mapping in state: {unmapped}/{n} rows")

    bad_src = Counter(r.get("label_source") for r in rows
                      if r.get("label_source") not in LABEL_SOURCES)
    for v, c in bad_src.items():
        rep["hard"].append(f"label_source `{v}` not in {sorted(LABEL_SOURCES)} on {c} rows")

    # flags: present-on-every-row or absent-on-every-row; partial = ambiguous
    for flag in ("known_benchmark", "short_state", "license_nc"):
        have = sum(1 for r in rows if flag in r)
        if 0 < have < n:
            rep["soft"].append(f"`{flag}` on only {have}/{n} rows (partial)")

    rep["license"] = sorted(set(r.get("license", "?") for r in rows))
    rep["label_source"] = sorted(set(str(r.get("label_source")) for r in rows))
    rep["qtype"] = sorted(set(str(r.get("qtype")) for r in rows))
    rep["known_benchmark"] = sorted(set(str(r.get("known_benchmark", "ABSENT")) for r in rows))
    rep["license_nc"] = sorted(set(str(r.get("license_nc", "ABSENT")) for r in rows))
    rep["probe"] = sorted(set(str(r.get("probe", "ABSENT")) for r in rows))
    rep["lang"] = sorted(set(str(r.get("lang")) for r in rows))
    return rep


def main():
    files = collect_files("--all" in sys.argv)
    fails = 0
    print(f"{'file':<40} {'n':>5}  {'qtype':<8} {'label_source':<14} {'known_bm':<10} lic")
    print("-" * 110)
    reps = []
    for p in files:
        r = audit(p)
        reps.append(r)
        print(f"{p[len(BASE)+1:]:<40} {r['n']:>5}  {','.join(r['qtype']):<8} "
              f"{','.join(r['label_source']):<14} {','.join(r['known_benchmark']):<10} "
              f"{','.join(r['license'])}")
    print()
    for r in reps:
        if r["hard"] or r["soft"]:
            print(f"## {r['path']}")
            for h in r["hard"]:
                print(f"   FAIL {h}")
                fails += 1
            for s in r["soft"]:
                print(f"   warn {s}")
    print(f"\nfiles={len(files)} rows={sum(r['n'] for r in reps)} hard_failures={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
