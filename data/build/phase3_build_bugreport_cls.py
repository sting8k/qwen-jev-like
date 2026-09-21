"""Bug-report classification builder — HackerOne disclosed reports -> data/raw/bugreport_cls/*.jsonl

Contract: docs/PROTOCOL.md (bug-report classification).
Biscuit approved 2026-09-22; user approved the same day at **n=300** (not 600).
Survey evidence: the dataset survey in the research repository.

What the label is: `substate` is the decision a real triager recorded on this
report. Not derived, not LLM. Severity is only a triager decision when
`severity.author_type == "Team"` -- otherwise it is the reporter grading their
own bug, so those rows get no severity question.

Licence posture (§4.6): the report text is copyright of the finder. This set is
MEASURE-ONLY. The JSONL lands in data/raw/bugreport_cls/ which is gitignored; what gets
committed is this builder plus the manifest of report ids (ids and labels are
facts about public records, not the creative text).
robots.txt has no Disallow and a keyword scan of the General T&C and /terms
found no anti-scraping clause, but no acceptable-use page was reachable:
**"không thấy cấm, chưa xác nhận được phép"**.

Run (no GPU, ~2 min at REQ_SLEEP=0.2):  .venv-data/bin/python data/build/phase3_build_bugreport_cls.py
Re-runs reuse the on-disk report cache, so the fetch happens once.
"""
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

# repo root: this file lives in data/build/, so two levels up (same as the
# phase3_build_* builders -- moving this file means fixing this line).
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "bugreport_cls"          # gitignored: cache + output JSONL
CACHE = RAW / "reports"
OUT = RAW / "bugreport_cls_300.jsonl"
MANIFEST = ROOT / "data" / "build" / "bugreport_cls_manifest.json"  # committed

SEED = 42
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
GQL = "https://hackerone.com/graphql"

# §4.2 -- 300 rows, 41.7 % noise. `duplicate` is excluded on purpose (§4.3): a
# duplicate verdict needs the rest of the queue, which the model never sees.
TARGET = {"spam": 10, "not-applicable": 50, "informative": 65, "resolved": 175}
PROGRAM_CAP = 24            # 8 % of 300 -- curl alone was 43 % of N-A in the survey
MAX_WORDS = 1500            # §4.5; only ~2 % of surveyed rows exceed this
FULL_ENUM_MAX = 2000        # pools at or under this are read whole (see collect_cards)
REQ_SLEEP = 0.2             # seconds between requests (user set 2026-09-22)
SEVERITY_OPTS = ["none", "low", "medium", "high", "critical"]

# §4.3 -- FIXED weakness->family map, written BEFORE the run and printed with the
# result. First match wins. Collapsing H1's long tail (45 distinct names in 118
# labelled survey rows; top-25 only reached 83 %) is a relabel of a human label,
# so label_source stays "human". Do not touch this table after seeing numbers.
# Finalised against the 45 weakness NAMES in the survey cache (vocabulary, not
# outcomes) before the build ran. `other` deliberately keeps real small classes
# such as Open Redirect and UI Redressing rather than forcing them into a family.
WEAKNESS_FAMILIES = [
    ("xss", ["cross-site scripting", "xss"]),
    ("csrf", ["cross-site request forgery", "csrf"]),
    ("ssrf", ["server-side request forgery", "ssrf"]),
    ("injection", ["sql injection", "command injection", "code injection", "injection",
                   "xml external entity", "xxe", "deserializ", "path traversal",
                   "directory traversal", "template injection", "request smuggling"]),
    ("memory_safety", ["buffer overflow", "buffer over-read", "buffer over-write",
                       "memory corruption", "use after free", "integer overflow",
                       "integer underflow", "out-of-bounds", "double free",
                       "null pointer", "null termination", "stack overflow", "heap",
                       "type confusion", "array index", "format string"]),
    ("authn", ["authentication", "2fa", "mfa", "session fixation", "credential",
               "brute force", "password", "session expiration"]),
    ("access_control", ["access control", "authorization", "privilege",
                        "insecure direct object", "idor", "permission"]),
    ("info_disclosure", ["information disclosure", "information exposure",
                         "cleartext storage", "cleartext transmission",
                         "privacy violation", "directory listing",
                         "exposure of sensitive", "sensitive data"]),
    ("dos", ["denial of service", "resource consumption", "resource exhaustion",
             "rate limit", "allocation of resources"]),
    ("crypto", ["cryptograph", "ssl", "tls", "certificate", "randomness", "hash",
                "encryption", "cipher"]),
    ("business_logic", ["business logic", "violation of secure design",
                        "improper input validation", "race condition",
                        "insufficient validation"]),
]
FAMILY_OTHER = "other"


def program_mentioned(program: str, state: str) -> tuple:
    """Does the report text name its own program? Returns (strict, substring).

    §4.9(a): the program handle is never put into the state, but the prose names
    it anyway in about half the rows, and knowing the program is worth +15 points
    on outcome. Both rules are emitted because the survey number (164/300) was
    computed with the loose one:
      substring -- handle appears anywhere, lowercased. What §4.9(a) reported.
      strict    -- word-boundary AND handle >=3 chars. A 1-char handle ("x")
                   matches almost any text, so it is not evidence of anything.
    Measured on this set: substring 164, word-boundary 159, strict 155. The
    feared shell-command false positive does not occur -- 0 of the 24 `curl`
    rows have "curl" only as a shell invocation.
    """
    if not program:
        return (False, False)
    h, low = program.lower(), state.lower()
    substr = h in low
    strict = len(h) >= 3 and re.search(r"\b" + re.escape(h) + r"\b", low) is not None
    return (strict, substr)


def family_of(name: str) -> str:
    low = (name or "").lower()
    for fam, keys in WEAKNESS_FAMILIES:
        if any(k in low for k in keys):
            return fam
    return FAMILY_OTHER


SEARCH = """query HacktivitySearchQuery($queryString: String!, $from: Int, $size: Int, $sort: SortInput!) {
  search(index: CompleteHacktivityReportIndex, query_string: $queryString, from: $from, size: $size, sort: $sort) {
    total_count
    nodes { ... on HacktivityDocument { _id severity_rating cwe submitted_at
            report { title substate disclosed_at } team { handle } } }
  }
}"""


def post(query, variables, tries=3):
    body = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(GQL, data=body, headers={
            "Content-Type": "application/json", "User-Agent": UA,
            "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read())
            if "errors" in d:
                raise RuntimeError(d["errors"][0].get("message", "graphql error"))
            return d
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def search_page(qs, frm, size):
    d = post(SEARCH, {"queryString": qs, "from": frm, "size": size,
                      "sort": {"field": "latest_disclosable_activity_at",
                               "direction": "DESC"}})
    return d["data"]["search"]


def collect_cards(substate, want, rng):
    """Cards carry team handle + id, so the program cap costs no report fetch.

    Any pool small enough to enumerate is enumerated whole. Hacktivity is sorted
    by latest activity and programs disclose in bursts, so the most recent slice
    of a class is not a sample of it: the first build read 200 of the 533
    not-applicable cards and 173 of them belonged to programs already at the cap,
    leaving the class short at 27/50. Enumerating lets the cap choose *across*
    the class instead of fighting one burst.
    """
    qs = f"substate:{substate}"
    total = search_page(qs, 0, 1)["total_count"]
    window = min(total, 9900)
    cards, seen = [], set()
    enumerate_all = window <= FULL_ENUM_MAX
    offsets = (list(range(0, window, 25)) if enumerate_all
               else sorted(rng.sample(range(0, window - 25), k=max(4, want // 4))))
    for off in offsets:
        if not enumerate_all and len(cards) >= want:
            break
        page = search_page(qs, off, 25)
        for n in page["nodes"]:
            if n["_id"] not in seen:
                seen.add(n["_id"])
                cards.append(n)
        time.sleep(REQ_SLEEP)
    handles = Counter((c.get("team") or {}).get("handle") for c in cards)
    print(f"[cards] {substate:16} total={total:6d} collected={len(cards)} "
          f"programs={len(handles)} top={handles.most_common(2)}", flush=True)
    return cards, total


def get_report(rid, tries=4):
    """404 means the report is gone; anything else is retried.

    At REQ_SLEEP=0.2 a throttle response is likely, and skipping on it would
    quietly shrink a class (or bias it toward whatever was cached) instead of
    failing loudly. Only a real 404 drops a candidate.
    """
    p = CACHE / f"{rid}.json"
    if p.exists():
        return json.loads(p.read_text())
    url = f"https://hackerone.com/reports/{rid}.json"
    for attempt in range(tries):
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                j = json.loads(r.read())
            p.write_text(json.dumps(j))
            time.sleep(REQ_SLEEP)
            return j
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  report {rid} HTTP 404 -- dropped", flush=True)
                return None
            wait = 5 * (attempt + 1)
            print(f"  report {rid} HTTP {e.code} -- retry in {wait}s "
                  f"({attempt+1}/{tries})", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 5 * (attempt + 1)
            print(f"  report {rid} {type(e).__name__} -- retry in {wait}s "
                  f"({attempt+1}/{tries})", flush=True)
            time.sleep(wait)
    raise SystemExit(f"report {rid}: {tries} attempts failed -- stopping rather "
                     f"than building a short class")


def pick(substate, cards, rng, program_count):
    """Walk seeded-shuffled cards, fetch, keep rows with text under the program cap."""
    rng.shuffle(cards)
    kept, skipped_empty, skipped_cap = [], 0, 0
    for c in cards:
        if len(kept) >= TARGET[substate]:
            break
        handle = (c.get("team") or {}).get("handle") or "?"
        if program_count[handle] >= PROGRAM_CAP:
            skipped_cap += 1
            continue
        j = get_report(c["_id"])
        if j is None:
            continue
        if not (j.get("vulnerability_information") or "").strip():
            skipped_empty += 1
            continue
        j["_card"] = c
        kept.append(j)
        program_count[handle] += 1
    print(f"[pick]  {substate:16} kept={len(kept):3d}/{TARGET[substate]} "
          f"(empty_text={skipped_empty}, program_cap={skipped_cap})", flush=True)
    if len(kept) < TARGET[substate]:
        print(f"  !! short by {TARGET[substate]-len(kept)} -- report it, do not "
              f"silently widen the filter", flush=True)
    return kept


def main():
    rng = random.Random(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-9B-AWQ"))

    # scarce classes first so `resolved` cannot eat the program budget
    order = sorted(TARGET, key=lambda s: TARGET[s])
    program_count, chosen, totals = Counter(), {}, {}
    for sub in order:
        cards, total = collect_cards(sub, want=TARGET[sub] * 4, rng=rng)
        totals[sub] = total
        chosen[sub] = pick(sub, cards, rng, program_count)

    rows, manifest = [], []
    fam_counter, no_weakness, team_sev = Counter(), 0, 0
    for sub in order:
        for j in chosen[sub]:
            rid = j["id"]
            title = (j.get("title") or "").strip()
            body = (j.get("vulnerability_information") or "").strip()
            words = body.split()
            if len(words) > MAX_WORDS:
                body = " ".join(words[:MAX_WORDS])
            # §4.5: the program name is deliberately NOT in the state
            state = f"{title}\n\n{body}"
            ntok = len(tok(state, add_special_tokens=False)["input_ids"])
            outcome = sub.replace("-", "_")
            sev = j.get("severity") or {}
            sev_rating = sev.get("rating") if isinstance(sev, dict) else None
            sev_author = sev.get("author_type") if isinstance(sev, dict) else None
            wk = j.get("weakness") or {}
            wk_name = wk.get("name") if isinstance(wk, dict) else None
            fam = family_of(wk_name) if wk_name else None
            has_attach = bool(j.get("attachments"))
            program = (j.get("team") or {}).get("handle")

            if fam:
                fam_counter[fam] += 1
            else:
                no_weakness += 1
            if sev_author == "Team" and sev_rating in SEVERITY_OPTS:
                team_sev += 1

            base = {
                "group_id": f"h1-{rid}",
                "source": "HackerOne disclosed reports (hacktivity)",
                "source_url": f"https://hackerone.com/reports/{rid}",
                "license": "hackerone-disclosed-report",
                "license_nc": True,
                "redistribute": False,
                "license_note": "không thấy cấm, chưa xác nhận được phép",
                "known_benchmark": False,
                "label_source": "human",
                "lang": "en",
                "split_hint": "TEST",
                "state": state,
                "state_tokens": ntok,
                "state_words": len(body.split()),
                "truncated": len(words) > MAX_WORDS,
                "has_attachment": has_attach,
                "program": program,          # metadata only -- not in the state
                "report_id": rid,
                "seed": SEED,
            }
            rows.append({**base, "id": f"bugreport-{rid}-is_valid", "qtype": "noul",
                         "question_key": "is_valid",
                         "question_desc": "The security team accepted this report as a real, "
                                          "actionable vulnerability and resolved it.",
                         "label": outcome == "resolved"})
            rows.append({**base, "id": f"bugreport-{rid}-outcome", "qtype": "choice",
                         "question_key": "outcome",
                         "question_desc": "The triage decision the program recorded for this report.",
                         "options": ["informative", "not_applicable", "resolved", "spam"],
                         "label": outcome})
            if fam:
                rows.append({**base, "id": f"bugreport-{rid}-weakness", "qtype": "choice",
                             "question_key": "weakness",
                             "question_desc": "The weakness family the security team assigned.",
                             "options": None,  # filled once the set is known
                             "label": fam, "weakness_raw": wk_name})
            if sev_author == "Team" and sev_rating in SEVERITY_OPTS:
                rows.append({**base, "id": f"bugreport-{rid}-severity", "qtype": "choice",
                             "question_key": "severity",
                             "question_desc": "The severity band the security team assigned.",
                             "options": SEVERITY_OPTS, "label": sev_rating})
            pm_strict, pm_substr = program_mentioned(program, state)
            manifest.append({"report_id": rid, "substate": outcome, "program": program,
                             "severity": sev_rating, "severity_author": sev_author,
                             "weakness_raw": wk_name, "weakness_family": fam,
                             "has_attachment": has_attach, "state_tokens": ntok,
                             "program_mentioned": pm_strict,
                             "program_mentioned_substr": pm_substr})

    # options for weakness: only families that actually occur (PHASE3 §7 rule)
    fams = sorted(fam_counter)
    for r in rows:
        if r["question_key"] == "weakness":
            r["options"] = fams

    RAW.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    MANIFEST.write_text(json.dumps(
        {"spec": "docs/PROTOCOL.md (bug-report classification)", "seed": SEED,
         "n_states": sum(len(v) for v in chosen.values()), "n_rows": len(rows),
         "target": TARGET, "program_cap": PROGRAM_CAP, "max_words": MAX_WORDS,
         "index_totals": totals,
         "weakness_families": {k: v for k, v in WEAKNESS_FAMILIES},
         "rows": manifest,
         # Which rule produced which count, recorded next to the flags rather
         # than in prose: the two differ on 9 of 300 reports, and a table that
         # filters on one of them while quoting the other's n is wrong while
         # looking right. Counted from the rows above, never hardcoded.
         "program_mentioned_rules": {
             "program_mentioned": "word-boundary match of team handle in "
                                  "title+body, handle length >= 3",
             "program_mentioned_substr": "lowercased substring of team handle "
                                         "anywhere in title+body (the rule "
                                         "behind the 164/300 in brief 4.9a)",
             "counts": {
                 "strict": sum(1 for m in manifest if m["program_mentioned"]),
                 "substr": sum(1 for m in manifest if m["program_mentioned_substr"]),
                 "n": len(manifest)},
         }}, indent=1))

    states = sum(len(v) for v in chosen.values())
    print(f"\nwrote {OUT}  states={states} rows={len(rows)}")
    print(f"manifest -> {MANIFEST}")
    print("\n== outcome (state-level) ==")
    for k, n in Counter(m["substate"] for m in manifest).most_common():
        flag = "  <- SIGNAL ONLY (n<50)" if n < 50 else ""
        print(f"  {k:16} {n:3d}  {n/states:5.1%}{flag}")
    print(f"  noise share: {sum(1 for m in manifest if m['substate']!='resolved')/states:.1%} "
          f"| disclosed-report prior 87/13 | real queue prior: UNKNOWN")
    print("\n== weakness families ==")
    for k, n in fam_counter.most_common():
        print(f"  {k:16} {n:3d}  {n/max(1,states-no_weakness):5.1%}")
    print(f"  no weakness (not asked): {no_weakness} ({no_weakness/states:.1%})")
    print("\n== program_mentioned (state never contains the handle; the prose might) ==")
    for key in ("program_mentioned", "program_mentioned_substr"):
        n = sum(1 for m in manifest if m[key])
        print(f"  {key:26} {n:3d}  {n/states:5.1%}")

    print("\n== has_attachment ==")
    for k, n in Counter(m["has_attachment"] for m in manifest).most_common():
        print(f"  {str(k):16} {n:3d}  {n/states:5.1%}")
    print("\n== severity ==")
    print(f"  Team-authored (asked):   {team_sev:3d}  {team_sev/states:5.1%}")
    for k, n in Counter(m["severity_author"] for m in manifest).most_common():
        print(f"  author={str(k):10} {n:3d}")
    print("\n== programs (cap %d) ==" % PROGRAM_CAP)
    pc = Counter(m["program"] for m in manifest)
    print(f"  distinct={len(pc)} max={pc.most_common(1)[0]} top5={pc.most_common(5)}")
    print("\n== state tokens ==")
    ts = sorted(m["state_tokens"] for m in manifest)
    print(f"  median={ts[len(ts)//2]} p90={ts[int(0.9*len(ts))-1]} max={ts[-1]} "
          f"| >800 tok: {sum(1 for t in ts if t>800)}/{len(ts)}")


if __name__ == "__main__":
    main()
