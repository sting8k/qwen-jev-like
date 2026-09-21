"""Fetch the four upstream latency presets that `run_bench.py` measures against.

They are someone else's work, so this repository links them instead of
vendoring them: the script pulls the files at a pinned revision and verifies
each one against the SHA-256 recorded here. Those are the exact bytes the
published latency table was produced from, so a mismatch means the comparison
would no longer be the same comparison, and the fetch refuses rather than
carrying on with different inputs.

    python data/build/fetch_bench_presets.py

Writes reference/presets/*.json, which is git-ignored. No model weights.

Upstream: https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import sys

REPO_ID = "harshatheg/Qwen-2.5-1B-RLCD"
REVISION = "2af86848be75847ccb3553b0941cc51d6ef7e4e9"
OUT = pathlib.Path(__file__).resolve().parents[2] / "reference" / "presets"

# sha256 of each file as measured on 2026-09-22 at the revision above, and as
# used for every published latency number that names these presets.
EXPECTED = {
    "fintech_fraud": "1dd2500cf4c3d82f62810dc17aacd662238d89aeb6472dd5371c7c1e22f185cf",
    "code_security": "79b592ced937b63735ae7179181e424344977580f13b3a5c2a88b17c27a4d3b1",
    "support_triage": "dcad89d77955893b1db6488e467a5ebe50b9e0376b8835dfb58132edea636fcc",
    "high_cardinality_255": "2beff0679b0cd0b9406540d046b1a3ae2e153e12a8ecf18d8d6bb16fc96e57b6",
}


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("needs huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    bad = []
    print(f"{REPO_ID} @ {REVISION[:12]}")
    for name, want in EXPECTED.items():
        src = hf_hub_download(REPO_ID, f"presets/{name}.json", revision=REVISION)
        data = pathlib.Path(src).read_bytes()
        got = hashlib.sha256(data).hexdigest()
        ok = got == want
        print(f"  {name:22s} {got}  {'ok' if ok else 'MISMATCH, expected ' + want}")
        if not ok:
            bad.append(name)
            continue
        shutil.copyfile(src, OUT / f"{name}.json")

    if bad:
        print(f"\n{len(bad)} of {len(EXPECTED)} files do not match the recorded "
              f"digest: {', '.join(bad)}.\nNothing was written for those. The "
              f"latency table in docs/RESULTS.md describes the recorded bytes, "
              f"so it does not describe these.", file=sys.stderr)
        return 1

    print(f"\n{len(EXPECTED)} files verified -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
