# examples/

`cloudflare/jev_examples.json` — request/response pairs published by Cloudflare
for their hosted typed-decision endpoint, captured 2026-09-18 against version
`jev-1.13` and pinned since.

`run_sanity.py` replays them as a direction check: the comparison is whether
this engine picks the same option, not whether it reports the same probability.

Nothing else is kept here. The other vendors' documentation this project read
while working out the contract is cited under References in the README rather
than copied.
