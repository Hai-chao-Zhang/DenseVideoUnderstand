# Complete protocol-screened leaderboard

As of the 2026-09-14 [target/reference consistency hold](HIGHMOTION_TARGET_HOLD.md),
the current public view contains 29 Educational results. A separate Educational
GRT comparison shows all 12 candidate/control rows; the CSV therefore contains
41 records, not 41 unique main-leaderboard methods. All High-Motion results are
withheld, including the 18 previously protocol-screened archive candidates.
There is no ranked or unranked High-Motion numeric table or current CSV row.
The standalone High-Motion section explains the review and retains its existing
`highmotion-aligned` anchor for link compatibility.

Run with Python/PyYAML from a checkout, or with the installed minimal package:

```bash
python -m tools.densevideo.build_complete_leaderboard --verify-only
python -m tools.densevideo.build_complete_leaderboard --output /path/to/new-directory
```

Default resources are resolved from the source checkout or installed wheel,
independent of the working directory. No model, dataset, credentials, network or
GPU is needed. Existing output directories are refused. Catchable write failures
trigger best-effort cleanup of only this invocation's created files and empty
directories; concurrently created or replaced paths are preserved. This is not
atomic publication or crash durability. Open `leaderboard.html`
in the new directory: its CSS is inline, JavaScript is not required, and all
relative data links point to included files.
Links to current CSV and public-audit JSON carry `?v=20260914-target-hold` so cached
pre-hold assets are not reused; immutable historical artifact links are unchanged.

## Source contract

The generator first invokes the frozen `2026-08-20` verifier. This checks the
immutable 32-row historical snapshot and independently recomputes Open MOS and
Token F1 from 7,608 numeric records across all 12 educational comparison methods.
The pinned declared precision is retained after verification, avoiding harmless
floating-point re-summation changes to published values.

The additional `2026-09-14/manifest.json` is pinned by SHA-256 in the generator.
It binds the original frozen/provenance hashes, exact family/method/role
coverage, and byte hashes of two new canonical inputs:

- `comparison_telemetry.json`: nine candidate/matched-control telemetry records
  with original summary hashes. Candidate patch ratios, throughput and source
  hashes are cross-checked against the older provenance. Archived baselines have
  no invented telemetry. These are historical per-request aggregate measurements,
  not repeated hardware speed measurements.
- `highmotion-audit.json`: the immutable historical 27-run source/protocol audit.
  Ordered identities, eight-endpoint input and target policies, sample coverage,
  metric checks, and the exact historical 18-candidate membership are still
  validated. Historical `rank_eligible` flags do not confer current release
  eligibility. No High-Motion method produces a current browser or CSV row.

The historical manifest remains byte-identical, including its earlier
47-result/12-control (59-record CSV) coverage contract from 2026-09-14 before the
target hold. Current release policy is separate and hardcoded in the generator,
with no user override flag. Current `public-audit.json` and its JavaScript contain
an empty `highmotion_additional`, status
`highmotion_release_status: "held_target_reference_consistency_review"`,
`highmotion_hold_date`, a nonempty `highmotion_hold_reason`,
`highmotion_historical_protocol_screened_candidates: 18`, and
`highmotion_release_eligible_rows: 0`. The current CSV has 29 + 12 = 41 records.
The retained `data/leaderboard.js` and `data/highmotion-audit.json` are explicitly
historical artifacts; neither is a current result payload or release permission.

Generated `data/public-audit.json`, browser JavaScript and CSV are never accepted
as independent numeric inputs. The website's `scripts/build_audit_page.py` is a
thin wrapper around this same canonical module, so it can check every generated
asset byte-for-byte without maintaining separate ranking or numeric logic.

## Limits

This is reproduction and validation of archived artifacts, not fresh inference.
The historical High-Motion audit checks saved identities, prompts, targets,
scores and wrapper policies; it does not establish whether reference trajectories
follow the body part requested in the question. It also does not certify decoded
frame bytes, fully pinned model revisions, identical hardware or a fresh GPU
replay. The first 1,000 items are not the full 3,243-item evaluation. The new
bounded four-reference finding does not characterize every item or GRT performance.

All three promoted educational GRT profiles exceed their contracted Open MOS and
Token F1 floors and use fewer patch projections. This is not a significance claim
or a win on every metric: Qwen 3B has lower reported throughput than its matched
all-patch control. No High-Motion GRT win is claimed by the screened leaderboard.
