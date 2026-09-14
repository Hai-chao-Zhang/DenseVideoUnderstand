# Complete protocol-screened leaderboard

The complete public view contains 47 results: 29 Educational methods and 18
High-Motion methods on the fixed 1,000-item preview. A separate educational GRT
comparison shows all 12 candidate/control rows; the CSV therefore contains 59
records, not 59 unique main-leaderboard methods. Nine non-aligned High-Motion runs
are excluded from every leaderboard view and CSV, including legacy High-Motion
GRT, midpoint-sampled InternVL, and full-trajectory-scored Gemini API runs.

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
- `highmotion-audit.json`: the complete 27-run source/protocol audit. Only its 18
  eligible methods produce browser rows. Ordered identities, eight-endpoint input
  and target policies, sample coverage, metric checks, and exact eligible-method
  membership are validated. Excluded evidence remains intact, outside rankings.

Generated `data/public-audit.json`, browser JavaScript and CSV are never accepted
as independent numeric inputs. The website's `scripts/build_audit_page.py` is a
thin wrapper around this same canonical module, so it can check every generated
asset byte-for-byte without maintaining separate ranking or numeric logic.

## Limits

This is reproduction and validation of archived artifacts, not fresh inference.
The High-Motion audit checks saved identities, prompts, targets, scores and wrapper
policies; it does not certify decoded-frame bytes, fully pinned model revisions,
identical hardware or a fresh GPU replay. The first 1,000 items are not the full
3,243-item High-Motion evaluation.

All three promoted educational GRT profiles exceed their contracted Open MOS and
Token F1 floors and use fewer patch projections. This is not a significance claim
or a win on every metric: Qwen 3B has lower reported throughput than its matched
all-patch control. No High-Motion GRT win is claimed by the screened leaderboard.
