# Complete versioned leaderboard

The default 2026-09-15 view contains 29 Educational methods, 19 corrected-reference
High-Motion preview methods and 12 Educational GRT comparison rows: **60 CSV
records**, not 60 distinct main-leaderboard methods. HTML and CSV show no legacy
High-Motion result rows. All five new HF 0.5B GRT/baseline differences are visible,
including the Transition Accuracy regression. The preview is the fixed first
1,000 of 3,243 source records, with per-metric valid-reference coverage.

```bash
python -m tools.densevideo.build_complete_leaderboard --verify-only
python -m tools.densevideo.build_complete_leaderboard --output /path/to/new-directory
```

Use Python/PyYAML from the `main` checkout or the installed
minimal package. All three dated bundles are included in wheels; commands work
outside a checkout. No model, dataset access, credentials, network or GPU is used
by this numerical export. Open the resulting `leaderboard.html`: its CSS is inline,
it needs no JavaScript, and relative data links point to included files.

Existing output directories are refused. Catchable write failures trigger
best-effort cleanup of this invocation's own files and empty directories; other
writers' paths are preserved. This is not atomic publication or crash durability.

## Source contracts

- `2026-08-20`: immutable 32-row historical snapshot and 7,608 numeric records
  across all 12 Educational comparison methods. Open MOS/Token F1 means are
  independently recomputed; validated declared precision is retained.
- `2026-09-14`: authenticated Educational telemetry and unchanged 27-run legacy
  High-Motion protocol audit. Its old eligibility flags are historical evidence,
  not permission to republish old scores. Original manifests remain byte-identical.
- `2026-09-15`: five-file corrected-reference bundle, pinned by manifest SHA-256
  `1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc`.
  The loader verifies member bytes and constructor/scorer/validator source hashes,
  exact 19-method/1,000-record populations, matching identities/prompts, original
  positions, masks, metric-specific denominators and source-audit receipts.
  It independently reaggregates all 19,000 numeric records. Hash/coverage failures
  stop export; there is no fallback to legacy numbers.

The v2 bundle has no raw answers, predictions or videos. Public numeric validation
does not rerun inference, source projection or raw-prediction scoring. The
[actual execution/results audit](HIGHMOTION_V2_RESULTS.md) distinguishes these
checks and records the new GPU run and independent raw-prediction rescoring.

Current `public-audit.json` contains a separate authenticated `highmotion_v2`
summary. The legacy fields `highmotion_additional: []`, zero legacy eligible rows,
and the dated legacy hold remain intact. Only `highmotion_v2` supplies current
High-Motion rows; old `data/leaderboard.js` and `data/highmotion-audit.json` remain
historical evidence, never current numeric inputs. CSV/audit links in the v2 HTML
use a manifest-derived cache key.

Website `scripts/build_audit_page.py` is a thin wrapper around this same generator.
Its JSON, JavaScript, HTML and CSV are outputs, not independent numeric sources.
Explicit alternative v2 bundles require both `--highmotion-v2-bundle` and
`--highmotion-v2-manifest-sha256`; the same strict numeric contract applies.

For the separate dated 2026-09-14 hold view only:

```bash
python -m tools.densevideo.build_complete_leaderboard --legacy-reference-hold --verify-only
```

That historical view retains its 41 records and original numeric/evidence bytes;
its code link now uses an immutable commit instead of a retired branch. It cannot
accept a v2 override. It is not the current website leaderboard.

## Interpretation

The new HF 0.5B GRT improves Grid Accuracy, Token F1, ADE and FDE against its
configuration-checked archived baseline under the corrected reference; Transition
Accuracy regresses. This is a small preview point-estimate improvement, not a
full-split, fresh paired-baseline or statistical-significance result. Archived
weight revisions and consumed tensors cannot be recovered by numeric checks.

Educational numbers remain historical, not newly rerun. Qwen 3B's lower throughput
than its matched all-patch control remains visible. No win on every quality or
efficiency metric is claimed. Source access, licensing and manuscript alignment
remain separate from numerical leaderboard reproduction.
