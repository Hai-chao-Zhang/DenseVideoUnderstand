# Publication and score audit

Audit date: 2026-09-13. Audited evaluation-code commit:
`78284318c9c8664df5fc6785410a0a9c4e494436`.
Public source: [DIVE-Bench leaderboard](https://www.zhanghaichao.xyz/DenseVideoUnderstand/).

The three promoted LPM GRT scores agree with the website and the frozen
per-sample evidence. This establishes historical score consistency. No model
inference or Open MOS judge was rerun during this audit, and this audit does
not determine paper acceptance.

## Verified LPM results

Each result uses all 634 LPM items. Open MOS values below are unrounded means;
the website stores Open MOS at six significant digits. Token F1 and patch
projection ratios in the website data equal the selected run summaries.

| Profile | Candidate Open MOS | Strongest contracted Open MOS floor | Candidate Token F1 | Strongest contracted F1 floor | Patch projection ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| LLaVA-OneVision Qwen2 0.5B Route31 | 0.11987381703470032 | 0.1167192429022082 | 0.014196368250678385 | 0.014087875254888587 | 0.8671541327517492 |
| Qwen2.5-VL 3B t03 | 1.5899053627760253 | 1.5583596214511042 | 0.09958369460437803 | 0.09913584373152082 | 0.8856587587716475 |
| Qwen2.5-VL 7B route floor | 1.5914826498422714 | 1.5757097791798107 | 0.048908690353566486 | 0.04877739573780698 | 0.8477193716603361 |

The archived public Qwen baselines have Open MOS 1.135646687697161 (3B)
and 1.2839116719242902 (7B). They are different runs from the stronger matched
controls in this table. GRT comparisons should report matched controls as
well as archived public baselines; their score difference must not all be
attributed to gating. The portable provenance contains each of the three
contracted floors per family.

The LLaVA-OneVision 7B candidate failed its full Open MOS gate and was not
promoted. The website's three-of-four family qualification is consistent
with that decision. These are observed point-estimate gates, not statistical
significance claims. The verified LPM profiles use eight sampled video
frames; their results do not establish an advantage at an independently
measured high-frame-rate operating point. Patch projection ratio is not
end-to-end FLOPs. `effective_fps` is sampling density, while `throughput_fps`
is wall-clock processing throughput.

## What was checked

- The live website JavaScript equals the 2026-08-20 frozen release byte for
  byte: SHA256 `c88492ec535a3bc4f47fe9e91f66bc050a857f757cd2c478db598e889d9dc1b1`.
  It contains 29 LPM rows, three High-Motion rows, and three promoted GRT
  families.
- The audited commit's `leaderboard.md` equals the frozen Markdown byte for
  byte. The original workspace's uncommitted 2026-08-23 `leaderboard.md`
  differs: it omits all Open MOS values and all three verified GRT LPM rows,
  and includes a newer 27-row High-Motion matrix. That file is not the
  website release and was not used for this minimal branch.
- Each promoted selection, MOS matrix, and selected summary matches the
  release manifest's SHA256. Twelve source sample logs (three candidates
  and their nine controls) also match their recorded SHA256 values.
- The exported 7,608 numeric records join MOS rows to saved sample metrics
  by document identity. Question, reference answer, and prediction agree
  (prediction comparison ignores only surrounding whitespace). Every
  method contains exactly document IDs 0 through 633. Recomputed MOS and
  Token F1 means match the stored selections within floating-point
  aggregation tolerance.
- All six published candidate numeric fields were compared: Open MOS,
  Token F1, patch projection ratio, reference patch ratio, sampling density,
  and throughput. Open MOS matches its documented six-significant-digit
  representation; the remaining fields match exactly as floats.
- Source bytes in the original audited checkout match the frozen Route31 inventory (17 files),
  Qwen3 recovery driver inventory (three files), Qwen7 full inventory (seven
  files), and Qwen7 prerequisite screen inventory (29 files). The compact
  source inventory records the relevant inference/task files separately
  from packaging and model-registration edits in the minimal branch.

The minimal branch is not byte-identical to that original checkout: it prunes
unrelated registrations, makes Decord imports optional for PyAV use, adds the
paper-named task aliases and a fixed preview selector, and makes video-path
resolution fail on missing files without flattening EgoDex action directories.
These release changes do not alter GRT kernel calculations or scoring formulas;
the original source hashes remain historical provenance, not hashes of every
current file. The published snapshot has a clean root commit so that the private
development repository's ancestry is not distributed with the minimal code.

The audited High-Motion task functions reproduce the saved prompt,
  target, and all five numeric metrics for every one of its 1,000 samples.
  Aggregates reproduce website values after the website's six-significant-
  digit rounding. This verifies the task/scorer path, while fresh model
  inference remains untested.

## Publication and reproduction limitations

The historical full campaign validation is currently not portable or fully
replayable from its original paths. Route31's native evidence validator
passes. The full four-family validator stops at a missing legacy LLaVA7
archived result. Independent Qwen3 and Qwen7 native validation stops because
the shared, mutable public `summary.csv` no longer matches its frozen
checksum. These failures concern the historical upstream evidence graph;
the matrices, selected summaries, and per-sample logs used in the numeric
checks above still match their pins. Do not describe the entire old
campaign chain as freshly revalidated.

The portable release intentionally carries small numeric evidence rather
than the complete historical campaign tree. It can reconstruct and check
the published scores offline. Re-running inference additionally requires
the documented model revisions, dataset revision and access, decoder and
framework dependencies, and appropriate hardware. Regenerating judge
scores requires the original predictions/reference text and the pinned
judge protocol, which are not included in this numeric-only package.

The High-Motion website row is a historical 1,000-item preview. Its GRT
wrapper is `llava_ov_dense_video`; the saved run used Decord, eight frames,
`dense_frame_fps=1`, a ten-second clip limit, threshold 0.05, and enabled
scene merging. The original general leaderboard config now has different
arguments. `configs/densevideo/grt_highmotion_historical.yaml` records the
actual saved arguments. The original run did not freeze a full source
snapshot or model revision, so this config alone cannot certify an exact
fresh inference reproduction. Its saved Git hash identifies a base commit
with uncommitted evaluation code; it must not be treated as a complete
runtime source pin.

Before describing the combined paper/code/data release as complete, align
the paper version and two task names, make the chosen benchmark data
revision accessible, and resolve the public website's "code forthcoming"
and High-Motion mirror status. The current arXiv version and local ECCV
paper differ in scope according to the separate paper audit. Framework
submission should state the intended paper/version and dataset access
requirements explicitly. No claims about framework acceptance are made
by this numerical audit.

## Portable files

`release/2026-08-20/leaderboard.js` and `leaderboard.md` are exact public
snapshot artifacts. `leaderboard.csv` preserves public score columns and
removes operational paths/telemetry details that the website does not
display. The original full CSV checksum is recorded in `provenance.json`.
The three `*_numeric.csv` files contain only method index, document index,
Open MOS, and Token F1. `sample_identities.csv` records hashed identities;
`provenance.json` maps method indices, revisions, source checksums, and
aggregate telemetry. No videos, reference answers, generated answers, or
credentials are included.
`highmotion_numeric.csv` additionally carries document index and the five
saved High-Motion metrics for all 1,000 preview items.
