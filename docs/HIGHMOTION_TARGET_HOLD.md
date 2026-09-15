# High-Motion target/reference consistency hold

Effective date: 2026-09-14. Status: `held_target_reference_consistency_review`.

All High-Motion results are withheld from the current leaderboard, including all
previously protocol-screened candidates. The current view contains 29 Educational
results and 12 Educational GRT comparison rows, for 41 CSV records. There is no
ranked or unranked High-Motion numeric table and no High-Motion current CSV row.

## Bounded finding and open question

An independently confirmed check of four available canonical construction references found that
the stored trajectories match a left-index joint projection
(`leftIndexFingerMetacarpal`), while the canonical question and current task ask
for a right-hand target (the palm/ring-base region). This raises a target/reference
consistency issue: reproducing saved labels or metrics cannot by itself establish
that the reference follows the body part requested in the question.

The initial finding covered four references. A subsequent fixed-formula source
audit covered all 3,243 references (826,253 frames), with every local HDF5 file
SHA-256 matched to its canonical archive member. Of these, 2,871 complete
trajectories match the fixed left-index projection within the predeclared
0.001-pixel component tolerance; 3,121 references have all grid labels match
that projection. These are different checks, and neither matches every item.
Negative-depth and offscreen projections are present; both checked joints lack
confidence data in 454 references. The audit completed with data issues, not a
universal left-index-equivalence or universal incorrect-label conclusion.

This follow-up used only two fixed joints and the prespecified projection; it
did not fit a transform, search for a better mapping, or inspect model predictions.
It does not recover the original constructor or establish intended anatomy,
visibility, or confidence validity. Its private diagnostic report SHA-256 is
`13d8ce987cca4b5d5c534743809a426f184db811bfb1f5b94d23982f99f81e08`.
The constructor and target mapping still require review. Neither this source
audit nor a successful runtime smoke check establishes GRT quality, a matched
GRT advantage, or a new performance result.

No reference answers, joint coordinates, videos or private source data are
published by this note. No dataset, model, scoring rule, sampling protocol, GRT
gate or model implementation has been changed as part of the hold.

## Historical evidence remains unchanged

The immutable 2026-08-20 32-row snapshot and the historical 27-run High-Motion
audit remain byte-identical. The frozen 2026-09-14 manifest SHA-256 remains
`5767bd60a3d8efe24722d9d4b12a374c7e566d50c370cbacdc02cdfab3162232`.
Its earlier 47-result/12-control, 59-record coverage contract predates this hold.
The verifier still checks the historical 18 protocol-screened candidates and
nine protocol exclusions. Those historical `rank_eligible` flags concern the
archived protocol audit and are not current release permission.

The current generator emits an empty `highmotion_additional`, the hold status and
reason, and zero release-eligible High-Motion rows. It retains the original
historical evidence files separately and offers no flag to bypass the hold.
Educational numbers, ranks and all three promoted GRT quality gates are unchanged.

The generic `tools.densevideo.build_leaderboard` exporter also refuses all three
High-Motion task aliases before writing Markdown or CSV. Mixed Educational and
High-Motion inputs fail explicitly rather than silently dropping the held rows;
there is no bypass flag. This is a publication guard, not a task removal: raw
evaluation, metric collection, full/preview task registrations, and the separate
immutable historical reconstruction remain available. The promoted
`dive-reproduce` entry point continues to offer only the three Educational profiles.

## Conditions for revisiting the hold

Resolve and document the constructor's intended body-part target against the
canonical questions and archived references; establish the affected scope with
source-bound evidence; then explicitly review any required data/protocol changes
and revalidation before releasing High-Motion comparisons. Neither historical
metric reconstruction nor new GPU inference alone satisfies this review.
