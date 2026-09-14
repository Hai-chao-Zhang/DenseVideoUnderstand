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

This is a bounded four-reference finding, not an audit of all 3,243 items. The
constructor and target mapping require review; the
scope of any affected examples is not yet established. It is not evidence about
GRT's quality, a matched GRT ablation, or a new performance result. A successful
runtime smoke check cannot resolve this semantic question.

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

## Conditions for revisiting the hold

Resolve and document the constructor's intended body-part target against the
canonical questions and archived references; establish the affected scope with
source-bound evidence; then explicitly review any required data/protocol changes
and revalidation before releasing High-Motion comparisons. Neither historical
metric reconstruction nor new GPU inference alone satisfies this review.
