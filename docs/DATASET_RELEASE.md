# DIVE-Bench dataset release audit

Initial read-only audit: 2026-09-13. Owner-authorized configuration follow-up:
2026-09-14. The follow-up created a private two-task annotation entry and updated
only the old repositories' cards; it did not delete data, change source visibility
or gating, or redistribute video archives. Historical observations below are
explicitly scoped to their audit dates.

## Task names and versions

The manuscript shipped at `paper/ECCV_Dense_Video_Understanding.pdf`, sections
4.2 and 4.3, names the two scenarios **Educational High-FPS Videos** and
**High-Motion High-FPS Videos**. Use these names in user-facing documentation.
Stable identifiers are:

| Paper name | Canonical task identifier | Existing compatibility alias | HF configuration |
| --- | --- | --- | --- |
| Educational High-FPS Videos | `dive_bench_educational_high_fps` | `densevideo` | `educational_high_fps` |
| High-Motion High-FPS Videos | `dive_bench_high_motion_high_fps` | `densevideo_highmotion` | `high_motion_high_fps` |

The identifiers are now implemented in the submitted draft integrations
[VLMEvalKit #1686](https://github.com/open-compass/VLMEvalKit/pull/1686) and
[lmms-eval #1521](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1521),
but have not been merged upstream.

The public [arXiv version 2509.14199v2](https://arxiv.org/html/2509.14199v2), dated
2025-09-18, describes the earlier educational/subtitle benchmark; it does not
contain the later two-scenario manuscript. Updating code terminology does not
update that paper. Cite the public version accurately and distinguish the bundled
manuscript when describing the added high-motion track.

## Hugging Face configuration completed on 2026-09-14

[`haichaozhang/DIVE-Bench`](https://huggingface.co/datasets/haichaozhang/DIVE-Bench)
now contains exactly two configurations, each with a `test` split, at revision
`d80461fccf879d5efdeece0edce8608a72d64f10`:

- `educational_high_fps`: 634 questions over 317 videos.
- `high_motion_high_fps`: 3,243 clips; the first-1,000 preview remains a client
  selection, not a third Hub configuration.

The entry is **private** to preserve the private high-motion source's audience
while release rights and access are resolved. It contains byte-identical source
annotations and component-specific terms, not copied video archives. This is a
completed owner-authorized configuration, not a claim of public data availability.

Both compatibility source cards now expose their one named task explicitly:

| Source | Updated card revision | Access policy (unchanged) |
| --- | --- | --- |
| `haichaozhang/DenseVideoEvaluation` | `c3ff65dfc37239ebee05bd190cfa5b5126f49146` | Public metadata, `gated=auto` |
| `haichaozhang/highmotion_densevideounderstand` | `25cc1aeaef5209776625ce4e72a3ba425d4ae929` | Private |

All non-card source file object hashes remained unchanged. Historical data
revisions below remain usable, so pinned evaluation integrations do not silently
switch to a new dataset. The explicit educational configuration excludes the
identical slides Parquet, avoiding accidental duplication to 1,268 rows.
Fresh-cache `datasets.load_dataset` checks passed for both configurations in
the combined entry and each named configuration in the two source repositories:
four actual remote loads, with exact annotation SHA-256, row counts, unique
video/identity counts and high-motion ordered-content validation. These checks
used the owner account and do not establish anonymous or third-party access.

## Initial access observations (2026-09-13)

| Repository | Anonymous observation | Standard cached-token observation |
| --- | --- | --- |
| [`haichaozhang/DenseVideoEvaluation`](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation) | Metadata public, `private=false`, `gated=auto`; annotation HEAD returns 401 `GatedRepo` | Metadata public; annotation HEAD returns 403 `GatedRepo` because this authenticated account has not been granted access |
| [`haichaozhang/highmotion_densevideounderstand`](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand) | Repository API and annotation HEAD return 401 | Repository API returns 404 `RepositoryNotFoundError` |

Those initial high-motion responses established that the referenced repository was
inaccessible to both tested callers, without distinguishing deletion from privacy.
The owner-authenticated follow-up below resolves that ambiguity. Do not advertise an operational public download
until a clean account can retrieve the annotations and required videos. Existing
local caches do not prove public accessibility. Educational access requires the
Hub access agreement as well as authentication; see the official
[gated-dataset guide](https://huggingface.co/docs/hub/datasets-gated).

Owner-authenticated follow-up on 2026-09-14 confirmed that the high-motion
repository is **private**, at revision
`d44407f607fdf020c59b816884f06ed6d453cf26`. Its original Hub Parquet SHA-256 is
`518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd`.
That serialization and the locally audited `39f9da7a...` file have identical
ordered task content (hash `90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789`).
This resolves source identity, not anonymous access or redistribution rights.

The initial educational metadata resolved to revision
`5cc61a045c8e5e95d1d9c87e22ccd0f699575aea` and lists:

| File | Bytes | Observation |
| --- | ---: | --- |
| `LPM_videos.parquet` | 16,801,231 | Release annotation table |
| `LPM_slides.parquet` | 16,801,231 | Identical LFS object to `LPM_videos.parquet` |
| `videos.zip` | 27,197,039,235 | Required external video archive; not downloaded by this audit |
| `README.md` | 7,677 | Historical card had no explicit `configs` mapping; fixed in the follow-up |
| `.gitattributes` | 110 | Repository metadata |

Both Parquet names have SHA-256
`9ab09ea35a66ca86fc7fdce4e539171ea5eef38f8162874f74dd1b2809dafb66`.
Select `LPM_videos.parquet` explicitly; loading both names would duplicate the
same examples. The card's `openrail` label is not a complete description of the
upstream assets' terms (see below).

## Verified annotation schema and counts

The pinned educational release was inspected from its existing HF Arrow cache,
whose metadata records the exact revision and `LPM_videos.parquet` source. It has
**634 QA rows over 317 distinct video paths**: 317 subtitle questions with `_sub`
qid suffixes and 317 OCR questions with `_ocr` suffixes. All `type` values are
`QA`; use the qid or explicit question routing for the two question types. All
634 qids are distinct and all answers are nonempty.

The minimal evaluator needs string fields `video`, `video_path`, `question`,
`answer`, `qid`, `type`; integer `frame_count`, `width`, `height`; floating-point
`fps`, `segment_start`, `segment_end`. Additional source fields are
`speaker_index`, `captions`, `ocr_answer` (strings), and `ocr` (a list of OCR
records). Each OCR record contains `text` plus integer fields `Unnamed: 0`,
`block_num`, `conf`, `height`, `left`, `level`, `line_num`, `page_num`, `par_num`,
`top`, `width`, and `word_num`.

An educational path is, for example,
`DenseVideo-LPM/videos/dDPcUO41nkE.mp4`. Resolve it under an explicit data root;
retain the relative path or use the known educational `videos/<basename>`
layout. The published evaluator explicitly maps this Parquet file to a `test`
split; the updated source card now declares that split explicitly too.

The locally available `Egodex_traj.parquet` has SHA-256
`39f9da7aca9020d79f383953646a5893f09c6f8e5f60433560011280ee987b2d`,
**3,243 rows and 3,243 distinct video paths** across 111 action directories.
Its `video` and `qid` are local numeric names reused across directories, each
having only 277 distinct values. **Do not use `qid` alone as a global key**: use
the stable row index or `(video_path, qid)` and retain the original qid as metadata.

It shares the educational columns, except that `ocr` is an empty/null-element
list and it adds `answer_traj` (a JSON-encoded coordinate sequence string).
`answer` is a JSON-encoded list of semantic 3-by-3 grid labels. Every answer and
coordinate sequence has exactly `frame_count` entries. All rows are 1920 by
1080 at 30 FPS; frame counts range from 15 to 5,237. Paths have the form
`egodex/add_remove_lid/0.mp4`; preserve the action directory to avoid basename
collisions. The initial non-owner audit could not fetch the remote source.
The owner follow-up verified its annotation bytes, equivalent ordered task
content, revision and source card; its archive contains all 3,243 referenced
MP4 paths. This is not a fresh full-video decode or GPU evaluation claim.

The existing public high-motion leaderboard uses the **first 1,000 rows** and
the scorer's target policy is eight uniform endpoint-inclusive frames, not a
full 3,243-row dense-frame evaluation. Actual legacy GRT inputs do not follow
that policy: 787/1,000 contain fewer than eight frames and 148 clips are
truncated at ten seconds. Keep that result unranked and separate from aligned
input protocols. The legacy code selects the prefix by default. Its explicit
`DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES=0` option selects all 3,243 examples; it does
not change the default eight-frame budget. Report both example subset and frame
policy with every result.

## Corrections required before paper-result equivalence claims

The bundled manuscript's Table 8 gives 317 educational QA pairs, whereas the
published release evaluates 634. Its 277 high-motion clips match the number of
distinct local numeric ids, whereas the annotation table has 3,243 distinct
paths; treating the 277 reused ids as distinct clips undercounts the table.
The educational duration range is approximately 190.3 to 7,072.2 seconds;
362 of the 634 rows are at most 30 minutes. The manuscript's universal
greater-than-30-minutes description therefore also needs correction.

The manuscript describes GPT-3.5 MOS and processing-speed Effective FPS. The
current released leaderboard uses a pinned open-model MOS judge and defines
`effective_fps` as sampled frames divided by full source-video duration. Report
the open-judge identity and sampling-density definition explicitly; use
`throughput_fps` for sampled frames per request wall-clock second. New public
scores must not be presented as a direct reproduction of old paper tables with
different judges, question counts, or sampling policies.

## Public metric contract

Educational text normalization strips outer whitespace, lowercases and collapses
whitespace; it does not strip punctuation. CER is character Levenshtein distance
divided by `max(reference_character_count, 1)`; WER uses whitespace tokens and
the analogous denominator. Both may exceed 1. Token-F1 uses multiset token
overlap; both empty sequences score 1, exactly one empty sequence scores 0.
Exact Match compares normalized strings. Aggregate per-example values by the
arithmetic mean, not by concatenating all references. Any accelerated mode that
skips large edit distances must report missing coverage instead of silently
calling the result full CER/WER.

For high-motion, normalize region aliases to `r1c1` through `r3c3`. The center of
cell `(r,c)` is `((2*c-1)/6, (2*r-1)/6)` in normalized image coordinates. Sample
the target at the same endpoint-inclusive integer `linspace(0,F-1,min(K,F))`
indices as the model frames. Grid accuracy is the fraction of matching frame
labels; Grid ADE is mean Euclidean center distance; Grid FDE is the distance at
the final reference frame. A missing or unparseable prediction receives
`sqrt(2)` distance and no correct-label credit. Extra predicted positions are
ignored for displacement and grid accuracy. Aggregate per-clip metrics by their
arithmetic mean. Grid transition accuracy compares consecutive displacement
vectors rounded to six decimals. Optional Token-F1 uses the parsed canonical
labels as whitespace tokens; it does not measure trajectory order.

The released Open MOS protocol is separate from inline GPT-3.5 scoring. Its
default judge is `Qwen/Qwen3-VL-32B-Instruct`, prompt version
`densevideo-open-mos-v1`, strict response schema `strict-json-pred-score-v1`
(`pred` yes/no, integer `score` 0 through 5), and 12,000-character default
reference/prediction truncation. Exact published reproduction additionally needs
the release manifest's judge revision, generation settings, request fingerprints
and per-example score artifacts. Disabled or failed judge calls must not be
represented as genuine zero MOS observations.

## Hub organization and compatibility

The configured `haichaozhang/DIVE-Bench` entry point has the two named
configurations, each with an explicit `test` split. A single repository is a
convenience, not an HF requirement: the official
[data-file configuration guide](https://huggingface.co/docs/hub/datasets-data-files-configuration)
supports multiple configurations in one card. Do not concatenate the two tasks
or average their unrelated primary metrics.

Existing repository ids and pinned revisions remain usable; the updated source
cards include migration links. Preserve `densevideo` and `densevideo_highmotion`
as evaluator aliases. Preserve the exact published row order and annotate the
1,000-row public high-motion profile explicitly; a full-split profile should have
a distinct configuration/result label. Include stable example ids, source ids,
annotation checksums, video-path manifests, source revisions and the extraction
commands. Validate both metadata loading and resolution of every required video
in a clean cache before announcing the new hub as available.

Source terms must stay attached to their respective assets. The
[official LPM repository](https://github.com/dondongwon/LPMDataset#license)
states CC BY-NC-SA 4.0 for its repository data and identifies the source videos
as under the standard YouTube license. The
[official EgoDex repository](https://github.com/apple/ml-egodex#license)
states CC BY-NC-ND for its dataset, separately from its code license. The current
educational `openrail` card tag does not replace either upstream statement.
Document component-specific terms and attribution; prefer original-provider
downloads plus deterministic annotation construction when redistribution rights
for transformed videos or annotations have not been established. A combined
card should not imply one new blanket license for all upstream content.

The owner authorized and completed the configuration-only migration above.
It does not grant third-party access, resolve redistribution rights, or make the
private high-motion annotations publicly downloadable.
