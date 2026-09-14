# DIVE-Bench dataset release audit

Read-only audit performed on 2026-09-13. No Hugging Face repository was renamed,
merged, uploaded, deleted, or made public by this audit.

## Task names and versions

The manuscript shipped at `paper/ECCV_Dense_Video_Understanding.pdf`, sections
4.2 and 4.3, names the two scenarios **Educational High-FPS Videos** and
**High-Motion High-FPS Videos**. Use these names in user-facing documentation.
Recommended stable identifiers are:

| Paper name | Canonical task identifier | Existing compatibility alias | Proposed HF configuration |
| --- | --- | --- | --- |
| Educational High-FPS Videos | `dive_bench_educational_high_fps` | `densevideo` | `educational_high_fps` |
| High-Motion High-FPS Videos | `dive_bench_high_motion_high_fps` | `densevideo_highmotion` | `high_motion_high_fps` |

These canonical identifiers are a proposed integration naming scheme, not names
already registered by the upstream evaluation frameworks.

The public [arXiv version 2509.14199v2](https://arxiv.org/html/2509.14199v2), dated
2025-09-18, describes the earlier educational/subtitle benchmark; it does not
contain the later two-scenario manuscript. Updating code terminology does not
update that paper. Cite the public version accurately and distinguish the bundled
manuscript when describing the added high-motion track.

## Current Hugging Face access

| Repository | Anonymous observation | Standard cached-token observation |
| --- | --- | --- |
| [`haichaozhang/DenseVideoEvaluation`](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation) | Metadata public, `private=false`, `gated=auto`; annotation HEAD returns 401 `GatedRepo` | Metadata public; annotation HEAD returns 403 `GatedRepo` because this authenticated account has not been granted access |
| [`haichaozhang/highmotion_densevideounderstand`](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand) | Repository API and annotation HEAD return 401 | Repository API returns 404 `RepositoryNotFoundError` |

The high-motion responses establish that the referenced repository is inaccessible
to both tested callers. They do not establish whether its owner deleted it,
renamed it, or kept it private. Do not advertise an operational public download
until a clean account can retrieve the annotations and required videos. Existing
local caches do not prove public accessibility. Educational access requires the
Hub access agreement as well as authentication; see the official
[gated-dataset guide](https://huggingface.co/docs/hub/datasets-gated).

The educational metadata currently resolves to revision
`5cc61a045c8e5e95d1d9c87e22ccd0f699575aea` and lists:

| File | Bytes | Observation |
| --- | ---: | --- |
| `LPM_videos.parquet` | 16,801,231 | Release annotation table |
| `LPM_slides.parquet` | 16,801,231 | Identical LFS object to `LPM_videos.parquet` |
| `videos.zip` | 27,197,039,235 | Required external video archive; not downloaded by this audit |
| `README.md` | 7,677 | Card has no explicit `configs` mapping |
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
split; the card currently does not declare that split itself.

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
collisions. The remote high-motion schema, config, revision, archive layout and
license card could not be independently fetched.

The existing public high-motion leaderboard is the **first 1,000 rows, with eight
uniform endpoint-inclusive frames per clip**, not a full 3,243-row dense-frame
evaluation. The legacy code selects that prefix by default. Its explicit
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

## Recommended Hub organization and compatibility

Use one discoverable `haichaozhang/DIVE-Bench` entry point with the two named
configurations, each with an explicit `test` split. A single repository is a
convenience, not an HF requirement: the official
[data-file configuration guide](https://huggingface.co/docs/hub/datasets-data-files-configuration)
supports multiple configurations in one card. Do not concatenate the two tasks
or average their unrelated primary metrics.

Keep existing repository ids and pinned revisions usable. Add migration links to
their cards when authorized; preserve `densevideo` and `densevideo_highmotion`
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

This document recommends a migration; it does not assert owner permission to
change existing repositories, grant dataset access, or redistribute assets.
