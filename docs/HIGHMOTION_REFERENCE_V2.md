# High-Motion reference correction v2 — not yet a published score release

This correction preserves the original questions and inference inputs. It fixes
the ground truth to the **right-hand ring-finger metacarpal joint**,
`transforms/rightRingFingerMetacarpal`, the explicitly named proxy for the right
palm-center/ring-finger-base target. It does not silently change a right-hand
question into a left-hand question, and it does not reuse old aggregate scores.

The target, construction, validity and scoring rules below are fixed before
examining new GRT outcomes. Choosing them is a documented benchmark correction,
not a claim to have recovered the original annotation-generation script.

## Why correct GT instead of changing questions?

The owner requested question/GT agreement without rerunning existing baselines.
Changing model inputs would make their saved predictions belong to a different
task. The 18 protocol-screened baselines have complete, source-bound predictions
for the same original 1,000-example preview. Keeping their original questions,
effective prompts, video identities and frame selection allows **CPU rescoring**
against one corrected reference set. No baseline model inference is needed.

The exact archived effective prompt—not only `doc.question`—must be checked for
every reused prediction. The new GRT run uses the HF `llava_hf` 0.5B implementation,
eight endpoint-inclusive PyAV-seek frames, BF16, resolution 384, eager attention,
router off and a 64-token output cap. The separate native LLaVA checkpoint is not
substituted for this HF baseline. Existing archival metrics are never copied into
the v2 table.

## One constructor, all source records

Version: `highmotion-right-ring-v2`. Reference policy:
`finite-positive-depth-in-frame-positive-confidence-v1`.

All 3,243 original rows, ordering, questions, video paths and frame counts are
retained. For every original frame, in float64:

```text
camera_from_joint = inverse(world_from_camera) @ world_from_right_ring_metacarpal
p = camera_from_joint[:3, 3]
h = camera_intrinsic @ p
pixel_xy = h[:2] / h[2]
```

There are no fitted offsets, frame shifts, hand mirroring, score-dependent joint
choices or tolerance-based mixtures of historical and recomputed labels.
Camera/joint input hashes bind each new reference to its source HDF5 file.

A scored reference requires finite transforms/projection, an invertible camera,
positive camera-space depth, a point inside `[0,width) × [0,height)`, and available,
finite joint confidence strictly above zero and at most one. Missing confidence
is unknown, not evidence of visibility. Invalid references are explicit nulls
with a validity mask and reason; they are not clamped into edge cells or assigned
`middle`. The nine grid cells have half-open boundaries; an internal boundary
belongs to the cell right/below it.

These are source-supported validity rules, **not a guarantee of visual annotation
accuracy**. EgoDex says confidence is optional, zero indicates occluded/undetected
joints, and its projection visualization warns about reprojection error. See the
[official dataset description](https://github.com/apple/ml-egodex) and
[projection example](https://github.com/apple/ml-egodex/blob/main/visualize_2d.py).

## Sampling, masks and metrics

Frames, labels and masks use the same original endpoint-inclusive indices:
`np.linspace(0, T - 1, min(8, T), dtype=int)`. Invalid frames are not removed before
sampling. The inference prompt keeps the original number of frame slots and is
never shortened using the ground-truth mask.

Scorer version: `position-preserving-masked-grid-v1`. A comma-separated sequence
or JSON list is accepted; canonical region names and `rNcN` aliases are equivalent.
An empty or unrecognized prediction remains an incorrect slot, not a deletion
that shifts subsequent answers. Additional output slots are counted and penalize
F1; they cannot contribute extra correct matches.

- Grid Accuracy and ADE use valid reference slots; missing/malformed predictions
  are wrong and receive the maximum normalized grid distance for ADE.
- FDE uses the original final sampled slot only if its reference is valid.
- Transition Accuracy uses adjacent original sampled slots only when both have
  valid references. It never bridges a masked gap or awards automatic perfect
  transition accuracy to a one-slot sequence.
- Token F1 uses canonical label bags on scored slots. Wrong/missing slots remain
  wrong tokens; surplus predictions only increase the precision denominator.
- Undefined row metrics are null. Aggregation is the macro mean of defined row
  metrics, with separate row/frame/edge denominators for every metric.

Rows with no scored reference remain in the dataset and coverage report. They do
not become automatic zeroes or perfect scores. A 1,000-record cache comparison
is labeled **preview-1000**, not a full 3,243-record evaluation; the number of
records with valid scored positions is additionally disclosed.

## Reuse and publication boundaries

Install the optional source-construction dependencies with
`python -m pip install '.[reference]'`. The installed commands
`dive-highmotion-reference --help` and `dive-highmotion-rescore --help` describe
the required local inputs and SHA-256 pins. They also run as
`python -m tools.densevideo.build_highmotion_reference_v2` and
`python -m tools.densevideo.rescore_highmotion_v2` from a checkout.
The constructor requires the source owner's complete ordered HDF5 binding
manifest; its pinned prior ZIP-member audit is recorded, not silently rerun.

`build_highmotion_reference_v2` constructs a separate reference artifact;
`rescore_highmotion_v2` reads it and saved predictions without loading any model.
Both refuse output replacement. Original annotations, prediction files and
historical results remain untouched. Constructor/scorer tests are synthetic,
not evidence that the full real-data correction or GPU evaluation has completed.

For new GRT generation, retain the original task and its original `doc.answer`
metadata in the sample JSONL, then supply the v2 Parquet **separately** to the
rescoring command. Its legacy-answer hash check binds the predictions to the
unchanged original task; it does not score against that legacy answer. Any
metrics emitted during original-task generation are legacy metrics, not v2
leaderboard entries. The same corrected Parquet, prompt cache and scorer must
be passed when rescoring every corresponding baseline.

GRT comparisons must use the same corrected reference bytes, scorer version,
effective questions, sample IDs/order and stated input configuration. The primary
High-Motion ranking metric remains Grid Accuracy; all five metrics and their
coverage must be shown. No candidate is described as winning unless its actual
score satisfies the stated comparison; no automatic publication is performed.

The archived baselines lack recorded consumed-tensor hashes, package versions
and precise model revision pins. Therefore the comparison can establish
**new GRT predictions versus rescored, configuration-checked archived predictions**,
not a freshly rerun, byte-identical paired experiment or statistical significance.

Publishing constructor code does not grant redistribution rights for source
videos or derived annotations. Existing source access restrictions and dataset
terms remain unchanged. No new public benchmark availability is claimed here.
