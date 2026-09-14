# Reproduction contract

There are three different checks. Rebuilding the released numeric leaderboard
does not run a model. Running fresh inference does not regenerate Open MOS unless
the judge is requested. Passing a fresh quality gate does not mean its scores are
identical to the historical website. The runner reports these separately.

See [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) for the verified historical
numbers and [DATASET_RELEASE.md](DATASET_RELEASE.md) for data access, paper-version,
count and license limitations. No full GPU inference or judge rerun was performed
as part of the minimal-code release audit. A clean public-data download remains
unverified; existing local caches are not proof of public access.

## Installation

Use a dedicated Python environment: this distribution owns the `lmms_eval`
namespace and must not be installed alongside another lmms-eval checkout or
distribution. Its adapters reflect the frozen release, not every model in the
upstream framework. From the minimal branch:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

The core supported reproduction path uses PyAV, not Decord, and pins
Transformers 4.57.6, Qwen-VL-Utils 0.0.14, Torch 2.9.x and Torchvision 0.24.x.
Install a mutually compatible CUDA build of Torch/Torchvision for your hardware.
The local CPU verification used Python 3.10, Torch 2.9.0+cu126,
Torchvision 0.24.0, Transformers 4.57.6, Accelerate 1.12.0, Datasets 4.5.0,
NumPy 2.2.6 and PyAV 15.1.0. Wider dependency ranges are packaging compatibility
ranges, not a claim of bitwise-identical inference on every version or GPU.

`dive-reproduce`, `dive-leaderboard` and `dive-leaderboard-complete` also work from an installed wheel. At
build time, every dated directory under the canonical source `release/` tree is
copied into package resources; there is no second checked-in bundle. The commands
resolve the packaged dated bundles independently of the working directory.
Use `--bundle` only to audit an explicit alternate bundle. No credentials,
videos, model weights, reference answers or historical generated answers are
distributed in the numeric bundle.

## Offline published scores

Generate the current 47-result view and its 12 comparison rows:

```bash
dive-leaderboard-complete --verify-only
dive-leaderboard-complete --output /tmp/dive-complete-table
```

Open the generated `leaderboard.html` without needing JavaScript or external
assets. The [complete generation contract](COMPLETE_LEADERBOARD.md) describes
which archived inputs are verified and which nine High-Motion runs are excluded.
For the separate immutable 32-row historical snapshot:

```bash
dive-leaderboard --verify-only
dive-leaderboard --output /tmp/dive-published-table
```

The output directory must be new. Optional `--website` additionally checks the
live JavaScript against the pinned release. This reconstructs the historical
32-row table and verifies the exported per-example numeric evidence. It does not
recover the complete private historical campaign graph or run inference.

## Fresh educational GRT runs

The profiles contain immutable model revisions, three matched arms and their
actual generation budgets. Each uses eight sampled frames and all 634 published
educational QA rows by default:

| Profile | Model | Global generation cap | Additional routing |
| --- | --- | ---: | --- |
| `route31` | LLaVA-OneVision Qwen2 0.5B HF | 128 | Subtitle cap 31; OCR cap 128 |
| `qwen3` | Qwen2.5-VL 3B | 128 | Motion threshold 0.3 |
| `qwen7` | Qwen2.5-VL 7B | 48 | Subtitle threshold/floor 3.0/0.80; OCR 30.0/0.55 |

Authorize access to the educational dataset on Hugging Face and authenticate
using the standard Hub login or `HF_TOKEN`. Do not put tokens in commands, model
arguments, output files or commits. The task selects only `LPM_videos.parquet`
at revision `5cc61a045c8e5e95d1d9c87e22ccd0f699575aea`; the similarly named
`LPM_slides.parquet` is a duplicate and must not be concatenated with it.
Set `DENSEVIDEO_DATA_ROOT` to the extracted video root, preserving the annotated
relative paths, for example `DenseVideo-LPM/videos/<video>.mp4`.

The default command only prints a plan. It creates no directory, downloads no
weights and does not use a GPU:

```bash
dive-reproduce --profile route31 --output /tmp/dive-route31
dive-reproduce --profile qwen3 --output /tmp/dive-qwen3
dive-reproduce --profile qwen7 --output /tmp/dive-qwen7
```

Execute a one-example smoke check first, using one explicitly selected GPU and
a new output directory. Then omit `--limit` for a full run:

```bash
CUDA_VISIBLE_DEVICES=0 dive-reproduce --profile qwen7 --limit 1 \
  --output /tmp/dive-qwen7-smoke --execute
CUDA_VISIBLE_DEVICES=0 dive-reproduce --profile qwen7 \
  --output /tmp/dive-qwen7-full --execute --with-mos
```

The second command runs base, native all-patch control and candidate serially,
then loads the judge after the inference processes have exited. The optional
32B bfloat16 judge requires substantial GPU memory in addition to the evaluated
model; plan capacity before executing. CPU offload or different precision is not
silently substituted by these profiles. A strict CUDA determinism error stops
the worker rather than enabling a warning-only fallback.

The inference launcher sets `PYTHONHASHSEED=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms/cuDNN, disables cuDNN
benchmarking and TF32, and requires one visible physical GPU. The bootstrap seed
is zero; the evaluator explicitly retains its historical Python/NumPy/Torch/
few-shot seeds `0,1234,1234,1234`. Each arm records its GPU UUID, host, framework
versions and deterministic settings; a changed runtime or device fails validation.
These settings cannot guarantee identical outputs across different hardware,
drivers or library builds. They apply to inference workers; the separate MOS
command retains the recorded judge protocol and is not advertised as an
independently proven cross-hardware deterministic judge.

Workers also reject multi-process MPI, Slurm and distributed-launch metadata,
even when each process sees just one GPU. Evaluation errors propagate as nonzero
exit codes: the retained evaluator uses strict `DEBUG` exception handling, and
configuration-file overrides are rejected. Debug logs can contain dataset text
or predictions; keep raw run directories private and review them before sharing.

Every fresh arm must contain exactly document IDs `0..633` (or `0..limit-1` for
smoke), matching published hashed qid/video/type identities, with no duplicate,
missing, empty-output or malformed metric rows. Prompts, references and video
identities must agree across all three arms. The runner checks resolved model
arguments, generation caps and seeds against the plan. Patch telemetry must have
one row per request and valid integer counts; compute ratios are ratios of sums,
not means of per-request ratios.

Route31 and Qwen7 require byte-identical base/all predictions. Qwen3 reports this
comparison but does not impose that equality: its historical base/all controls
were distinct and had different scores. For Qwen7, the launcher additionally
records actual request document order. The validator recomputes every strict
`planned_keep_ratio < route_floor` decision, checks its native/linear projection
mode and patch counts, and requires every native-fallback prediction to equal the
same-session base prediction. This small logging wrapper does not change the
model's inputs, outputs or patch-projection implementation.

With `--with-mos`, the runner checks all three arms with
`Qwen/Qwen3-VL-32B-Instruct` at
`0cfaf48183f594c314753d30a4c4974bc75f3ccb`, bfloat16, batch 8, generation cap 64,
6,000-character truncation, and the pinned strict JSON scoring protocol.
It verifies per-document coverage, judge fingerprints, input/prediction matching
and valid integer scores. A full candidate must strictly improve both Open MOS
and Token F1 over fresh base, fresh all-patch and the archived public control;
both actual and reference patch-compute ratios must be at most 0.98. Failure
returns a nonzero exit code and retains the report. Smoke runs never qualify for
publication, regardless of their scores. Without MOS, success means inference
integrity only, and quality is explicitly `not_evaluated`.

Outputs include `reproduction_plan.json`, three logs,
`runs/<method>/densevideo/`, `summary.csv`, and
`reproduction_validation.json`; optional judge artifacts are under `mos/`.
The validation report distinguishes the fresh quality gate from numeric equality
to the historical candidate. Never substitute a fresh result into the released
table merely because the command completed. Fresh output logs include dataset
text and predictions; review dataset terms before sharing them publicly.

## High-motion historical preview

The canonical full task is `dive_bench_high_motion_high_fps`; the historical
website subset is explicitly `dive_bench_high_motion_high_fps_preview1000`.
`densevideo_highmotion` remains a legacy alias with its original environment
override behavior. Match `DENSEVIDEO_HIGHMOTION_NUM_FRAMES` to the model's actual
uniform frame budget. The historical GRT website row had an eight-frame
maximum, but raw logs show 787/1,000 inputs used only two through seven frames
and 148 clips were truncated at ten seconds. It is out of protocol for
eight-frame endpoint-inclusive whole-clip targets, even though its saved
metrics can be recomputed. Do not rank it against aligned baselines or claim
it demonstrates a high-motion GRT improvement. Preserve
`egodex/<action>/<clip>.mp4` paths: numeric basenames repeat
across actions and are not global identifiers.

`configs/densevideo/grt_highmotion_historical.yaml` records the saved legacy
LLaVA wrapper and arguments. That path additionally requires Decord 0.6.0 and
the external LLaVA-NeXT implementation providing `llava.model` and its SigLIP
encoder; these are not needed by the three educational profiles. The optional
`legacy-decord` extra only installs Decord, not LLaVA-NeXT. Decord has no suitable
wheel on some platforms, including the ARM test host. Use a separate compatible
environment for historical legacy-wrapper work. Its exact historical model/code
revision was not frozen, so this release does not promise a fresh exact inference
reproduction of that older high-motion row.

## Source modifications and attribution

The compact source inventory records historical hashes from the audited source
commit, not hashes of every modified release file. The minimal branch removes
unrelated models/tasks/campaign infrastructure and changes packaging/registration.
It also makes Decord imports optional in the PyAV path, removes an unused task
Decord import, fixes the canonical preview to exactly 1,000 rows, and fails on
missing video files instead of accepting a nonexistent path. Canonical high-motion
tasks check the ordered annotation-content SHA256 before selecting a split. High-motion path
resolution retains action directories to prevent basename collisions. These are
release, dependency and input-validation changes, not newly tuned GRT thresholds
or replacement scoring formulas. New aliases follow the bundled paper's two task
names. The frozen numeric release remains unchanged.

Adopted from lmms-eval: <https://github.com/EvolvingLMMs-Lab/lmms-eval>.
Copyright (c) 2024 LMMs-Lab. The original core pipeline is MIT licensed;
`lmms_eval/models` and `lmms_eval/tasks` use Apache-2.0. Retain the notices in
`LICENSE`, `LICENSE-APACHE` and the source when redistributing modifications. Dataset and model
licenses are separate from this code license; the wheel's combined SPDX
expression is not a license grant for videos, annotations or model weights.
