# DIVE-Bench and Gated Residual Tokenization (GRT)

Minimal evaluation and reproduction code for the [project](https://www.zhanghaichao.xyz/DenseVideoUnderstand/)
and [bundled manuscript](paper/ECCV_Dense_Video_Understanding.pdf).
This branch contains the DIVE-Bench tasks, GRT implementations and controls,
Open MOS scoring, and a portable numerical leaderboard evidence bundle. It omits
unrelated benchmarks/models, private experiment orchestration, and raw datasets.

## Release status

The three promoted Educational GRT results match the official 2026-08-20
leaderboard snapshot. Per-example numerical evidence independently reconstructs
their Open MOS and Token F1; the complete 32-row leaderboard rebuilds byte for
byte. This verifies historical results, **not a fresh GPU rerun**. The legacy
High-Motion result covers 1,000 preview items, not the full 3,243-item split.
Its legacy GRT input sampler was not aligned to the target protocol: 787 of
1,000 inputs contained fewer than eight frames, and 148 clips were truncated.
That row is retained only in the historical evidence archive and is excluded
from the current leaderboard and CSV; numeric reconstruction alone does not
validate model inputs.

On 2026-09-14, all three supported Educational profiles completed a
[two-item, nine-arm GPU smoke check](docs/GPU_SMOKE_VALIDATION.md), including
fresh Open MOS and matching consumed input tensors. Full 634-example inference
and score equality remain unverified; smoke results are not leaderboard entries.

| GRT profile | Open MOS (website) | Token F1 | Reference patch compute ratio |
| --- | ---: | ---: | ---: |
| LLaVA-OneVision 0.5B Route31 | 0.119874 | 0.0141963683 | 0.867154133 |
| Qwen2.5-VL 3B t03 | 1.58991 | 0.0995836946 | 0.885658759 |
| Qwen2.5-VL 7B route floor | 1.59148 | 0.0489086904 | 0.847719372 |

Read the [score/publication audit](docs/PUBLICATION_AUDIT.md) and
[dataset and manuscript audit](docs/DATASET_RELEASE.md) before citing these
as paper-table reproductions. The bundled manuscript has unresolved count,
judge, and FPS-definition differences. LLaVA-OneVision 7B did not pass its MOS
gate and is not a promoted GRT result. Qwen's archived website baselines differ
from the matched controls; do not attribute their whole score gap to GRT.

Code and numeric evidence can be inspected without data access. Full inference
requires separately licensed videos and authorized HF dataset access. At the
initial access audit, Educational was gated and High-Motion was inaccessible to
both tested callers. Owner-authenticated follow-up confirmed that High-Motion
is private, at revision `d44407f607fdf020c59b816884f06ed6d453cf26`.
This repository does not grant those rights or promise a public download.

## Rebuild the complete leaderboard (CPU, no data or model downloads)

```bash
git clone --branch release/dive-bench-minimal --single-branch \
  https://github.com/Hai-chao-Zhang/DenseVideoUnderstand.git DIVE-Bench
cd DIVE-Bench
python -m pip install 'PyYAML>=6'
python -m tools.densevideo.build_complete_leaderboard --verify-only
python -m tools.densevideo.build_complete_leaderboard --output outputs/complete
```

Open `outputs/complete/leaderboard.html`. This rebuilds the current 47-result
view (29 Educational and 18 protocol-screened High-Motion methods), plus all 12
Educational GRT comparison rows. The CSV contains 59 records. Nine non-aligned
High-Motion runs are excluded from every table and CSV, not displayed in an
unranked section. The standalone HTML needs no JavaScript or external assets.
The [complete generation contract](docs/COMPLETE_LEADERBOARD.md) explains its
pinned evidence, checks and limits. This is archived-result verification, not
new GPU inference.

For the separate immutable 32-row **historical** snapshot:

```bash
python -m tools.densevideo.rebuild_published_leaderboard --verify-only
python -m tools.densevideo.rebuild_published_leaderboard --output outputs/historical
```

The historical verifier checks pinned SHA256 hashes, 634 identities per Educational method,
all three contracted quality floors, patch counts, the 1,000-item High-Motion
metrics, frozen CSV/website agreement, and exact Markdown regeneration. Add
`--website` to compare the website's retained historical `data/leaderboard.js`
against the pinned 32-row snapshot. It does not audit the complete HTML page's
new 47-result/12-control overlay; the complete command above verifies that view.
An existing output directory is refused. This
bundle contains numeric scores and hashed identities, not reference answers,
predictions, videos, or credentials. Non-GRT rows are preserved from the public
snapshot; their inference is not reproduced by the minimal profiles.

An installed wheel provides both dated evidence bundles as package data,
generated from the canonical `release/` tree during the build. Both
`dive-leaderboard-complete --verify-only` and `dive-leaderboard --verify-only`
work outside a checkout; `--bundle` remains available for an explicit bundle.

## Install the evaluator

Use a dedicated Python 3.10 or 3.11 environment with a compatible CUDA/PyTorch
installation. Do not install this distribution alongside upstream `lmms-eval`:
both own the `lmms_eval` Python namespace.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m lmms_eval --help
python -m pytest -q
```

The supported GRT profiles pin Transformers 4.57.6, PyTorch 2.9/torchvision 0.24,
and qwen-vl-utils 0.0.14. They use PyAV and do not require Decord or a local
LLaVA-NeXT checkout. The separate historical `llava_ov_dense_video` High-Motion
wrapper requires Decord and LLaVA-NeXT; its complete historical environment and
model revision were not frozen, so exact new-inference equivalence is unproven.

## Data and task names

| Paper task | Registered task | Examples |
| --- | --- | ---: |
| Educational High-FPS Videos | `dive_bench_educational_high_fps` | 634 QA / 317 videos |
| High-Motion High-FPS Videos | `dive_bench_high_motion_high_fps` | 3,243 |
| High-Motion historical preview | `dive_bench_high_motion_high_fps_preview1000` | first 1,000 |

`densevideo` and `densevideo_highmotion` are compatibility aliases. The latter
defaults to 1,000 but retains its old environment override; the canonical
preview ID is fixed at 1,000. Do not mix full-split and preview scores.

The owner-configured [DIVE-Bench Hub entry](https://huggingface.co/datasets/haichaozhang/DIVE-Bench)
now has `educational_high_fps` and `high_motion_high_fps` test configurations at
revision `d80461fccf879d5efdeece0edce8608a72d64f10`. This annotation-only entry is
currently **private**, preserving the high-motion source's access boundary.
The old source cards expose their corresponding named task too; source video
archives and the evaluator's historical data pins are unchanged.

Accept the [Educational access agreement](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)
and authenticate with `hf auth login`. Its pinned revision is
`5cc61a045c8e5e95d1d9c87e22ccd0f699575aea`. Load **only** `LPM_videos.parquet`:
`LPM_slides.parquet` is the same file and would double-count all questions.
High-Motion is referenced at
[`haichaozhang/highmotion_densevideounderstand`](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand).
Obtain access from its owner; inaccessible data must not be replaced silently.

The loader downloads/extracts the authorized video archive into the HF cache.
Archive extraction accepts only regular files and directories, rejecting path
traversal, links, and special files before extraction begins.
For existing videos, set `DENSEVIDEO_DATA_ROOT` to a directory containing
`DenseVideo-LPM/videos/<video>.mp4` and/or `egodex/<action>/<clip>.mp4`.
Never flatten High-Motion filenames: its numeric clip ids repeat across actions.
Missing files cause an error. Dataset licenses are distinct from the code
licenses; see the [source terms and completed Hub configuration](docs/DATASET_RELEASE.md).

Example Educational evaluation (one item is a smoke test only):

```bash
python -m lmms_eval --model llava_hf \
  --model_args pretrained=llava-hf/llava-onevision-qwen2-0.5b-ov-hf,revision=74dd0bf867a4cda7950c17663794267c60cf4b40,device_map=auto,dtype=bfloat16,max_frames_num=8,video_decode_backend=pyav_seek \
  --tasks dive_bench_educational_high_fps --batch_size 1 --limit 1 \
  --gen_kwargs max_new_tokens=128,temperature=0 \
  --log_samples --output_path outputs/educational-smoke
```

High-Motion target sampling defaults to eight endpoint-inclusive uniform frames.
Any new frame-budget experiment must set `DENSEVIDEO_HIGHMOTION_NUM_FRAMES`
and the model's frame budget consistently and report the changed protocol.
Canonical Educational tasks report objective metrics only. Open MOS is a
separate pinned judge; disabled inline GPT scoring is not a zero MOS result.

## Reproduce GRT with matched controls

The profile runner prints a plan by default and writes nothing:

```bash
python -m tools.densevideo.reproduce_grt --profile route31 --output outputs/route31
```

After data and model access are available, select one GPU and run all three
arms serially on that same device. `--with-mos` additionally runs the pinned
Qwen3-VL-32B text judge and requires sufficient GPU memory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.densevideo.reproduce_grt \
  --profile route31 --output outputs/route31-full --execute --with-mos
```

Repeat with `--profile qwen3` or `--profile qwen7` and a fresh output directory.
Use `--limit 2` for a smoke test covering subtitle and OCR; it is not
published-score reproduction.
Model revisions, eight-frame sampling, thresholds, generation caps (128/128/48),
and the judge revision are recorded in
[`profiles.json`](tools/densevideo/profiles.json). Published MOS uses batch 8,
cap 64 and 6,000-character trimming, which differ from the general scorer's
defaults. See [reproduction safeguards](docs/REPRODUCTION.md).

Patch compute ratios measure first-layer visual patch projections, not total
model FLOPs. `effective_fps` means temporal sampling density, not speed;
`throughput_fps` is measured processing throughput. Eight sampled frames do not
by themselves establish high-FPS temporal coverage or statistical significance.

## Citation and licenses

The public arXiv paper describes the earlier scope; the bundled manuscript adds
the High-Motion task. State which version/protocol is used.

```bibtex
@article{zhang2025dive,
  title={Dense Video Understanding with Gated Residual Tokenization},
  author={Zhang, Haichao and Chai, Wenhao and He, Shwai and Li, Ang and Fu, Yun},
  journal={arXiv preprint arXiv:2509.14199},
  year={2025}
}
```

Evaluation infrastructure derives from [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
Its original notices are preserved in [LICENSE](LICENSE): the core pipeline is
MIT, while the model/task additions are Apache-2.0. Dataset assets retain their
separate upstream terms. This release modifies model/task registration, optional
decoder imports, split selection and file-resolution safeguards; see the audit.
