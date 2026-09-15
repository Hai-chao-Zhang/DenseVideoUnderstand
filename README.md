# 🤿 DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION

### Dense Information Video Evaluation (DIVE) Benchmark

<p align="center">
  <a href="https://arxiv.org/abs/2509.14199">
    <img src="https://img.shields.io/badge/ArXiv-2509.14199-red?style=for-the-badge&logo=arxiv" alt="ArXiv paper" />
  </a>
  <a href="paper/ECCV_Dense_Video_Understanding.pdf">
    <img src="https://img.shields.io/badge/Paper-Revised_Manuscript-8b5cf6?style=for-the-badge" alt="Revised manuscript" />
  </a>
  <a href="https://www.zhanghaichao.xyz/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=google-chrome" alt="Project website" />
  </a>
  <a href="https://www.zhanghaichao.xyz/DenseVideoUnderstand/leaderboard.html">
    <img src="https://img.shields.io/badge/Results-Leaderboard-218c74?style=for-the-badge" alt="Leaderboard" />
  </a>
  <a href="https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d?style=for-the-badge&logo=huggingface" alt="Educational dataset; access agreement required" />
  </a>
</p>

## 👥 Authors

<p align="center">
  <a href="https://zhanghaichao.xyz"><b>Haichao Zhang<sup>1</sup></b></a> ·
  <a href="https://wenhaochai.com/"><b>Wenhao Chai<sup>2</sup></b></a> ·
  <a href="https://shwai-he.github.io/"><b>Shwai He<sup>3</sup></b></a> ·
  <a href="https://www.ang-li.com/"><b>Ang Li<sup>3</sup></b></a> ·
  <a href="https://www1.ece.neu.edu/~yunfu/"><b>Yun Fu<sup>1</sup></b></a>
</p>
<p align="center">
  <sub><b>1</b> Northeastern University &nbsp;|&nbsp; <b>2</b> Princeton University &nbsp;|&nbsp; <b>3</b> University of Maryland, College Park</sub>
</p>

**DIVE-Bench** evaluates video-language models on tasks where useful evidence is
distributed densely over time. **Gated Residual Tokenization (GRT)** explores how
to reduce redundant visual-token computation while retaining changing content.
This repository provides the benchmark tasks, GRT implementations, evaluation
tools, and reproducible numerical leaderboards.

<p align="center">
  <img src="assets/DIVE.jpeg" alt="DIVE project illustration: a diver exploring video information" width="1080" />
</p>

The [arXiv paper](https://arxiv.org/abs/2509.14199) introduces the earlier
Educational scope; the [revised manuscript](paper/ECCV_Dense_Video_Understanding.pdf)
includes both Educational and High-Motion tasks.

## 📅 News

- **2026/09/15** — Evaluation and GRT code are available on `main`, together with
  the complete leaderboard and the corrected-reference High-Motion v2 preview.
- **2026/09/14** — DIVE-Bench/GRT integration submitted to
  [VLMEvalKit #1686](https://github.com/open-compass/VLMEvalKit/pull/1686) and
  [lmms-eval #1521](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1521).
  Both are open draft submissions, not merged upstream.
- **2025/09/18** — Initial DIVE Educational test-split release.

## 🔍 What is DIVE?

**Dense Information Video Evaluation (DIVE)** studies video understanding beyond
isolated keyframes. Educational videos require recovering information from
subtitles and visual text, while high-motion videos test fine-grained spatial
trajectories across time.

| Paper task | Evaluation task ID | Scope |
| --- | --- | --- |
| Educational High-FPS Videos | `dive_bench_educational_high_fps` | 634 QA / 317 videos |
| High-Motion High-FPS Videos | `dive_bench_high_motion_high_fps` | 3,243 examples |

The fixed High-Motion preview uses
`dive_bench_high_motion_high_fps_preview1000` (the first 1,000 examples).
Preview and full-split results are separate protocols.

### Datasets

- [Educational High-FPS Videos](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)
- [High-Motion High-FPS Videos](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand)
- [Two-task DIVE-Bench entry](https://huggingface.co/datasets/haichaozhang/DIVE-Bench):
  `educational_high_fps` and `high_motion_high_fps` test configurations.

Data access is separate from this public code release. At the release audit,
Educational was gated, and the High-Motion source and combined annotation entry
were private. Obtain authorized access and authenticate with `hf auth login`.
Videos and dataset rights are not redistributed by this repository. See
[data access, pinned revisions, and source terms](docs/DATASET_RELEASE.md).

Use only `LPM_videos.parquet` for Educational evaluation: `LPM_slides.parquet`
duplicates its questions. For existing videos, set `DENSEVIDEO_DATA_ROOT` to
the root containing `DenseVideo-LPM/videos/<video>.mp4` and/or
`egodex/<action>/<clip>.mp4`; preserve action directories because clip IDs repeat.
High-Motion v2 requires the [corrected-reference recipe](docs/HIGHMOTION_V2_REPRODUCTION.md),
not the legacy GT scores emitted by the original generation task.

## 🔍 What is GRT?

**Gated Residual Tokenization** uses changes between sampled frames to guide
visual patch computation. The released implementations combine motion-based
gating with reuse of previous patch embeddings; fixed model-specific profiles
define thresholds, routing, and generation budgets.

The Educational reproduction runner includes LLaVA-OneVision **0.5B**
(`route31`), Qwen2.5-VL **3B** (`qwen3`), and Qwen2.5-VL **7B** (`qwen7`), with
baseline, all-patch control, and GRT arms. High-Motion has a separate fixed
**HF LLaVA-OneVision 0.5B GRT** recipe. See the
[Educational profiles](tools/densevideo/profiles.json) and
[High-Motion reproduction guide](docs/HIGHMOTION_V2_REPRODUCTION.md).

## ⚙️ Quick start

### Generate the leaderboard — CPU only

No dataset, model download, or GPU is needed to verify and export the published
numeric evidence:

```bash
git clone --branch main --single-branch \
  https://github.com/Hai-chao-Zhang/DenseVideoUnderstand.git DIVE-Bench
cd DIVE-Bench
python -m pip install 'PyYAML>=6'
python -m tools.densevideo.build_complete_leaderboard --verify-only
python -m tools.densevideo.build_complete_leaderboard --output outputs/complete
```

Open `outputs/complete/leaderboard.html`. It contains **29 Educational results,
19 High-Motion v2 preview results, and 12 Educational comparison rows** (60 CSV
records, not 60 unique methods). Choose a new output directory for each export.
This reconstructs saved numerical results; it does not run model inference.
See the [complete leaderboard guide](docs/COMPLETE_LEADERBOARD.md).

### Install the evaluator

From the cloned repository, use a dedicated Python **3.10 or 3.11** environment
and a compatible CUDA/PyTorch installation for GPU evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,reference]'
python -m lmms_eval --help
python -m pytest -q
```

Do not install this distribution alongside upstream `lmms-eval`: both own the
`lmms_eval` namespace. Supported profiles use PyAV, Transformers 4.57.6,
PyTorch 2.9 / torchvision 0.24, and qwen-vl-utils 0.0.14. See
[environment and reproduction requirements](docs/REPRODUCTION.md#installation).

### Evaluate DIVE-Bench

With authorized data and model access, run an Educational smoke test:

```bash
python -m lmms_eval --model llava_hf \
  --model_args pretrained=llava-hf/llava-onevision-qwen2-0.5b-ov-hf,revision=74dd0bf867a4cda7950c17663794267c60cf4b40,device_map=auto,dtype=bfloat16,max_frames_num=8,video_decode_backend=pyav_seek \
  --tasks dive_bench_educational_high_fps --batch_size 1 --limit 1 \
  --gen_kwargs max_new_tokens=128,temperature=0 \
  --log_samples --output_path outputs/educational-smoke
```

This one-item run checks execution, not published-score reproduction.
Open MOS requires the separate pinned judge. For High-Motion, follow the
[source verification, GRT generation, and v2 scoring steps](docs/HIGHMOTION_V2_REPRODUCTION.md).

### Reproduce GRT

Inspect an Educational three-arm plan without downloading models or using a GPU:

```bash
python -m tools.densevideo.reproduce_grt --profile route31 --output outputs/route31
```

After data and model access are available, execute on one selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.densevideo.reproduce_grt \
  --profile route31 --output outputs/route31-full --execute --with-mos
```

Use `--profile qwen3` or `--profile qwen7` for the other Educational models.
`--with-mos` also runs the pinned Qwen3-VL-32B text judge and needs sufficient
GPU memory. The runner executes matched arms serially; `--limit 2` is a smoke
test, not a leaderboard result. Full settings and validation are in the
[reproduction guide](docs/REPRODUCTION.md).

## 📊 Results and reproducibility

See the [interactive leaderboard](https://www.zhanghaichao.xyz/DenseVideoUnderstand/#leaderboard)
or [complete HTML tables](https://www.zhanghaichao.xyz/DenseVideoUnderstand/leaderboard.html).

On the fixed **1,000-example High-Motion v2 preview**, HF 0.5B GRT improves
Grid Accuracy from **0.0468682595 to 0.0495036226** and Token F1 from
**0.0418983286 to 0.0450524706**. ADE and FDE improve too; **Transition Accuracy
decreases**. The comparison uses 18 CPU-rescored archived baselines and one new
GRT GPU run, with the same corrected right-hand references and validity masks.
Of 1,000 records, 861 have valid references (6,015 of 8,000 sampled positions).
These are preview point estimates, not full 3,243-example results or a
statistical-significance claim. Baseline inference was not repeated, and missing
historical weight/tensor receipts prevent a newly rerun byte-identical paired
claim. [All five metrics, coverage, and execution details →](docs/HIGHMOTION_V2_RESULTS.md)

Educational results reconstruct the **2026-08-20 website snapshot**, not a fresh
full GPU rerun. The three supported profiles passed a
[bounded GPU smoke check](docs/GPU_SMOKE_VALIDATION.md). Archived website baselines
and matched controls are different comparisons; their whole score gap cannot
be attributed to GRT. Eight sampled frames do not establish high-FPS coverage,
and patch-compute ratios are not total FLOPs or throughput. Paper-version,
count, judge, and protocol differences are documented in the
[publication audit](docs/PUBLICATION_AUDIT.md) and
[dataset/manuscript notes](docs/DATASET_RELEASE.md).

## 📜 Citation

If you find DIVE-Bench or GRT useful, please cite our work and state the
paper version and evaluation protocol used:

```bibtex
@article{zhang2025dive,
  title={Dense Video Understanding with Gated Residual Tokenization},
  author={Zhang, Haichao and Chai, Wenhao and He, Shwai and Li, Ang and Fu, Yun},
  journal={arXiv preprint arXiv:2509.14199},
  year={2025}
}
```

## ⚖️ Acknowledgments and licenses

Our evaluation infrastructure builds on
[LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Its original MIT and
Apache-2.0 component notices are retained in [LICENSE](LICENSE) and
[LICENSE-APACHE](LICENSE-APACHE). The original project's BSD-3-Clause notice for
Haichao Zhang is retained in [LICENSE-BSD](LICENSE-BSD). These notices do not
replace one another; datasets and source videos retain their separate terms.
