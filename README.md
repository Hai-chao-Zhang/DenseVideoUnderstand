# 🤿 **DENSE VIDEO UNDERSTANDING WITH GATED RESIDUAL TOKENIZATION**
### **Dense Information Video Evaluation (DIVE) Benchmark**

## Project website

### Public audit update — 15 September 2026

The current minimal evaluation/GRT code and corrected High-Motion release are on
[`fix/highmotion-target-v2`](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/fix/highmotion-target-v2).
Framework integrations are submitted Draft PRs:
[VLMEvalKit #1686](https://github.com/open-compass/VLMEvalKit/pull/1686) and
[lmms-eval #1521](https://github.com/EvolvingLMMs-Lab/lmms-eval/pull/1521), not merged/accepted.
Historical README instructions below are retained as release-development history;
use the current release branch's installation and reproduction instructions.

`leaderboard.html` provides the current leaderboard without JavaScript:
29 Educational results, 19 High-Motion v2 results, and 12 educational GRT
comparison rows (60 CSV records; the controls are not 12 additional methods).
High-Motion v2 uses the **fixed first 1,000 source records, not the full 3,243-record
evaluation**. Questions and row order are unchanged. Corrected references name
`rightRingFingerMetacarpal` explicitly as the right-palm/ring-finger-base proxy;
invalid reference positions are masked without shifting later predictions.
Each metric shows its own scored-row and slot/edge coverage; undefined values
are not zero. See the
[reference contract](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/blob/fix/highmotion-target-v2/docs/HIGHMOTION_REFERENCE_V2.md).

The v2 comparison contains 18 archived baselines rescored on CPU and one new
fixed HF 0.5B GRT run. GRT's primary Grid Accuracy exceeds its corresponding
archived HF 0.5B baseline under the same corrected-reference scoring, but
**Transition Accuracy regresses**. The release retains every metric; it does
not claim universal improvement or statistical significance. Archived baseline
weight revisions and consumed input-tensor identity remain unproven: this is
not a freshly rerun byte-identical paired experiment.

The byte-identical 32-row `data/leaderboard.js` snapshot and complete 27-run
`data/highmotion-audit.json` remain historical evidence. The latter retains its
earlier 18 protocol-screened candidates and nine protocol exclusions; their old
scores remain withheld and are not mixed into the corrected v2 cohort. The
browser still ignores cached or injected legacy High-Motion overlays and accepts
only the separately authenticated v2 summary. Educational numbers are unchanged.

The educational comparison exposes all 12 candidate/control rows, including
archived baselines, quality baselines and all-patch controls. Qwen 3B GRT improves
Open MOS/Token F1 and patch reuse but has lower mean throughput than its all-patch
control. The page does not claim every quality/efficiency metric improves.

All current leaderboard and evidence assets come from the release branch's canonical
`tools.densevideo.build_complete_leaderboard` generator. It verifies the pinned
release manifests, independently recomputes all 12 educational quality means
from 7,608 numeric records, and verifies the unchanged 27-run High-Motion audit
while withholding its old scores. The separate
[15 September numeric bundle](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/fix/highmotion-target-v2/release/2026-09-15)
is authenticated and its v2 numeric contributions are reaggregated, without
rerunning model inference or source projection. Its manifest SHA-256 is
`1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc`.
Website JSON/CSV are outputs, not editable numeric sources. The JavaScript-free
HTML includes its own CSS and can be viewed offline.

From the website checkout, clone the code separately (replace the example
destination), then regenerate and check the site with Python/PyYAML:

```bash
git clone --branch fix/highmotion-target-v2 --single-branch \
  https://github.com/Hai-chao-Zhang/DenseVideoUnderstand.git /path/to/DIVE-Bench
export DIVE_MINIMAL_ROOT=/path/to/DIVE-Bench
python -m pip install 'PyYAML>=6'
python scripts/build_audit_page.py
python scripts/build_audit_page.py --check
python -m unittest discover -s tests -v
python -m pytest tests -q
```

Alternatively pass `--minimal-root /path/to/DIVE-Bench` to the wrapper, or install
the current release package. The pinned High-Motion v2 bundle is included by
default; no extra v2 flags are needed. This numeric rebuild requires no GPU,
model download, dataset access or token.

From the code checkout, `python -m tools.densevideo.build_complete_leaderboard
--output /path/to/new-directory` creates the same current offline view; an
existing output directory is refused. For the explicitly historical 41-record
reference-hold view, use `python -m tools.densevideo.build_complete_leaderboard
--legacy-reference-hold --output /path/to/new-legacy-directory`. Do not use that
legacy output to replace the current site. Original release evidence remains
unchanged.

High-Motion source data remain private/access-controlled, and video/source
licenses must be respected. Public numerical evidence does not grant source
data access or redistribution rights. Actual reference construction and GPU
generation require authorized inputs; use the
[v2 reproduction recipe](https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/blob/fix/highmotion-target-v2/docs/HIGHMOTION_V2_REPRODUCTION.md),
including its isolated original-annotation cache and model-byte checks.

Four tests that inspect external, mutable Qwen3/LLaVA7 historical campaign trees are
opt-in via `DIVE_RUN_ARCHIVED_CAMPAIGN_TESTS=1`; all synthetic validator, UI and
new protocol-cohort checks run by default. When opting in, explicitly set
`DIVE_QWEN3_ARCHIVED_CONTRACT` to the Qwen3 contract JSON and
`DIVE_LLAVA7_ARCHIVED_FAILURE_ROOT` to the LLaVA7 failure-evidence directory.
There are no machine-specific default paths: missing configuration fails clearly
instead of silently skipping an explicitly requested check. The Qwen3 completion
must be `completed.json` alongside its configured contract. Those four archival checks currently
fail against legacy manifest paths/checksums and are **not claimed to pass**.
The portable release's frozen numeric evidence is independently checked by its
own verifier. The new High-Motion GPU run covers the fixed 1,000-record preview;
this does not establish full 3,243-record High-Motion evaluation or a fresh rerun
of every Educational model and judge.

Deployment source is this repository's `webpage` branch (GitHub Pages), not
`main` or the minimal evaluation branch. The original live deployment was
`98f8bb8d3ee7b67e389acc7aa108a250957e5a34`; update only reviewed website files.

### Historical release builder

The webpage branch is a dependency-free static research site. It separates content, presentation, interaction, and the published result snapshot into index.html, styles.css, app.js, and data/leaderboard.js.

Preview it locally from the repository root:

    python -m http.server 8000

Then open http://localhost:8000. Leaderboard values live in data/leaderboard.js and are synchronized from verified release artifacts with `scripts/sync_leaderboard.py`.

Run the dependency-free site contract checks with:

    python -m unittest discover -s tests -v

The Qwen dual-route GRT wrapper lives in the evaluation repository as an
additive `lmms_eval` plugin. In that repository, install the package and enable
the plugin before using the `qwen2_5_vl_dual_route` model name:

    python -m pip install -e .
    export LMMS_EVAL_PLUGINS=densevideo_qwen_dual_plugin

For an uninstalled source checkout, also export the repository root on
`PYTHONPATH`. The environment variable remains required after installation;
it makes plugin activation explicit and keeps the upstream registry additive.

### Publish an additive release-v2 snapshot

Release v2 is an honest mixed-disposition contract. It always audits the four
families `route31`, `llava7`, `qwen3`, and `qwen7`. A `promoted` family carries
a passed full selection and may add or replace a leaderboard row; a
`not_promoted` family remains visible in provenance with hashed failure
evidence but contributes no candidate row.

The frozen baseline contains 27 LPM rows. Route31 replaces its legacy 0.5B GRT
row, Qwen2.5-VL 3B adds one verified row, LLaVA7 remains `not_promoted`, and
Qwen7 contributes a row only if its terminal 634-sample completion passes.
Consequently, the v2 manifest computes the final LPM count as 28 or 29; neither
the builder nor the site sync should hard-code that choice.

The published 20 Aug 2026 snapshot records Qwen7 as `promoted`, so the current
site contains 29 LPM rows, three verified GRT methods, and one explicit
`not_promoted` LLaVA7 family audit.

Create the manifest in the evaluation repository only after Qwen7 has a
terminal result. If its full completion passes, use the completion form:

    python tools/densevideo/create_grt_release_manifest_v2.py \
      --route31-selection /path/to/hf_route31_full/selection.json \
      --llava7-failed-root /path/to/llava7_full_campaign \
      --qwen3-completed /path/to/qwen3_full/completed.json \
      --qwen7-completed /path/to/qwen7_full/full_completed.json \
      --output /path/to/release/dive_grt_multifamily_release_v2.json

If the terminal Qwen7 gate fails, use the failure form instead; both failure
arguments are required together:

    python tools/densevideo/create_grt_release_manifest_v2.py \
      --route31-selection /path/to/hf_route31_full/selection.json \
      --llava7-failed-root /path/to/llava7_full_campaign \
      --qwen3-completed /path/to/qwen3_full/completed.json \
      --qwen7-failed-root /path/to/qwen7_full_campaign \
      --qwen7-failed-selection /path/to/qwen7_full/selection.json \
      --output /path/to/release/dive_grt_multifamily_release_v2.json

Build the release with the atomic legacy MOS completion. `--summary-csv` is
repeatable when the 27 baseline summaries are split across files:

    python tools/densevideo/build_public_leaderboard_complete.py \
      --model-config configs/densevideo/leaderboard_models_extended.yaml \
      --summary-csv /path/to/release_inputs/baseline_summaries.csv \
      --output-base /path/to/evaluation_outputs \
      --output-dir /path/to/release \
      --expected-mos-judge-model Qwen/Qwen3-VL-32B-Instruct \
      --expected-mos-judge-revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb \
      --expected-mos-judge-fingerprint 64c644b2bd2696677afd3507a6bd6efa03e902d712c0e55e26ab9ab144c0a76d \
      --grt-release-manifest /path/to/release/dive_grt_multifamily_release_v2.json \
      --legacy-open-mos-completed /path/to/legacy_open_mos_rejudge_v1/completed.json

The completion injects exactly nine disjoint 634-row legacy score files. The
builder records their completion, contract, matrix, method paths, row counts,
and hashes in `leaderboard_sources.json`. Do not combine
`--legacy-open-mos-completed` with manual `--open-mos-csv` or `--mos-dir`
arguments. Omitting `--expected-lpm-rows` and `--expected-highmotion-rows` is
intentional: the builder reads and validates both counts from the v2 manifest.

Stage the site payload rather than overwriting the published snapshot during
review:

    python scripts/sync_leaderboard.py \
      --site-data data/leaderboard.js \
      --leaderboard-csv /path/to/release/leaderboard.csv \
      --metadata-json /path/to/release/leaderboard_sources.json \
      --grt-release-manifest /path/to/release/dive_grt_multifamily_release_v2.json \
      --output data/leaderboard.next.js

The site independently re-hashes the manifest, all four audit dispositions,
the promoted artifacts, every `not_promoted` evidence bundle, and the legacy
MOS completion before writing. Omitting `--expected-lpm-count` and
`--expected-highmotion-count` is also intentional: sync derives them from the
same manifest. Validate the staged result before replacing
`data/leaderboard.js`:

    node --check data/leaderboard.next.js
    python -m unittest discover -s tests -v

### Legacy v1 compatibility

The original one-selection workflow remains available for an authenticated
historical snapshot by using `--selection-json` instead of
`--grt-release-manifest`. The legacy
`create_grt_multifamily_release_manifest.py` v1 schema is also still accepted
for previously produced artifacts, but it means that all four families passed
and therefore has a fixed 30-row LPM contract. It cannot represent
`not_promoted` families and must not be used for a new mixed pass/fail release.



<p align="center">
  <a href="https://arxiv.org/pdf/2509.14199">
    <img src="https://img.shields.io/badge/ArXiv-2509.14199-red?style=for-the-badge&logo=arxiv" alt="ArXiv"/>
  </a>
  <a href="https://zhanghaichao.xyz/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=google-chrome" alt="Website"/>
  </a>
  <a href="https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-ffcc4d?style=for-the-badge&logo=huggingface" alt="HuggingFace Dataset"/>
  </a>
  <a href="https://github.com/hai-chao-zhang/DenseVideoUnderstand/">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>

The **first-ever benchmark** dedicated to **Dense Video Understanding**, focusing on **QA-driven high-frame-rate** comprehension where **answer-relevant information** appears **in nearly every frame**.

---

<p align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/66393f5a1231260674ae798e/uOmH6pKW5yqk6PstJ4H8R.jpeg"
     alt="DIVE" width="1080">
</p>

---

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
<p align="center">
  <img src="https://brand.northeastern.edu/wp-content/uploads/2025/01/seal-yellow.svg" height="60" alt="NEU Seal"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Northeastern_University_wordmark.svg" height="30" alt="NEU Wordmark"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Princeton_University_Shield.svg" height="60" alt="Princeton Shield"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/Princeton_text_logo.svg" height="32" alt="Princeton Wordmark"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://prg.cs.umd.edu/img/logo/umd-logo-transparent.png" height="60" alt="UMD Logo"/>
  <img src="https://commons.wikimedia.org/wiki/Special:FilePath/University_of_Maryland_wordmark.svg" height="32" alt="UMD Wordmark"/>
</p>



---

## 📅 **Timeline**

| Date | Status | Description |
|------|--------|-------------|
| **2025/09/18** | ✅ | Release the **DIVE benchmark (test split only)** |
| *TBD* | ✅ | Release **evaluation/test code** on [GitHub](https://github.com/hai-chao-zhang/DenseVideoUnderstand/) |
| *TBD* | ⭕ | Merge DIVE into **LMMS-EVAL** VLM test kit |
| *TBD* | ⭕ | Release **multi-FPS versions** of the dataset |
| *TBD* | ⭕ | Add **diverse dense video task categories** |
| *TBD* | ⭕ | **Release full GRT model and training/inference code** |
| *Future Ideas* | 💡 | Contact us with suggestions for new tasks or collaborations |
---

## 🔍 What is DIVE?
**DIVE (Dense Information Video Evaluation)** is a benchmark designed for scenarios where useful content is densely distributed across frames (e.g., educational/lecture videos, surgical procedures, sign language). Existing VLLM pipelines downsample aggressively to control token cost, which **drops critical temporal details**.


## 🔍 What is GRT?
### GRT in a Nutshell (method overview)
**Gated Residual Tokenization (GRT)** is our token-efficiency framework:
1. **Motion-Gated Tokenization (inter-tokenization):** detect static regions via motion cues and **skip** them during tokenization → **sub-linear token/time growth** w.r.t. FPS.
2. **Semantic Scene Token Merging (intra-tokenization):** **merge redundant tokens** within scenes while preserving dynamic semantics.

For details, see the paper: [arXiv:2509.14199](https://arxiv.org/html/2509.14199).

> ⚠️ **Note:** The **benchmark (test)** is released now. **GRT model/implementation** will be released later.

---

## 🧪 Tasks
- **Dense Video QA (DIVE)** – question answering that requires **frame-dense reasoning**  
  ↳ Dataset on 🤗: **https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation**

Minimal loading example:
```python
from datasets import load_dataset
ds = load_dataset("haichaozhang/DenseVideoEvaluation", split="test")
print(ds[0])
```

---

## ⚙️ Usage (Evaluation via LMMS-EVAL)
We are preparing a PR to integrate DIVE into **[LMMS-EVAL](https://github.com/EvolvingLMMs-Lab/lmms-eval)**.

### Install LMMS-EVAL
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

### Run (example with LLaVA-OneVision; customize as needed)
```bash
accelerate launch   --num_processes=1   -m lmms_eval   --model llava_onevision   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen"   --tasks mme   --batch_size 1   --log_samples   --log_samples_suffix fps0.005   --output_path ./logs/   --verbosity=DEBUG
```

### Run (placeholder for our dense-video variant)
```bash
accelerate launch   --num_processes=1   -m lmms_eval   --model llava_ov_dense_video   --model_args "pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,use_gated_tok=True,use_vision_merge=False,profiling=False,dense_frame_fps=0.001"   --tasks mvbench   --batch_size 1   --log_samples   --output_path ./logs/   --verbosity=DEBUG
```

---

## 🗓️ Timeline
- ✅ **2025/09/18** – Release **DIVE benchmark (test split)**
- ⭕ Merge DIVE into **LMMS-EVAL** (PR in preparation)
- ⭕ Release **multi-FPS** variants of the dataset
- ⭕ Add **more dense-video task categories**
- ⭕ **Release full GRT model + training/inference code**
- 💡 Ideas or requests? Open an issue or reach out!

---

## 📎 Links
- 📄 Paper: [arXiv 2509.14199](https://arxiv.org/pdf/2509.14199)  
- 🤗 Dataset: [haichaozhang/DenseVideoEvaluation](https://huggingface.co/datasets/haichaozhang/DenseVideoEvaluation)  
- 🌐 Project: [Website](https://zhanghaichao.xyz/DenseVideoUnderstand/)  
- 💻 Repo: [GitHub](https://github.com/hai-chao-zhang/DenseVideoUnderstand/)

---

## 📜 Citation
If you find DIVE/GRT useful, please cite:
```bibtex
@article{zhang2025dive,
  title={Dense Video Understanding with Gated Residual Tokenization},
  author={Haichao Zhang and Wenhao Chai and Shwai He and Ang Li and Yun Fu},
  journal={arXiv preprint arXiv:2509.14199},
  year={2025}
}
```

## ⚖️ License
- **Dataset (DIVE)**: OpenRAIL (see dataset card for terms)  
- **Code**: to be announced with the model release
