# High-Motion v2: source verification and cached rescoring

These tools implement the [fixed reference and scoring contract](HIGHMOTION_REFERENCE_V2.md),
retained unchanged as the frozen pre-outcome protocol specification. Its earlier
hold language is historical; the completed fixed candidate and current comparison
are documented in [High-Motion v2 results](HIGHMOTION_V2_RESULTS.md). Running the
tools does not automatically approve a new result for publication.
Use authorized source data; this repository neither contains source videos nor
grants permission to redistribute them or derived annotations.

## Verify the sources and construct references (CPU)

Install `python -m pip install '.[reference]'`. Obtain the original authorized
`Egodex_traj.parquet` and `video.zip` at revision
`d44407f607fdf020c59b816884f06ed6d453cf26` of
[`haichaozhang/highmotion_densevideounderstand`](https://huggingface.co/datasets/haichaozhang/highmotion_densevideounderstand),
and the corresponding local EgoDex test HDF5 files. Replace the example local
paths below. All output directories must be new; their parents must exist.

```bash
dive-highmotion-bind \
  --annotation /data/highmotion/Egodex_traj.parquet \
  --annotation-sha256 518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd \
  --archive /data/highmotion/video.zip \
  --archive-sha256 ed06c8ef7e43f61901a11898dd5d4b9d265d411299c2174c772668e567f79e49 \
  --source-root /data/egodex/test \
  --output-dir outputs/highmotion-bindings
```

The binder actually reads and hashes the complete archive and all 3,243 required
HDF5 members, and checks their local counterparts. It does not extract the ZIP,
generate labels, or use model results. This can be substantial I/O: use an
appropriate compute environment, not a restricted shared login node.

Read `manifest_sha256` from `outputs/highmotion-bindings/binding_report.json`,
then supply that value as `BINDING_SHA256` below:

```bash
dive-highmotion-reference \
  --annotation /data/highmotion/Egodex_traj.parquet \
  --annotation-sha256 518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd \
  --source-root /data/egodex/test \
  --binding-manifest outputs/highmotion-bindings/source_bindings.json \
  --binding-manifest-sha256 BINDING_SHA256 \
  --output-dir outputs/highmotion-reference-v2
```

The resulting `high_motion_high_fps_v2.parquet` retains all 3,243 original rows
and questions. `build_report.json` records its checksum and full/eight-frame
reference coverage. Invalid slots have null answers, a mask and a reason.
Questions are not rewritten to accommodate a model's predictions.

## Generate the fixed HF 0.5B GRT candidate (GPU)

The released fixed HF 0.5B GRT candidate completed its 1,000-record generation;
the command below reproduces its fixed generation protocol, not a guarantee of
identical predictions on every runtime or GPU. It retains the original input
task so cached baseline predictions remain usable. Obtain authorized dataset/model access and
set the media root to the directory containing `egodex/<action>/<id>.mp4`.
The first-stage result's legacy GT metrics are **not v2 scores**.

Use the dedicated environment described in [REPRODUCTION.md](REPRODUCTION.md#installation)
and run from this verified checkout, without an inherited `PYTHONPATH` pointing
at another lmms-eval installation. The generation runtime recorded Python
3.10.19, Torch 2.9.0+cu126, Transformers 4.57.6, PyAV 15.1.0, Datasets 4.5.0,
Accelerate 1.12.0, Hugging Face Hub 0.36.2, Pillow 12.1.1 and NumPy 2.2.6.
Broader package compatibility ranges do not promise identical predictions.

First prepare a **new task cache**, separate from the authorized model hub cache.
Replace both `/data/...` paths. While authorized network access is available,
the following prewarms the exact model revision and checks its weight,
processor, tokenizer and configuration bytes, as well as the original Parquet.
It can download about 1.8 GB and hashes the weight file; use an appropriate
compute environment. Authentication, if needed, uses your existing local HF
login; do not put tokens into this command or its logs.

```bash
set -euo pipefail
export HM_ORIGINAL_PARQUET=/data/highmotion/Egodex_traj.parquet
export HM_MODEL_HUB=/data/huggingface/hub
HM_RUN_CACHE=$(mktemp -d "${TMPDIR:-/tmp}/dive-hm-v2.XXXXXXXX")
export HM_RUN_CACHE
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ANNOTATION_SHA256 = "518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd"
MODEL_REVISION = "74dd0bf867a4cda7950c17663794267c60cf4b40"
MODEL_FILE_SHA256 = {
    "model.safetensors": "07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e",
    "added_tokens.json": "33e0fb93dadacb864bd2f2e8441e147daa2baceb67f94d3ef5283b495572cea0",
    "chat_template.json": "2466d1704df30f0067f28d8e30e0190a1bf74e5b430942697af974d162a056bd",
    "config.json": "839a4fba0bd6949f0db22d4f840935cb0318f6ac28b29c9ce1a5b15735a4a740",
    "generation_config.json": "89dc53229f50b59570b6852056dafeac8116c458f1a748bff491b6d4d24d3b51",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "preprocessor_config.json": "3644c108b9f0fa53e62ff422a9be6639642f0e64dab4a71f961c7911d4386384",
    "processor_config.json": "04e9899e93f2a412c94e153cab4081457c9a44defb3b2c0b9df673d42c42cdd0",
    "special_tokens_map.json": "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b",
    "tokenizer.json": "3c0ce3213b50ff38d8aa1e91136a2d2cb142a3f569246170872e439cb2a29d15",
    "tokenizer_config.json": "494a5592a446535be00acc531ccf7a53fd6c6c392c122d444c389160261572e0",
    "video_preprocessor_config.json": "0ea9b672282b78353960b6069a521ed496a9f07c033e1e3362bff669234caa8d",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}

def checked_sha256(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("SHA-256 mismatch: " + str(path))

annotation = Path(os.environ["HM_ORIGINAL_PARQUET"]).resolve(strict=True)
checked_sha256(annotation, ANNOTATION_SHA256)
model_hub = Path(os.environ["HM_MODEL_HUB"]).expanduser().resolve()
snapshot = Path(snapshot_download(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", revision=MODEL_REVISION,
    cache_dir=str(model_hub), allow_patterns=list(MODEL_FILE_SHA256),
))
if snapshot.name != MODEL_REVISION:
    raise RuntimeError("Unexpected model snapshot revision")
for name, expected in MODEL_FILE_SHA256.items():
    checked_sha256(snapshot / name, expected)

run_cache = Path(os.environ["HM_RUN_CACHE"]).resolve(strict=True)
task_cache = run_cache / "hfhome" / "highmotion_densevideounderstand"
task_cache.mkdir(parents=True, exist_ok=False)
(task_cache / "Egodex_traj.parquet").symlink_to(annotation)
for name in ("datasets", "modules", "assets", "xet", "torch", "xdg", "cuda", "tmp"):
    (run_cache / name).mkdir(exist_ok=False)
print("Original annotation and pinned model verified; isolated task cache ready.")
PY
```

Do not proceed after a failed check. Keep the source Parquet and checked model
files unchanged for the run. The task-local Parquet symlink is intentional:
the legacy loader searches local files before the Hub revision, and disabling
prepared Arrow alone does not prevent a different cached Parquet being chosen.
This fresh cache also lets the existing-video-cache check skip a redundant
dataset snapshot download; actual videos still come from `DENSEVIDEO_DATA_ROOT`.

Run the following in the **same shell**, after preparation succeeds. All task
caches are private to this attempt; only the separately populated model hub is
reused. Offline mode is enabled only now, after the pinned files are available.

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME="${HM_RUN_CACHE}/hfhome" \
HF_HUB_CACHE="${HM_MODEL_HUB}" HUGGINGFACE_HUB_CACHE="${HM_MODEL_HUB}" \
HF_DATASETS_CACHE="${HM_RUN_CACHE}/datasets" HF_MODULES_CACHE="${HM_RUN_CACHE}/modules" \
HF_ASSETS_CACHE="${HM_RUN_CACHE}/assets" HF_XET_CACHE="${HM_RUN_CACHE}/xet" \
TORCH_HOME="${HM_RUN_CACHE}/torch" XDG_CACHE_HOME="${HM_RUN_CACHE}/xdg" \
CUDA_CACHE_PATH="${HM_RUN_CACHE}/cuda" TMPDIR="${HM_RUN_CACHE}/tmp" \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
DENSEVIDEO_DATA_ROOT=/data/highmotion/media \
DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES=1000 \
DENSEVIDEO_HIGHMOTION_NUM_FRAMES=8 \
DENSEVIDEO_DISABLE_PREPARED_ARROW_CACHE=1 \
DENSEVIDEO_SKIP_VIDEO_SNAPSHOT_IF_CACHE_EXISTS=1 \
DENSEVIDEO_ENABLE_GPT_EVAL=0 DENSEVIDEO_FAST_TEXT_METRICS=0 \
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m tools.densevideo.deterministic_worker -- \
  --model llava_hf \
  --model_args pretrained=llava-hf/llava-onevision-qwen2-0.5b-ov-hf,revision=74dd0bf867a4cda7950c17663794267c60cf4b40,trust_remote_code=True,device_map=auto,dtype=bfloat16,max_frames_num=8,max_image_size=384,attn_implementation=eager,profiling=True,video_decode_backend=pyav_seek,prompt_router=off,use_gated_tok=True,gate_policy=motion,gate_metric=ssim,gate_diff_threshold=0.001,gate_projection_mode=linear_consistent,gate_refresh_interval_frames=0 \
  --tasks densevideo_highmotion --batch_size 1 \
  --seed 0,1234,1234,1234 \
  --gen_kwargs max_new_tokens=64,temperature=0 \
  --log_samples --output_path outputs/grt-hm-v2-predictions
```

This HF patch-gating implementation is not the separate native LLaVA checkpoint
or a scene-merging configuration. Preserve the resulting sample JSONL and its
original document metadata. No baseline inference command is needed below.
Keep both the isolated original-Parquet cache and prepared-Arrow-cache override
above: an old cached task must not replace the intended original annotation.
The compatibility task itself is not a checksum
validator; require all saved inputs to pass the separate source-bound rescoring
checks before treating its predictions as comparable results.

## Preserve cached inputs and score both methods identically

Baseline inference is unnecessary when complete original predictions exist.
Require the original `doc_id`, document metadata, one `filtered_resps` string
and actual effective `input` for every preview row. The verifier checks all
1,000 records against the unchanged original questions and one common prompt
cache. Aggregate scores alone are insufficient.

Use the Parquet checksum from `build_report.json` for `REFERENCE_SHA256` and
the actual file checksums for the prediction and prompt-cache placeholders:

```bash
dive-highmotion-rescore \
  --references outputs/highmotion-reference-v2/high_motion_high_fps_v2.parquet \
  --references-sha256 REFERENCE_SHA256 \
  --predictions /data/predictions/baseline_samples.jsonl \
  --predictions-sha256 BASELINE_SAMPLES_SHA256 \
  --prompt-cache /data/predictions/baseline_samples.jsonl \
  --prompt-cache-sha256 BASELINE_SAMPLES_SHA256 \
  --method llava_onevision_0_5b --prediction-source archived_baseline \
  --expected-count 1000 --output outputs/baseline-v2
```

After a complete new GRT run, invoke the same command with its sample JSONL,
its own checksum, `--prediction-source new_grt`, its method name and a new output
directory. **Keep the reference and common baseline prompt-cache inputs the same.**
The new run must retain the original `doc.answer` metadata for source binding;
the scorer obtains its actual scoring labels from the separately supplied v2
Parquet. Never publish the old-reference metrics emitted during generation.

A 1,000-record preview is not a full 3,243-record result. Undefined reference
positions and metric-specific denominators must be disclosed for every method.
Comparison with archived predictions also cannot retroactively prove their
exact model-weight revisions, runtime packages or consumed input tensors.

All three installed commands have equivalent checkout entry points under
`python -m tools.densevideo`: `bind_highmotion_sources`,
`build_highmotion_reference_v2`, and `rescore_highmotion_v2`.
