# High-Motion v2: source verification and cached rescoring

These tools implement the [fixed reference and scoring contract](HIGHMOTION_REFERENCE_V2.md).
They do not establish a GRT win or lift the current High-Motion leaderboard hold.
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

This is the candidate's fixed generation protocol, not a declaration of a
successful run or superiority. It retains the original input task so cached
baseline predictions remain usable. Obtain authorized dataset/model access and
set the media root to the directory containing `egodex/<action>/<id>.mp4`.
The first-stage result's legacy GT metrics are **not v2 scores**.

```bash
CUDA_VISIBLE_DEVICES=0 \
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
Keep the prepared-Arrow-cache override above: an old cached task must not replace
the intended original annotation. The binder and reference constructor verify
the original Parquet checksum. The compatibility task itself is not a checksum
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
