# Educational GPU smoke validation — 2026-09-14

**Scope: execution and input-alignment checks, not full benchmark reproduction
or evidence that GRT improves quality.** No new leaderboard score is published
by this check. Full 634-example matched evaluation remains pending.

The frozen source was public commit
`a52c0360de0df175a476e815f4aaa27886131b23`, tree
`0a3b843c5dad5a705ccf9e05844065826b24f880`. Subsequent MOS failure-status
reporting changes do not change model implementations, profiles, sampling,
generation, judge settings, or quality gates; they were not part of this run.

One NVIDIA GH200 allocation completed successfully in 13 minutes 27 seconds.
Route31, Qwen 3B and Qwen 7B each executed base, all-patch control and GRT
serially, using the same physical GPU throughout. Each arm covered the first
two pinned Educational examples, one subtitle and one OCR question. The
32B judge also ran on this allocation without CPU/disk offload.

The runtime used Python 3.10.19, PyTorch 2.9.0+cu126, torchvision 0.24.0,
Transformers 4.57.6, Accelerate 1.12.0, PyAV 15.1.0, Datasets 4.5.0,
NumPy 2.2.6 and qwen-vl-utils 0.0.14. Exact model/judge revisions and
generation settings were those in [profiles.json](../tools/densevideo/profiles.json).

Checks passed for all nine arms:

- Exactly two typed, unique sample identities and complete predictions.
- Resolved model arguments, generation caps and evaluation seeds.
- Identical eight-frame pre-vision tensor bytes, dtype, shape, token IDs,
  attention masks and frame grids across each profile's three arms.
- The same deterministic runtime and physical GPU; TF32 disabled.
- Complete fresh MOS coverage bound to the pinned judge and fresh predictions.
- Recomputed patch telemetry, matched controls and applicable route-floor checks.

A separate capacity-only probe scored eight long reference-as-prediction
requests with the pinned BF16 judge at batch 8 and generation cap 64. Peak
allocated GPU memory was 83,987,359,232 bytes. This is a capacity observation,
not an accuracy result or a guarantee for every future input.

All saved reports remained `smoke_only`. A separate read-only audit recomputed
the nine arms' configurations, identities, telemetry, MOS bindings and input
fingerprints from their raw outputs. Receipt SHA-256 values are:

| Private receipt | SHA-256 |
|---|---|
| `debug_completed.json` | `e65cc41495178642c39bf53db792281597e16fe4cf31646e39f52bea5edeaa1f` |
| `judge_capacity.json` | `5bfead9b60b04f478a81527632e0148498018dedc558e72e69f5efd3e2b3dcb3` |
| `independent_smoke_audit.json` | `31bea341bc1844b3a47a5fa1244f63d9a27dba1983cf065b5ff8a5eaeea157ba` |

These hashes identify retained private artifacts; hashes alone are not
independently accessible proof. Raw logs, dataset text, videos and generated
answers are not redistributed. Authorized data access is still required to
rerun the [documented commands](REPRODUCTION.md).

Input hashing includes CPU-copy overhead, so these timings are not a replication
of historical uninstrumented latency. Two examples cannot establish full-split
coverage, historical score equality, statistical superiority or High-Motion
correctness. No High-Motion inference was part of this job.
