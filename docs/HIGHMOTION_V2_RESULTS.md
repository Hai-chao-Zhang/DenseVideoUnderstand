# High-Motion v2: completed HF 0.5B GRT preview

Audit date: 2026-09-15. Version: `highmotion-right-ring-v2`.
This evaluates the fixed first **1,000 records**, not the full 3,243-record
benchmark. All 18 baselines reuse complete archived predictions; only GRT ran
new GPU inference. All 19 methods use the same corrected GT, original questions,
effective prompts, eight sampled frame positions and validity masks.

| Metric | Rescored HF 0.5B baseline | New HF 0.5B GRT | GRT minus baseline |
| --- | ---: | ---: | ---: |
| Grid Accuracy ↑ | 0.04686825949892152 | 0.049503622587246277 | +0.002635363088324759 |
| Token F1 ↑ | 0.04189832856152993 | 0.045052470607675664 | +0.0031541420461457317 |
| Grid ADE ↓ | 1.0964107152122904 | 1.0862032974124285 | −0.010207417799861895 |
| Grid FDE ↓ | 1.077715207452239 | 1.067214162928631 | −0.010501044523608005 |
| Transition Accuracy ↑ | 0.005684104627766599 | 0.00534037558685446 | −0.0003437290409121392 |

GRT exceeds its corresponding baseline on primary Grid Accuracy by **0.263536
percentage points**. Four metrics improve; Transition Accuracy regresses. These
are small preview point estimates, not statistical significance or a win on every
metric. The absolute scores remain low.

## Reference correction and coverage

The [frozen pre-outcome specification](HIGHMOTION_REFERENCE_V2.md) defines
`rightRingFingerMetacarpal` as the named right-palm/ring-finger-base proxy. Its
target, validity mask and scorer were fixed before inspecting this run's outputs.
No questions, frames or scoring rules were changed to obtain this improvement.
Invalid source references are null with explicit masks/reasons, not center-cell
labels or shifted positions. Source validity is not proof of RGB visibility or
perfect reprojection.

All 3,243 source rows remain; 631,061 of 826,253 full-frame references are valid.
All 1,000 preview predictions remain counted, including 139 rows with no defined
reference score. Metrics are macro means over their defined per-row values:

| Metric population | Scored rows | Valid original positions / edges |
| --- | ---: | ---: |
| Grid Accuracy, ADE, Token F1 | 861 | 6,015 positions out of 8,000 |
| FDE | 669 | 669 original final positions |
| Transition Accuracy | 852 | 5,101 adjacent valid edges |

## Execution and independent checks

The completed run used one NVIDIA GH200, Python 3.10.19, PyTorch 2.9.0+cu126,
Transformers 4.57.6, PyAV 15.1 and NumPy 2.2.6. Checkpoint:
`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, revision
`74dd0bf867a4cda7950c17663794267c60cf4b40`; weight SHA-256:
`07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e`.
Parameters were GPU-resident BF16, with strict deterministic execution and TF32
disabled. No baseline GPU job was launched.

GRT used motion/SSIM gating, threshold 0.001, `linear_consistent` projection,
refresh interval 0, router off, eight endpoint-inclusive PyAV-seek frames,
384-pixel processing, eager attention and a 64-token cap. The
[reproduction guide](HIGHMOTION_V2_REPRODUCTION.md) gives the full command and
isolated input-cache setup. This is the HF wrapper, not the native LLaVA checkpoint
or Educational Route31 profile. Generation-stage legacy GT metrics are **not v2**.

The run passed checks on 1,000 video hashes, effective prompts, saved source-ordered
IDs/original metadata, eight decoded RGB frames per request, and consumed token
and video tensor hashes. Processing uses the released tokenizer's length-sorted
order, restoring source order in saved predictions. Released evaluator/model/
preprocessing code matches the deployed code. Independent CPU rescoring of both
raw GRT and HF 0.5B baseline predictions reproduced every saved numeric row and
aggregate; independent Decimal reaggregation checked all 19 methods.

Archived baselines lack pinned historical weight revisions, runtime packages
and consumed-tensor receipts. Configuration and prompt agreement do not recover
that evidence: this is **not a freshly rerun byte-identical pair**. No hardware
speedup or end-to-end FLOP claim follows from this comparison.

## Numeric evidence and access

The default complete-leaderboard command authenticates the five-file
[`2026-09-15` bundle](../release/2026-09-15/) using manifest SHA-256
`1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc`.
It includes 19 × 1,000 numeric contributions, sanitized generation receipt,
construction report and independent reference audit. No raw answer/prediction
text, video, credential or local path is included. Public numeric validation
does not rerun inference, reference geometry, or raw-prediction scoring.

Reference Parquet SHA-256:
`140ae69160d5777bbf24381ea02980481c40a638cdc2e3f69b20894c95fd6f8b`.
GRT prediction SHA-256:
`dd3d6f45c0aa3d35e2116a367aaac1cae81fec807187180f249ff9fbaf04aaa5`.
Archived HF 0.5B prediction SHA-256:
`f9616d1effa3bf94112be04d5ce9eb090b702a346662ffd43aaaa7d68aa08bd6`.

Authorized source access remains necessary for new inference/reference rebuilding.
The two-task Hub entry and High-Motion source remain private; this code/numeric
release changes neither visibility nor redistribution rights. Legacy GT files
and old aggregate scores must not be used as v2 answers or scores.
