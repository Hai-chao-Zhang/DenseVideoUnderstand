import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.densevideo.collect_run_metrics import (  # noqa: E402
    _dense_gate_summary,
    _mean_first_available,
    _parse_log_metrics,
)


def test_parse_baseline_profile_contract(tmp_path: Path):
    log_path = tmp_path / "base.log"
    log_path.write_text(
        "\n".join(
            [
                "[FPS_STATS] strategy=uniform total_frames=3000 capped_frames=3000 "
                "sampled_frames=8 duration_s=100 effective_fps=0.08",
                "[DENSE_METRICS] sampled_frames=8 effective_fps=0.08 wall_time_s=2 "
                "throughput_fps=4 recompute_ratio=1.0 gate_policy=disabled",
            ]
        ),
        encoding="utf-8",
    )

    dense_rows, fps_rows, prune_rows, run_meta = _parse_log_metrics(log_path)

    assert prune_rows == []
    assert run_meta["run_status"] is None
    assert dense_rows[0]["recompute_ratio"] == 1.0
    assert dense_rows[0]["gate_policy"] == "disabled"
    assert _mean_first_available([dense_rows, fps_rows], "effective_fps") == 0.08


def test_fps_stats_is_a_fallback_and_zero_is_not_missing(tmp_path: Path):
    log_path = tmp_path / "fps-only.log"
    log_path.write_text(
        "[FPS_STATS] sampled_frames=0 duration_s=10 effective_fps=0.0\n",
        encoding="utf-8",
    )

    dense_rows, fps_rows, _, _ = _parse_log_metrics(log_path)

    assert _mean_first_available([dense_rows, fps_rows], "effective_fps") == 0.0


def test_parse_mixed_route_profile_contract(tmp_path: Path):
    log_path = tmp_path / "mixed.log"
    log_path.write_text(
        "[DENSE_METRICS] prompt_router=ocr_motion_subtitle_exact "
        "question_route=ocr_motion requested_frames=8 reference_frames=16 "
        "video_decode_backend=pyav_seek effective_max_new_tokens=128 "
        "frozen_cache_hit=false frozen_cache_manifest_sha256=none "
        "recomputed_patches=25 orig_patches=100 reference_orig_patches=200 "
        "recompute_ratio=0.25 patch_projection_recompute_ratio=0.25 "
        "patch_projection_compute_ratio_vs_reference=0.125 gate_policy=motion\n",
        encoding="utf-8",
    )

    dense_rows, _, _, _ = _parse_log_metrics(log_path)

    assert dense_rows == [
        {
            "prompt_router": "ocr_motion_subtitle_exact",
            "question_route": "ocr_motion",
            "requested_frames": 8.0,
            "reference_frames": 16.0,
            "video_decode_backend": "pyav_seek",
            "effective_max_new_tokens": 128.0,
            "frozen_cache_hit": "false",
            "frozen_cache_manifest_sha256": "none",
            "recomputed_patches": 25.0,
            "orig_patches": 100.0,
            "reference_orig_patches": 200.0,
            "recompute_ratio": 0.25,
            "patch_projection_recompute_ratio": 0.25,
            "patch_projection_compute_ratio_vs_reference": 0.125,
            "gate_policy": "motion",
        }
    ]


def test_mixed_route_patch_metrics_use_ratio_of_sums_and_count_routes():
    rows = [
        {
            "recomputed_patches": 100,
            "orig_patches": 100,
            "reference_orig_patches": 200,
            "recompute_ratio": 1.0,
            "patch_projection_recompute_ratio": 1.0,
            "patch_projection_compute_ratio_vs_reference": 0.5,
            "gate_policy": "all",
            "prompt_router": "ocr_motion_subtitle_exact",
            "question_route": "subtitle_exact",
            "requested_frames": 8,
            "reference_frames": 16,
            "video_decode_backend": "decord_frozen",
            "effective_max_new_tokens": 31,
            "frozen_cache_hit": "true",
            "frozen_cache_manifest_sha256": "a" * 64,
        },
        {
            "recomputed_patches": 10,
            "orig_patches": 900,
            "reference_orig_patches": 1800,
            "recompute_ratio": 10 / 900,
            "patch_projection_recompute_ratio": 10 / 900,
            "patch_projection_compute_ratio_vs_reference": 10 / 1800,
            "gate_policy": "motion",
            "prompt_router": "ocr_motion_subtitle_exact",
            "question_route": "ocr_motion",
            "requested_frames": 8,
            "reference_frames": 16,
            "video_decode_backend": "pyav_seek",
            "effective_max_new_tokens": 128,
            "frozen_cache_hit": "false",
            "frozen_cache_manifest_sha256": "none",
        },
    ]

    summary = _dense_gate_summary(rows)

    assert summary["total_recomputed_patches"] == 110
    assert summary["total_orig_patches"] == 1000
    assert summary["total_reference_orig_patches"] == 2000
    assert summary["mean_recompute_ratio"] == 110 / 1000
    assert summary["reference_patch_compute_ratio"] == 110 / 2000
    assert summary["mean_patch_projection_recompute_ratio"] == 110 / 1000
    assert summary["mean_patch_projection_compute_ratio_vs_reference"] == 110 / 2000
    assert summary["gate_policy"] == "all+motion"
    assert summary["gate_policy_counts"] == '{"all":1,"motion":1}'
    assert summary["prompt_router"] == "ocr_motion_subtitle_exact"
    assert summary["question_route"] == "ocr_motion+subtitle_exact"
    assert summary["question_route_counts"] == '{"ocr_motion":1,"subtitle_exact":1}'
    assert summary["video_decode_backend"] == "decord_frozen+pyav_seek"
    assert summary["video_decode_backend_counts"] == '{"decord_frozen":1,"pyav_seek":1}'
    assert summary["mean_effective_max_new_tokens"] == 79.5
    assert summary["effective_max_new_tokens_counts"] == '{"128":1,"31":1}'
    assert summary["frozen_cache_hit"] == "false+true"
    assert summary["frozen_cache_hit_counts"] == '{"false":1,"true":1}'
    assert summary["frozen_cache_manifest_sha256"] == ("a" * 64) + "+none"
    assert summary["mean_requested_frames"] == 8
    assert summary["mean_reference_frames"] == 16
    assert summary["mean_reference_orig_patches"] == 1000


def test_projection_and_forced_refresh_telemetry_is_aggregated():
    rows = [
        {
            "gate_projection_mode": "linear_consistent",
            "gate_refresh_interval_frames": 4,
            "forced_refresh_frames": 0,
            "forced_refresh_patches": 0,
            "forced_refresh_frame_indices": "none",
        },
        {
            "gate_projection_mode": "linear_consistent",
            "gate_refresh_interval_frames": 4,
            "forced_refresh_frames": 1,
            "forced_refresh_patches": 729,
            "forced_refresh_frame_indices": "4",
        },
    ]

    summary = _dense_gate_summary(rows)

    assert summary["gate_projection_mode"] == "linear_consistent"
    assert summary["gate_projection_mode_counts"] == '{"linear_consistent":2}'
    assert summary["mean_gate_refresh_interval_frames"] == 4
    assert summary["total_forced_refresh_frames"] == 1
    assert summary["mean_forced_refresh_frames"] == 0.5
    assert summary["total_forced_refresh_patches"] == 729
    assert summary["mean_forced_refresh_patches"] == 364.5
    assert summary["forced_refresh_frame_indices"] == "4+none"


def test_legacy_recompute_ratio_falls_back_to_unweighted_mean():
    summary = _dense_gate_summary(
        [
            {"recompute_ratio": 0.25, "gate_policy": "motion"},
            {"recompute_ratio": 0.75, "gate_policy": "motion"},
        ]
    )

    assert summary["mean_recompute_ratio"] == 0.5
    assert summary["reference_patch_compute_ratio"] is None
    assert summary["gate_policy"] == "motion"
    assert summary["gate_policy_counts"] == '{"motion":2}'


def test_reference_ratio_is_missing_when_any_instrumented_route_lacks_denominator():
    summary = _dense_gate_summary(
        [
            {
                "recomputed_patches": 100,
                "orig_patches": 100,
                "reference_orig_patches": 200,
            },
            {
                "recomputed_patches": 25,
                "orig_patches": 100,
            },
        ]
    )

    assert summary["mean_recompute_ratio"] == 125 / 200
    assert summary["reference_patch_compute_ratio"] is None


def test_partial_patch_counts_do_not_fall_back_to_unweighted_ratio():
    summary = _dense_gate_summary(
        [
            {
                "recomputed_patches": 25,
                "orig_patches": 100,
                "recompute_ratio": 0.25,
            },
            {
                "recomputed_patches": 50,
                "recompute_ratio": 0.5,
            },
        ]
    )

    assert summary["mean_recompute_ratio"] is None
