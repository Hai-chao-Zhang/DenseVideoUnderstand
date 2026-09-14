#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _parse_value(value):
    num = _to_float(value)
    return num if num is not None else value


def _find_latest_result_json(run_output_dir: Path) -> Optional[Path]:
    candidates = list(run_output_dir.rglob("*_results.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pick_metric(task_results: Dict, metric_name: str):
    direct_keys = [metric_name, f"{metric_name},none"]
    for key in direct_keys:
        if key in task_results:
            return task_results[key]
    for key, value in task_results.items():
        if key.startswith(f"{metric_name},"):
            return value
    return None


def _normalize_metric_key(key: str) -> str:
    key = str(key)
    if "," in key:
        key = key.split(",", 1)[0]
    return key.strip()


def _flatten_task_metrics(task_results: Dict) -> Dict:
    metrics = {}
    for key, value in task_results.items():
        name = _normalize_metric_key(key)
        if not name:
            continue
        if isinstance(value, (int, float, str)) or value is None:
            metrics[name] = value
    return metrics


def _compute_main_score(metrics: Dict):
    grid_ade = _to_float(metrics.get("grid_ade"))
    if grid_ade is not None:
        return -grid_ade

    for key in ["gpt_eval_score", "token_f1", "grid_acc"]:
        value = _to_float(metrics.get(key))
        if value is not None:
            return value

    for key, value in metrics.items():
        if "acc" in key.lower():
            value = _to_float(value)
            if value is not None:
                return value
    return None


def _parse_result_metrics(result_json_path: Optional[Path], task_name: str) -> Dict:
    out = {
        "cer": None,
        "wer": None,
        "token_f1": None,
        "exact_match": None,
        "gpt_eval_score": None,
        "gpt_eval_accuracy": None,
        "grid_acc": None,
        "grid_ade": None,
        "grid_fde": None,
        "grid_transition_acc": None,
        "main_score": None,
        "samples": None,
    }
    if result_json_path is None or not result_json_path.exists():
        return out

    with result_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    task_results = payload.get("results", {}).get(task_name, {})
    dynamic_metrics = _flatten_task_metrics(task_results)
    out.update(dynamic_metrics)

    for metric_name in [
        "cer",
        "wer",
        "token_f1",
        "exact_match",
        "gpt_eval_score",
        "gpt_eval_accuracy",
        "grid_acc",
        "grid_ade",
        "grid_fde",
        "grid_transition_acc",
    ]:
        out[metric_name] = _pick_metric(task_results, metric_name)
    out["main_score"] = _compute_main_score(out)

    n_samples = payload.get("n-samples", {}).get(task_name, {})
    out["samples"] = n_samples.get("effective")
    return out


def _mean(rows: List[Dict], key: str):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _sum(rows: List[Dict], key: str):
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals)


def _ratio_of_sums(rows: List[Dict], numerator_key: str, denominator_key: str):
    pairs = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        has_numerator = isinstance(numerator, (int, float))
        has_denominator = isinstance(denominator, (int, float))
        if has_numerator != has_denominator:
            # A partial route would make independent sums look valid while
            # silently comparing different request populations.
            return None
        if has_numerator:
            pairs.append((numerator, denominator))
    if not pairs:
        return None
    denominator_sum = sum(denominator for _, denominator in pairs)
    if denominator_sum <= 0:
        return None
    return sum(numerator for numerator, _ in pairs) / denominator_sum


def _string_counts(rows: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _aggregate_strings(rows: List[Dict], key: str):
    values = sorted(_string_counts(rows, key))
    return "+".join(values) if values else None


def _value_counts(rows: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            label = str(value).lower()
        elif isinstance(value, (int, float)):
            label = str(int(value)) if float(value).is_integer() else str(value)
        elif isinstance(value, str) and value:
            label = value
        else:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _dense_gate_summary(rows: List[Dict]) -> Dict:
    """Aggregate patch-projection telemetry across homogeneous or mixed routes.

    Per-request ratios cannot be averaged when requests contain different
    numbers of patches.  Prefer ratios of the corresponding patch-count sums,
    while retaining a fallback for logs produced before patch counts were
    emitted.  Reference-compute telemetry deliberately has no legacy fallback:
    it is only meaningful when every instrumented route reports its reference
    denominator.
    """

    total_recomputed = _sum(rows, "recomputed_patches")
    total_orig = _sum(rows, "orig_patches")
    total_reference_orig = _sum(rows, "reference_orig_patches")
    recompute_ratio = _ratio_of_sums(rows, "recomputed_patches", "orig_patches")
    has_patch_counts = any(
        isinstance(row.get("recomputed_patches"), (int, float))
        or isinstance(row.get("orig_patches"), (int, float))
        for row in rows
    )
    if recompute_ratio is None and not has_patch_counts:
        recompute_ratio = _mean(rows, "patch_projection_recompute_ratio")
    if recompute_ratio is None and not has_patch_counts:
        recompute_ratio = _mean(rows, "recompute_ratio")

    reference_ratio = _ratio_of_sums(rows, "recomputed_patches", "reference_orig_patches")
    gate_policy_counts = _string_counts(rows, "gate_policy")
    gate_projection_mode_counts = _string_counts(rows, "gate_projection_mode")
    question_route_counts = _string_counts(rows, "question_route")
    video_decode_backend_counts = _string_counts(rows, "video_decode_backend")
    effective_max_new_tokens_counts = _value_counts(
        rows, "effective_max_new_tokens"
    )
    frozen_cache_hit_counts = _value_counts(rows, "frozen_cache_hit")

    return {
        "total_recomputed_patches": total_recomputed,
        "total_orig_patches": total_orig,
        "total_reference_orig_patches": total_reference_orig,
        # Compatibility field: the name is retained, but patch counts now make
        # its aggregation a ratio-of-sums rather than an unweighted mean.
        "mean_recompute_ratio": recompute_ratio,
        "reference_patch_compute_ratio": reference_ratio,
        "mean_patch_projection_recompute_ratio": recompute_ratio,
        "mean_patch_projection_compute_ratio_vs_reference": reference_ratio,
        "gate_policy": _aggregate_strings(rows, "gate_policy"),
        "gate_policy_counts": json.dumps(gate_policy_counts, sort_keys=True, separators=(",", ":")),
        "gate_projection_mode": _aggregate_strings(rows, "gate_projection_mode"),
        "gate_projection_mode_counts": json.dumps(
            gate_projection_mode_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "mean_gate_refresh_interval_frames": _mean(
            rows,
            "gate_refresh_interval_frames",
        ),
        "total_forced_refresh_frames": _sum(rows, "forced_refresh_frames"),
        "mean_forced_refresh_frames": _mean(rows, "forced_refresh_frames"),
        "total_forced_refresh_patches": _sum(rows, "forced_refresh_patches"),
        "mean_forced_refresh_patches": _mean(rows, "forced_refresh_patches"),
        "forced_refresh_frame_indices": _aggregate_strings(
            rows,
            "forced_refresh_frame_indices",
        ),
        "prompt_router": _aggregate_strings(rows, "prompt_router"),
        "question_route": _aggregate_strings(rows, "question_route"),
        "question_route_counts": json.dumps(question_route_counts, sort_keys=True, separators=(",", ":")),
        "video_decode_backend": _aggregate_strings(rows, "video_decode_backend"),
        "video_decode_backend_counts": json.dumps(
            video_decode_backend_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "mean_effective_max_new_tokens": _mean(
            rows, "effective_max_new_tokens"
        ),
        "effective_max_new_tokens_counts": json.dumps(
            effective_max_new_tokens_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "frozen_cache_hit": _aggregate_strings(rows, "frozen_cache_hit"),
        "frozen_cache_hit_counts": json.dumps(
            frozen_cache_hit_counts,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "frozen_cache_manifest_sha256": _aggregate_strings(
            rows, "frozen_cache_manifest_sha256"
        ),
        "mean_requested_frames": _mean(rows, "requested_frames"),
        "mean_reference_frames": _mean(rows, "reference_frames"),
        "mean_reference_orig_patches": _mean(rows, "reference_orig_patches"),
    }


def _first_string(rows: List[Dict], key: str):
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_value(rows: List[Dict], key: str):
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _mean_first_available(row_groups: List[List[Dict]], key: str):
    for rows in row_groups:
        value = _mean(rows, key)
        if value is not None:
            return value
    return None


def _first_available(row_groups: List[List[Dict]], key: str):
    for rows in row_groups:
        value = _first_value(rows, key)
        if value is not None:
            return value
    return None


def _parse_log_metrics(run_log: Path):
    dense_metrics = []
    fps_stats = []
    prune_metrics = []
    run_meta = {
        "run_wall_time_s": None,
        "run_status": None,
        "return_code": None,
    }

    kv_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")

    if not run_log.exists():
        return dense_metrics, fps_stats, prune_metrics, run_meta

    with run_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[DENSE_METRICS]" in line:
                parsed = {}
                for key, value in kv_pattern.findall(line):
                    parsed[key] = _parse_value(value)
                if parsed:
                    dense_metrics.append(parsed)
            elif "[PRUNE_METRICS]" in line:
                parsed = {}
                for key, value in kv_pattern.findall(line):
                    parsed[key] = _parse_value(value)
                if parsed:
                    prune_metrics.append(parsed)
            elif "[FPS_STATS]" in line:
                parsed = {}
                for key, value in kv_pattern.findall(line):
                    parsed[key] = _parse_value(value)
                if parsed:
                    fps_stats.append(parsed)
            elif "[RUN_META]" in line:
                kvs = dict(kv_pattern.findall(line))
                if "run_wall_time_s" in kvs:
                    run_meta["run_wall_time_s"] = _to_float(kvs["run_wall_time_s"])
                if "run_status" in kvs:
                    run_meta["run_status"] = kvs["run_status"]
                if "return_code" in kvs:
                    run_meta["return_code"] = _to_float(kvs["return_code"])

    return dense_metrics, fps_stats, prune_metrics, run_meta


def _append_summary_csv(summary_csv: Path, row: Dict):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "task",
        "method",
        "input_fps",
        "main_score",
        "cer",
        "wer",
        "token_f1",
        "exact_match",
        "gpt_eval_score",
        "gpt_eval_accuracy",
        "grid_acc",
        "grid_ade",
        "grid_fde",
        "grid_transition_acc",
        "samples",
        "mean_effective_fps",
        "mean_wall_time_s",
        "mean_throughput_fps",
        "mean_pre_tokens",
        "mean_post_tokens",
        "pruning_enabled",
        "prune_mode",
        "prune_apply_mode",
        "mean_post_tokens_before_prune",
        "mean_post_tokens_after_prune",
        "mean_prune_keep_ratio_actual",
        "scene_merge_applied",
        "mean_scene_merge_before_tokens",
        "mean_scene_merge_after_tokens",
        "mean_scene_merge_keep_ratio_actual",
        "mean_codec_k_frames",
        "mean_codec_p_frames",
        "mean_codec_b_frames",
        "mean_codec_unknown_frames",
        "frame_type_source",
        "mean_retention_ratio",
        "mean_gate_keep_ratio",
        "mean_recomputed_patches",
        "mean_orig_patches",
        "total_recomputed_patches",
        "total_orig_patches",
        "total_reference_orig_patches",
        "mean_recompute_ratio",
        "reference_patch_compute_ratio",
        "mean_patch_projection_recompute_ratio",
        "mean_patch_projection_compute_ratio_vs_reference",
        "gate_policy",
        "gate_policy_counts",
        "gate_metric",
        "gate_projection_mode",
        "gate_projection_mode_counts",
        "mean_gate_refresh_interval_frames",
        "total_forced_refresh_frames",
        "mean_forced_refresh_frames",
        "total_forced_refresh_patches",
        "mean_forced_refresh_patches",
        "forced_refresh_frame_indices",
        "prompt_router",
        "question_route",
        "question_route_counts",
        "video_decode_backend",
        "video_decode_backend_counts",
        "mean_effective_max_new_tokens",
        "effective_max_new_tokens_counts",
        "frozen_cache_hit",
        "frozen_cache_hit_counts",
        "frozen_cache_manifest_sha256",
        "mean_requested_frames",
        "mean_reference_frames",
        "mean_reference_orig_patches",
        "mean_merge_ratio",
        "mean_tokenization_time_s",
        "mean_total_video_frames",
        "mean_capped_video_frames",
        "run_wall_time_s",
        "run_status",
        "return_code",
        "result_json",
        "run_log",
    ]

    rows = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
    for name in row.keys():
        if name not in fieldnames:
            fieldnames.append(name)

    rows = [
        item
        for item in rows
        if not (
            str(item.get("run_name")) == str(row.get("run_name"))
            and str(item.get("task")) == str(row.get("task"))
            and str(item.get("method")) == str(row.get("method"))
            and str(item.get("input_fps")) == str(row.get("input_fps"))
        )
    ]
    rows.append(row)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow({k: item.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Collect one densevideo run into summary CSV.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-fps", required=True)
    parser.add_argument("--task-name", default="densevideo")
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--run-output-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    run_log = Path(args.run_log)
    run_output_dir = Path(args.run_output_dir)
    summary_csv = Path(args.summary_csv)

    result_json = _find_latest_result_json(run_output_dir)
    result_metrics = _parse_result_metrics(result_json, args.task_name)
    dense_rows, fps_rows, prune_rows, run_meta = _parse_log_metrics(run_log)
    prune_source_rows = [dense_rows, prune_rows]
    dense_gate_summary = _dense_gate_summary(dense_rows)

    row = {
        "run_name": args.run_name,
        "task": args.task_name,
        "method": args.method,
        "input_fps": args.input_fps,
        "main_score": result_metrics.get("main_score"),
        "cer": result_metrics["cer"],
        "wer": result_metrics["wer"],
        "token_f1": result_metrics["token_f1"],
        "exact_match": result_metrics["exact_match"],
        "gpt_eval_score": result_metrics["gpt_eval_score"],
        "gpt_eval_accuracy": result_metrics["gpt_eval_accuracy"],
        "grid_acc": result_metrics.get("grid_acc"),
        "grid_ade": result_metrics.get("grid_ade"),
        "grid_fde": result_metrics.get("grid_fde"),
        "grid_transition_acc": result_metrics.get("grid_transition_acc"),
        "samples": result_metrics["samples"],
        # Some baseline wrappers only know the sampling density in their
        # loader. Prefer the per-request line, but retain the FPS_STATS value
        # instead of silently rendering an empty leaderboard cell.
        "mean_effective_fps": _mean_first_available([dense_rows, fps_rows], "effective_fps"),
        "mean_wall_time_s": _mean(dense_rows, "wall_time_s"),
        "mean_throughput_fps": _mean(dense_rows, "throughput_fps"),
        "mean_pre_tokens": _mean(dense_rows, "pre_tokens"),
        "mean_post_tokens": _mean(dense_rows, "post_tokens"),
        "pruning_enabled": _first_available(prune_source_rows, "pruning_enabled"),
        "prune_mode": _first_available(prune_source_rows, "prune_mode"),
        "prune_apply_mode": _first_available(prune_source_rows, "prune_apply_mode"),
        "mean_post_tokens_before_prune": _mean_first_available(prune_source_rows, "post_tokens_before_prune"),
        "mean_post_tokens_after_prune": _mean_first_available(prune_source_rows, "post_tokens_after_prune"),
        "mean_prune_keep_ratio_actual": _mean_first_available(prune_source_rows, "prune_keep_ratio_actual"),
        "scene_merge_applied": _first_string(dense_rows, "scene_merge_applied"),
        "mean_scene_merge_before_tokens": _mean(dense_rows, "scene_merge_before_tokens"),
        "mean_scene_merge_after_tokens": _mean(dense_rows, "scene_merge_after_tokens"),
        "mean_scene_merge_keep_ratio_actual": _mean(dense_rows, "scene_merge_keep_ratio_actual"),
        "mean_codec_k_frames": _mean(dense_rows, "codec_k_frames"),
        "mean_codec_p_frames": _mean(dense_rows, "codec_p_frames"),
        "mean_codec_b_frames": _mean(dense_rows, "codec_b_frames"),
        "mean_codec_unknown_frames": _mean(dense_rows, "codec_unknown_frames"),
        "frame_type_source": _first_string(dense_rows, "frame_type_source"),
        "mean_retention_ratio": _mean(dense_rows, "retention_ratio"),
        "mean_gate_keep_ratio": _mean(dense_rows, "gate_keep_ratio"),
        "mean_recomputed_patches": _mean(dense_rows, "recomputed_patches"),
        "mean_orig_patches": _mean(dense_rows, "orig_patches"),
        **dense_gate_summary,
        "gate_metric": _first_string(dense_rows, "gate_metric"),
        "mean_merge_ratio": _mean(dense_rows, "merge_ratio"),
        "mean_tokenization_time_s": _mean(dense_rows, "tokenization_time_s"),
        "mean_total_video_frames": _mean(fps_rows, "total_frames"),
        "mean_capped_video_frames": _mean(fps_rows, "capped_frames"),
        "run_wall_time_s": run_meta.get("run_wall_time_s"),
        "run_status": run_meta.get("run_status"),
        "return_code": run_meta.get("return_code"),
        "result_json": str(result_json) if result_json else "",
        "run_log": str(run_log),
    }
    for key, value in result_metrics.items():
        if key not in row:
            row[key] = value

    _append_summary_csv(summary_csv, row)


if __name__ == "__main__":
    main()
