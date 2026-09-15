"""Validate complete High-Motion v2 numeric reports without inference or I/O.

The caller must authenticate the supplied reference/prompt pins, prediction
files, reports and generation receipt before calling this validator. Checking a
digest's syntax is not checking the underlying file. This module independently
reaggregates saved numeric contributions; it does not reconstruct predictions or
reference geometry and never grants publication permission.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy

from tools.densevideo.highmotion_v2_scoring import (
    MASK_POLICY,
    MAX_DISTANCE,
    METRICS,
    SCORER_VERSION,
    TARGET_JOINT,
    VERSION,
    aggregate_scores,
    require,
    sample_indices,
)

BASELINE_METHODS = (
    "qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b",
    "qwen2_vl_2b", "qwen2_5_vl_3b", "qwen2_5_vl_7b", "qwen2_5_vl_32b",
    "qwen2_5_vl_72b", "llava_onevision_0_5b", "llava_onevision_original",
    "qwen2_vl_7b", "llava_onevision_1_5_8b", "llava_onevision_2_8b",
    "videollama3_2b", "videollama3_7b", "longva_7b", "phi4_multimodal",
)
BASELINE_METHOD = "llava_onevision_0_5b"
PREVIEW_SAMPLES = 1000
SOURCE_RECORDS = 3243
POINT_TOLERANCE = 1e-12
AGGREGATE_TOLERANCE = 1e-15
LOWER_IS_BETTER = frozenset(("grid_ade", "grid_fde"))
METRIC_MAXIMA = {
    metric: MAX_DISTANCE if metric in LOWER_IS_BETTER else 1.0
    for metric in METRICS
}
COMPARISON_CAVEAT = (
    "New GRT predictions versus rescored, configuration-checked archived predictions; "
    "archived baseline weight revisions and consumed-tensor identity are unproven. "
    "This is not a freshly rerun byte-identical paired experiment or a "
    "statistical-significance claim."
)
_SHARED_ROW_FIELDS = (
    "doc_id", "identity_sha256", "prompt_sha256", "source_frame_indices",
    "denominators", "sampled_slots", "valid_slots",
)
_COUNT_FIELDS = (
    "records", "records_with_scored_slots", "records_without_scored_slots",
    "sampled_slots", "valid_slots",
)
_COVERAGE_SHAPES = frozenset(
    (
        bits.bit_count(), int(bool(bits & 128)),
        sum(bool(bits & (1 << index)) and bool(bits & (1 << (index + 1)))
            for index in range(7)),
    )
    for bits in range(256)
)


def _sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA-256: " + label)


def _integer(value, label, *, maximum=None):
    require(type(value) is int and value >= 0
            and (maximum is None or value <= maximum), "Invalid integer: " + label)


def _metric_map(value, label):
    require(isinstance(value, dict) and set(value) == set(METRICS),
            "Expected exactly five metrics: " + label)


def _metric(value, metric):
    require(type(value) in (int, float) and math.isfinite(value)
            and 0 <= value <= METRIC_MAXIMA[metric], "Invalid metric: " + metric)


def _validate_row(row, index):
    require(isinstance(row, dict), "Numeric record must be an object")
    require(type(row.get("doc_id")) is int and row["doc_id"] == index,
            "Missing, duplicate or reordered numeric document IDs")
    for key in ("identity_sha256", "prompt_sha256"):
        _sha(row.get(key), key)
    _integer(row.get("sampled_slots"), "sampled_slots", maximum=8)
    require(row["sampled_slots"] == 8, "Preview must preserve all eight sampled slots")
    _integer(row.get("valid_slots"), "valid_slots", maximum=8)
    indices = row.get("source_frame_indices")
    require(isinstance(indices, list) and len(indices) == 8
            and all(type(value) is int and value >= 0 for value in indices),
            "Invalid source frame indices")
    require(indices == sample_indices(indices[-1] + 1),
            "Source frame indices are not the original endpoint-inclusive eight slots")
    metrics, denominators = row.get("metrics"), row.get("denominators")
    _metric_map(metrics, "row values")
    _metric_map(denominators, "row denominators")
    for metric in METRICS:
        denominator = denominators[metric]
        _integer(denominator, metric + " denominator", maximum=8)
        require((metrics[metric] is None) == (denominator == 0),
                "Null metric must correspond exactly to a zero denominator")
        if metrics[metric] is not None:
            _metric(metrics[metric], metric)
    require(all(denominators[key] == row["valid_slots"]
                for key in ("grid_acc", "grid_ade", "token_f1")),
            "Valid-slot metric denominators disagree")
    require((row["valid_slots"], denominators["grid_fde"],
             denominators["grid_transition_acc"]) in _COVERAGE_SHAPES,
            "Impossible final-slot or adjacent-transition coverage")
    for key in ("prediction_slots", "malformed_or_missing_scored_slots",
                "surplus_prediction_slots"):
        _integer(row.get(key), key)
    require(row["malformed_or_missing_scored_slots"] <= row["valid_slots"],
            "Malformed prediction count exceeds scored slots")
    require(row["surplus_prediction_slots"] == max(0, row["prediction_slots"] - 8),
            "Surplus prediction count disagrees with retained output positions")


def _compare_aggregate(report, recomputed):
    for key in ("benchmark_version", "scorer_version", "target_joint",
                "reference_policy", "aggregation"):
        require(report.get(key) == recomputed[key], "Aggregate metadata changed: " + key)
    for key in _COUNT_FIELDS:
        _integer(report.get(key), key)
        require(report[key] == recomputed[key], "Aggregate count changed: " + key)
    for key in ("metric_scored_records", "metric_scored_slots_or_edges"):
        _metric_map(report.get(key), key)
        for metric in METRICS:
            _integer(report[key][metric], key + ":" + metric)
            require(report[key][metric] == recomputed[key][metric],
                    "Aggregate metric denominator changed: " + key + ":" + metric)
    _metric_map(report.get("metrics"), "aggregate values")
    for metric in METRICS:
        actual, expected = report["metrics"][metric], recomputed["metrics"][metric]
        require((actual is None) == (expected is None), "Aggregate null status changed")
        if actual is not None:
            _metric(actual, metric)
            require(math.isclose(actual, expected, rel_tol=0, abs_tol=AGGREGATE_TOLERANCE),
                    "Aggregate metric disagrees with numeric records: " + metric)


def _validate_report(report, reference_sha256, input_sequence_sha256, grt_method):
    require(isinstance(report, dict), "Rescoring report must be an object")
    require(report.get("method") in (*BASELINE_METHODS, grt_method), "Unexpected method")
    require(report.get("status") == "rescored_cached_predictions", "Wrong rescoring status")
    require(report.get("scope") == "first-1000-source-rows", "Wrong preview scope")
    require(type(report.get("source_reference_records")) is int
            and report["source_reference_records"] == SOURCE_RECORDS, "Wrong source population")
    for key in ("full_source_coverage", "automatic_publication", "inference_performed",
                "grt_superiority_verified"):
        require(report.get(key) is False, "Unexpected claim in report: " + key)
    require(report.get("references_sha256") == reference_sha256,
            "Corrected reference pin differs across methods")
    require(report.get("input_sequence_sha256") == input_sequence_sha256,
            "Original effective input sequence differs across methods")
    for key in ("predictions_sha256", "prompt_cache_sha256"):
        _sha(report.get(key), key)
    expected_source = "new_grt" if report["method"] == grt_method else "archived_baseline"
    require(report.get("prediction_source") == expected_source, "Wrong prediction provenance role")
    rows = report.get("records_detail")
    require(isinstance(rows, list) and len(rows) == PREVIEW_SAMPLES,
            "Every method needs all 1,000 numeric records")
    for index, row in enumerate(rows):
        _validate_row(row, index)
    require(len({row["identity_sha256"] for row in rows}) == PREVIEW_SAMPLES,
            "Duplicate source identities in numeric records")
    recomputed = aggregate_scores(rows)
    _compare_aggregate(report, recomputed)
    return recomputed


def _validate_receipt(receipt, candidate):
    require(isinstance(receipt, dict), "Missing GRT generation receipt")
    require(receipt.get("generation_completed") is True, "GRT generation is incomplete")
    require(type(receipt.get("preview_samples")) is int
            and receipt["preview_samples"] == PREVIEW_SAMPLES, "GRT receipt is not full preview")
    require(type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0,
            "GRT generation worker did not complete successfully")
    require(receipt.get("legacy_scores_are_publishable") is False,
            "Legacy scores are not corrected-reference results")
    for key in ("samples_sha256", "results_sha256", "worker_log_sha256"):
        _sha(receipt.get(key), "GRT receipt " + key)
    require(receipt["samples_sha256"] == candidate["predictions_sha256"],
            "GRT receipt belongs to different predictions")
    for key in ("full3243_benchmark_completed", "baseline_gpu_rerun", "winner_claim",
                "automatic_publication", "baseline_exact_weight_or_tensor_identity_proven"):
        if key in receipt:
            require(receipt[key] is False, "Unsupported generation-receipt claim: " + key)
    require(not receipt.get("validation_error"), "GRT generation receipt contains a validation error")


def validate_reports(reports, *, reference_sha256, input_sequence_sha256,
                     grt_method, grt_receipt):
    """Return ranked numeric summaries and honest deltas, including negative outcomes.

    Exactly the 18 fixed archived baselines and one explicit GRT report are
    required. All input objects are left unchanged. Raw predictions, references,
    source paths and generation logs are neither accepted as numeric evidence nor
    copied into the returned public-facing summary.
    """
    _sha(reference_sha256, "caller reference pin")
    _sha(input_sequence_sha256, "caller input-sequence pin")
    require(isinstance(grt_method, str)
            and re.fullmatch(r"[a-z][a-z0-9_.-]*", grt_method) is not None
            and grt_method not in BASELINE_METHODS, "Invalid or colliding GRT method ID")
    require(isinstance(reports, (list, tuple)) and len(reports) == len(BASELINE_METHODS) + 1,
            "Expected exactly 18 baselines and one GRT report")
    by_method, aggregates = {}, {}
    for report in reports:
        recomputed = _validate_report(report, reference_sha256, input_sequence_sha256, grt_method)
        method = report["method"]
        require(method not in by_method, "Duplicate method report")
        by_method[method], aggregates[method] = report, recomputed
    require(set(by_method) == set(BASELINE_METHODS) | {grt_method}, "Missing baseline or GRT report")
    baseline = by_method[BASELINE_METHOD]
    for report in reports:
        require(report["prompt_cache_sha256"] == baseline["predictions_sha256"],
                "Prompt cache must be the fixed corresponding 0.5B baseline prediction file")
        for expected, actual in zip(baseline["records_detail"], report["records_detail"]):
            require(all(actual[key] == expected[key] for key in _SHARED_ROW_FIELDS),
                    "Per-row identity, prompt, sampling or coverage differs across methods")
    _validate_receipt(grt_receipt, by_method[grt_method])

    ranked = []
    for method, aggregate in aggregates.items():
        ranked.append({
            "method": method, "samples": PREVIEW_SAMPLES,
            "prediction_source": by_method[method]["prediction_source"],
            "predictions_sha256": by_method[method]["predictions_sha256"],
            **aggregate["metrics"],
            **{key: aggregate[key] for key in _COUNT_FIELDS},
            "metric_scored_records": deepcopy(aggregate["metric_scored_records"]),
            "metric_scored_slots_or_edges": deepcopy(aggregate["metric_scored_slots_or_edges"]),
        })
    ranked.sort(key=lambda row: (row["grid_acc"] is None,
                                -row["grid_acc"] if row["grid_acc"] is not None else 0,
                                row["method"]))
    previous_score, previous_rank = None, None
    for position, row in enumerate(ranked, 1):
        score = row["grid_acc"]
        row["rank"] = (None if score is None else
                       previous_rank if score == previous_score else position)
        previous_score, previous_rank = score, row["rank"]

    deltas, oriented, better = {}, {}, {}
    for metric in METRICS:
        candidate = aggregates[grt_method]["metrics"][metric]
        control = aggregates[BASELINE_METHOD]["metrics"][metric]
        delta = None if candidate is None or control is None else candidate - control
        deltas[metric] = delta
        oriented[metric] = None if delta is None else (-delta if metric in LOWER_IS_BETTER else delta)
        better[metric] = None if delta is None else oriented[metric] > POINT_TOLERANCE
    return {
        "status": "numeric_reports_validated", "benchmark_version": VERSION,
        "target_joint": TARGET_JOINT, "reference_policy": MASK_POLICY,
        "scorer_version": SCORER_VERSION, "references_sha256": reference_sha256,
        "input_sequence_sha256": input_sequence_sha256,
        "scope": "first-1000-source-rows", "source_reference_records": SOURCE_RECORDS,
        "full_source_coverage": False, "method_count": len(ranked), "rows": ranked,
        "ranking": "Grid Accuracy descending; exact ties share competition rank, ordered by method; null unranked",
        "comparison": {
            "baseline_method": BASELINE_METHOD, "grt_method": grt_method,
            "primary_metric": "grid_acc", "point_tolerance": POINT_TOLERANCE,
            "baseline_metrics": deepcopy(aggregates[BASELINE_METHOD]["metrics"]),
            "grt_metrics": deepcopy(aggregates[grt_method]["metrics"]),
            "grt_minus_baseline": deltas, "oriented_improvements": oriented,
            "metric_outperform": better, "grid_acc_outperform": better["grid_acc"] is True,
        },
        "comparison_caveat": COMPARISON_CAVEAT,
        "underlying_file_hashes_verified_by_this_function": False,
        "inference_performed": False, "baseline_gpu_rerun": False,
        "baseline_exact_weight_or_tensor_identity_proven": False,
        "statistical_significance_claim": False, "automatic_publication": False,
    }
