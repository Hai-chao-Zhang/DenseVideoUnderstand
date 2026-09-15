"""Read an explicitly SHA-pinned, numeric-only High-Motion v2 release bundle.

This checks saved evidence and independently reaggregates numeric contributions.
It does not rerun inference, source projection, or per-prediction scoring. No
default release is enabled here: an actual complete bundle must be supplied.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from tools.densevideo.highmotion_v2_bundle import (
    BASELINE_METHOD,
    BASELINE_METHODS,
    validate_reports,
)
from tools.densevideo.highmotion_v2_scoring import (
    MASK_POLICY,
    SCORER_VERSION,
    TARGET_JOINT,
    VERSION,
    require,
)

SOURCE_ANNOTATION_SHA256 = "518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd"
GRT_METHOD = "grt_llava_hf_0_5b_motion_ssim_t0001"
MODEL_LABELS = {
    "qwen3_vl_2b": "Qwen3-VL-2B-Instruct", "qwen3_vl_4b": "Qwen3-VL-4B-Instruct",
    "qwen3_vl_8b": "Qwen3-VL-8B-Instruct", "qwen3_vl_32b": "Qwen3-VL-32B-Instruct",
    "qwen2_vl_2b": "Qwen2-VL-2B-Instruct", "qwen2_vl_7b": "Qwen2-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen2.5-VL-3B-Instruct", "qwen2_5_vl_7b": "Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_32b": "Qwen2.5-VL-32B-Instruct", "qwen2_5_vl_72b": "Qwen2.5-VL-72B-Instruct",
    "llava_onevision_0_5b": "LLaVA-OneVision HF 0.5B", "llava_onevision_original": "LLaVA-OneVision HF 7B",
    "llava_onevision_1_5_8b": "LLaVA-OneVision-1.5-8B-Instruct",
    "llava_onevision_2_8b": "LLaVA-OneVision-2-8B-Instruct",
    "videollama3_2b": "VideoLLaMA3-2B", "videollama3_7b": "VideoLLaMA3-7B",
    "longva_7b": "LongVA-7B", "phi4_multimodal": "Phi-4-multimodal-instruct",
}
SOURCE_FILES = {
    "highmotion_reference_v2.py": "1ff34718c54f4fb6cf1a574af26b94d8188c8d278b58ab16f1e62cc959d0cfd7",
    "build_highmotion_reference_v2.py": "3130848ce66665f77225f147540ca6bdc81e7ff4109eedd8b29f95b6b99902ac",
    "highmotion_v2_scoring.py": "4a8786ba44800047d3eedafd18c6a5169ddb192e610c57f0c0683c506b837e9d",
    "highmotion_v2_bundle.py": "ae60f97a6040eab8c948436a517dc787250920113269e0f09907fc6712aaf1c8",
}
BUNDLE_FILES = frozenset(("numeric_reports.json.gz", "generation_receipt.json",
                          "reference_build_report.json", "reference_audit.json"))
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_NUMERIC_BYTES = 64 * 1024 * 1024
PUBLIC_REPORT_FIELDS = frozenset((
    "aggregation", "automatic_publication", "benchmark_version", "comparison_caveat",
    "full_source_coverage", "grt_superiority_verified", "inference_performed", "input_sequence_sha256",
    "method", "metric_scored_records", "metric_scored_slots_or_edges", "metrics", "prediction_source",
    "predictions_sha256", "prompt_cache_sha256", "records", "records_detail", "records_with_scored_slots",
    "records_without_scored_slots", "reference_policy", "references_sha256", "sampled_slots", "scope",
    "scorer_version", "source_reference_records", "status", "target_joint", "valid_slots",
))
PUBLIC_ROW_FIELDS = frozenset((
    "denominators", "doc_id", "identity_sha256", "malformed_or_missing_scored_slots", "metrics",
    "prediction_slots", "prompt_sha256", "sampled_slots", "source_frame_indices",
    "surplus_prediction_slots", "valid_slots",
))
PUBLIC_GENERATION_FIELDS = frozenset((
    "generation_completed", "preview_samples", "exit_code", "samples_sha256", "results_sha256",
    "worker_log_sha256", "legacy_scores_are_publishable", "full3243_benchmark_completed",
    "baseline_gpu_rerun", "winner_claim", "automatic_publication",
    "baseline_exact_weight_or_tensor_identity_proven",
))
PUBLIC_MANIFEST_FIELDS = frozenset((
    "schema_version", "benchmark_version", "target_joint", "reference_policy", "scorer_version",
    "scope", "source_reference_records", "method_count", "source_annotation_sha256", "audit_date",
    "references_sha256", "input_sequence_sha256", "source_code_sha256", "grt_method", "files",
))
PUBLIC_BUILD_FIELDS = frozenset((
    "schema_version", "status", "benchmark_version", "target_joint", "reference_policy", "created_utc",
    "rows", "annotation_source_sha256", "original_annotation_unchanged", "binding_manifest_sha256",
    "canonical_source_revision", "canonical_zip_sha256_from_prior_verified_manifest",
    "zip_members_reread_in_this_build", "local_hdf5_actual_bytes_verified", "ordered_video_paths_sha256",
    "ordered_hdf5_sha256", "ordered_question_sha256", "parquet_name", "parquet_sha256",
    "preserved_columns", "full_frame_coverage", "uniform_eight_frame_coverage", "sampling",
    "invalid_reason_counts_overlap", "rows_removed", "model_inference_performed", "model_results_used",
    "performance_or_superiority_claim", "automatic_publication",
    "source_reprojection_is_not_verified_rgb_visibility",
))
PUBLIC_AUDIT_FIELDS = frozenset((
    "status", "original_annotation_sha256", "reference_sha256", "input_sequence_sha256",
    "source_hdf_bindings_verified_from_pinned_manifest", "max_macro_abs_difference",
    "automatic_publication", "grt_outputs_read", "gpu_inference_performed", "full_frame_coverage",
    "sampled_full_coverage", "full_nonanswer_columns_equal", "preview_rows", "preview_scored_rows",
    "preview_valid_slots", "preview_fde_rows", "preview_transition_rows", "preview_transition_edges",
    "baselines_verified", "elapsed_seconds", "per_prediction_metric_rescore_performed",
    "prepare_summary_sha256", "source_hdf_projection_rerun",
))
COVERAGE_FIELDS = frozenset(("rows", "frames", "valid_frames", "invalid_frames", "all_invalid_rows",
                             "fully_valid_rows", "invalid_reason_counts"))
INVALID_REASONS = frozenset((
    "missing_confidence", "invalid_confidence", "missing_camera", "missing_joint", "nonfinite_camera",
    "nonfinite_joint", "missing_intrinsic", "nonfinite_intrinsic", "noninvertible_camera",
    "nonfinite_camera_point", "nonpositive_depth", "invalid_projection", "out_of_frame",
))
SOURCE_COLUMNS = frozenset((
    "video", "speaker_index", "frame_count", "fps", "width", "height", "segment_start", "segment_end",
    "video_path", "question", "qid", "type", "captions", "ocr", "ocr_answer",
))
SAVED_COMPARISON_CAVEAT = (
    "Archived runs have no consumed-tensor hashes or pinned weight revisions; "
    "configuration/prompt checks are not a fresh byte-identical paired reproduction"
)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Invalid SHA-256: " + label)


def _read_checked(root, name, expected):
    _sha(expected, name)
    path = root / name
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked bundle member: " + name)
    require(path.stat().st_size <= MAX_FILE_BYTES, "Oversized bundle member: " + name)
    with path.open("rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    require(len(raw) <= MAX_FILE_BYTES and _digest(raw) == expected,
            "Bundle member SHA-256 or size mismatch: " + name)
    return raw


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _coverage(value, *, sampled=False):
    require(isinstance(value, dict) and set(value) == COVERAGE_FIELDS,
            "Unexpected reference coverage fields")
    require(value.get("rows") == 3243,
            "Reference coverage must retain all 3,243 rows")
    for key in ("rows", "frames", "valid_frames", "invalid_frames", "all_invalid_rows", "fully_valid_rows"):
        require(type(value.get(key)) is int and value[key] >= 0, "Invalid reference coverage count")
    require(value["frames"] > 0 and value["valid_frames"] + value["invalid_frames"] == value["frames"],
            "Reference frame coverage does not add up")
    require(value["all_invalid_rows"] + value["fully_valid_rows"] <= 3243,
            "Reference row coverage does not add up")
    if sampled:
        require(value["frames"] == 3243 * 8, "Reference sampling must retain all original eight slots")
    reasons = value.get("invalid_reason_counts")
    require(isinstance(reasons, dict) and set(reasons) <= INVALID_REASONS
            and all(type(count) is int
                                             and 0 <= count <= value["invalid_frames"]
                                             for key, count in reasons.items()),
            "Invalid reference-reason counts")
    require(sum(reasons.values()) >= value["invalid_frames"], "Missing invalid-reference reasons")


def _columns(value):
    require(isinstance(value, list) and all(isinstance(item, str) for item in value)
            and len(set(value)) == len(value) and set(value) <= SOURCE_COLUMNS
            and {"question", "video_path", "qid", "frame_count"}.issubset(value),
            "Missing or unexpected unchanged-input audit columns")


def _source_metadata(build, audit):
    """Validate every optional metadata value before retaining saved report bytes."""
    require(isinstance(build, dict) and set(build) <= PUBLIC_BUILD_FIELDS,
            "Unexpected source construction fields")
    require(isinstance(audit, dict) and set(audit) <= PUBLIC_AUDIT_FIELDS,
            "Unexpected independent source-audit fields")
    for key in ("binding_manifest_sha256", "canonical_zip_sha256_from_prior_verified_manifest",
                "ordered_video_paths_sha256", "ordered_hdf5_sha256", "ordered_question_sha256"):
        if key in build:
            _sha(build[key], key)
    if "schema_version" in build:
        require(type(build["schema_version"]) is int and build["schema_version"] == 1,
                "Wrong source report schema")
    if "canonical_source_revision" in build:
        require(isinstance(build["canonical_source_revision"], str)
                and re.fullmatch(r"[0-9a-f]{40}", build["canonical_source_revision"]) is not None,
                "Invalid canonical source revision")
    if "created_utc" in build:
        value = build["created_utc"]
        require(isinstance(value, str) and re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?\+00:00", value),
            "Invalid source construction timestamp")
        require(datetime.fromisoformat(value).utcoffset() == timedelta(0), "Source timestamp is not UTC")
    expected = {"parquet_name": "high_motion_high_fps_v2.parquet",
                "sampling": "endpoint-inclusive numpy.linspace(0,T-1,min(8,T),dtype=int)",
                "invalid_reason_counts_overlap": True, "zip_members_reread_in_this_build": False,
                "source_reprojection_is_not_verified_rgb_visibility": True}
    for key, value in expected.items():
        if key in build:
            require(type(build[key]) is type(value) and build[key] == value,
                    "Unexpected source construction metadata: " + key)
    if "preserved_columns" in build:
        _columns(build["preserved_columns"])
        require(build["preserved_columns"] == audit.get("full_nonanswer_columns_equal"),
                "Source construction and audit disagree on preserved columns")
    for key in ("source_hdf_projection_rerun", "per_prediction_metric_rescore_performed"):
        if key in audit:
            require(audit[key] is False, "Unsupported independent audit operation: " + key)
    if "prepare_summary_sha256" in audit:
        _sha(audit["prepare_summary_sha256"], "prepare_summary_sha256")
    if "elapsed_seconds" in audit:
        require(type(audit["elapsed_seconds"]) in (int, float)
                and math.isfinite(audit["elapsed_seconds"]) and audit["elapsed_seconds"] >= 0,
                "Invalid independent audit duration")
    if "baselines_verified" in audit:
        baselines = audit["baselines_verified"]
        require(isinstance(baselines, list) and len(baselines) == len(BASELINE_METHODS)
                and all(isinstance(row, dict) and set(row) == {"method", "report_sha256"}
                        and isinstance(row["method"], str) for row in baselines)
                and {row["method"] for row in baselines} == set(BASELINE_METHODS),
                "Independent audit must identify all 18 archived baseline reports")
        for row in baselines:
            _sha(row["report_sha256"], "audited baseline report")


def _validate_source_reports(manifest, build, audit, summary):
    _source_metadata(build, audit)
    require(build.get("status") == "source_reference_built"
            and type(build.get("rows")) is int and build["rows"] == 3243
            and build.get("benchmark_version") == VERSION and build.get("target_joint") == TARGET_JOINT
            and build.get("reference_policy") == MASK_POLICY, "Wrong source construction report")
    require(build.get("annotation_source_sha256") == SOURCE_ANNOTATION_SHA256
            and build.get("parquet_sha256") == manifest["references_sha256"],
            "Reference construction identifies different source/output bytes")
    require(build.get("original_annotation_unchanged") is True
            and type(build.get("rows_removed")) is int and build["rows_removed"] == 0
            and type(build.get("local_hdf5_actual_bytes_verified")) is int
            and build["local_hdf5_actual_bytes_verified"] == 3243,
            "Incomplete or input-changing source construction")
    for key in ("model_inference_performed", "model_results_used", "performance_or_superiority_claim",
                "automatic_publication"):
        require(build.get(key) is False, "Unsupported source construction claim: " + key)
    _coverage(build.get("full_frame_coverage"))
    _coverage(build.get("uniform_eight_frame_coverage"), sampled=True)
    require(audit.get("status") == "passed" and audit.get("original_annotation_sha256") == SOURCE_ANNOTATION_SHA256
            and audit.get("reference_sha256") == manifest["references_sha256"]
            and audit.get("input_sequence_sha256") == manifest["input_sequence_sha256"]
            and audit.get("source_hdf_bindings_verified_from_pinned_manifest") is True
            and type(audit.get("max_macro_abs_difference")) in (int, float)
            and audit["max_macro_abs_difference"] == 0,
            "Independent reference/numeric audit did not pass")
    for key in ("automatic_publication", "grt_outputs_read", "gpu_inference_performed"):
        require(audit.get(key) is False, "Unsupported independent source-audit claim")
    require(audit.get("full_frame_coverage") == build["full_frame_coverage"]
            and audit.get("sampled_full_coverage") == build["uniform_eight_frame_coverage"],
            "Independent and constructor reference coverage differ")
    _columns(audit.get("full_nonanswer_columns_equal"))
    baseline = next(row for row in summary["rows"] if row["method"] == BASELINE_METHOD)
    expected = {"preview_rows": 1000, "preview_scored_rows": baseline["records_with_scored_slots"],
                "preview_valid_slots": baseline["valid_slots"],
                "preview_fde_rows": baseline["metric_scored_records"]["grid_fde"],
                "preview_transition_rows": baseline["metric_scored_records"]["grid_transition_acc"],
                "preview_transition_edges": baseline["metric_scored_slots_or_edges"]["grid_transition_acc"]}
    require(all(type(audit.get(key)) is int and audit[key] == value for key, value in expected.items()),
            "Preview scoring coverage differs from independent reference audit")


def load_highmotion_v2_release(bundle, expected_manifest_sha256):
    """Authenticate four numeric artifacts and return a validated display summary."""
    require(bundle is not None, "An explicit corrected-reference release bundle is required")
    root = Path(bundle).expanduser().resolve(strict=True)
    require(root.is_dir(), "Release bundle is not a directory")
    require({path.name for path in root.iterdir()} == BUNDLE_FILES | {"manifest.json"},
            "Unexpected release directory membership")
    manifest = _json(_read_checked(root, "manifest.json", expected_manifest_sha256))
    require(isinstance(manifest, dict) and set(manifest) == PUBLIC_MANIFEST_FIELDS,
            "Unexpected release manifest fields")
    require(type(manifest.get("schema_version")) is int
            and manifest["schema_version"] == 1
            and manifest.get("benchmark_version") == VERSION
            and manifest.get("target_joint") == TARGET_JOINT
            and manifest.get("reference_policy") == MASK_POLICY
            and manifest.get("scorer_version") == SCORER_VERSION,
            "Wrong corrected-reference release version")
    require(manifest.get("scope") == "first-1000-source-rows"
            and type(manifest.get("source_reference_records")) is int
            and manifest["source_reference_records"] == 3243
            and type(manifest.get("method_count")) is int and manifest["method_count"] == 19
            and manifest.get("source_annotation_sha256") == SOURCE_ANNOTATION_SHA256,
            "Wrong release source population or comparison cohort")
    require(manifest.get("grt_method") == GRT_METHOD, "Unsupported GRT release configuration")
    require(isinstance(manifest.get("audit_date"), str)
            and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", manifest["audit_date"]), "Invalid release date")
    date.fromisoformat(manifest["audit_date"])
    for key in ("references_sha256", "input_sequence_sha256"):
        _sha(manifest.get(key), key)
    require(manifest.get("source_code_sha256") == SOURCE_FILES, "Source constructor/scorer pins differ")
    for name, expected in SOURCE_FILES.items():
        require(_digest(Path(__file__).with_name(name).read_bytes()) == expected,
                "Installed source differs from the numeric contract: " + name)
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == BUNDLE_FILES, "Wrong numeric bundle membership")
    raw = {name: _read_checked(root, name, digest) for name, digest in files.items()}
    with gzip.GzipFile(fileobj=io.BytesIO(raw["numeric_reports.json.gz"]), mode="rb") as compressed:
        decoded = compressed.read(MAX_NUMERIC_BYTES + 1)
    require(len(decoded) <= MAX_NUMERIC_BYTES, "Oversized decompressed numeric evidence")
    reports = _json(decoded)
    receipt = _json(raw["generation_receipt.json"])
    require(isinstance(reports, list) and all(isinstance(report, dict)
            and set(report) <= PUBLIC_REPORT_FIELDS for report in reports),
            "Numeric release contains unexpected report fields")
    for report in reports:
        if "comparison_caveat" in report:
            require(report["comparison_caveat"] == SAVED_COMPARISON_CAVEAT,
                    "Unexpected saved comparison caveat")
        rows = report.get("records_detail")
        require(isinstance(rows, list) and all(isinstance(row, dict)
                and set(row) == PUBLIC_ROW_FIELDS for row in rows),
                "Numeric release contains unexpected per-row fields")
    require(isinstance(receipt, dict) and set(receipt) == PUBLIC_GENERATION_FIELDS,
            "Generation receipt must contain only its public integrity fields")
    summary = validate_reports(
        reports, reference_sha256=manifest["references_sha256"],
        input_sequence_sha256=manifest["input_sequence_sha256"],
        grt_method=manifest.get("grt_method"), grt_receipt=receipt,
    )
    build, audit = _json(raw["reference_build_report.json"]), _json(raw["reference_audit.json"])
    _validate_source_reports(manifest, build, audit, summary)
    # Public-facing data contain only numeric summaries, bounded metadata and
    # source hashes. Do not copy arbitrary manifest/receipt fields or local paths.
    summary.update({
        "release_integrity_verified": True, "release_manifest_sha256": expected_manifest_sha256,
        "audit_date": manifest["audit_date"], "source_annotation_sha256": SOURCE_ANNOTATION_SHA256,
        "source_code_sha256": dict(SOURCE_FILES), "evidence_file_sha256": dict(files),
        "full_frame_coverage": build["full_frame_coverage"],
        "uniform_eight_frame_coverage": build["uniform_eight_frame_coverage"],
        "verification": "authenticated saved numeric evidence and reaggregation; no new inference, source projection or per-prediction rescoring",
        "reference_geometry_recomputed_by_this_loader": False,
        "raw_prediction_files_rehashed_by_this_loader": False,
    })
    for row in summary["rows"]:
        row["model"] = ("GRT · LLaVA-OneVision HF 0.5B (motion SSIM 0.001)"
                        if row["method"] == manifest["grt_method"] else MODEL_LABELS[row["method"]])
    return summary
