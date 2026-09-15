"""Synthetic numeric-only bundles; never access actual data or GPU results."""

import gzip
import hashlib
import json
from copy import deepcopy

import pytest
from test_highmotion_v2_bundle import GRT as SYNTHETIC_GRT
from test_highmotion_v2_bundle import INPUT_SHA, REFERENCE_SHA, receipt, report

from tools.densevideo import highmotion_v2_release as release
from tools.densevideo.highmotion_v2_bundle import BASELINE_METHODS
from tools.densevideo.highmotion_v2_scoring import (
    MASK_POLICY,
    SCORER_VERSION,
    TARGET_JOINT,
    VERSION,
)

GRT = release.GRT_METHOD


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


@pytest.fixture(scope="module")
def originals():
    reports = [report(method) for method in BASELINE_METHODS]
    candidate = report(SYNTHETIC_GRT)
    candidate["method"] = GRT
    reports.append(candidate)
    coverage = {"rows": 3243, "frames": 3243 * 15, "valid_frames": 3243 * 15,
                "invalid_frames": 0, "all_invalid_rows": 0, "fully_valid_rows": 3243,
                "invalid_reason_counts": {}}
    sampled = dict(coverage, frames=3243 * 8, valid_frames=3243 * 8)
    build = {"status": "source_reference_built", "rows": 3243, "benchmark_version": VERSION,
             "target_joint": TARGET_JOINT, "reference_policy": MASK_POLICY,
             "annotation_source_sha256": release.SOURCE_ANNOTATION_SHA256,
             "parquet_sha256": REFERENCE_SHA, "original_annotation_unchanged": True,
             "rows_removed": 0, "local_hdf5_actual_bytes_verified": 3243,
             "model_inference_performed": False, "model_results_used": False,
             "performance_or_superiority_claim": False, "automatic_publication": False,
             "full_frame_coverage": coverage, "uniform_eight_frame_coverage": sampled}
    audit = {"status": "passed", "original_annotation_sha256": release.SOURCE_ANNOTATION_SHA256,
             "reference_sha256": REFERENCE_SHA, "input_sequence_sha256": INPUT_SHA,
             "source_hdf_bindings_verified_from_pinned_manifest": True, "max_macro_abs_difference": 0.0,
             "automatic_publication": False, "grt_outputs_read": False, "gpu_inference_performed": False,
             "full_frame_coverage": coverage, "sampled_full_coverage": sampled,
             "full_nonanswer_columns_equal": ["question", "video_path", "qid", "frame_count"],
             "preview_rows": 1000, "preview_scored_rows": 1000, "preview_valid_slots": 8000,
             "preview_fde_rows": 1000, "preview_transition_rows": 1000, "preview_transition_edges": 7000}
    return {"numeric_reports.json.gz": reports, "generation_receipt.json": receipt(),
            "reference_build_report.json": build, "reference_audit.json": audit}


def write_bundle(path, originals, mutate=None):
    documents = deepcopy(originals)
    manifest = {"schema_version": 1, "benchmark_version": VERSION, "target_joint": TARGET_JOINT,
                "reference_policy": MASK_POLICY, "scorer_version": SCORER_VERSION,
                "scope": "first-1000-source-rows", "source_reference_records": 3243, "method_count": 19,
                "source_annotation_sha256": release.SOURCE_ANNOTATION_SHA256, "audit_date": "2026-09-15",
                "references_sha256": REFERENCE_SHA, "input_sequence_sha256": INPUT_SHA,
                "source_code_sha256": dict(release.SOURCE_FILES), "grt_method": GRT}
    if mutate:
        mutate(documents, manifest)
    path.mkdir()
    manifest["files"] = {}
    for name, document in documents.items():
        raw = encoded(document)
        if name.endswith(".gz"):
            raw = gzip.compress(raw, mtime=0)
        (path / name).write_bytes(raw)
        manifest["files"][name] = sha(raw)
    raw = encoded(manifest)
    (path / "manifest.json").write_bytes(raw)
    return sha(raw)


def test_complete_authenticated_bundle_returns_only_numeric_summaries(tmp_path, originals):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    summary = release.load_highmotion_v2_release(path, digest)
    assert summary["release_integrity_verified"] is True
    assert summary["release_manifest_sha256"] == digest
    assert len(summary["rows"]) == 19 and summary["rows"][0]["method"] == GRT
    assert summary["comparison"]["grid_acc_outperform"] is True
    assert all("records_detail" not in row and "doc" not in row for row in summary["rows"])
    for key in ("inference_performed", "reference_geometry_recomputed_by_this_loader",
                "raw_prediction_files_rehashed_by_this_loader", "automatic_publication"):
        assert summary[key] is False
    assert "reaggregation" in summary["verification"]


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("benchmark_version", "legacy"), ("target_joint", "leftIndexFingerMetacarpal"),
    ("scope", "full3243"), ("source_reference_records", 1000), ("method_count", 18),
    ("source_annotation_sha256", "c" * 64), ("audit_date", "2026-99-01"),
    ("references_sha256", "not-a-checksum"), ("input_sequence_sha256", "bad"),
    ("source_code_sha256", {}), ("grt_method", "llava_onevision_0_5b"),
    ("source_reference_records", 3243.0), ("method_count", 19.0),
])
def test_wrong_release_contract_rejected(tmp_path, originals, field, value):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, lambda docs, manifest: manifest.update({field: value}))
    with pytest.raises((ValueError, TypeError)):
        release.load_highmotion_v2_release(path, digest)


@pytest.mark.parametrize("name", ["manifest.json", *sorted(release.BUNDLE_FILES)])
def test_every_actual_file_hash_is_checked(tmp_path, originals, name):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    with (path / name).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        release.load_highmotion_v2_release(path, digest)


@pytest.mark.parametrize("field,value", [("rows_removed", 1), ("rows_removed", False),
                                        ("original_annotation_unchanged", False),
                                        ("local_hdf5_actual_bytes_verified", 3242),
                                        ("local_hdf5_actual_bytes_verified", 3243.0), ("rows", 3243.0),
                                        ("model_results_used", True), ("parquet_sha256", "c" * 64)])
def test_bad_source_construction_rejected(tmp_path, originals, field, value):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, lambda docs, manifest:
                          docs["reference_build_report.json"].update({field: value}))
    with pytest.raises(ValueError):
        release.load_highmotion_v2_release(path, digest)


@pytest.mark.parametrize("field,value", [("preview_valid_slots", 7999), ("preview_rows", 999),
                                        ("max_macro_abs_difference", False), ("grt_outputs_read", True),
                                        ("full_nonanswer_columns_equal", ["qid"])])
def test_bad_independent_reference_audit_rejected(tmp_path, originals, field, value):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, lambda docs, manifest:
                          docs["reference_audit.json"].update({field: value}))
    with pytest.raises(ValueError):
        release.load_highmotion_v2_release(path, digest)


@pytest.mark.parametrize("where", ["report", "row", "receipt", "build", "audit", "manifest"])
def test_private_fields_cannot_enter_numeric_release(tmp_path, originals, where):
    def mutate(docs, manifest):
        target = (docs["generation_receipt.json"] if where == "receipt" else
                  docs["reference_build_report.json"] if where == "build" else
                  docs["reference_audit.json"] if where == "audit" else
                  manifest if where == "manifest" else
                  docs["numeric_reports.json.gz"][0] if where == "report" else
                  docs["numeric_reports.json.gz"][0]["records_detail"][0])
        target["private_prompt_or_path"] = "must not enter the release"
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, mutate)
    with pytest.raises(ValueError, match="fields"):
        release.load_highmotion_v2_release(path, digest)


def test_partial_numeric_coverage_is_not_a_release(tmp_path, originals):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, lambda docs, manifest: docs["numeric_reports.json.gz"].pop())
    with pytest.raises(ValueError, match="18 baselines"):
        release.load_highmotion_v2_release(path, digest)


def test_duplicate_json_and_nonfinite_constants_rejected():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
        with pytest.raises(ValueError):
            release._json(raw)


def test_oversized_member_rejected_before_read(tmp_path, originals, monkeypatch):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    monkeypatch.setattr(release, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="Oversized"):
        release.load_highmotion_v2_release(path, digest)


def test_bounded_decompression(tmp_path, originals, monkeypatch):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    monkeypatch.setattr(release, "MAX_NUMERIC_BYTES", 1024)
    with pytest.raises(ValueError, match="decompressed"):
        release.load_highmotion_v2_release(path, digest)


def test_symlinked_member_is_rejected(tmp_path, originals):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    source = path / "reference_audit.json"
    retained = tmp_path / "retained.json"
    source.rename(retained)
    source.symlink_to(retained)
    with pytest.raises(ValueError, match="symlinked"):
        release.load_highmotion_v2_release(path, digest)


def test_unlisted_file_cannot_enter_publication_directory(tmp_path, originals):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals)
    (path / "private_log.txt").write_text("synthetic private text")
    with pytest.raises(ValueError, match="directory membership"):
        release.load_highmotion_v2_release(path, digest)


def test_free_text_cannot_hide_inside_saved_caveat(tmp_path, originals):
    path = tmp_path / "bundle"
    digest = write_bundle(path, originals, lambda docs, manifest:
                          docs["numeric_reports.json.gz"][0].update(comparison_caveat="private text"))
    with pytest.raises(ValueError, match="caveat"):
        release.load_highmotion_v2_release(path, digest)


@pytest.mark.parametrize("where", ["coverage", "reason"])
def test_nested_coverage_metadata_is_bounded(originals, where):
    coverage = deepcopy(originals["reference_build_report.json"]["full_frame_coverage"])
    target = coverage if where == "coverage" else coverage["invalid_reason_counts"]
    target["private_prompt_or_path"] = 0
    with pytest.raises(ValueError):
        release._coverage(coverage)


def test_full_saved_metadata_schema_is_supported(originals):
    build = deepcopy(originals["reference_build_report.json"])
    audit = deepcopy(originals["reference_audit.json"])
    build.update(schema_version=1, created_utc="2026-09-15T01:02:03.123456+00:00",
                 canonical_source_revision="d" * 40, parquet_name="high_motion_high_fps_v2.parquet",
                 preserved_columns=sorted(release.SOURCE_COLUMNS),
                 sampling="endpoint-inclusive numpy.linspace(0,T-1,min(8,T),dtype=int)",
                 invalid_reason_counts_overlap=True, zip_members_reread_in_this_build=False,
                 source_reprojection_is_not_verified_rgb_visibility=True)
    for key in ("binding_manifest_sha256", "canonical_zip_sha256_from_prior_verified_manifest",
                "ordered_video_paths_sha256", "ordered_hdf5_sha256", "ordered_question_sha256"):
        build[key] = "f" * 64
    audit.update(full_nonanswer_columns_equal=build["preserved_columns"], elapsed_seconds=42.5,
                 prepare_summary_sha256="e" * 64, source_hdf_projection_rerun=False,
                 per_prediction_metric_rescore_performed=False,
                 baselines_verified=[{"method": method, "report_sha256": "c" * 64}
                                     for method in BASELINE_METHODS])
    release._source_metadata(build, audit)
    release._columns(audit["full_nonanswer_columns_equal"])


@pytest.mark.parametrize("where,field,value", [
    ("build", "binding_manifest_sha256", "private text"),
    ("build", "canonical_source_revision", "/private/path"),
    ("build", "created_utc", "2026-09-15T01:02:03+00:00 private text"),
    ("build", "created_utc", "2026-09-99T01:02:03+00:00"),
    ("build", "parquet_name", "private/path.parquet"),
    ("build", "sampling", "private prompt"),
    ("build", "preserved_columns", ["question", "video_path", "qid", "frame_count", "private text"]),
    ("build", "invalid_reason_counts_overlap", 1),
    ("audit", "elapsed_seconds", True),
    ("audit", "elapsed_seconds", float("inf")),
    ("audit", "prepare_summary_sha256", "private text"),
    ("audit", "source_hdf_projection_rerun", True),
    ("audit", "per_prediction_metric_rescore_performed", True),
    ("audit", "baselines_verified", [{"method": "private text", "report_sha256": "a" * 64}]),
])
def test_optional_source_metadata_cannot_smuggle_text_or_claims(originals, where, field, value):
    build = deepcopy(originals["reference_build_report.json"])
    audit = deepcopy(originals["reference_audit.json"])
    (build if where == "build" else audit)[field] = value
    with pytest.raises(ValueError):
        release._source_metadata(build, audit)


@pytest.mark.parametrize("value", [[], "private text", None])
def test_source_reports_must_be_objects(originals, value):
    with pytest.raises(ValueError, match="fields"):
        release._source_metadata(value, originals["reference_audit.json"])
    with pytest.raises(ValueError, match="fields"):
        release._source_metadata(originals["reference_build_report.json"], value)


@pytest.mark.parametrize("columns", [
    ["question", "video_path", "qid", "frame_count", "private text"],
    ["question", "video_path", "qid", "frame_count", "qid"],
    ["question", "video_path", "qid", "frame_count", {}],
])
def test_input_column_names_are_bounded(columns):
    with pytest.raises(ValueError, match="columns"):
        release._columns(columns)
