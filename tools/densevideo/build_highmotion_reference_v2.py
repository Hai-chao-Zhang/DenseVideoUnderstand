"""Build an immutable, source-bound High-motion v2 annotation artifact on CPU."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re

from tools.densevideo.highmotion_reference_v2 import (
    MASK_POLICY,
    TARGET_JOINT,
    VERSION,
    _dimension,
    project_reference,
)


EXPECTED_ROWS = 3243
PARQUET_NAME = "high_motion_high_fps_v2.parquet"
REPORT_NAME = "build_report.json"
ADDED_FIELDS = (
    "reference_valid", "reference_invalid_reason", "benchmark_version",
    "target_joint", "reference_policy", "source_hdf5_sha256",
    "legacy_question_sha256", "legacy_answer_sha256",
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _check_sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _verified_bytes(path, expected, name):
    _check_sha(expected, name)
    value = Path(path).read_bytes()
    if _digest(value) != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    return value


def _source_path(root, video_path):
    if not isinstance(video_path, str) or "\\" in video_path:
        raise ValueError("Invalid video_path")
    path = PurePosixPath(video_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 3 or path.parts[0] != "egodex":
        raise ValueError("Unsafe or unexpected canonical video_path")
    if path.suffix != ".mp4" or str(path) != video_path:
        raise ValueError("Noncanonical video_path")
    candidate = root.joinpath(*path.with_suffix(".hdf5").parts[1:])
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        raise ValueError("Source HDF5 is missing, symlinked, or outside source-root")
    return candidate


def _coverage():
    return {"rows": 0, "frames": 0, "valid_frames": 0, "invalid_frames": 0,
            "all_invalid_rows": 0, "fully_valid_rows": 0, "invalid_reason_counts": Counter()}


def _add_coverage(target, valid, reasons):
    target["rows"] += 1
    target["frames"] += len(valid)
    target["valid_frames"] += sum(valid)
    target["invalid_frames"] += len(valid) - sum(valid)
    target["all_invalid_rows"] += int(not any(valid))
    target["fully_valid_rows"] += int(all(valid))
    for reason in reasons:
        target["invalid_reason_counts"].update(reason.split("|") if reason else [])


def build_reference_v2(annotation, annotation_sha256, source_root, binding_manifest,
                       binding_manifest_sha256, output_dir, *, expected_rows=EXPECTED_ROWS):
    """Build all source-bound rows; expected_rows override is for synthetic tests.

    The pinned binding manifest records a prior complete ZIP-member SHA audit.
    This builder verifies actual local HDF5 bytes against those bindings; it
    does not reread the large ZIP and does not claim to have done so. Existing
    output directories are rejected, including incomplete previous attempts.
    """
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected_rows = _dimension(expected_rows, "expected_rows")
    annotation = Path(annotation)
    binding_manifest = Path(binding_manifest)
    source_root = Path(source_root).resolve(strict=True)
    output_dir = Path(output_dir)
    if not source_root.is_dir() or not output_dir.parent.is_dir():
        raise ValueError("Source-root and output parent must already exist")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory must not already exist")
    annotation_bytes = _verified_bytes(annotation, annotation_sha256, "annotation")
    manifest_bytes = _verified_bytes(binding_manifest, binding_manifest_sha256, "binding_manifest")
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "passed":
        raise ValueError("Binding manifest did not pass")
    archive = manifest.get("canonical_archive", {})
    if archive.get("all_3243_hdf5_members_read_and_sha256_matched_to_original_test") is not True:
        raise ValueError("Missing prior complete canonical ZIP-member verification")
    archive_sha = _check_sha(archive.get("sha256_from_prior_verified_manifest"), "canonical_archive")
    revision = manifest.get("canonical_source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Missing canonical source revision")
    bindings = manifest.get("canonical_test_hdf5_bindings")
    if not isinstance(bindings, list) or len(bindings) != expected_rows:
        raise ValueError("Canonical HDF5 binding count differs")
    table = pq.read_table(pa.BufferReader(annotation_bytes))
    required = {"video_path", "question", "answer", "answer_traj", "qid", "frame_count", "width", "height"}
    if not required.issubset(table.column_names) or table.num_rows != expected_rows:
        raise ValueError("Annotation schema or canonical row count differs")
    if len(set(table.column_names)) != len(table.column_names) or set(ADDED_FIELDS) & set(table.column_names):
        raise ValueError("Ambiguous columns or an already versioned annotation")
    rows = table.to_pylist()
    paths = [row["video_path"] for row in rows]
    if paths != [item.get("video_path") for item in bindings] or len(set(paths)) != expected_rows:
        raise ValueError("Canonical annotation/binding identities or ordered membership differ")
    original_umask = os.umask(0o077)
    try:
        output_dir.mkdir(mode=0o700)
        columns = {name: [] for name in ("answer", "answer_traj") + ADDED_FIELDS}
        full, sampled = _coverage(), _coverage()
        hdf_hashes = []
        for row, binding in zip(rows, bindings):
            if not all(isinstance(row[name], str) for name in ("question", "answer", "answer_traj")):
                raise ValueError("Legacy question and answers must be strings")
            frame_count = _dimension(row["frame_count"], "frame_count")
            width, height = _dimension(row["width"], "width"), _dimension(row["height"], "height")
            path = _source_path(source_root, row["video_path"])
            data = _verified_bytes(path, binding.get("hdf5_sha256"), "source_hdf5")
            if len(data) != _dimension(binding.get("hdf5_bytes"), "hdf5_bytes"):
                raise ValueError("Source HDF5 size differs from canonical binding")
            with h5py.File(io.BytesIO(data), "r") as source:
                def value(key):
                    return np.asarray(source[key]) if key in source else None

                camera, joint = value("transforms/camera"), value(f"transforms/{TARGET_JOINT}")
                for transforms in (camera, joint):
                    if transforms is not None and transforms.shape != (frame_count, 4, 4):
                        raise ValueError("HDF5 transforms differ from annotation frame_count")
                result = project_reference(value("camera/intrinsic"), camera, joint,
                                           value(f"confidences/{TARGET_JOINT}"), width, height)
            if len(result["answer"]) != frame_count:
                raise ValueError("Generated reference frame count differs")
            for name in ("answer", "answer_traj", "reference_valid", "reference_invalid_reason"):
                columns[name].append(_json(result[name]))
            for name, value_ in (("benchmark_version", VERSION), ("target_joint", TARGET_JOINT),
                                 ("reference_policy", MASK_POLICY), ("source_hdf5_sha256", _digest(data)),
                                 ("legacy_question_sha256", _digest(row["question"].encode("utf-8"))),
                                 ("legacy_answer_sha256", _digest(row["answer"].encode("utf-8")))):
                columns[name].append(value_)
            hdf_hashes.append(_digest(data))
            _add_coverage(full, result["reference_valid"], result["reference_invalid_reason"])
            indices = np.linspace(0, frame_count - 1, min(8, frame_count), dtype=int)
            _add_coverage(sampled, [result["reference_valid"][int(i)] for i in indices],
                          [result["reference_invalid_reason"][int(i)] for i in indices])

        updated = table
        for name in ("answer", "answer_traj"):
            index = updated.schema.get_field_index(name)
            updated = updated.set_column(index, updated.schema.field(index),
                                         pa.array(columns[name], type=updated.schema.field(index).type))
        for name in ADDED_FIELDS:
            updated = updated.append_column(name, pa.array(columns[name], type=pa.string()))
        unchanged = [name for name in table.column_names if name not in ("answer", "answer_traj")]
        if not table.select(unchanged).equals(updated.select(unchanged), check_metadata=True):
            raise ValueError("A preserved annotation field changed")
        parquet_path = output_dir / PARQUET_NAME
        with parquet_path.open("xb") as handle:
            pq.write_table(updated, handle, compression="zstd")
            handle.flush()
            os.fsync(handle.fileno())
        # Re-read the actual output, not only the in-memory table.
        actual = pq.read_table(parquet_path)
        if not actual.equals(updated, check_metadata=True):
            raise ValueError("Written Parquet failed round-trip equality")
        _verified_bytes(annotation, annotation_sha256, "annotation_after_build")
        _verified_bytes(binding_manifest, binding_manifest_sha256, "binding_manifest_after_build")
        report = {
            "schema_version": 1, "status": "source_reference_built",
            "benchmark_version": VERSION, "target_joint": TARGET_JOINT, "reference_policy": MASK_POLICY,
            "created_utc": datetime.now(timezone.utc).isoformat(), "rows": expected_rows,
            "annotation_source_sha256": annotation_sha256, "original_annotation_unchanged": True,
            "binding_manifest_sha256": binding_manifest_sha256, "canonical_source_revision": revision,
            "canonical_zip_sha256_from_prior_verified_manifest": archive_sha,
            "zip_members_reread_in_this_build": False, "local_hdf5_actual_bytes_verified": expected_rows,
            "ordered_video_paths_sha256": _digest(_json(paths).encode("utf-8")),
            "ordered_hdf5_sha256": _digest(b"".join(bytes.fromhex(item) for item in hdf_hashes)),
            "ordered_question_sha256": _digest(_json(columns["legacy_question_sha256"]).encode("utf-8")),
            "parquet_name": PARQUET_NAME, "parquet_sha256": _digest(parquet_path.read_bytes()),
            "preserved_columns": unchanged, "full_frame_coverage": full,
            "uniform_eight_frame_coverage": sampled,
            "sampling": "endpoint-inclusive numpy.linspace(0,T-1,min(8,T),dtype=int)",
            "invalid_reason_counts_overlap": True, "rows_removed": 0,
            "model_inference_performed": False, "model_results_used": False,
            "performance_or_superiority_claim": False, "automatic_publication": False,
            "source_reprojection_is_not_verified_rgb_visibility": True,
        }
        with (output_dir / REPORT_NAME).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return report
    finally:
        os.umask(original_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--binding-manifest", type=Path, required=True)
    parser.add_argument("--binding-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_reference_v2(**vars(args))
    print(json.dumps({key: report[key] for key in ("status", "benchmark_version", "rows", "parquet_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
