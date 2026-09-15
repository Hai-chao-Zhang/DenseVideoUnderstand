"""Verify locally authorized High-motion sources without generating labels.

This CPU-only tool replaces any dependency on a private experiment manifest.
It verifies the complete canonical ZIP, then matches each required HDF5 member
to a local source file without extracting or redistributing the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile


ANNOTATION_SHA256 = "518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd"
ARCHIVE_SHA256 = "ed06c8ef7e43f61901a11898dd5d4b9d265d411299c2174c772668e567f79e49"
SOURCE_REVISION = "d44407f607fdf020c59b816884f06ed6d453cf26"
EXPECTED_ROWS = 3243
MANIFEST_NAME = "source_bindings.json"
REPORT_NAME = "binding_report.json"
BLOCK_BYTES = 8 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _expected_sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             f"{name} must be a lowercase SHA-256")
    return value


def _snapshot(path):
    value = Path(path).stat()
    _require(stat.S_ISREG(value.st_mode), "Input must be a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _handle_snapshot(handle):
    value = os.fstat(handle.fileno())
    _require(stat.S_ISREG(value.st_mode), "Opened input must be a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _stream_sha(handle):
    digest, count = hashlib.sha256(), 0
    for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def _safe_name(name, *, directory=False):
    _require(isinstance(name, str) and bool(name) and "\\" not in name and "\x00" not in name,
             "Unsafe archive/member path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and ".." not in path.parts and path.parts[0] == "egodex",
             "Unsafe or unexpected canonical archive prefix")
    expected = str(path) + ("/" if directory else "")
    _require(name == expected, "Noncanonical archive/member path")
    return path


def _archive_entries(archive):
    entries = {}
    for info in archive.infolist():
        _require(info.orig_filename == info.filename, "Truncated archive filename")
        _safe_name(info.filename, directory=info.is_dir())
        _require(info.filename not in entries, "Duplicate archive entry")
        mode = stat.S_IFMT(info.external_attr >> 16)
        _require(mode in (0, stat.S_IFREG, stat.S_IFDIR), "Symlink or special archive entry")
        if mode == stat.S_IFDIR:
            _require(info.is_dir(), "Archive directory type/name mismatch")
        if mode == stat.S_IFREG:
            _require(not info.is_dir(), "Archive regular-file type/name mismatch")
        _require(not info.flag_bits & 1, "Encrypted archive entries are not supported")
        entries[info.filename] = info
    return entries


def _local_hdf5(root, relative):
    candidate = root.joinpath(*relative.with_suffix(".hdf5").parts[1:])
    _require(not candidate.is_symlink() and candidate.is_file()
             and candidate.resolve().is_relative_to(root),
             "Local HDF5 is missing, symlinked, or outside source-root")
    return candidate


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def bind_sources(annotation, annotation_sha256, archive, archive_sha256, source_root,
                 output_dir, *, expected_rows=EXPECTED_ROWS):
    """Produce a constructor-compatible binding manifest from verified sources.

    ``expected_rows`` is overridable only in the library for synthetic tests;
    the public CLI fixes the canonical revision, hashes and all 3,243 rows.
    Local inputs are read only. An existing output directory is always refused.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    _require(type(expected_rows) is int and expected_rows > 0, "Invalid expected row count")
    _expected_sha(annotation_sha256, "annotation_sha256")
    _expected_sha(archive_sha256, "archive_sha256")
    annotation, archive, output_dir = Path(annotation), Path(archive), Path(output_dir)
    source_root = Path(source_root).resolve(strict=True)
    _require(source_root.is_dir() and output_dir.parent.is_dir(), "Source-root and output parent must exist")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory must not already exist")
    annotation_before, archive_before = _snapshot(annotation), _snapshot(archive)
    with annotation.open("rb") as handle:
        _require(_handle_snapshot(handle) == annotation_before, "Annotation changed before opening")
        annotation_bytes = handle.read()
        _require(_handle_snapshot(handle) == annotation_before, "Annotation changed during reading")
    _require(hashlib.sha256(annotation_bytes).hexdigest() == annotation_sha256, "Annotation SHA-256 mismatch")
    table = pq.read_table(pa.BufferReader(annotation_bytes), columns=["video_path"])
    _require(table.num_rows == expected_rows, "Wrong canonical annotation row count")
    if expected_rows == EXPECTED_ROWS:
        _require(annotation_sha256 == ANNOTATION_SHA256 and archive_sha256 == ARCHIVE_SHA256,
                 "Full canonical binding requires the published annotation and archive hashes")
    paths = table["video_path"].to_pylist()
    _require(all(isinstance(path, str) for path in paths) and len(set(paths)) == expected_rows,
             "Canonical video identities must be unique strings")
    relative_paths = [_safe_name(name) for name in paths]
    _require(all(len(path.parts) == 3 and path.suffix == ".mp4" for path in relative_paths),
             "Canonical video paths must be egodex/<action>/<id>.mp4")

    original_umask = os.umask(0o077)
    try:
        output_dir.mkdir(mode=0o700)
        bindings, local_snapshots = [], []
        with archive.open("rb") as archive_handle:
            _require(_handle_snapshot(archive_handle) == archive_before, "Archive changed before opening")
            actual_archive_sha, archive_bytes = _stream_sha(archive_handle)
            _require(actual_archive_sha == archive_sha256, "Complete archive SHA-256 mismatch")
            _require(archive_bytes == archive_before[2], "Archive size changed during checksum")
            _require(_handle_snapshot(archive_handle) == archive_before, "Archive changed during checksum")
            archive_handle.seek(0)
            with zipfile.ZipFile(archive_handle, "r") as zipped:
                entries = _archive_entries(zipped)
                required_hdf5 = {str(path.with_suffix(".hdf5")) for path in relative_paths}
                actual_hdf5 = {name for name, info in entries.items() if not info.is_dir() and name.endswith(".hdf5")}
                _require(actual_hdf5 == required_hdf5, "Archive HDF5 membership differs from canonical annotations")
                for relative in relative_paths:
                    video = str(relative)
                    _require(video in entries and not entries[video].is_dir(), "Paired canonical video member is missing")
                    member = str(relative.with_suffix(".hdf5"))
                    info = entries[member]
                    _require(info.file_size > 0, "Empty HDF5 source member")
                    local = _local_hdf5(source_root, relative)
                    before = _snapshot(local)
                    _require(before[2] == info.file_size, "Local/archive HDF5 sizes differ")
                    with local.open("rb") as local_handle:
                        _require(_handle_snapshot(local_handle) == before, "Local HDF5 changed before opening")
                        local_sha, local_bytes = _stream_sha(local_handle)
                        _require(_handle_snapshot(local_handle) == before, "Local HDF5 changed during checksum")
                    with zipped.open(info, "r") as member_handle:
                        member_sha, member_bytes = _stream_sha(member_handle)
                    _require(local_bytes == member_bytes == info.file_size and local_sha == member_sha,
                             "Actual ZIP HDF5 member differs from local source")
                    _require(_snapshot(local) == before, "Local HDF5 changed during verification")
                    local_snapshots.append((local, before))
                    bindings.append({"video_path": video, "hdf5_sha256": local_sha, "hdf5_bytes": local_bytes})
            _require(_handle_snapshot(archive_handle) == archive_before, "Archive changed during member verification")
        for local, before in local_snapshots:
            _require(_snapshot(local) == before, "A previously verified local HDF5 changed")
        _require(_snapshot(annotation) == annotation_before and _snapshot(archive) == archive_before,
                 "An original input changed during binding")

        manifest = {
            "status": "passed", "canonical_source_revision": SOURCE_REVISION,
            "canonical_archive": {
                "all_3243_hdf5_members_read_and_sha256_matched_to_original_test": True,
                # Name retained for compatibility with the v2 constructor. The
                # value is the actual complete checksum performed above.
                "sha256_from_prior_verified_manifest": actual_archive_sha,
                "full_archive_sha256_recomputed_in_this_audit": True,
                "archive_sha256_source": "actual complete archive bytes streamed in this binding run",
            },
            "canonical_test_hdf5_bindings": bindings,
        }
        manifest_path = output_dir / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report = {
            "schema_version": 1, "status": "canonical_sources_bound", "rows": expected_rows,
            "canonical_source_revision": SOURCE_REVISION, "annotation_sha256": annotation_sha256,
            "archive_sha256": actual_archive_sha, "archive_bytes": archive_bytes,
            "full_archive_sha256_recomputed": True, "actual_hdf5_members_verified": len(bindings),
            "local_hdf5_bytes_verified": sum(item["hdf5_bytes"] for item in bindings),
            "ordered_video_paths_sha256": hashlib.sha256(json.dumps(
                paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "ordered_hdf5_sha256": hashlib.sha256(b"".join(
                bytes.fromhex(item["hdf5_sha256"]) for item in bindings)).hexdigest(),
            "manifest_name": MANIFEST_NAME, "manifest_sha256": manifest_sha,
            "original_inputs_metadata_unchanged": True, "archive_extraction_performed": False,
            "derived_labels_generated": False, "model_results_used": False,
            "model_inference_performed": False, "automatic_publication": False,
            "redistribution_authorization_claim": False, "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(output_dir / REPORT_NAME, report)
        return report
    finally:
        os.umask(original_umask)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Authorized local EgoDex test root containing <action>/<id>.hdf5")
    parser.add_argument("--output-dir", type=Path, required=True, help="New private directory; must not exist")
    args = parser.parse_args(argv)
    _require(args.annotation_sha256 == ANNOTATION_SHA256 and args.archive_sha256 == ARCHIVE_SHA256,
             "CLI requires the published canonical annotation and archive SHA-256 values")
    report = bind_sources(**vars(args))
    print(json.dumps({key: report[key] for key in ("status", "rows", "manifest_name", "manifest_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
