"""Synthetic CPU-only fixtures for the portable, no-extraction source binder."""

import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.densevideo import bind_highmotion_sources as binder
from tools.densevideo.build_highmotion_reference_v2 import build_reference_v2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zip_entry(name, data, mode=stat.S_IFREG | 0o600):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    return info, data


@pytest.fixture
def source_data(tmp_path):
    root = tmp_path / "authorized-test"
    action = root / "synthetic"
    action.mkdir(parents=True)
    rows, entries = [], [zip_entry("egodex/", b"", stat.S_IFDIR | 0o700)]
    for index in range(2):
        local = action / f"{index}.hdf5"
        camera = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        joint = camera.copy()
        joint[:, :3, 3] = [1, 1, 1]
        with h5py.File(local, "w") as source:
            source.create_dataset("camera/intrinsic", data=np.eye(3, dtype=np.float32))
            source.create_dataset("transforms/camera", data=camera)
            source.create_dataset("transforms/rightRingFingerMetacarpal", data=joint)
            source.create_dataset("confidences/rightRingFingerMetacarpal", data=np.ones(3, dtype=np.float32))
        video = f"egodex/synthetic/{index}.mp4"
        rows.append({"video_path": video, "question": "Original right-hand question", "qid": 0,
                     "answer": '["middle","middle","middle"]', "answer_traj": "[[0,0],[0,0],[0,0]]",
                     "frame_count": 3, "width": 9, "height": 9})
        entries.extend([zip_entry(video, b"synthetic-video-not-decoded"),
                        zip_entry(str(Path(video).with_suffix(".hdf5")), local.read_bytes())])
    annotation = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist(rows), annotation)
    archive = tmp_path / "video.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for info, data in entries:
            handle.writestr(info, data)
    return {"annotation": annotation, "annotation_sha256": sha(annotation), "archive": archive,
            "archive_sha256": sha(archive), "source_root": root, "output_dir": tmp_path / "bindings",
            "expected_rows": 2}, entries


def rewrite_archive(arguments, entries):
    with zipfile.ZipFile(arguments["archive"], "w") as handle:
        for info, data in entries:
            handle.writestr(info, data)
    arguments["archive_sha256"] = sha(arguments["archive"])


def test_manifest_is_portable_ordered_and_constructor_compatible(source_data, tmp_path):
    arguments, _ = source_data
    annotation_bytes = arguments["annotation"].read_bytes()
    archive_bytes = arguments["archive"].read_bytes()
    report = binder.bind_sources(**arguments)
    output = arguments["output_dir"]
    manifest_path = output / binder.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "passed"
    assert manifest["canonical_source_revision"] == binder.SOURCE_REVISION
    assert manifest["canonical_archive"]["full_archive_sha256_recomputed_in_this_audit"] is True
    assert manifest["canonical_archive"]["sha256_from_prior_verified_manifest"] == arguments["archive_sha256"]
    assert [item["video_path"] for item in manifest["canonical_test_hdf5_bindings"]] == [
        "egodex/synthetic/0.mp4", "egodex/synthetic/1.mp4"]
    for index, row in enumerate(manifest["canonical_test_hdf5_bindings"]):
        local = arguments["source_root"] / "synthetic" / f"{index}.hdf5"
        assert row["hdf5_sha256"] == sha(local)
        assert row["hdf5_bytes"] == local.stat().st_size
    assert report["manifest_sha256"] == sha(manifest_path)
    assert report["full_archive_sha256_recomputed"] is True
    assert report["actual_hdf5_members_verified"] == 2
    assert report["original_inputs_metadata_unchanged"] is True
    for name in ("archive_extraction_performed", "derived_labels_generated", "model_results_used",
                 "model_inference_performed", "automatic_publication", "redistribution_authorization_claim"):
        assert report[name] is False
    assert arguments["annotation"].read_bytes() == annotation_bytes
    assert arguments["archive"].read_bytes() == archive_bytes
    assert output.stat().st_mode & 0o777 == 0o700
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert (output / binder.REPORT_NAME).stat().st_mode & 0o777 == 0o600
    assert str(tmp_path) not in manifest_path.read_text()
    built = build_reference_v2(arguments["annotation"], arguments["annotation_sha256"],
                               arguments["source_root"], manifest_path, report["manifest_sha256"],
                               tmp_path / "synthetic-reference", expected_rows=2)
    assert built["rows"] == 2 and built["full_frame_coverage"]["valid_frames"] == 6


@pytest.mark.parametrize("field", ["annotation_sha256", "archive_sha256"])
def test_wrong_full_input_hash_fails(source_data, field):
    arguments, _ = source_data
    arguments[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        binder.bind_sources(**arguments)
    assert not (arguments["output_dir"] / binder.MANIFEST_NAME).exists()


@pytest.mark.parametrize("name", ["../outside", "egodex/../outside", "egodex//extra", "/egodex/extra",
                                  "other/extra", "egodex\\extra", "egodex/./extra"])
def test_unsafe_central_entry_rejected_before_local_reads(source_data, name):
    arguments, entries = source_data
    rewrite_archive(arguments, entries + [zip_entry(name, b"unused")])
    with pytest.raises(ValueError, match="[Uu]nsafe|[Nn]oncanonical"):
        binder.bind_sources(**arguments)


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600,
                                  stat.S_IFCHR | 0o600, stat.S_IFSOCK | 0o600])
def test_symlink_and_special_central_entries_rejected(source_data, mode):
    arguments, entries = source_data
    rewrite_archive(arguments, entries + [zip_entry("egodex/extra", b"unused", mode)])
    with pytest.raises(ValueError, match="Symlink or special"):
        binder.bind_sources(**arguments)


def test_duplicate_entry_rejected(source_data):
    arguments, entries = source_data
    with pytest.warns(UserWarning, match="Duplicate name"):
        rewrite_archive(arguments, entries + [entries[1]])
    with pytest.raises(ValueError, match="Duplicate archive entry"):
        binder.bind_sources(**arguments)


@pytest.mark.parametrize("name,mode", [("egodex/bad", stat.S_IFDIR | 0o700),
                                     ("egodex/bad/", stat.S_IFREG | 0o600)])
def test_archive_type_suffix_mismatch_rejected(source_data, name, mode):
    arguments, entries = source_data
    rewrite_archive(arguments, entries + [zip_entry(name, b"", mode)])
    with pytest.raises(ValueError, match="type/name mismatch"):
        binder.bind_sources(**arguments)


def test_missing_hdf_member_rejected(source_data):
    arguments, entries = source_data
    rewrite_archive(arguments, entries[:-1])
    with pytest.raises(ValueError, match="HDF5 membership"):
        binder.bind_sources(**arguments)


def test_extra_hdf_member_rejected(source_data):
    arguments, entries = source_data
    rewrite_archive(arguments, entries + [zip_entry("egodex/synthetic/extra.hdf5", b"extra")])
    with pytest.raises(ValueError, match="HDF5 membership"):
        binder.bind_sources(**arguments)


def test_missing_paired_video_rejected(source_data):
    arguments, entries = source_data
    rewrite_archive(arguments, [item for item in entries if item[0].filename != "egodex/synthetic/0.mp4"])
    with pytest.raises(ValueError, match="Paired canonical video"):
        binder.bind_sources(**arguments)


def test_same_size_different_local_hdf_bytes_rejected(source_data):
    arguments, _ = source_data
    local = arguments["source_root"] / "synthetic/0.hdf5"
    contents = local.read_bytes()
    local.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    with pytest.raises(ValueError, match="ZIP HDF5 member differs"):
        binder.bind_sources(**arguments)


def test_local_size_mismatch_rejected(source_data):
    arguments, _ = source_data
    local = arguments["source_root"] / "synthetic/0.hdf5"
    local.write_bytes(local.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="sizes differ"):
        binder.bind_sources(**arguments)


def test_local_symlink_rejected(source_data):
    arguments, _ = source_data
    local = arguments["source_root"] / "synthetic/0.hdf5"
    moved = local.with_suffix(".original")
    local.rename(moved)
    local.symlink_to(moved.name)
    with pytest.raises(ValueError, match="symlinked"):
        binder.bind_sources(**arguments)


def test_annotation_change_during_zip_read_is_detected(source_data, monkeypatch):
    arguments, _ = source_data
    original = binder._stream_sha
    count = 0

    def changed(handle):
        nonlocal count
        result = original(handle)
        if count == 0:
            path = arguments["annotation"]
            current = path.stat()
            os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000_000))
        count += 1
        return result

    monkeypatch.setattr(binder, "_stream_sha", changed)
    with pytest.raises(ValueError, match="original input changed"):
        binder.bind_sources(**arguments)
    assert not (arguments["output_dir"] / binder.MANIFEST_NAME).exists()


def test_duplicate_annotation_paths_rejected(source_data):
    arguments, _ = source_data
    table = pq.read_table(arguments["annotation"])
    rows = table.to_pylist()
    rows[1]["video_path"] = rows[0]["video_path"]
    pq.write_table(pa.Table.from_pylist(rows), arguments["annotation"])
    arguments["annotation_sha256"] = sha(arguments["annotation"])
    with pytest.raises(ValueError, match="unique strings"):
        binder.bind_sources(**arguments)


def test_default_requires_full_count(source_data):
    arguments, _ = source_data
    arguments.pop("expected_rows")
    with pytest.raises(ValueError, match="row count"):
        binder.bind_sources(**arguments)


def test_full_scope_library_also_rejects_unpublished_hashes(source_data, monkeypatch):
    arguments, _ = source_data
    monkeypatch.setattr(binder, "EXPECTED_ROWS", 2)
    with pytest.raises(ValueError, match="Full canonical binding requires"):
        binder.bind_sources(**arguments)


def test_existing_outputs_refused_and_umask_restored(source_data):
    arguments, _ = source_data
    initial = os.umask(0o022)
    try:
        report = binder.bind_sources(**arguments)
        assert os.umask(0o022) == 0o022
        with pytest.raises(FileExistsError):
            binder.bind_sources(**arguments)
        assert sha(arguments["output_dir"] / binder.MANIFEST_NAME) == report["manifest_sha256"]
    finally:
        os.umask(initial)


def test_cli_forbids_noncanonical_hashes_and_count_override(monkeypatch, tmp_path):
    monkeypatch.setattr(binder, "bind_sources", lambda **kwargs: pytest.fail("must not run binding"))
    args = ["--annotation", str(tmp_path / "annotation"), "--annotation-sha256", "0" * 64,
            "--archive", str(tmp_path / "archive"), "--archive-sha256", "1" * 64,
            "--source-root", str(tmp_path), "--output-dir", str(tmp_path / "output")]
    with pytest.raises(ValueError, match="published canonical"):
        binder.main(args)
    with pytest.raises(SystemExit):
        binder.main(args + ["--expected-rows", "2"])


def test_cli_uses_fixed_canonical_scope(monkeypatch, tmp_path, capsys):
    seen = []

    def fake_bind(**kwargs):
        seen.append(kwargs)
        return {"status": "synthetic-cli-test", "rows": 3243,
                "manifest_name": binder.MANIFEST_NAME, "manifest_sha256": "0" * 64}

    monkeypatch.setattr(binder, "bind_sources", fake_bind)
    args = ["--annotation", str(tmp_path / "annotation"), "--annotation-sha256", binder.ANNOTATION_SHA256,
            "--archive", str(tmp_path / "archive"), "--archive-sha256", binder.ARCHIVE_SHA256,
            "--source-root", str(tmp_path), "--output-dir", str(tmp_path / "output")]
    assert binder.main(args) == 0
    assert len(seen) == 1 and "expected_rows" not in seen[0]
    assert json.loads(capsys.readouterr().out)["rows"] == 3243
