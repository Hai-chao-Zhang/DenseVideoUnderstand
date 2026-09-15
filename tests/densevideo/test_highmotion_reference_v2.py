"""Synthetic CPU tests: never read canonical data, weights, or predictions."""

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.densevideo.build_highmotion_reference_v2 import (
    PARQUET_NAME,
    REPORT_NAME,
    build_reference_v2,
    main,
)
from tools.densevideo.highmotion_reference_v2 import MASK_POLICY, TARGET_JOINT, VERSION, project_reference


def scene(points):
    camera = np.repeat(np.eye(4)[None], len(points), axis=0)
    joint = camera.copy()
    joint[:, :3, 3] = points
    return np.eye(3), camera, joint, np.ones(len(points))


def project(points, width=9, height=9):
    return project_reference(*scene(points), width, height)


def test_all_nine_cells():
    result = project([[x, y, 1] for y in (1, 4, 7) for x in (1, 4, 7)])
    assert result["answer"] == ["topleft", "top", "topright", "left", "middle", "right",
                                "bottomleft", "bottom", "bottomright"]
    assert result["reference_valid"] == [True] * 9
    assert result["reference_invalid_reason"] == [""] * 9


def test_boundaries_are_half_open_without_rounding_or_clamping():
    result = project([[0, 0, 1], [3, 3, 1], [6, 6, 1], [2.99999, 2.99999, 1],
                      [9, 0, 1], [0, 9, 1], [-0.01, 1, 1], [1, -0.01, 1]])
    assert result["answer"] == ["topleft", "middle", "bottomright", "topleft"] + [None] * 4
    assert result["reference_invalid_reason"][-4:] == ["out_of_frame"] * 4


def test_inverse_camera_direction_and_no_axis_flip():
    intrinsic, camera, joint, confidence = scene([[5, 7, 1]])
    camera[0, :3, 3] = [1, 2, 0]
    result = project_reference(intrinsic, camera, joint, confidence, 9, 9)
    assert result["answer_traj"] == [[4.0, 5.0]]
    assert result["answer"] == ["middle"]


def test_no_center_sentinel_for_nonpositive_depth():
    result = project([[0, 0, -1], [0, 0, 0]])
    assert result["answer"] == [None, None]
    assert result["answer_traj"] == [None, None]
    assert result["reference_invalid_reason"] == ["nonpositive_depth"] * 2


def test_confidence_policy_and_reason_overlap():
    intrinsic, camera, joint, _ = scene([[1, 1, 1]] * 7 + [[20, 1, 1]])
    confidence = np.array([1, 0.00001, 0, -1, 1.01, np.nan, np.inf, 0])
    result = project_reference(intrinsic, camera, joint, confidence, 9, 9)
    assert result["reference_valid"] == [True, True] + [False] * 6
    assert result["reference_invalid_reason"][-1] == "invalid_confidence|out_of_frame"
    assert all(value is None for value in result["answer_traj"][2:])


def test_missing_confidence_is_unknown_not_visible():
    intrinsic, camera, joint, _ = scene([[1, 1, 1]])
    result = project_reference(intrinsic, camera, joint, None, 9, 9)
    assert result["reference_invalid_reason"] == ["missing_confidence"]
    assert result["answer"] == [None]


@pytest.mark.parametrize("position,name", [(0, "intrinsic"), (1, "camera"), (2, "joint")])
def test_missing_source_arrays_are_masked(position, name):
    values = list(scene([[1, 1, 1]]))
    values[position] = None
    result = project_reference(*values, 9, 9)
    assert result["reference_invalid_reason"] == [f"missing_{name}"]
    assert result["answer_traj"] == [None]


@pytest.mark.parametrize("position,name", [(0, "intrinsic"), (1, "camera"), (2, "joint")])
def test_nonfinite_source_arrays_are_masked(position, name):
    values = list(scene([[1, 1, 1]]))
    values[position].flat[0] = np.nan
    result = project_reference(*values, 9, 9)
    assert result["reference_invalid_reason"] == [f"nonfinite_{name}"]


def test_singular_camera_masks_only_its_frame():
    intrinsic, camera, joint, confidence = scene([[1, 1, 1], [1, 1, 1]])
    camera[0] = 0
    result = project_reference(intrinsic, camera, joint, confidence, 9, 9)
    assert result["reference_valid"] == [False, True]
    assert result["reference_invalid_reason"] == ["noninvertible_camera", ""]


def test_zero_projective_denominator_is_masked():
    intrinsic, camera, joint, confidence = scene([[1, 1, 1]])
    intrinsic[2] = 0
    result = project_reference(intrinsic, camera, joint, confidence, 9, 9)
    assert result["reference_invalid_reason"] == ["invalid_projection"]


@pytest.mark.parametrize("position,bad", [(0, np.eye(4)), (1, np.eye(4)), (2, np.zeros((2, 4, 4))),
                                         (3, np.ones((1, 1)))])
def test_malformed_shapes_raise(position, bad):
    values = list(scene([[1, 1, 1]]))
    values[position] = bad
    with pytest.raises(ValueError):
        project_reference(*values, 9, 9)


@pytest.mark.parametrize("width,height", [(True, 9), (9, False), (0, 9), (9, -1), (9.0, 9)])
def test_invalid_image_dimensions_raise(width, height):
    with pytest.raises(ValueError):
        project([[1, 1, 1]], width, height)


def test_both_transform_arrays_missing_cannot_invent_frame_count():
    with pytest.raises(ValueError):
        project_reference(np.eye(3), None, None, None, 9, 9)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def fixture_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    task = source / "synthetic"
    task.mkdir()
    rows, bindings = [], []
    for index in range(2):
        hdf = task / f"{index}.hdf5"
        intrinsic, camera, joint, confidence = scene([[1, 1, 1]] * 10)
        if index == 0:
            confidence[4] = 0  # Unsampled slot under endpoint-inclusive 8-of-10.
        else:
            confidence = None  # Retain a fully invalid row, never silently drop.
        with h5py.File(hdf, "w") as handle:
            handle.create_dataset("camera/intrinsic", data=intrinsic.astype(np.float32))
            handle.create_dataset("transforms/camera", data=camera.astype(np.float32))
            handle.create_dataset(f"transforms/{TARGET_JOINT}", data=joint.astype(np.float32))
            # Wrong-side source is deliberately very different.
            handle.create_dataset("transforms/leftIndexFingerMetacarpal", data=np.zeros_like(joint))
            if confidence is not None:
                handle.create_dataset(f"confidences/{TARGET_JOINT}", data=confidence.astype(np.float32))
        video = f"egodex/synthetic/{index}.mp4"
        rows.append({"video_path": video, "question": f"RIGHT-hand ring-finger base\n原问题 {index}",
                     "answer": '["middle"]', "answer_traj": "[[0.0,0.0]]", "qid": 0,
                     "frame_count": 10, "width": 9, "height": 9, "fps": 30.0,
                     "segment_start": 0.0, "segment_end": 1.0, "extra": {"keep": index}})
        bindings.append({"video_path": video, "hdf5_sha256": digest(hdf), "hdf5_bytes": hdf.stat().st_size})
    annotation = tmp_path / "legacy.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"legacy": b"preserved"})
    pq.write_table(table, annotation)
    manifest = tmp_path / "bindings.json"
    manifest.write_text(json.dumps({"status": "passed", "canonical_source_revision": "a" * 40,
                                    "canonical_archive": {
                                        "all_3243_hdf5_members_read_and_sha256_matched_to_original_test": True,
                                        "sha256_from_prior_verified_manifest": "b" * 64},
                                    "canonical_test_hdf5_bindings": bindings}), encoding="utf-8")
    return {"annotation": annotation, "annotation_sha256": digest(annotation), "source_root": source,
            "binding_manifest": manifest, "binding_manifest_sha256": digest(manifest),
            "output_dir": tmp_path / "new-output", "expected_rows": 2}


def test_builder_preserves_questions_and_all_nonanswer_fields(fixture_data):
    before = Path(fixture_data["annotation"]).read_bytes()
    report = build_reference_v2(**fixture_data)
    old = pq.read_table(fixture_data["annotation"])
    new = pq.read_table(fixture_data["output_dir"] / PARQUET_NAME)
    unchanged = [name for name in old.column_names if name not in ("answer", "answer_traj")]
    assert old.select(unchanged).equals(new.select(unchanged), check_metadata=True)
    assert Path(fixture_data["annotation"]).read_bytes() == before
    assert new.num_rows == 2 and new["qid"].to_pylist() == [0, 0]
    assert new["benchmark_version"].to_pylist() == [VERSION] * 2
    assert new["target_joint"].to_pylist() == [TARGET_JOINT] * 2
    assert new["reference_policy"].to_pylist() == [MASK_POLICY] * 2
    assert json.loads(new["answer"][0].as_py()) == ["topleft"] * 4 + [None] + ["topleft"] * 5
    assert json.loads(new["answer"][1].as_py()) == [None] * 10
    assert report["parquet_sha256"] == digest(fixture_data["output_dir"] / PARQUET_NAME)
    assert json.loads((fixture_data["output_dir"] / REPORT_NAME).read_text()) == report
    assert report["full_frame_coverage"]["valid_frames"] == 9
    assert report["full_frame_coverage"]["all_invalid_rows"] == 1
    assert report["uniform_eight_frame_coverage"]["frames"] == 16
    assert report["uniform_eight_frame_coverage"]["valid_frames"] == 8
    assert report["uniform_eight_frame_coverage"]["all_invalid_rows"] == 1
    assert not report["zip_members_reread_in_this_build"]
    assert not report["model_inference_performed"] and not report["model_results_used"]
    assert not report["performance_or_superiority_claim"] and not report["automatic_publication"]
    assert report["rows_removed"] == 0
    assert fixture_data["output_dir"].stat().st_mode & 0o777 == 0o700
    assert (fixture_data["output_dir"] / PARQUET_NAME).stat().st_mode & 0o777 == 0o600
    for question, sha in zip(old["question"].to_pylist(), new["legacy_question_sha256"].to_pylist()):
        assert hashlib.sha256(question.encode("utf-8")).hexdigest() == sha


def test_output_is_exclusive_and_never_overwritten(fixture_data):
    build_reference_v2(**fixture_data)
    old_sha = digest(fixture_data["output_dir"] / PARQUET_NAME)
    with pytest.raises(FileExistsError):
        build_reference_v2(**fixture_data)
    assert digest(fixture_data["output_dir"] / PARQUET_NAME) == old_sha


@pytest.mark.parametrize("field", ["annotation_sha256", "binding_manifest_sha256"])
def test_pinned_input_hash_mismatch_rejected(fixture_data, field):
    fixture_data[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_reference_v2(**fixture_data)
    assert not fixture_data["output_dir"].exists()


def rewrite_manifest(fixture_data, mutate):
    path = fixture_data["binding_manifest"]
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    fixture_data["binding_manifest_sha256"] = digest(path)


def test_ordered_membership_must_match(fixture_data):
    rewrite_manifest(fixture_data, lambda m: m["canonical_test_hdf5_bindings"].reverse())
    with pytest.raises(ValueError, match="ordered membership"):
        build_reference_v2(**fixture_data)


def test_manifest_requires_prior_actual_zip_member_binding(fixture_data):
    rewrite_manifest(fixture_data, lambda m: m["canonical_archive"].clear())
    with pytest.raises(ValueError, match="ZIP-member"):
        build_reference_v2(**fixture_data)


def test_actual_hdf5_hash_mismatch_fails_without_success_report(fixture_data):
    path = fixture_data["source_root"] / "synthetic/0.hdf5"
    with h5py.File(path, "r+") as source:
        source[f"transforms/{TARGET_JOINT}"][0, 0, 3] = 8
    with pytest.raises(ValueError, match="source_hdf5 SHA-256"):
        build_reference_v2(**fixture_data)
    assert not (fixture_data["output_dir"] / REPORT_NAME).exists()


def test_hdf_size_binding_is_checked(fixture_data):
    rewrite_manifest(fixture_data, lambda m: m["canonical_test_hdf5_bindings"][0].update(hdf5_bytes=1))
    with pytest.raises(ValueError, match="size differs"):
        build_reference_v2(**fixture_data)


def test_source_symlink_rejected(fixture_data):
    path = fixture_data["source_root"] / "synthetic/0.hdf5"
    moved = path.with_suffix(".stored")
    path.rename(moved)
    path.symlink_to(moved.name)
    with pytest.raises(ValueError, match="symlinked"):
        build_reference_v2(**fixture_data)


def test_source_path_traversal_rejected(fixture_data):
    old = pq.read_table(fixture_data["annotation"])
    paths = old["video_path"].to_pylist()
    paths[0] = "egodex/../escape.mp4"
    old = old.set_column(old.schema.get_field_index("video_path"), "video_path", pa.array(paths))
    pq.write_table(old, fixture_data["annotation"])
    fixture_data["annotation_sha256"] = digest(fixture_data["annotation"])
    rewrite_manifest(fixture_data, lambda m: m["canonical_test_hdf5_bindings"][0].update(video_path=paths[0]))
    with pytest.raises(ValueError, match="Unsafe"):
        build_reference_v2(**fixture_data)


def test_frame_count_mismatch_fails_and_umask_restored(fixture_data):
    old = pq.read_table(fixture_data["annotation"])
    old = old.set_column(old.schema.get_field_index("frame_count"), "frame_count", pa.array([11, 10]))
    pq.write_table(old, fixture_data["annotation"])
    fixture_data["annotation_sha256"] = digest(fixture_data["annotation"])
    initial = os.umask(0o022)
    try:
        with pytest.raises(ValueError, match="frame_count"):
            build_reference_v2(**fixture_data)
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(initial)


def test_default_count_is_full_3243_and_cli_has_no_reduced_flag(fixture_data):
    fixture_data.pop("expected_rows")
    with pytest.raises(ValueError, match="binding count"):
        build_reference_v2(**fixture_data)
    with pytest.raises(SystemExit):
        main(["--expected-rows", "2"])
