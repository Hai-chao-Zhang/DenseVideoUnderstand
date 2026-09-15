import hashlib
import json
from pathlib import Path

import pytest

from lmms_eval.tasks.densevideo import utils


class Rows:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def select(self, indices):
        return list(indices)


def test_preview_is_fixed_even_with_legacy_full_override(monkeypatch):
    monkeypatch.setenv("DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES", "0")
    monkeypatch.setattr(utils, "validate_highmotion_release", lambda dataset: dataset)
    assert utils.highmotion_preview_1000(Rows(3243)) == list(range(1000))


def test_preview_rejects_incomplete_source():
    with pytest.raises(ValueError, match="3243"):
        utils.highmotion_preview_1000(Rows(999))


def test_ordered_highmotion_content_guard(monkeypatch):
    rows = [{"video_path": f"egodex/action/{i}.mp4", "qid": str(i), "question": "Track.",
             "answer": "middle", "frame_count": 1} for i in range(3243)]
    projected = [[r["video_path"], r["qid"], r["question"], r["answer"], r["frame_count"]] for r in rows]
    checksum = hashlib.sha256(json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(utils, "HIGHMOTION_CONTENT_SHA256", checksum)
    assert utils.validate_highmotion_release(rows) is rows
    with pytest.raises(ValueError, match="content/order"):
        utils.validate_highmotion_release(list(reversed(rows)))
    rows[0]["answer"] = "top"
    with pytest.raises(ValueError, match="content/order"):
        utils.validate_highmotion_release(rows)


def test_missing_video_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("DENSEVIDEO_DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="DENSEVIDEO_DATA_ROOT"):
        utils.lpm_doc_to_visual({"video_path": "egodex/release-test-action/missing.mp4"})


def test_highmotion_never_uses_colliding_flat_basename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("DENSEVIDEO_DATA_ROOT", str(tmp_path))
    flat = tmp_path / "DenseVideoEvaluation" / "videos" / "collision-0.mp4"
    flat.parent.mkdir(parents=True)
    flat.touch()
    with pytest.raises(FileNotFoundError):
        utils._resolve_lpm_video_path("egodex/action-a/collision-0.mp4")
    correct = tmp_path / "egodex" / "action-a" / "collision-0.mp4"
    correct.parent.mkdir(parents=True)
    correct.touch()
    assert Path(utils._resolve_lpm_video_path("egodex/action-a/collision-0.mp4")).resolve() == correct


def test_educational_archive_layout_resolves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DENSEVIDEO_DATA_ROOT", str(tmp_path))
    video = tmp_path / "DenseVideoEvaluation" / "videos" / "release-test-video.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    assert utils._resolve_lpm_video_path("DenseVideo-LPM/videos/release-test-video.mp4") == str(video)
