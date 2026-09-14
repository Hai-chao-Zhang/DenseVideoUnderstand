import json
import shutil
from pathlib import Path

import pytest

from tools.densevideo.rebuild_published_leaderboard import main, verify_bundle

BUNDLE = Path(__file__).resolve().parents[2] / "release" / "2026-08-20"


def test_numerical_audit_and_exact_markdown():
    report, markdown = verify_bundle(BUNDLE)
    assert report["leaderboard_rows"] == 32
    assert len(report["families"]) == 3
    assert all(f["point_gates"] == "pass" for f in report["families"])
    assert markdown == (BUNDLE / "leaderboard.md").read_bytes()


@pytest.mark.parametrize("filename", ["leaderboard.js", "leaderboard.csv", "provenance.json",
                                     "route31_numeric.csv", "qwen3_numeric.csv", "qwen7_numeric.csv",
                                     "sample_identities.csv", "highmotion_numeric.csv", "leaderboard.md"])
def test_tampered_artifact_rejected(tmp_path, filename):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    with (bundle / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError):
        verify_bundle(bundle)


def test_export_no_overwrite(tmp_path, capsys):
    output = tmp_path / "rebuilt"
    main(["--bundle", str(BUNDLE), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    for filename in ("leaderboard.md", "leaderboard.js", "leaderboard.csv"):
        assert (output / filename).read_bytes() == (BUNDLE / filename).read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--bundle", str(BUNDLE), "--output", str(output)])
    assert exc.value.code == 1


def test_verify_only_does_not_write(tmp_path):
    main(["--bundle", str(BUNDLE), "--verify-only"])
    assert not list(tmp_path.iterdir())


def test_conflicting_options_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--bundle", str(BUNDLE), "--verify-only", "--output", str(tmp_path)])
    assert exc.value.code == 2
