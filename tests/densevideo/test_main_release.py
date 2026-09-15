"""Main-branch integration contracts; no network or model inference."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_main_keeps_the_original_project_notice_and_illustration():
    expected = {
        "LICENSE-BSD": "c6b2d9138b0654f84081f074bc6a6f4b10230f3d0b12da579bc88a6e06a46fdc",
        "assets/DIVE.jpeg": "66f1d7d7b2e12f749402bde7f868fdd242ac37c0e1f43cbc07d313cc278185a4",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert "LICENSE-BSD" in (ROOT / "MANIFEST.in").read_text()
    metadata = (ROOT / "pyproject.toml").read_text()
    assert 'license = "MIT AND Apache-2.0 AND BSD-3-Clause"' in metadata
    assert 'license-files = ["LICENSE", "LICENSE-APACHE", "LICENSE-BSD"]' in metadata


def test_main_is_the_documented_checkout_and_ci_target():
    assert "branches: [main]" in (ROOT / ".github/workflows/release-checks.yml").read_text()
    assert "--branch main" in (ROOT / "README.md").read_text()
    for name in ("README.md", "pyproject.toml", "docs/COMPLETE_LEADERBOARD.md"):
        text = (ROOT / name).read_text()
        for retired in ("fix/highmotion-target-v2", "release/dive-bench-minimal"):
            assert f"/tree/{retired}" not in text
            assert f"/blob/{retired}" not in text
            assert f"--branch {retired}" not in text
