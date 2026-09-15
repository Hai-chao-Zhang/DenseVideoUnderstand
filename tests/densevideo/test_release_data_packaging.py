"""Synthetic release resource packaging; no benchmark data or GPU access."""

import gzip
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from setuptools import Distribution
from setuptools.errors import SetupError

from build_support import release_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-09-15"
NUMERIC_NAME = "numeric_reports.json.gz"
NUMERIC_BYTES = gzip.compress(b'{"fixture":"synthetic numeric evidence only"}\n', mtime=0)


@pytest.fixture
def staging(tmp_path, monkeypatch):
    project = tmp_path / "synthetic-project"
    release_root = project / "release"
    release_root.mkdir(parents=True)
    monkeypatch.setattr(release_data, "__file__", str(project / "build_support" / "release_data.py"))
    # Isolate the actual release-staging hook from setuptools' unrelated Python
    # package discovery. Its filesystem validation/copy/output logic runs intact.
    monkeypatch.setattr(release_data.build_py, "run", lambda self: None)
    monkeypatch.setattr(release_data.build_py, "get_outputs", lambda self, include_bytecode=True: [])
    command = release_data.BuildPyWithReleaseData(Distribution({"packages": []}))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "wheel-stage")
    destination = Path(command.build_lib) / "tools" / "densevideo" / "_release_data"
    return release_root, destination, command


def write_artifact(root, relative, content=NUMERIC_BYTES):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_exact_numeric_archive_is_staged_byte_identically_with_historical_files(staging):
    source, destination, command = staging
    expected = {
        f"{DATE}/{NUMERIC_NAME}": NUMERIC_BYTES,
        f"{DATE}/manifest.json": b'{"synthetic":true}\n',
        "2026-08-20/leaderboard.js": b"window.SYNTHETIC = {};\n",
        "2026-08-20/leaderboard.csv": b"method,score\nsynthetic,0\n",
        "2026-08-20/README.md": b"Synthetic historical release fixture.\n",
    }
    for relative, content in expected.items():
        write_artifact(source, relative, content)
    command.run()
    assert {Path(path).relative_to(destination).as_posix() for path in command.get_outputs()} == set(expected)
    for relative, content in expected.items():
        assert (destination / relative).read_bytes() == content


@pytest.mark.parametrize("name", [
    "predictions.json.gz", "numeric_reports.gz", "numeric_reports.json.GZ",
    "numeric_reports.JSON.gz", "numeric_reports.json.gz.tmp", "weights.bin.gz",
    "nested/numeric_reports.json.gz", ".numeric_reports.json.gz",
    ".private/numeric_reports.json.gz",
])
def test_other_compressed_names_or_locations_are_rejected(staging, name):
    source, destination, command = staging
    write_artifact(source, f"{DATE}/{name}")
    with pytest.raises(SetupError, match="unexpected release artifact"):
        command.run()
    assert not (destination / DATE / name).exists()


def test_numeric_archive_symlink_is_rejected(staging, tmp_path):
    source, destination, command = staging
    target = write_artifact(tmp_path, "outside-release.json.gz")
    bundle = source / DATE
    bundle.mkdir()
    (bundle / NUMERIC_NAME).symlink_to(target)
    with pytest.raises(SetupError, match="cannot contain symlinks"):
        command.run()
    assert not (destination / DATE / NUMERIC_NAME).exists()


@pytest.mark.parametrize("directory", ["2026-09-15-highmotion-v2", "highmotion-v2", "20260915"])
def test_release_directory_date_contract_is_not_broadened(staging, directory):
    source, destination, command = staging
    write_artifact(source, f"{directory}/{NUMERIC_NAME}")
    with pytest.raises(SetupError, match="unexpected release directory name"):
        command.run()
    assert not destination.exists()


def test_no_release_bundle_is_still_an_error(staging):
    _, _, command = staging
    with pytest.raises(SetupError, match="contains no dated bundles"):
        command.run()


def test_real_sdist_includes_only_the_dated_root_numeric_archive(tmp_path):
    """Run setuptools sdist using the actual manifest and synthetic files only."""
    project = tmp_path / "sdist-project"
    project.mkdir()
    (project / "MANIFEST.in").write_text(
        (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (project / "setup.py").write_text(
        "from setuptools import setup\n"
        "setup(name='synthetic_grt_release', version='0.0.0', packages=['fixture_pkg'])\n",
        encoding="utf-8",
    )
    write_artifact(project, "fixture_pkg/__init__.py", b"")
    write_artifact(project, "README.md", b"Synthetic packaging fixture.\n")
    expected = f"release/{DATE}/{NUMERIC_NAME}"
    write_artifact(project, expected)
    write_artifact(project, f"release/{DATE}/manifest.json", b'{"synthetic":true}\n')
    excluded = [
        f"release/{DATE}/predictions.json.gz", f"release/{DATE}/weights.bin.gz",
        f"release/{DATE}/numeric_reports.json.GZ", f"release/{DATE}/nested/{NUMERIC_NAME}",
        f"release/{DATE}/.private/{NUMERIC_NAME}", f"release/not-a-date/{NUMERIC_NAME}",
    ]
    for relative in excluded:
        write_artifact(project, relative)
    output = tmp_path / "sdist-output"
    subprocess.run([sys.executable, "setup.py", "sdist", "--dist-dir", str(output)],
                   cwd=project, check=True, capture_output=True, text=True)
    archives = list(output.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        members = {
            member.name.split("/", 1)[1]: member
            for member in archive.getmembers() if "/" in member.name
        }
        assert expected in members
        with archive.extractfile(members[expected]) as handle:
            assert handle.read() == NUMERIC_BYTES
        assert f"release/{DATE}/manifest.json" in members
        assert not set(excluded).intersection(members)
