"""Synthetic forwarding contracts; never load actual High-Motion results."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


@pytest.fixture
def builder(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "synthetic_v2_audit_builder", ROOT / "scripts/build_audit_page.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("DIVE_MINIMAL_ROOT", raising=False)
    return module


def mock_canonical(builder, monkeypatch, *, path="synthetic-generator.py"):
    outputs = {"leaderboard.html": "SYNTHETIC ONLY\n"}
    canonical = SimpleNamespace(__file__=path, build_outputs=Mock(return_value=outputs))
    importer = Mock(return_value=canonical)
    monkeypatch.setattr(builder, "importlib", SimpleNamespace(import_module=importer))
    return canonical, importer, outputs


def test_default_calls_historical_generator_without_new_keywords(builder, monkeypatch):
    canonical, importer, outputs = mock_canonical(builder, monkeypatch)
    assert builder.build_outputs() is outputs
    importer.assert_called_once_with("tools.densevideo.build_complete_leaderboard")
    canonical.build_outputs.assert_called_once_with()


def test_explicit_pair_is_forwarded_unchanged_without_numeric_copy(builder, monkeypatch):
    canonical, _, outputs = mock_canonical(builder, monkeypatch)
    bundle = Path("synthetic-reviewed-bundle")
    assert builder.build_outputs(highmotion_v2_bundle=bundle,
                                 highmotion_v2_manifest_sha256=DIGEST) is outputs
    canonical.build_outputs.assert_called_once_with(
        highmotion_v2_bundle=bundle, highmotion_v2_manifest_sha256=DIGEST)


@pytest.mark.parametrize("kwargs", [
    {"highmotion_v2_bundle": Path("synthetic")},
    {"highmotion_v2_manifest_sha256": DIGEST},
])
def test_unpaired_api_fails_before_root_lookup_or_import(builder, monkeypatch, kwargs):
    _, importer, _ = mock_canonical(builder, monkeypatch)
    with pytest.raises(ValueError, match="supplied together"):
        builder.build_outputs("missing-minimal-root-must-not-be-read", **kwargs)
    importer.assert_not_called()


@pytest.mark.parametrize("arguments", [
    ["--highmotion-v2-bundle", "synthetic"],
    ["--highmotion-v2-manifest-sha256", DIGEST],
])
def test_unpaired_cli_fails_before_build_or_writes(builder, monkeypatch, tmp_path, arguments):
    build = Mock(side_effect=AssertionError("No build before paired arguments"))
    monkeypatch.setattr(builder, "build_outputs", build)
    monkeypatch.setattr(builder, "ROOT", tmp_path / "uncreated-output")
    with pytest.raises(SystemExit) as error:
        builder.main(arguments)
    assert error.value.code == 2
    build.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("with_v2", [False, True])
def test_check_forwards_only_explicit_pair_and_does_not_write(
    builder, monkeypatch, tmp_path, with_v2,
):
    target = tmp_path / "leaderboard.html"
    target.write_text("SYNTHETIC ONLY\n", encoding="utf-8")
    canonical, _, _ = mock_canonical(builder, monkeypatch)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    before = target.stat().st_mtime_ns
    arguments = ["--check"]
    if with_v2:
        arguments += ["--highmotion-v2-bundle", "synthetic",
                      "--highmotion-v2-manifest-sha256", DIGEST]
    builder.main(arguments)
    assert target.read_text(encoding="utf-8") == "SYNTHETIC ONLY\n"
    assert target.stat().st_mtime_ns == before
    assert list(tmp_path.iterdir()) == [target]
    if with_v2:
        canonical.build_outputs.assert_called_once_with(
            highmotion_v2_bundle=Path("synthetic"), highmotion_v2_manifest_sha256=DIGEST)
    else:
        canonical.build_outputs.assert_called_once_with()


@pytest.mark.parametrize("failure", [ValueError("synthetic hash mismatch"),
                                    TypeError("old generator does not support v2 arguments")])
def test_failed_canonical_verification_preserves_existing_outputs(
    builder, monkeypatch, tmp_path, failure,
):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("must remain unchanged", encoding="utf-8")
    canonical, _, _ = mock_canonical(builder, monkeypatch)
    canonical.build_outputs.side_effect = failure
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        builder.main(["--highmotion-v2-bundle", "synthetic",
                      "--highmotion-v2-manifest-sha256", DIGEST])
    assert error.value.code == 1
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "must remain unchanged"


def test_selected_minimal_root_still_requires_import_origin_binding(builder, monkeypatch, tmp_path):
    root = tmp_path / "source"
    module_file = root / "tools/densevideo/build_complete_leaderboard.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# Synthetic module placeholder\n", encoding="utf-8")
    canonical, _, _ = mock_canonical(builder, monkeypatch, path=str(tmp_path / "outside.py"))
    monkeypatch.setattr(builder, "sys", SimpleNamespace(path=[]))
    with pytest.raises(ValueError, match="does not belong"):
        builder.build_outputs(root, highmotion_v2_bundle=Path("synthetic"),
                              highmotion_v2_manifest_sha256=DIGEST)
    canonical.build_outputs.assert_not_called()


def test_check_detects_stale_output_without_replacing_it(builder, monkeypatch, tmp_path):
    target = tmp_path / "leaderboard.html"
    target.write_text("old synthetic output", encoding="utf-8")
    mock_canonical(builder, monkeypatch)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        builder.main(["--check", "--highmotion-v2-bundle", "synthetic",
                      "--highmotion-v2-manifest-sha256", DIGEST])
    assert error.value.code == 1
    assert target.read_text(encoding="utf-8") == "old synthetic output"
