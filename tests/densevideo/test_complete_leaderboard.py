import csv
import hashlib
import io
import json
import math
import shutil
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tools.densevideo import build_complete_leaderboard as complete
from tools.densevideo.release_resources import resolve_release_bundle

BUNDLE = resolve_release_bundle(release="2026-09-14")
HISTORICAL = resolve_release_bundle(release="2026-08-20")
OUTPUT_HASHES = {
    "leaderboard.html": "cb7a5348cb091e4a1944a03dcf50121e55bb8dd301dcf201098a7e1441b42d7b",
    "data/leaderboard-complete.csv": "68def903c24ca4fd3aac9a62ae7edb0c672b123af02e6555b9accf5aa4db06a2",
    "data/public-audit.js": "d1084a00b92e59e66e8114efa9c23b81f2795b4e0aa5e70347c7c9a63e681ccd",
    "data/public-audit.json": "1cb83cf3a0add7c3c3894ef1a05759ce5e1e6e012ce56e25ddb5fe267efc87e5",
    "data/highmotion-audit.json": "f2358a0ae2f61ae15d0e03436c640fed128b26cf591db3a9e97e668aa562683d",
    "data/leaderboard.js": "c88492ec535a3bc4f47fe9e91f66bc050a857f757cd2c478db598e889d9dc1b1",
}


def source_objects():
    return (
        json.loads((BUNDLE / "manifest.json").read_text()),
        json.loads((HISTORICAL / "provenance.json").read_text()),
        json.loads((BUNDLE / "comparison_telemetry.json").read_text()),
        json.loads((BUNDLE / "highmotion-audit.json").read_text()),
    )


def test_complete_offline_golden_bytes_and_coverage(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline generator attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    monkeypatch.setattr("tools.densevideo.rebuild_published_leaderboard.urlopen", no_network)
    outputs = complete.build_outputs()
    assert {name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in outputs.items()} == OUTPUT_HASHES
    rows = list(csv.DictReader(io.StringIO(outputs["data/leaderboard-complete.csv"])))
    assert len(rows) == 41
    main = [r for r in rows if r["cohort"] != "educational_grt_controls"]
    assert len({(r["cohort"], r["method"]) for r in main}) == 29
    assert {r["cohort"] for r in rows} == {
        "educational_published", "educational_grt_controls",
    }
    audit = json.loads(outputs["data/public-audit.json"])
    evidence = json.loads(outputs["data/highmotion-audit.json"])
    assert len(evidence["rows"]) == 27
    assert audit["highmotion_additional"] == []
    assert audit["highmotion_release_status"] == "held_target_reference_consistency_review"
    assert audit["highmotion_hold_date"] == "2026-09-14"
    assert audit["highmotion_hold_reason"] == complete.HIGHMOTION_HOLD_REASON
    assert audit["highmotion_release_eligible_rows"] == 0
    assert audit["highmotion_historical_protocol_screened_candidates"] == 18
    assert sum(row["rank_eligible"] for row in evidence["rows"]) == 18
    assert outputs["data/highmotion-audit.json"].encode() == (BUNDLE / "highmotion-audit.json").read_bytes()
    assert hashlib.sha256((BUNDLE / "manifest.json").read_bytes()).hexdigest() == complete.MANIFEST_SHA256
    assert "grt_llava_ov_0_5b" not in outputs["leaderboard.html"]
    assert "highmotion_historical_unaligned" not in outputs["data/leaderboard-complete.csv"]
    assert outputs["data/leaderboard.js"].encode() == (HISTORICAL / "leaderboard.js").read_bytes()


def test_current_hold_preserves_all_29_educational_rows_exactly():
    frozen, audit, current_highmotion, _, _ = complete.load_data()
    historical = json.loads((HISTORICAL / "leaderboard.js").read_text().split(
        "window.DIVE_LEADERBOARD = ", 1)[1].strip().removesuffix(";"))
    assert frozen == historical
    assert current_highmotion == []
    outputs = complete.build_outputs()
    published = [row for row in csv.DictReader(io.StringIO(outputs["data/leaderboard-complete.csv"]))
                 if row["cohort"] == "educational_published"]
    assert len(published) == len(historical["tracks"]["lpm"]) == 29
    for actual, expected in zip(published, historical["tracks"]["lpm"]):
        for key, value in expected.items():
            assert actual[key] == ("" if value is None else str(value))
    assert len(audit["families"]) == 3
    assert all(len(family["methods"]) == 4 for family in audit["families"])


@pytest.mark.parametrize("target", ["rows", "payload", "status", "eligibility"])
def test_renderer_rejects_stale_or_forged_highmotion_release_payload(target):
    frozen, audit, aligned, _, _ = complete.load_data()
    if target == "rows":
        aligned.append({"method": "forged", "grid_acc": 1.0})
    elif target == "payload":
        audit["highmotion_additional"] = [{"method": "forged", "grid_acc": 1.0}]
    elif target == "status":
        audit["highmotion_release_status"] = "released"
    else:
        audit["highmotion_release_eligible_rows"] = 18
    with pytest.raises(ValueError, match="release hold forbids"):
        complete.render_outputs(frozen, audit, aligned)


def test_historical_protocol_candidates_remain_verified_not_release_eligible():
    objects = source_objects()
    complete.validate_contract(*objects)
    manifest, _, _, evidence = objects
    assert manifest["counts"]["csv_rows"] == 59  # Immutable pre-hold historical contract.
    assert len(manifest["eligible_highmotion_methods"]) == 18
    assert [row["method"] for row in evidence["rows"] if row["rank_eligible"]] == manifest["eligible_highmotion_methods"]
    current = json.loads(complete.build_outputs()["data/public-audit.json"])
    assert current["highmotion_release_eligible_rows"] == 0
    assert not current["highmotion_additional"]


def test_all_12_quality_values_are_bound_to_7608_numeric_records():
    outputs = complete.build_outputs()
    audit = json.loads(outputs["data/public-audit.json"])
    provenance = json.loads((HISTORICAL / "provenance.json").read_text())
    count = 0
    for family, visible in zip(provenance["families"], audit["families"]):
        with (HISTORICAL / family["numeric_file"]).open() as stream:
            numeric = list(csv.DictReader(stream))
        for source, displayed in zip(family["methods"], visible["methods"]):
            samples = [r for r in numeric if int(r["method_index"]) == source["method_index"]]
            assert len(samples) == 634
            assert {int(r["doc_id"]) for r in samples} == set(range(634))
            count += len(samples)
            for metric in ("open_mos", "token_f1"):
                mean = math.fsum(float(r[metric]) for r in samples) / 634
                assert displayed[metric] == source[metric]
                assert math.isclose(displayed[metric], mean, rel_tol=1e-12, abs_tol=1e-14)
    assert count == 7608


class ViewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local_links = []
        self.scripts = []
        self.external_styles = []
        self.rows_per_table = []
        self.in_body = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            href = attrs.get("href", "")
            if href and not href.startswith(("https://", "#")):
                self.local_links.append(href)
        if tag == "script":
            self.scripts.append(attrs)
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.external_styles.append(attrs)
        if tag == "tbody":
            self.rows_per_table.append(0)
            self.in_body = True
        if tag == "tr" and self.in_body:
            self.rows_per_table[-1] += 1

    def handle_endtag(self, tag):
        if tag == "tbody":
            self.in_body = False


def test_standalone_html_needs_no_missing_assets_or_javascript():
    outputs = complete.build_outputs()
    page = outputs["leaderboard.html"]
    parser = ViewParser()
    parser.feed(page)
    assert parser.rows_per_table == [29, 12]
    assert not parser.scripts and not parser.external_styles
    assert "<style>" in page
    assert {urlsplit(link).path for link in parser.local_links} <= set(outputs)
    for link in parser.local_links:
        parsed = urlsplit(link)
        if parsed.path in {"data/leaderboard-complete.csv", "data/public-audit.json"}:
            assert parsed.query == "v=20260914-target-hold"
        else:
            assert not parsed.query  # Immutable historical artifact links are unchanged.
    assert "0.10125" not in page
    assert "not a fresh GPU rerun" in page
    assert "target/reference consistency audit" in page
    assert "41 CSV records" in page
    assert "not a finding about all 3,243 items or GRT performance" in page
    section = page.split('<h2 id="highmotion-aligned">', 1)[1].split('<h2 id="grt-controls">', 1)[0]
    assert "<table" not in section and "<tbody" not in section
    evidence = json.loads(outputs["data/highmotion-audit.json"])
    assert all(row["method"] not in section for row in evidence["rows"])
    for metric in ("grid_acc", "grid_ade", "grid_fde", "transition_acc", "protocol_status"):
        assert metric not in outputs["data/leaderboard-complete.csv"].splitlines()[0]
        assert metric not in outputs["data/public-audit.json"]


@pytest.mark.parametrize("filename", [
    "manifest.json", "comparison_telemetry.json", "highmotion-audit.json",
])
def test_tampered_complete_source_bytes_are_rejected(tmp_path, filename):
    bundle = tmp_path / "changed"
    shutil.copytree(BUNDLE, bundle)
    with (bundle / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        complete.build_outputs(bundle)


@pytest.mark.parametrize("filename", [
    "provenance.json", "route31_numeric.csv", "qwen3_numeric.csv", "qwen7_numeric.csv",
])
def test_tampered_educational_numeric_sources_are_rejected(tmp_path, filename):
    historical = tmp_path / "historical"
    shutil.copytree(HISTORICAL, historical)
    if filename == "provenance.json":
        data = json.loads((historical / filename).read_text())
        data["families"][0]["methods"][-1]["open_mos"] += 0.25
        (historical / filename).write_text(json.dumps(data))
    else:
        with (historical / filename).open("ab") as stream:
            stream.write(b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        complete.build_outputs(historical_bundle=historical)


@pytest.mark.parametrize("case", [
    "duplicate_family", "missing_control", "duplicate_control", "changed_role",
    "candidate_throughput", "candidate_source_hash", "nonfinite_telemetry",
    "empty_highmotion", "duplicate_highmotion", "promoted_legacy", "wrong_identity",
    "wrong_sampling", "wrong_target_count", "wrong_metric", "nonfinite_metric",
    "evidence_schema",
])
def test_semantic_contract_rejects_known_failure_modes(case):
    manifest, provenance, telemetry, evidence = map(deepcopy, source_objects())
    candidate = next(r for r in telemetry["rows"] if r["method"].startswith("grt_qwen2_5_vl_3b"))
    aligned = next(r for r in evidence["rows"] if r["rank_eligible"])
    if case == "duplicate_family":
        provenance["families"][0] = deepcopy(provenance["families"][1])
    elif case == "missing_control":
        provenance["families"][0]["methods"].pop(1)
    elif case == "duplicate_control":
        provenance["families"][0]["methods"][1] = deepcopy(provenance["families"][0]["methods"][2])
    elif case == "changed_role":
        provenance["families"][0]["methods"][1]["role"] = "candidate"
    elif case == "candidate_throughput":
        candidate["throughput_fps"] = 100
    elif case == "candidate_source_hash":
        candidate["telemetry_summary_sha256"] = "0" * 64
    elif case == "nonfinite_telemetry":
        candidate["mean_wall_time_s"] = float("inf")
    elif case == "empty_highmotion":
        evidence["rows"] = []
    elif case == "duplicate_highmotion":
        evidence["rows"][0] = deepcopy(evidence["rows"][1])
    elif case == "promoted_legacy":
        evidence["rows"][0]["rank_eligible"] = True
        evidence["rows"][0]["protocol_status"] = "aligned_preview"
    elif case == "wrong_identity":
        aligned["ordered_identity_sha256"] = "0" * 64
    elif case == "wrong_sampling":
        aligned["input_sampling_policy"] = "segment_midpoints"
    elif case == "wrong_target_count":
        aligned["target_match"] = 999
    elif case == "wrong_metric":
        aligned["metrics"]["grid_acc"] += 0.1
    elif case == "nonfinite_metric":
        aligned["metrics"]["grid_acc"] = float("nan")
    elif case == "evidence_schema":
        evidence["schema_version"] = 2
    with pytest.raises(ValueError):
        complete.validate_contract(manifest, provenance, telemetry, evidence)


def test_default_generation_is_offline_outside_cwd_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    complete.main(["--verify-only"])
    checked = json.loads(capsys.readouterr().out)
    assert checked["leaderboard_rows"] == 29
    assert checked["highmotion_rows"] == 0
    assert checked["highmotion_excluded_rows"] == 27
    assert checked["csv_rows"] == 41
    assert not list(tmp_path.iterdir())
    output = tmp_path / "standalone"
    complete.main(["--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report["output_sha256"] == OUTPUT_HASHES
    for name, digest in OUTPUT_HASHES.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(SystemExit) as exc:
        complete.main(["--output", str(output)])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        complete.main(["--verify-only", "--output", str(output)])
    assert exc.value.code == 2


def test_failed_verification_writes_no_output(tmp_path):
    changed = tmp_path / "changed"
    shutil.copytree(BUNDLE, changed)
    (changed / "highmotion-audit.json").unlink()
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as exc:
        complete.main(["--bundle", str(changed), "--output", str(output)])
    assert exc.value.code == 1
    assert not output.exists()


def test_second_artifact_partial_write_failure_removes_only_owned_paths(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preexisting data")
    output = existing / "new-parent" / "standalone"
    original_open = Path.open
    count = 0

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def write(self, value):
            self.stream.write(value[:12])
            self.stream.flush()
            raise OSError("injected second-artifact write failure")

    def injected_open(path, mode="r", *args, **kwargs):
        nonlocal count
        stream = original_open(path, mode, *args, **kwargs)
        if mode == "x":
            count += 1
            if count == 2:
                return FailingWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", injected_open)
    with pytest.raises(SystemExit) as exc:
        complete.main(["--output", str(output)])
    assert exc.value.code == 1 and count == 2
    assert not output.exists()
    assert not output.parent.exists()
    assert sentinel.read_text() == "preexisting data"
    assert list(existing.iterdir()) == [sentinel]


def test_concurrent_file_is_not_overwritten_or_removed_on_cleanup(tmp_path, monkeypatch, capsys):
    output = tmp_path / "standalone"
    concurrent = output / "data/leaderboard-complete.csv"
    original_open = Path.open

    def injected_open(path, mode="r", *args, **kwargs):
        if path == concurrent and mode == "x":
            with original_open(path, "w") as stream:
                stream.write("another writer's data")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected_open)
    with pytest.raises(SystemExit) as exc:
        complete.main(["--output", str(output)])
    assert exc.value.code == 1
    assert "cleanup was incomplete" in capsys.readouterr().err
    assert concurrent.read_text() == "another writer's data"
    assert not (output / "leaderboard.html").exists()


def test_preexisting_output_is_untouched_by_cleanup(tmp_path):
    output = tmp_path / "standalone"
    output.mkdir()
    sentinel = output / "leaderboard.html"
    sentinel.write_text("existing output")
    with pytest.raises(SystemExit) as exc:
        complete.main(["--output", str(output)])
    assert exc.value.code == 1
    assert sentinel.read_text() == "existing output"
    assert list(output.iterdir()) == [sentinel]
