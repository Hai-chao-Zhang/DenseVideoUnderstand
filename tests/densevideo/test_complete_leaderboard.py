import csv
import hashlib
import io
import json
import math
import shutil
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tools.densevideo import build_complete_leaderboard as complete
from tools.densevideo.release_resources import resolve_release_bundle

BUNDLE = resolve_release_bundle(release="2026-09-14")
HISTORICAL = resolve_release_bundle(release="2026-08-20")
OUTPUT_HASHES = {
    "leaderboard.html": "1d0ae49af95d005522dddb6ce66e499de459ea55af4ca29b9755ff7a3608b4cd",
    "data/leaderboard-complete.csv": "28b52958ab58761c3124f8b74ceb2dd64fa832f5d6551a50b19f591f7167b105",
    "data/public-audit.js": "78ec8821ad125a72d413bf08f8072d5a28e5d2d4948a0e74dc0b183db3f12a3f",
    "data/public-audit.json": "19a2756517190aace4fcd65e658306fc78ace881512009fc76f99c40a91182a7",
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
    assert len(rows) == 59 and len(rows[0]) == 32
    main = [r for r in rows if r["cohort"] != "educational_grt_controls"]
    assert len({(r["cohort"], r["method"]) for r in main}) == 47
    assert {r["cohort"] for r in rows} == {
        "educational_published", "highmotion_aligned_preview1000", "educational_grt_controls",
    }
    audit = json.loads(outputs["data/public-audit.json"])
    evidence = json.loads(outputs["data/highmotion-audit.json"])
    assert len(evidence["rows"]) == 27
    assert len(audit["highmotion_additional"]) == 18
    assert {r["method"] for r in audit["highmotion_additional"]} == {
        r["method"] for r in evidence["rows"] if r["rank_eligible"]
    }
    assert [r["rank"] for r in audit["highmotion_additional"]] == list(range(1, 19))
    assert all(r["samples"] == 1000 for r in audit["highmotion_additional"])
    assert "grt_llava_ov_0_5b" not in outputs["leaderboard.html"]
    assert "highmotion_historical_unaligned" not in outputs["data/leaderboard-complete.csv"]
    assert outputs["data/leaderboard.js"].encode() == (HISTORICAL / "leaderboard.js").read_bytes()


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
    assert parser.rows_per_table == [29, 18, 12]
    assert not parser.scripts and not parser.external_styles
    assert "<style>" in page
    assert set(parser.local_links) <= set(outputs)
    assert "0.10125" not in page
    assert "not a fresh GPU rerun" in page
    assert "not a new GPU replay" in page
    assert "Model revisions were not fully pinned" in page


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
    assert json.loads(capsys.readouterr().out)["leaderboard_rows"] == 47
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
