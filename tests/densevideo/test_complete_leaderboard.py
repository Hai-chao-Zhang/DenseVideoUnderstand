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
    "leaderboard.html": "5237d1fc09a13c5ad85b9bbe56d84655c591aefff6b79bc6ea529130770a5423",
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


def test_main_links_and_legacy_permalink_preserve_original_html():
    base = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand"
    assert complete.V2_CODE_URL == f"{base}/tree/main"
    assert complete.V2_GUIDE_URL == f"{base}/blob/main/docs/HIGHMOTION_V2_REPRODUCTION.md"
    assert complete.CODE_URL == f"{base}/tree/9ee16af0d03d7f31e726b71f00e4586972afb062"
    page = complete.build_outputs(include_highmotion_v2=False)["leaderboard.html"]
    assert page.count(complete.CODE_URL) == 1
    # The historical HTML changes only its retired branch link, not any result.
    original = page.replace(complete.CODE_URL, f"{base}/tree/release/dive-bench-minimal")
    assert hashlib.sha256(original.encode()).hexdigest() == (
        "cb7a5348cb091e4a1944a03dcf50121e55bb8dd301dcf201098a7e1441b42d7b"
    )


def test_default_current_release_is_the_authenticated_v2_preview(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    complete.main(["--verify-only"])
    report = json.loads(capsys.readouterr().out)
    assert report["release"] == "2026-09-15"
    assert report["leaderboard_rows"] == 48 and report["highmotion_rows"] == 19
    assert report["comparison_rows"] == 12 and report["csv_rows"] == 60
    assert report["highmotion_v2_manifest_sha256"] == complete.HIGHMOTION_V2_MANIFEST_SHA256
    assert report["highmotion_grid_acc_outperform"] is True
    assert not list(tmp_path.iterdir())
    outputs = complete.build_outputs()
    audit = json.loads(outputs["data/public-audit.json"])
    corrected = audit["highmotion_v2"]
    assert corrected["release_integrity_verified"] is True
    assert corrected["scope"] == "first-1000-source-rows"
    comparison = corrected["comparison"]
    assert comparison["grt_metrics"]["grid_acc"] == 0.049503622587246277
    assert comparison["baseline_metrics"]["grid_acc"] == 0.04686825949892152
    assert comparison["metric_outperform"] == {
        "grid_acc": True, "grid_ade": True, "grid_fde": True,
        "grid_transition_acc": False, "token_f1": True,
    }
    assert all(row["samples"] == 1000 and row["valid_slots"] == 6015
               and row["records_with_scored_slots"] == 861 for row in corrected["rows"])
    parser = ViewParser()
    parser.feed(outputs["leaderboard.html"])
    assert parser.rows_per_table == [29, 19, 5, 12]
    assert not parser.scripts and not parser.external_styles
    records = list(csv.DictReader(io.StringIO(outputs["data/leaderboard-complete.csv"])))
    assert len(records) == 60
    assert sum(row["cohort"] == "highmotion_right_ring_v2_preview1000" for row in records) == 19
    assert "highmotion_historical_unaligned" not in outputs["data/leaderboard-complete.csv"]


def test_current_release_never_falls_back_to_legacy_when_v2_fails(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        raise ValueError("Corrupt current v2 release")

    monkeypatch.setattr(complete, "_load_highmotion_v2", reject)
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        complete.main(["--output", str(output)])
    assert error.value.code == 1 and not output.exists()
    assert complete.build_outputs(include_highmotion_v2=False)


def test_explicit_legacy_hold_refuses_corrected_bundle(monkeypatch):
    with pytest.raises(ValueError, match="cannot accept"):
        complete.build_outputs(highmotion_v2_bundle="unused", highmotion_v2_manifest_sha256="c" * 64,
                               include_highmotion_v2=False)
    with pytest.raises(SystemExit) as error:
        complete.main(["--legacy-reference-hold", "--highmotion-v2-bundle", "unused",
                       "--highmotion-v2-manifest-sha256", "c" * 64])
    assert error.value.code == 2


def test_legacy_hold_offline_golden_bytes_and_coverage(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline generator attempted network access")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    monkeypatch.setattr("tools.densevideo.rebuild_published_leaderboard.urlopen", no_network)
    outputs = complete.build_outputs(include_highmotion_v2=False)
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
    outputs = complete.build_outputs(include_highmotion_v2=False)
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


def test_legacy_generation_is_offline_outside_cwd_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    complete.main(["--legacy-reference-hold", "--verify-only"])
    checked = json.loads(capsys.readouterr().out)
    assert checked["leaderboard_rows"] == 29
    assert checked["highmotion_rows"] == 0
    assert checked["highmotion_excluded_rows"] == 27
    assert checked["csv_rows"] == 41
    assert not list(tmp_path.iterdir())
    output = tmp_path / "standalone"
    complete.main(["--legacy-reference-hold", "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report["output_sha256"] == OUTPUT_HASHES
    for name, digest in OUTPUT_HASHES.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with pytest.raises(SystemExit) as exc:
        complete.main(["--legacy-reference-hold", "--output", str(output)])
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


def synthetic_highmotion_v2_summary(*, candidate_acc=0.625, undefined_fde=False):
    """Mocked-loader fixture only; these values are not experimental results."""
    from tools.densevideo.highmotion_v2_bundle import BASELINE_METHODS, COMPARISON_CAVEAT
    from tools.densevideo.highmotion_v2_scoring import (
        MASK_POLICY,
        METRICS,
        SCORER_VERSION,
        TARGET_JOINT,
        VERSION,
    )

    candidate_method = "synthetic_grt_v2_fixture"
    base = {"grid_acc": 0.5, "grid_ade": 0.5, "grid_fde": None if undefined_fde else 0.5,
            "grid_transition_acc": 0.5, "token_f1": 0.5}
    candidate = {**base, "grid_acc": candidate_acc, "grid_ade": 0.75}
    rows = []
    for method in (*BASELINE_METHODS, candidate_method):
        metric_rows = dict.fromkeys(METRICS, 1000)
        metric_slots = dict.fromkeys(METRICS, 8000)
        metric_slots.update(grid_fde=1000, grid_transition_acc=7000)
        if undefined_fde:
            metric_slots.update(grid_acc=7000, grid_ade=7000, token_f1=7000,
                                grid_transition_acc=6000)
            metric_rows["grid_fde"] = metric_slots["grid_fde"] = 0
        rows.append({
            "method": method, "samples": 1000, "records": 1000,
            "model": "Synthetic display name: " + method,
            "sampled_slots": 8000, "valid_slots": 7000 if undefined_fde else 8000,
            "records_with_scored_slots": 1000, "records_without_scored_slots": 0,
            "prediction_source": "new_grt" if method == candidate_method else "archived_baseline",
            "predictions_sha256": hashlib.sha256(method.encode()).hexdigest(),
            **(candidate if method == candidate_method else base),
            "metric_scored_records": metric_rows, "metric_scored_slots_or_edges": metric_slots,
        })
    rows.sort(key=lambda row: (-row["grid_acc"], row["method"]))
    previous, rank = None, None
    for position, row in enumerate(rows, 1):
        rank = rank if row["grid_acc"] == previous else position
        row["rank"], previous = rank, row["grid_acc"]
    delta = {metric: None if base[metric] is None else candidate[metric] - base[metric]
             for metric in METRICS}
    oriented = {metric: None if value is None else -value if metric in ("grid_ade", "grid_fde") else value
                for metric, value in delta.items()}
    better = {metric: None if value is None else value > 1e-12 for metric, value in oriented.items()}
    return {
        "status": "numeric_reports_validated", "benchmark_version": VERSION,
        "scorer_version": SCORER_VERSION, "target_joint": TARGET_JOINT,
        "reference_policy": MASK_POLICY, "references_sha256": "a" * 64,
        "input_sequence_sha256": "b" * 64, "scope": "first-1000-source-rows",
        "source_reference_records": 3243, "full_source_coverage": False,
        "automatic_publication": False, "method_count": 19, "rows": rows,
        "release_integrity_verified": True, "release_manifest_sha256": "c" * 64,
        "comparison_caveat": "SYNTHETIC FIXTURE. " + COMPARISON_CAVEAT,
        "comparison": {
            "baseline_method": "llava_onevision_0_5b", "grt_method": candidate_method,
            "primary_metric": "grid_acc", "point_tolerance": 1e-12,
            "baseline_metrics": base, "grt_metrics": candidate,
            "grt_minus_baseline": delta, "oriented_improvements": oriented,
            "metric_outperform": better, "grid_acc_outperform": better["grid_acc"],
        },
        "provenance": {"fixture_only": True, "manifest_sha256": "c" * 64},
    }


def install_mocked_v2_loader(monkeypatch, summary):
    calls = []

    def mocked_loader(bundle, manifest_sha256):
        calls.append((bundle, manifest_sha256))
        return deepcopy(summary)

    monkeypatch.setattr(complete, "_load_highmotion_v2", mocked_loader)
    return calls


def test_optional_v2_loads_only_explicit_bundle_and_retains_legacy_hold(monkeypatch):
    summary = synthetic_highmotion_v2_summary()
    calls = install_mocked_v2_loader(monkeypatch, summary)
    before = complete.build_outputs(include_highmotion_v2=False)
    assert calls == []
    output = complete.build_outputs(highmotion_v2_bundle=Path("synthetic-not-a-real-bundle"),
                                    highmotion_v2_manifest_sha256="c" * 64)
    assert calls == [(Path("synthetic-not-a-real-bundle"), "c" * 64)]
    audit = json.loads(output["data/public-audit.json"])
    assert audit["highmotion_v2"] == summary
    assert audit["highmotion_additional"] == []
    assert audit["highmotion_release_eligible_rows"] == 0
    assert audit["highmotion_release_status"] == complete.HIGHMOTION_RELEASE_STATUS
    for name in ("data/leaderboard.js", "data/highmotion-audit.json"):
        assert output[name] == before[name]
    assert audit["families"] == json.loads(before["data/public-audit.json"])["families"]


def test_optional_v2_appends_19_csv_rows_without_changing_41_educational_rows(monkeypatch):
    original = list(csv.DictReader(io.StringIO(complete.build_outputs(include_highmotion_v2=False)["data/leaderboard-complete.csv"])))
    summary = synthetic_highmotion_v2_summary()
    install_mocked_v2_loader(monkeypatch, summary)
    output = complete.build_outputs(highmotion_v2_bundle=Path("synthetic-bundle"),
                                    highmotion_v2_manifest_sha256="c" * 64)
    records = list(csv.DictReader(io.StringIO(output["data/leaderboard-complete.csv"])))
    assert len(records) == 60
    for old, current in zip(original, records[:41]):
        assert all(current[key] == value for key, value in old.items())
    assert {row["cohort"] for row in records[41:]} == {"highmotion_right_ring_v2_preview1000"}
    for actual, expected in zip(records[41:], summary["rows"]):
        assert actual["method"] == expected["method"]
        assert actual["samples"] == "1000"
        assert actual["benchmark_version"] == "highmotion-right-ring-v2"
        for metric in complete.HM_METRICS:
            assert actual[metric] == str(expected[metric])
            assert actual[metric + "_scored_records"] == "1000"
            assert actual[metric + "_scored_slots_or_edges"] == str(expected["metric_scored_slots_or_edges"][metric])
        assert "metric_scored_records" not in actual
        assert "metric_scored_slots_or_edges" not in actual
    assert "grt_llava_ov_0_5b" not in output["data/leaderboard-complete.csv"]


@pytest.mark.parametrize("candidate_acc", [0.625, 0.375, 0.5])
def test_optional_v2_html_shows_all_metrics_coverage_and_negative_deltas(monkeypatch, candidate_acc):
    summary = synthetic_highmotion_v2_summary(candidate_acc=candidate_acc)
    install_mocked_v2_loader(monkeypatch, summary)
    output = complete.build_outputs(highmotion_v2_bundle=Path("synthetic-bundle"),
                                    highmotion_v2_manifest_sha256="c" * 64)
    page = output["leaderboard.html"]
    parser = ViewParser()
    parser.feed(page)
    assert parser.rows_per_table == [29, 19, 5, 12]
    assert not parser.scripts and not parser.external_styles
    assert {urlsplit(link).path for link in parser.local_links} <= set(output)
    assert "60 CSV records" in page and "preview-1000" in page
    assert "not a full 3,243-record evaluation" in page
    assert "1000 rows / 8000 slots" in page and "1000 rows / 7000 edges" in page
    assert "1000 rows / 1000 slots" in page
    assert "GRT minus baseline" in page and "Oriented improvement" in page
    assert '<td title="-0.25">-0.25</td>' in page  # GRT ADE regression is retained.
    assert "Legacy High-Motion results remain withheld" in page
    assert "weight revisions and consumed-tensor identity are unproven" in page
    assert complete.V2_CODE_URL in page and complete.V2_GUIDE_URL in page
    assert 'data/leaderboard-complete.csv?v=hm-v2-cccccccccccccccc' in page
    assert 'data/public-audit.json?v=hm-v2-cccccccccccccccc' in page
    if candidate_acc > 0.5:
        assert "GRT exceeds the corresponding HF 0.5B baseline on observed Grid Accuracy." in page
    else:
        assert "GRT does not exceed the corresponding HF 0.5B baseline on observed Grid Accuracy." in page
    assert "grt_llava_ov_0_5b" not in page


def test_optional_v2_null_metrics_display_undefined_coverage_not_zero(monkeypatch):
    install_mocked_v2_loader(monkeypatch, synthetic_highmotion_v2_summary(undefined_fde=True))
    output = complete.build_outputs(highmotion_v2_bundle=Path("synthetic-bundle"),
                                    highmotion_v2_manifest_sha256="c" * 64)
    assert 'title="Undefined or unranked; see valid-reference coverage">—</td>' in output["leaderboard.html"]
    assert "0 rows / 0 slots" in output["leaderboard.html"]
    records = list(csv.DictReader(io.StringIO(output["data/leaderboard-complete.csv"])))
    assert all(row["grid_fde"] == "" and row["grid_fde_scored_records"] == "0"
               for row in records[41:])


@pytest.mark.parametrize("kwargs", [
    {"highmotion_v2_bundle": Path("synthetic-bundle")},
    {"highmotion_v2_manifest_sha256": "c" * 64},
])
def test_optional_v2_arguments_must_be_paired_before_loading(monkeypatch, kwargs):
    def unexpected(*args, **kwargs):
        pytest.fail("Unpaired arguments must fail before any loader")

    monkeypatch.setattr(complete, "load_data", unexpected)
    monkeypatch.setattr(complete, "_load_highmotion_v2", unexpected)
    with pytest.raises(ValueError, match="supplied together"):
        complete.build_outputs(**kwargs)


@pytest.mark.parametrize("arguments", [
    ["--highmotion-v2-bundle", "synthetic-bundle"],
    ["--highmotion-v2-manifest-sha256", "c" * 64],
])
def test_optional_v2_cli_unpaired_arguments_fail_before_writes(arguments, tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        complete.main(arguments + ["--output", str(output)])
    assert error.value.code == 2 and not output.exists()


def test_optional_v2_cli_reports_actual_preview_counts_without_writing(monkeypatch, tmp_path, capsys):
    install_mocked_v2_loader(monkeypatch, synthetic_highmotion_v2_summary(candidate_acc=0.375))
    monkeypatch.chdir(tmp_path)
    complete.main(["--verify-only", "--highmotion-v2-bundle", "synthetic-bundle",
                   "--highmotion-v2-manifest-sha256", "c" * 64])
    report = json.loads(capsys.readouterr().out)
    assert report["leaderboard_rows"] == 48 and report["highmotion_rows"] == 19
    assert report["csv_rows"] == 60 and report["comparison_rows"] == 12
    assert report["highmotion_legacy_release_status"] == complete.HIGHMOTION_RELEASE_STATUS
    assert report["highmotion_v2_scope"] == "first-1000-source-rows"
    assert report["highmotion_grid_acc_outperform"] is False
    assert report["automatic_publication"] is False
    assert not list(tmp_path.iterdir())


def test_optional_v2_loader_failure_cannot_create_output(monkeypatch, tmp_path):
    def rejected(*args, **kwargs):
        raise ValueError("synthetic release manifest mismatch")

    monkeypatch.setattr(complete, "_load_highmotion_v2", rejected)
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        complete.main(["--highmotion-v2-bundle", "synthetic-bundle",
                       "--highmotion-v2-manifest-sha256", "c" * 64, "--output", str(output)])
    assert error.value.code == 1 and not output.exists()


def test_v2_cannot_be_injected_through_old_audit_field_or_bypass_legacy_guard():
    frozen, audit, aligned, _, _ = complete.load_data()
    summary = synthetic_highmotion_v2_summary()
    audit["highmotion_v2"] = summary
    with pytest.raises(ValueError, match="explicitly supplied"):
        complete.render_outputs(frozen, audit, aligned)
    audit["highmotion_additional"] = [{"method": "old_legacy_row"}]
    with pytest.raises(ValueError, match="release hold forbids"):
        complete.render_outputs(frozen, audit, aligned, highmotion_v2=summary)


@pytest.mark.parametrize("field,value", [("status", "unchecked"), ("benchmark_version", "legacy"),
                                        ("method_count", 18), ("full_source_coverage", True),
                                        ("release_integrity_verified", False),
                                        ("release_manifest_sha256", "d" * 64)])
def test_renderer_refuses_wrong_v2_summary_schema(monkeypatch, field, value):
    summary = synthetic_highmotion_v2_summary()
    summary[field] = value
    install_mocked_v2_loader(monkeypatch, summary)
    with pytest.raises(ValueError):
        complete.build_outputs(highmotion_v2_bundle=Path("synthetic-bundle"),
                               highmotion_v2_manifest_sha256="c" * 64)


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
