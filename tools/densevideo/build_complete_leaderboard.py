"""Rebuild the current release-policy-screened leaderboard from pinned evidence.

This verifies saved evidence, without performing inference. Educational means
are independently recomputed by the frozen-bundle verifier. High-Motion v2 uses
corrected-reference numeric reports for 18 cached baselines and a completed GRT
run. Legacy High-Motion numbers remain withheld. No data, network or GPU needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
from pathlib import Path

from tools.densevideo.rebuild_published_leaderboard import (
    PROVENANCE_SHA256,
    SNAPSHOT_SHA256,
    checked_bytes,
    close,
    require,
    verify_bundle,
)
from tools.densevideo.release_resources import resolve_release_bundle

RELEASE = "2026-09-14"
MANIFEST_SHA256 = "5767bd60a3d8efe24722d9d4b12a374c7e566d50c370cbacdc02cdfab3162232"
HIGHMOTION_V2_RELEASE = "2026-09-15"
HIGHMOTION_V2_MANIFEST_SHA256 = "1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc"
HIGHMOTION_RELEASE_STATUS = "held_target_reference_consistency_review"
HIGHMOTION_HOLD_DATE = "2026-09-14"
HIGHMOTION_HOLD_REASON = (
    "2026-09-14 target/reference consistency hold: an initial bounded check of four "
    "canonical construction references found stored trajectories matching a left-index "
    "joint projection while the task asks for a right-hand target. Constructor/target "
    "consistency requires review; this is not a finding about all 3,243 items or GRT "
    "performance. All High-Motion results are withheld, including previously "
    "protocol-screened archive candidates."
)
CODE_URL = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/release/dive-bench-minimal"
V2_CODE_URL = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/fix/highmotion-target-v2"
V2_GUIDE_URL = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/blob/fix/highmotion-target-v2/docs/HIGHMOTION_V2_REPRODUCTION.md"
STANDALONE_CSS = """
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#172938;background:#f6f3ec}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:32px 20px}
h1{font-size:clamp(1.8rem,5vw,3rem)}h2{margin-top:40px}p{line-height:1.65}
a{color:#155a76;overflow-wrap:anywhere}nav{line-height:2}.table-scroll{overflow:auto;
max-width:100%;border:1px solid #bbc7c8;border-radius:8px;background:#fff}
table{border-collapse:collapse;width:100%;font-size:.85rem}caption{text-align:left;
padding:14px;font-weight:700}th,td{padding:10px;border-bottom:1px solid #dce3e3;
text-align:left;vertical-align:top}th{background:#eaf0ef}td{min-width:80px}
td:nth-child(2),td:nth-child(3){overflow-wrap:anywhere;min-width:180px}
@media(max-width:600px){main{padding:20px 12px}th,td{padding:8px}}
""".strip()
TELEMETRY_FIELDS = {
    "family", "method", "patch_ratio", "throughput_fps",
    "mean_wall_time_s", "telemetry_summary_sha256",
}
HM_METRICS = {
    "grid_acc": 1, "grid_ade": math.sqrt(2), "grid_fde": math.sqrt(2),
    "grid_transition_acc": 1, "token_f1": 1,
}


def finite_number(value, label, *, lower=0, upper=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Non-finite or nonnumeric {label}")
    require(value >= lower and (upper is None or value <= upper),
            f"Out-of-range {label}")
    return value


def validate_contract(manifest, provenance, telemetry, evidence):
    """Verify the historical contract; its eligibility flags are not release policy."""
    require(manifest["schema_version"] == telemetry["schema_version"] == evidence["schema_version"] == 1,
            "Unsupported complete-leaderboard schema")
    require(manifest["historical_leaderboard_sha256"] == SNAPSHOT_SHA256
            and manifest["numeric_provenance_sha256"] == PROVENANCE_SHA256,
            "Manifest identifies a different historical numerical bundle")
    require(manifest["counts"] == {
        "educational": 29, "highmotion_evidence": 27, "highmotion_eligible": 18,
        "highmotion_excluded": 9, "comparison_rows": 12, "csv_rows": 59,
    }, "Unexpected screened coverage contract")
    require(set(manifest["files"]) == {
        "highmotion-audit.json", "comparison_telemetry.json",
    }, "Unexpected complete-leaderboard input file")

    contracts = manifest["families"]
    families = provenance["families"]
    require([f["family"] for f in contracts] == ["route31", "qwen3", "qwen7"],
            "Expected exactly Route31, Qwen3 and Qwen7 family contracts")
    require([f["family"] for f in families] == [f["family"] for f in contracts],
            "Missing, duplicate or reordered educational families")
    expected_telemetry = set()
    for contract, family in zip(contracts, families):
        methods = family["methods"]
        require(len(methods) == len(contract["methods"]) == 4,
                "Every family requires all four comparison methods")
        require([(r["method"], r["role"]) for r in methods]
                == [(r["method"], r["role"]) for r in contract["methods"]],
                "Method identity, order or comparison role changed")
        require(len({r["method"] for r in methods}) == 4,
                "Duplicate comparison method")
        require([r["role"] for r in methods] == [
            "archived_public", "matched_control", "matched_control", "candidate",
        ], "Unexpected comparison role topology")
        require(family["selected_method"] == methods[-1]["method"],
                "Selected candidate does not match comparison contract")
        for method in methods:
            require(method["samples"] == 634, "Wrong educational sample count")
            finite_number(method["open_mos"], "Open MOS", upper=5)
            finite_number(method["token_f1"], "Token F1", upper=1)
            require(re.fullmatch(r"[0-9a-f]{64}", method["raw_sample_sha256"]),
                    "Invalid comparison sample hash")
            if method["role"] != "archived_public":
                expected_telemetry.add((family["family"], method["method"]))
        for metric in ("open_mos", "token_f1"):
            require(all(methods[-1][metric] > r[metric] for r in methods[:-1]),
                    "Candidate no longer exceeds every quality floor")

    rows = telemetry["rows"]
    require(len(rows) == 9 and {
        (r["family"], r["method"]) for r in rows
    } == expected_telemetry, "Incomplete or duplicate nine-method telemetry")
    for row in rows:
        require(set(row) == TELEMETRY_FIELDS, "Unexpected telemetry fields")
        finite_number(row["patch_ratio"], "patch ratio", lower=1e-15, upper=1)
        finite_number(row["throughput_fps"], "throughput", lower=1e-15)
        finite_number(row["mean_wall_time_s"], "request time", lower=1e-15)
        require(re.fullmatch(r"[0-9a-f]{64}", row["telemetry_summary_sha256"]),
                "Invalid telemetry source hash")
        family = next(f for f in families if f["family"] == row["family"])
        if row["method"] == family["selected_method"]:
            saved = family["candidate_telemetry"]
            close(row["patch_ratio"], saved["mean_recompute_ratio"], "candidate patches")
            close(row["throughput_fps"], saved["mean_throughput_fps"], "candidate throughput")
            require(row["telemetry_summary_sha256"] == family["source_sha256"]["summary_csv"],
                    "Candidate telemetry contradicts its original summary")
        else:
            require(row["patch_ratio"] == 1, "Matched all-patch route must recompute all patches")

    archived = evidence["rows"]
    require(len(archived) == 27
            and len({r["method"] for r in archived}) == 27
            and [r["method"] for r in archived] == manifest["highmotion_methods"],
            "Missing, duplicate or changed 27-run High-Motion evidence")
    eligible = []
    for row in archived:
        require(row["samples"] == 1000 and row["artifact_integrity_pass"] is True
                and row["ordered_first1000_identity_match"] is True,
                "High-Motion artifact integrity or fixed-preview identity failed")
        require(row["ordered_identity_sha256"] == evidence["preview_ordered_identity_sha256"],
                "High-Motion ordered identities disagree")
        require(row["rank_eligible"] in (True, False)
                and (row["protocol_status"] == "aligned_preview") == row["rank_eligible"],
                "Protocol eligibility contradicts status")
        for key, ceiling in HM_METRICS.items():
            finite_number(row["metrics"][key], key, upper=ceiling)
        if not row["rank_eligible"]:
            require(row["protocol_status"] == "historical_unaligned",
                    "Unknown excluded protocol status")
            continue
        eligible.append(row["method"])
        require(row["source"] == "open"
                and row["input_sampling_policy"] == "eight_endpoint_inclusive_uniform"
                and row["reference_sampling_policy"] == "eight_endpoint_inclusive_uniform",
                "Eligible High-Motion input and target policies must match")
        for field in ("unique_identity_count", "prompt_match", "target_match",
                      "nonempty_predictions", "score_record_prompt_match"):
            require(row[field] == 1000, f"Incomplete High-Motion {field}")
        require(row["doc_id_order_match"] is True and row["full_doc_content_match"] is True,
                "High-Motion original annotations or document ordering changed")
        require(row["metric_recomputation_max_per_sample_abs_error"] == 0
                and row["aggregate_metric_max_abs_error"] == 0
                and row["error_log_markers"] == 0,
                "High-Motion saved metric or generation integrity failed")
        require(row["fresh_gpu_reproduction"] is False,
                "Historical audit cannot certify new inference")
        for key in ("samples_sha256", "result_sha256", "ordered_identity_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", row[key]), f"Invalid High-Motion {key}")
        for metric in HM_METRICS:
            require(row["metrics"][metric] == row["recomputed_metrics"][metric],
                    "High-Motion aggregate contradicts its audited recomputation")
    require(len(eligible) == 18 and eligible == manifest["eligible_highmotion_methods"],
            "Expected exactly the 18 protocol-screened High-Motion methods")


def load_data(bundle=None, *, historical_bundle=None):
    bundle = resolve_release_bundle(bundle, release=RELEASE)
    historical = resolve_release_bundle(historical_bundle, release="2026-08-20")
    # This recomputes 7,608 educational sample records for all 12 compared methods,
    # checks every frozen CSV/JS value and validates the original evidence hashes.
    verify_bundle(historical)
    manifest = json.loads(checked_bytes(bundle, "manifest.json", MANIFEST_SHA256))
    source_bytes = {
        name: checked_bytes(bundle, name, digest)
        for name, digest in manifest["files"].items()
    }
    evidence = json.loads(source_bytes["highmotion-audit.json"])
    telemetry = json.loads(source_bytes["comparison_telemetry.json"])
    provenance = json.loads(checked_bytes(historical, "provenance.json", PROVENANCE_SHA256))
    validate_contract(manifest, provenance, telemetry, evidence)
    frozen_bytes = checked_bytes(historical, "leaderboard.js", SNAPSHOT_SHA256)
    frozen = json.loads(frozen_bytes.decode().split(
        "window.DIVE_LEADERBOARD = ", 1)[1].strip().removesuffix(";"))
    audit = {
        "audit_date": manifest["audit_date"],
        "result_snapshot": manifest["result_snapshot"],
        "historical_leaderboard_sha256": SNAPSHOT_SHA256,
        "numeric_provenance_sha256": PROVENANCE_SHA256,
        "code_commit": manifest["code_commit"],
        "judge": provenance["judge"],
        "families": [],
    }
    telemetry_by_method = {r["method"]: r for r in telemetry["rows"]}
    for contract, family in zip(manifest["families"], provenance["families"]):
        methods = []
        for display, method in zip(contract["methods"], family["methods"]):
            # Use the pinned declared precision after the independent sample-mean
            # verification; do not introduce floating-point re-summation drift.
            row = {k: v for k, v in method.items() if k != "method_index"}
            row["label"] = display["label"]
            values = telemetry_by_method.get(method["method"], {})
            for key in ("patch_ratio", "throughput_fps", "mean_wall_time_s",
                        "telemetry_summary_sha256"):
                row[key] = values.get(key)
            methods.append(row)
        audit["families"].append({
            "family": family["family"], "label": contract["label"], "methods": methods,
        })
    # Dated current release policy, intentionally without an override flag.
    # Do not rewrite the historical audit's 18 protocol-screened candidates or
    # treat saved prompt/score agreement as proof of target-reference semantics.
    aligned = []
    audit["highmotion_additional"] = []
    audit["highmotion_release_status"] = HIGHMOTION_RELEASE_STATUS
    audit["highmotion_hold_date"] = HIGHMOTION_HOLD_DATE
    audit["highmotion_hold_reason"] = HIGHMOTION_HOLD_REASON
    audit["highmotion_historical_protocol_screened_candidates"] = 18
    audit["highmotion_release_eligible_rows"] = 0
    audit["highmotion_evidence_file"] = "highmotion-audit.json"
    audit["highmotion_evidence_sha256"] = manifest["files"]["highmotion-audit.json"]
    require(len(frozen["tracks"]["lpm"]) == 29 and not aligned,
            "Current release must retain Educational results and withhold High-Motion")
    return frozen, audit, aligned, frozen_bytes, source_bytes["highmotion-audit.json"]


def cell(value, *, missing_title="Not reported"):
    if value is None:
        return '<td title="' + html.escape(missing_title, quote=True) + '">—</td>'
    if isinstance(value, float):
        return f'<td title="{value!r}">{value:.6g}</td>'
    return "<td>" + html.escape(str(value)) + "</td>"


def table(rows, columns, caption, *, missing_title="Not reported"):
    header = "".join('<th scope="col">' + html.escape(label) + "</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(cell(row.get(key), missing_title=missing_title)
                                    for key, _ in columns) + "</tr>")
    return '<div class="table-scroll" role="region" tabindex="0" aria-label="' + html.escape(caption, quote=True) + '"><table><caption>' + html.escape(caption) + "</caption><thead><tr>" + header + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table></div>"


def _load_highmotion_v2(bundle, manifest_sha256):
    # The explicit legacy-only path remains independent of the newer bundle.
    # Current/default and explicit v2 paths authenticate this release artifact.
    from tools.densevideo.highmotion_v2_release import load_highmotion_v2_release

    return load_highmotion_v2_release(bundle, expected_manifest_sha256=manifest_sha256)


def _render_highmotion_v2(summary):
    """Format an already authenticated loader result, not an alternate verifier."""
    from tools.densevideo.highmotion_v2_bundle import BASELINE_METHODS
    from tools.densevideo.highmotion_v2_scoring import (
        MASK_POLICY,
        SCORER_VERSION,
        TARGET_JOINT,
        VERSION,
    )

    require(isinstance(summary, dict) and summary.get("status") == "numeric_reports_validated"
            and summary.get("benchmark_version") == VERSION
            and summary.get("scorer_version") == SCORER_VERSION
            and summary.get("target_joint") == TARGET_JOINT
            and summary.get("reference_policy") == MASK_POLICY,
            "Unsupported corrected High-Motion numeric summary")
    require(summary.get("scope") == "first-1000-source-rows"
            and summary.get("source_reference_records") == 3243
            and summary.get("full_source_coverage") is False
            and summary.get("automatic_publication") is False,
            "Wrong corrected High-Motion scope or publication claim")
    rows = summary.get("rows")
    require(isinstance(rows, list) and len(rows) == summary.get("method_count") == 19,
            "Corrected preview must contain 18 baselines and one GRT method")
    comparison = summary["comparison"]
    require(comparison["baseline_method"] == "llava_onevision_0_5b"
            and comparison["grt_method"] not in BASELINE_METHODS
            and len({row["method"] for row in rows}) == 19
            and {row["method"] for row in rows} == set(BASELINE_METHODS) | {comparison["grt_method"]},
            "Corrected High-Motion method topology changed")
    labels = {
        "grid_acc": "Grid Accuracy ↑", "grid_ade": "Grid ADE ↓", "grid_fde": "Grid FDE ↓",
        "grid_transition_acc": "Transition Accuracy ↑", "token_f1": "Token F1 ↑",
    }
    columns = [("rank", "Rank"), ("model", "Model"), ("method", "Method ID"),
               ("samples", "Cached/new records")]
    for metric, label in labels.items():
        columns.extend([(metric, label), (metric + "_coverage", label + " coverage")])
    columns.append(("prediction_source", "Prediction provenance"))
    display_rows, csv_rows = [], []
    for row in rows:
        require(row["samples"] == row["records"] == 1000 and row["sampled_slots"] == 8000,
                "Corrected preview record or sampled-slot count changed")
        display = dict(row)
        display["model"] = row.get("model") or row["method"]
        flat = {key: value for key, value in row.items()
                if key not in ("metric_scored_records", "metric_scored_slots_or_edges")}
        flat["model"] = display["model"]
        require(all(value is None or type(value) in (str, bool, int, float) for value in flat.values()),
                "Nested fields must not be silently stringified in CSV")
        for metric, ceiling in HM_METRICS.items():
            count = row["metric_scored_records"][metric]
            slots = row["metric_scored_slots_or_edges"][metric]
            require(type(count) is int and 0 <= count <= 1000
                    and type(slots) is int and 0 <= slots <= 8000
                    and (row[metric] is None) == (count == 0) == (slots == 0),
                    "Invalid corrected metric coverage")
            if row[metric] is not None:
                finite_number(row[metric], "corrected " + metric, upper=ceiling)
            unit = "edges" if metric == "grid_transition_acc" else "slots"
            display[metric + "_coverage"] = f"{count} rows / {slots} {unit}"
            flat[metric + "_scored_records"] = count
            flat[metric + "_scored_slots_or_edges"] = slots
        for key in ("benchmark_version", "scorer_version", "target_joint", "reference_policy",
                    "references_sha256", "input_sequence_sha256", "scope"):
            flat[key] = summary[key]
        display_rows.append(display)
        csv_rows.append(flat)
    deltas = []
    for metric, label in labels.items():
        outcome = comparison["metric_outperform"][metric]
        require(outcome is None or type(outcome) is bool, "Invalid point-comparison outcome")
        deltas.append({
            "metric": label, "baseline": comparison["baseline_metrics"][metric],
            "grt": comparison["grt_metrics"][metric],
            "delta": comparison["grt_minus_baseline"][metric],
            "oriented": comparison["oriented_improvements"][metric],
            "exceeds_tolerance": "undefined" if outcome is None else "yes" if outcome else "no",
        })
    require(type(comparison["grid_acc_outperform"]) is bool,
            "Primary comparison must be an explicitly computed boolean")
    primary = ("GRT exceeds the corresponding HF 0.5B baseline on observed Grid Accuracy."
               if comparison["grid_acc_outperform"] else
               "GRT does not exceed the corresponding HF 0.5B baseline on observed Grid Accuracy.")
    missing_title = "Undefined or unranked; see valid-reference coverage"
    sections = [
        '<h2 id="highmotion-v2">High-Motion v2: right-ring reference correction, preview-1000</h2>',
        ('<p>19 methods: 18 archived baselines rescored on CPU and one new HF 0.5B GRT run. '
        'All use the same fixed first 1,000 source records, not a full 3,243-record evaluation. '
        'The reference target is <code>rightRingFingerMetacarpal</code>, the named '
        'right-palm/ring-finger-base proxy; questions and original sampled positions are unchanged. '
        'This versioned correction does not claim to recover the original annotation constructor.</p>'),
        ('<p>Rank by Grid Accuracy descending; exact ties share a competition rank and are ordered '
        'by method ID. Null scores are unranked. Every metric is a macro mean over its defined '
        'per-record values, with visible metric-specific row and slot/edge coverage. '
        'A dash means undefined under the reference mask, not zero. Invalid slots never shift '
        'later predictions; FDE uses the original final slot, and transitions require adjacent '
        'valid original slots. Token F1 uses canonical label bags on valid positions, with surplus '
        'outputs penalized. All 1,000 records remain counted even when no positions are scoreable.</p>'),
        '<p>' + html.escape(summary["comparison_caveat"]) + '</p>',
        table(display_rows, columns, "19 corrected-reference preview methods; eight slots per record",
              missing_title=missing_title),
        '<h3 id="highmotion-v2-comparison">GRT versus its corresponding HF 0.5B baseline</h3><p>'
        + html.escape(primary) + ' Point tolerance: '
        + html.escape(str(comparison["point_tolerance"]))
        + '. Every observed difference is retained, including regressions. '
        'This is a point-estimate comparison, not a statistical-significance claim.</p>',
        table(deltas, [("metric", "Metric"), ("baseline", "Rescored HF 0.5B baseline"),
                       ("grt", "GRT"), ("delta", "GRT minus baseline"),
                       ("oriented", "Oriented improvement (positive is better)"),
                       ("exceeds_tolerance", "Exceeds point tolerance")],
              "All five corrected-reference GRT versus baseline differences",
              missing_title=missing_title),
        ('<h2 id="highmotion-aligned">Legacy High-Motion results remain withheld</h2><p>'
        'The old target/reference hold and immutable '
        '<a href="data/highmotion-audit.json">27-run historical protocol audit</a> remain intact. '
        'No old High-Motion score is mixed into the corrected v2 table or its CSV cohort. '
        'Reference, scorer, prediction and release provenance are available in '
        '<a href="data/public-audit.json?v=hm-v2-' + summary["release_manifest_sha256"][:16]
        + '">the versioned numeric audit</a>.</p>'),
    ]
    return csv_rows, sections


def render_outputs(frozen, audit, aligned, *, highmotion_v2=None):
    require(aligned == [] and audit.get("highmotion_additional") == []
            and audit.get("highmotion_release_status") == HIGHMOTION_RELEASE_STATUS
            and audit.get("highmotion_release_eligible_rows") == 0,
            "High-Motion release hold forbids ranked or unranked numeric rows")
    require((highmotion_v2 is None and "highmotion_v2" not in audit)
            or (highmotion_v2 is not None and audit.get("highmotion_v2") == highmotion_v2),
            "Corrected High-Motion payload was not explicitly supplied by the loader")
    lpm_columns = [("rank", "Rank"), ("model", "Model"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("cer", "CER ↓"), ("wer", "WER ↓"), ("exact_match", "Exact match ↑"), ("recompute_ratio", "Patch recompute ↓"), ("reference_recompute_ratio", "Reference patch compute ↓"), ("effective_fps", "Sampling density (fps)"), ("throughput_fps", "Mean throughput (fps) ↑"), ("source", "Source")]
    control_columns = [("family_label", "Family"), ("label", "Control"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("patch_ratio", "Patch recompute ↓"), ("throughput_fps", "Mean throughput (fps) ↑"), ("mean_wall_time_s", "Mean request time (s) ↓")]
    controls = [dict(row, family_label=family["label"]) for family in audit["families"] for row in family["methods"]]
    intro = '<p>Audit and release hold: 14 September 2026. Educational scores retain the 20 August snapshot. These are historical artifacts, not a fresh GPU rerun. <a href="data/leaderboard-complete.csv?v=20260914-target-hold">Download CSV</a> · <a href="data/public-audit.json?v=20260914-target-hold">Source hashes and release status</a> · <a href="' + CODE_URL + '">Public reproduction code</a>.</p>'
    coverage = '<p><strong>29 Educational results, plus 12 educational GRT comparison rows (41 CSV records). No High-Motion results are currently released.</strong> Educational: 634 QA / 317 videos. A dash means unreported, not zero. All numeric cells retain full precision in their title and downloadable data.</p>'
    navigation = '<nav aria-label="Tables and release status"><a href="#educational">Educational</a> · <a href="#highmotion-aligned">High-Motion release hold</a> · <a href="#grt-controls">GRT vs all controls</a></nav>'
    hm_sections = [
        '<h2 id="highmotion-aligned">High-Motion High-FPS Videos: target/reference consistency audit</h2><p>' + html.escape(HIGHMOTION_HOLD_REASON) + '</p>',
        '<p>There is no ranked or unranked High-Motion results table, and no High-Motion CSV entry. The immutable <a href="data/highmotion-audit.json">historical 27-run protocol audit</a> still verifies 18 archived protocol-screened candidates; those historical eligibility flags are not current release permission. Matching saved prompts, target labels and recomputed scores does not establish that the reference trajectory follows the body part requested by the question.</p>',
    ]
    hm_rows = []
    if highmotion_v2 is not None:
        hm_rows, hm_sections = _render_highmotion_v2(highmotion_v2)
        cache_key = "hm-v2-" + highmotion_v2["release_manifest_sha256"][:16]
        intro = '<p>Educational scores retain the 20 August historical snapshot. High-Motion v2 compares CPU-rescored archived baselines with one new GRT run; no baseline inference was repeated. <a href="data/leaderboard-complete.csv?v=' + cache_key + '">Download CSV</a> · <a href="data/public-audit.json?v=' + cache_key + '">Versioned source hashes and comparison limits</a> · <a href="' + V2_CODE_URL + '">Public reproduction code</a> · <a href="' + V2_GUIDE_URL + '">Corrected-reference reproduction guide</a>.</p>'
        coverage = '<p><strong>29 Educational results, 19 corrected High-Motion preview results, and 12 educational GRT comparison rows (60 CSV records).</strong> Educational: 634 QA / 317 videos. High-Motion v2: the fixed 1,000-record preview of 3,243 source records; metric-specific valid-reference coverage is shown separately. All numeric cells retain full precision in their title and downloadable data.</p>'
        navigation = '<nav aria-label="Tables and release status"><a href="#educational">Educational</a> · <a href="#highmotion-v2">High-Motion v2 preview</a> · <a href="#highmotion-v2-comparison">High-Motion GRT comparison</a> · <a href="#grt-controls">Educational GRT controls</a></nav>'
    pieces = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>DIVE-Bench — Complete Protocol-Screened Leaderboard</title><style>" + STANDALONE_CSS + "</style></head><body><main class=\"section-shell audit-page\">",
              '<a href="https://www.zhanghaichao.xyz/DenseVideoUnderstand/#leaderboard">← Interactive project page</a><h1>Complete protocol-screened leaderboard</h1>',
              intro, coverage, navigation,
              '<h2 id="educational">Educational High-FPS Videos</h2><p>29 published methods. Rank by reported Open MOS, then Token F1; missing Open MOS sorts last and is never filled in. Open MOS judge: Qwen/Qwen3-VL-32B-Instruct. The verified GRT profiles use eight sampled frames, not a measured high-FPS operating point.</p>',
              table(frozen["tracks"]["lpm"], lpm_columns, "29 educational methods; 634 items each"),
              *hm_sections,
              '<h2 id="grt-controls">GRT vs archived and matched controls</h2><p>All three promoted educational candidates exceed every contracted Open MOS and Token F1 floor, with 11.43–15.23% fewer patch projections than the matched all-patch route. LLaVA-OneVision 7B failed its MOS gate and is not promoted. These are observed point estimates, not statistical-significance claims. Archived Qwen baselines differ from stronger matched controls: their entire gap must not be attributed to GRT.</p>',
              table(controls, control_columns, "All 12 educational comparison rows; 634 items and eight sampled frames per method"),
              '<p><strong>Not all metrics improve.</strong> Qwen 3B GRT reports mean throughput 1.67264 fps versus 1.74994 for its all-patch control (about 4.42% lower), even though its Open MOS, Token F1 and patch reuse improve. Route31 mean request time is essentially unchanged versus its all-patch control. Throughput is the mean of per-request sampled-frame rates, not total frames divided by total campaign time, and these single historical runs do not establish repeated speedup. Patch ratios measure patch projection only, not end-to-end FLOPs.</p>',
              '<p><a href="data/leaderboard.js">Original immutable 32-row historical snapshot</a> remains byte-identical to the archived numerical bundle; it is not the current release-policy view. The hold does not rewrite historical evidence or change Educational scores, ranks or GRT gates. Full dataset access, GPU/judge reproduction and manuscript alignment remain separate release checks.</p></main></body></html>']
    records = []
    for cohort, rows in [("educational_published", frozen["tracks"]["lpm"]), ("educational_grt_controls", controls)]:
        records.extend(dict(row, cohort=cohort) for row in rows)
    records.extend(dict(row, cohort="highmotion_right_ring_v2_preview1000") for row in hm_rows)
    fields = ["cohort"] + sorted({key for row in records for key in row if key != "cohort"})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return {
        "leaderboard.html": "\n".join(pieces) + "\n",
        "data/leaderboard-complete.csv": buffer.getvalue(),
        "data/public-audit.js": "/* Generated by tools.densevideo.build_complete_leaderboard; verified release evidence. */\nwindow.DIVE_PUBLIC_AUDIT = " + json.dumps(audit, ensure_ascii=False, indent=2) + ";\n",
    }


def build_outputs(bundle=None, *, historical_bundle=None, highmotion_v2_bundle=None,
                  highmotion_v2_manifest_sha256=None, include_highmotion_v2=True):
    """Return every view/data asset, with no file writes or network access."""
    require((highmotion_v2_bundle is None) == (highmotion_v2_manifest_sha256 is None),
            "Corrected High-Motion bundle and manifest SHA-256 must be supplied together")
    require(type(include_highmotion_v2) is bool, "include_highmotion_v2 must be boolean")
    require(include_highmotion_v2 or highmotion_v2_bundle is None,
            "Legacy reference-hold view cannot accept a v2 bundle")
    if include_highmotion_v2 and highmotion_v2_bundle is None:
        highmotion_v2_bundle = resolve_release_bundle(release=HIGHMOTION_V2_RELEASE)
        highmotion_v2_manifest_sha256 = HIGHMOTION_V2_MANIFEST_SHA256
    frozen, audit, aligned, frozen_bytes, evidence_bytes = load_data(
        bundle, historical_bundle=historical_bundle,
    )
    highmotion_v2 = None
    if highmotion_v2_bundle is not None:
        highmotion_v2 = _load_highmotion_v2(highmotion_v2_bundle, highmotion_v2_manifest_sha256)
        require(isinstance(highmotion_v2, dict)
                and highmotion_v2.get("release_integrity_verified") is True
                and highmotion_v2.get("release_manifest_sha256") == highmotion_v2_manifest_sha256,
                "Corrected High-Motion loader did not authenticate the requested release manifest")
        audit["highmotion_v2"] = highmotion_v2
    outputs = render_outputs(frozen, audit, aligned, highmotion_v2=highmotion_v2)
    outputs["data/public-audit.json"] = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    outputs["data/highmotion-audit.json"] = evidence_bytes.decode("utf-8")
    outputs["data/leaderboard.js"] = frozen_bytes.decode("utf-8")
    return outputs


def write_outputs(output, outputs):
    """Write exclusively, rolling back owned paths on catchable failures.

    This is best-effort exception cleanup, not atomic publication or crash
    durability. Never recursively remove a tree or unlink another writer's file.
    """
    output = Path(output).absolute()
    directories = {}
    files = {}

    def identity(stat):
        return stat.st_dev, stat.st_ino

    def make_directory(path):
        path.mkdir(exist_ok=False)
        directories[path] = identity(path.lstat())

    try:
        missing = []
        parent = output.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for parent in reversed(missing):
            try:
                make_directory(parent)
            except FileExistsError:
                if not parent.is_dir():
                    raise
        make_directory(output)
        for name, value in outputs.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "Invalid output artifact path")
            parent = output
            for part in relative.parts[:-1]:
                parent = parent / part
                if parent not in directories:
                    make_directory(parent)
                require(identity(parent.lstat()) == directories[parent],
                        "Output directory was replaced during generation")
            target = output / relative
            with target.open("x", encoding="utf-8") as stream:
                files[target] = identity(os.fstat(stream.fileno()))
                stream.write(value)
    except BaseException as error:
        incomplete = []
        for paths, remove in ((files, Path.unlink), (directories, Path.rmdir)):
            for path, created_identity in reversed(list(paths.items())):
                try:
                    if identity(path.lstat()) != created_identity:
                        incomplete.append(str(path))
                        continue
                    remove(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    incomplete.append(str(path))
        if incomplete:
            raise OSError(
                "Output write failed; cleanup was incomplete (paths preserved): "
                + ", ".join(incomplete)
            ) from error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Explicit 2026-09-14 evidence directory")
    parser.add_argument("--historical-bundle", type=Path, help="Explicit frozen 2026-08-20 bundle")
    parser.add_argument("--highmotion-v2-bundle", type=Path,
                        help="Optional separate corrected-reference preview release bundle")
    parser.add_argument("--highmotion-v2-manifest-sha256",
                        help="Reviewed immutable manifest SHA-256 for the optional v2 bundle")
    parser.add_argument("--legacy-reference-hold", action="store_true",
                        help="Rebuild only the dated 2026-09-14 hold view, not the current v2 leaderboard")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only and args.output:
        parser.error("--verify-only and --output are mutually exclusive")
    if (args.highmotion_v2_bundle is None) != (args.highmotion_v2_manifest_sha256 is None):
        parser.error("--highmotion-v2-bundle and --highmotion-v2-manifest-sha256 are required together")
    if args.legacy_reference_hold and args.highmotion_v2_bundle is not None:
        parser.error("--legacy-reference-hold cannot be combined with a v2 bundle")
    try:
        outputs = build_outputs(args.bundle, historical_bundle=args.historical_bundle,
                                highmotion_v2_bundle=args.highmotion_v2_bundle,
                                highmotion_v2_manifest_sha256=args.highmotion_v2_manifest_sha256,
                                include_highmotion_v2=not args.legacy_reference_hold)
        if args.output:
            write_outputs(args.output, outputs)
        report = {
            "status": "verified",
            "verification": "source-bound historical artifacts; no fresh inference",
            "release": RELEASE,
            "leaderboard_rows": 29,
            "educational_rows": 29,
            "highmotion_rows": 0,
            "highmotion_evidence_rows": 27,
            "highmotion_excluded_rows": 27,
            "highmotion_historical_protocol_screened_candidates": 18,
            "highmotion_release_status": HIGHMOTION_RELEASE_STATUS,
            "highmotion_hold_reason": HIGHMOTION_HOLD_REASON,
            "comparison_rows": 12,
            "csv_rows": 41,
            "html": "leaderboard.html",
            "manifest_sha256": MANIFEST_SHA256,
            "output_sha256": {
                name: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for name, value in outputs.items()
            },
        }
        if not args.legacy_reference_hold:
            summary = json.loads(outputs["data/public-audit.json"])["highmotion_v2"]
            report.update({
                "verification": "Historical Educational artifacts and authenticated corrected-reference numeric reports; no model inference performed by this command",
                "release": summary.get("audit_date", HIGHMOTION_V2_RELEASE),
                "leaderboard_rows": 29 + summary["method_count"],
                "highmotion_rows": summary["method_count"], "csv_rows": 41 + summary["method_count"],
                "highmotion_release_status": summary["status"],
                "highmotion_legacy_release_status": HIGHMOTION_RELEASE_STATUS,
                "highmotion_hold_reason": "Legacy High-Motion results remain withheld; the separate corrected v2 preview does not reuse their old scores",
                "highmotion_v2_version": summary["benchmark_version"],
                "highmotion_v2_manifest_sha256": summary["release_manifest_sha256"],
                "highmotion_v2_references_sha256": summary["references_sha256"],
                "highmotion_v2_scope": summary["scope"],
                "highmotion_grid_acc_outperform": summary["comparison"]["grid_acc_outperform"],
                "automatic_publication": False,
            })
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Complete leaderboard verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
