"""Rebuild the current release-policy-screened leaderboard from pinned evidence.

This verifies historical artifacts, not fresh inference. The educational quality
means are independently recomputed by the frozen-bundle verifier. The immutable
27-run High-Motion audit is still verified, but its historical protocol screening
does not establish current release eligibility: all High-Motion results are held
pending target/reference consistency review. No data, network or GPU are needed.
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


def cell(value):
    if value is None:
        return '<td title="Not reported">—</td>'
    if isinstance(value, float):
        return f'<td title="{value!r}">{value:.6g}</td>'
    return "<td>" + html.escape(str(value)) + "</td>"


def table(rows, columns, caption):
    header = "".join('<th scope="col">' + html.escape(label) + "</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(cell(row.get(key)) for key, _ in columns) + "</tr>")
    return '<div class="table-scroll" role="region" tabindex="0" aria-label="' + html.escape(caption, quote=True) + '"><table><caption>' + html.escape(caption) + "</caption><thead><tr>" + header + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table></div>"


def render_outputs(frozen, audit, aligned):
    require(aligned == [] and audit.get("highmotion_additional") == []
            and audit.get("highmotion_release_status") == HIGHMOTION_RELEASE_STATUS
            and audit.get("highmotion_release_eligible_rows") == 0,
            "High-Motion release hold forbids ranked or unranked numeric rows")
    lpm_columns = [("rank", "Rank"), ("model", "Model"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("cer", "CER ↓"), ("wer", "WER ↓"), ("exact_match", "Exact match ↑"), ("recompute_ratio", "Patch recompute ↓"), ("reference_recompute_ratio", "Reference patch compute ↓"), ("effective_fps", "Sampling density (fps)"), ("throughput_fps", "Mean throughput (fps) ↑"), ("source", "Source")]
    control_columns = [("family_label", "Family"), ("label", "Control"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("patch_ratio", "Patch recompute ↓"), ("throughput_fps", "Mean throughput (fps) ↑"), ("mean_wall_time_s", "Mean request time (s) ↓")]
    controls = [dict(row, family_label=family["label"]) for family in audit["families"] for row in family["methods"]]
    pieces = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>DIVE-Bench — Complete Protocol-Screened Leaderboard</title><style>" + STANDALONE_CSS + "</style></head><body><main class=\"section-shell audit-page\">",
              '<a href="https://www.zhanghaichao.xyz/DenseVideoUnderstand/#leaderboard">← Interactive project page</a><h1>Complete protocol-screened leaderboard</h1>',
              '<p>Audit and release hold: 14 September 2026. Educational scores retain the 20 August snapshot. These are historical artifacts, not a fresh GPU rerun. <a href="data/leaderboard-complete.csv?v=20260914-target-hold">Download CSV</a> · <a href="data/public-audit.json?v=20260914-target-hold">Source hashes and release status</a> · <a href="' + CODE_URL + '">Public reproduction code</a>.</p>',
              '<p><strong>29 Educational results, plus 12 educational GRT comparison rows (41 CSV records). No High-Motion results are currently released.</strong> Educational: 634 QA / 317 videos. A dash means unreported, not zero. All numeric cells retain full precision in their title and downloadable data.</p>',
              '<nav aria-label="Tables and release status"><a href="#educational">Educational</a> · <a href="#highmotion-aligned">High-Motion release hold</a> · <a href="#grt-controls">GRT vs all controls</a></nav>',
              '<h2 id="educational">Educational High-FPS Videos</h2><p>29 published methods. Rank by reported Open MOS, then Token F1; missing Open MOS sorts last and is never filled in. Open MOS judge: Qwen/Qwen3-VL-32B-Instruct. The verified GRT profiles use eight sampled frames, not a measured high-FPS operating point.</p>',
              table(frozen["tracks"]["lpm"], lpm_columns, "29 educational methods; 634 items each"),
              '<h2 id="highmotion-aligned">High-Motion High-FPS Videos: target/reference consistency audit</h2><p>' + html.escape(HIGHMOTION_HOLD_REASON) + '</p>',
              '<p>There is no ranked or unranked High-Motion results table, and no High-Motion CSV entry. The immutable <a href="data/highmotion-audit.json">historical 27-run protocol audit</a> still verifies 18 archived protocol-screened candidates; those historical eligibility flags are not current release permission. Matching saved prompts, target labels and recomputed scores does not establish that the reference trajectory follows the body part requested by the question.</p>',
              '<h2 id="grt-controls">GRT vs archived and matched controls</h2><p>All three promoted educational candidates exceed every contracted Open MOS and Token F1 floor, with 11.43–15.23% fewer patch projections than the matched all-patch route. LLaVA-OneVision 7B failed its MOS gate and is not promoted. These are observed point estimates, not statistical-significance claims. Archived Qwen baselines differ from stronger matched controls: their entire gap must not be attributed to GRT.</p>',
              table(controls, control_columns, "All 12 educational comparison rows; 634 items and eight sampled frames per method"),
              '<p><strong>Not all metrics improve.</strong> Qwen 3B GRT reports mean throughput 1.67264 fps versus 1.74994 for its all-patch control (about 4.42% lower), even though its Open MOS, Token F1 and patch reuse improve. Route31 mean request time is essentially unchanged versus its all-patch control. Throughput is the mean of per-request sampled-frame rates, not total frames divided by total campaign time, and these single historical runs do not establish repeated speedup. Patch ratios measure patch projection only, not end-to-end FLOPs.</p>',
              '<p><a href="data/leaderboard.js">Original immutable 32-row historical snapshot</a> remains byte-identical to the archived numerical bundle; it is not the current release-policy view. The hold does not rewrite historical evidence or change Educational scores, ranks or GRT gates. Full dataset access, GPU/judge reproduction and manuscript alignment remain separate release checks.</p></main></body></html>']
    records = []
    for cohort, rows in [("educational_published", frozen["tracks"]["lpm"]), ("educational_grt_controls", controls)]:
        records.extend(dict(row, cohort=cohort) for row in rows)
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


def build_outputs(bundle=None, *, historical_bundle=None):
    """Return every view/data asset, with no file writes or network access."""
    frozen, audit, aligned, frozen_bytes, evidence_bytes = load_data(
        bundle, historical_bundle=historical_bundle,
    )
    outputs = render_outputs(frozen, audit, aligned)
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
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing path")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only and args.output:
        parser.error("--verify-only and --output are mutually exclusive")
    try:
        outputs = build_outputs(args.bundle, historical_bundle=args.historical_bundle)
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
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Complete leaderboard verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
