#!/usr/bin/env python3
"""Build dependency-free complete tables without modifying the frozen release."""

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_URL = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/release/dive-bench-minimal"
FROZEN_SHA256 = "c88492ec535a3bc4f47fe9e91f66bc050a857f757cd2c478db598e889d9dc1b1"


def load_data():
    raw = (ROOT / "data/leaderboard.js").read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError("Historical leaderboard changed; do not overwrite frozen scores")
    frozen = json.loads(raw.decode().split("window.DIVE_LEADERBOARD = ", 1)[1].strip().removesuffix(";"))
    audit = json.loads((ROOT / "data/public-audit.json").read_text())
    if audit["historical_leaderboard_sha256"] != FROZEN_SHA256:
        raise ValueError("Audit does not identify the frozen leaderboard")
    if len(frozen["tracks"]["lpm"]) != 29 or len(frozen["tracks"]["highmotion"]) != 3:
        raise ValueError("Unexpected historical coverage")
    for family in audit["families"]:
        methods = family["methods"]
        candidate = [row for row in methods if row["role"] == "candidate"]
        if len(methods) != 4 or len(candidate) != 1 or any(row["samples"] != 634 for row in methods):
            raise ValueError("Incomplete GRT comparison")
        for metric in ("open_mos", "token_f1"):
            if not all(candidate[0][metric] >= row[metric] for row in methods):
                raise ValueError("GRT quality-floor statement no longer holds")
        if not 0 < candidate[0]["patch_ratio"] < 1:
            raise ValueError("Invalid candidate patch ratio")
    additional = audit["highmotion_additional"]
    if additional:
        if audit["highmotion_evidence_file"] != "highmotion-audit.json":
            raise ValueError("Unexpected evidence filename")
        evidence_bytes = (ROOT / "data/highmotion-audit.json").read_bytes()
        if hashlib.sha256(evidence_bytes).hexdigest() != audit["highmotion_evidence_sha256"]:
            raise ValueError("High-Motion evidence checksum mismatch")
        evidence = json.loads(evidence_bytes)
        evidence_rows = {row["method"]: row for row in evidence["rows"]}
        eligible_ids = {row["method"] for row in evidence["rows"] if row["protocol_status"] == "aligned_preview" and row["rank_eligible"] is True}
        if len(evidence["rows"]) != 27 or len(evidence_rows) != 27 or len(eligible_ids) != 18:
            raise ValueError("Expected complete 27-method evidence with 18 eligible methods")
        if len(additional) != 18 or {row["method"] for row in additional} != eligible_ids:
            raise ValueError("Expected exactly the 18 protocol-screened High-Motion methods")
        if any(row["samples"] != 1000 for row in additional):
            raise ValueError("Cannot mix preview and full-split results")
        for row in additional:
            source = evidence_rows[row["method"]]
            if not source["artifact_integrity_pass"] or not source["ordered_first1000_identity_match"]:
                raise ValueError("Unaudited High-Motion row")
            if row["protocol_status"] != "aligned_preview" or row.get("rank_eligible") is not True or source["rank_eligible"] is not True:
                raise ValueError("High-Motion ranking contradicts protocol evidence")
            for key in ("samples_sha256", "result_sha256", "ordered_identity_sha256"):
                if row[key] != source[key]:
                    raise ValueError("High-Motion source identity contradicts its evidence")
            for metric in ("grid_acc", "grid_ade", "grid_fde", "token_f1", "transition_acc"):
                source_key = "grid_transition_acc" if metric == "transition_acc" else metric
                if row[metric] != source["metrics"][source_key]:
                    raise ValueError("High-Motion aggregate contradicts its evidence")
        aligned = sorted(
            (row for row in additional if row["protocol_status"] == "aligned_preview"),
            key=lambda row: (-row["grid_acc"], -row["token_f1"], row["method"]),
        )
        for rank, row in enumerate(aligned, 1):
            row["rank"] = rank
        audit["highmotion_additional"] = aligned
    else:
        raise ValueError("Screened High-Motion audit is required; frozen rows are not a fallback")
    return frozen, audit, aligned


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


def build_outputs():
    frozen, audit, aligned = load_data()
    lpm_columns = [("rank", "Rank"), ("model", "Model"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("cer", "CER ↓"), ("wer", "WER ↓"), ("exact_match", "Exact match ↑"), ("recompute_ratio", "Patch recompute ↓"), ("reference_recompute_ratio", "Reference patch compute ↓"), ("effective_fps", "Sampling density (fps)"), ("throughput_fps", "Mean throughput (fps) ↑"), ("source", "Source")]
    hm_columns = [("rank", "Rank"), ("model", "Model"), ("method", "Method ID"), ("samples", "Items"), ("grid_acc", "Grid accuracy ↑"), ("grid_ade", "Grid ADE ↓"), ("grid_fde", "Grid FDE ↓"), ("transition_acc", "Transition accuracy ↑"), ("token_f1", "Token F1 ↑"), ("protocol_note", "Protocol / status"), ("source", "Source")]
    control_columns = [("family_label", "Family"), ("label", "Control"), ("method", "Method ID"), ("samples", "Items"), ("open_mos", "Open MOS ↑"), ("token_f1", "Token F1 ↑"), ("patch_ratio", "Patch recompute ↓"), ("throughput_fps", "Mean throughput (fps) ↑"), ("mean_wall_time_s", "Mean request time (s) ↓")]
    controls = [dict(row, family_label=family["label"]) for family in audit["families"] for row in family["methods"]]
    pieces = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>DIVE-Bench — Complete Protocol-Screened Leaderboard</title><link rel=\"stylesheet\" href=\"styles.css?v=20260914-screened\"></head><body><main class=\"section-shell audit-page\">",
              '<a href="index.html#leaderboard">← Interactive project page</a><h1>Complete protocol-screened leaderboard</h1>',
              '<p>Audit: 14 September 2026. Educational scores retain the 20 August snapshot; additional High-Motion runs are from 21–23 August. These are historical artifacts, not a fresh GPU rerun. <a href="data/leaderboard-complete.csv">Download CSV</a> · <a href="data/public-audit.json">Source hashes and protocol evidence</a> · <a href="' + CODE_URL + '">Public reproduction code</a>.</p>',
              '<p><strong>47 protocol-screened results: 29 Educational + 18 High-Motion, plus 12 educational GRT comparison rows (59 CSV records).</strong> Read by protocol, not as a single composite score. Educational: 634 QA / 317 videos. High-Motion: fixed first 1,000 of 3,243 items, not a full-split evaluation. A dash means unreported, not zero. All numeric cells retain full precision in their title and downloadable data.</p>',
              '<nav aria-label="Tables"><a href="#educational">Educational</a> · <a href="#highmotion-aligned">Aligned High-Motion preview</a> · <a href="#grt-controls">GRT vs all controls</a></nav>',
              '<h2 id="educational">Educational High-FPS Videos</h2><p>29 published methods. Rank by reported Open MOS, then Token F1; missing Open MOS sorts last and is never filled in. Open MOS judge: Qwen/Qwen3-VL-32B-Instruct. The verified GRT profiles use eight sampled frames, not a measured high-FPS operating point.</p>',
              table(frozen["tracks"]["lpm"], lpm_columns, "29 educational methods; 634 items each"),
              '<h2 id="highmotion-aligned">High-Motion High-FPS Videos: aligned preview archive</h2><p>Saved wrappers/configurations specify endpoint-inclusive uniform eight-frame input and an eight-position target for the same fixed 1,000 items. Identities, prompts, targets and every saved sample metric were checked. This is a protocol-compatible archive, not a new GPU replay or an exact saved frame-byte audit. Model revisions were not fully pinned; hardware, image resolution and decoding may differ across families. <a href="data/highmotion-audit.json">Full protocol audit and source hashes</a>.</p>',
              table(aligned, hm_columns, f"{len(aligned)} aligned High-Motion methods; 1,000 items each"),
              '<p>Nine non-aligned High-Motion runs, including legacy GRT, are excluded from all leaderboard tables and the CSV. Their evidence is retained only in the <a href="data/highmotion-audit.json">complete 27-run protocol audit</a>, not as leaderboard results. Malformed predictions in the retained cohort remain scored penalties; no such samples were silently dropped.</p>',
              '<h2 id="grt-controls">GRT vs archived and matched controls</h2><p>All three promoted educational candidates exceed every contracted Open MOS and Token F1 floor, with 11.43–15.23% fewer patch projections than the matched all-patch route. LLaVA-OneVision 7B failed its MOS gate and is not promoted. These are observed point estimates, not statistical-significance claims. Archived Qwen baselines differ from stronger matched controls: their entire gap must not be attributed to GRT.</p>',
              table(controls, control_columns, "All 12 educational comparison rows; 634 items and eight sampled frames per method"),
              '<p><strong>Not all metrics improve.</strong> Qwen 3B GRT reports mean throughput 1.67264 fps versus 1.74994 for its all-patch control (about 4.42% lower), even though its Open MOS, Token F1 and patch reuse improve. Route31 mean request time is essentially unchanged versus its all-patch control. Throughput is the mean of per-request sampled-frame rates, not total frames divided by total campaign time, and these single historical runs do not establish repeated speedup. Patch ratios measure patch projection only, not end-to-end FLOPs.</p>',
              '<p><a href="data/leaderboard.js">Original immutable 32-row snapshot</a> remains byte-identical to the released numerical bundle. The new protocol audit does not rewrite that historical evidence or present incompatible rows as matched wins. Full dataset access, GPU/judge reproduction and manuscript alignment remain separate release checks.</p></main></body></html>']
    records = []
    for cohort, rows in [("educational_published", frozen["tracks"]["lpm"]), ("highmotion_aligned_preview1000", aligned), ("educational_grt_controls", controls)]:
        records.extend(dict(row, cohort=cohort) for row in rows)
    fields = ["cohort"] + sorted({key for row in records for key in row if key != "cohort"})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return {
        "leaderboard.html": "\n".join(pieces) + "\n",
        "data/leaderboard-complete.csv": buffer.getvalue(),
        "data/public-audit.js": "/* Generated by scripts/build_audit_page.py; source: public-audit.json. */\nwindow.DIVE_PUBLIC_AUDIT = " + json.dumps(audit, ensure_ascii=False, indent=2) + ";\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated output without writing")
    args = parser.parse_args()
    for relative, text in build_outputs().items():
        path = ROOT / relative
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit("Generated file is stale: " + relative)
        else:
            path.write_text(text)
    print("Complete HTML, CSV and browser audit payload are consistent.")


if __name__ == "__main__":
    main()
