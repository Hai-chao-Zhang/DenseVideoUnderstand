"""Verify the public numerical evidence and rebuild the frozen leaderboard.

This is an artifact-level reproduction, not fresh model inference. It needs
only Python and PyYAML; no dataset, GPU, credentials, or model downloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

from tools.densevideo.build_leaderboard import TASK_COLUMNS, TASK_LABELS, markdown_table, rank_rows

SNAPSHOT_SHA256 = "c88492ec535a3bc4f47fe9e91f66bc050a857f757cd2c478db598e889d9dc1b1"
PROVENANCE_SHA256 = "ea2d25f47a10662dbc064931e7a9b6fd6bf782e2ec000c371314ac0c2651edcf"
CSV_SHA256 = "dc269f911d346672f097cbfd69497a7875f895a868afa3977494a7fbbf07597e"
TRACKS = {"lpm": "densevideo", "highmotion": "densevideo_highmotion"}
SITE_FIELDS = {
    "model": "display_name", "recompute_ratio": "mean_recompute_ratio",
    "transition_acc": "grid_transition_acc",
    "reference_recompute_ratio": "reference_patch_compute_ratio",
    "effective_fps": "mean_effective_fps", "throughput_fps": "mean_throughput_fps",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(bundle, name, digest):
    path = bundle / name
    require(path.parent.resolve() == bundle.resolve(), "Invalid bundle filename")
    value = path.read_bytes()
    require(hashlib.sha256(value).hexdigest() == digest, f"SHA256 mismatch: {name}")
    return value


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), f"Non-finite {label}")
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14),
            f"Numeric mismatch: {label}: {actual} != {expected}")


def rebuild_markdown(rows):
    # Preserve the historical timestamp and labels, even as new tasks use paper names.
    lines = ["# DIVE-Bench Leaderboard", "",
             "- generated_at: `2026-08-20T06:20:21`",
             "- include_closed_source: `true`", "- source_artifacts: `21`",
             "- open_mos_judge: `Qwen/Qwen3-VL-32B-Instruct`", ""]
    for task in TRACKS.values():
        selected = rank_rows([dict(row) for row in rows if row["task"] == task], task)
        lines.extend([f"## {TASK_LABELS[task]}", "",
                      markdown_table(selected, TASK_COLUMNS[task]), ""])
    return "\n".join(lines).encode("utf-8")


def verify_bundle(bundle, *, website=False):
    bundle = Path(bundle)
    provenance = json.loads(checked_bytes(bundle, "provenance.json", PROVENANCE_SHA256))
    site_bytes = checked_bytes(bundle, "leaderboard.js", SNAPSHOT_SHA256)
    matched = re.fullmatch(r"\s*(?:/\*.*?\*/\s*)?window\.DIVE_LEADERBOARD\s*=\s*(\{.*\});\s*",
                           site_bytes.decode("utf-8"), re.DOTALL)
    require(matched is not None, "Unexpected website data format")
    site = json.loads(matched.group(1))
    checked_bytes(bundle, "leaderboard.csv", CSV_SHA256)
    rows = read_rows(bundle / "leaderboard.csv")
    require(len(rows) == 32, "Expected exactly 32 leaderboard rows")
    by_key = {(row["task"], row["method"]): row for row in rows}
    require(len(by_key) == len(rows), "Duplicate leaderboard method/task")
    for track, task in TRACKS.items():
        require(len(site["tracks"][track]) == provenance["snapshot_rows"][track],
                f"Wrong row count: {track}")
        for entry in site["tracks"][track]:
            row = by_key[(task, entry["method"])]
            for key, value in entry.items():
                actual = row.get(SITE_FIELDS.get(key, key), "")
                if value is None:
                    require(actual == "", f"Unexpected CSV value: {entry['method']}/{key}")
                elif isinstance(value, (float, int)):
                    # The legacy High-Motion exporter rounded its entire table to 6g.
                    converted = float(actual)
                    if track == "highmotion":
                        converted = float(f"{converted:.6g}")
                    close(converted, value, f"CSV/site {entry['method']}/{key}")
                else:
                    require(actual == value, f"CSV/site mismatch: {entry['method']}/{key}")

    identity = provenance["identity"]
    checked_bytes(bundle, identity["file"], identity["sha256"])
    identities = read_rows(bundle / identity["file"])
    require(len(identities) == 634 and {int(r["doc_id"]) for r in identities} == set(range(634)),
            "Expected 634 unique document identities")
    for row in identities:
        require(re.fullmatch(r"[0-9a-f]{64}", row["identity_sha256"]), "Invalid identity hash")

    result_families = []
    for family in provenance["families"]:
        checked_bytes(bundle, family["numeric_file"], family["numeric_sha256"])
        numeric = read_rows(bundle / family["numeric_file"])
        grouped = defaultdict(list)
        for row in numeric:
            grouped[int(row["method_index"])].append(row)
        methods = family["methods"]
        require(set(grouped) == {m["method_index"] for m in methods}, "Unexpected method indices")
        means = {}
        for method in methods:
            samples = grouped[method["method_index"]]
            require(len(samples) == 634 and {int(r["doc_id"]) for r in samples} == set(range(634)),
                    f"Incomplete/duplicate sample IDs: {method['method']}")
            means[method["method"]] = {}
            for metric, ceiling in (("open_mos", 5), ("token_f1", 1)):
                values = [float(r[metric]) for r in samples]
                require(all(math.isfinite(v) and 0 <= v <= ceiling for v in values),
                        f"Invalid {metric}: {method['method']}")
                mean = math.fsum(values) / len(values)
                close(mean, method[metric], f"{method['method']}/{metric}")
                means[method["method"]][metric] = mean
        candidate = means[family["selected_method"]]
        for metric in ("open_mos", "token_f1"):
            require(all(candidate[metric] > value[metric] for method, value in means.items()
                        if method != family["selected_method"]),
                    f"Quality gate failed: {family['family']}/{metric}")
        published = by_key[("densevideo", family["selected_method"])]
        require(float(published["open_mos"]) == float(f"{candidate['open_mos']:.6g}"),
                "Published Open MOS must use six significant digits")
        close(float(published["token_f1"]), candidate["token_f1"], "Published Token F1")
        telemetry = family["candidate_telemetry"]
        for key in ("mean_recompute_ratio", "reference_patch_compute_ratio",
                    "mean_effective_fps", "mean_throughput_fps"):
            close(float(published[key]), telemetry[key], f"Published {key}")
        ratio = telemetry["total_recomputed_patches"] / telemetry["total_reference_orig_patches"]
        close(ratio, telemetry["reference_patch_compute_ratio"], "Patch compute ratio")
        require(0 < ratio <= 0.98, "Patch compute gate failed")
        result_families.append({"profile": family["family"], **candidate,
                                "reference_patch_compute_ratio": ratio, "point_gates": "pass"})

    hm = provenance["highmotion"]
    checked_bytes(bundle, hm["numeric_file"], hm["numeric_sha256"])
    numeric = read_rows(bundle / hm["numeric_file"])
    require(len(numeric) == 1000 and {int(r["doc_id"]) for r in numeric} == set(range(1000)),
            "Expected exactly 1000 High-Motion preview samples")
    published = by_key[("densevideo_highmotion", hm["method"])]
    for metric in ("grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1"):
        mean = math.fsum(float(r[metric]) for r in numeric) / len(numeric)
        close(mean, hm["results"][metric + ",none"], f"High-Motion {metric}")
        close(float(published[metric]), mean, f"Published High-Motion {metric}")

    rebuilt = rebuild_markdown(rows)
    require(rebuilt == (bundle / "leaderboard.md").read_bytes(), "Markdown rebuild differs from snapshot")
    if website:
        request = Request(provenance["website_data"], headers={
            "User-Agent": "DIVE-Bench-release-audit/0.1", "Accept": "application/javascript"})
        with urlopen(request, timeout=30) as response:
            current = response.read()
        require(hashlib.sha256(current).hexdigest() == SNAPSHOT_SHA256,
                "Live website differs from the pinned 2026-08-20 release")
    return {"status": "verified", "verification": "historical artifacts; no fresh inference",
            "website_checked": website, "leaderboard_rows": 32, "lpm_samples_per_method": 634,
            "highmotion_preview_samples": 1000, "families": result_families}, rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("release/2026-08-20"),
                        help="Release bundle directory (relative to current directory by default)")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are refused")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--website", action="store_true", help="Also compare live website bytes (network)")
    args = parser.parse_args(argv)
    if args.verify_only and args.output:
        parser.error("--verify-only and --output are mutually exclusive")
    try:
        report, markdown = verify_bundle(args.bundle, website=args.website)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "leaderboard.md").write_bytes(markdown)
            for filename in ("leaderboard.csv", "leaderboard.js"):
                (args.output / filename).write_bytes((args.bundle / filename).read_bytes())
            (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
