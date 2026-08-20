import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_leaderboard.py"
sys.path.insert(0, str(ROOT))

from scripts.sync_leaderboard import (  # noqa: E402
    LEGACY_OPEN_MOS_INVENTORY_METHODS,
    LEGACY_OPEN_MOS_METHODS,
    LEGACY_OPEN_MOS_OMITTED_METHODS,
    SyncError,
    _inspect_multifamily_matrix,
    _open_mos_means_match,
    _site_row,
    _validate_multifamily_summary,
    build_multifamily_release_payload,
    build_release_payload,
    load_site_payload,
    render_site_payload,
    rerank,
    validate_multifamily_release_binding,
    validate_legacy_open_mos_release_binding,
    validate_strict_multifamily_selection,
)


WINNER = "grt_llava_ov_0_5b_verified"
ROUTE31_WINNER = "grt_llava_onevision_0_5b_hf_route31_t0001"
RUN_FINGERPRINT = "d" * 64
JUDGE_REVISION = "e" * 40
JUDGE_FINGERPRINT = "f" * 64
REAL_ROUTE31_SELECTION = Path(
    os.environ.get(
        "DIVE_ROUTE31_SELECTION",
        "/work/nvme/bdqf/william/charles/dive_grt_quality_outputs/"
        "quality_20260819_062149/hf_route31_full/selection.json",
    )
)
REAL_ROUTE31_RELEASE = Path(
    os.environ.get(
        "DIVE_ROUTE31_RELEASE",
        "/tmp/dive-route31-release-dryrun.sTZ4FV/release-27x3-no-legacy-mos",
    )
)
WINNER_VALUES = {
    "open_mos": "0.216719",
    "token_f1": "0.0240879",
    "cer": "0.9376500000000001",
    "wer": "0.93756",
    "exact_match": "0",
    "mean_recompute_ratio": "0.8123456789012345",
    "reference_patch_compute_ratio": "0.7345678901234567",
    "mean_effective_fps": "0.00324489123456789",
    "mean_throughput_fps": "5.179849123456789",
}


# Freeze the 27-row v1 release input in the test suite.  Production
# ``data/leaderboard.js`` is now a 29-row v2 snapshot, so using it as a source
# fixture and then appending synthetic winners silently double-counts the v2
# winners.  These are the compact, release-authoritative fields from the legacy
# 27-row leaderboard; the helper below materializes the site schema from them.
LEGACY_LPM_BASELINE = (
    (
        11,
        "LLaVA-OneVision Qwen2 7B",
        "llava_onevision_original",
        "open",
        "0.367508",
        "0.01752016317982999",
        "1.0405626588277064",
        "1.039818796076794",
    ),
    (
        12,
        "Qwen3-VL 32B Instruct",
        "qwen3_vl_32b",
        "open",
        "0.173502",
        "0.01846755449340384",
        "1.0523939762097037",
        "1.0508510729252385",
    ),
    (
        5,
        "Qwen3-VL 2B Instruct",
        "qwen3_vl_2b",
        "open",
        "1.13407",
        "0.03314477298861916",
        "1.0455027913245973",
        "1.0365009880881166",
    ),
    (
        7,
        "Qwen3-VL 4B Instruct",
        "qwen3_vl_4b",
        "open",
        "1.13091",
        "0.03486060433625551",
        "1.0559297830448988",
        "1.0511873568663221",
    ),
    (
        2,
        "Qwen3-VL 8B Instruct",
        "qwen3_vl_8b",
        "open",
        "1.29811",
        "0.036039834008412965",
        "1.0588887332730432",
        "1.0519802408339594",
    ),
    (
        9,
        "Qwen2-VL 2B Instruct",
        "qwen2_vl_2b",
        "open",
        "0.790221",
        "0.02009328088724416",
        "1.0369238650622847",
        "1.0334005714792545",
    ),
    (
        6,
        "Qwen2.5-VL 3B Instruct",
        "qwen2_5_vl_3b",
        "open",
        "1.13249",
        "0.03433883131964184",
        "1.043925743144367",
        "1.033987739698653",
    ),
    (
        3,
        "Qwen2.5-VL 7B Instruct",
        "qwen2_5_vl_7b",
        "open",
        "1.28707",
        "0.03478459844603571",
        "1.0462430278543418",
        "1.0397471201356405",
    ),
    (
        10,
        "Qwen2.5-VL 32B Instruct",
        "qwen2_5_vl_32b",
        "open",
        "0.498423",
        "0.02502756241563325",
        "1.0544758077551875",
        "1.0485573638968613",
    ),
    (
        8,
        "Qwen2.5-VL 72B Instruct",
        "qwen2_5_vl_72b",
        "open",
        "0.94164",
        "0.02686277110574796",
        "1.059336941355189",
        "1.047885420247722",
    ),
    (
        13,
        "LLaVA-OneVision Qwen2 0.5B",
        "llava_onevision_0_5b",
        "open",
        "0.116719",
        "0.014087875254888579",
        "1.0376504189479812",
        "1.0375605573912878",
    ),
    (
        14,
        "GRT (LLaVA-OneVision Qwen2 0.5B)",
        "grt_llava_ov_0_5b",
        "open",
        "0.0867508",
        "0.010325861687397282",
        "1.0556364257393647",
        "1.0500822110119228",
    ),
    (
        27,
        "InternVL2.5 2B",
        "internvl2_5_2b",
        "open",
        "",
        "0.011929782390987282",
        "1.0172852136789552",
        "1.0203659095386695",
    ),
    (
        15,
        "LLaVA-OneVision 1.5 8B Instruct",
        "llava_onevision_1_5_8b",
        "open",
        "",
        "0.03497477424334058",
        "1.0724296597671577",
        "1.0568565590454495",
    ),
    (
        16,
        "Qwen2-VL 7B Instruct",
        "qwen2_vl_7b",
        "open",
        "",
        "0.03050962641631123",
        "1.0530886301628082",
        "1.0478847813325716",
    ),
    (
        23,
        "InternVL2.5 8B",
        "internvl2_5_8b",
        "open",
        "",
        "0.018853306384938217",
        "1.057382352168392",
        "1.057194212650538",
    ),
    (
        25,
        "InternVL2.5 1B",
        "internvl2_5_1b",
        "open",
        "",
        "0.017842716799282916",
        "1.0314727804633639",
        "1.033601478031424",
    ),
    (
        24,
        "InternVL2.5 4B",
        "internvl2_5_4b",
        "open",
        "",
        "0.017919950359298137",
        "1.0205119702954106",
        "1.0236172586536032",
    ),
    (
        26,
        "VideoLLaMA3 2B",
        "videollama3_2b",
        "open",
        "",
        "0.01584763764836983",
        "1.0244956136416596",
        "1.0268283128884375",
    ),
    (
        20,
        "VideoLLaMA3 7B",
        "videollama3_7b",
        "open",
        "",
        "0.02493691744497152",
        "1.0510580591302532",
        "1.0497489981549928",
    ),
    (
        18,
        "InternVL3 1B",
        "internvl3_1b",
        "open",
        "",
        "0.02716220358046363",
        "1.062001013014241",
        "1.0606025485342636",
    ),
    (
        22,
        "InternVL3 2B",
        "internvl3_2b",
        "open",
        "",
        "0.022309056166492085",
        "1.0340814107002563",
        "1.0357447841537042",
    ),
    (
        19,
        "InternVL3 8B",
        "internvl3_8b",
        "open",
        "",
        "0.02503030545885109",
        "1.0704625839346382",
        "1.0660084753836587",
    ),
    (
        17,
        "Phi-4 Multimodal Instruct",
        "phi4_multimodal",
        "open",
        "",
        "0.027528686731500093",
        "1.0300793296372575",
        "1.0276186962365996",
    ),
    (
        21,
        "LongVA 7B",
        "longva_7b",
        "open",
        "",
        "0.024661617889657445",
        "1.0467750893746477",
        "1.0424050061066892",
    ),
    (
        1,
        "Gemini 3.1 Pro Preview (gemini-3.1-pro-preview)",
        "gemini-3.1-pro-preview",
        "gemini",
        "1.60726",
        "0.15963300061702457",
        "1.26881212090543",
        "1.2416608170285448",
    ),
    (
        4,
        "Gemini 3.6 Flash (gemini-3.6-flash)",
        "gemini-3.6-flash",
        "gemini",
        "1.25237",
        "0.08146113759889202",
        "1.1016801978435222",
        "1.0679640573224862",
    ),
)


def legacy_site_payload():
    """Return a self-contained v1 site payload for legacy and synthetic tests."""

    lpm = []
    for (
        rank,
        model,
        method,
        source,
        open_mos,
        token_f1,
        cer,
        wer,
    ) in LEGACY_LPM_BASELINE:
        raw = {
            "task": "densevideo",
            "rank": str(rank),
            "display_name": model,
            "method": method,
            "samples": "634",
            "source": source,
            "open_mos": open_mos,
            "token_f1": token_f1,
            "cer": cer,
            "wer": wer,
            "exact_match": "0",
            "mean_recompute_ratio": "",
            "reference_patch_compute_ratio": "",
            "mean_effective_fps": "",
            "mean_throughput_fps": "",
        }
        if method == "grt_llava_ov_0_5b":
            raw.update(
                {
                    "mean_recompute_ratio": "0.2493027728706626",
                    "mean_effective_fps": "0.8001304384858123",
                    "mean_throughput_fps": "24.73282587066247",
                }
            )
        lpm.append(_site_row(raw, method, 634, strict_telemetry=False))
    lpm = rerank(lpm, "open_mos")
    if len(lpm) != 27 or len({row["method"] for row in lpm}) != 27:
        raise AssertionError("legacy LPM fixture must contain 27 unique methods")

    highmotion = [
        {
            "rank": 1,
            "model": "GRT (LLaVA-OneVision Qwen2 0.5B)",
            "method": "grt_llava_ov_0_5b",
            "samples": 1000,
            "grid_acc": Decimal("0.10125"),
            "grid_ade": Decimal("0.712142"),
            "grid_fde": Decimal("0.686488"),
            "transition_acc": Decimal("0.0148571"),
            "token_f1": Decimal("0.181281"),
            "effective_fps": Decimal("0.988599"),
            "source": "open",
        },
        {
            "rank": 2,
            "model": "Gemini 3.6 Flash (gemini-3.6-flash)",
            "method": "gemini-3.6-flash",
            "samples": 1000,
            "grid_acc": Decimal("0.0885007"),
            "grid_ade": Decimal("0.502699"),
            "grid_fde": Decimal("0.571509"),
            "transition_acc": Decimal("0.863788"),
            "token_f1": 0,
            "effective_fps": None,
            "source": "gemini",
        },
        {
            "rank": 3,
            "model": "Gemini 3.1 Pro Preview (gemini-3.1-pro-preview)",
            "method": "gemini-3.1-pro-preview",
            "samples": 1000,
            "grid_acc": Decimal("0.066019"),
            "grid_ade": Decimal("0.779814"),
            "grid_fde": Decimal("0.849685"),
            "transition_acc": Decimal("0.603178"),
            "token_f1": 0,
            "effective_fps": None,
            "source": "gemini",
        },
    ]
    return {
        "generatedAt": "2026-08-17T00:00:00",
        "sourceArtifacts": 3,
        "openMosJudge": "Qwen/Qwen3-VL-32B-Instruct",
        "primaryGrtMethod": "grt_llava_ov_0_5b",
        "tracks": {"lpm": lpm, "highmotion": highmotion},
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path):
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _legacy_open_mos_fixture(directory, route_entry):
    root = directory / "legacy_open_mos_rejudge_v1"
    fields = (
        "sample_id",
        "doc_id",
        "method",
        "open_mos_correctness",
        "open_mos_score",
        "mos_judge_model",
        "mos_judge_revision",
        "request_fingerprint",
        "judge_fingerprint",
        "error",
        "question",
        "answer",
        "pred",
        "open_mos_review",
    )
    rows_by_method = {}
    for method_index, method in enumerate(LEGACY_OPEN_MOS_METHODS):
        rows_by_method[method] = [
            {
                "sample_id": f"{method}::{index}",
                "doc_id": "" if method.startswith("gemini-") else str(index),
                "method": method,
                "open_mos_correctness": "yes",
                "open_mos_score": str((method_index + index) % 6),
                "mos_judge_model": "Qwen/Qwen3-VL-32B-Instruct",
                "mos_judge_revision": JUDGE_REVISION,
                "request_fingerprint": hashlib.sha256(
                    f"{method}:{index}".encode()
                ).hexdigest(),
                "judge_fingerprint": JUDGE_FINGERPRINT,
                "error": "",
                "question": f"question {index}",
                "answer": f"answer {index}",
                "pred": f"prediction {method_index} {index}",
                "open_mos_review": json.dumps(
                    {"pred": "yes", "score": (method_index + index) % 6},
                    separators=(",", ":"),
                ),
            }
            for index in range(634)
        ]

    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    matrix = root / "mos/matrix.csv"
    write_rows(
        matrix,
        [row for method in LEGACY_OPEN_MOS_METHODS for row in rows_by_method[method]],
    )
    matrix_jsonl = root / "mos/matrix.jsonl"
    matrix_jsonl.write_text("{}\n", encoding="utf-8")
    matrix_markdown = root / "mos/matrix.md"
    matrix_markdown.write_text("# fixture\n", encoding="utf-8")
    input_manifest = root / "provenance/input_manifest.json"
    input_manifest.parent.mkdir(parents=True, exist_ok=True)
    input_manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    methods = {}
    for method in LEGACY_OPEN_MOS_METHODS:
        path = root / "per_method" / f"{method}.csv"
        write_rows(path, rows_by_method[method])
        methods[method] = {**_descriptor(path), "rows": 634}
    judge = {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "revision": JUDGE_REVISION,
        "judge_fingerprint": JUDGE_FINGERPRINT,
    }
    release_merge = {
        "explicit_per_method_methods": list(LEGACY_OPEN_MOS_METHODS),
        "omit_legacy_methods": list(LEGACY_OPEN_MOS_OMITTED_METHODS),
        "route31_open_mos_matrix": route_entry["open_mos_matrix"],
        "route31_selection": route_entry["selection"],
    }
    contract = root / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "campaign": "legacy_open_mos_rejudge_v1",
                "expected_samples": 634,
                "inputs": [],
                "judge": judge,
                "methods": list(LEGACY_OPEN_MOS_INVENTORY_METHODS),
                "release_merge": release_merge,
                "schema_version": 1,
                "scored_methods": list(LEGACY_OPEN_MOS_METHODS),
                "source_files": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = root / "completed.json"
    completed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign": "legacy_open_mos_rejudge_v1",
                "status": "passed",
                "expected_samples": 634,
                "inventory_methods": list(LEGACY_OPEN_MOS_INVENTORY_METHODS),
                "methods": list(LEGACY_OPEN_MOS_METHODS),
                "rows": 9 * 634,
                "judge": judge,
                "contract": _descriptor(contract),
                "scorer_command": ["python", "score_open_mos_matrix.py"],
                "artifacts": {
                    "input_manifest": _descriptor(input_manifest),
                    "matrix_csv": _descriptor(matrix),
                    "matrix_jsonl": _descriptor(matrix_jsonl),
                    "matrix_markdown": _descriptor(matrix_markdown),
                    "per_method": methods,
                },
                "release_builder": {
                    "route31_open_mos_matrix": route_entry["open_mos_matrix"],
                    "route31_selection": route_entry["selection"],
                    "explicit_legacy_mos": methods,
                    "omitted_legacy_methods": list(LEGACY_OPEN_MOS_OMITTED_METHODS),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    binding = {
        "schema_version": 1,
        "campaign": "legacy_open_mos_rejudge_v1",
        "expected_samples": 634,
        "rows": 9 * 634,
        "judge": judge,
        "completion": _descriptor(completed),
        "contract": _descriptor(contract),
        "matrix_csv": _descriptor(matrix),
        "methods": methods,
    }
    sources = {
        (Path(item["path"]).resolve(), item["sha256"]) for item in methods.values()
    }
    by_method = {method: item["path"] for method, item in methods.items()}
    return binding, sources, by_method, completed


def _selection(sample_manifest, open_mos_matrix, selected_summary):
    artifacts = {
        "sample_manifest": _descriptor(sample_manifest),
        "open_mos_matrix": _descriptor(open_mos_matrix),
        f"summary_{WINNER}": _descriptor(selected_summary),
    }
    return {
        "status": "passed",
        "metric": "open_mos",
        "secondary_metric": "token_f1",
        "selected_method": WINNER,
        "selected_score": WINNER_VALUES["open_mos"],
        "selected_secondary_score": WINNER_VALUES["token_f1"],
        "selected_recompute_ratio": WINNER_VALUES["mean_recompute_ratio"],
        "selected_reference_patch_compute_ratio": WINNER_VALUES[
            "reference_patch_compute_ratio"
        ],
        "required_score": "0.116719",
        "secondary_required_score": "0.0140879",
        "max_recompute_ratio": "0.98",
        "max_reference_recompute_ratio": "0.98",
        "passing_candidates": [WINNER],
        "sample_counts": {WINNER: 634},
        "artifact_provenance": {
            "schema_version": 1,
            "run_fingerprint": RUN_FINGERPRINT,
            "sample_manifest": artifacts["sample_manifest"]["path"],
            "sample_manifest_sha256": artifacts["sample_manifest"]["sha256"],
            "open_mos_matrix": artifacts["open_mos_matrix"]["path"],
            "open_mos_matrix_sha256": artifacts["open_mos_matrix"]["sha256"],
            "artifacts": artifacts,
            "revisions": {
                "candidate_model": "model-revision",
                "dataset": "dataset-revision",
                "open_mos_judge": JUDGE_REVISION,
            },
        },
    }


def _winner_row(**updates):
    row = {
        "task": "densevideo",
        "rank": "12",
        "display_name": "GRT (LLaVA-OneVision Qwen2 0.5B, verified)",
        "method": WINNER,
        "samples": "634",
        "source": "open",
        **WINNER_VALUES,
    }
    row.update(updates)
    return row


class LeaderboardSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dive-site-sync-test-")
        self.directory = Path(self.temporary.name)
        self.csv_path = self.directory / "leaderboard.csv"
        self.selection_path = self.directory / "selection.json"
        self.metadata_path = self.directory / "leaderboard_sources.json"
        self.output_path = self.directory / "leaderboard.js"
        self.site_data_path = self.directory / "legacy-leaderboard.js"
        self.sample_manifest = self.directory / "sample_manifest.json"
        self.matrix_path = self.directory / "matrix.csv"
        self.summary_path = self.directory / "winner_summary.csv"
        self.api_summary_path = self.directory / "api_summary.csv"
        self.sample_manifest.write_text('{"samples": 634}\n', encoding="utf-8")
        self.matrix_path.write_text("method,score\nfixture,5\n", encoding="utf-8")
        self.summary_path.write_text("method,samples\nfixture,634\n", encoding="utf-8")
        self.api_summary_path.write_text("Model,#Videos\napi,634\n", encoding="utf-8")
        self.site_data_path.write_text(
            render_site_payload(legacy_site_payload()), encoding="utf-8"
        )
        self.write_release_csv(_winner_row())
        self.write_selection(self.make_selection())
        self.refresh_metadata()

    def make_selection(self):
        return _selection(self.sample_manifest, self.matrix_path, self.summary_path)

    def make_route31_selection(self):
        public = "llava_onevision_0_5b"
        base = "llava_onevision_0_5b_hf_route31_base"
        exact = "llava_onevision_0_5b_hf_route31_exact"
        methods = (public, base, exact, ROUTE31_WINNER)
        artifact_names = {
            "actual_vs_virtual_diagnostic",
            "archived_public_floor",
            "archived_public_result",
            "archived_public_sample",
            "base_exact_control",
            "candidate_route_invariants",
            "diagnostic_mos_matrix",
            "diagnostic_sample",
            "driver_snapshot",
            "evaluation_contract",
            "mos_integrity",
            "offline_evidence",
            "open_mos_matrix",
            "resolved_run_checksums",
            "resolved_runs",
            "runtime",
            "sample_source_checksums",
            "sample_sources",
            "source_snapshot",
            "summary_archived_public",
            "summary_route31_base",
            "summary_route31_candidate",
            "summary_route31_exact",
            "telemetry_contract",
        }
        route_root = self.directory / "route31"
        route_root.mkdir()
        driver = route_root / "driver.sbatch"
        driver.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        source_snapshot = route_root / "source_snapshot.sha256"
        source_snapshot.write_text("fixture source snapshot\n", encoding="utf-8")
        driver_manifest = route_root / "driver.sha256"
        driver_manifest.write_text(
            f"{_sha256(driver)}  {driver.resolve()}\n", encoding="utf-8"
        )
        campaign = hashlib.sha256(
            f"{_sha256(source_snapshot)}\n{_sha256(driver)}\n".encode("utf-8")
        ).hexdigest()
        evaluation = route_root / "evaluation_contract.json"
        evaluation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_fingerprint": campaign,
                    "expected_samples": 634,
                    "expected_question_routes": {"ocr": 317, "subtitle": 317},
                    "generation": {
                        "default_max_new_tokens": 128,
                        "subtitle_max_new_tokens": 31,
                        "temperature": 0,
                    },
                    "video_decode_backend": "pyav_seek",
                    "model_revision": "a" * 40,
                    "dataset_revision": "b" * 40,
                    "judge_revision": JUDGE_REVISION,
                    "evidence_sha256": hashlib.sha256(
                        b"offline_evidence\n"
                    ).hexdigest(),
                    "archived_public_result_sha256": hashlib.sha256(
                        b"archived_public_result\n"
                    ).hexdigest(),
                    "methods": {
                        base: {"role": "base", "threshold": None},
                        exact: {"role": "exact", "threshold": None},
                        ROUTE31_WINNER: {"role": "candidate", "threshold": "0.001"},
                    },
                    "quality_contract": {
                        "gates": ["archived_public", "fresh_controls", "combined"],
                        "metrics": ["open_mos", "token_f1"],
                        "comparison": "strictly_greater",
                        "max_recompute_ratio": 0.98,
                        "max_reference_patch_compute_ratio": 0.98,
                        "mos_batch_size": 8,
                    },
                    "driver": {
                        "path": str(driver.resolve()),
                        "sha256": _sha256(driver),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_paths = {}
        for name in artifact_names:
            if name == "open_mos_matrix":
                path = self.matrix_path
            elif name == "summary_route31_candidate":
                path = self.summary_path
            elif name == "driver_snapshot":
                path = driver_manifest
            elif name == "evaluation_contract":
                path = evaluation
            elif name == "source_snapshot":
                path = source_snapshot
            elif name == "telemetry_contract":
                path = route_root / name
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "passed",
                            "limits": {
                                "max_recompute_ratio": 0.98,
                                "max_reference_patch_compute_ratio": 0.98,
                            },
                            "methods": {
                                base: {"role": "base"},
                                exact: {"role": "exact"},
                                ROUTE31_WINNER: {
                                    "role": "candidate",
                                    "passes_recompute_caps": True,
                                    "effective_max_new_tokens_counts": {
                                        "128": 317,
                                        "31": 317,
                                    },
                                    "question_route_counts": {
                                        "ocr": 317,
                                        "subtitle": 317,
                                    },
                                    "mean_recompute_ratio": WINNER_VALUES[
                                        "mean_recompute_ratio"
                                    ],
                                    "reference_patch_compute_ratio": WINNER_VALUES[
                                        "reference_patch_compute_ratio"
                                    ],
                                    "mean_effective_fps": WINNER_VALUES[
                                        "mean_effective_fps"
                                    ],
                                    "mean_throughput_fps": WINNER_VALUES[
                                        "mean_throughput_fps"
                                    ],
                                },
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                path = route_root / name
                path.write_text(f"{name}\n", encoding="utf-8")
            artifact_paths[name] = path
        gates = {}
        for name in ("archived_public", "fresh_controls"):
            gate = route_root / f"{name}_gate.json"
            baselines = {public} if name == "archived_public" else {base, exact}
            gate.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "selected_method": ROUTE31_WINNER,
                        "metric": "open_mos",
                        "secondary_metric": "token_f1",
                        "passing_candidates": [ROUTE31_WINNER],
                        "candidate_scores": {ROUTE31_WINNER: WINNER_VALUES["open_mos"]},
                        "candidate_secondary_scores": {
                            ROUTE31_WINNER: WINNER_VALUES["token_f1"]
                        },
                        "sample_counts": {
                            method: 634 for method in baselines | {ROUTE31_WINNER}
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gates[name] = _descriptor(gate)
        return {
            "status": "passed",
            "attempt_id": "fixture_0",
            "metric": "open_mos",
            "secondary_metric": "token_f1",
            "selected_method": ROUTE31_WINNER,
            "selected_score": WINNER_VALUES["open_mos"],
            "selected_secondary_score": WINNER_VALUES["token_f1"],
            "selected_recompute_ratio": WINNER_VALUES["mean_recompute_ratio"],
            "selected_reference_patch_compute_ratio": WINNER_VALUES[
                "reference_patch_compute_ratio"
            ],
            "required_score": "0.116719",
            "secondary_required_score": "0.0140879",
            "max_recompute_ratio": "0.98",
            "max_reference_recompute_ratio": "0.98",
            "passing_candidates": [ROUTE31_WINNER],
            "sample_counts": {method: 634 for method in methods},
            "candidate_scores": {ROUTE31_WINNER: WINNER_VALUES["open_mos"]},
            "candidate_secondary_scores": {ROUTE31_WINNER: WINNER_VALUES["token_f1"]},
            "baseline_scores": {
                public: "0.116719",
                base: "0.116719",
                exact: "0.116719",
            },
            "baseline_secondary_scores": {
                public: "0.0140879",
                base: "0.0139814",
                exact: "0.0139814",
            },
            "dual_gate_provenance": gates,
            "artifact_provenance": {
                "schema_version": 1,
                "attempt_id": "fixture_0",
                "campaign_fingerprint": campaign,
                "same_snapshot_fresh_methods": True,
                "archived_floor_is_fresh": False,
                "default_max_new_tokens": 128,
                "subtitle_max_new_tokens": 31,
                "samples": 634,
                "artifacts": {
                    name: _descriptor(path) for name, path in artifact_paths.items()
                },
            },
        }

    def use_route31_selection(self):
        self.write_release_csv(
            _winner_row(
                method=ROUTE31_WINNER,
                display_name="GRT (LLaVA-OneVision Qwen2 0.5B, verified Route31)",
            )
        )
        self.write_selection(self.make_route31_selection())
        self.refresh_metadata()

    def write_selection(self, selection):
        self.selection_path.write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )

    def refresh_metadata(self, *, embed_selection=False):
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        selection_provenance = selection["artifact_provenance"]
        if "campaign_fingerprint" in selection_provenance:
            selection_schema = "hf_route31_full_campaign_v1"
            fingerprint_field = "campaign_fingerprint"
        else:
            selection_schema = "legacy_run_fingerprint_v1"
            fingerprint_field = "run_fingerprint"
        payload = {
            "generated_at": "2026-08-19T12:34:56",
            "summary_csv": [str(self.summary_path.resolve())],
            "open_mos_artifacts": [str(self.matrix_path.resolve())],
            "api_summaries": [str(self.api_summary_path.resolve())],
            "open_mos_judge": "Qwen/Qwen3-VL-32B-Instruct",
            "open_mos_judge_provenance": {
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "revision": JUDGE_REVISION,
                "judge_fingerprint": JUDGE_FINGERPRINT,
            },
            "artifact_provenance": {
                "schema_version": 1,
                "leaderboard_csv": _descriptor(self.csv_path),
                "source_artifacts": {
                    "summary_csv": [_descriptor(self.summary_path)],
                    "open_mos_artifacts": [_descriptor(self.matrix_path)],
                    "api_summaries": [_descriptor(self.api_summary_path)],
                },
                "verified_grt_selection": {
                    **_descriptor(self.selection_path),
                    "selection_schema": selection_schema,
                    fingerprint_field: selection_provenance[fingerprint_field],
                    "selected_method": selection["selected_method"],
                },
            },
        }
        if embed_selection:
            payload["full_selection"] = selection
        self.metadata_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_csv(self, *rows):
        fieldnames = list(_winner_row())
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def csv_row_from_site(row):
        def value(field):
            item = row.get(field)
            return "" if item is None else str(item)

        return {
            "task": "densevideo",
            "rank": str(row["rank"]),
            "display_name": row["model"],
            "method": row["method"],
            "samples": str(row["samples"]),
            "source": row.get("source", "open"),
            "open_mos": value("open_mos"),
            "token_f1": value("token_f1"),
            "cer": value("cer"),
            "wer": value("wer"),
            "exact_match": value("exact_match"),
            "mean_recompute_ratio": value("recompute_ratio"),
            "reference_patch_compute_ratio": value("reference_recompute_ratio"),
            "mean_effective_fps": value("effective_fps"),
            "mean_throughput_fps": value("throughput_fps"),
        }

    def release_rows(self, *winner_rows):
        original = legacy_site_payload()
        winner_methods = {str(row["method"]) for row in winner_rows}
        retained = [
            dict(row)
            for row in original["tracks"]["lpm"]
            if row["method"] != "grt_llava_ov_0_5b"
            and row["method"] not in winner_methods
        ]
        winner_site_rows = [
            _site_row(
                row,
                str(row["method"]),
                634,
                strict_telemetry=False,
            )
            for row in winner_rows
        ]
        ranked = rerank([*retained, *winner_site_rows], "open_mos")
        rank_by_method = {str(row["method"]): int(row["rank"]) for row in ranked}
        raw_by_method = {str(row["method"]): dict(row) for row in winner_rows}
        result = []
        for row in ranked:
            method = str(row["method"])
            if method in raw_by_method:
                raw = raw_by_method[method]
                # Preserve an explicitly wrong rank so negative tests can
                # exercise the release-wide rank validator.
                if raw.get("rank") in (None, "", "0", 0):
                    raw["rank"] = str(rank_by_method[method])
                result.append(raw)
            else:
                result.append(self.csv_row_from_site(row))
        return result

    def write_release_csv(self, *winner_rows):
        self.write_csv(*self.release_rows(*winner_rows))

    def run_sync(self, *extra, output=None, check=True, refresh_metadata=True):
        if refresh_metadata:
            self.refresh_metadata()
        command = [
            sys.executable,
            str(SCRIPT),
            "--site-data",
            str(self.site_data_path),
            "--leaderboard-csv",
            str(self.csv_path),
            "--selection-json",
            str(self.selection_path),
            "--metadata-json",
            str(self.metadata_path),
            "--output",
            str(output or self.output_path),
            *extra,
        ]
        return subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_verified_winner_replaces_only_lpm_grt_and_preserves_precision(self):
        old = legacy_site_payload()
        result = self.run_sync()
        self.assertIn("status=passed", result.stdout)
        new = load_site_payload(self.output_path)

        self.assertEqual(len(new["tracks"]["lpm"]), 27)
        self.assertEqual(len(new["tracks"]["highmotion"]), 3)
        self.assertEqual(new["tracks"]["highmotion"], old["tracks"]["highmotion"])
        self.assertEqual(new["primaryGrtMethod"], WINNER)
        self.assertEqual(new["generatedAt"], "2026-08-19T12:34:56")
        self.assertEqual(new["sourceArtifacts"], 3)
        self.assertEqual(new["openMosJudge"], "Qwen/Qwen3-VL-32B-Instruct")

        old_methods = {row["method"] for row in old["tracks"]["lpm"]}
        new_methods = {row["method"] for row in new["tracks"]["lpm"]}
        self.assertEqual(
            new_methods,
            old_methods.difference({"grt_llava_ov_0_5b"}).union({WINNER}),
        )
        old_by_method = {row["method"]: row for row in old["tracks"]["lpm"]}
        new_by_method = {row["method"]: row for row in new["tracks"]["lpm"]}
        for method in old_methods.difference({"grt_llava_ov_0_5b"}):
            old_row = old_by_method[method]
            new_row = new_by_method[method]
            for key, value in old_row.items():
                if key != "rank":
                    self.assertEqual(new_row[key], value, f"{method}.{key}")
            for key in ("reference_recompute_ratio", "throughput_fps"):
                if key not in old_row:
                    self.assertIsNone(new_row[key], f"{method}.{key}")

        winner = new_by_method[WINNER]
        expected = {
            "open_mos": WINNER_VALUES["open_mos"],
            "token_f1": WINNER_VALUES["token_f1"],
            "cer": WINNER_VALUES["cer"],
            "wer": WINNER_VALUES["wer"],
            "exact_match": WINNER_VALUES["exact_match"],
            "recompute_ratio": WINNER_VALUES["mean_recompute_ratio"],
            "reference_recompute_ratio": WINNER_VALUES["reference_patch_compute_ratio"],
            "effective_fps": WINNER_VALUES["mean_effective_fps"],
            "throughput_fps": WINNER_VALUES["mean_throughput_fps"],
        }
        for field, value in expected.items():
            self.assertEqual(winner[field], Decimal(value), field)
        self.assertEqual(winner["rank"], 12)
        self.assertEqual(
            [row["rank"] for row in new["tracks"]["lpm"]], list(range(1, 28))
        )
        self.assertIn("Keep numeric values unrounded", self.output_path.read_text())

    def test_single_release_ignores_every_existing_lpm_value(self):
        original = legacy_site_payload()
        corrupted = deepcopy(original)
        corrupted["tracks"]["lpm"] = [
            {
                "rank": 999,
                "model": "untrusted stale row",
                "method": "not_a_release_method",
                "samples": 1,
                "open_mos": 5,
            }
        ]
        release_rows = self.release_rows(_winner_row())
        kwargs = {
            "metadata": {},
            "expected_lpm_count": 27,
            "expected_highmotion_count": 3,
            "generated_at": "2026-08-19T12:34:56",
            "source_artifacts": 3,
            "open_mos_judge": "Qwen/Qwen3-VL-32B-Instruct",
        }
        expected = build_release_payload(
            deepcopy(original), release_rows, self.make_selection(), **kwargs
        )
        actual = build_release_payload(
            corrupted, release_rows, self.make_selection(), **kwargs
        )
        self.assertEqual(actual["tracks"]["lpm"], expected["tracks"]["lpm"])
        self.assertEqual(
            actual["tracks"]["highmotion"], original["tracks"]["highmotion"]
        )

    def test_release_wide_lpm_contract_rejects_count_duplicate_and_samples(self):
        original = legacy_site_payload()
        valid_rows = self.release_rows(_winner_row())
        kwargs = {
            "metadata": {},
            "expected_lpm_count": 27,
            "expected_highmotion_count": 3,
            "generated_at": "2026-08-19T12:34:56",
            "source_artifacts": 3,
            "open_mos_judge": "Qwen/Qwen3-VL-32B-Instruct",
        }
        cases = []
        missing = deepcopy(valid_rows[:-1])
        cases.append((missing, "contains 26 LPM rows"))
        duplicate = deepcopy(valid_rows)
        duplicate[-1]["method"] = duplicate[0]["method"]
        cases.append((duplicate, "duplicate LPM method"))
        incomplete = deepcopy(valid_rows)
        ordinary = next(row for row in incomplete if row["method"] != WINNER)
        ordinary["samples"] = "633"
        cases.append((incomplete, "expected 634"))

        for rows, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(SyncError, message),
            ):
                build_release_payload(
                    deepcopy(original), rows, self.make_selection(), **kwargs
                )

    @unittest.skipUnless(
        REAL_ROUTE31_SELECTION.is_file()
        and (REAL_ROUTE31_RELEASE / "leaderboard.csv").is_file()
        and (REAL_ROUTE31_RELEASE / "leaderboard_sources.json").is_file(),
        "real Route31 dry-run artifacts are not available",
    )
    def test_real_route31_dryrun_replaces_entire_lpm_from_hashed_csv(self):
        output = self.directory / "real-route31.js"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--site-data",
                str(ROOT / "data" / "leaderboard.js"),
                "--leaderboard-csv",
                str(REAL_ROUTE31_RELEASE / "leaderboard.csv"),
                "--selection-json",
                str(REAL_ROUTE31_SELECTION),
                "--metadata-json",
                str(REAL_ROUTE31_RELEASE / "leaderboard_sources.json"),
                "--expected-lpm-count",
                "27",
                "--expected-highmotion-count",
                "3",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("status=passed", result.stdout)
        before = load_site_payload(ROOT / "data" / "leaderboard.js")
        after = load_site_payload(output)
        with (REAL_ROUTE31_RELEASE / "leaderboard.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            release_rows = [
                row for row in csv.DictReader(handle) if row["task"] == "densevideo"
            ]
        release_by_method = {row["method"]: row for row in release_rows}
        after_by_method = {row["method"]: row for row in after["tracks"]["lpm"]}
        self.assertEqual(len(after_by_method), 27)
        self.assertEqual(set(after_by_method), set(release_by_method))
        for method, source in release_by_method.items():
            self.assertEqual(after_by_method[method]["rank"], int(source["rank"]))
            self.assertEqual(after_by_method[method]["model"], source["display_name"])
        self.assertEqual(
            after_by_method[ROUTE31_WINNER]["open_mos"],
            Decimal(release_by_method[ROUTE31_WINNER]["open_mos"]),
        )
        # The old site has Gemini MOS values, while this provenance-safe
        # dry-run intentionally does not. The release CSV must win.
        self.assertIsNone(after_by_method["gemini-3.1-pro-preview"]["open_mos"])
        self.assertEqual(after["tracks"]["highmotion"], before["tracks"]["highmotion"])

    def test_route31_campaign_winner_uses_dual_schema_without_relaxing_legacy(self):
        self.use_route31_selection()

        result = self.run_sync(refresh_metadata=False)

        self.assertIn("status=passed", result.stdout)
        payload = load_site_payload(self.output_path)
        self.assertEqual(payload["primaryGrtMethod"], ROUTE31_WINNER)
        winner = next(
            row for row in payload["tracks"]["lpm"] if row["method"] == ROUTE31_WINNER
        )
        self.assertEqual(
            winner["model"],
            "GRT (LLaVA-OneVision Qwen2 0.5B, verified Route31)",
        )
        self.assertEqual(winner["samples"], 634)

    def test_route31_campaign_rejects_mixed_fingerprint_schemas(self):
        self.use_route31_selection()
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        selection["artifact_provenance"]["run_fingerprint"] = RUN_FINGERPRINT
        self.write_selection(selection)
        self.refresh_metadata()

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "exactly one of run_fingerprint or campaign_fingerprint", result.stderr
        )
        self.assertFalse(self.output_path.exists())

    def test_route31_campaign_rejects_missing_required_artifact(self):
        self.use_route31_selection()
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        selection["artifact_provenance"]["artifacts"].pop("telemetry_contract")
        self.write_selection(selection)
        self.refresh_metadata()

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete or expanded artifact set", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_route31_campaign_rejects_fingerprint_not_bound_to_source_and_driver(self):
        self.use_route31_selection()
        selection = json.loads(self.selection_path.read_text(encoding="utf-8"))
        selection["artifact_provenance"]["campaign_fingerprint"] = "a" * 64
        self.write_selection(selection)
        self.refresh_metadata()

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not bound to source + driver", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_route31_campaign_rejects_release_telemetry_drift(self):
        self.use_route31_selection()
        self.write_release_csv(
            _winner_row(
                method=ROUTE31_WINNER,
                display_name="GRT (LLaVA-OneVision Qwen2 0.5B, verified Route31)",
                mean_throughput_fps="9.0",
            )
        )
        self.refresh_metadata()

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "mean_throughput_fps differs from authenticated telemetry", result.stderr
        )
        self.assertFalse(self.output_path.exists())

    def test_output_is_byte_deterministic_and_valid_javascript(self):
        second = self.directory / "leaderboard-second.js"
        self.run_sync()
        self.run_sync(output=second)
        self.assertEqual(self.output_path.read_bytes(), second.read_bytes())
        subprocess.run(
            ["node", "--check", str(self.output_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_selection_can_be_embedded_in_metadata_with_explicit_release_metadata(self):
        embedded_path = self.directory / "embedded.json"
        self.refresh_metadata(embed_selection=True)
        embedded_path.write_bytes(self.metadata_path.read_bytes())
        output = self.directory / "embedded-output.js"
        command = [
            sys.executable,
            str(SCRIPT),
            "--site-data",
            str(self.site_data_path),
            "--leaderboard-csv",
            str(self.csv_path),
            "--metadata-json",
            str(embedded_path),
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        payload = load_site_payload(output)
        self.assertEqual(payload["primaryGrtMethod"], WINNER)
        self.assertEqual(payload["sourceArtifacts"], 3)

    def test_failed_quality_contract_does_not_touch_output(self):
        self.write_release_csv(_winner_row(open_mos="0.100000"))
        failed_selection = self.make_selection()
        failed_selection["selected_score"] = "0.100000"
        self.write_selection(failed_selection)
        self.output_path.write_text("do-not-touch\n", encoding="utf-8")
        result = self.run_sync(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not clear the full-gate requirement", result.stderr)
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "do-not-touch\n")

    def test_rank_disagreement_fails_closed(self):
        self.write_release_csv(_winner_row(rank="13"))
        result = self.run_sync(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source rank 13 disagrees with release rank 12", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_missing_reference_or_throughput_telemetry_is_rejected(self):
        for field in ("reference_patch_compute_ratio", "mean_throughput_fps"):
            with self.subTest(field=field):
                self.write_release_csv(_winner_row(**{field: ""}))
                result = self.run_sync(check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("missing numeric value", result.stderr)
                self.assertFalse(self.output_path.exists())

    def test_tampered_selection_evidence_is_rejected_by_sha(self):
        self.refresh_metadata()
        self.matrix_path.write_text("method,score\nfixture,0\n", encoding="utf-8")

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selection artifact 'open_mos_matrix' SHA-256", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_release_csv_must_match_metadata_path_and_sha(self):
        self.refresh_metadata()
        with self.csv_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release leaderboard_csv SHA-256", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_selection_file_must_match_release_metadata_sha(self):
        self.refresh_metadata()
        with self.selection_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release verified_grt_selection SHA-256", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_release_metadata_run_fingerprint_must_match_selection(self):
        self.refresh_metadata()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["artifact_provenance"]["verified_grt_selection"]["run_fingerprint"] = (
            "c" * 64
        )
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run_fingerprint does not match", result.stderr)
        self.assertFalse(self.output_path.exists())

    def test_selected_matrix_path_and_sha_must_be_release_source(self):
        self.refresh_metadata()
        replacement = self.directory / "other_matrix.csv"
        replacement.write_bytes(self.matrix_path.read_bytes())
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata["open_mos_artifacts"] = [str(replacement.resolve())]
        metadata["artifact_provenance"]["source_artifacts"]["open_mos_artifacts"] = [
            _descriptor(replacement)
        ]
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Open-MOS matrix is not bound", result.stderr)
        self.assertFalse(self.output_path.exists())

    def write_multifamily_matrix(self, path, methods, score_for):
        fields = (
            "sample_id",
            "doc_id",
            "method",
            "open_mos_score",
            "mos_judge_model",
            "mos_judge_revision",
            "judge_fingerprint",
            "error",
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method in methods:
                for index in range(634):
                    writer.writerow(
                        {
                            "sample_id": f"{method}::{index}",
                            "doc_id": index,
                            "method": method,
                            "open_mos_score": score_for(method, index),
                            "mos_judge_model": "Qwen/Qwen3-VL-32B-Instruct",
                            "mos_judge_revision": JUDGE_REVISION,
                            "judge_fingerprint": JUDGE_FINGERPRINT,
                            "error": "",
                        }
                    )

    def test_legacy_open_mos_binding_is_exact_and_tamper_evident(self):
        route_selection = self.directory / "route-selection.json"
        route_matrix = self.directory / "route-matrix.csv"
        route_selection.write_text('{"status": "passed"}\n', encoding="utf-8")
        route_matrix.write_text("method,score\nroute31,1\n", encoding="utf-8")
        route_entry = {
            "selection": _descriptor(route_selection),
            "open_mos_matrix": _descriptor(route_matrix),
        }
        binding, sources, by_method, completed = _legacy_open_mos_fixture(
            self.directory, route_entry
        )
        judge = {
            "model": "Qwen/Qwen3-VL-32B-Instruct",
            "revision": JUDGE_REVISION,
            "fingerprint": JUDGE_FINGERPRINT,
        }
        verified = validate_legacy_open_mos_release_binding(
            binding,
            metadata_directory=self.directory,
            judge=judge,
            open_mos_sources=sources,
            open_mos_by_method=by_method,
            route31_entry=route_entry,
        )
        self.assertEqual(tuple(verified["methods"]), LEGACY_OPEN_MOS_METHODS)

        with self.assertRaisesRegex(SyncError, "not a release source"):
            validate_legacy_open_mos_release_binding(
                binding,
                metadata_directory=self.directory,
                judge=judge,
                open_mos_sources={next(iter(sources))},
                open_mos_by_method=by_method,
                route31_entry=route_entry,
            )
        with self.assertRaisesRegex(SyncError, "header/judge"):
            validate_legacy_open_mos_release_binding(
                None,
                metadata_directory=self.directory,
                judge=judge,
                open_mos_sources=sources,
                open_mos_by_method=by_method,
                route31_entry=route_entry,
            )

        with completed.open("a", encoding="utf-8") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(SyncError, "SHA-256"):
            validate_legacy_open_mos_release_binding(
                binding,
                metadata_directory=self.directory,
                judge=judge,
                open_mos_sources=sources,
                open_mos_by_method=by_method,
                route31_entry=route_entry,
            )

    @unittest.skipUnless(
        os.environ.get("DIVE_LEGACY_OPEN_MOS_COMPLETED"),
        "set DIVE_LEGACY_OPEN_MOS_COMPLETED for the immutable real-artifact audit",
    )
    def test_real_legacy_open_mos_completion_opt_in(self):
        completed = Path(os.environ["DIVE_LEGACY_OPEN_MOS_COMPLETED"]).resolve()
        payload = json.loads(completed.read_text(encoding="utf-8"))
        methods = payload["artifacts"]["per_method"]
        release = payload["release_builder"]
        judge = {
            "model": payload["judge"]["model"],
            "revision": payload["judge"]["revision"],
            "fingerprint": payload["judge"]["judge_fingerprint"],
        }
        binding = {
            "schema_version": 1,
            "campaign": "legacy_open_mos_rejudge_v1",
            "expected_samples": 634,
            "rows": 9 * 634,
            "judge": {
                "model": judge["model"],
                "revision": judge["revision"],
                "judge_fingerprint": judge["fingerprint"],
            },
            "completion": _descriptor(completed),
            "contract": payload["contract"],
            "matrix_csv": payload["artifacts"]["matrix_csv"],
            "methods": methods,
        }
        verified = validate_legacy_open_mos_release_binding(
            binding,
            metadata_directory=completed.parent,
            judge=judge,
            open_mos_sources={
                (Path(item["path"]).resolve(), item["sha256"])
                for item in methods.values()
            },
            open_mos_by_method={
                method: item["path"] for method, item in methods.items()
            },
            route31_entry={
                "selection": release["route31_selection"],
                "open_mos_matrix": release["route31_open_mos_matrix"],
            },
        )
        self.assertEqual(
            verified["completion"]["sha256"],
            "da4d7bac14c5d77c13d36a2deee9415a7216b7f4657ba2b7e0efd4385d78794f",
        )

    def test_multifamily_mean_accepts_real_76_over_634_rounding(self):
        matrix = self.directory / "ratio-matrix.csv"
        self.write_multifamily_matrix(
            matrix,
            ("base", "winner"),
            lambda method, index: 1 if method == "winner" and index < 76 else 0,
        )
        means = _inspect_multifamily_matrix(
            matrix,
            family="synthetic",
            expected_methods={"base", "winner"},
            expected_samples=634,
            judge={
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "revision": JUDGE_REVISION,
                "fingerprint": JUDGE_FINGERPRINT,
            },
        )
        rounded_selection = Decimal(str(76 / 634))
        self.assertNotEqual(means["winner"], rounded_selection)
        self.assertTrue(
            _open_mos_means_match(
                means,
                {"base": 0, "winner": rounded_selection},
                family="synthetic",
            )
        )

    def test_multifamily_matrix_rejects_docset_drift_and_score_six(self):
        matrix = self.directory / "invalid-matrix.csv"
        self.write_multifamily_matrix(
            matrix,
            ("base", "winner"),
            lambda method, index: 4,
        )
        with matrix.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["doc_id"] = "9999"
        with matrix.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(SyncError, "doc_id sets differ"):
            _inspect_multifamily_matrix(
                matrix,
                family="synthetic",
                expected_methods={"base", "winner"},
                expected_samples=634,
                judge={
                    "model": "Qwen/Qwen3-VL-32B-Instruct",
                    "revision": JUDGE_REVISION,
                    "fingerprint": JUDGE_FINGERPRINT,
                },
            )
        self.write_multifamily_matrix(
            matrix,
            ("base", "winner"),
            lambda method, index: 6 if method == "winner" and index == 0 else 4,
        )
        with self.assertRaisesRegex(SyncError, "outside \\[0,5\\]"):
            _inspect_multifamily_matrix(
                matrix,
                family="synthetic",
                expected_methods={"base", "winner"},
                expected_samples=634,
                judge={
                    "model": "Qwen/Qwen3-VL-32B-Instruct",
                    "revision": JUDGE_REVISION,
                    "fingerprint": JUDGE_FINGERPRINT,
                },
            )

    def test_multifamily_rejects_29_2_header_before_artifact_reads(self):
        manifest = {
            "schema_version": 1,
            "schema": "dive_grt_multifamily_release_v1",
            "benchmark": "DIVE-Bench",
            "task": "densevideo",
            "expected_samples": 634,
            "expected_lpm_rows": 29,
            "expected_highmotion_rows": 2,
            "judge": {
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "revision": JUDGE_REVISION,
                "fingerprint": JUDGE_FINGERPRINT,
            },
            "families": [],
        }
        with self.assertRaisesRegex(SyncError, "header/count contract"):
            validate_multifamily_release_binding(
                manifest,
                manifest_path=self.directory / "manifest.json",
                leaderboard_csv=self.csv_path,
                leaderboard_rows=[],
                metadata={},
                metadata_path=self.metadata_path,
            )

    def test_multifamily_selection_rejects_missing_controls_and_fractional_count(self):
        candidate = {
            "method": "grt_qwen3",
            "open_mos": Decimal("0.3"),
            "token_f1": Decimal("0.2"),
            "recompute_ratio": Decimal("0.75"),
            "reference_recompute_ratio": Decimal("0.7"),
        }
        selection = {
            "status": "passed",
            "metric": "open_mos",
            "secondary_metric": "token_f1",
            "selected_method": "grt_qwen3",
            "selected_score": 0.3,
            "selected_secondary_score": 0.2,
            "selected_recompute_ratio": 0.75,
            "selected_reference_patch_compute_ratio": 0.7,
            "required_score": 0.2,
            "secondary_required_score": 0.1,
            "max_recompute_ratio": 0.98,
            "max_reference_recompute_ratio": 0.98,
            "passing_candidates": ["grt_qwen3"],
            "candidate_scores": {"grt_qwen3": 0.3},
            "candidate_secondary_scores": {"grt_qwen3": 0.2},
            "baseline_scores": {"qwen2_5_vl_3b": 0.2},
            "baseline_secondary_scores": {"qwen2_5_vl_3b": 0.1},
            "sample_counts": {"qwen2_5_vl_3b": 634, "grt_qwen3": 634},
        }
        with self.assertRaisesRegex(SyncError, "exact control/candidate set"):
            validate_strict_multifamily_selection(
                selection,
                candidate,
                family="qwen3",
                base_method="qwen2_5_vl_3b",
                expected_samples=634,
            )
        generic = deepcopy(selection)
        generic["baseline_scores"] = {"base": 0.2}
        generic["baseline_secondary_scores"] = {"base": 0.1}
        generic["sample_counts"] = {"base": 634, "grt_qwen3": 634.9}
        with self.assertRaisesRegex(SyncError, "must be an integer"):
            validate_strict_multifamily_selection(
                generic,
                candidate,
                family="synthetic",
                base_method="base",
                expected_samples=634,
            )

    def test_multifamily_summary_rejects_blank_status(self):
        summary = self.directory / "blank-status.csv"
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("method", "task", "run_status", "samples", "result_json"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "method": "winner",
                    "task": "densevideo",
                    "run_status": "",
                    "samples": 634,
                    "result_json": "result.json",
                }
            )
        with self.assertRaisesRegex(SyncError, "not one complete"):
            _validate_multifamily_summary(
                summary,
                family="synthetic",
                selected_method="winner",
                expected_samples=634,
            )

    def test_multifamily_payload_adds_four_winners_removes_only_legacy_lpm_grt(self):
        original = legacy_site_payload()
        contracts = {
            "route31": (
                "llava_onevision_0_5b",
                "grt_llava_ov_0_5b",
                "0.216719",
                "0.0240879",
            ),
            "llava7": ("llava_onevision_original", None, "0.467508", "0.0275202"),
            "qwen3": ("qwen2_5_vl_3b", None, "1.23249", "0.0443388"),
            "qwen7": ("qwen2_5_vl_7b", None, "1.38707", "0.0447846"),
        }
        verified = []
        source_rows = []
        for family, (base, replaced, open_mos, token_f1) in contracts.items():
            method = f"grt_{family}_verified"
            raw = {
                "task": "densevideo",
                "rank": "0",
                "display_name": f"GRT {family} verified",
                "method": method,
                "samples": "634",
                "open_mos": open_mos,
                "token_f1": token_f1,
                "cer": "0.9",
                "wer": "0.9",
                "exact_match": "0",
                "mean_recompute_ratio": "0.75",
                "reference_patch_compute_ratio": "0.70",
                "mean_effective_fps": "0.02",
                "mean_throughput_fps": "4.5",
                "source": "open",
            }
            candidate = _site_row(raw, method, 634)
            verified.append(
                {
                    "family": family,
                    "base_method": base,
                    "replaces_method": replaced,
                    "selected_method": method,
                    "campaign_fingerprint": family[0] * 64,
                    "candidate": candidate,
                }
            )
            source_rows.append(raw)

        source_rows = self.release_rows(*source_rows)
        source_only_name = "CSV-authoritative Qwen3-VL 8B"
        next(row for row in source_rows if row["method"] == "qwen3_vl_8b")[
            "display_name"
        ] = source_only_name

        payload = build_multifamily_release_payload(
            deepcopy(original),
            source_rows,
            verified,
            metadata={},
            expected_lpm_count=30,
            expected_highmotion_count=3,
            generated_at="2026-08-19T12:34:56",
            source_artifacts=12,
            open_mos_judge="Qwen/Qwen3-VL-32B-Instruct",
        )

        self.assertEqual(len(payload["tracks"]["lpm"]), 30)
        self.assertEqual(
            payload["tracks"]["highmotion"], original["tracks"]["highmotion"]
        )
        self.assertEqual(payload["primaryGrtMethod"], "grt_route31_verified")
        self.assertEqual(
            payload["grtMethods"], [item["selected_method"] for item in verified]
        )
        methods = {row["method"] for row in payload["tracks"]["lpm"]}
        self.assertNotIn("grt_llava_ov_0_5b", methods)
        for item in verified:
            self.assertIn(item["selected_method"], methods)
            self.assertIn(item["base_method"], methods)
        self.assertIn(
            "grt_llava_ov_0_5b",
            {row["method"] for row in payload["tracks"]["highmotion"]},
        )
        self.assertEqual(
            next(
                row
                for row in payload["tracks"]["lpm"]
                if row["method"] == "qwen3_vl_8b"
            )["model"],
            source_only_name,
        )

        corrupted = deepcopy(original)
        corrupted["tracks"]["lpm"] = [{"corrupted": "ignored in full"}]
        from_corrupted_site = build_multifamily_release_payload(
            corrupted,
            source_rows,
            verified,
            metadata={},
            expected_lpm_count=30,
            expected_highmotion_count=3,
            generated_at="2026-08-19T12:34:56",
            source_artifacts=12,
            open_mos_judge="Qwen/Qwen3-VL-32B-Instruct",
        )
        self.assertEqual(from_corrupted_site["tracks"]["lpm"], payload["tracks"]["lpm"])
        self.assertEqual(
            from_corrupted_site["tracks"]["highmotion"],
            original["tracks"]["highmotion"],
        )

    def test_invalid_selection_run_fingerprint_is_rejected(self):
        selection = self.make_selection()
        selection["artifact_provenance"]["run_fingerprint"] = "mutable-tag"
        self.write_selection(selection)
        self.refresh_metadata()

        result = self.run_sync(check=False, refresh_metadata=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run_fingerprint must be a lowercase SHA-256", result.stderr)
        self.assertFalse(self.output_path.exists())


if __name__ == "__main__":
    unittest.main()
