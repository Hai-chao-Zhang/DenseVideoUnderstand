from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sync_leaderboard as sync  # noqa: E402


_CONTRACT_LOCATION = os.environ.get("DIVE_QWEN3_ARCHIVED_CONTRACT", "").strip()
REAL_CONTRACT = Path(_CONTRACT_LOCATION).expanduser() if _CONTRACT_LOCATION else None
REAL_COMPLETION = REAL_CONTRACT.parent / "completed.json" if REAL_CONTRACT else None


def telemetry_row(
    *, recomputed: int, original: int, effective: float, wall: float, throughput: float
) -> str:
    ratio = recomputed / original
    return (
        f"[FPS_STATS] sampled_frames=8 effective_fps={effective:.6f}\n"
        "[DENSE_METRICS] sampled_frames=8 "
        f"effective_fps={effective:.6f} wall_time_s={wall:.6f} "
        f"throughput_fps={throughput:.6f} recomputed_patches={recomputed} "
        f"orig_patches={original} reference_orig_patches={original} "
        f"recompute_ratio={ratio:.6f} "
        f"patch_projection_recompute_ratio={ratio:.6f} "
        f"patch_projection_compute_ratio_vs_reference={ratio:.6f}\n"
    )


class Qwen3RecoveryReleaseTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("DIVE_RUN_ARCHIVED_CAMPAIGN_TESTS") == "1",
        "external historical campaign check is opt-in (DIVE_RUN_ARCHIVED_CAMPAIGN_TESTS=1)",
    )
    def test_real_prepared_contract_deeply_verifies_without_writes(self) -> None:
        self.assertIsNotNone(REAL_CONTRACT, "DIVE_QWEN3_ARCHIVED_CONTRACT is required when archival checks are enabled")
        self.assertTrue(REAL_CONTRACT.is_file(), "Configured Qwen3 contract is not a readable file")
        before = sync.sha256_file(REAL_CONTRACT)
        contract = json.loads(REAL_CONTRACT.read_text(encoding="utf-8"))
        verified = sync._validate_qwen3_recovery_contract(
            REAL_CONTRACT.resolve(),
            contract_sha=before,
            recovery_fingerprint=contract["recovery_fingerprint"],
            campaign_fingerprint=contract["campaign_fingerprint"],
            recovery_driver_sha=contract["recovery_driver"]["sha256"],
        )
        self.assertEqual(
            verified["recovery_fingerprint"], contract["recovery_fingerprint"]
        )
        for method, row in contract["legacy"]["reused_methods"].items():
            summary_path = Path(row["artifacts"]["summary"]["path"])
            summary = sync._summary_row(
                summary_path,
                family="qwen3",
                method=method,
                expected_samples=634,
                expected_result=Path(row["artifacts"]["result"]["path"]),
            )
            sync._validate_recovery_raw_telemetry(
                Path(row["artifacts"]["driver_log"]["path"]),
                method=method,
                summary=summary,
            )
        self.assertEqual(sync.sha256_file(REAL_CONTRACT), before)

    def test_recovery_schema_is_qwen3_only_and_screen_is_never_releaseable(self) -> None:
        self.assertIn(
            sync.QWEN3_RECOVERY_COMPLETION_SCHEMA,
            sync.MULTIFAMILY_CONTRACTS["qwen3"]["completion_schema"],
        )
        self.assertNotIn(
            sync.QWEN3_RECOVERY_COMPLETION_SCHEMA,
            sync.MULTIFAMILY_CONTRACTS["qwen7"]["completion_schema"],
        )
        self.assertNotIn(
            "qwen7_cap48_next_screen_completed_v1",
            sync.MULTIFAMILY_CONTRACTS["qwen7"]["completion_schema"],
        )

    def test_recovery_canonical_fingerprint_detects_contract_mutation(self) -> None:
        payload = {
            "schema_version": 1,
            "protocol": sync.QWEN3_RECOVERY_PROTOCOL,
            "recovery_fingerprint": "0" * 64,
            "expected_samples": 634,
        }
        fingerprint = sync._canonical_recovery_fingerprint(payload)
        payload["recovery_fingerprint"] = fingerprint
        self.assertEqual(sync._canonical_recovery_fingerprint(payload), fingerprint)
        payload["expected_samples"] = 633
        self.assertNotEqual(sync._canonical_recovery_fingerprint(payload), fingerprint)

    def test_raw_telemetry_requires_every_row_and_reproduces_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="site-recovery-telemetry-") as raw:
            log = Path(raw) / "run.log"
            log.write_text(
                telemetry_row(
                    recomputed=5,
                    original=10,
                    effective=0.2,
                    wall=2,
                    throughput=4,
                )
                + telemetry_row(
                    recomputed=2,
                    original=10,
                    effective=0.4,
                    wall=4,
                    throughput=2,
                )
                + "[RUN_META] run_status=success return_code=0\n",
                encoding="utf-8",
            )
            summary = {
                "run_log": str(log),
                "mean_effective_fps": "0.3",
                "mean_wall_time_s": "3",
                "mean_throughput_fps": "3",
                "mean_recomputed_patches": "3.5",
                "mean_orig_patches": "10",
                "total_recomputed_patches": "7",
                "total_orig_patches": "20",
                "total_reference_orig_patches": "20",
                "mean_recompute_ratio": "0.35",
                "reference_patch_compute_ratio": "0.35",
                "mean_patch_projection_recompute_ratio": "0.35",
                "mean_patch_projection_compute_ratio_vs_reference": "0.35",
            }
            with patch.object(sync, "MULTIFAMILY_EXPECTED_SAMPLES", 2):
                sync._validate_recovery_raw_telemetry(
                    log.resolve(), method="synthetic", summary=summary
                )
                summary["mean_recompute_ratio"] = "0.3501"
                with self.assertRaisesRegex(sync.SyncError, "raw/summary telemetry"):
                    sync._validate_recovery_raw_telemetry(
                        log.resolve(), method="synthetic", summary=summary
                    )

    def test_strict_integer_rejects_fractional_sample_count(self) -> None:
        with self.assertRaisesRegex(sync.SyncError, "must be an integer"):
            sync._integer(634.9, "samples")
        with self.assertRaisesRegex(sync.SyncError, "non-finite|must be an integer"):
            sync._integer(math.nan, "samples")

    def test_recovery_doc_id_normalization_accepts_integer_zero_fail_closed(self) -> None:
        self.assertEqual(sync._normalized_recovery_doc_id(0, "integer"), "0")
        self.assertEqual(sync._normalized_recovery_doc_id("0", "string"), "0")
        for value in (None, "", "   ", False, 0.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(sync.SyncError, "doc_id"):
                    sync._normalized_recovery_doc_id(value, "invalid")

    @unittest.skipUnless(
        os.environ.get("DIVE_RUN_ARCHIVED_CAMPAIGN_TESTS") == "1",
        "external historical campaign check is opt-in (DIVE_RUN_ARCHIVED_CAMPAIGN_TESTS=1)",
    )
    def test_real_completion_passes_site_independent_consumer(self) -> None:
        self.assertIsNotNone(REAL_COMPLETION, "DIVE_QWEN3_ARCHIVED_CONTRACT is required when archival checks are enabled")
        self.assertTrue(REAL_COMPLETION.is_file(), "Configured Qwen3 completion is not a readable file")
        completed = sync._load_hashed_json(REAL_COMPLETION, "qwen3 completion")
        artifacts = completed["artifacts"]

        def pair(name: str) -> tuple[Path, str]:
            return sync._recovery_descriptor(
                artifacts[name],
                base_directory=REAL_COMPLETION.parent,
                label=f"qwen3 {name}",
            )

        selection_pair = pair("combined_selection")
        matrix_pair = pair("mos_matrix_csv")
        matrix_jsonl_pair = pair("mos_matrix_jsonl")
        selection = sync._load_hashed_json(selection_pair[0], "qwen3 selection")
        with matrix_pair[0].open("r", encoding="utf-8", newline="") as handle:
            csv_doc_zero = next(
                row for row in csv.DictReader(handle) if row.get("doc_id") == "0"
            )
        jsonl_doc_zero = next(
            row
            for row in (
                json.loads(line)
                for line in matrix_jsonl_pair[0]
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
            if row.get("method") == csv_doc_zero["method"]
            and row.get("sample_id") == csv_doc_zero["sample_id"]
        )
        self.assertEqual(csv_doc_zero["doc_id"], "0")
        self.assertIs(type(jsonl_doc_zero["doc_id"]), int)
        self.assertEqual(jsonl_doc_zero["doc_id"], 0)
        winner_marker = sync._load_hashed_json(
            pair(f"method_{selection['selected_method']}")[0],
            "qwen3 winner marker",
        )
        summary_pair = sync._recovery_descriptor(
            winner_marker["summary"],
            base_directory=REAL_COMPLETION.parent,
            label="qwen3 winner summary",
        )
        sync._validate_multifamily_completion(
            REAL_COMPLETION,
            entry={
                "family": "qwen3",
                "selected_method": selection["selected_method"],
                "completion_schema": sync.QWEN3_RECOVERY_COMPLETION_SCHEMA,
                "campaign_fingerprint": completed["campaign_fingerprint"],
                "revisions": {
                    "model": "66285546d2b821cf421d4f5eb2576359d3770cd3",
                    "dataset": "5cc61a045c8e5e95d1d9c87e22ccd0f699575aea",
                    "judge": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
                },
            },
            selection=selection,
            selection_pair=selection_pair,
            matrix_pair=matrix_pair,
            summary_pair=summary_pair,
            judge={
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "revision": "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
                "fingerprint": "64c644b2bd2696677afd3507a6bd6efa03e902d712c0e55e26ab9ab144c0a76d",
            },
        )


if __name__ == "__main__":
    unittest.main()
