from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_dual_release_fixture import write_dual_bundle, write_json  # noqa: E402
from scripts import qwen_dual_release as dual  # noqa: E402
from scripts.sync_leaderboard import (  # noqa: E402
    MULTIFAMILY_CONTRACTS,
    QWEN3_RECOVERY_COMPLETION_SCHEMA,
    QWEN7_FLOOR_FULL_COMPLETION_SCHEMA,
    SyncError,
    _validate_multifamily_completion,
    validate_strict_multifamily_selection,
)


JUDGE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
JUDGE_REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"
JUDGE_FINGERPRINT = "64c644b2bd2696677afd3507a6bd6efa03e902d712c0e55e26ab9ab144c0a76d"


class QwenDualReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="site-qwen-dual-")
        self.bundle = write_dual_bundle(
            Path(self.temporary.name) / "dual",
            judge_model=JUDGE_MODEL,
            judge_revision=JUDGE_REVISION,
            judge_fingerprint=JUDGE_FINGERPRINT,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evidence(self, family: str) -> dict:
        return dual.validate_completion(
            self.bundle["completion"],
            family=family,
            judge_model=JUDGE_MODEL,
            judge_revision=JUDGE_REVISION,
            judge_fingerprint=JUDGE_FINGERPRINT,
        )

    def validate_site_binding(self, family: str, evidence: dict | None = None) -> None:
        evidence = evidence or self.evidence(family)
        entry = {
            "family": family,
            "selected_method": evidence["selected_method"],
            "completion_schema": dual.COMPLETION_SCHEMA,
            "campaign_fingerprint": evidence["campaign_fingerprint"],
            "revisions": {
                "model": evidence["model_revision"],
                "dataset": evidence["dataset_revision"],
                "judge": JUDGE_REVISION,
            },
        }

        def pair(descriptor: dict[str, str]) -> tuple[Path, str]:
            return Path(descriptor["path"]), descriptor["sha256"]

        _validate_multifamily_completion(
            self.bundle["completion"],
            entry=entry,
            selection=evidence["selection_payload"],
            selection_pair=pair(evidence["selection"]),
            matrix_pair=pair(evidence["open_mos_matrix"]),
            summary_pair=pair(evidence["summary_csv"]),
            judge={
                "model": JUDGE_MODEL,
                "revision": JUDGE_REVISION,
                "fingerprint": JUDGE_FINGERPRINT,
            },
        )

    def test_site_dispatch_validates_both_atomic_family_views(self) -> None:
        self.validate_site_binding("qwen3")
        self.validate_site_binding("qwen7")
        self.assertEqual(
            MULTIFAMILY_CONTRACTS["qwen3"]["completion_schema"],
            frozenset(
                {
                    "followup_parallel_completed_v1",
                    dual.COMPLETION_SCHEMA,
                    QWEN3_RECOVERY_COMPLETION_SCHEMA,
                }
            ),
        )
        self.assertEqual(
            MULTIFAMILY_CONTRACTS["qwen7"]["completion_schema"],
            frozenset(
                {
                    "followup_parallel_completed_v1",
                    dual.COMPLETION_SCHEMA,
                    QWEN7_FLOOR_FULL_COMPLETION_SCHEMA,
                }
            ),
        )

    def test_site_dual_selection_uses_exact_dual_controls(self) -> None:
        evidence = self.evidence("qwen3")
        candidate = {
            "method": evidence["selected_method"],
            "open_mos": Decimal("2.0"),
            "token_f1": Decimal("0.04"),
            "recompute_ratio": Decimal("0.8"),
            "reference_recompute_ratio": Decimal("0.8"),
        }
        selected, methods = validate_strict_multifamily_selection(
            evidence["selection_payload"],
            candidate,
            family="qwen3",
            base_method="qwen2_5_vl_3b",
            expected_samples=634,
            completion_schema=dual.COMPLETION_SCHEMA,
        )
        self.assertEqual(selected, dual.FAMILY_SPECS["qwen3"]["candidate"])
        self.assertEqual(set(methods), set(evidence["methods"]))

    def test_site_dispatch_rejects_rehashed_nonhard_all634_report(self) -> None:
        evidence = self.evidence("qwen7")
        completion = json.loads(self.bundle["completion"].read_text(encoding="utf-8"))
        report_path = Path(completion["complement_report"]["path"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["families"]["qwen7"]["all_634_is_sole_hard_quality_gate"] = False
        write_json(report_path, report)
        completion["complement_report"] = dual.descriptor(report_path)
        write_json(self.bundle["completion"], completion)

        with self.assertRaisesRegex(SyncError, "hard-gate semantics"):
            self.validate_site_binding("qwen7", evidence)


if __name__ == "__main__":
    unittest.main()
