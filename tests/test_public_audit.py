import csv
import hashlib
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from test_site_contract import _leaderboard_data, _render_ui


ROOT = Path(__file__).resolve().parents[1]


class PublicAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _leaderboard_data()
        cls.audit = json.loads((ROOT / "data/public-audit.json").read_text())
        # The generated browser payload adds protocol-specific ranks.
        browser = (ROOT / "data/public-audit.js").read_text()
        cls.browser = json.loads(browser.split("window.DIVE_PUBLIC_AUDIT = ", 1)[1].strip().removesuffix(";"))
        cls.html = (ROOT / "leaderboard.html").read_text()

    def test_generated_assets_are_reproducible_and_javascript_valid(self):
        subprocess.run(["python", "scripts/build_audit_page.py", "--check"], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["node", "--check", "data/public-audit.js"], cwd=ROOT, check=True, capture_output=True)

    def test_frozen_32_rows_remain_byte_identical(self):
        self.assertEqual(hashlib.sha256((ROOT / "data/leaderboard.js").read_bytes()).hexdigest(), "c88492ec535a3bc4f47fe9e91f66bc050a857f757cd2c478db598e889d9dc1b1")
        self.assertEqual(sum(map(len, self.frozen["tracks"].values())), 32)

    def test_all_quality_controls_and_mixed_throughput_outcome_are_visible(self):
        for family in self.audit["families"]:
            rows = family["methods"]
            self.assertEqual(len(rows), 4)
            candidate = next(row for row in rows if row["role"] == "candidate")
            for control in rows:
                self.assertEqual(control["samples"], 634)
                self.assertGreaterEqual(candidate["open_mos"], control["open_mos"])
                self.assertGreaterEqual(candidate["token_f1"], control["token_f1"])
                self.assertIn(control["method"], self.html)
            self.assertLess(candidate["patch_ratio"], 1)
        qwen = next(f for f in self.audit["families"] if f["family"] == "qwen3")
        candidate = next(r for r in qwen["methods"] if r["role"] == "candidate")
        all_patch = next(r for r in qwen["methods"] if r["label"] == "Matched all-patch control")
        self.assertLess(candidate["throughput_fps"], all_patch["throughput_fps"])
        self.assertIn("Not all metrics improve", self.html)
        rendered = _render_ui(self.frozen, self.browser)
        self.assertEqual(rendered["comparisonHtml"].count("<tr>"), 12)

    def test_full_highmotion_archive_has_two_explicit_protocol_cohorts(self):
        rows = self.browser["highmotion_additional"]
        self.assertEqual(len(rows), 27)
        aligned = [row for row in rows if row["protocol_status"] == "aligned_preview"]
        historical = [row for row in rows if row["protocol_status"] != "aligned_preview"]
        self.assertEqual(len(aligned), 18)
        self.assertEqual(len(historical), 9)
        self.assertEqual([row["rank"] for row in aligned], list(range(1, 19)))
        self.assertTrue(all(row["rank"] is None for row in historical))
        self.assertTrue(all(row["samples"] == 1000 for row in rows))
        self.assertIn("grt_llava_ov_0_5b", {row["method"] for row in historical})

    def test_browser_only_ranks_aligned_highmotion_rows(self):
        aligned = _render_ui(self.frozen, self.browser, "highmotion")
        self.assertEqual(aligned["tableHtml"].count("<tr>"), 18)
        self.assertNotIn("grt_llava_ov_0_5b", aligned["tableHtml"])
        historical = _render_ui(self.frozen, self.browser, "highmotion_historical")
        self.assertEqual(historical["tableHtml"].count("<tr>"), 9)
        self.assertNotIn('data-sort="rank"', historical["tableHead"])
        self.assertIn("Unranked", historical["rankingRule"])
        self.assertNotIn("Grid Accuracy leader", historical["summaryHtml"])
        self.assertIn("literal-text overlap scorer", (ROOT / "app.js").read_text())

    def test_missing_audit_asset_never_promotes_legacy_rows(self):
        aligned = _render_ui(self.frozen, None, "highmotion")
        self.assertNotIn("grt_llava_ov_0_5b", aligned["tableHtml"])
        self.assertIn("Showing 0 of 0 methods", aligned["tableCount"])
        historical = _render_ui(self.frozen, None, "highmotion_historical")
        self.assertEqual(historical["tableHtml"].count("<tr>"), 3)
        self.assertNotIn('data-sort="rank"', historical["tableHead"])

    def test_protocol_notes_and_comparison_labels_escape_markup(self):
        injected = deepcopy(self.browser)
        injected["highmotion_additional"][0]["protocol_note"] = '<img src=x onerror="alert(1)">'
        injected["families"][0]["methods"][0]["label"] = "<script>alert(1)</script>"
        rendered = _render_ui(self.frozen, injected, "highmotion")
        self.assertIn("&lt;img", rendered["tableHtml"])
        self.assertNotIn("<img", rendered["tableHtml"])
        self.assertIn("&lt;script", rendered["comparisonHtml"])
        self.assertNotIn("<script", rendered["comparisonHtml"])

    def test_complete_html_and_csv_preserve_every_method_and_cohort(self):
        with (ROOT / "data/leaderboard-complete.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in {row["cohort"] for row in rows}}
        self.assertEqual(counts, {"educational_published": 29, "highmotion_aligned_preview1000": 18, "highmotion_historical_unaligned": 9, "educational_grt_controls": 12})
        for row in rows:
            self.assertIn(row["method"], self.html)
        self.assertNotIn("<script", self.html)
        self.assertIn("213/1,000", self.html)
        self.assertIn("148 clips", self.html)

    def test_public_page_links_real_release_and_records_limits(self):
        index = (ROOT / "index.html").read_text()
        for stale in ("Code forthcoming", "pending code release", "not public yet"):
            self.assertNotIn(stale, index)
        self.assertIn("pull/1686", index)
        self.assertIn("pull/1521", index)
        self.assertIn("not a fresh GPU rerun", index)
        self.assertIn("leaderboard.html", index)


if __name__ == "__main__":
    unittest.main()
