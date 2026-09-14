import csv
import hashlib
import importlib.util
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from test_site_contract import _leaderboard_data, _render_ui


ROOT = Path(__file__).resolve().parents[1]


class PublicAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = _leaderboard_data()
        cls.audit = json.loads((ROOT / "data/public-audit.json").read_text())
        cls.evidence = json.loads((ROOT / "data/highmotion-audit.json").read_text())
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

    def test_full_27_run_evidence_is_preserved_but_payload_contains_only_18_eligible(self):
        rows = self.browser["highmotion_additional"]
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(self.audit["highmotion_additional"]), 18)
        self.assertEqual(len(self.evidence["rows"]), 27)
        aligned = [row for row in self.evidence["rows"] if row["rank_eligible"]]
        historical = [row for row in self.evidence["rows"] if not row["rank_eligible"]]
        self.assertEqual(len(historical), 9)
        self.assertEqual({row["method"] for row in rows}, {row["method"] for row in aligned})
        self.assertEqual([row["rank"] for row in rows], list(range(1, 19)))
        self.assertTrue(all(row["samples"] == 1000 and row["rank_eligible"] is True and row["protocol_status"] == "aligned_preview" for row in rows))
        self.assertIn("grt_llava_ov_0_5b", {row["method"] for row in historical})
        self.assertEqual(hashlib.sha256((ROOT / "data/highmotion-audit.json").read_bytes()).hexdigest(), "f2358a0ae2f61ae15d0e03436c640fed128b26cf591db3a9e97e668aa562683d")

    def test_browser_only_ranks_aligned_highmotion_rows(self):
        aligned = _render_ui(self.frozen, self.browser, "highmotion")
        self.assertEqual(aligned["tableHtml"].count("<tr>"), 18)
        self.assertNotIn("grt_llava_ov_0_5b", aligned["tableHtml"])
        for row in self.evidence["rows"]:
            if not row["rank_eligible"]:
                self.assertNotIn(row["method"], aligned["tableHtml"])
        unknown = _render_ui(self.frozen, self.browser, "highmotion_historical")
        self.assertEqual(unknown["tableHtml"].count("<tr>"), 29)
        self.assertNotIn("grt_llava_ov_0_5b", unknown["tableHtml"])
        self.assertNotIn('id="tab-highmotion-historical"', (ROOT / "index.html").read_text())

    def test_missing_audit_asset_never_promotes_legacy_rows(self):
        aligned = _render_ui(self.frozen, None, "highmotion")
        self.assertNotIn("grt_llava_ov_0_5b", aligned["tableHtml"])
        self.assertIn("Showing 0 of 0 methods", aligned["tableCount"])
        for payload in ({}, {"highmotion_additional": []}):
            rendered = _render_ui(self.frozen, payload, "highmotion")
            self.assertIn("Showing 0 of 0 methods", rendered["tableCount"])
            self.assertNotIn("grt_llava_ov_0_5b", rendered["tableHtml"])

    def test_browser_rejects_noneligible_or_wrong_split_overlay_rows(self):
        poisoned = deepcopy(self.browser)
        poisoned["highmotion_additional"] += [row for row in self.evidence["rows"] if not row["rank_eligible"]]
        rendered = _render_ui(self.frozen, poisoned, "highmotion")
        self.assertEqual(rendered["tableHtml"].count("<tr>"), 18)
        for key, value in (("rank_eligible", False), ("protocol_status", "historical_unaligned"), ("samples", 3243)):
            with self.subTest(key=key):
                poisoned = deepcopy(self.browser)
                for row in poisoned["highmotion_additional"]:
                    row[key] = value
                rendered = _render_ui(self.frozen, poisoned, "highmotion")
                self.assertIn("Showing 0 of 0 methods", rendered["tableCount"])

    def test_builder_has_no_second_numeric_truth_and_checks_generated_audit(self):
        spec = importlib.util.spec_from_file_location("audit_builder", ROOT / "scripts/build_audit_page.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        expected = builder.build_outputs()
        original_read = Path.read_text
        cases = []
        for key, value in (("rank_eligible", False), ("protocol_status", "historical_unaligned"), ("samples", 3243), ("ordered_identity_sha256", "0" * 64)):
            changed = deepcopy(self.audit)
            changed["highmotion_additional"][0][key] = value
            cases.append((key, changed))
        empty = deepcopy(self.audit)
        empty["highmotion_additional"] = []
        cases.append(("empty", empty))
        missing = deepcopy(self.audit)
        del missing["highmotion_additional"]
        cases.append(("missing", missing))
        duplicate = deepcopy(self.audit)
        duplicate["highmotion_additional"][0] = deepcopy(duplicate["highmotion_additional"][1])
        cases.append(("duplicate", duplicate))
        for label, changed in cases:
            with self.subTest(case=label):
                def substituted_read(path, *args, **kwargs):
                    if path == ROOT / "data/public-audit.json":
                        return json.dumps(changed)
                    return original_read(path, *args, **kwargs)
                with patch.object(Path, "read_text", substituted_read):
                    # Browser JSON is generated output, never a numeric input.
                    self.assertEqual(builder.build_outputs(), expected)
                    with self.assertRaises(SystemExit) as error:
                        builder.main(["--check"])
                    self.assertEqual(error.exception.code, 1)

    def test_protocol_notes_and_comparison_labels_escape_markup(self):
        injected = deepcopy(self.browser)
        injected["highmotion_additional"][0]["protocol_note"] = '<img src=x onerror="alert(1)">'
        injected["families"][0]["methods"][0]["label"] = "<script>alert(1)</script>"
        rendered = _render_ui(self.frozen, injected, "highmotion")
        self.assertIn("&lt;img", rendered["tableHtml"])
        self.assertNotIn("<img", rendered["tableHtml"])
        self.assertIn("&lt;script", rendered["comparisonHtml"])
        self.assertNotIn("<script", rendered["comparisonHtml"])

    def test_complete_html_and_csv_include_only_47_eligible_results_and_12_controls(self):
        with (ROOT / "data/leaderboard-complete.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in {row["cohort"] for row in rows}}
        self.assertEqual(counts, {"educational_published": 29, "highmotion_aligned_preview1000": 18, "educational_grt_controls": 12})
        self.assertEqual(len(rows), 59)
        self.assertEqual(len(rows[0]), 32)
        main = [row for row in rows if row["cohort"] != "educational_grt_controls"]
        self.assertEqual(len({(row["cohort"], row["method"]) for row in main}), 47)
        for row in rows:
            self.assertIn(row["method"], self.html)
        self.assertNotIn("<script", self.html)
        self.assertEqual(self.html.count("<tbody>"), 3)
        self.assertIn("Complete protocol-screened leaderboard", self.html)
        self.assertNotIn("grt_llava_ov_0_5b", self.html)
        self.assertNotIn("highmotion-historical", self.html)
        hm_html = self.html.split('<h2 id="highmotion-aligned">', 1)[1].split('<h2 id="grt-controls">', 1)[0]
        hm_ids = {row["method"] for row in rows if row["cohort"] == "highmotion_aligned_preview1000"}
        for row in self.evidence["rows"]:
            if not row["rank_eligible"]:
                self.assertNotIn(row["method"], hm_ids)
                self.assertNotIn(row["method"], hm_html)
        for page in (self.html, (ROOT / "index.html").read_text()):
            self.assertNotIn("0.10125", page)
            self.assertIn("data/highmotion-audit.json", page)

    def test_public_page_links_real_release_and_records_limits(self):
        index = (ROOT / "index.html").read_text()
        for stale in ("Code forthcoming", "pending code release", "not public yet"):
            self.assertNotIn(stale, index)
        self.assertIn("pull/1686", index)
        self.assertIn("pull/1521", index)
        self.assertIn("not a fresh GPU rerun", index)
        self.assertIn("leaderboard.html", index)

    def test_paper_scope_dataset_access_and_visual_inputs_are_explicit(self):
        index = (ROOT / "index.html").read_text()
        self.assertIn("arXiv version covers the earlier educational scope", index)
        self.assertIn("paper/ECCV_Dense_Video_Understanding.pdf", index)
        self.assertIn("Educational dataset (gated)", index)
        self.assertIn("educational source videos", index)
        self.assertIn("not an audio-input protocol", index)
        self.assertNotIn("Read, listen", index)

    def test_complete_reproduction_uses_new_code_pin_but_keeps_manuscript_pin(self):
        index = (ROOT / "index.html").read_text()
        app = (ROOT / "app.js").read_text()
        code_pin = "e59a708043131271f42f7715cf11b322730554a1"
        paper_pin = "2a79fcce2707b1eb74648a5ed135c469b17eb4e1"
        for source in (index, app):
            self.assertIn("git checkout " + code_pin, source)
            self.assertIn("build_complete_leaderboard --verify-only", source)
            self.assertIn("build_complete_leaderboard --output outputs/leaderboard-complete", source)
            self.assertIn("47 screened results + 12 comparison rows", source)
        self.assertIn("blob/" + paper_pin + "/paper/ECCV_Dense_Video_Understanding.pdf", index)
        self.assertEqual(index.count(paper_pin), 1)
        self.assertNotIn(paper_pin, app)
        self.assertIn("47 results plus 12 educational GRT comparison rows", index)

    def test_external_archive_defaults_do_not_contain_private_machine_paths(self):
        for filename, variable in (
            ("test_qwen3_recovery_release.py", "DIVE_QWEN3_ARCHIVED_CONTRACT"),
            ("test_release_v2.py", "DIVE_LLAVA7_ARCHIVED_FAILURE_ROOT"),
        ):
            source = (ROOT / "tests" / filename).read_text()
            self.assertNotIn("/work/nvme/", source)
            self.assertNotIn("/u/yli8/", source)
            self.assertIn(variable, source)
            self.assertIn("is required when archival checks are enabled", source)


if __name__ == "__main__":
    unittest.main()
