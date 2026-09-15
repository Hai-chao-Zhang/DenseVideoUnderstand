import csv
import hashlib
import importlib.util
import json
import re
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

    def test_full_27_run_evidence_keeps_legacy_hold_separate_from_v2(self):
        rows = self.browser["highmotion_additional"]
        self.assertEqual(rows, [])
        self.assertEqual(self.audit["highmotion_additional"], [])
        for payload in (self.browser, self.audit):
            self.assertEqual(payload["highmotion_release_status"], "held_target_reference_consistency_review")
            self.assertEqual(payload["highmotion_historical_protocol_screened_candidates"], 18)
            self.assertTrue(payload["highmotion_hold_reason"])
        self.assertEqual(len(self.evidence["rows"]), 27)
        aligned = [row for row in self.evidence["rows"] if row["rank_eligible"]]
        historical = [row for row in self.evidence["rows"] if not row["rank_eligible"]]
        self.assertEqual(len(historical), 9)
        self.assertEqual(len(aligned), 18)
        self.assertIn("grt_llava_ov_0_5b", {row["method"] for row in historical})
        self.assertEqual(hashlib.sha256((ROOT / "data/highmotion-audit.json").read_bytes()).hexdigest(), "f2358a0ae2f61ae15d0e03436c640fed128b26cf591db3a9e97e668aa562683d")
        self.assertEqual(self.audit["highmotion_v2"], self.browser["highmotion_v2"])
        corrected = self.audit["highmotion_v2"]
        self.assertIs(corrected["release_integrity_verified"], True)
        self.assertEqual(corrected["release_manifest_sha256"],
                         "1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc")
        self.assertEqual(corrected["benchmark_version"], "highmotion-right-ring-v2")
        self.assertEqual(corrected["scope"], "first-1000-source-rows")
        self.assertEqual(corrected["source_reference_records"], 3243)
        self.assertEqual(corrected["method_count"], 19)
        self.assertIs(corrected["full_source_coverage"], False)

    def test_browser_displays_only_authenticated_highmotion_v2_rows(self):
        aligned = _render_ui(self.frozen, self.browser, "highmotion")
        self.assertIn("Showing 19 of 19 methods", aligned["tableCount"])
        self.assertEqual(aligned["badgeCount"], 1)
        self.assertIn("metric-coverage", aligned["tableHtml"])
        self.assertIn("fixed", aligned["summaryHtml"].lower())
        self.assertNotIn("grt_llava_ov_0_5b", aligned["tableHtml"])
        for row in self.browser["highmotion_v2"]["rows"]:
            self.assertIn(row["method"], aligned["tableHtml"])
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

    def test_browser_rejects_stale_aligned_and_noneligible_overlay_rows(self):
        poisoned = deepcopy(self.browser)
        poisoned.pop("highmotion_v2")
        poisoned["highmotion_additional"] = deepcopy(self.evidence["rows"])
        rendered = _render_ui(self.frozen, poisoned, "highmotion")
        self.assertIn("Showing 0 of 0 methods", rendered["tableCount"])
        for key, value in (("rank_eligible", True), ("protocol_status", "aligned_preview"), ("samples", 1000)):
            with self.subTest(key=key):
                poisoned = deepcopy(self.browser)
                poisoned.pop("highmotion_v2")
                poisoned["highmotion_additional"] = deepcopy(self.evidence["rows"])
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
        for key, value in (("highmotion_release_status", "approved"), ("highmotion_hold_reason", ""), ("highmotion_historical_protocol_screened_candidates", 27)):
            changed = deepcopy(self.audit)
            changed[key] = value
            cases.append((key, changed))
        injected = deepcopy(self.audit)
        injected["highmotion_additional"] = deepcopy(self.evidence["rows"])
        cases.append(("injected_historical_rows", injected))
        missing = deepcopy(self.audit)
        del missing["highmotion_additional"]
        cases.append(("missing", missing))
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

    def test_withheld_notes_cannot_inject_markup_and_comparison_labels_escape_it(self):
        injected = deepcopy(self.browser)
        injected.pop("highmotion_v2")
        injected["highmotion_additional"] = deepcopy(self.evidence["rows"])
        injected["highmotion_additional"][0]["protocol_note"] = '<img src=x onerror="alert(1)">'
        injected["highmotion_hold_reason"] = '<img src=x onerror="alert(1)">'
        injected["families"][0]["methods"][0]["label"] = "<script>alert(1)</script>"
        rendered = _render_ui(self.frozen, injected, "highmotion")
        self.assertNotIn("<img", rendered["tableHtml"])
        self.assertIn("withheld", rendered["tableHtml"])
        self.assertIn("&lt;script", rendered["comparisonHtml"])
        self.assertNotIn("<script", rendered["comparisonHtml"])

    def test_complete_html_and_csv_preserve_educational_and_append_19_v2_rows(self):
        with (ROOT / "data/leaderboard-complete.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in {row["cohort"] for row in rows}}
        self.assertEqual(counts, {"educational_published": 29, "educational_grt_controls": 12,
                                  "highmotion_right_ring_v2_preview1000": 19})
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(row["grid_acc"] == "" for row in rows
                            if row["cohort"] == "educational_published"))
        main = [row for row in rows if row["cohort"] != "educational_grt_controls"]
        self.assertEqual(len({(row["cohort"], row["method"]) for row in main}), 48)
        for row in rows:
            self.assertIn(row["method"], self.html)
        self.assertNotIn("<script", self.html)
        self.assertEqual(self.html.count("<tbody>"), 4)
        self.assertIn("Complete protocol-screened leaderboard", self.html)
        self.assertNotIn("grt_llava_ov_0_5b", self.html)
        self.assertNotIn("highmotion-historical", self.html)
        hm_html = self.html.split('<h2 id="highmotion-aligned">', 1)[1].split('<h2 id="grt-controls">', 1)[0]
        self.assertNotIn("highmotion_aligned_preview1000", counts)
        for row in self.evidence["rows"]:
            self.assertNotIn(row["method"], hm_html)
        corrected_rows = {row["method"]: row for row in rows
                          if row["cohort"] == "highmotion_right_ring_v2_preview1000"}
        for expected in self.audit["highmotion_v2"]["rows"]:
            actual = corrected_rows[expected["method"]]
            self.assertEqual(actual["samples"], "1000")
            self.assertEqual(actual["rank"], "" if expected["rank"] is None else str(expected["rank"]))
            for metric in ("grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1"):
                self.assertEqual(actual[metric], "" if expected[metric] is None else str(expected[metric]))
                self.assertEqual(actual[metric + "_scored_records"], str(expected["metric_scored_records"][metric]))
                self.assertEqual(actual[metric + "_scored_slots_or_edges"], str(expected["metric_scored_slots_or_edges"][metric]))
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

    def test_changed_result_links_are_cache_versioned(self):
        index = (ROOT / "index.html").read_text()
        assets = ("leaderboard.html", "data/leaderboard-complete.csv", "data/public-audit.json")
        manifest_key = self.audit["highmotion_v2"]["release_manifest_sha256"][:16]
        for page, version in ((self.html, "?v=hm-v2-" + manifest_key),
                              (index, "?v=20260915-main-release")):
            links = re.findall(r'href="([^"]+)"', page)
            changed = [link for link in links if any(link.startswith(asset) for asset in assets)]
            self.assertTrue(changed)
            for link in changed:
                self.assertIn(version, link)
                self.assertLess(link.index(version), link.index("#") if "#" in link else len(link))
        for asset in ("app.js", "styles.css", "data/public-audit.js"):
            self.assertIn(asset + "?v=20260915-main-release", index)
        self.assertIn("data/leaderboard.js?v=1f1b5a79c74ff763", index)
        self.assertNotIn("20260914-target-hold", index)

    def test_paper_scope_dataset_access_and_visual_inputs_are_explicit(self):
        index = (ROOT / "index.html").read_text()
        self.assertIn("arXiv version covers the earlier educational scope", index)
        self.assertIn("paper/ECCV_Dense_Video_Understanding.pdf", index)
        self.assertIn("Educational dataset (gated)", index)
        self.assertIn("educational source videos", index)
        self.assertIn("not an audio-input protocol", index)
        self.assertNotIn("Read, listen", index)

    def test_complete_reproduction_uses_current_branch_and_manifest_but_keeps_manuscript_pin(self):
        index = (ROOT / "index.html").read_text()
        app = (ROOT / "app.js").read_text()
        manifest_pin = "1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc"
        paper_pin = "2a79fcce2707b1eb74648a5ed135c469b17eb4e1"
        for source in (index, app):
            self.assertIn("git clone --branch main --single-branch", source)
            self.assertIn(manifest_pin, source)
            self.assertNotIn("git checkout 9ee16af0d03d7f31e726b71f00e4586972afb062", source)
            self.assertNotIn("b720d636624166638a185202fd47d3eb56e17b38", source)
            self.assertIn("build_complete_leaderboard --verify-only", source)
            self.assertIn("build_complete_leaderboard --output outputs/leaderboard-complete", source)
            self.assertIn("29 Educational + 19 High-Motion preview + 12 comparison rows", source)
            self.assertIn("HIGHMOTION_REFERENCE_V2.md", source)
            self.assertIn("HIGHMOTION_V2_REPRODUCTION.md", source)
        self.assertIn("blob/" + paper_pin + "/paper/ECCV_Dense_Video_Understanding.pdf", index)
        self.assertEqual(index.count(paper_pin), 1)
        self.assertNotIn(paper_pin, app)
        self.assertIn("60 CSV records", index)
        self.assertIn("release/2026-09-15/manifest.json", index)
        self.assertIn("release/2026-09-15/numeric_reports.json.gz", index)

    def test_current_code_links_survive_temporary_branch_cleanup(self):
        repository = "https://github.com/Hai-chao-Zhang/DenseVideoUnderstand"
        sources = {name: (ROOT / name).read_text()
                   for name in ("index.html", "app.js", "README.md", "leaderboard.html")}
        for name, source in sources.items():
            with self.subTest(source=name):
                self.assertNotIn("fix/highmotion-target-v2", source)
                if name != "leaderboard.html":
                    self.assertIn("git clone --branch main --single-branch", source)
                links = re.findall(re.escape(repository) + r"/(?:blob|tree)/([^\s\"<>`)]+)", source)
                for target in links:
                    revision = target.split("/", 1)[0]
                    self.assertTrue(revision == "main" or re.fullmatch(r"[0-9a-f]{40}", revision),
                                    "Code links must use main or an immutable commit: " + target)
        index = sources["index.html"]
        for target in ("docs/PUBLICATION_AUDIT.md", "docs/REPRODUCTION.md",
                       "docs/HIGHMOTION_REFERENCE_V2.md", "docs/HIGHMOTION_V2_REPRODUCTION.md",
                       "release/2026-09-15/manifest.json", "release/2026-09-15/numeric_reports.json.gz"):
            self.assertIn(repository + "/blob/main/" + target, index)
        self.assertIn(repository + "/tree/main", index)
        self.assertIn(repository + "/blob/9ee16af0d03d7f31e726b71f00e4586972afb062/"
                      "docs/HIGHMOTION_TARGET_HOLD.md", index)

    def test_current_v2_claims_match_all_five_source_bound_differences(self):
        summary = self.audit["highmotion_v2"]
        comparison = summary["comparison"]
        by_method = {row["method"]: row for row in summary["rows"]}
        baseline = by_method[comparison["baseline_method"]]
        grt = by_method[comparison["grt_method"]]
        self.assertEqual(comparison["baseline_method"], "llava_onevision_0_5b")
        for metric in ("grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1"):
            delta = grt[metric] - baseline[metric]
            oriented = -delta if metric in ("grid_ade", "grid_fde") else delta
            self.assertEqual(comparison["grt_minus_baseline"][metric], delta)
            self.assertEqual(comparison["oriented_improvements"][metric], oriented)
            self.assertEqual(comparison["metric_outperform"][metric], oriented > comparison["point_tolerance"])
        for metric in ("grid_acc", "grid_ade", "grid_fde", "token_f1"):
            self.assertTrue(comparison["metric_outperform"][metric])
        self.assertFalse(comparison["metric_outperform"]["grid_transition_acc"])
        self.assertLess(comparison["grt_minus_baseline"]["grid_transition_acc"], 0)
        index = (ROOT / "index.html").read_text()
        self.assertIn("Transition Accuracy regresses", index)
        self.assertIn("not a full 3,243-record evaluation", index)
        self.assertIn("Exact historical baseline weight revisions and consumed-tensor identity remain unproven", index)
        self.assertIn("private source videos, HDF5 files and derived annotations are not bundled", index)

    def test_qualification_and_comparison_headings_are_educational_on_both_tabs(self):
        index = (ROOT / "index.html").read_text()
        self.assertIn(
            'id="grt-family-qualification-title">Four-family Educational GRT qualification',
            index,
        )
        self.assertIn(
            'id="grt-comparison-title">Educational GRT: does it beat the baselines?',
            index,
        )
        for track in ("lpm", "highmotion"):
            rendered = _render_ui(self.frozen, self.browser, track)
            self.assertEqual(
                rendered["qualification"]["count"], "3 of 4 Educational GRT families promoted"
            )
            self.assertEqual(rendered["comparisonHtml"].count("<tr>"), 12)
            self.assertEqual(rendered["tableHtml"].count("<tr>"), 29 if track == "lpm" else 19)

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
