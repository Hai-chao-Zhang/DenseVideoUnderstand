import json
import re
import subprocess
import unittest
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])


def _leaderboard_data():
    loader = r"""
const fs = require("fs");
const vm = require("vm");
const context = { window: {} };
vm.runInNewContext(fs.readFileSync("data/leaderboard.js", "utf8"), context);
process.stdout.write(JSON.stringify(context.window.DIVE_LEADERBOARD));
"""
    result = subprocess.run(
        ["node", "-e", loader],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _render_ui(dataset, audit=None, track=None):
    loader = r"""
const fs = require("fs");
const vm = require("vm");

class Element {
  constructor(tagName, id) {
    this.tagName = String(tagName || "div").toUpperCase();
    this.id = id || "";
    this.children = [];
    this.attributes = {};
    this.className = "";
    this.hidden = false;
    this.innerHTML = "";
    this._text = "";
    this.style = {};
    this.value = "";
    this.classList = { toggle() {}, remove() {} };
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join("");
  }
  appendChild(child) { this.children.push(child); return child; }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] || null; }
  addEventListener() {}
  querySelectorAll() { return []; }
  focus() {}
  click() {}
  select() {}
}

const ids = [
  "model-search", "source-filter", "leader-summary", "table-count",
  "ranking-rule", "leaderboard-caption", "metric-definitions",
  "grt-family-qualification", "grt-family-qualification-count",
  "grt-family-qualification-list", "lpm-method-count",
  "source-artifact-count", "open-mos-judge", "snapshot-date",
  "snapshot-date-long", "footer-sync-date", "active-code", "code-filename",
  "code-panel", "grt-comparison-body"
];
const elements = {};
ids.forEach((id) => { elements[id] = new Element("div", id); });
elements["source-filter"].value = "all";
const tableHead = new Element("thead", "table-head");
const tableBody = new Element("tbody", "table-body");
const table = new Element("table", "leaderboard-table");
table.querySelector = (selector) => selector === "thead" ? tableHead : tableBody;
elements["leaderboard-table"] = table;

const document = {
  body: new Element("body", "body"),
  createElement: (tagName) => new Element(tagName),
  execCommand: () => false,
  getElementById: (id) => elements[id] || null,
  querySelector: () => null,
  querySelectorAll: () => []
};
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const window = { DIVE_LEADERBOARD: input.dataset, DIVE_PUBLIC_AUDIT: input.audit, isSecureContext: false };
const context = { window, document, navigator: {}, clearTimeout, setTimeout };
const source = fs.readFileSync("app.js", "utf8").replace("  installLeaderboard();", "  window.selectTrackForTest = selectTrack;\n  installLeaderboard();");
vm.runInNewContext(source, context);
if (input.track) window.selectTrackForTest(input.track);

function tagsBelow(node) {
  return node.children.reduce(
    (tags, child) => tags.concat([child.tagName], tagsBelow(child)),
    []
  );
}
const qualificationList = elements["grt-family-qualification-list"];
process.stdout.write(JSON.stringify({
  badgeCount: (tableBody.innerHTML.match(/model-badge grt/g) || []).length,
  summaryHtml: elements["leader-summary"].innerHTML,
  methodCount: elements["lpm-method-count"].textContent,
  qualification: {
    count: elements["grt-family-qualification-count"].textContent,
    hidden: elements["grt-family-qualification"].hidden,
    items: qualificationList.children.map((item) => ({
      status: item.attributes["data-status"],
      text: item.textContent
    })),
    tags: tagsBelow(qualificationList)
  },
  tableCount: elements["table-count"].textContent,
  tableHtml: tableBody.innerHTML,
  tableHead: tableHead.innerHTML,
  comparisonHtml: elements["grt-comparison-body"].innerHTML,
  rankingRule: elements["ranking-rule"].textContent
}));
"""
    result = subprocess.run(
        ["node", "-e", loader],
        cwd=ROOT,
        check=True,
        capture_output=True,
        input=json.dumps({"dataset": dataset, "audit": audit, "track": track}),
        text=True,
    )
    return json.loads(result.stdout)


class SiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "index.html").read_text(encoding="utf-8")
        cls.app = (ROOT / "app.js").read_text(encoding="utf-8")
        cls.data = _leaderboard_data()

    def test_javascript_is_syntactically_valid(self):
        for relative_path in ("app.js", "data/leaderboard.js"):
            subprocess.run(
                ["node", "--check", relative_path],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_html_ids_are_unique(self):
        parser = _IdCollector()
        parser.feed(self.html)
        duplicates = sorted({value for value in parser.ids if parser.ids.count(value) > 1})
        self.assertEqual(duplicates, [])

    def test_primary_grt_method_is_data_driven_and_present(self):
        method = self.data["primaryGrtMethod"]
        lpm_methods = {row["method"] for row in self.data["tracks"]["lpm"]}
        self.assertIn(method, lpm_methods)
        self.assertIn(
            "dataset.primaryGrtMethod ? [dataset.primaryGrtMethod] : []", self.app
        )

    def test_grt_badge_uses_only_the_explicit_method_array(self):
        self.assertIn("grtMethodIds().indexOf(method) !== -1", self.app)
        self.assertNotIn('row.method.indexOf("grt_")', self.app)

    def test_sorting_restores_focus_after_rebuilding_the_header(self):
        self.assertIn(
            "tableHead.querySelector('[data-sort=\"' + key + '\"]')", self.app
        )
        self.assertIn("if (replacement) replacement.focus();", self.app)

    def test_legacy_payload_without_grt_methods_falls_back_to_primary(self):
        legacy = deepcopy(self.data)
        legacy.pop("grtMethods", None)
        legacy.pop("grtFamilies", None)

        rendered = _render_ui(legacy)

        self.assertEqual(rendered["badgeCount"], 1)
        self.assertIn("1 verified profile", rendered["summaryHtml"])
        primary = next(
            row
            for row in legacy["tracks"]["lpm"]
            if row["method"] == legacy["primaryGrtMethod"]
        )
        self.assertIn(primary["model"], rendered["summaryHtml"])

    def test_explicit_empty_grt_method_array_does_not_badge_primary(self):
        v2_without_promotions = deepcopy(self.data)
        v2_without_promotions["grtMethods"] = []

        rendered = _render_ui(v2_without_promotions)

        self.assertEqual(rendered["badgeCount"], 0)
        self.assertIn("0 verified profiles", rendered["summaryHtml"])
        self.assertIn("— lowest patch recompute", rendered["summaryHtml"])
        primary = next(
            row
            for row in self.data["tracks"]["lpm"]
            if row["method"] == self.data["primaryGrtMethod"]
        )
        self.assertNotIn(primary["model"] + " ·", rendered["summaryHtml"])

    def test_snapshot_metadata_has_runtime_targets(self):
        expected_ids = {
            "lpm-method-count",
            "source-artifact-count",
            "snapshot-date",
            "snapshot-date-long",
            "footer-sync-date",
            "open-mos-judge",
        }
        parser = _IdCollector()
        parser.feed(self.html)
        self.assertTrue(expected_ids.issubset(set(parser.ids)))
        for element_id in expected_ids:
            self.assertIn('"' + element_id + '"', self.app)

    def test_runtime_counts_replace_non_authoritative_html_fallbacks(self):
        self.assertNotIn('id="lpm-method-count">27<', self.html)
        self.assertNotIn('id="table-count">Showing 27 methods<', self.html)
        self.assertIn('id="lpm-method-count">—<', self.html)
        self.assertIn('id="table-count">Showing — methods<', self.html)
        self.assertIn("lpmRows.length.toLocaleString", self.app)
        self.assertIn('dataset.tracks[state.track].length', self.app)

    def test_v2_qualification_is_accessible_and_text_only(self):
        self.assertIn(
            'id="grt-family-qualification" '
            'aria-labelledby="grt-family-qualification-title" hidden',
            self.html,
        )
        self.assertIn(
            'id="grt-family-qualification-list" '
            'aria-label="Educational GRT family qualification results"',
            self.html,
        )
        start = self.app.index("function renderGrtQualification()")
        end = self.app.index("function renderGlossary()", start)
        renderer = self.app[start:end]
        self.assertIn('entry.status === "promoted"', renderer)
        self.assertIn('entry.status === "not_promoted"', renderer)
        self.assertIn(
            'strict_quality_gate_failed: "Strict full-set quality gate not met"',
            self.app,
        )
        self.assertIn(
            'strict_quality_gate_passed: "Strict full-set quality gate passed"',
            self.app,
        )
        self.assertIn(
            'closedLabel(grtFamilyLabels, entry.family, "Unknown family")',
            renderer,
        )
        self.assertIn(
            'closedLabel(grtReasonLabels, entry.reasonCode, '
            '"Qualification reason unavailable")',
            renderer,
        )
        self.assertIn("document.createElement", renderer)
        self.assertIn(".textContent =", renderer)
        self.assertNotIn("innerHTML", renderer)

    def test_v2_qualification_does_not_create_rows_or_execute_markup(self):
        row = {
            "rank": 1,
            "model": "Verified without prefix",
            "method": "verified_without_prefix",
            "samples": 634,
            "open_mos": 1.2,
            "token_f1": 0.1,
            "cer": 0.9,
            "wer": 0.9,
            "exact_match": 0,
            "recompute_ratio": 0.5,
            "reference_recompute_ratio": 0.4,
            "effective_fps": 0.2,
            "throughput_fps": 4.5,
            "source": "open",
        }
        prefix_only = deepcopy(row)
        prefix_only.update(
            rank=2,
            model="Prefix is not authority",
            method="grt_unverified_prefix",
        )
        injected_method = '<script id="fake-row">alert(1)</script>'
        injected_reason = '<img src=x onerror="alert(1)">'
        dataset = {
            "generatedAt": "2026-08-20T00:00:00Z",
            "sourceArtifacts": 2,
            "openMosJudge": "judge",
            "primaryGrtMethod": row["method"],
            "grtMethods": [row["method"], injected_method],
            "grtFamilies": [
                {
                    "family": "route31",
                    "status": "promoted",
                    "reasonCode": "strict_quality_gate_passed",
                    "method": row["method"],
                    "baseMethod": "base-route31",
                    "campaignFingerprint": "a" * 64,
                },
                {
                    "family": '<img src=x onerror="family()">',
                    "status": "not_promoted",
                    "reasonCode": injected_reason,
                    "method": None,
                    "baseMethod": "base-llava7",
                    "campaignFingerprint": "b" * 64,
                },
                {
                    "family": "qwen3",
                    "status": "promoted",
                    "reasonCode": "strict_quality_gate_passed",
                    "method": injected_method,
                    "baseMethod": "base-qwen3",
                    "campaignFingerprint": "c" * 64,
                },
                {
                    "family": "qwen7",
                    "status": "not_promoted",
                    "reasonCode": "strict_quality_gate_failed",
                    "method": None,
                    "baseMethod": "base-qwen7",
                    "campaignFingerprint": "d" * 64,
                },
            ],
            "tracks": {"lpm": [row, prefix_only], "highmotion": []},
        }

        rendered = _render_ui(dataset)

        self.assertEqual(rendered["badgeCount"], 1)
        self.assertEqual(rendered["methodCount"], "2")
        self.assertEqual(rendered["tableCount"], "Showing 2 of 2 methods")
        self.assertFalse(rendered["qualification"]["hidden"])
        self.assertEqual(rendered["qualification"]["count"], "2 of 4 Educational GRT families promoted")
        self.assertEqual(
            [item["status"] for item in rendered["qualification"]["items"]],
            ["promoted", "not_promoted", "promoted", "not_promoted"],
        )
        qualification_text = " ".join(
            item["text"] for item in rendered["qualification"]["items"]
        )
        self.assertIn(injected_method, qualification_text)
        self.assertNotIn(injected_reason, qualification_text)
        self.assertNotIn("family()", qualification_text)
        self.assertIn("Unknown family", qualification_text)
        self.assertIn("Qualification reason unavailable", qualification_text)
        self.assertIn("Strict full-set quality gate passed", qualification_text)
        self.assertIn("Strict full-set quality gate not met", qualification_text)
        self.assertNotIn("SCRIPT", rendered["qualification"]["tags"])
        self.assertNotIn("IMG", rendered["qualification"]["tags"])
        self.assertNotIn("fake-row", rendered["tableHtml"])
        self.assertEqual(rendered["tableHtml"].count("<tr>"), 2)

    def test_v1_family_entries_remain_supported_without_status_ui(self):
        legacy = deepcopy(self.data)
        legacy["grtMethods"] = [legacy["primaryGrtMethod"]]
        legacy["grtFamilies"] = [
            {
                "method": legacy["primaryGrtMethod"],
                "baseMethod": "base-" + str(index),
                "campaignFingerprint": str(index) * 64,
            }
            for index in range(1, 5)
        ]

        rendered = _render_ui(legacy)

        self.assertTrue(rendered["qualification"]["hidden"])
        self.assertEqual(rendered["qualification"]["items"], [])
        self.assertEqual(rendered["methodCount"], str(len(legacy["tracks"]["lpm"])))
        self.assertEqual(rendered["badgeCount"], 1)

    def test_final_assets_are_versioned_and_frozen_data_retains_its_key(self):
        assets = re.findall(
            r'(?:href|src)="(styles\.css|data/leaderboard\.js|data/public-audit\.js|app\.js)'
            r'\?v=([0-9a-z-]+)"',
            self.html,
        )
        self.assertEqual(
            [path for path, _version in assets],
            ["styles.css", "data/leaderboard.js", "data/public-audit.js", "app.js"],
        )
        self.assertEqual(dict(assets)["data/leaderboard.js"], "1f1b5a79c74ff763")
        self.assertEqual(len({version for path, version in assets if path != "data/leaderboard.js"}), 1)

    def test_sampling_density_is_not_presented_as_throughput(self):
        visible_sources = self.html + self.app
        self.assertNotIn("Effective FPS", visible_sources)
        self.assertNotIn("effective FPS", visible_sources)
        self.assertIn("Sampling density (fps)", visible_sources)
        self.assertIn("not processing throughput", visible_sources)

    def test_grt_telemetry_schema_separates_compute_density_and_speed(self):
        for field in (
            "recompute_ratio",
            "reference_recompute_ratio",
            "effective_fps",
            "throughput_fps",
        ):
            self.assertIn('{ key: "' + field + '"', self.app)
        self.assertIn("first-layer visual patch projections", self.app)
        self.assertIn("end-to-end request wall time", self.app)

    def test_copy_uses_fallback_when_clipboard_permission_is_rejected(self):
        self.assertIn("function copyFallback()", self.app)
        self.assertIn(".then(confirm).catch(copyFallback)", self.app)
        self.assertIn('showStatus("Copy failed")', self.app)

    def test_scrollable_regions_are_keyboard_accessible_and_named(self):
        self.assertIn(
            'class="table-scroll" role="region" tabindex="0" '
            'aria-label="Scrollable leaderboard table"',
            self.html,
        )
        self.assertGreaterEqual(self.html.count('<pre tabindex="0">'), 2)

    def test_dynamic_cells_escape_html_and_nullish_titles_are_not_leaked(self):
        self.assertIn("escapeHtml(row.model)", self.app)
        self.assertIn("escapeHtml(value)", self.app)
        self.assertIn(
            'value === null || value === undefined || value === "" ? "Not reported"',
            self.app,
        )
        self.assertNotIn('value === null ? "Not reported"', self.app)


if __name__ == "__main__":
    unittest.main()
