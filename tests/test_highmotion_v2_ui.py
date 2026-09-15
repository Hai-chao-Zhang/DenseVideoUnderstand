"""High-Motion v2 UI contracts using synthetic summaries, never real predictions."""

import hashlib
import json
import subprocess
import unittest
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1")
BASELINES = (
    "qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b",
    "qwen2_vl_2b", "qwen2_5_vl_3b", "qwen2_5_vl_7b", "qwen2_5_vl_32b",
    "qwen2_5_vl_72b", "llava_onevision_0_5b", "llava_onevision_original",
    "qwen2_vl_7b", "llava_onevision_1_5_8b", "llava_onevision_2_8b",
    "videollama3_2b", "videollama3_7b", "longva_7b", "phi4_multimodal",
)
BASELINE = "llava_onevision_0_5b"
GRT = "synthetic_grt_v2"


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _comparison(rows):
    by_method = {row["method"]: row for row in rows}
    baseline = {key: by_method[BASELINE][key] for key in METRICS}
    candidate = {key: by_method[GRT][key] for key in METRICS}
    delta = {
        key: None if candidate[key] is None or baseline[key] is None
        else candidate[key] - baseline[key]
        for key in METRICS
    }
    oriented = {
        key: None if delta[key] is None else
        (-delta[key] if key in ("grid_ade", "grid_fde") else delta[key])
        for key in METRICS
    }
    better = {key: None if oriented[key] is None else oriented[key] > 1e-12
              for key in METRICS}
    return {
        "baseline_method": BASELINE, "grt_method": GRT,
        "primary_metric": "grid_acc", "point_tolerance": 1e-12,
        "baseline_metrics": baseline, "grt_metrics": candidate,
        "grt_minus_baseline": delta, "oriented_improvements": oriented,
        "metric_outperform": better, "grid_acc_outperform": better["grid_acc"] is True,
    }


def _summary(*, null_fde=False, all_null=False):
    """Mirror the numeric validator's output, without importing private source."""
    rows = []
    for method in (GRT, *sorted(BASELINES)):
        grt = method == GRT
        values = {
            "grid_acc": 0.625 if grt else 0.5,
            "grid_ade": 0.375 if grt else 0.5,
            "grid_fde": 0.0 if grt else 0.5,
            "grid_transition_acc": 4 / 7 if grt else 3 / 7,
            "token_f1": 0.75 if grt else 0.625,
        }
        valid_slots = 7000 if null_fde else 8000
        metric_rows = dict.fromkeys(METRICS, 1000)
        metric_slots = dict.fromkeys(METRICS, valid_slots)
        metric_slots.update(grid_fde=1000, grid_transition_acc=6000 if null_fde else 7000)
        if null_fde:
            values["grid_fde"] = None
            metric_rows["grid_fde"] = metric_slots["grid_fde"] = 0
        if all_null:
            values = dict.fromkeys(METRICS)
            metric_rows = dict.fromkeys(METRICS, 0)
            metric_slots = dict.fromkeys(METRICS, 0)
            valid_slots = 0
        rows.append({
            "method": method,
            "model": "Synthetic GRT 0.5B" if grt else "Synthetic baseline " + method,
            "source": "open", "samples": 1000, "records": 1000,
            "prediction_source": "new_grt" if grt else "archived_baseline",
            "predictions_sha256": _sha("synthetic predictions " + method),
            "records_with_scored_slots": 0 if all_null else 1000,
            "records_without_scored_slots": 1000 if all_null else 0,
            "sampled_slots": 8000, "valid_slots": valid_slots,
            "metric_scored_records": metric_rows,
            "metric_scored_slots_or_edges": metric_slots,
            "rank": None if all_null else 1 if grt else 2,
            **values,
        })
    if all_null:
        rows.sort(key=lambda row: row["method"])
    return {
        "status": "numeric_reports_validated",
        "benchmark_version": "highmotion-right-ring-v2",
        "target_joint": "rightRingFingerMetacarpal",
        "reference_policy": "finite-positive-depth-in-frame-positive-confidence-v1",
        "scorer_version": "position-preserving-masked-grid-v1",
        "references_sha256": "a" * 64, "input_sequence_sha256": "b" * 64,
        "scope": "first-1000-source-rows", "source_reference_records": 3243,
        "full_source_coverage": False, "method_count": 19, "rows": rows,
        "ranking": "Grid Accuracy descending; exact ties share competition rank, ordered by method; null unranked",
        "comparison": _comparison(rows),
        "comparison_caveat": (
            "New GRT predictions versus rescored, configuration-checked archived predictions; "
            "archived baseline weight revisions and consumed-tensor identity are unproven. "
            "This is not a freshly rerun byte-identical paired experiment or a "
            "statistical-significance claim."
        ),
        "underlying_file_hashes_verified_by_this_function": False,
        "inference_performed": False, "baseline_gpu_rerun": False,
        "baseline_exact_weight_or_tensor_identity_proven": False,
        "statistical_significance_claim": False, "automatic_publication": False,
        "release_integrity_verified": True, "release_manifest_sha256": "c" * 64,
    }


def _dataset():
    loader = """
const fs = require('fs'), vm = require('vm');
const context = { window: {} };
vm.runInNewContext(fs.readFileSync('data/leaderboard.js', 'utf8'), context);
process.stdout.write(JSON.stringify(context.window.DIVE_LEADERBOARD));
"""
    result = subprocess.run(["node", "-e", loader], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return json.loads(result.stdout)


def _render(dataset, summary=None, *, actions=(), comparator=None):
    # Only closure access is injected. Actual event handlers perform search/sort.
    loader = r"""
const fs = require("fs"), vm = require("vm");
class Element {
  constructor(tagName, id) {
    this.tagName = String(tagName || "div").toUpperCase(); this.id = id || "";
    this.children = []; this.attributes = {}; this.className = ""; this.hidden = false;
    this.innerHTML = ""; this._text = ""; this.style = {}; this.value = "";
    this.listeners = {}; this.classList = { toggle() {}, remove() {} };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  appendChild(child) { this.children.push(child); return child; }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] || null; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  dispatch(name) { if (this.listeners[name]) this.listeners[name]({ preventDefault() {} }); }
  querySelectorAll() { return []; }
  focus() {} click() { this.dispatch("click"); } select() {}
}
const ids = ["model-search", "source-filter", "leader-summary", "table-count",
  "ranking-rule", "leaderboard-caption", "metric-definitions", "grt-family-qualification",
  "grt-family-qualification-count", "grt-family-qualification-list", "lpm-method-count",
  "source-artifact-count", "open-mos-judge", "snapshot-date", "snapshot-date-long",
  "footer-sync-date", "active-code", "code-filename", "code-panel", "grt-comparison-body"];
const elements = {};
ids.forEach(id => { elements[id] = new Element("div", id); });
elements["source-filter"].value = "all";
const head = new Element("thead"), body = new Element("tbody"), table = new Element("table");
let sortButtons = {};
head.querySelectorAll = selector => {
  if (selector !== "[data-sort]") return [];
  sortButtons = {};
  return [...head.innerHTML.matchAll(/data-sort="([^"]+)"/g)].map(match => {
    const button = new Element("button"); button.setAttribute("data-sort", match[1]);
    sortButtons[match[1]] = button; return button;
  });
};
head.querySelector = selector => {
  const match = /data-sort="([^"]+)"/.exec(selector);
  return match ? sortButtons[match[1]] || null : null;
};
table.querySelector = selector => selector === "thead" ? head : body;
elements["leaderboard-table"] = table;
const document = {
  body: new Element("body"), createElement: tag => new Element(tag), execCommand: () => false,
  getElementById: id => elements[id] || null, querySelector: () => null, querySelectorAll: () => []
};
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const window = { DIVE_LEADERBOARD: input.dataset, DIVE_PUBLIC_AUDIT: input.audit, isSecureContext: false };
const context = { window, document, navigator: {}, clearTimeout, setTimeout };
const source = fs.readFileSync("app.js", "utf8").replace("  installLeaderboard();",
  "  window.testSelectTrack = selectTrack;\n" +
  "  window.testCompare = function (rows, key, direction) { state.sortKey = key; state.sortDirection = direction; return rows.slice().sort(compareRows); };\n" +
  "  installLeaderboard();");
vm.runInNewContext(source, context);
window.testSelectTrack("highmotion");
function snapshot() {
  return { tableHtml: body.innerHTML, tableHead: head.innerHTML,
    badgeCount: (body.innerHTML.match(/model-badge grt/g) || []).length,
    tableCount: elements["table-count"].textContent, tableEmpty: table.getAttribute("data-empty"),
    caption: elements["leaderboard-caption"].textContent,
    rankingRule: elements["ranking-rule"].textContent,
    glossary: elements["metric-definitions"].innerHTML,
    summaryHtml: elements["leader-summary"].innerHTML };
}
const snapshots = [snapshot()];
input.actions.forEach(action => {
  if (action.kind === "track") window.testSelectTrack(action.value);
  else if (action.kind === "search") {
    elements["model-search"].value = action.value; elements["model-search"].dispatch("input");
  } else if (action.kind === "source") {
    elements["source-filter"].value = action.value; elements["source-filter"].dispatch("change");
  } else if (action.kind === "sort") {
    if (!sortButtons[action.value]) throw new Error("Missing sort button: " + action.value);
    sortButtons[action.value].click();
  } else throw new Error("Unknown synthetic action");
  snapshots.push(snapshot());
});
let compared = null;
if (input.comparator) {
  const item = input.comparator;
  compared = window.testCompare(item.rows, item.key, item.direction).map(row => row.method);
}
process.stdout.write(JSON.stringify({ snapshots, compared }));
"""
    result = subprocess.run(
        ["node", "-e", loader], cwd=ROOT, check=True, capture_output=True,
        input=json.dumps({"dataset": dataset, "audit": {"highmotion_v2": summary},
                          "actions": list(actions), "comparator": comparator}), text=True,
    )
    return json.loads(result.stdout)


class _TableRows(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = {"text": "", "titles": [], "class": attrs.get("class", "")}
        if self.cell is not None and "title" in attrs:
            self.cell["titles"].append(attrs["title"])

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row and "rank-cell" in self.row[0]["class"]:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def _rows(snapshot):
    return _TableRows(snapshot["tableHtml"]).rows


class HighMotionV2UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _dataset()

    def assert_held(self, summary):
        data = deepcopy(self.data)
        # A stale legacy row must never be used as fallback for an invalid v2.
        data["tracks"]["highmotion"] = [{
            "method": "legacy_unverified_sentinel", "model": "Legacy sentinel",
            "rank": 1, "grid_acc": 1, "source": "open", "samples": 1000,
        }]
        snapshot = _render(data, summary)["snapshots"][0]
        self.assertEqual(_rows(snapshot), [])
        self.assertEqual(snapshot["badgeCount"], 0)
        self.assertEqual(snapshot["tableEmpty"], "true")
        self.assertIn("withheld", snapshot["tableHtml"].lower())
        self.assertNotIn("legacy_unverified_sentinel", snapshot["tableHtml"])

    def test_missing_payload_retains_hold_without_legacy_fallback(self):
        self.assert_held(None)

    def test_valid_summary_shows_all_methods_and_exactly_one_grt_badge(self):
        snapshot = _render(self.data, _summary())["snapshots"][0]
        rows = _rows(snapshot)
        self.assertEqual(len(rows), 19)
        self.assertEqual(snapshot["badgeCount"], 1)
        self.assertEqual(snapshot["tableEmpty"], "false")
        self.assertEqual(snapshot["tableCount"], "Showing 19 of 19 methods")
        self.assertEqual({row[2]["text"] for row in rows}, set(BASELINES) | {GRT})
        self.assertEqual(rows[0][2]["text"], GRT)
        self.assertTrue(all(row[3]["text"] == "1,000" for row in rows))

    def test_v2_grt_badge_does_not_inherit_legacy_educational_method_ids(self):
        data = deepcopy(self.data)
        data["grtMethods"] = list(BASELINES)
        data["primaryGrtMethod"] = BASELINE
        snapshot = _render(data, _summary())["snapshots"][0]
        self.assertEqual(len(_rows(snapshot)), 19)
        self.assertEqual(snapshot["badgeCount"], 1)
        for row in _rows(snapshot):
            self.assertEqual("GRTopen weights" in row[1]["text"], row[2]["text"] == GRT)

    def test_caption_glossary_and_caveat_disambiguate_scope_and_reference(self):
        snapshot = _render(self.data, _summary())["snapshots"][0]
        self.assertIn("v2", snapshot["caption"].lower())
        self.assertRegex(snapshot["caption"], r"1,?000")
        self.assertRegex(snapshot["caption"].lower(), r"not full.*3,?243")
        self.assertIn("preview", snapshot["caption"].lower())
        explained = (snapshot["glossary"] + snapshot["summaryHtml"]).lower()
        self.assertIn("right", explained)
        self.assertIn("ring", explained)
        self.assertIn("valid", explained)
        self.assertIn("final", explained)
        self.assertIn("adjacent", explained)
        self.assertIn("archiv", explained)
        self.assertIn("unproven", explained)
        self.assertIn("statistical", explained)

    def test_each_metric_displays_rows_and_slots_or_edges_coverage(self):
        snapshot = _render(self.data, _summary())["snapshots"][0]
        row = _rows(snapshot)[0]
        for index, count, unit in ((4, 8000, "slots"), (5, 8000, "slots"),
                                   (6, 1000, "slots"), (7, 7000, "edges"),
                                   (8, 8000, "slots")):
            with self.subTest(column=index):
                text = row[index]["text"] + " " + " ".join(row[index]["titles"])
                self.assertRegex(text, r"1,?000\s+(?:scored\s+)?(?:rows|records)")
                self.assertRegex(text, rf"{count // 1000},?000\s+(?:valid\s+)?{unit}")
        self.assertIn("0.5714", row[7]["text"])
        self.assertNotIn("Not reported", " ".join(row[7]["titles"]))

    def test_zero_is_zero_and_null_is_dash_with_zero_metric_coverage(self):
        zero = _rows(_render(self.data, _summary())["snapshots"][0])[0][6]
        self.assertTrue(zero["text"].startswith("0"))
        self.assertEqual(zero["titles"][0], "0")
        self.assertNotIn("—", zero["text"])
        snapshot = _render(self.data, _summary(null_fde=True))["snapshots"][0]
        self.assertEqual(len(_rows(snapshot)), 19)
        for row in _rows(snapshot):
            self.assertTrue(row[6]["text"].startswith("—"))
            self.assertRegex(row[6]["text"], r"0\s+(?:rows|records)")
            self.assertRegex(row[6]["text"], r"0\s+slots")

    def test_partial_reference_coverage_does_not_shrink_the_preview_population(self):
        summary = _summary()
        for row in summary["rows"]:
            row.update(records_with_scored_slots=800, records_without_scored_slots=200,
                       valid_slots=5600)
            row["metric_scored_records"] = dict.fromkeys(METRICS, 800)
            row["metric_scored_records"]["grid_fde"] = 600
            row["metric_scored_slots_or_edges"] = dict.fromkeys(METRICS, 5600)
            row["metric_scored_slots_or_edges"].update(grid_fde=600, grid_transition_acc=4800)
        snapshot = _render(self.data, summary)["snapshots"][0]
        rows = _rows(snapshot)
        self.assertEqual(len(rows), 19)
        self.assertTrue(all(row[3]["text"] == "1,000" for row in rows))
        self.assertRegex(rows[0][4]["text"], r"800\s+rows.*5,?600\s+slots")
        self.assertRegex(rows[0][6]["text"], r"600\s+rows.*600\s+slots")
        self.assertRegex(rows[0][7]["text"], r"800\s+rows.*4,?800\s+edges")

    def test_valid_negative_grt_outcome_is_retained_without_a_winner_claim(self):
        summary = _summary()
        for row in summary["rows"]:
            row["grid_acc"] = 0.375 if row["method"] == GRT else 0.5
            row["rank"] = 19 if row["method"] == GRT else 1
        summary["rows"].sort(key=lambda row: (-row["grid_acc"], row["method"]))
        summary["comparison"] = _comparison(summary["rows"])
        snapshot = _render(self.data, summary)["snapshots"][0]
        rows = _rows(snapshot)
        self.assertEqual(len(rows), 19)
        self.assertEqual(rows[-1][2]["text"], GRT)
        self.assertEqual(rows[-1][0]["text"], "19")
        self.assertEqual(snapshot["badgeCount"], 1)
        self.assertIn("-0.125", snapshot["summaryHtml"])

    def test_null_primary_scores_have_unranked_dash_and_no_invented_rank(self):
        snapshot = _render(self.data, _summary(all_null=True))["snapshots"][0]
        rows = _rows(snapshot)
        self.assertEqual(len(rows), 19)
        self.assertTrue(all(row[0]["text"] == "—" for row in rows))
        self.assertTrue(all(row[4]["text"].startswith("—") for row in rows))
        self.assertNotIn("rank-null", snapshot["tableHtml"])

    def test_server_competition_ranks_survive_sort_and_search(self):
        snapshots = _render(self.data, _summary(), actions=(
            {"kind": "sort", "value": "model"},
            {"kind": "search", "value": "QWEN3_VL_2B"},
        ))["snapshots"]
        expected = {method: "1" if method == GRT else "2" for method in (*BASELINES, GRT)}
        for snapshot in snapshots:
            for row in _rows(snapshot):
                self.assertEqual(row[0]["text"], expected[row[2]["text"]])
        self.assertEqual(len(_rows(snapshots[-1])), 1)
        self.assertEqual(_rows(snapshots[-1])[0][0]["text"], "2")

    def test_search_no_match_is_not_a_reference_hold_and_reset_restores_rows(self):
        snapshots = _render(self.data, _summary(), actions=(
            {"kind": "search", "value": "__no_synthetic_model_matches__"},
            {"kind": "search", "value": ""},
        ))["snapshots"]
        self.assertEqual(snapshots[1]["tableCount"], "Showing 0 of 19 methods")
        self.assertIn("No models match this filter.", snapshots[1]["tableHtml"])
        self.assertNotIn("withheld", snapshots[1]["tableHtml"].lower())
        self.assertEqual(len(_rows(snapshots[2])), 19)

    def test_cross_tab_educational_keeps_29_rows_and_its_own_metrics(self):
        snapshots = _render(self.data, _summary(), actions=(
            {"kind": "track", "value": "lpm"},
            {"kind": "track", "value": "highmotion"},
        ))["snapshots"]
        self.assertEqual(snapshots[1]["tableCount"], "Showing 29 of 29 methods")
        self.assertEqual(len(_rows(snapshots[1])), 29)
        self.assertIn("Open MOS", snapshots[1]["tableHead"])
        self.assertNotIn("Grid Acc", snapshots[1]["tableHead"])
        self.assertEqual(len(_rows(snapshots[2])), 19)
        self.assertEqual(snapshots[2]["badgeCount"], 1)

    def test_source_filter_all_open_and_api_respects_v2_provenance(self):
        snapshots = _render(self.data, _summary(), actions=(
            {"kind": "source", "value": "open"},
            {"kind": "source", "value": "gemini"},
            {"kind": "source", "value": "all"},
            {"kind": "track", "value": "lpm"},
        ))["snapshots"]
        for index in (0, 1, 3):
            self.assertEqual(len(_rows(snapshots[index])), 19)
            self.assertEqual(snapshots[index]["badgeCount"], 1)
            self.assertTrue(all(row[-1]["text"] == "open" for row in _rows(snapshots[index])))
        self.assertEqual(snapshots[2]["tableCount"], "Showing 0 of 19 methods")
        self.assertEqual(snapshots[2]["tableEmpty"], "true")
        self.assertIn("No models match this filter.", snapshots[2]["tableHtml"])
        self.assertNotIn("withheld", snapshots[2]["tableHtml"].lower())
        self.assertEqual(snapshots[4]["tableCount"], "Showing 29 of 29 methods")
        self.assertEqual(len(_rows(snapshots[4])), 29)

    def test_null_sort_values_remain_last_in_both_directions(self):
        # Mixed null metric coverage is invalid for a shared-reference payload.
        # Test the generic comparator in isolation instead of weakening that rule.
        rows = [{"method": "null", "rank": 1, "metric": None},
                {"method": "zero", "rank": 2, "metric": 0},
                {"method": "positive", "rank": 3, "metric": 0.5}]
        for direction, expected in (("asc", ["zero", "positive", "null"]),
                                    ("desc", ["positive", "zero", "null"])):
            with self.subTest(direction=direction):
                result = _render(self.data, _summary(), comparator={
                    "rows": rows, "key": "metric", "direction": direction,
                })
                self.assertEqual(result["compared"], expected)

    def test_invalid_top_level_contract_fails_closed(self):
        changes = {
            "release_integrity_verified": [False, None, "true", 1],
            "release_manifest_sha256": [None, "c" * 63, "C" * 64, "z" * 64],
            "references_sha256": [None, "a" * 63, "A" * 64],
            "input_sequence_sha256": [None, "b" * 65, "B" * 64],
            "status": ["pending", None],
            "benchmark_version": ["legacy", None],
            "target_joint": ["leftIndexFingerMetacarpal", None],
            "reference_policy": ["unmasked", None],
            "scorer_version": ["legacy", None],
            "scope": ["full-source", "first-999-source-rows", None],
            "source_reference_records": [1000, "3243", True],
            "method_count": [18, 20, "19"],
            "full_source_coverage": [True, None],
            "underlying_file_hashes_verified_by_this_function": [True],
            "inference_performed": [True], "baseline_gpu_rerun": [True],
            "baseline_exact_weight_or_tensor_identity_proven": [True],
            "statistical_significance_claim": [True], "automatic_publication": [True],
        }
        for key, values in changes.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    summary = _summary()
                    summary[key] = value
                    self.assert_held(summary)

    def test_missing_required_contract_fields_fail_closed(self):
        for key in ("release_integrity_verified", "release_manifest_sha256", "status",
                    "references_sha256", "input_sequence_sha256", "scope", "comparison"):
            with self.subTest(key=key):
                summary = _summary()
                del summary[key]
                self.assert_held(summary)

    def test_invalid_rows_counts_metric_bounds_and_hashes_fail_closed(self):
        mutations = (
            ("samples", 999), ("samples", "1000"), ("records", 999),
            ("sampled_slots", 7999), ("valid_slots", 8001),
            ("records_with_scored_slots", 999), ("records_without_scored_slots", 1),
            ("predictions_sha256", "bad"), ("prediction_source", "archived_baseline"),
            ("grid_acc", None), ("grid_acc", -0.01), ("grid_acc", 1.01),
            ("grid_acc", "0.625"), ("grid_acc", True),
            ("grid_ade", 1.42), ("grid_fde", -0.01),
            ("grid_transition_acc", 1.01), ("token_f1", -0.01),
            ("rank", 3), ("rank", None), ("rank", "1"),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                summary = _summary()
                summary["rows"][0][key] = value
                self.assert_held(summary)
        for change in ("missing", "duplicate", "unknown", "reordered"):
            with self.subTest(change=change):
                summary = _summary()
                if change == "missing":
                    summary["rows"].pop()
                elif change == "duplicate":
                    summary["rows"][-1] = deepcopy(summary["rows"][-2])
                elif change == "unknown":
                    summary["rows"][-1]["method"] = "unapproved_baseline"
                else:
                    summary["rows"].reverse()
                self.assert_held(summary)

    def test_invalid_metric_coverage_and_cross_method_disagreement_fail_closed(self):
        changes = (
            ("metric_scored_records", "grid_acc", 0),
            ("metric_scored_records", "grid_acc", 1001),
            ("metric_scored_records", "grid_fde", 999),
            ("metric_scored_slots_or_edges", "grid_acc", 7999),
            ("metric_scored_slots_or_edges", "grid_fde", 1001),
            ("metric_scored_slots_or_edges", "grid_transition_acc", 7001),
            ("metric_scored_slots_or_edges", "token_f1", "8000"),
        )
        for field, metric, value in changes:
            with self.subTest(field=field, metric=metric, value=value):
                summary = _summary()
                summary["rows"][0][field][metric] = value
                self.assert_held(summary)
        summary = _summary(null_fde=True)
        summary["rows"][0]["grid_fde"] = 0
        self.assert_held(summary)
        for field in ("metric_scored_records", "metric_scored_slots_or_edges"):
            summary = _summary()
            summary["rows"][0][field]["unrecognized_metric"] = 0
            self.assert_held(summary)

    def test_invalid_comparison_cannot_create_an_unsupported_winner(self):
        for key, value in (("baseline_method", "qwen3_vl_2b"),
                           ("grt_method", BASELINE), ("primary_metric", "token_f1"),
                           ("point_tolerance", 0), ("grid_acc_outperform", False)):
            with self.subTest(key=key):
                summary = _summary()
                summary["comparison"][key] = value
                self.assert_held(summary)
        for field, value in (("baseline_metrics", 0.0), ("grt_metrics", 0.0),
                             ("grt_minus_baseline", 0.0), ("oriented_improvements", 0.0),
                             ("metric_outperform", False)):
            with self.subTest(field=field):
                summary = _summary()
                summary["comparison"][field]["grid_acc"] = value
                self.assert_held(summary)

    def test_synthetic_model_display_is_escaped(self):
        summary = _summary()
        summary["rows"][0]["model"] = 'Synthetic <img src=x onerror="bad()"> GRT'
        snapshot = _render(self.data, summary)["snapshots"][0]
        self.assertEqual(len(_rows(snapshot)), 19)
        self.assertNotIn("<img", snapshot["tableHtml"])
        self.assertIn("&lt;img", snapshot["tableHtml"])


if __name__ == "__main__":
    unittest.main()
