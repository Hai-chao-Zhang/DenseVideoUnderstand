(function () {
  "use strict";

  var dataset = window.DIVE_LEADERBOARD;
  var publicAudit = window.DIVE_PUBLIC_AUDIT;
  var highmotionV2 = validatedHighmotionV2(publicAudit && publicAudit.highmotion_v2);
  if (dataset) {
    dataset = JSON.parse(JSON.stringify(dataset));
    // 2026-09-14 target/reference-consistency hold. Neither the immutable
    // archive nor a cached previously screened overlay may repopulate rankings.
    dataset.tracks.highmotion = [];
    delete dataset.tracks.highmotion_historical;
    // Only the separately authenticated, versioned numeric summary may lift
    // this display hold. Never merge archived High-Motion numbers into v2.
    if (highmotionV2) {
      dataset.tracks.highmotion = highmotionV2.rows.map(function (row) {
        var display = Object.assign({}, row);
        display.transition_acc = row.grid_transition_acc;
        display.source = "open";
        display.protocol_note = "v2 · fixed 1,000-record preview · corrected reference; cached baselines";
        return display;
      });
    }
  }
  var table = document.getElementById("leaderboard-table");
  var tableHead = table ? table.querySelector("thead") : null;
  var tableBody = table ? table.querySelector("tbody") : null;
  var searchInput = document.getElementById("model-search");
  var sourceFilter = document.getElementById("source-filter");
  var summary = document.getElementById("leader-summary");
  var tableCount = document.getElementById("table-count");
  var rankingRule = document.getElementById("ranking-rule");
  var caption = document.getElementById("leaderboard-caption");
  var glossary = document.getElementById("metric-definitions");
  var grtQualification = document.getElementById("grt-family-qualification");
  var grtQualificationCount = document.getElementById("grt-family-qualification-count");
  var grtQualificationList = document.getElementById("grt-family-qualification-list");

  function validatedHighmotionV2(payload) {
    var metrics = ["grid_acc", "grid_ade", "grid_fde", "grid_transition_acc", "token_f1"];
    var baselines = ["qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b", "qwen3_vl_32b",
      "qwen2_vl_2b", "qwen2_5_vl_3b", "qwen2_5_vl_7b", "qwen2_5_vl_32b", "qwen2_5_vl_72b",
      "llava_onevision_0_5b", "llava_onevision_original", "qwen2_vl_7b", "llava_onevision_1_5_8b",
      "llava_onevision_2_8b", "videollama3_2b", "videollama3_7b", "longva_7b", "phi4_multimodal"];
    function object(value) { return value && typeof value === "object" && !Array.isArray(value); }
    function hash(value) { return typeof value === "string" && /^[0-9a-f]{64}$/.test(value); }
    function count(value, maximum) { return Number.isInteger(value) && value >= 0 && value <= maximum; }
    function metricMap(value) {
      return object(value) && Object.keys(value).length === 5 && metrics.every(function (key) {
        return Object.prototype.hasOwnProperty.call(value, key);
      });
    }
    function same(a, b) { return a === b || (typeof a === "number" && typeof b === "number" && Math.abs(a - b) <= 1e-15); }
    function metricMaximum(key) { return key === "grid_ade" || key === "grid_fde" ? Math.SQRT2 : 1; }
    try {
      if (!object(payload) || payload.release_integrity_verified !== true || !hash(payload.release_manifest_sha256)
          || !hash(payload.references_sha256) || !hash(payload.input_sequence_sha256)
          || payload.status !== "numeric_reports_validated" || payload.benchmark_version !== "highmotion-right-ring-v2"
          || payload.target_joint !== "rightRingFingerMetacarpal"
          || payload.reference_policy !== "finite-positive-depth-in-frame-positive-confidence-v1"
          || payload.scorer_version !== "position-preserving-masked-grid-v1"
          || payload.source_reference_records !== 3243 || payload.scope !== "first-1000-source-rows"
          || payload.method_count !== 19 || !Array.isArray(payload.rows) || payload.rows.length !== 19
          || ["full_source_coverage", "inference_performed", "baseline_gpu_rerun", "automatic_publication",
            "underlying_file_hashes_verified_by_this_function", "baseline_exact_weight_or_tensor_identity_proven",
            "statistical_significance_claim"].some(function (key) {
            return payload[key] !== false;
          }) || typeof payload.comparison_caveat !== "string" || !payload.comparison_caveat.trim()) return null;
      var comparison = payload.comparison;
      if (!object(comparison) || comparison.baseline_method !== "llava_onevision_0_5b"
          || typeof comparison.grt_method !== "string" || !/^[a-z][a-z0-9_.-]*$/.test(comparison.grt_method)
          || baselines.indexOf(comparison.grt_method) !== -1 || comparison.primary_metric !== "grid_acc"
          || comparison.point_tolerance !== 1e-12
          || ["baseline_metrics", "grt_metrics", "grt_minus_baseline", "oriented_improvements", "metric_outperform"].some(function (key) {
            return !metricMap(comparison[key]);
          })) return null;
      var methods = Object.create(null);
      var sharedCoverage = null;
      for (var index = 0; index < payload.rows.length; index += 1) {
        var row = payload.rows[index];
        if (!object(row) || typeof row.method !== "string" || methods[row.method]
            || (baselines.indexOf(row.method) === -1 && row.method !== comparison.grt_method)
            || typeof row.model !== "string" || !row.model.trim() || row.model.length > 300
            || !hash(row.predictions_sha256) || row.samples !== 1000 || row.records !== 1000
            || row.sampled_slots !== 8000 || !count(row.valid_slots, 8000)
            || !count(row.records_with_scored_slots, 1000) || !count(row.records_without_scored_slots, 1000)
            || row.records_with_scored_slots + row.records_without_scored_slots !== 1000
            || row.valid_slots < row.records_with_scored_slots || row.valid_slots > 8 * row.records_with_scored_slots
            || row.prediction_source !== (row.method === comparison.grt_method ? "new_grt" : "archived_baseline")
            || !metricMap(row.metric_scored_records) || !metricMap(row.metric_scored_slots_or_edges)) return null;
        for (var m = 0; m < metrics.length; m += 1) {
          var key = metrics[m];
          var records = row.metric_scored_records[key];
          var slots = row.metric_scored_slots_or_edges[key];
          var value = row[key];
          if (!count(records, 1000) || !count(slots, key === "grid_transition_acc" ? 7000 : 8000)
              || records > row.records_with_scored_slots || (records === 0) !== (slots === 0)
              || (value === null) !== (records === 0)
              || (value !== null && (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > metricMaximum(key)))) return null;
          if (key === "grid_acc" || key === "grid_ade" || key === "token_f1") {
            if (records !== row.records_with_scored_slots || slots !== row.valid_slots) return null;
          } else if (key === "grid_fde") {
            if (slots !== records) return null;
          } else if (slots < records || slots > 7 * records) return null;
        }
        var coverage = [row.records_with_scored_slots, row.records_without_scored_slots, row.valid_slots].concat(
          metrics.map(function (key) { return row.metric_scored_records[key]; }),
          metrics.map(function (key) { return row.metric_scored_slots_or_edges[key]; })).join(",");
        if (sharedCoverage !== null && coverage !== sharedCoverage) return null;
        sharedCoverage = coverage;
        if (row.grid_acc === null ? row.rank !== null : (!count(row.rank, 19) || row.rank === 0)) return null;
        if (index > 0 && row.rank !== null && (payload.rows[index - 1].rank === null
            || row.rank < payload.rows[index - 1].rank)) return null;
        methods[row.method] = row;
      }
      if (!baselines.every(function (method) { return methods[method]; }) || !methods[comparison.grt_method]) return null;
      // Check the supplied ranks; do not replace or recompute display ranks.
      var ordered = payload.rows.slice().sort(function (a, b) {
        if (a.grid_acc === null) return b.grid_acc === null ? 0 : 1;
        if (b.grid_acc === null) return -1;
        return b.grid_acc - a.grid_acc;
      });
      for (var position = 0; position < ordered.length; position += 1) {
        var current = ordered[position];
        if (current.grid_acc !== null && current.rank !== (position > 0 && current.grid_acc === ordered[position - 1].grid_acc
            ? ordered[position - 1].rank : position + 1)) return null;
      }
      var baseline = methods[comparison.baseline_method];
      var candidate = methods[comparison.grt_method];
      for (var metricIndex = 0; metricIndex < metrics.length; metricIndex += 1) {
        var metric = metrics[metricIndex];
        var delta = candidate[metric] === null ? null : candidate[metric] - baseline[metric];
        var oriented = delta === null ? null : (metric === "grid_ade" || metric === "grid_fde" ? -delta : delta);
        if (!same(comparison.baseline_metrics[metric], baseline[metric]) || !same(comparison.grt_metrics[metric], candidate[metric])
            || !same(comparison.grt_minus_baseline[metric], delta) || !same(comparison.oriented_improvements[metric], oriented)
            || comparison.metric_outperform[metric] !== (oriented === null ? null : oriented > comparison.point_tolerance)) return null;
      }
      if (comparison.grid_acc_outperform !== (comparison.metric_outperform.grid_acc === true)) return null;
      return JSON.parse(JSON.stringify(payload));
    } catch (error) {
      return null;
    }
  }

  var grtFamilyLabels = {
    route31: "Route31",
    llava7: "LLaVA 7B",
    qwen3: "Qwen 3B",
    qwen7: "Qwen 7B"
  };
  var grtReasonLabels = {
    strict_quality_gate_failed: "Strict full-set quality gate not met",
    strict_quality_gate_passed: "Strict full-set quality gate passed"
  };

  var state = {
    track: "lpm",
    search: "",
    source: "all",
    sortKey: "rank",
    sortDirection: "asc"
  };

  var columns = {
    lpm: [
      { key: "rank", label: "Rank", format: "integer" },
      { key: "model", label: "Model", format: "model" },
      { key: "method", label: "Method ID", format: "method" },
      { key: "samples", label: "Samples", format: "integer" },
      { key: "open_mos", label: "Open MOS ↑", format: "score", primary: true },
      { key: "token_f1", label: "Token F1 ↑", format: "score" },
      { key: "cer", label: "CER ↓", format: "score" },
      { key: "wer", label: "WER ↓", format: "score" },
      { key: "exact_match", label: "Exact Match ↑", format: "score" },
      { key: "recompute_ratio", label: "Patch recompute ↓", format: "ratio" },
      { key: "reference_recompute_ratio", label: "Reference compute ↓", format: "ratio" },
      { key: "effective_fps", label: "Sampling density (fps)", format: "score" },
      { key: "throughput_fps", label: "Throughput (fps) ↑", format: "score" },
      { key: "source", label: "Source", format: "source" }
    ],
    highmotion: [
      { key: "rank", label: "Rank", format: "integer" },
      { key: "model", label: "Model", format: "model" },
      { key: "method", label: "Method ID", format: "method" },
      { key: "samples", label: "Samples", format: "integer" },
      { key: "grid_acc", label: "Grid Acc ↑", format: "score", primary: true },
      { key: "grid_ade", label: "Grid ADE ↓", format: "score" },
      { key: "grid_fde", label: "Grid FDE ↓", format: "score" },
      { key: "transition_acc", label: "Transition Acc ↑", format: "score" },
      { key: "token_f1", label: "Token F1 ↑", format: "score" },
      { key: "effective_fps", label: "Sampling density (fps)", format: "score" },
      { key: "protocol_note", label: "Protocol / status", format: "text" },
      { key: "source", label: "Source", format: "source" }
    ]
  };

  var metricDefinitions = {
    lpm: [
      ["Open MOS ↑", "Mean semantic-match rating on a 0–5 scale from the declared open text judge; higher is better."],
      ["Token F1 ↑", "Token-overlap F1 between the generated answer and reference answer."],
      ["CER ↓", "Character error rate from edit distance. Insertions can produce values above one."],
      ["WER ↓", "Word error rate from edit distance. Lower values indicate closer transcription."],
      ["Patch recompute ↓", "Fraction of first-layer visual patch projections recomputed by an instrumented gated wrapper; this is not end-to-end FLOPs."],
      ["Reference compute ↓", "Recomputed patch projections divided by the patch projections requested by the corresponding ungated reference route."],
      ["Sampling density (fps)", "Sampled frames divided by full source-video duration. This is not processing throughput."],
      ["Throughput (fps) ↑", "Sampled frames divided by end-to-end request wall time; unlike sampling density, this is a processing-rate measurement."]
    ],
    highmotion: [
      ["Grid Acc ↑", "Accuracy at the eight aligned target positions over the nine semantic regions of the 3×3 image grid."],
      ["Grid ADE ↓", "Mean Euclidean distance between aligned predicted and reference grid-cell centers in normalized image coordinates (range 0 to √2); a missing position receives √2."],
      ["Grid FDE ↓", "Euclidean distance at the reference trajectory's final position in normalized image coordinates (range 0 to √2); a missing final position receives √2."],
      ["Transition Acc ↑", "Fraction of consecutive steps whose predicted 2D grid displacement exactly matches the reference displacement."],
      ["Token F1 ↑", "Order-insensitive token-overlap F1 over canonicalized grid labels; it measures region-label overlap, not temporal ordering."],
      ["Sampling density (fps)", "Sampled frames divided by full source-video duration when profiling telemetry is available; not processing throughput."]
    ]
  };

  if (highmotionV2) {
    columns.highmotion = columns.highmotion.filter(function (column) { return column.key !== "effective_fps"; });
    metricDefinitions.highmotion = [
      ["Scope", "High-Motion v2 uses the fixed first 1,000 source records, not the full 3,243-record benchmark. Every method uses the same valid-reference mask; cell coverage lists scored rows and slots or edges."],
      ["Target", "The right-hand ring-finger metacarpal (rightRingFingerMetacarpal) is the source-defined palm proxy in this version. A valid projected reference is not proof of RGB visibility."],
      ["Grid Acc ↑", "Macro mean of per-row accuracy over valid sampled reference positions. Invalid references are excluded, not scored as correct."],
      ["Grid ADE ↓", "Macro mean of per-row normalized grid-center distances over valid reference positions. Missing or malformed predictions receive √2."],
      ["Grid FDE ↓", "Distance at the original final sampled position. Undefined when that reference is invalid; the last valid earlier position is not substituted."],
      ["Transition Acc ↑", "Macro mean of accuracy over adjacent valid sampled positions only. Masked gaps are never bridged."],
      ["Token F1 ↑", "Canonical region-label overlap on valid positions, with surplus prediction slots penalized. This is not a temporal-order metric."],
      ["Missing values", "An em dash means undefined or unreported, never zero. Baselines are rescored archived predictions; exact historical weights and input tensors are not proven."]
    ];
  }

  var codeSamples = {
    setup: {
      filename: "setup.sh",
      value: [
        "git clone --branch fix/highmotion-target-v2 --single-branch \\",
        "  https://github.com/Hai-chao-Zhang/DenseVideoUnderstand.git DIVE-Bench",
        "cd DIVE-Bench",
        "python -m pip install 'PyYAML>=6'",
        "# 29 Educational + 19 High-Motion preview + 12 comparison rows (60 CSV records).",
        "# Pinned v2 manifest: 1f64ff54ec8eb09d72c37d6ef3a944e8ccae0a58fe5b4d45fabdfb0a7449d0dc",
        "python -m tools.densevideo.build_complete_leaderboard --verify-only",
        "python -m tools.densevideo.build_complete_leaderboard --output outputs/leaderboard-complete",
        "# Open outputs/leaderboard-complete/leaderboard.html; output must be new."
      ].join("\n")
    },
    lpm: {
      filename: "reproduce_grt.sh",
      value: [
        "# Use a dedicated environment with a compatible CUDA build.",
        "python -m venv .venv",
        ". .venv/bin/activate",
        "python -m pip install -e '.[test]'",
        "# First prints the pinned three-arm plan without running a GPU.",
        "dive-reproduce --profile qwen7 --output outputs/qwen7-full",
        "# Requires authorized videos and one visible GPU; also runs the judge.",
        "CUDA_VISIBLE_DEVICES=0 dive-reproduce --profile qwen7 \\",
        "  --output outputs/qwen7-full --execute --with-mos",
        "# Other profiles: route31, qwen3. Output must not already exist."
      ].join("\n")
    },
    motion: {
      filename: "task_names.txt",
      value: [
        "Educational High-FPS Videos:",
        "  dive_bench_educational_high_fps                 # 634 QA / 317 videos",
        "High-Motion input-task compatibility names:",
        "  dive_bench_high_motion_high_fps                # full 3,243 source items",
        "  dive_bench_high_motion_high_fps_preview1000    # fixed first 1,000 source items",
        "",
        "Legacy aliases: densevideo, densevideo_highmotion",
        "Published High-Motion scores use highmotion-right-ring-v2 references.",
        "Legacy task metrics are NOT corrected v2 scores; score saved predictions separately.",
        "The current release is the fixed 1,000-record preview, not the full 3,243.",
        "GRT improves Grid Accuracy; Transition Accuracy regresses against its 0.5B baseline.",
        "See docs/HIGHMOTION_REFERENCE_V2.md and docs/HIGHMOTION_V2_REPRODUCTION.md.",
        "Fresh inference requires separately authorized private source media."
      ].join("\n")
    }
  };

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatNumber(value, style) {
    if (value === null || value === undefined || value === "") return "—";
    if (style === "integer") return Number(value).toLocaleString("en-US");
    if (style === "ratio") return (Number(value) * 100).toFixed(2) + "%";
    if (Number(value) === 0) return "0";
    if (Math.abs(Number(value)) < 0.01) return Number(value).toFixed(5).replace(/0+$/, "");
    if (Math.abs(Number(value)) < 1) return Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    return Number(value).toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }

  function grtMethodIds() {
    if (dataset && Array.isArray(dataset.grtMethods)) return dataset.grtMethods;
    return dataset && dataset.primaryGrtMethod ? [dataset.primaryGrtMethod] : [];
  }

  function isGrtMethod(method) {
    if (state.track === "highmotion") return Boolean(highmotionV2 && method === highmotionV2.comparison.grt_method);
    return grtMethodIds().indexOf(method) !== -1;
  }

  function closedLabel(labels, key, fallback) {
    return typeof key === "string" && Object.prototype.hasOwnProperty.call(labels, key) ? labels[key] : fallback;
  }

  function modelCell(row) {
    var badges = [];
    if (isGrtMethod(row.method)) badges.push('<span class="model-badge grt">GRT</span>');
    if (row.source === "open") badges.push('<span class="model-badge">open weights</span>');
    else badges.push('<span class="model-badge">API</span>');
    return '<span class="model-name" title="' + escapeHtml(row.model) + '">' + escapeHtml(row.model) + '</span><span class="model-badges">' + badges.join("") + "</span>";
  }

  function cellContent(row, column) {
    var value = row[column.key];
    if (column.format === "text") return '<span class="protocol-cell">' + escapeHtml(value || "Historical protocol; see audit") + "</span>";
    if (column.format === "model") return modelCell(row);
    if (column.format === "method") return '<span class="method-id" title="' + escapeHtml(value) + '">' + escapeHtml(value) + "</span>";
    if (column.format === "source") return '<span class="source-chip source-' + escapeHtml(value) + '">' + escapeHtml(value) + "</span>";
    if (column.key === "rank") {
      if (value === null || value === undefined) return '<span title="Not ranked">—</span>';
      var medalClass = value <= 3 ? " rank-" + value : "";
      return '<span class="rank-medal' + medalClass + '">' + escapeHtml(value) + "</span>";
    }
    var formatted = formatNumber(value, column.format);
    var title = value === null || value === undefined || value === "" ? "Not reported" : escapeHtml(value);
    var metric = column.key === "transition_acc" ? "grid_transition_acc" : column.key;
    if (state.track === "highmotion" && highmotionV2 && Object.prototype.hasOwnProperty.call(row.metric_scored_records, metric)) {
      var coverage = formatNumber(row.metric_scored_records[metric], "integer") + " rows · " +
        formatNumber(row.metric_scored_slots_or_edges[metric], "integer") + (metric === "grid_transition_acc" ? " edges" : " slots");
      return '<span title="' + title + '">' + formatted + '</span><br><small class="metric-coverage">' + coverage + '</small>';
    }
    return '<span title="' + title + '">' + formatted + "</span>";
  }

  function compareRows(left, right) {
    var a = left[state.sortKey];
    var b = right[state.sortKey];
    if (a === null || a === undefined || a === "") return b === null || b === undefined || b === "" ? left.rank - right.rank : 1;
    if (b === null || b === undefined || b === "") return -1;
    var result;
    if (typeof a === "number" && typeof b === "number") result = a - b;
    else result = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
    if (result === 0) result = left.rank - right.rank;
    return state.sortDirection === "asc" ? result : -result;
  }

  function filteredRows() {
    if (!dataset || !dataset.tracks || !dataset.tracks[state.track]) return [];
    return dataset.tracks[state.track]
      .filter(function (row) {
        var searchable = (row.model + " " + row.method).toLowerCase();
        var matchesText = !state.search || searchable.indexOf(state.search) !== -1;
        var matchesSource = state.source === "all" || row.source === state.source;
        return matchesText && matchesSource;
      })
      .slice()
      .sort(compareRows);
  }

  function renderHead() {
    var cells = columns[state.track].map(function (column) {
      var active = state.sortKey === column.key;
      var ariaSort = active ? (state.sortDirection === "asc" ? "ascending" : "descending") : "none";
      var mark = active ? (state.sortDirection === "asc" ? "↑" : "↓") : "↕";
      return '<th scope="col" aria-sort="' + ariaSort + '"><button type="button" data-sort="' + column.key + '">' + escapeHtml(column.label) + '<span class="sort-mark" aria-hidden="true">' + mark + "</span></button></th>";
    });
    tableHead.innerHTML = "<tr>" + cells.join("") + "</tr>";
    tableHead.querySelectorAll("[data-sort]").forEach(function (button) {
      button.addEventListener("click", function () {
        var key = button.getAttribute("data-sort");
        if (state.sortKey === key) state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
        else {
          state.sortKey = key;
          state.sortDirection = key === "rank" || key === "model" || key === "method" || key === "cer" || key === "wer" || key === "grid_ade" || key === "grid_fde" || key === "recompute_ratio" || key === "reference_recompute_ratio" ? "asc" : "desc";
        }
        renderTable();
        var replacement = tableHead.querySelector('[data-sort="' + key + '"]');
        if (replacement) replacement.focus();
      });
    });
  }

  function renderBody(rows) {
    table.setAttribute("data-empty", rows.length ? "false" : "true");
    if (!rows.length) {
      var emptyMessage = state.track === "highmotion" && !highmotionV2 ? "High-Motion results withheld pending target/reference consistency review." : "No models match this filter.";
      tableBody.innerHTML = '<tr class="table-empty"><td colspan="' + columns[state.track].length + '">' + emptyMessage + '</td></tr>';
      return;
    }
    tableBody.innerHTML = rows.map(function (row) {
      var cells = columns[state.track].map(function (column) {
        var classes = [];
        if (column.key === "rank") classes.push("rank-cell");
        if (column.primary) classes.push("metric-primary");
        if (row[column.key] === null || row[column.key] === undefined || row[column.key] === "") classes.push("metric-missing");
        return '<td class="' + classes.join(" ") + '">' + cellContent(row, column) + "</td>";
      });
      return "<tr>" + cells.join("") + "</tr>";
    }).join("");
  }

  function renderTable() {
    if (!tableHead || !tableBody) return;
    var rows = filteredRows();
    renderHead();
    renderBody(rows);
    var total = dataset && dataset.tracks ? dataset.tracks[state.track].length : 0;
    tableCount.textContent = "Showing " + rows.length + " of " + total + " methods";
  }

  function bestRow(rows, key, lowerIsBetter, source) {
    return rows
      .filter(function (row) { return row[key] !== null && row[key] !== undefined && (!source || row.source === source); })
      .sort(function (a, b) { return lowerIsBetter ? a[key] - b[key] : b[key] - a[key]; })[0];
  }

  function summaryCard(label, row, metric, metricLabel, lowerIsBetter, source) {
    var winner = row || bestRow(dataset.tracks[state.track], metric, lowerIsBetter, source);
    if (!winner) return "";
    return '<div class="summary-card"><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(winner.model) + '</strong><small>' + escapeHtml(metricLabel) + " " + formatNumber(winner[metric], "score") + "</small></div>";
  }

  function renderSummary() {
    if (!summary || !dataset) return;
    var rows = dataset.tracks[state.track];
    if (state.track === "lpm") {
      var grtMethods = grtMethodIds();
      var grtRows = rows.filter(function (row) { return grtMethods.indexOf(row.method) !== -1; });
      var telemetryGrt = grtRows
        .filter(function (row) { return row.recompute_ratio !== null && row.recompute_ratio !== undefined; })
        .sort(function (a, b) { return a.recompute_ratio - b.recompute_ratio; })[0];
      var scoredCount = rows.filter(function (row) { return row.open_mos !== null && row.open_mos !== undefined; }).length;
      summary.innerHTML = [
        summaryCard("Highest reported Open MOS (" + scoredCount + "/" + rows.length + " scored)", null, "open_mos", "Open MOS", false),
        summaryCard("Top open model", null, "open_mos", "Open MOS", false, "open"),
        '<div class="summary-card"><span>Reported GRT telemetry (' + grtRows.length + ' verified profile' + (grtRows.length === 1 ? '' : 's') + ')</span><strong>' + (telemetryGrt ? formatNumber(telemetryGrt.recompute_ratio, "ratio") : "—") + ' lowest patch recompute</strong><small>' + (telemetryGrt ? escapeHtml(telemetryGrt.model) + ' · ' : '') + (telemetryGrt ? formatNumber(telemetryGrt.reference_recompute_ratio, "ratio") : "—") + ' reference compute · ' + (telemetryGrt ? formatNumber(telemetryGrt.effective_fps, "score") : "—") + ' sampling density · ' + (telemetryGrt ? formatNumber(telemetryGrt.throughput_fps, "score") : "—") + " throughput (fps)</small></div>"
      ].join("");
    } else if (highmotionV2) {
      var coverage = rows[0];
      var delta = highmotionV2.comparison.grt_minus_baseline.grid_acc;
      var deltaText = delta === null ? "—" : (delta > 0 ? "+" : "") + (delta === 0 ? "0" : Number(delta).toPrecision(5));
      summary.innerHTML = [
        '<div class="summary-card"><span>High-Motion v2 scope</span><strong>Fixed 1,000-record preview</strong><small>Not the full 3,243-record benchmark · ' +
          formatNumber(coverage.records_with_scored_slots, "integer") + ' rows with valid references · ' + formatNumber(coverage.valid_slots, "integer") + ' valid sampled slots</small></div>',
        '<div class="summary-card"><span>GRT vs corresponding 0.5B baseline</span><strong>' + escapeHtml(deltaText) +
          ' Grid Acc difference</strong><small>New GRT predictions compared with the same corrected-reference rescoring of the archived baseline. No statistical-significance claim.</small></div>',
        '<div class="summary-card"><span>Comparison evidence</span><strong>' + rows.length + ' methods · common reference mask</strong><small>' +
          escapeHtml(highmotionV2.comparison_caveat) + '</small></div>'
      ].join("");
    } else {
      summary.innerHTML = '<div class="summary-card"><span>High-Motion release status</span><strong>Results withheld</strong><small>Target/reference consistency review is incomplete. No GRT superiority claim is supported.</small></div>';
    }
  }

  function renderGrtQualification() {
    if (!grtQualification || !grtQualificationCount || !grtQualificationList) return;
    var families = dataset && Array.isArray(dataset.grtFamilies) ? dataset.grtFamilies : [];
    var hasV2Statuses = families.length > 0 && families.every(function (entry) {
      return entry && (entry.status === "promoted" || entry.status === "not_promoted");
    });

    grtQualificationList.textContent = "";
    if (!hasV2Statuses) {
      grtQualification.hidden = true;
      return;
    }

    var promoted = families.filter(function (entry) { return entry.status === "promoted"; });
    grtQualificationCount.textContent = promoted.length + " of " + families.length + " Educational GRT families promoted";

    families.forEach(function (entry) {
      var item = document.createElement("li");
      var family = document.createElement("strong");
      var status = document.createElement("span");
      var detail = document.createElement("small");
      var familyLabel = closedLabel(grtFamilyLabels, entry.family, "Unknown family");
      var reasonLabel = closedLabel(grtReasonLabels, entry.reasonCode, "Qualification reason unavailable");

      item.className = "summary-card";
      item.setAttribute("data-status", entry.status);
      family.textContent = familyLabel;
      status.textContent = entry.status === "promoted" ? "Promoted" : "Not promoted";
      status.setAttribute("aria-label", familyLabel + " qualification status: " + status.textContent);
      if (entry.status === "promoted") {
        detail.textContent = "Published method: " + String(entry.method || "—") + " · " + reasonLabel;
      } else {
        detail.textContent = "No leaderboard row · " + reasonLabel;
      }

      item.appendChild(family);
      item.appendChild(status);
      item.appendChild(detail);
      grtQualificationList.appendChild(item);
    });
    grtQualification.hidden = false;
  }

  function renderGlossary() {
    glossary.innerHTML = metricDefinitions[state.track].map(function (item) {
      return '<div class="metric-definition"><dt>' + escapeHtml(item[0]) + "</dt><dd>" + escapeHtml(item[1]) + "</dd></div>";
    }).join("");
  }

  function renderGrtComparison() {
    var body = document.getElementById("grt-comparison-body");
    if (!body || !publicAudit || !Array.isArray(publicAudit.families)) return;
    body.innerHTML = publicAudit.families.map(function (family) {
      return family.methods.map(function (row) {
        var values = [row.open_mos, row.token_f1, row.patch_ratio, row.throughput_fps, row.mean_wall_time_s];
        return '<tr><th scope="row">' + escapeHtml(family.label) + '<br><small>' + escapeHtml(row.label) + '</small></th>' + values.map(function (value, index) {
          return '<td title="' + escapeHtml(value === null || value === undefined || value === "" ? "Not reported" : value) + '">' + formatNumber(value, index === 2 ? "ratio" : "score") + '</td>';
        }).join("") + '</tr>';
      }).join("");
    }).join("");
  }

  function selectTrack(track) {
    if (track !== "lpm" && track !== "highmotion") return;
    state.track = track;
    state.sortKey = "rank";
    state.sortDirection = "asc";
    document.querySelectorAll(".track-tab").forEach(function (button) {
      var selected = button.getAttribute("data-track") === track;
      button.classList.toggle("is-active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    caption.textContent = track === "lpm" ? "DIVE-Bench Educational High-FPS Videos leaderboard" :
      highmotionV2 ? "DIVE-Bench High-Motion v2 · fixed 1,000-record preview (not full 3,243)" :
        "DIVE-Bench High-Motion High-FPS Videos: results withheld pending reference review";
    rankingRule.textContent = track === "lpm" ? "Open MOS (reported first; missing last), then Token F1" :
      highmotionV2 ? "Validated server ranks: Grid Acc descending; exact ties share rank; undefined values last" :
        "No ranking: target/reference review pending";
    renderSummary();
    renderGlossary();
    renderTable();
  }

  function snapshotDate(value) {
    var calendarDate = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value || ""));
    if (calendarDate) {
      return new Date(Date.UTC(Number(calendarDate[1]), Number(calendarDate[2]) - 1, Number(calendarDate[3])));
    }
    var parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function formatSnapshotDate(value, monthStyle) {
    var parsed = snapshotDate(value);
    if (!parsed) return null;
    return new Intl.DateTimeFormat("en", {
      day: "numeric",
      month: monthStyle,
      year: "numeric",
      timeZone: "UTC"
    }).format(parsed);
  }

  function renderDatasetMetadata() {
    var lpmRows = dataset.tracks && Array.isArray(dataset.tracks.lpm) ? dataset.tracks.lpm : [];
    var methodCount = document.getElementById("lpm-method-count");
    var artifactCount = document.getElementById("source-artifact-count");
    var judge = document.getElementById("open-mos-judge");
    var shortDate = formatSnapshotDate(dataset.generatedAt, "short");
    var longDate = formatSnapshotDate(dataset.generatedAt, "long");

    if (methodCount) methodCount.textContent = lpmRows.length.toLocaleString("en-US");
    if (artifactCount && Number.isFinite(Number(dataset.sourceArtifacts))) {
      artifactCount.textContent = Number(dataset.sourceArtifacts).toLocaleString("en-US");
    }
    if (judge && dataset.openMosJudge) judge.textContent = dataset.openMosJudge;
    if (shortDate) {
      ["snapshot-date", "footer-sync-date"].forEach(function (id) {
        var element = document.getElementById(id);
        if (element) element.textContent = shortDate;
      });
    }
    if (longDate) {
      var provenanceDate = document.getElementById("snapshot-date-long");
      if (provenanceDate) provenanceDate.textContent = longDate;
    }
  }

  function installLeaderboard() {
    if (!dataset) {
      if (table) table.setAttribute("data-empty", "true");
      if (tableBody) tableBody.innerHTML = '<tr class="table-empty"><td>Leaderboard data could not be loaded.</td></tr>';
      return;
    }
    renderDatasetMetadata();
    renderGrtQualification();
    renderGrtComparison();
    document.querySelectorAll(".track-tab").forEach(function (button) {
      button.addEventListener("click", function () { selectTrack(button.getAttribute("data-track")); });
      button.addEventListener("keydown", function (event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        var next = button.nextElementSibling || button.previousElementSibling;
        if (next) { next.focus(); next.click(); }
      });
    });
    searchInput.addEventListener("input", function () {
      state.search = searchInput.value.trim().toLowerCase();
      renderTable();
    });
    sourceFilter.addEventListener("change", function () {
      state.source = sourceFilter.value;
      renderTable();
    });
    selectTrack("lpm");
  }

  function installCodeTabs() {
    var activeCode = document.getElementById("active-code");
    var filename = document.getElementById("code-filename");
    var panel = document.getElementById("code-panel");
    var tabs = Array.prototype.slice.call(document.querySelectorAll(".code-tab"));

    function activate(button) {
      var key = button.getAttribute("data-code");
      tabs.forEach(function (candidate) {
        var selected = candidate === button;
        candidate.classList.toggle("is-active", selected);
        candidate.setAttribute("aria-selected", String(selected));
        candidate.tabIndex = selected ? 0 : -1;
      });
      activeCode.textContent = codeSamples[key].value;
      filename.textContent = codeSamples[key].filename;
      if (panel) panel.setAttribute("aria-labelledby", button.id);
    }

    tabs.forEach(function (button, index) {
      button.addEventListener("click", function () {
        activate(button);
      });
      button.addEventListener("keydown", function (event) {
        var nextIndex = null;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        tabs[nextIndex].focus();
        activate(tabs[nextIndex]);
      });
    });
  }

  function copyText(text, button) {
    function showStatus(message) {
      var previous = button.getAttribute("data-copy-label") || button.textContent;
      button.setAttribute("data-copy-label", previous);
      button.textContent = message;
      clearTimeout(button.copyStatusTimer);
      button.copyStatusTimer = setTimeout(function () {
        button.textContent = previous;
        button.removeAttribute("data-copy-label");
      }, 1400);
    }

    function confirm() {
      showStatus("Copied");
    }

    function reject() {
      showStatus("Copy failed");
    }

    function copyFallback() {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        if (document.execCommand("copy")) confirm();
        else reject();
      } catch (error) { reject(); }
      document.body.removeChild(textarea);
    }

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(confirm).catch(copyFallback);
      return;
    }
    copyFallback();
  }

  function installCopyButtons() {
    document.querySelectorAll("[data-copy-target]").forEach(function (button) {
      button.addEventListener("click", function () {
        var target = document.getElementById(button.getAttribute("data-copy-target"));
        if (target) copyText(target.textContent, button);
      });
    });
  }

  function installNavigation() {
    var toggle = document.querySelector(".nav-toggle");
    var nav = document.getElementById("site-nav");
    if (!toggle || !nav) return;
    toggle.addEventListener("click", function () {
      var open = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!open));
      nav.classList.toggle("is-open", !open);
    });
    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        toggle.setAttribute("aria-expanded", "false");
        nav.classList.remove("is-open");
      });
    });
  }

  installLeaderboard();
  installCodeTabs();
  installCopyButtons();
  installNavigation();
})();
