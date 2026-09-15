"""Synthetic numeric contributions only; no models, predictions or source data."""

import hashlib
import json
from copy import deepcopy

import pytest

from tools.densevideo import highmotion_v2_bundle as bundle
from tools.densevideo.highmotion_v2_scoring import METRICS, aggregate_scores

GRT = "synthetic_grt_v2"
REFERENCE_SHA = "a" * 64
INPUT_SHA = "b" * 64


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def numeric_row(index, *, grt=False, invalid=False):
    values = {
        "grid_acc": 0.625 if grt else 0.5,
        "grid_ade": 0.375 if grt else 0.5,
        "grid_fde": 0.0 if grt else 0.5,
        "grid_transition_acc": 4 / 7 if grt else 3 / 7,
        "token_f1": 0.75 if grt else 0.625,
    }
    denominators = dict.fromkeys(METRICS, 8)
    denominators.update(grid_fde=1, grid_transition_acc=7)
    if invalid:
        values = dict.fromkeys(METRICS)
        denominators = dict.fromkeys(METRICS, 0)
    return {
        "doc_id": index, "identity_sha256": digest(f"synthetic identity {index}"),
        "prompt_sha256": digest(f"synthetic prompt {index}"),
        "source_frame_indices": [0, 2, 4, 6, 8, 10, 12, 14],
        "sampled_slots": 8, "valid_slots": 0 if invalid else 8,
        "prediction_slots": 8, "malformed_or_missing_scored_slots": 0,
        "surplus_prediction_slots": 0,
        "metrics": values, "denominators": denominators,
    }


def report(method, *, invalid=False):
    grt = method == GRT
    rows = [numeric_row(index, grt=grt, invalid=invalid)
            for index in range(bundle.PREVIEW_SAMPLES)]
    return {
        **aggregate_scores(rows), "method": method,
        "status": "rescored_cached_predictions", "scope": "first-1000-source-rows",
        "prediction_source": "new_grt" if grt else "archived_baseline",
        "references_sha256": REFERENCE_SHA, "input_sequence_sha256": INPUT_SHA,
        "predictions_sha256": digest(method), "prompt_cache_sha256": digest(bundle.BASELINE_METHOD),
        "source_reference_records": 3243, "full_source_coverage": False,
        "inference_performed": False, "automatic_publication": False,
        "grt_superiority_verified": False, "records_detail": rows,
    }


@pytest.fixture(scope="module")
def original_reports():
    return [report(method) for method in (*bundle.BASELINE_METHODS, GRT)]


@pytest.fixture
def reports(original_reports):
    return deepcopy(original_reports)


def receipt():
    return {
        "generation_completed": True, "preview_samples": 1000, "exit_code": 0,
        "samples_sha256": digest(GRT), "results_sha256": "d" * 64,
        "worker_log_sha256": "e" * 64, "legacy_scores_are_publishable": False,
        "full3243_benchmark_completed": False, "baseline_gpu_rerun": False,
        "winner_claim": False, "automatic_publication": False,
        "baseline_exact_weight_or_tensor_identity_proven": False,
    }


def validate(reports, **kwargs):
    return bundle.validate_reports(
        reports, reference_sha256=kwargs.pop("reference_sha256", REFERENCE_SHA),
        input_sequence_sha256=kwargs.pop("input_sequence_sha256", INPUT_SHA),
        grt_method=kwargs.pop("grt_method", GRT), grt_receipt=kwargs.pop("grt_receipt", receipt()),
        **kwargs,
    )


def reaggregate(report):
    report.update(aggregate_scores(report["records_detail"]))


def test_complete_reports_rank_and_compare_without_mutation(original_reports):
    before = json.dumps(original_reports, sort_keys=True)
    summary = validate(original_reports)
    assert json.dumps(original_reports, sort_keys=True) == before
    assert summary["method_count"] == 19
    assert summary["rows"][0]["method"] == GRT
    assert summary["rows"][0]["rank"] == 1
    assert [row["method"] for row in summary["rows"][1:]] == sorted(bundle.BASELINE_METHODS)
    assert {row["rank"] for row in summary["rows"][1:]} == {2}
    assert summary["comparison"]["grid_acc_outperform"] is True
    assert summary["comparison"]["grt_minus_baseline"]["grid_ade"] == -0.125
    assert summary["comparison"]["oriented_improvements"]["grid_ade"] == 0.125
    assert set(summary["comparison"]["metric_outperform"]) == set(METRICS)
    for row in summary["rows"]:
        assert row["samples"] == 1000 and row["sampled_slots"] == row["valid_slots"] == 8000
        assert row["metric_scored_slots_or_edges"]["grid_transition_acc"] == 7000
    for key in ("inference_performed", "baseline_gpu_rerun", "automatic_publication",
                "statistical_significance_claim", "underlying_file_hashes_verified_by_this_function",
                "baseline_exact_weight_or_tensor_identity_proven", "full_source_coverage"):
        assert summary[key] is False
    assert "unproven" in summary["comparison_caveat"]
    assert all("records_detail" not in row for row in summary["rows"])


@pytest.mark.parametrize("candidate_acc", [0.375, 0.5, 0.5 + 1e-13])
def test_negative_tied_and_roundoff_outcomes_are_retained(reports, candidate_acc):
    for row in reports[-1]["records_detail"]:
        row["metrics"]["grid_acc"] = candidate_acc
    reaggregate(reports[-1])
    summary = validate(reports)
    assert summary["comparison"]["grid_acc_outperform"] is False
    assert len(summary["rows"]) == 19 and GRT in {row["method"] for row in summary["rows"]}
    assert summary["comparison"]["grt_minus_baseline"]["grid_acc"] == candidate_acc - 0.5


def test_all_invalid_coverage_remains_null_unranked_and_not_a_win():
    summary = validate([report(method, invalid=True) for method in (*bundle.BASELINE_METHODS, GRT)])
    assert all(row["rank"] is None and row["grid_acc"] is None for row in summary["rows"])
    assert all(row["records_without_scored_slots"] == 1000 for row in summary["rows"])
    assert summary["comparison"]["grt_minus_baseline"] == dict.fromkeys(METRICS)
    assert summary["comparison"]["grid_acc_outperform"] is False


def test_shared_partial_coverage_keeps_metric_specific_macro_denominators(reports):
    for item in reports:
        row = item["records_detail"][0]
        grt = item["method"] == GRT
        row["valid_slots"] = 3
        row["denominators"] = dict.fromkeys(METRICS, 3)
        row["denominators"].update(grid_fde=0, grid_transition_acc=1)
        row["metrics"].update(grid_acc=2 / 3 if grt else 1 / 3,
                              grid_fde=None, grid_transition_acc=1 if grt else 0)
        reaggregate(item)
    summary = validate(reports)
    for row in summary["rows"]:
        assert row["valid_slots"] == 7995
        assert row["metric_scored_records"]["grid_acc"] == 1000
        assert row["metric_scored_records"]["grid_fde"] == 999
        assert row["metric_scored_records"]["grid_transition_acc"] == 1000
        assert row["metric_scored_slots_or_edges"]["grid_transition_acc"] == 6994


def test_input_method_order_does_not_affect_ranking_or_comparison(original_reports):
    assert validate(original_reports) == validate(list(reversed(original_reports)))


def test_valid_but_unbound_prompt_cache_sha_is_rejected(reports):
    reports[-1]["prompt_cache_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Prompt cache must be the fixed"):
        validate(reports)


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "unknown", "missing-grt"])
def test_exact_fixed_method_set_is_required(reports, case):
    if case == "missing":
        reports.pop(0)
    elif case == "extra":
        reports.append(deepcopy(reports[0]))
    elif case == "duplicate":
        reports[1] = deepcopy(reports[0])
    elif case == "unknown":
        reports[0]["method"] = "unapproved_model"
    else:
        reports[-1]["method"] = bundle.BASELINE_METHOD
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("key,value", [
    ("benchmark_version", "legacy"), ("scorer_version", "other"),
    ("target_joint", "leftIndexFingerMetacarpal"), ("reference_policy", "other"),
    ("aggregation", "micro mean"), ("status", "passed"), ("scope", "first-2-source-rows"),
    ("references_sha256", "f" * 64), ("input_sequence_sha256", "f" * 64),
    ("predictions_sha256", "bad"), ("prompt_cache_sha256", "bad"),
    ("source_reference_records", 1000), ("full_source_coverage", True),
    ("automatic_publication", True), ("inference_performed", True),
    ("grt_superiority_verified", True), ("prediction_source", "new_grt"),
])
def test_report_version_scope_pins_roles_and_claims_fail_closed(reports, key, value):
    reports[0][key] = value
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("key,value", [
    ("generation_completed", False), ("generation_completed", 1),
    ("preview_samples", 2), ("exit_code", 1), ("exit_code", False),
    ("samples_sha256", "f" * 64), ("results_sha256", "bad"),
    ("worker_log_sha256", "bad"), ("legacy_scores_are_publishable", True),
    ("baseline_gpu_rerun", True), ("winner_claim", True), ("automatic_publication", True),
    ("full3243_benchmark_completed", True),
    ("baseline_exact_weight_or_tensor_identity_proven", True), ("validation_error", "failed"),
])
def test_incomplete_unbound_or_overclaiming_generation_receipt_is_rejected(original_reports, key, value):
    bad = receipt()
    bad[key] = value
    with pytest.raises(ValueError):
        validate(original_reports, grt_receipt=bad)


@pytest.mark.parametrize("field", ["doc_id", "identity_sha256", "prompt_sha256",
                                   "source_frame_indices", "coverage"])
def test_per_row_identity_prompt_and_coverage_must_match(reports, field):
    row = reports[0]["records_detail"][0]
    if field == "doc_id":
        row[field] = 1
    elif field in ("identity_sha256", "prompt_sha256"):
        row[field] = "f" * 64
    elif field == "source_frame_indices":
        row[field] = list(range(8))
    else:
        row.update(numeric_row(0, invalid=True))
        reaggregate(reports[0])
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("case", ["partial", "reordered", "duplicate-identity", "bad-indices",
                                  "bool-id", "impossible-edges", "impossible-final",
                                  "wrong-valid-denom", "wrong-surplus", "too-many-malformed"])
def test_malformed_numeric_structure_is_rejected(reports, case):
    rows = reports[0]["records_detail"]
    row = rows[0]
    if case == "partial":
        rows.pop()
    elif case == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif case == "duplicate-identity":
        rows[1]["identity_sha256"] = row["identity_sha256"]
    elif case == "bad-indices":
        row["source_frame_indices"][1] = 3
    elif case == "bool-id":
        row["doc_id"] = False
    elif case == "impossible-edges":
        row["denominators"]["grid_transition_acc"] = 6
    elif case == "impossible-final":
        row["denominators"]["grid_fde"] = 0
        row["metrics"]["grid_fde"] = None
    elif case == "wrong-valid-denom":
        row["denominators"]["grid_acc"] = 7
    elif case == "wrong-surplus":
        row["surplus_prediction_slots"] = 1
    else:
        row["malformed_or_missing_scored_slots"] = 9
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_row_metric_values_are_rejected(reports, value):
    reports[0]["records_detail"][0]["metrics"]["grid_acc"] = value
    with pytest.raises(ValueError):
        validate(reports)


@pytest.mark.parametrize("field", ["metrics", "records", "sampled_slots", "valid_slots",
                                   "records_with_scored_slots", "records_without_scored_slots",
                                   "metric_scored_records", "metric_scored_slots_or_edges"])
def test_every_stored_aggregate_is_independently_checked(reports, field):
    if field == "metrics":
        reports[0][field]["grid_acc"] += 1e-10
    elif field.startswith("metric_"):
        reports[0][field]["grid_acc"] += 1
    else:
        reports[0][field] += 1
    with pytest.raises(ValueError):
        validate(reports)


def test_minute_serialization_difference_is_not_used_for_ranking_or_delta(reports):
    reports[0]["metrics"]["grid_acc"] += 5e-16
    summary = validate(reports)
    row = next(row for row in summary["rows"] if row["method"] == reports[0]["method"])
    assert row["grid_acc"] == 0.5 and row["rank"] == 2


@pytest.mark.parametrize("kwargs", [
    {"reference_sha256": "bad"}, {"input_sequence_sha256": "bad"},
    {"grt_method": bundle.BASELINE_METHOD}, {"grt_method": ""}, {"grt_method": "<script>"},
])
def test_bad_caller_pins_or_grt_method_fail_closed(original_reports, kwargs):
    with pytest.raises(ValueError):
        validate(original_reports, **kwargs)
