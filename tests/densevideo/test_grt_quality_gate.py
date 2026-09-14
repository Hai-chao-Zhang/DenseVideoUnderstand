import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.densevideo.check_grt_quality_gate import (  # noqa: E402
    _parse_reference_ratio_overrides,
    select_candidate,
)


def _row(score, recompute, samples=128, reference_recompute=None):
    row = {
        "token_f1": str(score),
        "mean_recompute_ratio": str(recompute),
        "samples": str(samples),
    }
    if reference_recompute is not None:
        row["reference_patch_compute_ratio"] = str(reference_recompute)
    return row


def test_selects_highest_candidate_that_beats_both_controls():
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        "grt_llava_ov_0_5b_quality_t001": _row(0.0165, 0.24),
        "grt_llava_ov_0_5b_quality_t005": _row(0.0162, 0.20),
    }

    result = select_candidate(rows, expected_samples=128)

    assert result["selected_method"] == "grt_llava_ov_0_5b_quality_t001"
    assert result["required_score"] == pytest.approx(0.0160)
    assert result["passing_candidates"] == [
        "grt_llava_ov_0_5b_quality_t001",
        "grt_llava_ov_0_5b_quality_t005",
    ]
    assert [item["method"] for item in result["passing_candidate_details"]] == result[
        "passing_candidates"
    ]


def test_rejects_equal_score_or_unmeasured_recompute():
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        "grt_llava_ov_0_5b_quality_t001": _row(0.0160, 0.24),
        "grt_llava_ov_0_5b_quality_t005": _row(0.0170, ""),
    }

    with pytest.raises(ValueError, match="No GRT candidate strictly beats"):
        select_candidate(rows, expected_samples=128)


def test_recompute_gate_is_inclusive_and_cli_threshold_is_configurable():
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        "grt_llava_ov_0_5b_quality_t001": _row(0.0170, 0.99),
        "grt_llava_ov_0_5b_quality_t005": _row(0.0165, 0.98),
    }

    default_result = select_candidate(rows, expected_samples=128)
    assert default_result["selected_method"] == "grt_llava_ov_0_5b_quality_t005"
    assert default_result["max_recompute_ratio"] == pytest.approx(0.98)

    relaxed_result = select_candidate(
        rows,
        expected_samples=128,
        max_recompute_ratio=0.99,
    )
    assert relaxed_result["selected_method"] == "grt_llava_ov_0_5b_quality_t001"


def test_requires_both_primary_mos_and_secondary_token_f1():
    rows = {
        "llava_onevision_0_5b_quality_base": {
            **_row(0.0140, 1.0),
            "open_mos": "0.8",
        },
        "llava_ov_0_5b_same_wrapper_base": {
            **_row(0.0160, 1.0),
            "open_mos": "0.7",
        },
        "grt_llava_ov_0_5b_quality_t001": {
            **_row(0.0165, 0.24),
            "open_mos": "0.81",
        },
        "grt_llava_ov_0_5b_quality_t005": {
            **_row(0.0162, 0.20),
            "open_mos": "0.79",
        },
    }

    result = select_candidate(
        rows,
        expected_samples=128,
        metric="open_mos",
        secondary_metric="token_f1",
    )

    assert result["selected_method"] == "grt_llava_ov_0_5b_quality_t001"
    assert result["selected_secondary_score"] == pytest.approx(0.0165)
    assert result["candidate_secondary_scores"] == {
        "grt_llava_ov_0_5b_quality_t001": pytest.approx(0.0165),
        "grt_llava_ov_0_5b_quality_t005": pytest.approx(0.0162),
    }
    assert result["passing_candidates"] == ["grt_llava_ov_0_5b_quality_t001"]
    assert result["passing_candidate_details"] == [
        {
            "method": "grt_llava_ov_0_5b_quality_t001",
            "score": pytest.approx(0.81),
            "secondary_score": pytest.approx(0.0165),
            "recompute_ratio": pytest.approx(0.24),
            "reference_recompute_ratio": None,
            "reference_recompute_ratio_source": "missing",
        }
    ]


def test_followup_floor_is_max_of_archived_fresh_base_and_all_for_both_metrics():
    baselines = ("archived", "fresh_base", "all_control")
    rows = {
        "archived": {**_row(0.10, 1.0), "open_mos": "0.90"},
        "fresh_base": {**_row(0.30, 1.0), "open_mos": "0.70"},
        "all_control": {**_row(0.20, 1.0), "open_mos": "0.80"},
        "passes_both_maxima": {**_row(0.31, 0.80, reference_recompute=0.80), "open_mos": "0.91"},
        "misses_archived_mos": {**_row(0.40, 0.70, reference_recompute=0.70), "open_mos": "0.90"},
        "misses_fresh_f1": {**_row(0.30, 0.60, reference_recompute=0.60), "open_mos": "0.95"},
    }

    result = select_candidate(
        rows,
        baselines=baselines,
        candidates=("misses_archived_mos", "misses_fresh_f1", "passes_both_maxima"),
        expected_samples=128,
        metric="open_mos",
        secondary_metric="token_f1",
        max_reference_recompute_ratio=0.98,
    )

    assert result["required_score"] == pytest.approx(0.90)
    assert result["secondary_required_score"] == pytest.approx(0.30)
    assert result["passing_candidates"] == ["passes_both_maxima"]


def test_passing_candidates_use_the_documented_stable_rank_order():
    rows = {
        "llava_onevision_0_5b_quality_base": {**_row(0.0140, 1.0), "open_mos": "0.70"},
        "llava_ov_0_5b_same_wrapper_base": {**_row(0.0160, 1.0), "open_mos": "0.75"},
        "quality_first": {**_row(0.0180, 0.50), "open_mos": "0.90"},
        "secondary_first": {**_row(0.0190, 0.60), "open_mos": "0.80"},
        "compute_tiebreak": {**_row(0.0190, 0.40), "open_mos": "0.80"},
        "method_tiebreak_b": {**_row(0.0190, 0.40), "open_mos": "0.80"},
        "method_tiebreak_a": {**_row(0.0190, 0.40), "open_mos": "0.80"},
    }

    result = select_candidate(
        rows,
        candidates=(
            "method_tiebreak_b",
            "quality_first",
            "secondary_first",
            "compute_tiebreak",
            "method_tiebreak_a",
        ),
        expected_samples=128,
        metric="open_mos",
        secondary_metric="token_f1",
    )

    assert result["passing_candidates"] == [
        "quality_first",
        "compute_tiebreak",
        "method_tiebreak_a",
        "method_tiebreak_b",
        "secondary_first",
    ]


def test_reference_recompute_gate_is_inclusive_and_strictly_requires_new_field():
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        # Best quality fails the reference-compute ceiling.
        "grt_llava_ov_0_5b_quality_t001": _row(0.0180, 0.20, reference_recompute=0.91),
        # The boundary is inclusive.
        "grt_llava_ov_0_5b_quality_t005": _row(0.0170, 0.30, reference_recompute=0.90),
    }

    result = select_candidate(
        rows,
        expected_samples=128,
        max_reference_recompute_ratio=0.90,
    )

    assert result["selected_method"] == "grt_llava_ov_0_5b_quality_t005"
    assert result["selected_reference_recompute_ratio"] == pytest.approx(0.90)
    assert result["selected_reference_patch_compute_ratio"] == pytest.approx(0.90)
    assert result["max_reference_recompute_ratio"] == pytest.approx(0.90)

    rows["grt_llava_ov_0_5b_quality_t005"].pop("reference_patch_compute_ratio")
    with pytest.raises(ValueError, match="reference_recompute_ratios"):
        select_candidate(
            rows,
            expected_samples=128,
            max_reference_recompute_ratio=0.90,
        )


def test_reference_recompute_gate_validates_range_but_is_optional_for_legacy_rows():
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        "grt_llava_ov_0_5b_quality_t001": _row(0.0170, 0.20),
    }

    result = select_candidate(rows, expected_samples=128)
    assert result["selected_method"] == "grt_llava_ov_0_5b_quality_t001"
    assert result["selected_reference_recompute_ratio"] is None

    with pytest.raises(ValueError, match="max_reference_recompute_ratio"):
        select_candidate(rows, max_reference_recompute_ratio=1.01)


def test_explicit_legacy_reference_override_is_audited_and_enables_gate():
    legacy_method = "grt_llava_onevision_0_5b_hf_t0001"
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        legacy_method: _row(0.0170, 0.8229729218750002),
    }

    result = select_candidate(
        rows,
        candidates=[legacy_method],
        expected_samples=128,
        max_reference_recompute_ratio=0.90,
        reference_ratio_overrides={legacy_method: 0.8229729218750002},
    )

    assert result["selected_method"] == legacy_method
    assert result["selected_reference_patch_compute_ratio"] == pytest.approx(
        0.8229729218750002
    )
    assert result["selected_reference_patch_compute_ratio_source"] == "override"
    assert result["reference_ratio_overrides"] == {
        legacy_method: pytest.approx(0.8229729218750002)
    }
    assert result["candidate_reference_patch_compute_ratio_sources"] == {
        legacy_method: "override"
    }


def test_reference_override_never_replaces_populated_or_invalid_summary_field():
    legacy_method = "grt_llava_onevision_0_5b_hf_t0001"
    rows = {
        "llava_onevision_0_5b_quality_base": _row(0.0140, 1.0),
        "llava_ov_0_5b_same_wrapper_base": _row(0.0160, 1.0),
        legacy_method: _row(0.0170, 0.82, reference_recompute=0.95),
    }

    with pytest.raises(ValueError, match="'summary'"):
        select_candidate(
            rows,
            candidates=[legacy_method],
            max_reference_recompute_ratio=0.90,
            reference_ratio_overrides={legacy_method: 0.80},
        )

    rows[legacy_method]["reference_patch_compute_ratio"] = "not-a-number"
    with pytest.raises(ValueError, match="'invalid_summary'"):
        select_candidate(
            rows,
            candidates=[legacy_method],
            max_reference_recompute_ratio=0.90,
            reference_ratio_overrides={legacy_method: 0.80},
        )


def test_reference_override_cli_parser_is_repeatable_and_strict():
    assert _parse_reference_ratio_overrides(["raw=0.822", "adaptive=0"]) == {
        "raw": pytest.approx(0.822),
        "adaptive": pytest.approx(0.0),
    }

    for values in (
        ["missing-equals"],
        ["=0.5"],
        ["raw=not-a-number"],
        ["raw=-0.1"],
        ["raw=nan"],
        ["raw=inf"],
        ["raw=0.5", "raw=0.4"],
    ):
        with pytest.raises(ValueError):
            _parse_reference_ratio_overrides(values)
