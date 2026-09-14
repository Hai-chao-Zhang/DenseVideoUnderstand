#!/usr/bin/env python3
"""Select a GRT candidate only when it beats its configured quality controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


DEFAULT_BASELINES = (
    "llava_onevision_0_5b_quality_base",
    "llava_ov_0_5b_same_wrapper_base",
)
DEFAULT_CANDIDATES = (
    "grt_llava_ov_0_5b_quality_t001",
    "grt_llava_ov_0_5b_quality_t005",
)


def _to_float(value) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_reference_ratio_overrides(
    overrides: Optional[Mapping[str, object]],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for raw_method, raw_value in (overrides or {}).items():
        method = str(raw_method).strip()
        if not method:
            raise ValueError("reference ratio override method must not be empty")
        value = _to_float(raw_value)
        if value is None or not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"reference ratio override for {method!r} must be a finite non-negative number; "
                f"got {raw_value!r}"
            )
        normalized[method] = value
    return normalized


def _parse_reference_ratio_overrides(values: Iterable[str]) -> Dict[str, float]:
    overrides: Dict[str, str] = {}
    for raw_value in values:
        if raw_value.count("=") != 1:
            raise ValueError(
                f"Invalid --reference-ratio-override {raw_value!r}; expected METHOD=FLOAT"
            )
        raw_method, raw_ratio = raw_value.split("=", 1)
        method = raw_method.strip()
        if not method:
            raise ValueError(
                f"Invalid --reference-ratio-override {raw_value!r}; method must not be empty"
            )
        if method in overrides:
            raise ValueError(f"Duplicate --reference-ratio-override for method {method!r}")
        overrides[method] = raw_ratio.strip()
    return _normalize_reference_ratio_overrides(overrides)


def _reference_ratio_for_candidate(
    row: Dict[str, str],
    method: str,
    overrides: Mapping[str, float],
) -> Tuple[Optional[float], str]:
    field = "reference_patch_compute_ratio"
    raw_value = row.get(field)
    field_missing = field not in row or raw_value is None or str(raw_value).strip() == ""
    if not field_missing:
        value = _to_float(raw_value)
        if value is None or not math.isfinite(value) or value < 0.0:
            return None, "invalid_summary"
        return value, "summary"
    if method in overrides:
        return overrides[method], "override"
    return None, "missing"


def _read_latest_rows(paths: Iterable[Path]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("task") != "densevideo":
                    continue
                if str(row.get("run_status", "")).lower() == "failed":
                    continue
                method = str(row.get("method", "")).strip()
                if method:
                    rows[method] = row
    return rows


def _read_mos_scores(paths: Iterable[Path]) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("error", "")).strip():
                    continue
                method = str(row.get("method", "")).strip()
                score = _to_float(row.get("open_mos_score"))
                if method and score is not None:
                    scores.setdefault(method, []).append(score)
    return {method: sum(values) / len(values) for method, values in scores.items() if values}


def select_candidate(
    rows: Dict[str, Dict[str, str]],
    *,
    baselines: Iterable[str] = DEFAULT_BASELINES,
    candidates: Iterable[str] = DEFAULT_CANDIDATES,
    expected_samples: Optional[int] = None,
    metric: str = "token_f1",
    min_margin: float = 0.0,
    secondary_metric: Optional[str] = None,
    secondary_min_margin: float = 0.0,
    max_recompute_ratio: float = 0.98,
    max_reference_recompute_ratio: Optional[float] = None,
    reference_ratio_overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    max_recompute_ratio = float(max_recompute_ratio)
    if not 0.0 <= max_recompute_ratio <= 1.0:
        raise ValueError(
            f"max_recompute_ratio must be between 0 and 1; got {max_recompute_ratio}"
        )
    if max_reference_recompute_ratio is not None:
        max_reference_recompute_ratio = float(max_reference_recompute_ratio)
        if not 0.0 <= max_reference_recompute_ratio <= 1.0:
            raise ValueError(
                "max_reference_recompute_ratio must be between 0 and 1; "
                f"got {max_reference_recompute_ratio}"
            )
    reference_ratio_overrides = _normalize_reference_ratio_overrides(reference_ratio_overrides)

    baseline_scores: Dict[str, float] = {}
    baseline_secondary_scores: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}
    for method in baselines:
        row = rows.get(method)
        score = _to_float(row.get(metric)) if row else None
        samples = _to_float(row.get("samples")) if row else None
        if row is None or score is None or samples is None:
            raise ValueError(f"Missing usable {metric}/samples for baseline {method}")
        sample_counts[method] = int(samples)
        if expected_samples is not None and int(samples) != expected_samples:
            raise ValueError(f"Baseline {method} has {int(samples)} samples; expected {expected_samples}")
        baseline_scores[method] = score
        if secondary_metric:
            secondary_score = _to_float(row.get(secondary_metric))
            if secondary_score is None:
                raise ValueError(f"Missing usable {secondary_metric} for baseline {method}")
            baseline_secondary_scores[method] = secondary_score

    target = max(baseline_scores.values()) + float(min_margin)
    secondary_target = (
        max(baseline_secondary_scores.values()) + float(secondary_min_margin)
        if secondary_metric
        else None
    )
    passing: List[Dict[str, object]] = []
    candidate_scores: Dict[str, float] = {}
    candidate_secondary_scores: Dict[str, Optional[float]] = {}
    candidate_recompute_ratios: Dict[str, Optional[float]] = {}
    candidate_reference_recompute_ratios: Dict[str, Optional[float]] = {}
    candidate_reference_recompute_ratio_sources: Dict[str, str] = {}
    for method in candidates:
        row = rows.get(method)
        score = _to_float(row.get(metric)) if row else None
        samples = _to_float(row.get("samples")) if row else None
        recompute = _to_float(row.get("mean_recompute_ratio")) if row else None
        # This is intentionally strict: an override is allowed only for a
        # genuinely absent legacy field, never as a fallback for malformed or
        # failing summary telemetry.
        reference_recompute, reference_recompute_source = (
            _reference_ratio_for_candidate(row, method, reference_ratio_overrides)
            if row
            else (None, "missing")
        )
        secondary_score = _to_float(row.get(secondary_metric)) if row and secondary_metric else None
        if row is None or score is None or samples is None:
            continue
        sample_counts[method] = int(samples)
        candidate_scores[method] = score
        candidate_secondary_scores[method] = secondary_score
        candidate_recompute_ratios[method] = recompute
        candidate_reference_recompute_ratios[method] = reference_recompute
        candidate_reference_recompute_ratio_sources[method] = reference_recompute_source
        if expected_samples is not None and int(samples) != expected_samples:
            continue
        # A candidate with no measured savings is not a meaningful GRT win.
        if recompute is None or not (0.0 <= recompute <= max_recompute_ratio):
            continue
        if max_reference_recompute_ratio is not None and (
            reference_recompute is None
            or not (0.0 <= reference_recompute <= max_reference_recompute_ratio)
        ):
            continue
        if secondary_metric and (secondary_score is None or secondary_score <= secondary_target):
            continue
        if score > target:
            passing.append(
                {
                    "method": method,
                    "score": score,
                    "secondary_score": secondary_score,
                    "recompute_ratio": recompute,
                    "reference_recompute_ratio": reference_recompute,
                    "reference_recompute_ratio_source": reference_recompute_source,
                }
            )

    if not passing:
        raise ValueError(
            f"No GRT candidate strictly beats all controls on {metric}; "
            f"target>{target:.12g}, max_recompute_ratio<={max_recompute_ratio:.12g}, "
            f"max_reference_recompute_ratio={max_reference_recompute_ratio}, "
            f"candidates={candidate_scores}, recompute_ratios={candidate_recompute_ratios}, "
            f"secondary_scores={candidate_secondary_scores}, "
            f"reference_recompute_ratios={candidate_reference_recompute_ratios}, "
            f"reference_recompute_ratio_sources={candidate_reference_recompute_ratio_sources}, "
            f"reference_ratio_overrides={reference_ratio_overrides}"
        )

    passing.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["secondary_score"]) if item["secondary_score"] is not None else 0.0,
            float(item["recompute_ratio"]),
            str(item["method"]),
        )
    )
    winner = passing[0]
    passing_candidates = [str(item["method"]) for item in passing]
    passing_candidate_details = [dict(item) for item in passing]
    return {
        "status": "passed",
        "metric": metric,
        "min_margin": min_margin,
        "required_score": target,
        "selected_method": winner["method"],
        "selected_score": winner["score"],
        "selected_recompute_ratio": winner["recompute_ratio"],
        "max_recompute_ratio": max_recompute_ratio,
        "selected_reference_recompute_ratio": winner["reference_recompute_ratio"],
        "selected_reference_patch_compute_ratio": winner["reference_recompute_ratio"],
        "selected_reference_recompute_ratio_source": winner["reference_recompute_ratio_source"],
        "selected_reference_patch_compute_ratio_source": winner["reference_recompute_ratio_source"],
        "max_reference_recompute_ratio": max_reference_recompute_ratio,
        "baseline_scores": baseline_scores,
        "secondary_metric": secondary_metric,
        "secondary_required_score": secondary_target,
        "selected_secondary_score": winner["secondary_score"],
        "baseline_secondary_scores": baseline_secondary_scores,
        "candidate_scores": candidate_scores,
        "candidate_secondary_scores": candidate_secondary_scores,
        "passing_candidates": passing_candidates,
        "passing_candidate_details": passing_candidate_details,
        "candidate_recompute_ratios": candidate_recompute_ratios,
        "candidate_reference_recompute_ratios": candidate_reference_recompute_ratios,
        "candidate_reference_patch_compute_ratios": candidate_reference_recompute_ratios,
        "candidate_reference_recompute_ratio_sources": candidate_reference_recompute_ratio_sources,
        "candidate_reference_patch_compute_ratio_sources": candidate_reference_recompute_ratio_sources,
        "reference_ratio_overrides": reference_ratio_overrides,
        "sample_counts": sample_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", action="append", required=True)
    parser.add_argument("--baseline", action="append", default=[], help="Baseline method id; repeat for multiple controls.")
    parser.add_argument("--candidate", action="append", default=[], help="GRT candidate method id; repeat for a sweep.")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--metric", default="token_f1")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--secondary-metric", default="")
    parser.add_argument("--secondary-min-margin", type=float, default=0.0)
    parser.add_argument(
        "--max-recompute-ratio",
        type=float,
        default=0.98,
        help="Maximum inclusive recompute ratio for a useful GRT candidate (default: 0.98).",
    )
    parser.add_argument(
        "--max-reference-recompute-ratio",
        type=float,
        default=None,
        help=(
            "Optional maximum inclusive patch compute ratio against the reference route. "
            "Candidates must provide reference_patch_compute_ratio when this is set."
        ),
    )
    parser.add_argument(
        "--reference-ratio-override",
        action="append",
        default=[],
        metavar="METHOD=FLOAT",
        help=(
            "Explicit reference ratio for one legacy method whose summary field is absent. "
            "Repeat for multiple methods; never overrides a populated summary field."
        ),
    )
    parser.add_argument("--open-mos-csv", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    rows = _read_latest_rows(Path(value) for value in args.summary_csv)
    for method, score in _read_mos_scores(Path(value) for value in args.open_mos_csv).items():
        rows.setdefault(method, {})["open_mos"] = score
    try:
        reference_ratio_overrides = _parse_reference_ratio_overrides(
            args.reference_ratio_override
        )
        result = select_candidate(
            rows,
            baselines=args.baseline or DEFAULT_BASELINES,
            candidates=args.candidate or DEFAULT_CANDIDATES,
            expected_samples=args.expected_samples,
            metric=args.metric,
            min_margin=args.min_margin,
            secondary_metric=args.secondary_metric or None,
            secondary_min_margin=args.secondary_min_margin,
            max_recompute_ratio=args.max_recompute_ratio,
            max_reference_recompute_ratio=args.max_reference_recompute_ratio,
            reference_ratio_overrides=reference_ratio_overrides,
        )
    except ValueError as exc:
        result = {
            "status": "failed",
            "reason": str(exc),
            "metric": args.metric,
            "selected_method": None,
            "max_recompute_ratio": args.max_recompute_ratio,
            "max_reference_recompute_ratio": args.max_reference_recompute_ratio,
            "reference_ratio_overrides": args.reference_ratio_override,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[GRT_QUALITY_GATE] status=failed reason={exc}")
        raise SystemExit(1)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "[GRT_QUALITY_GATE] status=passed "
        f"selected_method={result['selected_method']} "
        f"selected_score={result['selected_score']} "
        f"recompute_ratio={result['selected_recompute_ratio']} "
        f"max_recompute_ratio={result['max_recompute_ratio']} "
        f"reference_patch_compute_ratio={result['selected_reference_patch_compute_ratio']} "
        f"reference_ratio_source={result['selected_reference_patch_compute_ratio_source']} "
        f"max_reference_recompute_ratio={result['max_reference_recompute_ratio']}"
    )


if __name__ == "__main__":
    main()
