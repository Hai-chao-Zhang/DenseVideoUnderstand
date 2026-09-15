#!/usr/bin/env python3
"""Strictly compare two LMMS sample JSONL files after aligning by document ID."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class SampleFormatError(ValueError):
    """Raised when a sample JSONL cannot support a strict comparison."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_at_path(row: Mapping[str, Any], field_path: str, *, source: Path, line_number: int) -> Any:
    value: Any = row
    for component in field_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise SampleFormatError(f"{source}:{line_number}: missing required field {field_path!r}")
        value = value[component]
    return value


def _doc_key(doc_id: Any, *, source: Path, line_number: int) -> str:
    if doc_id is None or isinstance(doc_id, (dict, list)):
        raise SampleFormatError(
            f"{source}:{line_number}: doc_id must be a non-null JSON scalar, got {type(doc_id).__name__}"
        )
    # Canonical JSON preserves the distinction between, for example, integer
    # 1 and string "1", which a strict deterministic comparison must retain.
    return _canonical_json(doc_id)


def load_samples(
    path: Path,
    *,
    doc_id_key: str = "doc_id",
    fields: Sequence[Tuple[str, str]] = (),
) -> Dict[str, Tuple[Any, int, Dict[str, Any]]]:
    # Keep only comparison fields. Real LMMS rows may embed very large OCR and
    # caption payloads under ``doc``; retaining both complete files can consume
    # orders of magnitude more memory than the strict comparison requires.
    samples: Dict[str, Tuple[Any, int, Dict[str, Any]]] = {}
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise SampleFormatError(f"Could not open {path}: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SampleFormatError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, Mapping):
                raise SampleFormatError(f"{path}:{line_number}: expected a JSON object")
            doc_id = _value_at_path(row, doc_id_key, source=path, line_number=line_number)
            key = _doc_key(doc_id, source=path, line_number=line_number)
            if key in samples:
                previous_line = samples[key][1]
                raise SampleFormatError(
                    f"{path}:{line_number}: duplicate doc_id={doc_id!r}; first seen on line {previous_line}"
                )
            values = {
                label: _value_at_path(row, field_path, source=path, line_number=line_number)
                for label, field_path in fields
            }
            samples[key] = (doc_id, line_number, values)

    if not samples:
        raise SampleFormatError(f"{path}: no sample rows found")
    return samples


def _preview(value: Any, limit: int = 240) -> str:
    rendered = _canonical_json(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def compare_sample_files(
    left_path: Path,
    right_path: Path,
    *,
    doc_id_key: str = "doc_id",
    fields: Sequence[Tuple[str, str]] = (
        ("question", "input"),
        ("reference", "target"),
        ("prediction", "filtered_resps"),
    ),
    max_details: int = 20,
) -> Dict[str, Any]:
    """Return a machine-readable strict comparison report.

    Rows may appear in different orders. Document IDs, row counts, required
    fields, JSON types, and values are otherwise compared exactly.
    """

    left_path = Path(left_path)
    right_path = Path(right_path)
    left = load_samples(left_path, doc_id_key=doc_id_key, fields=fields)
    right = load_samples(right_path, doc_id_key=doc_id_key, fields=fields)

    left_keys = set(left)
    right_keys = set(right)
    only_left_keys = sorted(left_keys - right_keys)
    only_right_keys = sorted(right_keys - left_keys)
    common_keys = sorted(left_keys & right_keys)
    mismatch_counts = {label: 0 for label, _ in fields}
    mismatch_details: List[Dict[str, Any]] = []

    for key in common_keys:
        doc_id, left_line, left_values = left[key]
        _, right_line, right_values = right[key]
        for label, field_path in fields:
            left_value = left_values[label]
            right_value = right_values[label]
            if _canonical_json(left_value) == _canonical_json(right_value):
                continue
            mismatch_counts[label] += 1
            if len(mismatch_details) < max(0, int(max_details)):
                mismatch_details.append(
                    {
                        "doc_id": doc_id,
                        "field": label,
                        "json_path": field_path,
                        "left_line": left_line,
                        "right_line": right_line,
                        "left": _preview(left_value),
                        "right": _preview(right_value),
                    }
                )

    only_left = [left[key][0] for key in only_left_keys]
    only_right = [right[key][0] for key in only_right_keys]
    identical = not only_left and not only_right and not any(mismatch_counts.values())
    return {
        "schema_version": 1,
        "status": "identical" if identical else "mismatch",
        "identical": identical,
        "left": str(left_path),
        "right": str(right_path),
        "left_rows": len(left),
        "right_rows": len(right),
        "aligned_rows": len(common_keys),
        "doc_id_key": doc_id_key,
        "fields": {label: field_path for label, field_path in fields},
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left_doc_ids": only_left[: max(0, int(max_details))],
        "only_right_doc_ids": only_right[: max(0, int(max_details))],
        "mismatch_counts": mismatch_counts,
        "mismatch_details": mismatch_details,
    }


def _field_specs(args: argparse.Namespace) -> Iterable[Tuple[str, str]]:
    return (
        ("question", args.question_key),
        ("reference", args.reference_key),
        ("prediction", args.prediction_key),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align two LMMS sample JSONLs by doc_id and require exact question, "
            "reference, and prediction equality. Exit 0 means identical; 1 means "
            "a value/doc_id mismatch; 2 means malformed input."
        )
    )
    parser.add_argument("left", type=Path, help="First LMMS *_samples_*.jsonl file.")
    parser.add_argument("right", type=Path, help="Second LMMS *_samples_*.jsonl file.")
    parser.add_argument("--doc-id-key", default="doc_id", help="Required doc ID field (dotted paths supported).")
    parser.add_argument("--question-key", default="input", help="Required question field.")
    parser.add_argument("--reference-key", default="target", help="Required reference field.")
    parser.add_argument("--prediction-key", default="filtered_resps", help="Required prediction field.")
    parser.add_argument("--max-details", type=int, default=20, help="Maximum mismatch details per report.")
    parser.add_argument("--report-json", type=Path, help="Optionally write the full comparison report as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = compare_sample_files(
            args.left,
            args.right,
            doc_id_key=args.doc_id_key,
            fields=tuple(_field_specs(args)),
            max_details=args.max_details,
        )
    except SampleFormatError as exc:
        print(f"[LMMS_SAMPLE_COMPARE] status=invalid error={exc}", file=sys.stderr)
        return 2

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = report["mismatch_counts"]
    print(
        "[LMMS_SAMPLE_COMPARE] "
        f"status={report['status']} aligned_rows={report['aligned_rows']} "
        f"left_rows={report['left_rows']} right_rows={report['right_rows']} "
        f"only_left={report['only_left_count']} only_right={report['only_right_count']} "
        f"question_mismatches={counts['question']} "
        f"reference_mismatches={counts['reference']} "
        f"prediction_mismatches={counts['prediction']}"
    )
    for detail in report["mismatch_details"]:
        print(
            f"  doc_id={detail['doc_id']!r} field={detail['field']} "
            f"left_line={detail['left_line']} right_line={detail['right_line']}\n"
            f"    left={detail['left']}\n"
            f"    right={detail['right']}"
        )
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
