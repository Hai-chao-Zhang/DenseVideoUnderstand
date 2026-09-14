#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_JUDGE = "Qwen/Qwen3-VL-32B-Instruct"


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            count += 1
    return count


def find_sample_jsonl(method: str, roots: Iterable[Path], expected_samples: int) -> Path:
    candidates: list[Path] = []
    rejected: list[str] = []
    for root in roots:
        run_dir = root / "runs" / method / "densevideo"
        if not run_dir.exists():
            continue
        candidates.extend(run_dir.rglob("*_samples_densevideo.jsonl"))

    for path in sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            sample_count = count_jsonl_rows(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rejected.append(f"{path}: {exc}")
            continue
        if sample_count == expected_samples:
            return path
        rejected.append(f"{path}: {sample_count} rows (expected {expected_samples})")

    details = "; ".join(rejected) if rejected else "no sample JSONL candidates found"
    raise FileNotFoundError(f"no complete LPM samples for {method}: {details}")


def validate_mos_csv(
    path: Path,
    method: str,
    expected_samples: int,
    *,
    judge_model: str = "",
    judge_revision: str = "",
) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return False, str(exc)
    if len(rows) != expected_samples:
        return False, f"{len(rows)} rows (expected {expected_samples})"
    wrong_method = sum(1 for row in rows if str(row.get("method", "")).strip() != method)
    errors = sum(1 for row in rows if str(row.get("error", "")).strip())
    missing_scores = sum(1 for row in rows if not str(row.get("open_mos_score", "")).strip())
    wrong_judge_model = sum(
        1
        for row in rows
        if judge_model and str(row.get("mos_judge_model", "")).strip() != judge_model
    )
    wrong_judge_revision = sum(
        1
        for row in rows
        if judge_revision and str(row.get("mos_judge_revision", "")).strip() != judge_revision
    )
    if wrong_method:
        return False, f"{wrong_method} rows use a different method id"
    if errors:
        return False, f"{errors} scored rows contain errors"
    if missing_scores:
        return False, f"{missing_scores} rows have no open_mos_score"
    if wrong_judge_model:
        return False, f"{wrong_judge_model} rows use a different judge model"
    if wrong_judge_revision:
        return False, f"{wrong_judge_revision} rows use a different judge revision"
    return True, "complete"


def find_complete_existing_mos(
    method: str,
    directories: Iterable[Path],
    expected_samples: int,
    *,
    judge_model: str = "",
    judge_revision: str = "",
) -> Optional[Path]:
    candidates = [directory / f"{method}.csv" for directory in directories]
    existing = [path for path in candidates if path.exists()]
    for path in sorted(existing, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        valid, _ = validate_mos_csv(
            path,
            method,
            expected_samples,
            judge_model=judge_model,
            judge_revision=judge_revision,
        )
        if valid:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill one complete DIVE-Bench Open MOS column.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--existing-mos-dir", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=634)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--judge-revision",
        default="",
        help="Optional immutable Hugging Face judge revision forwarded to the scorer.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "vllm", "transformers", "openai-compatible", "dry-run"),
        default="transformers",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--trim-char-limit", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the scorer command.")
    args = parser.parse_args()
    args.judge_revision = str(args.judge_revision or "").strip()

    if args.expected_samples <= 0:
        parser.error("--expected-samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    search_dirs = [args.output_dir, *args.existing_mos_dir]
    if not args.force:
        existing = find_complete_existing_mos(
            args.method,
            search_dirs,
            args.expected_samples,
            judge_model=args.judge_model,
            judge_revision=args.judge_revision,
        )
        if existing is not None:
            print(f"[OPEN_MOS_BACKFILL] method={args.method} status=skip_complete csv={existing}")
            return

    sample_path = find_sample_jsonl(args.method, args.input_root, args.expected_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / f"{args.method}.jsonl"
    out_csv = args.output_dir / f"{args.method}.csv"
    out_md = args.output_dir / f"{args.method}.md"
    scorer = Path(__file__).with_name("score_open_mos.py")
    command = [
        sys.executable,
        str(scorer),
        "--input-jsonl",
        str(sample_path),
        "--out-jsonl",
        str(out_jsonl),
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
        "--model",
        args.judge_model,
        "--judge-revision",
        args.judge_revision,
        "--method",
        args.method,
        "--run-name",
        f"{args.method}_densevideo_open_mos",
        "--backend",
        args.backend,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--trim-char-limit",
        str(args.trim_char_limit),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--resume",
    ]
    print(
        f"[OPEN_MOS_BACKFILL] method={args.method} samples={args.expected_samples} "
        f"input={sample_path}"
    )
    print(f"[OPEN_MOS_BACKFILL] command={shlex.join(command)}")
    if args.dry_run:
        return

    subprocess.run(command, check=True)
    valid, reason = validate_mos_csv(
        out_csv,
        args.method,
        args.expected_samples,
        judge_model=args.judge_model,
        judge_revision=args.judge_revision,
    )
    if not valid:
        raise RuntimeError(f"Open MOS output validation failed for {args.method}: {reason}")
    print(f"[OPEN_MOS_BACKFILL] method={args.method} status=complete csv={out_csv}")


if __name__ == "__main__":
    main()
