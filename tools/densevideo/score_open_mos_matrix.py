#!/usr/bin/env python3
"""Score several DIVE methods with one shared Open-MOS judge instance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.densevideo.run_open_mos_backfill import find_sample_jsonl  # noqa: E402
from tools.densevideo.score_open_mos import (  # noqa: E402
    JudgeRequest,
    TransformersJudge,
    DryRunJudge,
    build_messages,
    ground_truth_from_record,
    iter_jsonl,
    judge_fingerprint,
    make_result,
    prediction_from_record,
    question_from_record,
    read_done,
    request_fingerprint,
    reusable_result,
    sample_id_for_record,
    score_batch_resilient,
    write_summary,
)


DEFAULT_JUDGE = "Qwen/Qwen3-VL-32B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        type=Path,
        help="Root containing runs/<method>; repeat to search archived and fresh outputs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--judge-revision",
        default="",
        help="Optional immutable Hugging Face judge revision; passed to the loader and fingerprint.",
    )
    parser.add_argument("--backend", choices=("transformers", "dry-run"), default="transformers")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trim-char-limit", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.judge_revision = str(args.judge_revision or "").strip()

    if args.expected_samples <= 0 or args.batch_size <= 0:
        parser.error("--expected-samples and --batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "matrix.jsonl"
    out_csv = args.output_dir / "matrix.csv"
    out_md = args.output_dir / "matrix.md"
    judge_hash = judge_fingerprint(
        model=args.judge_model,
        revision=args.judge_revision,
        backend=args.backend,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        trim_char_limit=args.trim_char_limit,
        trust_remote_code=True,
        extra={"batch_size": args.batch_size},
    )
    done = read_done(out_jsonl) if args.resume else {}

    # Build current inputs before considering resume rows. A stable sample id
    # alone is not sufficient: predictions and judge settings can change while
    # the method/doc id remains the same.
    entries = []
    for method in args.method:
        sample_path = find_sample_jsonl(method, args.input_root, args.expected_samples)
        records = list(iter_jsonl(sample_path))
        if len(records) != args.expected_samples:
            raise RuntimeError(f"{method} has {len(records)} samples; expected {args.expected_samples}")
        for index, record in enumerate(records):
            sample_id = f"{method}::{sample_id_for_record(record, index)}"
            answer = ground_truth_from_record(record)
            pred = prediction_from_record(record)
            request = JudgeRequest(
                messages=build_messages(
                    question_from_record(record),
                    answer,
                    pred,
                    args.trim_char_limit,
                ),
                answer=answer,
                pred=pred,
            )
            request_hash = request_fingerprint(request)
            entries.append((method, record, sample_id, request, request_hash))

    # Failed, malformed, legacy, or stale-fingerprint rows are intentionally
    # excluded so --resume retries them. Exact inputs are canonicalized across
    # method labels under the same judge fingerprint.
    completed = {}
    cached_reviews = {}
    for method, record, sample_id, request, request_hash in entries:
        row = done.get(sample_id)
        if not reusable_result(row, request_hash, judge_hash):
            continue
        cache_key = (judge_hash, request_hash)
        if cache_key in cached_reviews:
            review, judge_model = cached_reviews[cache_key]
        else:
            review = str(row.get("open_mos_review", ""))
            judge_model = str(row.get("mos_judge_model", args.judge_model))
            cached_reviews[cache_key] = (review, judge_model)
        completed[sample_id] = make_result(
            record,
            sample_id,
            review,
            judge_model,
            "",
            method,
            f"{method}_densevideo_open_mos",
            request_hash,
            judge_hash,
            args.judge_revision,
        )

    cached_pending = []
    pending_by_key = {}
    for method, record, sample_id, request, request_hash in entries:
        if sample_id in completed:
            continue
        cache_key = (judge_hash, request_hash)
        if cache_key in cached_reviews:
            review, judge_model = cached_reviews[cache_key]
            cached_pending.append((method, record, sample_id, review, judge_model, request_hash))
            continue
        group = pending_by_key.setdefault(
            cache_key,
            {"request": request, "request_hash": request_hash, "items": []},
        )
        group["items"].append((method, record, sample_id))

    unique_pending = list(pending_by_key.values())
    pending_rows = len(cached_pending) + sum(len(group["items"]) for group in unique_pending)
    judge = None
    if unique_pending:
        if args.backend == "dry-run":
            judge = DryRunJudge(args.judge_model)
        else:
            judge = TransformersJudge(
                args.judge_model,
                args.dtype,
                True,
                args.max_new_tokens,
                args.judge_revision,
            )
    written = 0
    with out_jsonl.open("w", encoding="utf-8") as output:
        for row in completed.values():
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

        for method, record, sample_id, review, judge_model, request_hash in cached_pending:
            result = make_result(
                record,
                sample_id,
                review,
                judge_model,
                "",
                method,
                f"{method}_densevideo_open_mos",
                request_hash,
                judge_hash,
                args.judge_revision,
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

        for start in range(0, len(unique_pending), args.batch_size):
            batch = unique_pending[start : start + args.batch_size]
            outcomes = score_batch_resilient(judge, [group["request"] for group in batch])
            for group, (response, error) in zip(batch, outcomes):
                cacheable = response is not None
                for method, record, sample_id in group["items"]:
                    if response is None:
                        result = make_result(
                            record,
                            sample_id,
                            "",
                            args.judge_model,
                            error,
                            method,
                            f"{method}_densevideo_open_mos",
                            group["request_hash"],
                            judge_hash,
                            args.judge_revision,
                        )
                    else:
                        result = make_result(
                            record,
                            sample_id,
                            response.text,
                            response.model,
                            "",
                            method,
                            f"{method}_densevideo_open_mos",
                            group["request_hash"],
                            judge_hash,
                            args.judge_revision,
                        )
                    cacheable = cacheable and not bool(result.get("error"))
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    written += 1
                if cacheable:
                    cached_reviews[(judge_hash, group["request_hash"])] = (response.text, response.model)
                output.flush()
            print(
                f"[OPEN_MOS_MATRIX] rows={written}/{pending_rows} "
                f"unique_judged={min(start + len(batch), len(unique_pending))}/{len(unique_pending)} "
                f"total_completed={len(completed) + written}",
                flush=True,
            )

    rows = list(iter_jsonl(out_jsonl))
    write_summary(rows, out_csv, out_md, out_jsonl)
    for method in args.method:
        method_rows = [row for row in rows if row.get("method") == method]
        errors = [row for row in method_rows if row.get("error")]
        if len(method_rows) != args.expected_samples or errors:
            raise RuntimeError(
                f"Open-MOS validation failed for {method}: rows={len(method_rows)} "
                f"expected={args.expected_samples} errors={len(errors)}"
            )
    print(f"[OPEN_MOS_MATRIX] status=complete csv={out_csv}")


if __name__ == "__main__":
    main()
