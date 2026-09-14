#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TASK_INPUT_FPS = {
    "densevideo": "1.0",
    "densevideo_highmotion": "4.0",
}

PREDICTION_FIELDS = (
    "filtered_resps",
    "response",
    "responses",
    "prediction",
    "predictions",
    "model_response",
    "model_output",
    "output",
    "outputs",
    "generated_text",
    "completion",
    "resps",
)

NESTED_PREDICTION_FIELDS = (
    "text",
    "content",
    "answer",
    "response",
    "prediction",
    "output",
    "generated_text",
    "completion",
    "message",
)

GENERATION_ERROR_PATTERNS = (
    re.compile(r"\berror\b[^\r\n]*\bgenerat(?:e|ing|ion)\b", re.IGNORECASE),
    re.compile(r"\bgenerat(?:e|ing|ion)\b[^\r\n]*\b(?:error|failed|failure)\b", re.IGNORECASE),
    re.compile(r"\bempty generation\b", re.IGNORECASE),
    re.compile(r"\berror\b[^\r\n]*\b(?:loading|decoding) video\b", re.IGNORECASE),
)

ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True)
class RunOutputValidation:
    usable: bool
    reason: str
    result_path: Optional[Path] = None
    sample_paths: Tuple[Path, ...] = ()
    sample_count: int = 0
    nonempty_prediction_count: int = 0
    expected_sample_count: Optional[int] = None


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def selected_models(config: Dict[str, Any], model_ids: Iterable[str]) -> List[Dict[str, Any]]:
    requested = {item for item in model_ids if item}
    models = list(config.get("models", []))
    if not requested:
        return models
    out = [model for model in models if str(model.get("method", "")) in requested]
    missing = requested.difference({str(model.get("method", "")) for model in out})
    if missing:
        raise ValueError(f"Unknown model-id(s): {', '.join(sorted(missing))}")
    return out


def build_lmms_command(
    *,
    model_name: str,
    model_args: str,
    task: str,
    output_dir: Path,
    launcher: str,
    num_processes: str,
    batch_size: str,
    limit: Optional[str],
    gen_kwargs: Optional[str],
    extra_args: List[str],
) -> List[str]:
    if launcher == "accelerate":
        cmd = ["accelerate", "launch", "--num_processes", str(num_processes), "-m", "lmms_eval"]
    else:
        cmd = [sys.executable, "-m", "lmms_eval"]
    cmd.extend(
        [
            "--model",
            model_name,
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--batch_size",
            batch_size,
            "--log_samples",
            "--output_path",
            str(output_dir),
        ]
    )
    if limit:
        cmd.extend(["--limit", str(limit)])
    if gen_kwargs:
        cmd.extend(["--gen_kwargs", gen_kwargs])
    cmd.extend(extra_args)
    return cmd


def run_with_log(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"[RUN_META] command={shlex.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        return_code = proc.wait()
        wall = time.perf_counter() - start
        status = "success" if return_code == 0 else "failed"
        log_f.write(f"[RUN_META] run_wall_time_s={wall:.6f} run_status={status} return_code={return_code}\n")
        log_f.flush()
    return return_code


def has_result_json(run_output_dir: Path) -> bool:
    return any(run_output_dir.rglob("*_results.json"))


def _artifact_signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _snapshot_result_artifacts(run_output_dir: Path) -> Dict[Path, Tuple[int, int]]:
    snapshot: Dict[Path, Tuple[int, int]] = {}
    for path in run_output_dir.rglob("*_results.json"):
        try:
            snapshot[path] = _artifact_signature(path)
        except OSError:
            continue
    return snapshot


def _result_candidates(
    run_output_dir: Path,
    *,
    artifact_baseline: Optional[Mapping[Path, Tuple[int, int]]] = None,
) -> List[Path]:
    candidates = []
    for path in run_output_dir.rglob("*_results.json"):
        try:
            signature = _artifact_signature(path)
        except OSError:
            continue
        if artifact_baseline is not None and artifact_baseline.get(path) == signature:
            continue
        candidates.append(path)
    return sorted(candidates, key=lambda path: (_artifact_signature(path)[0], str(path)), reverse=True)


def _sample_paths_for_result(result_path: Path, task: Optional[str]) -> Tuple[Path, ...]:
    suffix = "_results.json"
    if not result_path.name.endswith(suffix):
        return ()
    prefix = result_path.name[: -len(suffix)]
    candidates = tuple(sorted(result_path.parent.glob(f"{prefix}_samples_*.jsonl")))
    if task:
        exact_name = f"{prefix}_samples_{task}.jsonl"
        exact = tuple(path for path in candidates if path.name == exact_name)
        if exact:
            return exact
    return candidates


def _coerce_sample_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _declared_sample_counts(payload: Mapping[str, Any], task: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    counts = payload.get("n-samples", payload.get("n_samples"))
    if counts is None:
        return None, None

    selected: List[Any]
    if isinstance(counts, Mapping) and ("original" in counts or "effective" in counts):
        selected = [counts]
    elif isinstance(counts, Mapping) and task and task in counts:
        selected = [counts[task]]
    elif isinstance(counts, Mapping):
        selected = list(counts.values())
    else:
        selected = [counts]

    originals: List[int] = []
    effective: List[int] = []
    for item in selected:
        if isinstance(item, Mapping):
            original_count = _coerce_sample_count(item.get("original"))
            effective_count = _coerce_sample_count(item.get("effective"))
        else:
            original_count = _coerce_sample_count(item)
            effective_count = original_count
        if original_count is not None:
            originals.append(original_count)
        if effective_count is not None:
            effective.append(effective_count)
    return (sum(originals) if originals else None, sum(effective) if effective else None)


def _expected_sample_count(payload: Mapping[str, Any], task: Optional[str], limit: Optional[str]) -> Optional[int]:
    original_count, effective_count = _declared_sample_counts(payload, task)
    expected_counts = []
    if effective_count is not None:
        expected_counts.append(effective_count)
    elif original_count is not None and not limit:
        expected_counts.append(original_count)

    if limit:
        try:
            parsed_limit = float(limit)
        except (TypeError, ValueError):
            parsed_limit = math.nan
        if math.isfinite(parsed_limit) and parsed_limit > 0:
            if parsed_limit < 1:
                if original_count is not None:
                    expected_counts.append(int(math.ceil(original_count * parsed_limit)))
            else:
                limited_count = int(parsed_limit)
                if original_count is not None:
                    limited_count = min(limited_count, original_count)
                expected_counts.append(limited_count)

    return max(expected_counts) if expected_counts else None


def _has_nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_nonempty_text(item) for item in value)
    if isinstance(value, Mapping):
        for key in NESTED_PREDICTION_FIELDS:
            if key in value and _has_nonempty_text(value[key]):
                return True
        if len(value) == 1:
            return _has_nonempty_text(next(iter(value.values())))
    return False


def _sample_prediction_value(sample: Mapping[str, Any]) -> Tuple[bool, Any]:
    scored_field = PREDICTION_FIELDS[0]
    if scored_field in sample:
        return True, sample[scored_field]

    found_prediction_field = False
    first_value: Any = None
    for field in PREDICTION_FIELDS[1:]:
        if field in sample:
            found_prediction_field = True
            if first_value is None:
                first_value = sample[field]
            if _has_nonempty_text(sample[field]):
                return True, sample[field]
    return found_prediction_field, first_value


def _sample_has_prediction(sample: Mapping[str, Any]) -> Tuple[bool, bool]:
    found_prediction_field, prediction = _sample_prediction_value(sample)
    return found_prediction_field, _has_nonempty_text(prediction)


def _highmotion_grid_label_count(prediction: Any) -> int:
    # Keep the smoke gate on the exact parser used by the task metrics. This
    # accepts harmless semantic/rNcN formatting variants while rejecting a
    # nonempty but truncated or unparseable trajectory.
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from lmms_eval.tasks.densevideo.utils import parse_grid_sequence

    return len(parse_grid_sequence(prediction))


def find_generation_error(run_log: Optional[Path]) -> Optional[str]:
    if run_log is None or not run_log.exists():
        return None
    try:
        with run_log.open("r", encoding="utf-8", errors="ignore") as log_f:
            for line in log_f:
                clean_line = ANSI_ESCAPE_RE.sub("", line)
                if any(pattern.search(clean_line) for pattern in GENERATION_ERROR_PATTERNS):
                    return " ".join(clean_line.split())[:500]
    except OSError as exc:
        return f"could not read run log: {exc}"
    return None


def _validate_result_candidate(result_path: Path, *, task: Optional[str], limit: Optional[str]) -> RunOutputValidation:
    try:
        with result_path.open("r", encoding="utf-8") as result_f:
            payload = json.load(result_f)
    except (OSError, json.JSONDecodeError) as exc:
        return RunOutputValidation(False, f"invalid result JSON {result_path}: {exc}", result_path=result_path)
    if not isinstance(payload, Mapping):
        return RunOutputValidation(False, f"result JSON is not an object: {result_path}", result_path=result_path)

    expected_count = _expected_sample_count(payload, task, limit)
    sample_paths = _sample_paths_for_result(result_path, task)
    if not sample_paths:
        return RunOutputValidation(
            False,
            f"missing sample JSONL paired with {result_path.name}",
            result_path=result_path,
            expected_sample_count=expected_count,
        )

    sample_count = 0
    nonempty_count = 0
    missing_prediction_count = 0
    empty_prediction_count = 0
    invalid_highmotion_predictions: List[str] = []
    # Exact trajectory structure is a smoke/debug contract. Full evaluation
    # must preserve malformed model outputs so the task metrics can penalize
    # them instead of discarding an otherwise complete 1,000-sample run.
    if task == "densevideo_highmotion" and limit is not None:
        raw_frame_count = os.getenv("DENSEVIDEO_HIGHMOTION_NUM_FRAMES", "8").strip()
        try:
            highmotion_frame_count = int(raw_frame_count)
        except ValueError:
            highmotion_frame_count = 0
        if highmotion_frame_count <= 0:
            return RunOutputValidation(
                False,
                f"DENSEVIDEO_HIGHMOTION_NUM_FRAMES must be a positive integer, got {raw_frame_count!r}",
                result_path=result_path,
                sample_paths=sample_paths,
                expected_sample_count=expected_count,
            )
    else:
        highmotion_frame_count = None
    for sample_path in sample_paths:
        try:
            with sample_path.open("r", encoding="utf-8") as sample_f:
                for line_number, line in enumerate(sample_f, start=1):
                    if not line.strip():
                        continue
                    sample_count += 1
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as exc:
                        return RunOutputValidation(
                            False,
                            f"invalid sample JSON at {sample_path}:{line_number}: {exc}",
                            result_path=result_path,
                            sample_paths=sample_paths,
                            sample_count=sample_count,
                            nonempty_prediction_count=nonempty_count,
                            expected_sample_count=expected_count,
                        )
                    if not isinstance(sample, Mapping):
                        return RunOutputValidation(
                            False,
                            f"sample is not an object at {sample_path}:{line_number}",
                            result_path=result_path,
                            sample_paths=sample_paths,
                            sample_count=sample_count,
                            nonempty_prediction_count=nonempty_count,
                            expected_sample_count=expected_count,
                        )
                    has_prediction_field, prediction = _sample_prediction_value(sample)
                    has_nonempty_prediction = _has_nonempty_text(prediction)
                    if not has_prediction_field:
                        missing_prediction_count += 1
                    elif has_nonempty_prediction:
                        nonempty_count += 1
                        if highmotion_frame_count is not None:
                            parsed_count = _highmotion_grid_label_count(prediction)
                            if parsed_count == 0 or parsed_count > highmotion_frame_count:
                                invalid_highmotion_predictions.append(
                                    f"{sample_path.name}:{line_number} parsed={parsed_count}"
                                )
                    else:
                        empty_prediction_count += 1
        except OSError as exc:
            return RunOutputValidation(
                False,
                f"could not read sample JSONL {sample_path}: {exc}",
                result_path=result_path,
                sample_paths=sample_paths,
                sample_count=sample_count,
                nonempty_prediction_count=nonempty_count,
                expected_sample_count=expected_count,
            )

    details = {
        "result_path": result_path,
        "sample_paths": sample_paths,
        "sample_count": sample_count,
        "nonempty_prediction_count": nonempty_count,
        "expected_sample_count": expected_count,
    }
    if expected_count is not None and expected_count <= 0:
        return RunOutputValidation(False, f"result declares a non-positive sample count: {expected_count}", **details)
    if sample_count == 0:
        return RunOutputValidation(False, "sample JSONL contains no samples", **details)
    if expected_count is not None and sample_count < expected_count:
        return RunOutputValidation(False, f"sample count {sample_count} is below expected {expected_count}", **details)
    if missing_prediction_count:
        return RunOutputValidation(False, f"{missing_prediction_count} samples have no recognized prediction field", **details)
    if nonempty_count == 0:
        return RunOutputValidation(False, f"all {empty_prediction_count} samples have empty predictions", **details)
    if empty_prediction_count:
        return RunOutputValidation(
            False,
            f"{empty_prediction_count} of {sample_count} samples have empty predictions",
            **details,
        )
    if invalid_highmotion_predictions:
        examples = ", ".join(invalid_highmotion_predictions[:3])
        return RunOutputValidation(
            False,
            f"{len(invalid_highmotion_predictions)} of {sample_count} densevideo_highmotion predictions do not "
            f"contain between one and {highmotion_frame_count} valid grid labels ({examples})",
            **details,
        )
    return RunOutputValidation(True, "usable result", **details)


def validate_run_output(
    run_output_dir: Path,
    *,
    task: Optional[str] = None,
    limit: Optional[str] = None,
    run_log: Optional[Path] = None,
    artifact_baseline: Optional[Mapping[Path, Tuple[int, int]]] = None,
) -> RunOutputValidation:
    generation_error = find_generation_error(run_log)
    if generation_error is not None:
        return RunOutputValidation(False, f"generation error in run log: {generation_error}")

    candidates = _result_candidates(run_output_dir, artifact_baseline=artifact_baseline)
    if not candidates:
        qualifier = "new " if artifact_baseline is not None else ""
        return RunOutputValidation(False, f"missing {qualifier}result JSON under {run_output_dir}")
    return _validate_result_candidate(candidates[0], task=task, limit=limit)


def has_usable_result(
    run_output_dir: Path,
    *,
    task: Optional[str] = None,
    limit: Optional[str] = None,
    run_log: Optional[Path] = None,
) -> bool:
    return validate_run_output(run_output_dir, task=task, limit=limit, run_log=run_log).usable


def append_run_meta(log_path: Path, **items: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        fields = " ".join(f"{key}={value}" for key, value in items.items())
        log_f.write(f"[RUN_META] {fields}\n")


def collect_metrics(
    *,
    run_name: str,
    method: str,
    task: str,
    run_log: Path,
    run_output_dir: Path,
    summary_csv: Path,
) -> int:
    cmd = [
        sys.executable,
        "tools/densevideo/collect_run_metrics.py",
        "--run-name",
        run_name,
        "--method",
        method,
        "--input-fps",
        TASK_INPUT_FPS.get(task, ""),
        "--task-name",
        task,
        "--run-log",
        str(run_log),
        "--run-output-dir",
        str(run_output_dir),
        "--summary-csv",
        str(summary_csv),
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the open-model DIVE-Bench leaderboard matrix.")
    parser.add_argument("--model-config", required=True, help="Explicit model profile YAML; use dive-reproduce for the pinned GRT profiles")
    parser.add_argument("--model-id", action="append", default=[], help="Run only this method id. Can be repeated.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Override tasks from the model config.")
    parser.add_argument("--output-root", default="outputs/densevideo_leaderboard")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="accelerate")
    parser.add_argument("--num-processes", default=os.getenv("EVAL_NUM_PROCESSES", "1"))
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", default=os.getenv("EVAL_LIMIT", ""))
    parser.add_argument("--gen-kwargs", default="max_new_tokens=256,temperature=0")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra lmms-eval CLI argument tokens, shell-split before use.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.model_config)
    config = load_config(config_path)
    models = selected_models(config, args.model_id)
    tasks = args.tasks if args.tasks is not None else list(config.get("tasks", ["densevideo", "densevideo_highmotion"]))
    output_root = Path(args.output_root)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary.csv"
    extra_args: List[str] = []
    for item in args.extra_arg:
        extra_args.extend(shlex.split(item))

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failures = []

    for model in models:
        method = str(model["method"])
        model_name = str(model["model"])
        model_args = str(model.get("model_args", ""))
        for task in tasks:
            run_name = f"{method}_{task}_{timestamp}"
            run_output_dir = output_root / "runs" / method / task
            run_log = output_root / "logs" / f"{method}_{task}.log"
            if args.skip_existing:
                existing_validation = validate_run_output(
                    run_output_dir,
                    task=task,
                    limit=args.limit or None,
                    run_log=run_log,
                )
                if existing_validation.usable:
                    print(
                        f"[OPEN_LEADERBOARD] Skipping existing usable run: {method} {task} "
                        f"({existing_validation.sample_count} samples)"
                    )
                    continue
                if has_result_json(run_output_dir):
                    print(
                        f"[OPEN_LEADERBOARD] Existing result is unusable; rerunning {method} {task}: "
                        f"{existing_validation.reason}",
                        file=sys.stderr,
                    )
            cmd = build_lmms_command(
                model_name=model_name,
                model_args=model_args,
                task=task,
                output_dir=run_output_dir,
                launcher=args.launcher,
                num_processes=args.num_processes,
                batch_size=args.batch_size,
                limit=args.limit or None,
                gen_kwargs=args.gen_kwargs or None,
                extra_args=extra_args,
            )
            print(f"[OPEN_LEADERBOARD] {method} {task}: {shlex.join(cmd)}", flush=True)
            if args.dry_run:
                continue

            artifact_baseline = _snapshot_result_artifacts(run_output_dir)
            return_code = run_with_log(cmd, run_log, env)
            validation = validate_run_output(
                run_output_dir,
                task=task,
                limit=args.limit or None,
                run_log=run_log,
                artifact_baseline=artifact_baseline,
            )
            if not validation.usable:
                validation_reason = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", validation.reason).strip("_")[:500]
                append_run_meta(
                    run_log,
                    run_status="failed",
                    usable_result="false",
                    validation_reason=validation_reason,
                    return_code=1,
                )
                return_code = return_code or 1
                print(
                    f"[OPEN_LEADERBOARD] unusable result for method={method} task={task}: "
                    f"{validation.reason}",
                    file=sys.stderr,
                )
                collect_code = 0
            else:
                collect_code = collect_metrics(
                    run_name=run_name,
                    method=method,
                    task=task,
                    run_log=run_log,
                    run_output_dir=run_output_dir,
                    summary_csv=summary_csv,
                )
            if return_code != 0 or collect_code != 0:
                failures.append((method, task, return_code, collect_code))
                if args.stop_on_error:
                    break
        if failures and args.stop_on_error:
            break

    if failures:
        for method, task, return_code, collect_code in failures:
            print(
                f"[OPEN_LEADERBOARD] failed method={method} task={task} "
                f"eval_return_code={return_code} collect_return_code={collect_code}",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print(f"[OPEN_LEADERBOARD] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
