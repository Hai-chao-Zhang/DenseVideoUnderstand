#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


csv.field_size_limit(sys.maxsize)


TASK_LABELS = {
    "densevideo": "DIVE-Bench Educational Dense Video",
    "densevideo_highmotion": "DIVE-Bench High-Motion Dense Video",
    "dive_bench_educational_high_fps": "DIVE-Bench Educational High-FPS Videos",
    "dive_bench_high_motion_high_fps": "DIVE-Bench High-Motion High-FPS Videos (full split)",
    "dive_bench_high_motion_high_fps_preview1000": "DIVE-Bench High-Motion High-FPS Videos (1000-item preview)",
}

HELD_HIGHMOTION_TASKS = frozenset({
    "densevideo_highmotion",
    "dive_bench_high_motion_high_fps",
    "dive_bench_high_motion_high_fps_preview1000",
})

TASK_COLUMNS = {
    "densevideo": [
        "rank",
        "display_name",
        "method",
        "samples",
        "open_mos",
        "token_f1",
        "cer",
        "wer",
        "exact_match",
        "mean_recompute_ratio",
        "reference_patch_compute_ratio",
        "mean_effective_fps",
        "mean_throughput_fps",
        "source",
    ],
    "densevideo_highmotion": [
        "rank",
        "display_name",
        "method",
        "samples",
        "grid_acc",
        "grid_ade",
        "grid_fde",
        "grid_transition_acc",
        "token_f1",
        "mean_effective_fps",
        "source",
    ],
}

DISPLAY = {
    "rank": "rank",
    "display_name": "model",
    "method": "method_id",
    "samples": "samples",
    "open_mos": "open_mos ↑",
    "token_f1": "token_f1 ↑",
    "cer": "cer ↓",
    "wer": "wer ↓",
    "exact_match": "exact_match ↑",
    "mean_recompute_ratio": "patch_projection_recompute_ratio ↓",
    "reference_patch_compute_ratio": "reference_patch_compute_ratio ↓",
    "mean_effective_fps": "sampling_density_fps",
    "mean_throughput_fps": "throughput_fps ↑",
    "grid_acc": "grid_acc ↑",
    "grid_ade": "grid_ade ↓",
    "grid_fde": "grid_fde ↓",
    "grid_transition_acc": "transition_acc ↑",
    "source": "source",
}

TASK_COLUMNS["dive_bench_educational_high_fps"] = TASK_COLUMNS["densevideo"]
for _task in ("dive_bench_high_motion_high_fps", "dive_bench_high_motion_high_fps_preview1000"):
    TASK_COLUMNS[_task] = TASK_COLUMNS["densevideo_highmotion"]


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "":
        return ""
    numeric = to_float(text)
    if numeric is not None:
        return f"{numeric:.6g}"
    return text.replace("\n", " ").replace("|", "\\|")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def require_publishable_rows(rows: List[Dict[str, Any]]) -> None:
    held_tasks = sorted({row.get("task", "") for row in rows} & HELD_HIGHMOTION_TASKS)
    if held_tasks:
        raise ValueError(
            "High-Motion publication is withheld pending target/reference consistency "
            f"review; refusing leaderboard output for: {', '.join(held_tasks)}. "
            "See docs/HIGHMOTION_TARGET_HOLD.md. Raw evaluation and historical "
            "evidence reconstruction remain available."
        )


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    require_publishable_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def load_model_config(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in data.get("models", []):
        method = str(item.get("method", "")).strip()
        if method:
            out[method] = dict(item)
    return out


def method_from_mos_row(row: Dict[str, str]) -> str:
    for key in ("method", "model", "run_name"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def load_open_mos(paths: Iterable[str]) -> Dict[str, float]:
    paths = [path for path in paths if path]
    if not paths:
        return {}
    scores: Dict[str, List[float]] = defaultdict(list)
    for raw_path in paths:
        for row in read_csv(Path(raw_path)):
            method = method_from_mos_row(row)
            score = None
            for col in ("open_mos_score", "score"):
                score = to_float(row.get(col))
                if score is not None:
                    break
            if method and score is not None:
                scores[method].append(score)
    return {method: sum(vals) / len(vals) for method, vals in scores.items() if vals}


def load_open_mos_judges(paths: Iterable[str]) -> List[str]:
    judges = {
        str(row.get("mos_judge_model", "")).strip()
        for raw_path in paths
        if raw_path
        for row in read_csv(Path(raw_path))
        if str(row.get("mos_judge_model", "")).strip()
    }
    return sorted(judges)


def is_closed_source(row: Dict[str, Any]) -> bool:
    provider = str(row.get("provider") or row.get("Provider") or "").lower()
    method = str(row.get("method") or row.get("Model") or row.get("model") or "").lower()
    return provider in {"openai", "gemini"} or method.startswith(("gpt-", "gemini-"))


def merge_lmms_rows(
    summary_paths: Iterable[str],
    model_config: Dict[str, Dict[str, Any]],
    mos_by_method: Dict[str, float],
    excluded_method_tasks: Optional[set[tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    excluded_method_tasks = excluded_method_tasks or set()
    rows_by_method_task: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw_path in summary_paths:
        for row in read_csv(Path(raw_path)):
            task = row.get("task", "")
            if task not in TASK_LABELS:
                continue
            if str(row.get("run_status", "")).strip().lower() == "failed":
                continue
            if not str(row.get("result_json", "")).strip():
                continue
            method = row.get("method", "") or row.get("run_name", "")
            if (method, task) in excluded_method_tasks:
                continue
            cfg = model_config.get(method, {})
            out = dict(row)
            out["method"] = method
            out["display_name"] = cfg.get("display_name", method)
            out["source"] = "open"
            if method in mos_by_method:
                out["open_mos"] = f"{mos_by_method[method]:.6g}"
            elif row.get("open_mos"):
                out["open_mos"] = row["open_mos"]
            rows_by_method_task[(method, task)] = out
    return list(rows_by_method_task.values())


def merge_api_rows(
    api_summary_paths: Iterable[str],
    include_closed_source: bool,
    mos_by_method: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not include_closed_source:
        return rows
    mos_by_method = mos_by_method or {}
    for raw_path in api_summary_paths:
        for row in read_csv(Path(raw_path)):
            if str(row.get("Status", "")).strip().lower() == "failed":
                continue
            subtask = row.get("Subtask", "")
            task = "densevideo" if subtask == "lpm" else "densevideo_highmotion" if subtask == "highfps" else subtask
            if task not in TASK_LABELS:
                continue
            out = {
                "task": task,
                "method": row.get("Model", ""),
                "display_name": row.get("DisplayName", "") or row.get("Model", ""),
                "samples": row.get("#SuccessfulVideos", "") or row.get("#Videos", ""),
                "source": row.get("Provider", "api"),
            }
            if out["method"] in mos_by_method:
                out["open_mos"] = f"{mos_by_method[out['method']]:.6g}"
            metric_value = row.get("MetricValue", "")
            if task == "densevideo":
                out["cer"] = row.get("CER", "")
                out["wer"] = row.get("WER", "")
                out["token_f1"] = row.get("TokenF1", metric_value if row.get("Metric") == "token_f1" else "")
                out["exact_match"] = row.get("ExactMatch", metric_value if row.get("Metric") == "accuracy" else "")
            else:
                out["grid_acc"] = row.get("GridAcc", metric_value if row.get("Metric") == "grid_acc" else "")
                out["grid_ade"] = row.get("GridADE", "")
                out["grid_fde"] = row.get("GridFDE", "")
                out["grid_transition_acc"] = row.get("GridTransitionAcc", "")
                out["token_f1"] = row.get("TokenF1", "")
            rows.append(out)
    return rows


def rank_rows(rows: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    primary = "open_mos" if task in {"densevideo", "dive_bench_educational_high_fps"} else "grid_acc"
    secondary = "token_f1"

    def sort_key(row: Dict[str, Any]):
        p = to_float(row.get(primary))
        s = to_float(row.get(secondary))
        if p is None:
            p = -1.0
        if s is None:
            s = -1.0
        return (-p, -s, str(row.get("display_name", row.get("method", ""))))

    ranked = sorted(rows, key=sort_key)
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = ["| " + " | ".join(DISPLAY.get(col, col) for col in columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    source_paths: List[str],
    include_closed_source: bool,
    open_mos_judges: Optional[List[str]] = None,
) -> None:
    require_publishable_rows(rows)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task", "")].append(row)

    lines = [
        "# DIVE-Bench Leaderboard",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- include_closed_source: `{str(include_closed_source).lower()}`",
        f"- source_artifacts: `{len(source_paths)}`",
    ]
    if open_mos_judges:
        lines.append(f"- open_mos_judge: `{', '.join(open_mos_judges)}`")
    lines.append("")

    if not rows:
        lines.append("No leaderboard rows found.")
    for task in TASK_LABELS:
        if task not in by_task:
            continue
        task_rows = rank_rows(by_task.get(task, []), task)
        lines.append(f"## {TASK_LABELS[task]}")
        lines.append("")
        if task_rows:
            lines.append(markdown_table(task_rows, TASK_COLUMNS[task]))
        else:
            lines.append("No rows.")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DIVE-Bench leaderboard markdown file.")
    parser.add_argument("--summary-csv", action="append", default=[], help="lmms-eval/collect_run_metrics summary.csv. Can be passed multiple times.")
    parser.add_argument("--open-mos-csv", action="append", default=[], help="Open MOS scored CSV from tools/densevideo/score_open_mos.py. Can be passed multiple times.")
    parser.add_argument("--api-summary-csv", action="append", default=[], help="Optional closed-source API summary CSV.")
    parser.add_argument(
        "--exclude-method-task",
        action="append",
        default=[],
        metavar="METHOD:TASK",
        help="Exclude one lmms-eval row, for example an invalid all-empty artifact.",
    )
    parser.add_argument("--model-config", default=None, help="Optional method display-name mapping YAML")
    parser.add_argument("--out-md", default="leaderboard.md")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--include-closed-source", nargs="?", const=True, default=False, type=parse_bool)
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    mos_by_method = load_open_mos(args.open_mos_csv)
    open_mos_judges = load_open_mos_judges(args.open_mos_csv)
    excluded_method_tasks = set()
    for value in args.exclude_method_task:
        if ":" not in value:
            parser.error(f"--exclude-method-task must be METHOD:TASK, got: {value}")
        excluded_method_tasks.add(tuple(value.rsplit(":", 1)))
    rows = merge_lmms_rows(args.summary_csv, model_config, mos_by_method, excluded_method_tasks)
    rows.extend(merge_api_rows(args.api_summary_csv, args.include_closed_source, mos_by_method))
    try:
        require_publishable_rows(rows)
    except ValueError as error:
        parser.error(str(error))
    if not args.include_closed_source:
        rows = [row for row in rows if not is_closed_source(row)]

    sources = list(args.summary_csv)
    sources.extend(args.open_mos_csv)
    sources.extend(args.api_summary_csv)
    out_md = Path(args.out_md)
    write_markdown(out_md, rows, sources, args.include_closed_source, open_mos_judges)

    if args.out_csv:
        all_columns = sorted({key for row in rows for key in row.keys()})
        write_csv(Path(args.out_csv), rows, all_columns)
    print(f"Wrote leaderboard: {out_md}")


if __name__ == "__main__":
    main()
