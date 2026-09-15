"""Synthetic publication-policy checks; no model or dataset artifacts are read."""

import copy
import csv
from datetime import datetime

import pytest

from tools.densevideo import build_leaderboard as exporter


HM_TASKS = (
    "densevideo_highmotion",
    "dive_bench_high_motion_high_fps",
    "dive_bench_high_motion_high_fps_preview1000",
)


def educational_row():
    return {
        "task": "densevideo",
        "method": "synthetic_edu",
        "display_name": "Synthetic educational",
        "result_json": "synthetic-not-read.json",
        "open_mos": "2",
        "token_f1": "0.5",
    }


def highmotion_row(task):
    return {
        "task": task,
        "method": "synthetic_hm",
        "result_json": "synthetic-not-read.json",
        "grid_acc": "1",
    }


def write_synthetic_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("task", HM_TASKS)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("kind", ["markdown", "csv"])
def test_direct_writers_refuse_before_any_write(tmp_path, task, mixed, existing, kind):
    rows = ([educational_row()] if mixed else []) + [highmotion_row(task)]
    unchanged = copy.deepcopy(rows)
    output = tmp_path / "output" / ("leaderboard.md" if kind == "markdown" else "leaderboard.csv")
    if existing:
        output.parent.mkdir()
        output.write_bytes(b"existing output must survive\n")

    with pytest.raises(ValueError, match="High-Motion publication is withheld") as error:
        if kind == "markdown":
            exporter.write_markdown(output, rows, [], False)
        else:
            exporter.write_csv(output, rows, ["task", "method"])

    assert task in str(error.value)
    assert "docs/HIGHMOTION_TARGET_HOLD.md" in str(error.value)
    assert rows == unchanged
    if existing:
        assert output.read_bytes() == b"existing output must survive\n"
    else:
        assert not output.parent.exists()


@pytest.mark.parametrize("task", HM_TASKS)
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_cli_refuses_both_outputs_before_calling_writers(
    tmp_path, monkeypatch, capsys, task, mixed, existing
):
    rows = ([educational_row()] if mixed else []) + [highmotion_row(task)]
    summary = tmp_path / "synthetic.csv"
    write_synthetic_csv(summary, rows)
    output_dir = tmp_path / "output"
    markdown = output_dir / "leaderboard.md"
    csv_output = output_dir / "leaderboard.csv"
    if existing:
        output_dir.mkdir()
        markdown.write_bytes(b"existing markdown\n")
        csv_output.write_bytes(b"existing csv\n")

    def unexpected_write(*args, **kwargs):
        pytest.fail("CLI must reject held rows before calling either output writer")

    monkeypatch.setattr(exporter, "write_markdown", unexpected_write)
    monkeypatch.setattr(exporter, "write_csv", unexpected_write)
    monkeypatch.setattr("sys.argv", [
        "build_leaderboard", "--summary-csv", str(summary),
        "--out-md", str(markdown), "--out-csv", str(csv_output),
    ])
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 2
    assert "High-Motion publication is withheld" in capsys.readouterr().err
    if existing:
        assert markdown.read_bytes() == b"existing markdown\n"
        assert csv_output.read_bytes() == b"existing csv\n"
    else:
        assert not output_dir.exists()


@pytest.mark.parametrize("subtask", ("highfps",) + HM_TASKS)
def test_cli_api_task_mapping_cannot_bypass_hold(tmp_path, monkeypatch, capsys, subtask):
    summary = tmp_path / "synthetic-api.csv"
    write_synthetic_csv(summary, [{
        "Subtask": subtask, "Model": "synthetic_api", "Provider": "api", "GridAcc": "1",
    }])
    output = tmp_path / "output" / "leaderboard.md"
    monkeypatch.setattr("sys.argv", [
        "build_leaderboard", "--api-summary-csv", str(summary),
        "--include-closed-source", "--out-md", str(output),
    ])
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 2
    assert "High-Motion publication is withheld" in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize("task", ("densevideo", "dive_bench_educational_high_fps"))
def test_educational_cli_bytes_unchanged(tmp_path, monkeypatch, task):
    class FrozenDatetime:
        @staticmethod
        def now():
            return datetime(2026, 9, 14, 12, 0, 0)

    monkeypatch.setattr(exporter, "datetime", FrozenDatetime)
    row = educational_row()
    row["task"] = task
    summary = tmp_path / "synthetic.csv"
    write_synthetic_csv(summary, [row])
    output = tmp_path / "leaderboard.md"
    csv_output = tmp_path / "leaderboard.csv"
    monkeypatch.setattr("sys.argv", [
        "build_leaderboard", "--summary-csv", str(summary),
        "--out-md", str(output), "--out-csv", str(csv_output),
    ])
    exporter.main()
    expected = (
        "# DIVE-Bench Leaderboard\n\n"
        "- generated_at: `2026-09-14T12:00:00`\n"
        "- include_closed_source: `false`\n"
        "- source_artifacts: `1`\n\n"
        f"## {exporter.TASK_LABELS[task]}\n\n"
        "| rank | model | method_id | samples | open_mos ↑ | token_f1 ↑ | cer ↓ | wer ↓ | "
        "exact_match ↑ | patch_projection_recompute_ratio ↓ | reference_patch_compute_ratio ↓ | "
        "sampling_density_fps | throughput_fps ↑ | source |\n"
        "| " + " | ".join(["---"] * 14) + " |\n"
        "| 1 | synthetic_edu | synthetic_edu |  | 2 | 0.5 |  |  |  |  |  |  |  | open |\n"
    )
    assert output.read_bytes() == expected.encode("utf-8")
    expected_csv = (
        "display_name,method,open_mos,rank,result_json,source,task,token_f1\r\n"
        f"synthetic_edu,synthetic_edu,2,1,synthetic-not-read.json,open,{task},0.5\r\n"
    )
    assert csv_output.read_bytes() == expected_csv.encode("utf-8")
