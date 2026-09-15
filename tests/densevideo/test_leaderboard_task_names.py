from tools.densevideo.build_leaderboard import TASK_COLUMNS, TASK_LABELS, markdown_table, rank_rows


def test_canonical_educational_ranks_by_mos():
    rows = [{"method": "a", "open_mos": 2, "token_f1": .1},
            {"method": "b", "open_mos": 1, "token_f1": .9}]
    assert rank_rows(rows, "dive_bench_educational_high_fps")[0]["method"] == "a"


def test_full_and_preview_remain_separate_raw_task_helpers():
    full = "dive_bench_high_motion_high_fps"
    preview = "dive_bench_high_motion_high_fps_preview1000"
    assert "(full split)" in TASK_LABELS[full]
    assert "(1000-item preview)" in TASK_LABELS[preview]
    for task, score in ((full, .2), (preview, .9)):
        rows = [{"task": task, "method": "synthetic", "grid_acc": score}]
        ranked = rank_rows(rows, task)
        assert ranked[0]["rank"] == 1
        assert "| 1 |" in markdown_table(ranked, TASK_COLUMNS[task])
