from tools.densevideo.build_leaderboard import rank_rows, write_markdown


def test_canonical_educational_ranks_by_mos():
    rows = [{"method": "a", "open_mos": 2, "token_f1": .1},
            {"method": "b", "open_mos": 1, "token_f1": .9}]
    assert rank_rows(rows, "dive_bench_educational_high_fps")[0]["method"] == "a"


def test_full_and_preview_are_separate_tables(tmp_path):
    rows = [{"task": "dive_bench_high_motion_high_fps", "method": "a", "grid_acc": .2},
            {"task": "dive_bench_high_motion_high_fps_preview1000", "method": "b", "grid_acc": .9}]
    path = tmp_path / "leaderboard.md"
    write_markdown(path, rows, [], False)
    text = path.read_text()
    assert "(full split)" in text
    assert "(1000-item preview)" in text
    assert text.count("| 1 |") == 2
    assert "Educational Dense Video" not in text
