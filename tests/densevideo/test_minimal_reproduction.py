import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.densevideo import reproduce_grt
from tools.densevideo.deterministic_worker import trace_floor_requests
from tools.densevideo.reproduce_grt import (
    build_plan,
    evaluate_fresh_quality,
    initial_quality_status,
    load_run_samples,
    main,
    profiles,
    run_requested_mos,
    validate_controls,
    validate_floor,
    validate_resolved_result,
    validate_telemetry,
)


@pytest.mark.parametrize("profile,cap", [("route31", 128), ("qwen3", 128), ("qwen7", 48)])
def test_published_plan_pins_revision_budget_and_serial_controls(profile, cap, tmp_path):
    plan = build_plan(profile, tmp_path)
    assert plan["samples"] == 634
    assert [x["role"] for x in plan["commands"]] == ["base", "all", "candidate"]
    for row in plan["commands"]:
        cmd = row["command"]
        assert "tools.densevideo.deterministic_worker" in cmd
        assert "revision=" in cmd[cmd.index("--model_args") + 1]
        assert cmd[cmd.index("--gen_kwargs") + 1] == f"max_new_tokens={cap},temperature=0"
        assert cmd[cmd.index("--seed") + 1] == "0,1234,1234,1234"
        assert Path(row["output"]).parts[-3:] == ("runs", row["method"], "densevideo")


@pytest.mark.parametrize(
    "profile,revision",
    [
        ("route31", "74dd0bf867a4cda7950c17663794267c60cf4b40"),
        ("qwen3", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
        ("qwen7", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ],
)
def test_exact_published_model_revision(profile, revision):
    for entry in profiles()["profiles"][profile]["roles"].values():
        assert f"revision={revision}," in entry["model_args"]


def test_profiles_match_retained_source_configs():
    import yaml

    root = Path(__file__).resolve().parents[2]
    for profile, filename in (
        ("route31", "grt_hf_route31_full_models.yaml"),
        ("qwen7", "grt_qwen7_floor_cap48_full_models.yaml"),
    ):
        config = yaml.safe_load((root / "configs/densevideo" / filename).read_text())
        wanted = {row["method"]: row for row in config["models"]}
        for row in profiles()["profiles"][profile]["roles"].values():
            assert row["model"] == wanted[row["method"]]["model"]
            assert row["model_args"] == wanted[row["method"]]["model_args"]


def test_plan_does_not_create_outputs_or_claim_smoke_as_full(tmp_path, capsys):
    target = tmp_path / "not-created"
    assert main(["--profile", "qwen7", "--output", str(target), "--limit", "1"]) == 0
    assert not target.exists()
    assert json.loads(capsys.readouterr().out)["published_profile"] is False


def mos_phase_inputs(tmp_path):
    output = tmp_path / "reproduction"
    output.mkdir()
    plan = {
        "samples": 2,
        "commands": [
            {"method": "base_method"},
            {"method": "all_method"},
            {"method": "candidate_method"},
        ],
    }
    report = {
        "status": "inference_validated",
        "historical_score_equality": "not_checked",
        "quality": initial_quality_status(True),
    }
    report_path = output / "reproduction_validation.json"
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return output, plan, report, report_path


def test_mos_request_status_is_truthful_before_judge_and_no_mos_is_unchanged():
    assert initial_quality_status(True) == {"status": "pending"}
    assert initial_quality_status(False) == {
        "status": "not_evaluated",
        "reason": "Open MOS not requested",
    }


@pytest.mark.parametrize(
    "phase,error",
    [
        ("judge_execution", subprocess.CalledProcessError(137, ["private", "secret"])),
        ("quality_validation", ValueError("private path and prompt must not be recorded")),
    ],
)
def test_requested_mos_exceptions_persist_sanitized_failure_and_propagate(
    tmp_path, monkeypatch, phase, error
):
    output, plan, report, report_path = mos_phase_inputs(tmp_path)
    observed = []

    def judge(*args, **kwargs):
        observed.append(json.loads(report_path.read_text(encoding="utf-8"))["quality"])
        if phase == "judge_execution":
            raise error

    def evaluate(*args, **kwargs):
        if phase == "quality_validation":
            raise error
        pytest.fail("quality validation must not run after judge failure")

    monkeypatch.setattr(reproduce_grt.subprocess, "run", judge)
    monkeypatch.setattr(reproduce_grt, "evaluate_fresh_quality", evaluate)
    with pytest.raises(type(error)) as raised:
        run_requested_mos(plan, {}, {}, output, {}, {}, report, report_path)
    assert raised.value is error
    assert observed == [{"status": "pending"}]
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["quality"] == {
        "status": "failed",
        "phase": phase,
        "error_class": type(error).__name__,
    }
    assert saved["historical_score_equality"] == "not_checked"
    assert "private path and prompt" not in report_path.read_text(encoding="utf-8")
    assert "secret" not in report_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "quality",
    [
        {},
        {"status": "pending", "historical_score_equality": False},
        {"status": "passed", "historical_score_equality": "not_checked"},
    ],
)
def test_malformed_quality_result_is_a_sanitized_validation_failure(
    tmp_path, monkeypatch, quality
):
    output, plan, report, report_path = mos_phase_inputs(tmp_path)
    monkeypatch.setattr(reproduce_grt.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(reproduce_grt, "evaluate_fresh_quality", lambda *args: quality)
    with pytest.raises(ValueError, match="invalid status"):
        run_requested_mos(plan, {}, {}, output, {}, {}, report, report_path)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["quality"] == {
        "status": "failed",
        "phase": "quality_validation",
        "error_class": "ValueError",
    }


@pytest.mark.parametrize(
    "quality",
    [
        {"status": "smoke_only", "historical_score_equality": False},
        {"status": "passed", "historical_score_equality": True, "gate": {"passed": True}},
        {"status": "failed", "historical_score_equality": False, "reason": "quality floor"},
    ],
)
def test_requested_mos_result_replaces_pending_status(tmp_path, monkeypatch, quality):
    output, plan, report, report_path = mos_phase_inputs(tmp_path)
    commands = []

    def judge(command, **kwargs):
        assert json.loads(report_path.read_text(encoding="utf-8"))["quality"] == {
            "status": "pending"
        }
        commands.append(command)

    monkeypatch.setattr(reproduce_grt.subprocess, "run", judge)
    monkeypatch.setattr(reproduce_grt, "evaluate_fresh_quality", lambda *args: quality)
    assert run_requested_mos(plan, {}, {}, output, {}, {}, report, report_path) is quality
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["quality"] == quality
    assert saved["historical_score_equality"] is quality["historical_score_equality"]
    assert commands[0].count("--method") == 3
    assert [commands[0][index + 1] for index, value in enumerate(commands[0]) if value == "--method"] == [
        "base_method",
        "all_method",
        "candidate_method",
    ]


def test_main_with_mos_persists_failed_judge_status_and_propagates(
    tmp_path, monkeypatch
):
    from tools.densevideo import rebuild_published_leaderboard, run_open_leaderboard

    output = tmp_path / "main-output"
    bundle = tmp_path / "bundle"
    artifacts = tmp_path / "artifacts"
    bundle.mkdir()
    artifacts.mkdir()
    sample = raw_sample(0)
    identity_bytes = (
        f"doc_id,identity_sha256\n0,{sample_digest(sample)}\n".encode()
    )
    (bundle / "sample_identities.csv").write_bytes(identity_bytes)
    (bundle / "provenance.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(reproduce_grt, "IDENTITY_SHA256", hashlib.sha256(identity_bytes).hexdigest())
    monkeypatch.setattr(reproduce_grt, "resolve_release_bundle", lambda override: bundle)
    monkeypatch.setattr(rebuild_published_leaderboard, "verify_bundle", lambda path: None)

    checked = {}
    plan = build_plan("qwen3", output.resolve(), 1)
    wanted = profiles()["profiles"]["qwen3"]
    for entry in plan["commands"]:
        sample_path = artifacts / f"{entry['role']}-samples.jsonl"
        result_path = artifacts / f"{entry['role']}-result.json"
        write_rows(sample_path, [sample])
        role = wanted["roles"][entry["role"]]
        result_path.write_text(
            json.dumps(
                {
                    "config": {
                        "model": role["model"],
                        "model_args": role["model_args"],
                        "gen_kwargs": {"max_new_tokens": wanted["max_new_tokens"], "temperature": 0},
                        "limit": 1,
                        "batch_size": "1",
                        "random_seed": 0,
                        "numpy_seed": 1234,
                        "torch_seed": 1234,
                        "fewshot_seed": 1234,
                    }
                }
            ),
            encoding="utf-8",
        )
        checked[entry["output"]] = SimpleNamespace(
            usable=True,
            sample_count=1,
            sample_paths=[sample_path],
            result_path=result_path,
        )
    monkeypatch.setattr(
        run_open_leaderboard,
        "validate_run_output",
        lambda path, **kwargs: checked[str(path)],
    )

    judge_error = subprocess.CalledProcessError(137, ["judge", "private-secret"])

    def execute(command, **kwargs):
        if "tools.densevideo.deterministic_worker" in command:
            kwargs["stdout"].write(
                "[DIVE_RUNTIME] " + json.dumps(RUNTIME) + "\n"
                "[DENSE_METRICS] recomputed_patches=10 orig_patches=10 "
                "reference_orig_patches=10\n"
            )
        elif "tools.densevideo.score_open_mos_matrix" in command:
            pending = json.loads(
                (output / "reproduction_validation.json").read_text(encoding="utf-8")
            )
            assert pending["quality"] == {"status": "pending"}
            raise judge_error
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reproduce_grt.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        main(
            [
                "--profile",
                "qwen3",
                "--output",
                str(output),
                "--bundle",
                str(bundle),
                "--limit",
                "1",
                "--execute",
                "--with-mos",
            ]
        )
    assert raised.value is judge_error
    saved = json.loads(
        (output / "reproduction_validation.json").read_text(encoding="utf-8")
    )
    assert saved["quality"] == {
        "status": "failed",
        "phase": "judge_execution",
        "error_class": "CalledProcessError",
    }
    assert "private-secret" not in json.dumps(saved)


@pytest.mark.parametrize("profile", ["route31", "qwen3", "qwen7"])
def test_all_profile_plans_resolve_the_shared_default_bundle(
    profile, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert main(["--profile", profile, "--output", str(tmp_path / profile)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert Path(plan["bundle"]).name == "2026-08-20"
    assert Path(plan["bundle"]).is_dir()


@pytest.mark.parametrize("limit", [0, 635, 634.9, True])
def test_invalid_limit_is_rejected(limit, tmp_path):
    with pytest.raises(ValueError):
        build_plan("qwen7", tmp_path, limit)


def test_qwen7_route_floor_profile_is_complete():
    args = profiles()["profiles"]["qwen7"]["roles"]["candidate"]["model_args"]
    for item in (
        "subtitle_gate_diff_threshold=3.0",
        "ocr_gate_diff_threshold=30.0",
        "subtitle_min_gate_keep_ratio=0.80",
        "ocr_min_gate_keep_ratio=0.55",
    ):
        assert item in args


def test_only_dive_tasks_are_included():
    root = Path(__file__).resolve().parents[2]
    task_dirs = {p.parent.name for p in (root / "lmms_eval/tasks").rglob("*.yaml")}
    assert task_dirs == {"densevideo"}


def raw_sample(doc_id=0):
    return {
        "doc_id": doc_id,
        "doc": {
            "qid": f"v{doc_id}_sub",
            "video": f"v{doc_id}",
            "video_path": f"v{doc_id}.mp4",
            "question": "What subtitles appear?",
            "answer": "example",
            "type": "QA",
        },
        "input": "What subtitles appear? Complete answer.",
        "target": "example",
        "filtered_resps": ["example"],
        "token_f1": 1.0,
    }


def sample_digest(row):
    doc = row["doc"]
    return hashlib.sha256(
        json.dumps(
            [row["doc_id"], doc["qid"], doc["video"], doc["type"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "string_id",
        "identity",
        "empty",
        "bad_f1",
        "missing_prompt",
        "invalid_route",
    ],
)
def test_sample_integrity_rejects_malformed_or_changed_rows(tmp_path, mutation):
    rows = [raw_sample(0), raw_sample(1)]
    identities = {row["doc_id"]: sample_digest(row) for row in rows}
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "string_id":
        rows[0]["doc_id"] = "0"
    elif mutation == "identity":
        rows[0]["doc"]["qid"] = "other_sub"
    elif mutation == "empty":
        rows[0]["filtered_resps"] = [""]
    elif mutation == "bad_f1":
        rows[0]["token_f1"] = float("nan")
    elif mutation == "missing_prompt":
        del rows[0]["input"]
    elif mutation == "invalid_route":
        rows[0]["doc"]["qid"] = "other"
        identities[0] = sample_digest(rows[0])
    with pytest.raises(ValueError):
        load_run_samples(write_rows(tmp_path / "samples.jsonl", rows), 2, identities)


def validated_samples(tmp_path, count=2):
    rows = [raw_sample(i) for i in range(count)]
    return load_run_samples(
        write_rows(tmp_path / "samples.jsonl", reversed(rows)),
        count,
        {row["doc_id"]: sample_digest(row) for row in rows},
    )


RUNTIME = {
    "hostname": "testhost",
    "gpu_uuid": "GPU-unique",
    "deterministic": True,
    "warn_only": False,
    "tf32": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "evaluation_seeds": [0, 1234, 1234, 1234],
}


def runtime_logs(tmp_path):
    logs = {}
    for role in ("base", "all", "candidate"):
        logs[role] = tmp_path / f"{role}.log"
        logs[role].write_text("[DIVE_RUNTIME] " + json.dumps(RUNTIME) + "\n")
    return logs


@pytest.mark.parametrize("profile,raises", [("route31", True), ("qwen3", False)])
def test_base_all_equality_preserves_profile_specific_contract(tmp_path, profile, raises):
    samples = {
        role: copy.deepcopy(validated_samples(tmp_path)) for role in ("base", "all", "candidate")
    }
    samples["all"][1]["prediction"] = "different"
    plan, logs = build_plan(profile, tmp_path, 2), runtime_logs(tmp_path)
    if raises:
        with pytest.raises(ValueError, match="base/all"):
            validate_controls(plan, samples, logs)
    else:
        assert validate_controls(plan, samples, logs)["base_all_identical"] == 1


@pytest.mark.parametrize("mutation", ["uuid", "tf32", "missing_runtime", "prompt"])
def test_control_validation_fails_closed_on_runtime_or_identity_changes(tmp_path, mutation):
    samples = {
        role: copy.deepcopy(validated_samples(tmp_path)) for role in ("base", "all", "candidate")
    }
    logs = runtime_logs(tmp_path)
    if mutation == "prompt":
        samples["candidate"][0]["identity"][-2] = "changed prompt"
    elif mutation == "missing_runtime":
        logs["candidate"].write_text("")
    else:
        runtime = dict(RUNTIME)
        runtime["gpu_uuid" if mutation == "uuid" else "tf32"] = (
            "GPU-different" if mutation == "uuid" else True
        )
        logs["candidate"].write_text("[DIVE_RUNTIME] " + json.dumps(runtime) + "\n")
    with pytest.raises(ValueError):
        validate_controls(build_plan("route31", tmp_path, 2), samples, logs)


def floor_log(tmp_path, *, planned=7, triggered=True, order=(1, 0)):
    lines = ["[DIVE_REQUEST_ORDER] " + json.dumps(order)]
    for _ in order:
        actual = 10 if triggered else planned
        policy = "all" if triggered else "motion"
        projection = "native_on_full" if triggered else "linear_consistent"
        lines.extend(
            [
                "[DENSE_ROUTE_GATE] question_route=subtitle min_gate_keep_ratio=0.8 "
                + "gate_diff_threshold=3.0 prompt_router=densevideo_lpm_first_sentence_v1 gate_policy=motion",
                "[DENSE_GATE_FLOOR] question_route=subtitle min_gate_keep_ratio=0.8 "
                + f"planned_recomputed_patches={planned} orig_patches=10 triggered={str(triggered).lower()} "
                f"actual_recomputed_patches={actual} actual_gate_policy={policy} "
                f"actual_projection={projection} planned_gate_keep_ratio={planned / 10} "
                f"actual_gate_keep_ratio={actual / 10}",
                f"[DENSE_METRICS] recomputed_patches={actual} orig_patches=10 gate_policy={policy}",
            ]
        )
    path = tmp_path / "floor.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_floor_validation_uses_actual_request_order_and_strict_boundary(tmp_path):
    samples = validated_samples(tmp_path)
    report = validate_floor(floor_log(tmp_path), samples, samples)
    assert report["fallback_ids"] == [1, 0]
    assert validate_floor(floor_log(tmp_path, planned=8, triggered=False), samples, samples) == {
        "checked": 2,
        "fallback_ids": [],
    }


@pytest.mark.parametrize("mutation", ["prediction", "counts", "decision", "order", "coverage"])
def test_floor_validation_rejects_false_fallback_and_partial_logs(tmp_path, mutation):
    base = validated_samples(tmp_path)
    candidate = copy.deepcopy(base)
    log = floor_log(tmp_path)
    if mutation == "prediction":
        candidate[1]["prediction"] = "changed"
    elif mutation == "counts":
        log.write_text(
            log.read_text().replace("actual_recomputed_patches=10", "actual_recomputed_patches=9")
        )
    elif mutation == "decision":
        log = floor_log(tmp_path, planned=8, triggered=True)
    elif mutation == "order":
        log = floor_log(tmp_path, order=(0, 0))
    elif mutation == "coverage":
        log.write_text(log.read_text().replace("[DENSE_METRICS]", "[OTHER]", 1))
    with pytest.raises(ValueError):
        validate_floor(log, candidate, base)


def test_trace_preserves_requests_responses_and_exposes_order(capsys):
    class Model:
        def generate_until(self, requests):
            self.seen = requests
            return ["answer"] * len(requests)

    requests = [SimpleNamespace(args=("prompt", {}, None, i)) for i in (3, 0)]
    trace_floor_requests(Model)
    model = Model()
    assert model.generate_until(iter(requests)) == ["answer", "answer"]
    assert model.seen == requests
    assert capsys.readouterr().out == "[DIVE_REQUEST_ORDER] [3, 0]\n"


def test_resolved_config_enforces_model_args_generation_and_seeds(tmp_path):
    plan = build_plan("qwen7", tmp_path, 2)
    entry = plan["commands"][0]
    model = profiles()["profiles"]["qwen7"]["roles"]["base"]
    config = {
        "model": model["model"],
        "model_args": model["model_args"],
        "gen_kwargs": {"max_new_tokens": 48, "temperature": 0},
        "limit": 2,
        "batch_size": "1",
        "random_seed": 0,
        "numpy_seed": 1234,
        "torch_seed": 1234,
        "fewshot_seed": 1234,
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"config": config}))
    validate_resolved_result(path, entry, plan)
    for field in ("model_args", "gen_kwargs", "torch_seed", "limit"):
        changed = dict(config, **{field: None})
        path.write_text(json.dumps({"config": changed}))
        with pytest.raises(ValueError):
            validate_resolved_result(path, entry, plan)


def test_patch_telemetry_requires_complete_counts_not_means(tmp_path):
    path = tmp_path / "metrics.log"
    path.write_text(
        "[DENSE_METRICS] recomputed_patches=5 orig_patches=10 reference_orig_patches=20\n"
        "[DENSE_METRICS] recomputed_patches=15 orig_patches=20 reference_orig_patches=20\n"
    )
    report = validate_telemetry(path, 2, control=False)
    assert report["mean_recompute_ratio"] == 20 / 30
    assert report["reference_patch_compute_ratio"] == 0.5
    with pytest.raises(ValueError):
        validate_telemetry(path, 2, control=True)
    with pytest.raises(ValueError):
        validate_telemetry(path, 3, control=False)


def test_judge_pins_and_coverage_are_checked_even_for_smoke(tmp_path):
    plan = build_plan("qwen3", tmp_path, 2)
    samples = {
        role: copy.deepcopy(validated_samples(tmp_path)) for role in ("base", "all", "candidate")
    }
    provenance = {"judge": {"fingerprint": "pinned"}}
    judge = profiles()["judge"]
    rows = [
        {
            "method": entry["method"],
            "doc_id": i,
            "open_mos_score": 2,
            "mos_judge_model": judge["model"],
            "mos_judge_revision": judge["revision"],
            "judge_fingerprint": "pinned",
            "pred": "example",
            "answer": "example",
            "question": "What subtitles appear?",
            "error": "",
        }
        for entry in plan["commands"]
        for i in range(2)
    ]
    telemetry = {role: {} for role in samples}
    matrix = write_rows(tmp_path / "matrix.jsonl", rows)
    result = evaluate_fresh_quality(plan, samples, telemetry, matrix, provenance)
    assert result["status"] == "smoke_only"
    for field in ("mos_judge_revision", "judge_fingerprint", "pred", "answer", "question"):
        changed = copy.deepcopy(rows)
        changed[0][field] = "wrong"
        with pytest.raises(ValueError):
            evaluate_fresh_quality(
                plan, samples, telemetry, write_rows(matrix, changed), provenance
            )
    with pytest.raises(ValueError):
        evaluate_fresh_quality(plan, samples, telemetry, write_rows(matrix, rows[:-1]), provenance)
