"""Portable serial execution of the three published LPM GRT profiles.

Prints a plan by default. --execute runs fresh inference; --with-mos also
scores all three fresh methods with the pinned judge. Archived public controls
are contained in the published evidence bundle, not silently substituted here.
"""

import argparse
import csv
import hashlib
import importlib.resources
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

EVALUATION_SEEDS = "0,1234,1234,1234"
IDENTITY_SHA256 = "27ea477ec792f645fabffec4b813ae2946c4e4cc48049ae15961cbb3de7b7ec7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def profiles():
    resource = importlib.resources.files("tools.densevideo").joinpath("profiles.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def build_plan(profile, output, limit=None):
    spec = profiles()["profiles"][profile]
    count = spec["samples"] if limit is None else limit
    if type(count) is not int or not 1 <= count <= spec["samples"]:
        raise ValueError("limit must be an integer between 1 and 634")
    commands = []
    for role in ("base", "all", "candidate"):
        model = spec["roles"][role]
        run_dir = output / "runs" / model["method"] / spec["task"]
        command = [
            sys.executable,
            "-m",
            "tools.densevideo.deterministic_worker",
            "--",
            "--model",
            model["model"],
            "--model_args",
            model["model_args"],
            "--tasks",
            spec["task"],
            "--batch_size",
            "1",
            "--limit",
            str(count),
            "--seed",
            EVALUATION_SEEDS,
            "--gen_kwargs",
            f"max_new_tokens={spec['max_new_tokens']},temperature=0",
            "--log_samples",
            "--output_path",
            str(run_dir),
        ]
        commands.append(
            {"role": role, "method": model["method"], "output": str(run_dir), "command": command}
        )
    return {
        "profile": profile,
        "samples": count,
        "task": spec["task"],
        "published_profile": count == spec["samples"],
        "commands": commands,
    }


def json_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"Expected an object in {path}")
                yield row


def load_run_samples(path, count, identities):
    """Validate typed row identity and keep only small comparison fields."""
    result = {}
    for row in json_rows(path):
        doc_id = row.get("doc_id")
        require(type(doc_id) is int and 0 <= doc_id < count, "Invalid integer doc_id")
        require(doc_id not in result, f"Duplicate doc_id {doc_id}")
        doc = row.get("doc")
        require(isinstance(doc, dict), "Sample has no typed doc")
        for field in ("qid", "video", "video_path", "question", "answer", "type"):
            require(isinstance(doc.get(field), str) and doc[field], f"Invalid doc.{field}")
        key = [doc_id, doc["qid"], doc["video"], doc["type"]]
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        require(digest == identities.get(doc_id), f"Published sample identity changed: {doc_id}")
        prediction = row.get("filtered_resps")
        require(
            isinstance(prediction, list)
            and len(prediction) == 1
            and isinstance(prediction[0], str)
            and prediction[0].strip(),
            f"Missing singleton nonempty prediction: {doc_id}",
        )
        require(
            isinstance(row.get("input"), str) and isinstance(row.get("target"), str),
            f"Missing typed prompt or reference: {doc_id}",
        )
        f1 = row.get("token_f1")
        require(
            type(f1) in (int, float) and math.isfinite(f1) and 0 <= f1 <= 1,
            f"Invalid per-sample token_f1: {doc_id}",
        )
        route = "subtitle" if doc["qid"].endswith("_sub") else "ocr"
        require(
            doc["qid"].endswith("_sub" if route == "subtitle" else "_ocr"),
            f"Unknown educational route: {doc_id}",
        )
        result[doc_id] = {
            "identity": [
                doc[field] for field in ("qid", "video", "video_path", "question", "answer", "type")
            ]
            + [row["input"], row["target"]],
            "prediction": prediction[0],
            "token_f1": f1,
            "route": route,
        }
    require(set(result) == set(range(count)), f"Samples must cover exact doc_id 0..{count - 1}")
    return result


def tagged_rows(path, tag, *, json_payload=False):
    prefix = f"[{tag}] "
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if prefix not in line:
                continue
            payload = line.split(prefix, 1)[1].strip()
            yield (
                json.loads(payload)
                if json_payload
                else dict(item.split("=", 1) for item in payload.split() if "=" in item)
            )


def runtime_identity(path):
    rows = list(tagged_rows(path, "DIVE_RUNTIME", json_payload=True))
    require(len(rows) == 1, "Expected exactly one deterministic runtime record")
    row = rows[0]
    require(
        row.get("deterministic") is True
        and row.get("warn_only") is False
        and row.get("tf32") is False
        and row.get("cudnn_deterministic") is True
        and row.get("cudnn_benchmark") is False
        and bool(row.get("gpu_uuid"))
        and row.get("evaluation_seeds") == [0, 1234, 1234, 1234],
        "Missing strict deterministic CUDA runtime settings",
    )
    return row


def validate_floor(log, candidate, base):
    """Recompute every floor decision and compare native-fallback predictions."""
    orders = list(tagged_rows(log, "DIVE_REQUEST_ORDER", json_payload=True))
    require(len(orders) == 1, "Missing unique floor request-order trace")
    order = orders[0]
    require(
        isinstance(order, list)
        and all(type(i) is int for i in order)
        and len(order) == len(candidate)
        and set(order) == set(candidate),
        "Floor request order does not cover exact sample IDs",
    )
    floors = list(tagged_rows(log, "DENSE_GATE_FLOOR"))
    dense = list(tagged_rows(log, "DENSE_METRICS"))
    routes = list(tagged_rows(log, "DENSE_ROUTE_GATE"))
    require(
        len(floors) == len(dense) == len(routes) == len(order),
        "Incomplete floor/route/dense telemetry",
    )
    fallback_ids = []
    for doc_id, floor, metrics, route in zip(order, floors, dense, routes):
        name = candidate[doc_id]["route"]
        minimum = {"subtitle": 0.80, "ocr": 0.55}[name]
        threshold = {"subtitle": 3.0, "ocr": 30.0}[name]
        planned, orig = int(floor["planned_recomputed_patches"]), int(floor["orig_patches"])
        require(0 < planned <= orig, "Invalid floor patch counts")
        triggered = planned / orig < minimum
        actual = orig if triggered else planned
        policy = "all" if triggered else "motion"
        require(
            floor.get("question_route") == route.get("question_route") == name
            and float(floor["min_gate_keep_ratio"]) == minimum
            and float(route["min_gate_keep_ratio"]) == minimum
            and float(route["gate_diff_threshold"]) == threshold
            and route.get("prompt_router") == "densevideo_lpm_first_sentence_v1"
            and route.get("gate_policy") == "motion"
            and floor.get("triggered") == str(triggered).lower()
            and floor.get("actual_gate_policy") == metrics.get("gate_policy") == policy
            and floor.get("actual_projection")
            == ("native_on_full" if triggered else "linear_consistent")
            and int(floor["actual_recomputed_patches"]) == actual
            and int(metrics["recomputed_patches"]) == actual
            and int(metrics["orig_patches"]) == orig
            and math.isclose(
                float(floor["planned_gate_keep_ratio"]), planned / orig, rel_tol=0, abs_tol=2e-6
            )
            and math.isclose(
                float(floor["actual_gate_keep_ratio"]), actual / orig, rel_tol=0, abs_tol=2e-6
            ),
            f"Incorrect floor decision: doc_id {doc_id}",
        )
        if triggered:
            require(
                candidate[doc_id]["prediction"] == base[doc_id]["prediction"],
                f"Native fallback differs from base: doc_id {doc_id}",
            )
            fallback_ids.append(doc_id)
    return {"checked": len(order), "fallback_ids": fallback_ids}


def validate_controls(plan, samples, logs):
    base = samples["base"]
    for role in ("all", "candidate"):
        require(set(samples[role]) == set(base), f"{role} sample IDs differ")
        require(
            all(base[i]["identity"] == samples[role][i]["identity"] for i in base),
            f"{role} prompt/reference/video identity differs",
        )
    equal = sum(base[i]["prediction"] == samples["all"][i]["prediction"] for i in base)
    strict = plan["profile"] in {"route31", "qwen7"}
    require(not strict or equal == len(base), "Same-session base/all predictions differ")
    runtimes = [runtime_identity(logs[role]) for role in ("base", "all", "candidate")]
    require(all(row == runtimes[0] for row in runtimes), "Serial arms changed device/runtime")
    report = {
        "samples": len(base),
        "base_all_identical": equal,
        "base_all_equality_required": strict,
        "runtime": runtimes[0],
    }
    if plan["profile"] == "qwen7":
        report["route_floor"] = validate_floor(logs["candidate"], samples["candidate"], base)
    return report


def validate_resolved_result(path, entry, plan):
    result = json.loads(path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    wanted = profiles()["profiles"][plan["profile"]]
    require(
        config.get("model") == wanted["roles"][entry["role"]]["model"]
        and config.get("model_args") == wanted["roles"][entry["role"]]["model_args"]
        and config.get("gen_kwargs")
        == {"max_new_tokens": wanted["max_new_tokens"], "temperature": 0}
        and config.get("limit") == plan["samples"]
        and str(config.get("batch_size")) == "1"
        and [config.get(key) for key in ("random_seed", "numpy_seed", "torch_seed", "fewshot_seed")]
        == [0, 1234, 1234, 1234],
        f"Resolved inference settings differ for {entry['role']}",
    )


def validate_telemetry(log, count, *, control):
    rows = list(tagged_rows(log, "DENSE_METRICS"))
    require(len(rows) == count, "Incomplete per-request patch telemetry")
    totals = {key: 0 for key in ("recomputed_patches", "orig_patches", "reference_orig_patches")}
    for row in rows:
        values = {key: int(row[key]) for key in totals}
        require(
            0 < values["recomputed_patches"] <= values["orig_patches"]
            and values["reference_orig_patches"] > 0,
            "Invalid measured patch counts",
        )
        require(
            not control or len(set(values.values())) == 1,
            "Base/all control does not compute all reference patches",
        )
        for key, value in values.items():
            totals[key] += value
    return {
        "mean_recompute_ratio": totals["recomputed_patches"] / totals["orig_patches"],
        "reference_patch_compute_ratio": totals["recomputed_patches"]
        / totals["reference_orig_patches"],
        **totals,
    }


def evaluate_fresh_quality(plan, samples, telemetry, matrix, provenance):
    """Validate MOS coverage and report gates separately from historical equality."""
    judge = profiles()["judge"]
    methods = {entry["method"]: entry["role"] for entry in plan["commands"]}
    scored = {role: {} for role in samples}
    for row in json_rows(matrix):
        method, doc_id = row.get("method"), row.get("doc_id")
        require(method in methods, "Unexpected MOS method")
        role = methods[method]
        require(
            type(doc_id) is int and doc_id in samples[role] and doc_id not in scored[role],
            "Unexpected or duplicate MOS doc_id",
        )
        score = row.get("open_mos_score")
        require(
            not row.get("error")
            and type(score) is int
            and 0 <= score <= 5
            and row.get("mos_judge_model") == judge["model"]
            and row.get("mos_judge_revision") == judge["revision"]
            and row.get("judge_fingerprint") == provenance["judge"]["fingerprint"]
            and row.get("pred") == samples[role][doc_id]["prediction"]
            and row.get("question") == samples[role][doc_id]["identity"][3]
            and row.get("answer") == samples[role][doc_id]["identity"][4],
            "MOS result differs from pinned judge protocol or fresh sample",
        )
        scored[role][doc_id] = score
    rows = {}
    for method, role in methods.items():
        require(set(scored[role]) == set(samples[role]), "Incomplete MOS document coverage")
        rows[method] = {
            "samples": plan["samples"],
            "open_mos": math.fsum(scored[role].values()) / plan["samples"],
            "token_f1": math.fsum(row["token_f1"] for row in samples[role].values())
            / plan["samples"],
            **telemetry[role],
        }
    report = {"fresh_metrics": rows, "status": "smoke_only", "historical_score_equality": False}
    if not plan["published_profile"]:
        return report
    family = next(row for row in provenance["families"] if row["family"] == plan["profile"])
    archived = next(row for row in family["methods"] if row["role"] == "archived_public")
    candidate = next(entry["method"] for entry in plan["commands"] if entry["role"] == "candidate")
    baselines = [entry["method"] for entry in plan["commands"] if entry["role"] != "candidate"]
    from tools.densevideo.check_grt_quality_gate import select_candidate

    try:
        gate = select_candidate(
            {**rows, archived["method"]: archived},
            baselines=baselines + [archived["method"]],
            candidates=[candidate],
            expected_samples=634,
            metric="open_mos",
            secondary_metric="token_f1",
            max_recompute_ratio=0.98,
            max_reference_recompute_ratio=0.98,
        )
        report.update(status="passed", gate=gate)
    except ValueError as error:
        report.update(status="failed", reason=str(error))
    published = next(row for row in family["methods"] if row["role"] == "candidate")
    report["candidate_delta_from_published"] = {
        key: rows[candidate][key] - published[key] for key in ("open_mos", "token_f1")
    }
    report["historical_score_equality"] = all(
        abs(delta) <= 1e-12 for delta in report["candidate_delta_from_published"].values()
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(profiles()["profiles"]), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("release/2026-08-20"),
        help="Published identity/numeric bundle; required for execution",
    )
    parser.add_argument("--limit", type=int, help="Smoke only; omitted means all 634 examples")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--with-mos", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    plan = build_plan(args.profile, output, args.limit)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    from tools.densevideo.rebuild_published_leaderboard import verify_bundle

    verify_bundle(args.bundle)
    identity_bytes = (args.bundle / "sample_identities.csv").read_bytes()
    require(
        hashlib.sha256(identity_bytes).hexdigest() == IDENTITY_SHA256,
        "Published identity bundle checksum changed",
    )
    identities = {
        int(row["doc_id"]): row["identity_sha256"]
        for row in csv.DictReader(identity_bytes.decode().splitlines())
    }
    provenance = json.loads((args.bundle / "provenance.json").read_text(encoding="utf-8"))
    # Refuse reuse of an output directory from another allocation/partial run.
    output.mkdir(parents=True, exist_ok=False)
    (output / "reproduction_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    env = dict(
        os.environ,
        PYTHONHASHSEED="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        LMMS_EVAL_PLUGINS="densevideo_qwen_dual_plugin",
        DENSEVIDEO_ENABLE_GPT_EVAL="0",
        DENSEVIDEO_FAST_TEXT_METRICS="0",
    )
    summary = output / "summary.csv"
    from tools.densevideo.run_open_leaderboard import validate_run_output

    samples, logs, telemetry = {}, {}, {}
    for entry in plan["commands"]:
        log = output / (entry["role"] + ".log")
        print(shlex.join(entry["command"]), flush=True)
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(
                entry["command"], env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
            )
        checked = validate_run_output(
            Path(entry["output"]), task=plan["task"], limit=str(plan["samples"]), run_log=log
        )
        if not checked.usable or checked.sample_count != plan["samples"]:
            raise RuntimeError(f"Incomplete {entry['role']} output: {checked.reason}")
        require(len(checked.sample_paths) == 1, "Expected one sample file per serial arm")
        validate_resolved_result(checked.result_path, entry, plan)
        role = entry["role"]
        samples[role] = load_run_samples(checked.sample_paths[0], plan["samples"], identities)
        logs[role] = log
        telemetry[role] = validate_telemetry(log, plan["samples"], control=role != "candidate")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.densevideo.collect_run_metrics",
                "--run-name",
                entry["method"],
                "--method",
                entry["method"],
                "--input-fps",
                "",
                "--task-name",
                plan["task"],
                "--run-log",
                str(log),
                "--run-output-dir",
                entry["output"],
                "--summary-csv",
                str(summary),
            ],
            check=True,
            env=env,
        )
    report = {
        "status": "inference_validated",
        "historical_score_equality": "not_checked",
        "integrity": validate_controls(plan, samples, logs),
        "telemetry": telemetry,
        "quality": {"status": "not_evaluated", "reason": "Open MOS not requested"},
    }
    report_path = output / "reproduction_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.with_mos:
        judge = profiles()["judge"]
        command = [
            sys.executable,
            "-m",
            "tools.densevideo.score_open_mos_matrix",
            "--input-root",
            str(output),
            "--output-dir",
            str(output / "mos"),
            "--expected-samples",
            str(plan["samples"]),
            "--judge-model",
            judge["model"],
            "--judge-revision",
            judge["revision"],
            "--batch-size",
            str(judge["batch_size"]),
            "--dtype",
            judge["dtype"],
            "--max-new-tokens",
            str(judge["max_new_tokens"]),
            "--trim-char-limit",
            str(judge["trim_char_limit"]),
        ]
        for entry in plan["commands"]:
            command.extend(["--method", entry["method"]])
        subprocess.run(command, check=True, env=env)
        report["quality"] = evaluate_fresh_quality(
            plan, samples, telemetry, output / "mos/matrix.jsonl", provenance
        )
        report["historical_score_equality"] = report["quality"]["historical_score_equality"]
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if report["quality"]["status"] == "failed":
            print(f"Fresh quality gate FAILED; inspect {report_path}", file=sys.stderr)
            return 1
    print(
        f"Validated {plan['samples']} samples per arm; quality={report['quality']['status']}; "
        f"outputs: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
