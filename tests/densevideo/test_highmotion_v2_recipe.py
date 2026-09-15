"""Execute documentation preparation with synthetic bytes and a mocked HF API."""

import ast
import hashlib
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOC = (ROOT / "docs/HIGHMOTION_V2_REPRODUCTION.md").read_text(encoding="utf-8")
PREPARATION = re.search(r"python - <<'PY'\n(.*?)\nPY\n", DOC, re.DOTALL).group(1)
GENERATION = re.search(r"```bash\n(CUDA_VISIBLE_DEVICES=0.*?)```", DOC, re.DOTALL).group(1)


def constants():
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(PREPARATION).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("ANNOTATION_SHA256", "MODEL_REVISION", "MODEL_FILE_SHA256")
    }


PINS = constants()


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    annotation = tmp_path / "original.parquet"
    annotation.write_bytes(b"Synthetic original annotation, not a benchmark.\n")
    hub = tmp_path / "authorized-model-hub"
    snapshot = hub / "models--synthetic" / "snapshots" / PINS["MODEL_REVISION"]
    snapshot.mkdir(parents=True)
    cache = tmp_path / "fresh-task-cache"
    cache.mkdir(mode=0o700)
    source = PREPARATION.replace(
        PINS["ANNOTATION_SHA256"], hashlib.sha256(annotation.read_bytes()).hexdigest(),
    )
    for name, expected in PINS["MODEL_FILE_SHA256"].items():
        content = ("Synthetic model file: " + name).encode("utf-8")
        (snapshot / name).write_bytes(content)
        source = source.replace(expected, hashlib.sha256(content).hexdigest())
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("HM_ORIGINAL_PARQUET", str(annotation))
    monkeypatch.setenv("HM_MODEL_HUB", str(hub))
    monkeypatch.setenv("HM_RUN_CACHE", str(cache))
    return {"annotation": annotation, "hub": hub, "snapshot": snapshot, "cache": cache,
            "calls": calls, "source": source}


def execute(preparation):
    # Execute this repository's reviewed recipe with synthetic data and a mocked Hub.
    exec(compile(preparation["source"], "documented-cache-preparation", "exec"), {})  # noqa: S102


def test_original_annotation_and_model_identity_pins_are_explicit():
    assert PINS["ANNOTATION_SHA256"] == "518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd"
    assert PINS["MODEL_REVISION"] == "74dd0bf867a4cda7950c17663794267c60cf4b40"
    assert PINS["MODEL_FILE_SHA256"]["model.safetensors"] == "07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e"
    assert set(PINS["MODEL_FILE_SHA256"]) == {
        "model.safetensors", "added_tokens.json", "chat_template.json", "config.json",
        "generation_config.json", "merges.txt", "preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        "video_preprocessor_config.json", "vocab.json",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in PINS["MODEL_FILE_SHA256"].values())


def test_guide_distinguishes_current_results_from_frozen_pre_outcome_protocol():
    assert "frozen pre-outcome protocol specification" in DOC
    assert "[High-Motion v2 results](HIGHMOTION_V2_RESULTS.md)" in DOC
    assert "candidate completed its 1,000-record generation" in DOC
    assert "not a guarantee of\nidentical predictions" in DOC
    assert "They do not establish a GRT win or lift the current High-Motion leaderboard hold." not in DOC


def test_preparation_creates_exact_local_annotation_binding_and_isolated_caches(preparation):
    execute(preparation)
    local = preparation["cache"] / "hfhome/highmotion_densevideounderstand/Egodex_traj.parquet"
    assert local.is_symlink()
    assert local.resolve() == preparation["annotation"]
    assert local.read_bytes() == preparation["annotation"].read_bytes()
    for name in ("datasets", "modules", "assets", "xet", "torch", "xdg", "cuda", "tmp"):
        assert (preparation["cache"] / name).is_dir()
    assert preparation["calls"] == [(
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        {"revision": PINS["MODEL_REVISION"], "cache_dir": str(preparation["hub"]),
         "allow_patterns": list(PINS["MODEL_FILE_SHA256"])},
    )]


def test_wrong_annotation_stops_before_model_download_or_cache_binding(preparation):
    preparation["annotation"].write_bytes(b"Wrong synthetic annotation")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        execute(preparation)
    assert preparation["calls"] == []
    assert list(preparation["cache"].iterdir()) == []


@pytest.mark.parametrize("name", sorted(PINS["MODEL_FILE_SHA256"]))
def test_wrong_model_component_stops_before_binding(preparation, name):
    (preparation["snapshot"] / name).write_bytes(b"Wrong synthetic model bytes")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        execute(preparation)
    assert list(preparation["cache"].iterdir()) == []


def test_wrong_snapshot_revision_is_rejected(preparation, monkeypatch):
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download",
                        lambda *args, **kwargs: str(preparation["snapshot"].parent / "wrong-revision"))
    with pytest.raises(RuntimeError, match="Unexpected model snapshot revision"):
        execute(preparation)
    assert list(preparation["cache"].iterdir()) == []


def test_preparation_does_not_reuse_or_overwrite_an_existing_task_cache(preparation):
    execute(preparation)
    local = preparation["cache"] / "hfhome/highmotion_densevideounderstand/Egodex_traj.parquet"
    before = local.lstat()
    with pytest.raises(FileExistsError):
        execute(preparation)
    assert local.lstat().st_ino == before.st_ino
    assert local.resolve() == preparation["annotation"]


def test_generation_uses_prepared_cache_and_offline_environment_without_changing_protocol():
    words = shlex.split(GENERATION.replace("\\\n", " "))
    environment = dict(word.split("=", 1) for word in words[:words.index("python")])
    assert environment["HF_HOME"] == "${HM_RUN_CACHE}/hfhome"
    assert environment["HF_HUB_CACHE"] == environment["HUGGINGFACE_HUB_CACHE"] == "${HM_MODEL_HUB}"
    for variable, directory in (
        ("HF_DATASETS_CACHE", "datasets"), ("HF_MODULES_CACHE", "modules"),
        ("HF_ASSETS_CACHE", "assets"), ("HF_XET_CACHE", "xet"), ("TORCH_HOME", "torch"),
        ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda"), ("TMPDIR", "tmp"),
    ):
        assert environment[variable] == "${HM_RUN_CACHE}/" + directory
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE",
                     "DENSEVIDEO_DISABLE_PREPARED_ARROW_CACHE",
                     "DENSEVIDEO_SKIP_VIDEO_SNAPSHOT_IF_CACHE_EXISTS"):
        assert environment[variable] == "1"
    assert environment["DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES"] == "1000"
    assert environment["DENSEVIDEO_HIGHMOTION_NUM_FRAMES"] == "8"
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert environment[name] == "8"
    cli = words[words.index("--") + 1:]
    assert cli[cli.index("--tasks") + 1] == "densevideo_highmotion"
    assert cli[cli.index("--seed") + 1] == "0,1234,1234,1234"
    assert cli[cli.index("--gen_kwargs") + 1] == "max_new_tokens=64,temperature=0"
    assert cli[cli.index("--batch_size") + 1] == "1"
    model = dict(item.split("=", 1) for item in cli[cli.index("--model_args") + 1].split(","))
    assert model["revision"] == PINS["MODEL_REVISION"]
    assert model["dtype"] == "bfloat16"
    assert model["max_frames_num"] == "8" and model["max_image_size"] == "384"
    assert model["attn_implementation"] == "eager" and model["video_decode_backend"] == "pyav_seek"
    assert model["prompt_router"] == "off" and model["gate_policy"] == "motion"
    assert model["gate_metric"] == "ssim" and model["gate_diff_threshold"] == "0.001"
    assert model["gate_projection_mode"] == "linear_consistent"
    assert model["gate_refresh_interval_frames"] == "0"


def test_bash_examples_parse_and_require_fresh_task_specific_directory():
    blocks = re.findall(r"```bash\n(.*?)```", DOC, re.DOTALL)
    assert blocks
    for block in blocks:
        subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
    assert 'HM_RUN_CACHE=$(mktemp -d "${TMPDIR:-/tmp}/dive-hm-v2.XXXXXXXX")' in DOC
    assert "HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python" in DOC
    assert "Do not proceed after a failed check." in DOC
    assert "same shell" in DOC
