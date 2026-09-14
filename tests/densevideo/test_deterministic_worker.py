"""Actual retained-harness exception handling and single-process boundaries."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from lmms_eval import __main__ as cli
from tools.densevideo import deterministic_worker as worker


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    for name in (*worker._PROCESS_SIZE_ENV, *worker._PROCESS_RANK_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = mock.MagicMock()
    runtime.__version__ = "mock-torch"
    runtime.version.cuda = "mock-cuda"
    runtime.cuda.is_available.return_value = True
    runtime.cuda.device_count.return_value = 1
    runtime.cuda.get_device_name.return_value = "mock-gpu"
    runtime.cuda.get_device_properties.return_value.uuid = "mock-uuid"
    runtime.distributed.is_available.return_value = True
    runtime.distributed.is_initialized.return_value = False
    runtime.distributed.get_world_size.return_value = 1
    runtime.are_deterministic_algorithms_enabled.return_value = True
    runtime.is_deterministic_algorithms_warn_only_enabled.return_value = False
    monkeypatch.setitem(sys.modules, "torch", runtime)
    accelerator = mock.MagicMock()
    accelerator.is_main_process = True
    monkeypatch.setattr(cli, "Accelerator", mock.Mock(return_value=accelerator))
    monkeypatch.setattr(cli, "eval_logger", mock.Mock())
    monkeypatch.setattr(worker, "version", lambda _: "mock-package")
    output = tmp_path / "unused-output"
    argv = ["worker", "--", "--model", "llava_hf", "--tasks", "densevideo", "--output_path", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate = mock.Mock(return_value=(None, None))
    monkeypatch.setattr(cli, "cli_evaluate_single", evaluate)
    return SimpleNamespace(torch=runtime, output=output, argv=argv, evaluate=evaluate)


@pytest.mark.parametrize("name", worker._PROCESS_SIZE_ENV)
@pytest.mark.parametrize("value", ["2", "0", "invalid"])
def test_multiprocess_sizes_fail_before_cuda_or_evaluation(boundary, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="single process"):
        worker.main()
    boundary.evaluate.assert_not_called()
    boundary.torch.cuda.is_available.assert_not_called()
    assert not boundary.output.exists()


@pytest.mark.parametrize("name", worker._PROCESS_RANK_ENV)
def test_nonzero_rank_fails(boundary, monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match="single process"):
        worker.main()
    boundary.evaluate.assert_not_called()


def test_initialized_multiprocess_group_fails(boundary):
    boundary.torch.distributed.is_initialized.return_value = True
    boundary.torch.distributed.get_world_size.return_value = 2
    with pytest.raises(RuntimeError, match="single process"):
        worker.main()
    boundary.evaluate.assert_not_called()


@pytest.mark.parametrize("verbosity", [None, "INFO", "WARNING", "DEBUG"])
def test_real_harness_error_branch_propagates_failure(boundary, verbosity):
    if verbosity is not None:
        boundary.argv.extend(["--verbosity", verbosity])
    boundary.evaluate.side_effect = RuntimeError("simulated inference failure")
    with pytest.raises(RuntimeError, match="simulated inference failure"):
        worker.main()
    boundary.evaluate.assert_called_once()
    assert boundary.evaluate.call_args.args[0].verbosity == "DEBUG"
    assert not boundary.output.exists()


def test_config_cannot_override_strict_harness_mode(boundary):
    boundary.argv.extend(["--config", "not-read.yaml"])
    with pytest.raises(RuntimeError, match="explicit profile flags"):
        worker.main()
    boundary.evaluate.assert_not_called()


def test_single_process_preserves_profile_generation_and_seed_flags(boundary, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("LOCAL_RANK", "-1")
    boundary.argv.extend(["--seed", "0,1234,1234,1234", "--gen_kwargs", "max_new_tokens=48,temperature=0"])
    worker.main()
    args = boundary.evaluate.call_args.args[0]
    assert args.model == "llava_hf"
    assert args.tasks == "densevideo"
    assert args.output_path == str(boundary.output)
    assert args.seed == [0, 1234, 1234, 1234]
    assert args.gen_kwargs == "max_new_tokens=48,temperature=0"
    assert args.verbosity == "DEBUG"
    assert not boundary.output.exists()
