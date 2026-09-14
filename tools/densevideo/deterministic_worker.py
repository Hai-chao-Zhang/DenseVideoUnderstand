"""Run one evaluation process with the published strict CUDA settings."""

import json
import os
import random
import socket
import sys
from importlib.metadata import version

_PROCESS_SIZE_ENV = ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS")
_PROCESS_RANK_ENV = ("RANK", "LOCAL_RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID")


def require_single_process(torch_module):
    """One visible GPU per rank is not proof of a single-process launch."""
    for name in (*_PROCESS_SIZE_ENV, *_PROCESS_RANK_ENV):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise RuntimeError(f"Invalid {name}; use a single process") from error
        allowed = (1,) if name in _PROCESS_SIZE_ENV else ((-1, 0) if name == "LOCAL_RANK" else (0,))
        if value not in allowed:
            raise RuntimeError(f"{name}={value}; use a single process for serial reproduction")
    distributed = torch_module.distributed
    if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() != 1:
        raise RuntimeError("Initialized distributed group is not a single process")


def trace_floor_requests(model_class):
    """Bind floor-decision log order to document IDs without changing inference."""
    original = model_class.generate_until

    def generate_until(self, requests):
        requests = list(requests)
        print("[DIVE_REQUEST_ORDER] " + json.dumps([r.args[3] for r in requests]), flush=True)
        return original(self, requests)

    model_class.generate_until = generate_until


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Start this worker with PYTHONHASHSEED=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Start this worker with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    import numpy as np
    import torch

    require_single_process(torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU for same-device serial reproduction")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(getattr(properties, "uuid", ""))
    if not gpu_uuid:
        raise RuntimeError("CUDA runtime must expose the physical GPU UUID for serial verification")
    print(
        "[DIVE_RUNTIME] "
        + json.dumps(
            {
                "hostname": socket.gethostname(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_uuid": gpu_uuid,
                "python": sys.version,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "tf32": torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "bootstrap_seed": 0,
                "evaluation_seeds": [0, 1234, 1234, 1234],
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "packages": {
                    name: version(name)
                    for name in (
                        "transformers",
                        "accelerate",
                        "av",
                        "datasets",
                        "numpy",
                        "pillow",
                        "qwen-vl-utils",
                        "huggingface-hub",
                        "torchvision",
                    )
                },
            }
        ),
        flush=True,
    )
    if "qwen2_5_vl_dual_route_floor" in sys.argv:
        from densevideo_qwen_dual_plugin.models.qwen2_5_vl_dual_route import (
            Qwen2_5_VL_DualRouteFloor,
        )

        trace_floor_requests(Qwen2_5_VL_DualRouteFloor)
    from lmms_eval.__main__ import cli_evaluate, parse_eval_args

    if len(sys.argv) > 1 and sys.argv[1] == "--":
        sys.argv.pop(1)
    args = parse_eval_args()
    if args.config:
        raise RuntimeError("Strict reproduction uses explicit profile flags, not --config overrides")
    # The retained harness otherwise logs evaluation errors and returns normally.
    # The parent runner also validates artifacts; the worker itself must fail too.
    args.verbosity = "DEBUG"
    cli_evaluate(args=args)


if __name__ == "__main__":
    main()
