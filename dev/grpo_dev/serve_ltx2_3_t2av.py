#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Online serving entrypoint for the standalone LTX2.3 T2AV GRPO adapter.

This file intentionally exists as a real Python script instead of being
executed through ``python -`` / heredoc. vLLM-Omni starts worker processes with
Python multiprocessing spawn, and spawn needs to reload the parent main module
from an actual file path. If the parent was launched from stdin, child processes
try to reload ``<stdin>`` and fail with FileNotFoundError.
"""

from __future__ import annotations

import os
import sys

# Import the standalone module before vLLM-Omni constructs the diffusion model.
# The package __init__ registers LTX23Pipeline -> LTX23T2AVPipelineWithLogProb.
import ltx2_3_t2av  # noqa: F401


def _env_arg(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    model = _env_arg("LTX23_T2AV_MODEL", "dg845/LTX-2.3-Diffusers")
    host = _env_arg("LTX23_T2AV_HOST", "0.0.0.0")
    port = _env_arg("LTX23_T2AV_PORT", "8098")
    data_parallel_size = _env_arg("LTX23_T2AV_DP", "8")

    # Keep the public model-class-name as LTX23Pipeline. The registry entry has
    # already been replaced by importing ltx2_3_t2av above.
    sys.argv = [
        "vllm",
        "serve",
        model,
        "--omni",
        "--host",
        host,
        "--port",
        port,
        "--model-class-name",
        "LTX23Pipeline",
        "--data-parallel-size",
        data_parallel_size,
    ]

    from vllm_omni.entrypoints.cli.main import main as vllm_omni_main

    vllm_omni_main()


if __name__ == "__main__":
    main()
