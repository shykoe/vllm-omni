# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Register the LTX-2.3 T2AV GRPO rollout adapter with vllm-omni."""

from __future__ import annotations

_REGISTERED = False


def register_rollout_adapter() -> None:
    """Replace the built-in ``LTX23Pipeline`` registry entry with the adapter.

    vLLM-Omni dispatches diffusion pipelines through
    ``vllm_omni.diffusion.registry.DiffusionModelRegistry`` by architecture
    name.  The LTX-2.3 checkpoint's ``model_index.json`` uses
    ``_class_name=LTX23Pipeline``; registering under the same architecture key
    lets both offline ``Omni`` and online ``vllm serve --omni`` instantiate this
    adapter without changing request-side model metadata.
    """

    global _REGISTERED
    if _REGISTERED:
        return

    from vllm_omni.diffusion.registry import register_diffusion_model

    register_diffusion_model(
        model_arch="LTX23Pipeline",
        module_name="ltx2_3_t2av.pipeline",
        class_name="LTX23T2AVPipelineWithLogProb",
        # Keep the same post-process function name, but resolve it from this
        # module path after the registry entry is replaced.
        post_process_func_name="get_ltx2_post_process_func",
    )
    _REGISTERED = True
