# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal LTX-2.3 T2AV GRPO rollout adapter for vllm-omni.

This package provides a minimal adaptation of :class:`LTX23Pipeline`
(``vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3``) that collects
Flow-Factory GRPO compatible intermediate artifacts (unified
video+audio ``all_latents``, SDE ``log_probs``, connector embeddings,
``prompt_ids``, ``video_seq_len``, index maps, resolved metadata) and
exposes them through ``DiffusionOutput.custom_output`` /
``OmniRequestOutput.custom_output``.

Importing this package triggers registration of
``LTX23T2AVPipelineWithLogProb`` as a replacement for the built-in
``LTX23Pipeline`` architecture in ``DiffusionModelRegistry``.

Usage (offline)::

    import ltx2_3_t2av  # triggers registration
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model="dg845/LTX-2.3-Diffusers",
        model_class_name="LTX23Pipeline",  # registry override hits the adapter
    )
    ...

Usage (online, ``vllm serve --omni``)::

    ``vllm serve --omni`` runs in a fresh process. Ensure this package is
    imported in the serving process before the diffusion model is constructed,
    for example from your training launcher's server wrapper.
"""

from .register import register_rollout_adapter

# Auto-register on import so that any process that imports this module
# (engine worker, offline ``Omni`` entrypoint, ``vllm serve --omni``) gets
# the adapter wired into :class:`DiffusionModelRegistry`.
register_rollout_adapter()

from .pipeline import LTX23T2AVPipelineWithLogProb  # noqa: E402
from .schedulers import FlowMatchSDEDiscreteScheduler  # noqa: E402

__all__ = [
    "LTX23T2AVPipelineWithLogProb",
    "FlowMatchSDEDiscreteScheduler",
    "register_rollout_adapter",
]
