# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from verl_omni/pipelines/schedulers/flow_match_sde.py

"""SDE version of ``FlowMatchEulerDiscreteScheduler`` with log-prob support.

This is the scheduler used by Flow-Factory / FlowGRPO to collect the old
policy ``log_prob`` used by GRPO's importance-sampling ratio. Implementation
follows the FlowGRPO paper (https://arxiv.org/abs/2505.05470) and the
``diffusers`` v0.37 branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchSDEDiscreteSchedulerOutput(BaseOutput):
    """Output of :meth:`FlowMatchSDEDiscreteScheduler.step`.

    Attributes:
        prev_sample: Previous-timestep sample ``x_{t-1}``.
        log_prob: Per-sample log-probability of ``prev_sample`` under the
            current reverse-SDE transition kernel.  ``None`` when
            ``return_logprobs=False``.
        prev_sample_mean: Mean of the reverse-SDE transition kernel.
        std_dev_t: Standard deviation of the reverse-SDE transition kernel.
    """

    prev_sample: torch.FloatTensor
    log_prob: Optional[torch.FloatTensor]
    prev_sample_mean: torch.FloatTensor
    std_dev_t: torch.FloatTensor


class FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """SDE variant of ``FlowMatchEulerDiscreteScheduler`` for GRPO rollouts.

    Compared to the ODE parent class, :meth:`step` optionally injects noise
    and computes ``log p(x_{t-1} | x_t)`` so that downstream GRPO training
    can reuse the rollout sample as an old-policy draw.
    """

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None,
        sde_type: Literal["sde", "cps"] = "sde",
        return_logprobs: bool = True,
    ):
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of the `scheduler.timesteps` values instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast for numerical stability when computing prev_sample.
        sample = sample.to(torch.float32)
        if prev_sample is not None:
            prev_sample = prev_sample.to(torch.float32)

        prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sample_previous_step(
            sample=sample,
            model_output=model_output,
            generator=generator,
            per_token_timesteps=per_token_timesteps,
            noise_level=noise_level,
            prev_sample=prev_sample,
            sde_type=sde_type,
            return_logprobs=return_logprobs,
        )

        self._step_index += 1
        if per_token_timesteps is None:
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample, log_prob, prev_sample_mean, std_dev_t)

        return FlowMatchSDEDiscreteSchedulerOutput(
            prev_sample=prev_sample,
            log_prob=log_prob,
            prev_sample_mean=prev_sample_mean,
            std_dev_t=std_dev_t,
        )

    def sample_previous_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.Tensor] = None,
        sde_type: Literal["cps", "sde"] = "sde",
        return_logprobs: bool = True,
    ):
        assert sde_type in ("sde", "cps")
        assert sample.dtype == torch.float32
        if prev_sample is not None:
            assert prev_sample.dtype == torch.float32

        if per_token_timesteps is not None:
            raise NotImplementedError(
                "per_token_timesteps is not supported by FlowMatchSDEDiscreteScheduler."
            )

        if timestep is None:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_prev = self.sigmas[sigma_idx + 1]
        else:
            sigma_idx = torch.tensor([self.index_for_timestep(t) for t in timestep])
            sigma = self.sigmas[sigma_idx].view(-1, *([1] * (len(sample.shape) - 1)))
            sigma_prev = self.sigmas[sigma_idx + 1].view(-1, *([1] * (len(sample.shape) - 1)))

        sigma_max = self.sigmas[1]
        dt = sigma_prev - sigma

        if sde_type == "sde":
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )
            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
            if return_logprobs:
                log_prob = (
                    -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                    - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
            else:
                log_prob = None
        else:  # cps
            std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)
            pred_original_sample = sample - sigma * model_output
            noise_estimate = sample + model_output * (1 - sigma)
            prev_sample_mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
                sigma_prev**2 - std_dev_t**2
            )
            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise
            if return_logprobs:
                log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
            else:
                log_prob = None

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim))) if log_prob is not None else None
        return prev_sample, log_prob, prev_sample_mean, std_dev_t
