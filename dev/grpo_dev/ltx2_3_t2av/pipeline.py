# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import os
from typing import Any, Literal

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import calculate_shift
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
    LTX23Pipeline,
    get_ltx2_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .schedulers import FlowMatchSDEDiscreteScheduler

__all__ = ["LTX23T2AVPipelineWithLogProb", "get_ltx2_post_process_func"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _maybe_to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_maybe_to_cpu(v) for v in value)
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


def _mean_log_prob(video_log_prob: torch.Tensor | None, audio_log_prob: torch.Tensor | None) -> torch.Tensor | None:
    if video_log_prob is None:
        return audio_log_prob
    if audio_log_prob is None:
        return video_log_prob
    return 0.5 * (video_log_prob + audio_log_prob)


class LTX23T2AVPipelineWithLogProb(LTX23Pipeline):
    """LTX-2.3 text-to-audio-video rollout pipeline with GRPO artifacts.

    This adapter keeps the normal vLLM-Omni LTX-2.3 generation behavior, but
    swaps the video/audio denoising schedulers to SDE-capable schedulers during
    rollout and exports Flow-Factory-compatible intermediate fields through
    :class:`~vllm_omni.diffusion.data.DiffusionOutput.custom_output`.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    @staticmethod
    def _get_prompt_field(prompt: Any, key: str, default: Any = None) -> Any:
        if isinstance(prompt, dict):
            return prompt.get(key, default)
        return default

    def _get_prompt_ids(
        self,
        prompt: list[str],
        max_sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.tokenizer is None:
            return None
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        text_inputs = self.tokenizer(
            [p.strip() for p in prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids.to(device)

    @staticmethod
    def _parse_sde_window(
        num_timesteps: int,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] | list[int] | None = None,
    ) -> tuple[int, int]:
        if sde_window_size is None:
            return 0, max(num_timesteps - 1, 0)

        if sde_window_range is None:
            start_min, start_max = 0, max(num_timesteps - sde_window_size, 0)
        else:
            start_min, start_max = int(sde_window_range[0]), int(sde_window_range[1])
            start_max = min(start_max, max(num_timesteps - sde_window_size, 0))

        start_min = max(start_min, 0)
        start_max = max(start_min, start_max)
        if start_min == start_max:
            start = start_min
        else:
            torch_generator = generator[0] if isinstance(generator, list) else generator
            start = torch.randint(start_min, start_max + 1, (1,), generator=torch_generator, device=device).item()
        return start, min(start + sde_window_size, max(num_timesteps - 1, 0))

    @staticmethod
    def _build_index_map(num_inference_steps: int, start: int, end: int, include_terminal: bool) -> torch.Tensor:
        index_map = torch.full((num_inference_steps + 1,), -1, dtype=torch.long)
        for original_idx in range(start, end + 1):
            index_map[original_idx] = original_idx - start
        if include_terminal and end + 1 <= num_inference_steps:
            index_map[end + 1] = end + 1 - start
        return index_map

    def _split_connector_outputs(
        self,
        connector_prompt_embeds: torch.Tensor,
        connector_audio_prompt_embeds: torch.Tensor,
        connector_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.do_classifier_free_guidance:
            negative_connector_prompt_embeds, connector_prompt_embeds = connector_prompt_embeds.chunk(2)
            negative_connector_audio_prompt_embeds, connector_audio_prompt_embeds = connector_audio_prompt_embeds.chunk(2)
            negative_connector_attention_mask, connector_attention_mask = connector_attention_mask.chunk(2)
            return (
                connector_prompt_embeds,
                connector_audio_prompt_embeds,
                connector_attention_mask,
                negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask,
            )
        return connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask, None, None, None

    def diffuse_with_log_prob(
        self,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        connector_prompt_embeds: torch.Tensor,
        connector_audio_prompt_embeds: torch.Tensor,
        connector_attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
        audio_scheduler: FlowMatchSDEDiscreteScheduler,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        frame_rate: float,
        padded_audio_num_frames: int,
        video_coords: torch.Tensor,
        audio_coords: torch.Tensor,
        guidance_scale: float,
        generator: torch.Generator | list[torch.Generator] | None,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: Literal["sde", "cps"],
        logprobs: bool,
        collect_latents: bool,
        attention_kwargs: dict[str, Any] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Run LTX-2.3 denoising and collect unified video+audio trajectories."""
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor | None] = []
        all_timesteps: list[torch.Tensor] = []
        start, end = sde_window
        video_seq_len = latents.shape[1]

        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)

        def collect_current(step_idx: int) -> None:
            # ``end`` is exclusive for log-prob collection and inclusive for
            # latent collection.  This mirrors Flow-Factory's compact
            # trajectory collector: log-prob at step ``i`` consumes latent
            # positions ``i`` and ``i + 1``.
            if collect_latents and start <= step_idx <= end:
                all_latents.append(torch.cat([latents, audio_latents], dim=1))

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                collect_current(i)

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(connector_prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(connector_prompt_embeds.dtype)
                ts = t.expand(latent_model_input.shape[0])

                with self._transformer_cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio = self.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=connector_prompt_embeds,
                        audio_encoder_hidden_states=connector_audio_prompt_embeds,
                        timestep=ts,
                        sigma=ts,
                        encoder_attention_mask=connector_attention_mask,
                        audio_encoder_attention_mask=connector_attention_mask,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=frame_rate,
                        audio_num_frames=padded_audio_num_frames,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )

                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond, noise_pred_video_cond = noise_pred_video.chunk(2)
                    x0_video_cond = latents - noise_pred_video_cond * self.scheduler.sigmas[i]
                    x0_video_uncond = latents - noise_pred_video_uncond * self.scheduler.sigmas[i]
                    x0_video_guided = x0_video_cond + (guidance_scale - 1) * (x0_video_cond - x0_video_uncond)

                    noise_pred_audio_uncond, noise_pred_audio_cond = noise_pred_audio.chunk(2)
                    x0_audio_cond = audio_latents - noise_pred_audio_cond * audio_scheduler.sigmas[i]
                    x0_audio_uncond = audio_latents - noise_pred_audio_uncond * audio_scheduler.sigmas[i]
                    x0_audio_guided = x0_audio_cond + (guidance_scale - 1) * (x0_audio_cond - x0_audio_uncond)

                    noise_pred_video = (latents - x0_video_guided) / self.scheduler.sigmas[i]
                    noise_pred_audio = (audio_latents - x0_audio_guided) / audio_scheduler.sigmas[i]

                cur_noise_level = noise_level if start <= i < end else 0.0
                return_logprobs = logprobs and start <= i < end
                torch_generator = generator[0] if isinstance(generator, list) else generator
                latents, video_log_prob, _, _ = self.scheduler.step(
                    noise_pred_video,
                    t,
                    latents,
                    generator=torch_generator,
                    noise_level=cur_noise_level,
                    sde_type=sde_type,
                    return_logprobs=return_logprobs,
                    return_dict=False,
                )
                audio_latents, audio_log_prob, _, _ = audio_scheduler.step(
                    noise_pred_audio,
                    t,
                    audio_latents,
                    generator=torch_generator,
                    noise_level=0.0,
                    sde_type=sde_type,
                    return_logprobs=False,
                    return_dict=False,
                )

                if start <= i < end:
                    all_log_probs.append(_mean_log_prob(video_log_prob, audio_log_prob) if return_logprobs else None)
                    all_timesteps.append(t)

                pbar.update()

        trajectory_latents = torch.stack(all_latents, dim=1) if all_latents else None
        trajectory_log_probs = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        trajectory_timesteps = (
            torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1) if all_timesteps else None
        )
        rollout = {
            "all_latents": trajectory_latents,
            "all_log_probs": trajectory_log_probs,
            "all_timesteps": trajectory_timesteps,
            "timesteps": timesteps.unsqueeze(0).expand(latents.shape[0], -1),
            "video_seq_len": video_seq_len,
            "latent_index_map": self._build_index_map(len(timesteps), start, end, include_terminal=False).to(
                latents.device
            ),
            "log_prob_index_map": self._build_index_map(len(timesteps), start, end - 1, include_terminal=False).to(
                latents.device
            ),
        }
        return latents, audio_latents, trajectory_log_probs, rollout

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput:
        # ---- Extract from request ----
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = req.sampling_params.height or height or 512
        width = req.sampling_params.width or width or 768
        num_frames = req.sampling_params.num_frames or num_frames or 121
        frame_rate = req.sampling_params.resolved_frame_rate or frame_rate or 24.0
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps or 40
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchSDEDiscreteScheduler.")
        num_videos_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        latents = req.sampling_params.latents if req.sampling_params.latents is not None else latents
        audio_latents = (
            req.sampling_params.audio_latents
            if req.sampling_params.audio_latents is not None
            else req.sampling_params.extra_args.get("audio_latents", audio_latents)
        )

        req_prompt_embeds = [self._get_prompt_field(p, "prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_prompt_embeds):
            prompt_embeds = torch.stack(req_prompt_embeds)

        req_negative_prompt_embeds = [self._get_prompt_field(p, "negative_prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_negative_prompt_embeds):
            negative_prompt_embeds = torch.stack(req_negative_prompt_embeds)

        req_prompt_attention_masks = [
            self._get_prompt_field(p, "prompt_attention_mask") or self._get_prompt_field(p, "attention_mask")
            for p in req.prompts
        ]
        if any(m is not None for m in req_prompt_attention_masks):
            prompt_attention_mask = torch.stack(req_prompt_attention_masks)

        req_negative_attention_masks = [
            self._get_prompt_field(p, "negative_prompt_attention_mask")
            or self._get_prompt_field(p, "negative_attention_mask")
            for p in req.prompts
        ]
        if any(m is not None for m in req_negative_attention_masks):
            negative_prompt_attention_mask = torch.stack(req_negative_attention_masks)

        if req.sampling_params.decode_timestep is not None:
            decode_timestep = req.sampling_params.decode_timestep
        if req.sampling_params.decode_noise_scale is not None:
            decode_noise_scale = req.sampling_params.decode_noise_scale
        if req.sampling_params.output_type is not None:
            output_type = req.sampling_params.output_type

        extra_args = req.sampling_params.extra_args or {}
        noise_level = _coalesce_not_none(extra_args.get("noise_level"), 0.7)
        sde_window_size = extra_args.get("sde_window_size")
        sde_window_range = extra_args.get("sde_window_range")
        sde_type = _coalesce_not_none(extra_args.get("sde_type"), "sde")
        logprobs = _coalesce_not_none(extra_args.get("logprobs"), True)
        collect_latents = _coalesce_not_none(extra_args.get("return_trajectory_latents"), True)

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt_list = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            prompt_list = prompt
        else:
            batch_size = prompt_embeds.shape[0]
            prompt_list = ["" for _ in range(batch_size)]

        device = self.device
        prompt_ids = None
        req_prompt_ids = [self._get_prompt_field(p, "prompt_ids") for p in req.prompts]
        if any(p is not None for p in req_prompt_ids):
            prompt_ids = torch.stack(
                [p if isinstance(p, torch.Tensor) else torch.as_tensor(p, device=device) for p in req_prompt_ids]
            ).to(device)
        elif prompt is not None:
            prompt_ids = self._get_prompt_ids(prompt_list, max_sequence_length, device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        self.connectors.to(device)
        tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=tokenizer_padding_side
        )
        self.connectors.to("cpu")
        if torch.cuda.is_available():
            torch.accelerator.empty_cache()

        (
            positive_connector_prompt_embeds,
            positive_connector_audio_prompt_embeds,
            positive_connector_attention_mask,
            negative_connector_prompt_embeds,
            negative_connector_audio_prompt_embeds,
            negative_connector_attention_mask,
        ) = self._split_connector_outputs(
            connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask
        )

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None and latents.ndim == 5:
            _, _, latent_num_frames, latent_height, latent_width = latents.shape

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None and audio_latents.ndim == 4:
            _, _, audio_num_frames, _ = audio_latents.shape

        num_mel_bins = self.audio_vae.config.mel_bins if self.audio_vae is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = self.audio_vae.config.latent_channels if self.audio_vae is not None else 8
        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        _ = retrieve_timesteps(audio_scheduler, num_inference_steps, device, timesteps, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            latents.device,
            fps=frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            padded_audio_num_frames,
            audio_latents.device,
        )

        if self.do_classifier_free_guidance:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        sde_window = self._parse_sde_window(
            len(timesteps),
            generator=generator,
            device=device,
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
        )
        latents, audio_latents, _, rollout = self.diffuse_with_log_prob(
            latents=latents,
            audio_latents=audio_latents,
            connector_prompt_embeds=connector_prompt_embeds,
            connector_audio_prompt_embeds=connector_audio_prompt_embeds,
            connector_attention_mask=connector_attention_mask,
            timesteps=timesteps,
            audio_scheduler=audio_scheduler,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            frame_rate=frame_rate,
            padded_audio_num_frames=padded_audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            guidance_scale=guidance_scale,
            generator=generator,
            noise_level=noise_level,
            sde_window=sde_window,
            sde_type=sde_type,
            logprobs=logprobs,
            collect_latents=collect_latents,
            attention_kwargs=attention_kwargs,
        )

        self._current_timestep = None

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = self._unpad_audio_latents(audio_latents, original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            original_audio_num_frames,
            num_mel_bins=latent_mel_bins,
        )

        if output_type == "latent":
            video = latents
            audio = audio_latents
        else:
            latents = latents.to(connector_prompt_embeds.dtype)
            if not self.vae.config.timestep_conditioning:
                timestep_decode = None
            else:
                from diffusers.utils.torch_utils import randn_tensor

                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size
                timestep_decode = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale_t = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

            self.vae.to(device)
            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep_decode, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
            self.vae.to("cpu")

            self.audio_vae.to(device)
            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            self.audio_vae.to("cpu")

            self.vocoder.to(device)
            audio = self.vocoder(generated_mel_spectrograms)
            self.vocoder.to("cpu")
            torch.accelerator.empty_cache()

        audio_sample_rate = int(
            getattr(getattr(self.vocoder, "config", None), "output_sampling_rate", self.audio_sampling_rate)
        )
        metadata = {
            "prompt": prompt_list,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "video_seq_len": rollout["video_seq_len"],
            "duration_s": duration_s,
            "audio_sample_rate": audio_sample_rate,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "noise_scale": noise_scale,
            "noise_level": noise_level,
            "sde_window": sde_window,
            "sde_type": sde_type,
            "sigmas": list(map(float, sigmas)) if sigmas is not None else None,
            "max_sequence_length": max_sequence_length,
        }
        custom_output = {
            "all_latents": _maybe_to_cpu(rollout["all_latents"]),
            "all_log_probs": _maybe_to_cpu(rollout["all_log_probs"]),
            "log_probs": _maybe_to_cpu(rollout["all_log_probs"]),
            "all_timesteps": _maybe_to_cpu(rollout["all_timesteps"]),
            "timesteps": _maybe_to_cpu(rollout["timesteps"]),
            "latent_index_map": _maybe_to_cpu(rollout["latent_index_map"]),
            "log_prob_index_map": _maybe_to_cpu(rollout["log_prob_index_map"]),
            "prompt_ids": _maybe_to_cpu(prompt_ids),
            "connector_prompt_embeds": _maybe_to_cpu(positive_connector_prompt_embeds),
            "connector_audio_prompt_embeds": _maybe_to_cpu(positive_connector_audio_prompt_embeds),
            "connector_attention_mask": _maybe_to_cpu(positive_connector_attention_mask),
            "negative_connector_prompt_embeds": _maybe_to_cpu(negative_connector_prompt_embeds),
            "negative_connector_audio_prompt_embeds": _maybe_to_cpu(negative_connector_audio_prompt_embeds),
            "negative_connector_attention_mask": _maybe_to_cpu(negative_connector_attention_mask),
            "metadata": _maybe_to_cpu(metadata),
            "rl_rollout": {
                "schema_version": 1,
                "tensor_format": "in_memory",
                "sample_index": 0,
                "fields": {
                    "timesteps": "timesteps",
                    "all_latents": "all_latents",
                    "log_probs": "all_log_probs",
                    "latent_index_map": "latent_index_map",
                    "log_prob_index_map": "log_prob_index_map",
                    "prompt_ids": "prompt_ids",
                    "connector_prompt_embeds": "connector_prompt_embeds",
                    "connector_audio_prompt_embeds": "connector_audio_prompt_embeds",
                    "connector_attention_mask": "connector_attention_mask",
                    "negative_connector_prompt_embeds": "negative_connector_prompt_embeds",
                    "negative_connector_audio_prompt_embeds": "negative_connector_audio_prompt_embeds",
                    "negative_connector_attention_mask": "negative_connector_attention_mask",
                },
                "metadata": metadata,
            },
        }

        return DiffusionOutput(
            output=(video, audio),
            custom_output=custom_output,
            trajectory_latents=_maybe_to_cpu(rollout["all_latents"]),
            trajectory_timesteps=_maybe_to_cpu(rollout["all_timesteps"]),
            trajectory_log_probs=_maybe_to_cpu(rollout["all_log_probs"]),
        )
