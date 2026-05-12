# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline smoke test for the LTX-2.3 T2AV GRPO rollout adapter.

Running ``python -m ltx2_3_t2av.test_rollout`` imports the standalone
``ltx2_3_t2av`` package and registers
``LTX23T2AVPipelineWithLogProb`` as the implementation for the built-in
``LTX23Pipeline`` architecture, runs one small offline generation through
``Omni``, and prints the shapes / types of the collected rollout artifacts.

Example::

    python -m ltx2_3_t2av.test_rollout \
        --model dg845/LTX-2.3-Diffusers \
        --height 512 --width 768 --num-frames 33 --num-inference-steps 4 \
        --output dev/grpo_dev/ltx2_3_t2av/output/rollout_smoke.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Import triggers DiffusionModelRegistry replacement for LTX23Pipeline.
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal LTX-2.3 T2AV rollout adapter smoke test.")
    parser.add_argument("--model", default="dg845/LTX-2.3-Diffusers", help="Diffusers model ID or local path.")
    parser.add_argument("--prompt", default="A cinematic close-up of ocean waves at golden hour.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-window-size", type=int, default=None)
    parser.add_argument("--sde-window-start", type=int, default=0)
    parser.add_argument("--sde-window-end", type=int, default=2)
    parser.add_argument("--sde-type", choices=("sde", "cps"), default="sde")
    parser.add_argument("--no-logprobs", action="store_true", help="Disable rollout log-prob collection.")
    parser.add_argument("--no-save-video", action="store_true", help="Only inspect rollout outputs; do not save mp4.")
    parser.add_argument("--output", default="dev/grpo_dev/ltx2_3_t2av/output/rollout_smoke.mp4")
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1)
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [len(value)]
    return type(value).__name__


def _first_output(result: Any) -> OmniRequestOutput:
    if isinstance(result, list):
        if not result:
            raise RuntimeError("Omni.generate returned an empty list.")
        result = result[0]
    if not isinstance(result, OmniRequestOutput):
        raise TypeError(f"Expected OmniRequestOutput, got {type(result)!r}")
    if result.is_pipeline_output and isinstance(result.request_output, OmniRequestOutput):
        result = result.request_output
    return result


def _extract_video_audio(output: OmniRequestOutput) -> tuple[Any, Any, int | None]:
    video = None
    audio = None
    audio_sample_rate = None

    if output.multimodal_output:
        audio = output.multimodal_output.get("audio")
        audio_sample_rate = output.multimodal_output.get("audio_sample_rate")

    if output.images:
        item = output.images[0]
        if isinstance(item, tuple) and len(item) == 2:
            video, audio = item
        elif isinstance(item, dict):
            video = item.get("frames") or item.get("video")
            audio = item.get("audio", audio)
            audio_sample_rate = item.get("audio_sample_rate", audio_sample_rate)
        else:
            video = item
    return video, audio, audio_sample_rate


def _save_video(output: OmniRequestOutput, output_path: str, fps: float) -> None:
    video, audio, audio_sample_rate = _extract_video_audio(output)
    if video is None:
        print("[save] no video payload found; skip saving")
        return

    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required to save video outputs.") from exc

    def normalize_frame(frame: Any) -> Any:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu()
            if frame.dim() == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.dim() == 3 and frame.shape[0] in (3, 4):
                frame = frame.permute(1, 2, 0)
            if frame.is_floating_point():
                frame = frame.clamp(-1, 1) * 0.5 + 0.5
            return frame.float().numpy()
        if isinstance(frame, np.ndarray):
            if np.issubdtype(frame.dtype, np.integer):
                return frame.astype(np.float32) / 255.0
            return frame
        return frame

    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        frames = video_tensor.float().numpy()
    elif isinstance(video, np.ndarray):
        frames = video[0] if video.ndim == 5 else video
        if np.issubdtype(frames.dtype, np.integer):
            frames = frames.astype(np.float32) / 255.0
    elif isinstance(video, list):
        frames = [normalize_frame(frame) for frame in video]
    else:
        frames = video

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if audio is not None:
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        frames_np = np.stack(frames, axis=0) if isinstance(frames, list) else np.asarray(frames)
        frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).round().clip(0, 255).astype("uint8")
        audio_np = audio[0] if isinstance(audio, list) and audio else audio
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.detach().cpu().float().numpy()
        if isinstance(audio_np, np.ndarray):
            audio_np = np.squeeze(audio_np).astype(np.float32)
        video_bytes = mux_video_audio_bytes(
            frames_u8,
            audio_np,
            fps=float(fps),
            audio_sample_rate=int(audio_sample_rate or 48000),
        )
        output_path_obj.write_bytes(video_bytes)
    else:
        export_to_video(frames, str(output_path_obj), fps=int(round(fps)))
    print(f"[save] wrote {output_path_obj}")


def main() -> None:
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
    )

    omni = Omni(
        model=args.model,
        model_class_name="LTX23Pipeline",
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
    )

    prompt: dict[str, Any] = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt["negative_prompt"] = args.negative_prompt

    extra_args: dict[str, Any] = {
        "noise_level": args.noise_level,
        "sde_type": args.sde_type,
        "logprobs": not args.no_logprobs,
        "return_trajectory_latents": True,
        "rl_rollout": {
            "enabled": True,
            "compute_log_prob": not args.no_logprobs,
            "return_trajectory_latents": True,
            "return_trajectory_log_probs": not args.no_logprobs,
        },
    }
    if args.sde_window_size is not None:
        extra_args["sde_window_size"] = args.sde_window_size
        extra_args["sde_window_range"] = [args.sde_window_start, args.sde_window_end]

    result = omni.generate(
        prompt,
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            guidance_scale_provided=True,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            seed=args.seed,
            extra_args=extra_args,
        ),
    )
    output = _first_output(result)
    custom = output.custom_output or {}

    summary = {
        "final_output_type": output.final_output_type,
        "has_audio": bool(output.multimodal_output and output.multimodal_output.get("audio") is not None),
        "audio_sample_rate": (output.multimodal_output or {}).get("audio_sample_rate"),
        "custom_keys": sorted(custom.keys()),
        "all_latents": _shape(custom.get("all_latents")),
        "all_log_probs": _shape(custom.get("all_log_probs")),
        "all_timesteps": _shape(custom.get("all_timesteps")),
        "timesteps": _shape(custom.get("timesteps")),
        "latent_index_map": _shape(custom.get("latent_index_map")),
        "log_prob_index_map": _shape(custom.get("log_prob_index_map")),
        "prompt_ids": _shape(custom.get("prompt_ids")),
        "connector_prompt_embeds": _shape(custom.get("connector_prompt_embeds")),
        "connector_audio_prompt_embeds": _shape(custom.get("connector_audio_prompt_embeds")),
        "connector_attention_mask": _shape(custom.get("connector_attention_mask")),
        "negative_connector_prompt_embeds": _shape(custom.get("negative_connector_prompt_embeds")),
        "negative_connector_audio_prompt_embeds": _shape(custom.get("negative_connector_audio_prompt_embeds")),
        "negative_connector_attention_mask": _shape(custom.get("negative_connector_attention_mask")),
        "metadata": custom.get("metadata"),
        "rl_rollout": custom.get("rl_rollout"),
    }
    print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))

    required = ["all_latents", "timesteps", "latent_index_map", "prompt_ids", "connector_prompt_embeds", "metadata"]
    missing = [key for key in required if custom.get(key) is None]
    if missing:
        raise RuntimeError(f"Missing required rollout fields: {missing}")
    if not args.no_logprobs and custom.get("all_log_probs") is None:
        raise RuntimeError("Missing all_log_probs while logprobs are enabled.")

    if not args.no_save_video:
        _save_video(output, args.output, fps=args.frame_rate)


if __name__ == "__main__":
    main()
