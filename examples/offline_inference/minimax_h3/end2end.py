# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 unified offline inference script.

Supports all three MiniMax-H3 tasks and their input combinations:

- ``t2va``:   text -> video + audio (FL2VA partition)
- ``fl2va``:  first-frame image + text -> video + audio (FL2VA partition)
- ``ref2va``: image + audio reference, or one or more video references,
              -> video + audio (Ref2VA partition)

One Omni instance loads one checkpoint partition. Point ``--model`` at the
``FL2VA`` directory for t2va/fl2va, or at the ``Ref2VA`` directory for ref2va.

Examples:
    # T2VA
    python end2end.py --model /path/to/MiniMax-H3/FL2VA --task t2va \
        --aspect-ratio 16:9 \
        --prompts "A quiet cinematic night scene with matching ambient sound."

    # FL2VA (first frame)
    python end2end.py --model /path/to/MiniMax-H3/FL2VA --task fl2va \
        --image-path first_frame.png --prompts "The car drives away."

    # Ref2VA (image + audio)
    python end2end.py --model /path/to/MiniMax-H3/Ref2VA --task ref2va \
        --image-path ref.png --audio-path ref.mp3 --prompts "The cat lip-syncs."

    # Ref2VA (one or more video references, comma-separated)
    python end2end.py --model /path/to/MiniMax-H3/Ref2VA --task ref2va \
        --video-path subject.mp4,background.mov --prompts "Replace the background."
"""

import argparse
import json
import os
import time

import numpy as np

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MINIMAX_H3_FPS = 24
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax-H3 offline inference (t2va / fl2va / ref2va).")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to one checkpoint partition directory (e.g. /path/to/MiniMax-H3/FL2VA "
        "for t2va/fl2va, or /path/to/MiniMax-H3/Ref2VA for ref2va).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["t2va", "fl2va", "ref2va"],
        help="Generation task. Default: auto-resolve from the checkpoint partition and "
        "the provided conditions (image -> fl2va, ref2va partition -> ref2va, else t2va).",
    )
    parser.add_argument("--prompts", nargs="+", default=None, help="Input text prompts.")

    # Condition inputs (one combination per task, validated in main).
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Condition image path. First frame for fl2va; reference image for ref2va "
        "(must be combined with --audio-path).",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Reference audio path (wav/mp3/m4a) for image+audio ref2va.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Reference video path(s) for video ref2va. Comma-separated for multiple "
        "videos. The reference soundtracks are used; --audio-path is not accepted.",
    )

    # Shape / schedule parameters.
    parser.add_argument("--height", type=int, default=None, help="Output video height (multiple of 32).")
    parser.add_argument("--width", type=int, default=None, help="Output video width (multiple of 32).")
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        default="16:9",
        choices=MINIMAX_H3_ASPECT_RATIOS,
        help="Output aspect ratio. T2VA requires a named ratio; FL2VA follows the input image. Default: 16:9.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Output duration in seconds (decimal allowed). Overrides --num-frames.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Output frame count (aligned by the pipeline). Default: 209 for t2va/fl2va, 124 for ref2va.",
    )
    parser.add_argument("--fps", type=int, default=MINIMAX_H3_FPS, help="Output fps. H3 fixes this at 24.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        choices=["lossless", "high"],
        help="MiniMax-H3 request quality. 'high' enables its native conservative Cache-DiT profile; "
        "'lossless' disables request-scoped Cache-DiT.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=0,
        help="Number of warmup generations before the measured/profiled run. Warmup outputs are "
        "discarded. On NPU the first generations pay kernel JIT/autotune and HCCL lazy-init costs, "
        "so use e.g. 3 before collecting latency or profiler data.",
    )
    parser.add_argument(
        "--num-measured",
        type=int,
        default=1,
        help="Number of measured generations after warmup. Latency is reported as median plus "
        "min/max across these runs. When --profiler-config is set, the profiler wraps one "
        "additional run after these, so profiler overhead never contaminates the reported latency.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--flow-shift", type=float, default=None, help="Video sigma shift (checkpoint default: 12).")
    parser.add_argument(
        "--audio-flow-shift", type=float, default=None, help="Audio sigma shift (checkpoint default: 3)."
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory for the generated MP4 files.")

    # Engine / parallelism parameters.
    parser.add_argument("--usp", type=int, default=1, help="Ulysses sequence-parallel degree.")
    parser.add_argument("--ring", type=int, default=1, help="Ring sequence-parallel degree.")
    parser.add_argument(
        "--text-encoder-tp-size",
        type=int,
        default=1,
        help="Shard the Qwen3-VL text encoder across the first N DiT ranks "
        "(must divide 64 attention heads and 8 KV heads).",
    )
    parser.add_argument("--vae-patch-parallel-size", type=int, default=1, help="VAE patch-parallel size.")
    parser.add_argument(
        "--vae-parallel-mode",
        type=str,
        default="tile",
        choices=["tile", "spatial_shard_height", "spatial_shard_width"],
        help="VAE patch-parallel mode. H3 supports the native 'tile' mode only.",
    )
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable model-level CPU offload (single-GPU accuracy/memory-first configuration).",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offload on DiT modules.",
    )
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8"],
        help="Enable load-time FP8 quantization for the MiniMax-H3 DiT. Text encoder and VAEs remain unquantized.",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default="none",
        choices=["none", "tea_cache", "cache_dit"],
        help="Diffusion cache backend. TeaCache is calibrated only for the FL2VA partition; alternatively use "
        "--quality high for MiniMax-H3's native Cache-DiT profile.",
    )
    parser.add_argument(
        "--cache-config",
        type=str,
        default=None,
        help='JSON object forwarded to Omni(cache_config=...), e.g. \'{"rel_l1_thresh":0.17}\' for TeaCache.',
    )
    parser.add_argument(
        "--diffusion-attention-backend",
        type=str,
        default=None,
        help="Diffusion attention backend, for example FLASH_ATTN.",
    )
    parser.add_argument("--init-timeout", type=int, default=600, help="Engine initialization timeout in seconds.")
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        default=False,
        help="Record per-stage pipeline durations (encode / denoise / decode) alongside latency.",
    )
    parser.add_argument(
        "--profiler-config",
        type=str,
        default=None,
        help=(
            "JSON object forwarded to Omni/profiler_config. Memory snapshot example: "
            '\'{"profiler":"torch","torch_profiler_dir":"./h3_prof","torch_profiler_with_memory":true,'
            '"torch_profiler_with_stack":false,"torch_profiler_record_shapes":false}\'. '
            "With torch_profiler_with_memory=true, each worker rank records NPU memory history "
            "between start/stop_profile and dumps memory_snapshot-*.pickle into the profiler dir."
        ),
    )
    return parser.parse_args()


def _split_paths(raw: str | None) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _resolve_task_and_mm_data(args) -> tuple[str | None, dict]:
    """Validate the condition inputs and build multi_modal_data for the task."""
    image_paths = _split_paths(args.image_path)
    audio_paths = _split_paths(args.audio_path)
    video_paths = _split_paths(args.video_path)
    for path in image_paths + audio_paths + video_paths:
        if not os.path.exists(path):
            raise ValueError(f"Condition file does not exist: {path}")

    task = args.task
    if task is None:
        # Mirror the pipeline's auto-resolution; the ref2va partition is
        # detected engine-side, so only disambiguate the FL2VA cases here.
        task = "fl2va" if image_paths else ("ref2va" if (audio_paths or video_paths) else "t2va")

    if len(image_paths) > 1:
        raise ValueError(f"MiniMax H3 supports exactly one condition image, got {len(image_paths)}")
    if len(audio_paths) > 1:
        raise ValueError(f"MiniMax H3 supports exactly one condition audio, got {len(audio_paths)}")

    if task == "t2va":
        if image_paths or audio_paths or video_paths:
            raise ValueError("t2va does not accept image/audio/video conditions.")
        mm_data: dict = {}
    elif task == "fl2va":
        if not image_paths:
            raise ValueError("fl2va requires --image-path (the first frame).")
        if audio_paths or video_paths:
            raise ValueError("fl2va does not accept audio or video conditions.")
        mm_data = {"image": image_paths[0]}
    else:  # ref2va
        if video_paths:
            if image_paths or audio_paths:
                raise ValueError(
                    "video ref2va uses the reference-video soundtracks; "
                    "do not combine --video-path with --image-path/--audio-path."
                )
            mm_data = {"video": video_paths}
        else:
            if not image_paths or not audio_paths:
                raise ValueError(
                    "ref2va requires either --image-path + --audio-path, or one or more --video-path entries."
                )
            mm_data = {"image": image_paths[0], "audio": audio_paths[0]}
    return args.task, mm_data


def _parse_json_object(raw: str | None, option: str) -> dict | None:
    if raw is None:
        return None
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {option} JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{option} must decode to a JSON object, got {type(config).__name__}")
    return config


def main():
    args = parse_args()
    if not args.model:
        raise ValueError(
            "--model is required. Point it at one checkpoint partition directory, "
            "e.g. /path/to/MiniMax-H3/FL2VA (t2va/fl2va) or /path/to/MiniMax-H3/Ref2VA (ref2va)."
        )
    if args.quantization == "fp8" and args.enable_layerwise_offload:
        raise ValueError("MiniMax-H3 online FP8 quantization is incompatible with --enable-layerwise-offload.")
    if args.cache_config is not None and args.cache_backend == "none":
        raise ValueError("--cache-config requires --cache-backend tea_cache or cache_dit.")
    if args.cache_backend == "tea_cache" and args.task == "ref2va":
        raise ValueError("MiniMax-H3 TeaCache supports the FL2VA partition only, not ref2va.")
    if args.cache_backend == "tea_cache" and args.quality is not None:
        raise ValueError("--quality configures Cache-DiT and cannot be combined with --cache-backend tea_cache.")
    os.makedirs(args.output, exist_ok=True)

    task, mm_data = _resolve_task_and_mm_data(args)
    prompts = args.prompts or ["A quiet cinematic night scene with matching ambient sound."]

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.usp,
        ring_degree=args.ring,
        text_encoder_tp_size=args.text_encoder_tp_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        vae_parallel_mode=args.vae_parallel_mode,
    )

    omni_kwargs = dict(
        model=args.model,
        parallel_config=parallel_config,
        trust_remote_code=True,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_tiling=args.vae_use_tiling,
        enforce_eager=args.enforce_eager,
        init_timeout=args.init_timeout,
        log_stats=args.log_stats,
        cache_backend=args.cache_backend,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )
    if args.quantization is not None:
        omni_kwargs["quantization_config"] = {
            "transformer": {"method": args.quantization},
            "text_encoder": None,
            "video_vae": None,
            "audio_vae": None,
        }
    cache_config = _parse_json_object(args.cache_config, "--cache-config")
    if args.cache_backend == "tea_cache" and cache_config is None:
        cache_config = {"rel_l1_thresh": 0.17}
    if cache_config is not None:
        omni_kwargs["cache_config"] = cache_config
    if args.diffusion_attention_backend is not None:
        omni_kwargs["diffusion_attention_backend"] = args.diffusion_attention_backend
    profiler_config = _parse_json_object(args.profiler_config, "--profiler-config")
    if profiler_config is not None:
        omni_kwargs["profiler_config"] = profiler_config

    print(f"\n{'=' * 60}")
    print("MiniMax-H3 Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Task: {task or 'auto'}")
    print(f"  Inference steps: {args.steps}")
    print(f"  Quality: {args.quality or 'default'}")
    print(f"  Seed: {args.seed}")
    print(f"  Warmup runs: {args.num_warmup}")
    print(f"  Measured runs: {args.num_measured}")
    print(f"  Output size: {args.width}x{args.height} (None = pipeline default)")
    print(f"  Aspect ratio: {args.aspect_ratio}")
    print(f"  Duration: {args.duration}s" if args.duration else f"  Num frames: {args.num_frames}")
    print(f"  Flow shift: video={args.flow_shift}, audio={args.audio_flow_shift} (None = checkpoint default)")
    print(
        f"  Parallel: usp={args.usp}, ring={args.ring}, text_encoder_tp_size={args.text_encoder_tp_size},"
        f" vae_patch_parallel_size={args.vae_patch_parallel_size} ({args.vae_parallel_mode})"
    )
    print(f"  cpu_offload={args.enable_cpu_offload}, layerwise_offload={args.enable_layerwise_offload}")
    print(f"  Quantization: {args.quantization or 'None (BF16)'}")
    print(f"  Cache: backend={args.cache_backend}, config={cache_config}")
    if mm_data:
        print(f"  Conditions: {mm_data}")
    print(f"  Prompts: {prompts}")
    print(f"{'=' * 60}\n")

    omni = Omni(**omni_kwargs)

    extra_args: dict = {"aspect_ratio": args.aspect_ratio}
    if task is not None:
        extra_args["task"] = task
    if args.duration is not None:
        extra_args["duration"] = args.duration
    if args.flow_shift is not None:
        extra_args["flow_shift"] = args.flow_shift
    if args.audio_flow_shift is not None:
        extra_args["audio_flow_shift"] = args.audio_flow_shift

    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames or 1,
        fps=args.fps,
        num_inference_steps=args.steps,
        seed=args.seed,
        output_type="np",
        quality=args.quality,
        extra_args=extra_args,
    )

    formatted_prompts: list = []
    for prompt in prompts:
        prompt_dict: dict = {"prompt": prompt}
        if mm_data:
            prompt_dict["multi_modal_data"] = mm_data
        formatted_prompts.append(prompt_dict)

    def _report_stage_durations(label: str, result_list) -> None:
        for index, result in enumerate(result_list or []):
            durations = getattr(result, "stage_durations", None) or {}
            peak_mb = getattr(result, "peak_memory_mb", None)
            if not durations and peak_mb is None:
                continue
            print(f"[Stages] {label} output {index}:", flush=True)
            # stage_durations mixes units: pipeline stage records are seconds,
            # while orchestrator-injected keys already end in _ms.
            normalized = {
                stage: (value if stage.endswith("_ms") else value * 1000.0) for stage, value in durations.items()
            }
            for stage, ms in sorted(normalized.items(), key=lambda kv: -kv[1]):
                print(f"    {stage}: {ms:.1f} ms", flush=True)
            if peak_mb is not None:
                print(f"    peak_memory_mb: {peak_mb}", flush=True)

    def _run_generate(label: str):
        t0 = time.perf_counter()
        result = omni.generate(
            prompts=formatted_prompts,
            sampling_params_list=sampling_params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - t0
        print(f"[Timing] {label}: latency={elapsed:.3f}s", flush=True)
        if args.enable_diffusion_pipeline_profiler:
            _report_stage_durations(label, result)
        return result, elapsed

    # Warmup runs (outputs discarded). NPU first-run costs (kernel JIT,
    # autotune, HCCL lazy init) are absorbed here so the measured/profiled
    # run reflects steady state.
    warmup_latencies: list[float] = []
    for i in range(args.num_warmup):
        _, dt = _run_generate(f"warmup {i + 1}/{args.num_warmup}")
        warmup_latencies.append(dt)

    # Measured runs. The profiler is deliberately NOT active here: torch
    # profiler overhead (especially with record_shapes) inflates latency, so a
    # profiled run must not be quoted as a latency number.
    num_measured = max(args.num_measured, 1)
    measured_latencies: list[float] = []
    outputs = None
    try:
        for i in range(num_measured):
            outputs, dt = _run_generate(f"measured {i + 1}/{num_measured}")
            measured_latencies.append(dt)

        if profiler_config is not None:
            print("[Profiler] Starting profiler (extra run, excluded from latency stats)...")
            omni.start_profile()
            try:
                _run_generate("profiled run (excluded from latency stats)")
            finally:
                try:
                    omni.stop_profile()
                    print(
                        f"[Profiler] Trace and memory snapshots written under "
                        f"{profiler_config.get('torch_profiler_dir')}"
                    )
                except Exception as exc:
                    print(f"[Profiler] stop_profile failed: {exc}")
    finally:
        omni.close()

    if not outputs:
        raise RuntimeError("No output returned from the model.")

    if warmup_latencies or measured_latencies:
        print(f"\n{'=' * 60}")
        print("Latency summary:")
        for i, dt in enumerate(warmup_latencies):
            print(f"  warmup {i + 1}: {dt:.3f}s (excluded)")
        for i, dt in enumerate(measured_latencies):
            print(f"  measured {i + 1}: {dt:.3f}s")
        if measured_latencies:
            ordered = sorted(measured_latencies)
            median = (
                ordered[len(ordered) // 2]
                if len(ordered) % 2
                else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
            )
            spread = ordered[-1] - ordered[0]
            rel = (spread / median * 100.0) if median else 0.0
            print(f"  median: {median:.3f}s  min: {ordered[0]:.3f}s  max: {ordered[-1]:.3f}s")
            print(f"  spread: {spread:.3f}s ({rel:.2f}% of median) over n={len(ordered)}")
        print(f"{'=' * 60}\n", flush=True)

    for index, result in enumerate(outputs):
        if not result.images:
            raise RuntimeError(f"No video frames found in output {index}.")
        # Current H3 post-processing returns uint8 frames in [0, 255]. Keep
        # compatibility with older pipelines that returned floats in [0, 1].
        frames = np.asarray(result.images[0])
        if frames.dtype == np.uint8:
            frames = np.ascontiguousarray(frames)
        else:
            frames = (np.clip(frames, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        mm = result.multimodal_output or {}
        audio = mm.get("audio")
        audio_np = None
        if audio is not None:
            # H3 returns (1, N, 2); drop the batch dim for mux_video_audio_bytes,
            # which expects mono (N,) or (N, C) / (C, N).
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim == 3 and audio_np.shape[0] == 1:
                audio_np = audio_np[0]
        fps = float(mm.get("fps", MINIMAX_H3_FPS))
        sample_rate = int(mm.get("audio_sample_rate", MINIMAX_H3_AUDIO_SAMPLE_RATE))

        save_path = os.path.join(args.output, f"minimax_h3_{task or 'auto'}_{index}.mp4")
        with open(save_path, "wb") as f:
            f.write(
                mux_video_audio_bytes(
                    frames,
                    audio_np,
                    fps=fps,
                    audio_sample_rate=sample_rate,
                )
            )
        print(f"[Output] Saved video+audio to {save_path}")


if __name__ == "__main__":
    main()
