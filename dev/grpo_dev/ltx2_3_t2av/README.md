# LTX-2.3 T2AV GRPO rollout adapter

This directory contains a minimal vLLM-Omni adapter for using `LTX23Pipeline` as an LTX-2.3 text-to-audio-video rollout engine for GRPO-style training.

## Files

- `__init__.py`: imports the package and auto-registers the adapter.
- `register.py`: replaces the built-in `LTX23Pipeline` registry entry with `LTX23T2AVPipelineWithLogProb`.
- `pipeline.py`: subclasses the vLLM-Omni LTX-2.3 pipeline, swaps in an SDE scheduler, and exports rollout artifacts.
- `schedulers.py`: local `FlowMatchSDEDiscreteScheduler` implementation with log-prob support.
- `test_rollout.py`: offline smoke test that runs one generation and prints rollout field shapes.

## Rollout fields

The adapter returns final `(video, audio)` as usual and adds GRPO data through `DiffusionOutput.custom_output` / `OmniRequestOutput.custom_output`:

- `all_latents`: compact unified video+audio latent trajectory.
- `all_log_probs` / `log_probs`: old-policy SDE log probabilities.
- `all_timesteps`: timesteps aligned to `all_log_probs`.
- `timesteps`: resolved scheduler timesteps.
- `latent_index_map`: train-step index to saved-latent index map.
- `log_prob_index_map`: train-step index to saved-log-prob index map.
- `prompt_ids`: token IDs used for the actual text encoder pass.
- `connector_prompt_embeds`: positive video connector embeddings.
- `connector_audio_prompt_embeds`: positive audio connector embeddings.
- `connector_attention_mask`: positive connector attention mask.
- `negative_connector_prompt_embeds`: negative video connector embeddings when CFG is enabled.
- `negative_connector_audio_prompt_embeds`: negative audio connector embeddings when CFG is enabled.
- `negative_connector_attention_mask`: negative connector attention mask when CFG is enabled.
- `metadata`: resolved generation metadata, including `height`, `width`, `num_frames`, `frame_rate`, `video_seq_len`, `audio_sample_rate`, `guidance_scale`, `noise_level`, `sde_window`, and `sigmas`.
- `rl_rollout`: schema wrapper pointing to the in-memory fields above.

## Offline smoke test

From the repository root:

```bash
python -m ltx2_3_t2av.test_rollout \
  --model dg845/LTX-2.3-Diffusers \
  --height 512 \
  --width 768 \
  --num-frames 33 \
  --num-inference-steps 4 \
  --guidance-scale 4.0 \
  --seed 41 \
  --output dev/grpo_dev/ltx2_3_t2av/output/rollout_smoke.mp4
```

Run the command from `/efs/kwinsheng/vllm-omni/dev/grpo_dev`. The module import initializes the standalone `ltx2_3_t2av` package, so `LTX23Pipeline` is registered to instantiate `LTX23T2AVPipelineWithLogProb`. It prints shape summaries for all rollout fields and raises an error if required fields are missing.

## Programmatic usage

```python
import ltx2_3_t2av  # triggers registry replacement
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="dg845/LTX-2.3-Diffusers", model_class_name="LTX23Pipeline")
outputs = omni.generate(
    {"prompt": "A cinematic close-up of ocean waves at golden hour"},
    OmniDiffusionSamplingParams(
        height=512,
        width=768,
        num_frames=33,
        num_inference_steps=4,
        guidance_scale=4.0,
        extra_args={
            "noise_level": 0.7,
            "sde_type": "sde",
            "logprobs": True,
            "return_trajectory_latents": True,
            "rl_rollout": {"enabled": True},
        },
    ),
)
rollout = outputs[0].custom_output
```

## Online serving note

`vllm serve --omni` runs in a fresh process. Ensure this package is imported in that process before the diffusion model is constructed. Use the real script entrypoint `../serve_ltx2_3_t2av.py` instead of `python -` / heredoc because vLLM-Omni worker processes use Python multiprocessing spawn and need to reload the parent main module from a real file path.

From `/efs/kwinsheng/vllm-omni/dev/grpo_dev`:

```bash
sh serving.sh > serving.log 2>&1
```

Optional overrides:

```bash
export LTX23_T2AV_MODEL=dg845/LTX-2.3-Diffusers
export LTX23_T2AV_HOST=0.0.0.0
export LTX23_T2AV_PORT=8098
export LTX23_T2AV_DP=8
```

## Current scope

This is a minimal first-step adapter for testing output correctness. The large tensors are returned in memory through `custom_output`. A production rollout service should write large tensors to a sidecar artifact such as `safetensors` and return `rl_rollout.artifact_uri` instead of embedding all tensors in the response object.
