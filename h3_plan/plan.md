https://github.com/vllm-project/vllm-omni/issues/7380#issue-5413090678
## Goal

Complete MiniMax H3 support in the existing vLLM-Omni ComfyUI nodes, add APIs only for H3 features that the server cannot currently express, and recreate the official and mainstream community workflows with remote vLLM-Omni execution. (ComfyUI location: `vllm-omni/apps/ComfyUI-vLLM-Omni`)

## Task model

Each checklist item is independently assignable and verifiable. Tasks are split by code boundary or workflow topology, not by different input examples that use the same implementation.

## Current baseline

- `/v1/videos` already supports T2VA, first-frame FL2VA, mixed Ref2VA, repeated `input_references`, reference ordering, and H3 reference limits.
- `VLLMOmniGenerateVideo` already selects T2VA, FL2VA, or Ref2VA from its connected inputs.
- `VLLMOmniMiniMaxH3Params`, `VLLMOmniVideoReferences`, `VLLMOmniRemoteLoRA`, and diffusion sampling nodes already exist.
- The current ComfyUI client supports only one FL2VA frame, caps each reference modality at two inputs, and rejects some valid mixed-reference combinations.
- Arbitrary timeline guides, latent-mask editing, and Fun ControlNet Union are not yet exposed through vLLM-Omni APIs.

## Core ComfyUI node tasks

- [ ] **NODE-01 — Complete H3 first/last-frame support in Generate Video.** Add separate optional `first_frame` and `last_frame` inputs while preserving compatibility with the existing `frame` input.
- [ ] **NODE-02 — Complete H3 reference support in Video References.** Expand the existing reference node and request serializer to support ordered mixed inputs up to 9 images, 3 videos, and 3 audio clips.

No separate H3 Text to Video or Reference to Video generation node is required; the existing Generate Video node already owns that behavior.

## Official base workflow tasks

- [ ] **WF-01 — Add the official H3 Text to Video workflow.** Use the existing Generate Video, H3 Params, sampling, and Remote LoRA nodes.
- [ ] **WF-02 — Add the official H3 Image to Video workflow.** Cover first-frame, last-frame, and first-and-last-frame generation in one template. Depends on NODE-01.
- [ ] **WF-03 — Add the official H3 Reference to Video workflow.** Accept any supported combination of image, video, and audio references in one template. Depends on NODE-02.

Turbo is an option inside WF-01 through WF-03, not a separate workflow family. Each template must expose the appropriate remote LoRA, step count, and flow-shift settings.

## Arbitrary multiframe guide tasks

- [ ] **GUIDE-01 — Add arbitrary timeline guide conditioning to the model and API.** Support ordered image or clip guides, optional audio, positive and negative frame indices, and valid `17k+5` guide lengths.
- [ ] **GUIDE-02 — Add H3 timeline guide support to the ComfyUI extension.** Allow guides to be chained and serialized to the API. Depends on GUIDE-01.
- [ ] **WF-04 — Add the official H3 Multiframe Reference workflow.** Recreate the four-anchor template with remote guide execution. Depends on GUIDE-02 and NODE-02.

## Latent-mask editing tasks

- [ ] **EDIT-01 — Add H3 latent initialization and noise-mask conditioning to the model and API.** Cover both video and audio latents with per-token preservation and regeneration semantics.
- [ ] **EDIT-02 — Add H3 latent-mask editing support to the ComfyUI extension.** Upload the source media and serialize video and audio masks. Depends on EDIT-01.
- [ ] **WF-05 — Add the H3 latent-mask editing workflow.** Provide inpainting, object removal, continuation, and extension examples in one template. Depends on EDIT-02.

## Fun ControlNet Union tasks

- [ ] **CTRL-01 — Add H3 Fun ControlNet Union serving.** Load the control model and expose control video, source video, mask, control type, and strength through the API.
- [ ] **CTRL-02 — Add H3 Fun ControlNet Union support to the ComfyUI extension.** Serialize all supported control modes to the API. Depends on CTRL-01.
- [ ] **WF-06 — Add the official H3 Fun ControlNet Union workflow.** Cover Canny, Depth, HED, MLSD, Pose, and inpainting, including the pose-extraction subgraph. Depends on CTRL-02.

## Community workflow task

- [ ] **WF-07 — Add a synchronized H3 video-upscale workflow.** Use ComfyUI post-processing nodes after remote generation and preserve the generated audio track.

## Dependency summary

```text
NODE-01 ----------------------------> WF-02
NODE-02 ----------------------------> WF-03, WF-04
GUIDE-01 ------------> GUIDE-02 -----> WF-04
EDIT-01 -------------> EDIT-02 ------> WF-05
CTRL-01 -------------> CTRL-02 ------> WF-06
Existing nodes ----------------------> WF-01, WF-07
```

Tasks on the same level may be assigned and implemented in parallel.

## Workflow requirements

Every shipped workflow must:

- use portable input paths and avoid machine-specific paths;
- use 24 FPS and preserve synchronized audio where applicable;
- respect H3's `17k+5` frame-length constraint;
- use H3-native resolution presets;
- call remote vLLM-Omni nodes instead of local H3 model-loader nodes.

Community graphs that only configure local inference should be translated to the equivalent server configuration:

| Community graph feature | vLLM-Omni equivalent |
| --- | --- |
| Low-VRAM loading or chunked feed-forward | Server offload/DLO configuration |
| Multi-GPU loader nodes | `vllm serve` parallelism flags |
| Sage, VSA, or block-sparse attention nodes | Server attention backend configuration |
| Local INT8 or NVFP4 loaders | Server checkpoint and quantization options |
| Local VAE tiling or FastVAE nodes | Server VAE tiling and parallelism |

## Requirements for every implementation PR

Every PR that completes one or more tasks must:

- include focused automated tests for the behavior it changes;
- update the relevant API documentation, extension README, or workflow notes;
- provide an exact validation command;
- include a recorded real-model result when it adds or changes a workflow;
- preserve backward compatibility unless the RFC explicitly permits a break;
- reference the completed task IDs in its description.

Tests and documentation are part of each implementation PR and must not be filed as separate tasks.

## Not separate tasks

- Existing base H3 API behavior that only needs coverage while implementing a node or workflow.
- Individual first-frame, last-frame, and first-and-last-frame I2V examples.
- Individual image, video, and audio combinations supported by the same R2V graph.
- Turbo variants of T2V, I2V, and R2V.
- Individual ControlNet types selected by the same Fun ControlNet Union graph.
- Tests, documentation, examples, or validation for another implementation task.

## Out of scope

- Supporting ComfyUI-repacked H3 checkpoints directly in vLLM-Omni.
- Configuring GPU topology from a ComfyUI graph.
- Pixel-identical parity with native ComfyUI.
- Treating every community optimization fork as a distinct workflow.
- Shipping a workflow before its required API or node task is complete.

## References

- [vLLM-Omni MiniMax H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md)
- [vLLM-Omni video API](../../serving/videos_api.md)
- [vLLM-Omni ComfyUI extension](../../../apps/ComfyUI-vLLM-Omni/README.md)
- [ComfyUI H3 workflow index](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [ComfyUI native H3 workflows and advanced techniques](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)
- [ComfyUI Multiframe Reference](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe)
- [ComfyUI Fun ControlNet Union](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fun-controlnet)
- [ComfyUI H3 implementation change](https://github.com/Comfy-Org/ComfyUI/pull/15224)
- [Official H3 T2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
- [Official H3 I2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json)
- [Official H3 R2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json)
- [Community H3 workflow collection](https://github.com/sepiablue-ai/minimax_h3_workflows)
- [MiniMax H3 Turbo workflows](https://github.com/ModelTC/Minimax-H3-Turbo)

