# GUIDE-01 / GUIDE-02 Implementation Plan

Status: implementation proposal, not an implemented API or a validation report.
Research date: 2026-09-11.
Repository baseline inspected: `c239cc813e9720b8af3be67a58ce204515659c75`.
Parent RFC: [plan.md](plan.md).

## 1. Scope and Deliverables

GUIDE-01 adds ordered timeline conditioning to the existing H3 pipeline and
`/v1/videos` plus `/v1/videos/sync`. GUIDE-02 adds a chainable ComfyUI guide
node and integrates it with the existing Generate Video node.

Deliver as two implementation PRs, each with its own tests and documentation:

| PR | Scope | Dependency |
| --- | --- | --- |
| GUIDE-01 | Contract, uploads, normalization, VAE encoding, packing, execution, backend tests/docs | API/semantic agreement |
| GUIDE-02 | Typed guide chain, node registration, Generate Video integration, serialization, client tests/docs | GUIDE-01 contract and working backend |

Required behavior includes image guides, clip guides, optional audio, ordered
chains, positive/negative frame indices, and valid H3 clip lengths. Ref2VA plus
guides is required: otherwise this work cannot unblock WF-04.

Do not implement WF-04, NODE-01, NODE-02, latent-mask editing, ControlNet, new
model loaders, a new generation node, or a new endpoint in these PRs. A focused
guide smoke test is part of GUIDE validation, not a shipped WF-04 template.

## 2. Findings That Determine the Design

Paths below are relative to the repository root. Existing behavior and proposed
behavior are intentionally separated.

| Existing code | Finding / consequence |
| --- | --- |
| `vllm_omni/entrypoints/openai/api_server.py::_parse_video_form` | Both video routes use multipart forms. Model-specific metadata uses `extra_params`, not a literal `extra_body` field. |
| `vllm_omni/entrypoints/openai/serving_video.py::_run_and_extract` | Ordinary image/video/audio inputs become `multi_modal_data`; extras become `sampling.extra_args`. Nested guide media is not decoded automatically. |
| `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py::_prepare_request_inputs` | Shared preparation for request and step execution. FL2VA only accepts one/two first/last images. |
| `pipeline_minimax_h3.py::_build_denoise_inputs` | Shared layout/anchor/schedule construction; the principal integration point. |
| `vllm_omni/diffusion/models/minimax_h3/packed_sequence.py` | FL2VA supports only first/last positions. Ref2VA places references before the target timeline. Neither supports arbitrary guides. |
| `vllm_omni/diffusion/models/minimax_h3/denoise_loop.py` | Already distinguishes fixed condition rows from generated target rows. Reuse this mechanism. |
| `vllm_omni/model_executor/models/minimax_h3/reference_video.py` | Existing reference preparation has reference-specific sizes and 2-15 second limits. Do not apply these unchanged to short guides. |
| `vllm_omni/model_executor/stage_input_processors/minimax_h3.py` | Split text encoding constructs ordinary reference presentations. Guide assets must survive to diffusion without becoming Qwen references. |
| `apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/utils/format.py` | `image_tensor_to_png_bytes` serializes only `tensor[0]`. It cannot serialize guide clips. |

The RFC's reference-order baseline needs a qualification: serving currently
groups mixed references by modality. It does not preserve arbitrary cross-modal
upload interleaving. Guide order must therefore have an independent manifest.
Do not expand this work into fixing reference ordering; that belongs to NODE-02.

## 3. Adopted Native Semantics

Use a pinned upstream reference, not a moving `master` branch:

- ComfyUI commit: `1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a`.
- [MiniMaxH3AddGuide](https://github.com/Comfy-Org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy_extras/nodes_minimax_h3.py).
- [PackedLayout and model forward](https://github.com/Comfy-Org/ComfyUI/blob/1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a/comfy/ldm/minimax/model.py).
- [Guide feature PR #15439](https://github.com/Comfy-Org/ComfyUI/pull/15439). RFC PR #15224 introduced base H3 support; it is not the complete guide reference.
- [Multiframe tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe).
- [Pinned multiframe template](https://github.com/Comfy-Org/workflow_templates/blob/18abde72b2b778610d90dc0d17cfbc37f0413344/templates/video_minimax_h3_multiframe_reference.json).

### 3.1 Guide meaning

A guide adds fixed visual/audio condition tokens with positions on the output
timeline. It does not overwrite generated latents, apply a target noise mask,
or guarantee pixel/sample-identical reconstruction.

Guides do not automatically enter the text encoder. An image needs a separate
ordinary reference connection if the prompt should refer to it as `<Picture N>`.
Guides neither consume ordinary reference slots nor renumber their labels.

### 3.2 Frames and indices

Let `N` be the actual aligned output frame count at 24 FPS. Resolve an index `i`:

```text
start = i if i >= 0 else N + i
```

Indices are zero-based pixel-frame positions, not VAE latent indices. Do not
snap a start to a latent boundary. For a visual guide of effective length `G`,
require `0 <= start` and `start + G <= N`.

Normalize available visual frames `M` using native behavior:

```text
M == 0: reject
1 <= M < 5: G = 1; use the first frame
M >= 5: G = 5 + 17 * floor((M - 5) / 17)
```

Output length rounds UP to `17k+5`; guide length rounds DOWN. Still images are
a separate one-frame case, not five-frame clips. Normalize the source first;
reject a normalized guide that does not fit instead of truncating it again to
the remaining output duration.

| Case with N=124 | Result |
| --- | --- |
| Image at -1 | Start 123; valid |
| 22-frame clip at -22 | Start 102; valid |
| 22-frame clip at -1 | Reject |
| Index -124 | Start 0 |
| Index -125 | Reject |
| 21 source frames | Use 5 frames |
| 23 source frames | Use 22 frames |

Initial API proposal has no explicit guide length or source-offset field. The
uploaded clip defines the source interval; a caller can trim it before upload.
This matches native Add Guide and avoids adding unnecessary controls. Include
effective source-frame counts in debug diagnostics so normalization is visible.

### 3.3 Spatial and audio handling

- Resize visual guides to target geometry with aspect-preserving center crop.
- Keep existing FL2VA image stretching unchanged.
- Explicit guide audio begins at the same frame index as its visual component.
- Audio may be shorter or longer than its associated visual component.
- Resample for the audio VAE, retain the established stereo/channel convention,
  and crop encoded audio to the remaining target audio capacity.
- Do not automatically use a video's embedded soundtrack. Require an explicit
  audio connection/upload, matching native Add Guide's separate audio input.
- Recommend accepting audio-only guide entries, as native Add Guide does.
  Confirm this small scope addition with maintainers before implementation.

For aligned output `N = 17k+5`:

```text
target_video_latent_t = 5k + 2
target_audio_t = round(N / 24 * 40)
max_guide_audio_t = floor(target_audio_t - (5 / 3) * start)
```

An audio-only guide uses a one-frame bound check for its start, then requires
at least one remaining audio latent position. Use the actual encoded audio
length when cropping. Native ComfyUI and the local audio VAE wrapper have
different crop/pad details; exact audio latent parity requires a separate
measurement and is not assumed by this plan.

### 3.4 Ordering, overlap, and strength

Preserve insertion order, including non-chronological and overlapping entries.
Do not sort, deduplicate, average, or define a last-write-wins rule. Native code
accepts overlaps but does not establish which conflicting guide dominates.

Do not add per-guide strength or masks. Reuse local condition-noise conventions
(visual 0.999, audio 1.0). They are not user-facing guide strength controls.

## 4. Proposed Public API and Internal Contract

This section proposes NEW fields. Names and placement need maintainer agreement.

### 4.1 Prefer a typed manifest plus multipart files

Continue using the two existing video endpoints. Add:

| Form field | Type | Meaning |
| --- | --- | --- |
| `timeline_guides` | JSON-encoded list | Ordered, schema-validated guide descriptors |
| `guide_files` | Repeated file upload | Files addressed by zero-based upload index |

Example manifest:

```json
[
  {"frame_index": 36, "image": {"upload_index": 0}},
  {
    "frame_index": -22,
    "video": {"upload_index": 1},
    "audio": {"upload_index": 2}
  }
]
```

The three `guide_files` parts are an image, a clip, and an audio file. Ordinary
`input_references` can coexist and have a completely separate index space.
`extra_params` remains responsible for `task` and existing H3 sampling metadata.

Reasons for this transport: preserve association/order, avoid base64 expansion
and inline-media logging, and avoid overloading reference grouping. Start with
uploads only; do not add URL fetching, file IDs, arbitrary server paths, or a
new upload service. If maintainers prefer `extra_params.timeline_guides`, retain
the same typed validation and upload binding rather than opaque passthrough.

### 4.2 Validation and ownership

1. Add a typed guide record near `VideoGenerationRequest` in `protocol/videos.py`.
   Use strict integers for frame/upload indices; reject booleans and unknown
   fields. Image/video are mutually exclusive; at least one media source is
   required. Audio-only support follows the decision in section 3.3.
2. Validate model capability and manifest/file associations before expensive
   decoding. Reject missing/out-of-range indices, wrong media types, orphaned
   uploads, and a nonempty manifest with no assets. Repeated use of a file is
   allowed; cleanup ownership is unique per uploaded file.
3. Bound file reads using existing upload helpers; validate content, not just
   suffix/MIME. Keep video/audio file-backed to preserve source streams.
4. Create an internal request-owned guide bundle containing ordered descriptors
   and cleanup paths. Add a narrow guide argument to serving functions; do not
   rebuild all reference APIs into a generic media framework.
5. Translate the bundle into H3's model-specific `sampling.extra_args` under a
   reserved internal guide key. Use plain serializable descriptors, not HTTP
   objects or GPU tensors. Reject external injection of that reserved key and
   ensure the final merge of `extra_params` cannot overwrite trusted descriptors.
6. Perform output-dependent validation after actual output shape resolution.
   Predictable input failures should use existing client-error semantics:
   immediate 4xx when detected before enqueue; structured failed-job errors
   when detected during async preparation, not an opaque server crash.
7. Own uploads until generation really finishes, including split-stage use.
   Cover partial parsing, queued cancellation, success, engine failure, sync
   timeout, and async cancellation. Do not delete files while a worker still
   reads them; follow and test actual cancellation semantics.

The offline Python path can accept trusted local sources through a model-side
adapter to the same normalizer. Public HTTP must not accept those internal paths.
Existing file-backed execution assumes worker filesystem visibility; do not
claim arbitrary cross-host artifact transfer support.

### 4.3 Task/checkpoint selection

Guides supplement the base task; no fourth task or checkpoint partition is needed.

| Request | Proposed routing |
| --- | --- |
| Ordinary references + guides | Existing Ref2VA routing/weights |
| First/last frames + guides | Existing FL2VA routing/weights; legacy anchors followed by explicit guides |
| Text + guides without ordinary inputs | Keep `t2va` task selection on FL2VA weights; guides use the separate conditioning channel |
| No guides | Existing behavior unchanged |

Guide-only `t2va` here identifies weight/task routing, not an unconditioned
generation promise. Confirm this naming with maintainers. Do not loosen the
existing rules for ordinary T2VA/FL2VA media or the legacy `frame_indices` parser.
Require explicit task selection where a single-partition server cannot satisfy
the chosen default, rather than switching to an unavailable transformer.

Validate base checkpoints first. Native LoRA, FastH3, Turbo, and sparse/cache
variants need explicit compatibility decisions; task name alone does not prove
they support guides. Reject an unsupported guide combination clearly while
preserving its existing no-guide behavior. Do not silently drop adapters or
claim untested checkpoint support.

## 5. GUIDE-01 Model Implementation

### 5.1 Normalize and encode once per request

Add one focused model-side module, for example
`vllm_omni/diffusion/models/minimax_h3/timeline_guides.py`, for source normalization
and guide validation. Avoid an abstract conditioning framework.

Extend `_prepare_request_inputs` in this order:

1. Resolve base task and target shape with existing code.
2. Parse trusted guide descriptors; resolve negative indices against aligned `N`.
3. Probe and decode only a bounded source interval at 24 FPS. For non-24-FPS
   clips, define a deterministic PTS-based resampling rule before length
   normalization; audio remains on source elapsed time, not frame ordinal.
4. Apply guide length normalization and output bounds checks; center-crop visual
   inputs to target geometry. A source longer than the output should be rejected
   with a trim instruction, not silently pre-cropped until it happens to fit.
5. Encode stills with `encode_image`; encode valid clips with `encode_video`.
   Assert actual returned latent shape. Reuse normalization/patchification in
   `vae.py`; do not use `_video_latent_t(1)` for stills.
6. Encode explicitly attached audio with `encode_waveform`; crop temporal
   samples in both channels correctly before flattening back to rows.
7. Return ordered prepared blocks with `resolved_frame_index`, effective frame
   count, visual rows/shape, and audio rows/per-channel length as applicable.
8. Keep guide sources completely out of Qwen reference presentation and labels.

Reuse low-level `reference_video.py` probe/decoder utilities only where their
policies match. Do not reuse its full Ref2VA validator: 5/22-frame guides must
not fail a two-second minimum or use a 2048-pixel reference-image canvas.

For distributed VAE execution, broadcast rank-0 preparation errors before any
subsequent collective. All ranks must enter image/clip encodes in a consistent
order; exercise short clips with tiled/patch-parallel encoding.

### 5.2 Build a combined packed layout

Extend `packed_sequence.py` with guide block support. Prefer extending the
existing heterogeneous block builder with optional `guide_blocks`, reusing its
coordinate helpers, while preserving no-guide layout results exactly. Retain
the legacy no-guide FL2VA constructor and parser.

Required physical sequence:

```text
[text | ordered guide blocks | ordinary reference blocks | target audio | target video | padding]
```

Within a guide, visual rows precede audio rows. Preserve existing reference
block ordering, including audio-before-video within a video reference block.
Distinguish physical token offsets from temporal coordinates.

Let `T0` be the target temporal origin after ordinary reference spans:

```text
T0 = text_length + sum(existing_reference_temporal_spans)
guide_origin = T0 + (5 / 3) * resolved_frame_index
guide_video_t[j] = guide_origin + (5 / 3) * sum((1,4,4,4,4)[i % 5] for i < j)
guide_audio_t[j] = guide_origin + j
```

Guide spans do NOT advance `T0`. Preserve fractional coordinates. A clip starts
its own local VAE temporal cadence; do not map its start to a target latent index.

Maintain `img_pos`, `audio_pos`, position IDs, token tags, padding, visual/audio
update masks, and physical `video_spans`. Include multi-frame guide spans where
attention consumers require them; use existing supported role semantics or
update all affected consumers. Do not merely invent a new role string. Keep
still guides out of multi-frame spans, following existing image behavior.

Expected row counts for a guide with latent shape `(T,Hl,Wl)` and audio length A:

```text
visual rows = T * (Hl / 2) * (Wl / 2), each with 96 values
audio rows = 2 * A, each with 32 values, channel-major
```

Condition rows get update flag `False`; target rows remain `True`. Concatenated
visual/audio anchor tensors and their metadata must follow the exact modality
row order of the new layout. Passing guide metadata without reordering anchors
is a silent correctness bug.

For FL2VA plus guides, adapt legacy first/last conditions into the combined
layout only for the new guided path. Keep legacy resize/text presentation and
first/last semantics. Do not turn first/last frames into ordinary references.

### 5.3 Reuse denoising and cover all execution paths

Extend `_build_denoise_inputs` and `_MINIMAX_H3_DENOISE_INPUT_KEYS` so request
`forward` and step `prepare_encode` consume identical prepared guide state.
Use existing condition-noise augmentation and anchor restoration; scheduler
updates apply only to target rows, and decoding returns target video/audio only.

Audit `diffuse`, `denoise_step`, `step_scheduler`, `post_decode`, and
`batched_packing.py` for metadata propagation. Test mixed guide counts/shapes
across batched requests and document isolation. Preserve existing restrictions
on DLO, fanout, and Cache-DiT in step mode rather than removing guards.

Test the disaggregated stage path independently: Stage 0 must ignore guides as
Qwen media, Stage 1 must still receive guide descriptors, and asset ownership
must extend through Stage 1. Do not modify generic workers unless a test proves
the existing model-specific transport is insufficient.

No learned transformer-layer changes are expected: it already embeds packed
media rows and position IDs. Verify this with real weights before treating it
as established support.

### 5.4 Resource limits

Arbitrary placement does not mean unbounded conditioning. Reuse existing
per-file limits where appropriate (currently image 30 MiB, video 50 MiB,
audio 15 MiB), but do not reuse the ordinary reference-count limit as a guide
semantic rule.

Add bounded aggregate upload bytes, source dimensions/decoded frames, audio
samples, probe/decode timeouts, and combined packed token budget. Estimate total
rows for guides AND references AND targets before allocating packed tensors.
Bound temporary files and concurrent preparation as well as GPU memory.

Guide-count/token-budget configuration names and defaults are a P0 decision,
based on a four-anchor and a clip-guide capacity measurement. They must be
defined before merge, not left unlimited. Tests should use injectable small
budgets to exercise rejection deterministically without allocating large media.

## 6. GUIDE-02 ComfyUI Implementation

### 6.1 Node and payload

Add `VLLMOmniMiniMaxH3AddGuide` with a dedicated typed socket such as
`H3_TIMELINE_GUIDES`:

| Input | Type | Behavior |
| --- | --- | --- |
| `frame_index` | INT | Start frame; negative values allowed |
| `guides` | Optional H3_TIMELINE_GUIDES | Previous chain |
| `image` | Optional IMAGE | Still or temporal frame batch |
| `video` | Optional VIDEO | Source clip, mutually exclusive with image |
| `audio` | Optional AUDIO | Explicit attached audio; optionally audio-only |

Return a fresh ordered list/container with one appended entry. Share media
objects without deep-copying tensors. Never mutate an upstream chain, which
may branch to multiple downstream nodes. Encode/upload only during generation,
not each time an Add Guide node executes.

The remote node needs no positive-conditioning or VAE socket: those are server
responsibilities. It also needs no target latent input; final output-bound
validation belongs to Generate Video/server, which know the aligned duration.

### 6.2 File changes

| File under `apps/ComfyUI-vLLM-Omni/` | Change |
| --- | --- |
| `comfyui_vllm_omni/utils/types.py` | Guide entry typing and ordered payload type |
| `comfyui_vllm_omni/nodes.py` | Add Guide node; optional `guides` input on `VLLMOmniGenerateVideo`; local validation |
| `comfyui_vllm_omni/utils/api_client.py` | Explicit guides argument; ordered manifest and file parts; preserve ordinary task routing |
| `comfyui_vllm_omni/utils/format.py` | Explicit IMAGE-batch clip serialization if existing VIDEO adapters are insufficient; lossless guide audio |
| `comfyui_vllm_omni/utils/models.py` | Only necessary H3 capability/canvas integration; do not put media into the sampling-parameter whitelist |
| `__init__.py` | Class/display-name registration |
| `README.md` | Chain usage, index/length/audio semantics, server version requirements |

Keep the frontend JavaScript unchanged unless normal static sockets cannot
express the chosen interface. No dynamic input-slot UI is required for chaining.

### 6.3 Serialization and compatibility

1. Serialize one-frame guides as PNG. Treat an IMAGE batch as temporal frames,
   not independent references; use the native length normalization rule.
2. For a multi-frame IMAGE batch, construct a 24-FPS clip using Comfy's video
   adapter or a small explicit serializer. Preserve frame count/order. Verify
   a lossless RGB-capable encoding accepted by the server rather than assuming
   the default MP4 save is lossless. Do not call the single-image helper on it.
3. Serialize VIDEO through existing helpers, with server-side authoritative
   FPS/length normalization. Ignore embedded audio unless separately connected.
4. Use an explicitly supported lossless audio format such as FLAC; do not use
   default MP3 where encoder delay can affect temporal guidance.
5. Build upload indices and the manifest in one pass. Merge task/H3 params/LoRA
   metadata once, preserving `audio_flow_shift` and existing sampling options.
6. Reject guides on unsupported models and reject non-24-FPS H3 requests; do not
   silently overwrite a user's FPS. The existing Wan-oriented 16-FPS/41-frame
   defaults are not valid guide examples.
7. For guide-only generation, provide an H3 aspect ratio derived from the chosen
   output canvas if the existing client cannot express it. Keep this adjustment
   scoped to H3 guide requests, not a redesign of H3 Params.
8. Return the existing VIDEO output, preserving decoded audio. Add an actual
   waveform regression fixture; an MP4 audio-stream header alone is insufficient.

Expected graph topology, not a shipped WF-04 workflow:

```text
image/clip A -> Add Guide(0) -> Add Guide(36) -> Add Guide(-1) -> Generate Video.guides
Video References ------------------------------------------> Generate Video.references
H3 Params / Sampling / Remote LoRA -------------------------> existing inputs
```

GUIDE-02 must allow `references + guides`. Existing `frame` versus `references`
exclusivity remains NODE-01/NODE-02's contract; guides are independent of it.

## 7. Implementation Sequence and Exit Gates

These are internal milestones, not new independently assigned RFC tasks.

| Milestone | Concrete work | Exit gate |
| --- | --- | --- |
| P0: contract | Post schema, pinned semantics, routing/audio policy and limits to RFC; coordinate NODE owners | Agreed API and explicitly documented compatibility matrix |
| P1: pure logic | Implement index/length normalization and guide block packing with synthetic tensors | Boundary, coordinate, order and mask tests pass without weights |
| P2: model vertical slice | Encode a middle-frame image guide; then clip and audio; combine with references | Offline real-model image guide works; shared request/step tests pass |
| P3: serving | Typed manifest, uploads, trusted dispatch, lifecycle and model capability validation | Async/sync mocked-server tests and API real-model smoke pass |
| P4: GUIDE-01 completion | Split-stage, batching/distributed cases, negative tests, API/recipe docs | GUIDE-01 checklist and recorded validation evidence complete |
| P5: ComfyUI | Guide payload/node, serialization, Generate Video integration, registration/docs | Branch-safe chain and client/server contract tests pass |
| P6: GUIDE-02 completion | Actual remote guide execution and audio-preserving output | GUIDE-02 checklist complete; publish stable interface to WF-04 owner |

Start with a middle-frame image, not frame 0: this proves new timeline behavior
rather than accidentally exercising existing FL2VA. Next prioritize Ref2VA plus
guides before UI polish. Client contract work can begin after P0, but do not
merge GUIDE-02 against a speculative backend.

## 8. Test Matrix

| Layer | Required focused cases |
| --- | --- |
| Schema | Empty/malformed records, bool/string indices, unknown fields, image+video conflict, missing/orphaned files, wrong MIME/content, reserved-path injection, wrong model |
| Length/index | M=0,1,2,4,5,6,21,22,23,38,39; indices 0,N-1,-1,-N,-N-1; clip start N-G vs N-G+1; requested vs aligned output length |
| Layout | Non-sorted and overlapping guides; offsets for frames 1,36,72,120; fractional audio coordinates; guide+reference target origin; exact anchor order; target-only decode; padding/video spans |
| VAE preparation | Still special case; clips 5/22/39; center crop; non-24-FPS source; empty/short audio; mono/stereo; audio longer than visual; source exceeding output |
| Denoising | Fixed visual/audio anchors after every step; target rows change; conditioned request/step equivalence; no-guide regression |
| Execution | Heterogeneous guide batches, request isolation, split-stage guide preservation, unchanged Qwen labels, distributed error propagation and short-clip encode order |
| Lifecycle | Both endpoints; partial decode failure, queue cancellation, engine error, sync timeout, async cancellation, success; no leaked or prematurely removed files |
| ComfyUI | Empty input rejection, branch-safe chains, IMAGE batch not reduced to first frame, manifest/file association, references+guides, retained sampling/LoRA metadata, waveform-preserving VIDEO output |
| Real weights | Middle image, tail image, 22-frame tail clip, visual+audio, references+multiple guides, legacy T2VA/FL2VA/Ref2VA regressions |

Existing tests to extend:

- `tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py`.
- `tests/diffusion/models/minimax_h3/test_minimax_h3_packing.py`.
- `tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py`.
- `tests/model_executor/stage_input_processors/test_minimax_h3.py`.
- `tests/entrypoints/openai_api/test_video_server.py`.
- `tests/entrypoints/openai_api/test_video_api_utils.py`.
- `tests/e2e/features/comfyui/test_comfyui_integration.py`.

Add `tests/diffusion/models/minimax_h3/test_minimax_h3_timeline_guides.py` only
for the new normalizer/preparation contract if it makes ownership clearer.
Keep layout and step coverage in their existing suites rather than duplicating
fixtures. Mock tests do not establish real-model quality or A/V synchronization.

### 8.1 Exact validation commands

Run from the repository root in an installed development/test environment.
These are planned validation commands, NOT commands run during plan preparation.

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py tests/diffusion/models/minimax_h3/test_minimax_h3_packing.py tests/diffusion/models/minimax_h3/test_minimax_h3_step_execution.py tests/model_executor/stage_input_processors/test_minimax_h3.py
```

```bash
python -m pytest -q tests/entrypoints/openai_api/test_video_server.py tests/entrypoints/openai_api/test_video_api_utils.py
```

```bash
python -m pytest -q tests/e2e/features/comfyui/test_comfyui_integration.py
```

If the proposed new normalizer test file is added:

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_timeline_guides.py
```

### 8.2 Real-model smoke procedure

Start an H3 server using an appropriate existing recipe profile, retaining its
exact command, checkpoint revision, hardware and attention/offload settings in
the PR record. Use Ref2VA weights for the combined-reference smoke below.
Do not assume this plan's workstation has the checkpoint access/RAM/GPU capacity.

After the proposed API exists, and with portable assets under `inputs/`, this
sync request validates transport for one semantic reference and two guides:

```bash
curl --fail-with-body --max-time 14400 http://localhost:8091/v1/videos/sync \
  -F 'model=MiniMaxAI/MiniMax-H3' \
  -F 'prompt=A continuous cinematic scene featuring <Picture 1>, with synchronized ambient sound.' \
  -F 'num_frames=124' \
  -F 'fps=24' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'seed=42' \
  -F 'extra_params={"task":"ref2va"}' \
  -F 'input_references=@inputs/reference.png;type=image/png' \
  -F 'timeline_guides=[{"frame_index":36,"image":{"upload_index":0}},{"frame_index":-22,"video":{"upload_index":1},"audio":{"upload_index":2}}]' \
  -F 'guide_files=@inputs/anchor.png;type=image/png' \
  -F 'guide_files=@inputs/clip_22_frames.mp4;type=video/mp4' \
  -F 'guide_files=@inputs/guide.flac;type=audio/flac' \
  --output guide_smoke.mp4
```

`clip_22_frames.mp4` must contain exactly 22 frames at 24 FPS. Input fixtures are
not created by this plan. The example intentionally omits a forced step count
so checkpoint-pinned schedule validation remains authoritative.

```bash
ffprobe -v error -count_frames -show_entries stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,sample_rate,channels,duration -of json guide_smoke.mp4
```

```bash
ffmpeg -v error -i guide_smoke.mp4 -f null -
```

Also exercise asynchronous create/poll/download and the actual ComfyUI chain.
Compare decoded audio/video start times and durations, accounting for codec
priming; propose an acceptance tolerance of one video frame plus documented
codec delay, and fix the exact measurement before recording results. Inspect
frames around anchors and compare a same-seed no-guide run. Avoid asserting
pixel equality as a quality criterion.

Record input hashes, effective frame counts/indices, full request, seed,
checkpoint/code revisions, steps/shifts, output artifact, waveform checks,
wall time, peak memory, and guide/no-guide observations. A four-image test at
0/36/72/120 is useful independent coverage, but is not automatically an exact
reproduction of the official workflow.

The pinned official multiframe template has one semantic picture reference and
explicit Add Guide nodes at 36/72/120; it does not contain an explicit guide at
0. Coordinate this topology detail with WF-04 rather than silently rewriting
the RFC's four-anchor description.

## 9. Documentation and Collaboration

GUIDE-01 updates `docs/serving/videos_api.md` and
`recipes/MiniMaxAI/MiniMax-H3.md`; update the disaggregated recipe if its usage
or restrictions change. Include the exact transport, normalization/index rules,
audio policy, supported checkpoint matrix, resource limits, and validation command.

GUIDE-02 updates the extension README and, if needed,
`docs/features/comfyui.md`. Explain chain branching, temporal IMAGE batches,
explicit audio, 24-FPS output, and guides versus semantic references. A prose
connection example is sufficient; WF-04 owns the official workflow JSON.

| Collaborator | Agreement needed |
| --- | --- |
| NODE-01 | Preserve `frame` compatibility and ownership of first/last sockets; agree on `guides` argument beside them |
| NODE-02 | Keep guide/reference payloads independent; agree on shared client form construction and references+guides tests |
| WF-04 | Publish stable node/socket names and index semantics; workflow ships only after GUIDE-02 and NODE-02 |
| EDIT-01/02 | Guides are fixed extra rows, not masked/preserved target rows |

Current ComfyUI restrictions on image-only/mixed references may prevent testing
the official topology before NODE-02 lands. GUIDE-01 must support it through the
API; GUIDE-02 can test an already-supported reference combination and contract
fixtures meanwhile. Do not silently take over NODE-02 to unblock a UI demo.

## 10. Merge Checklist and Open Decisions

Before coding the public interface, post these recommendations to the RFC:

- Typed top-level `timeline_guides` plus repeated `guide_files`; use upload-only
  sources in the initial implementation.
- Native auto-trimming, negative-start-index semantics, overlap acceptance,
  explicit audio, and no strength/mask controls.
- Include audio-only guides if maintainers agree; otherwise reject explicitly.
- Guide-only task routing on base FL2VA weights, and an explicit compatibility
  matrix for adapters/accelerated execution.
- Bounded guide assets and conditioning-token budgets, with documented defaults
  informed by a short capacity run.

GUIDE-01 is complete only when model, API, async/sync lifecycle, shared execution
paths, focused tests, documentation and recorded validation are covered. GUIDE-02
is complete only when chains serialize correctly, work with references, preserve
output audio, and execute remotely against GUIDE-01. Do not mark WF-04 complete.

No implementation code, tests, workflows, or real-model runs were performed while
preparing this plan. Existing unrelated workspace edits were left untouched.
