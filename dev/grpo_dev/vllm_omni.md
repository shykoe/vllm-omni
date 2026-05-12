# 目标

我希望使用 vllm_omni 作为 ltx2.3 t2av 的 rollout 引擎，参考 Flow-Factory LTX2.3 GRPO 的逻辑：

- [`Flow-Factory/src/flow_factory/trainers/grpo.py`](../../Flow-Factory/src/flow_factory/trainers/grpo.py)
- [`Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py)

对 vllm_omni 服务的请求需要返回一些额外的信息作为训练的输入。本文档补充 vllm_omni rollout 响应中需要返回的信息。

## Flow-Factory GRPO 对 rollout 结果的使用方式

[`GRPOTrainer.sample()`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:141) 在采样阶段会调用 adapter 的 inference，并要求 rollout 样本携带：

1. 奖励计算所需的最终媒体：生成视频、生成音频、音频采样率、prompt 等。
2. GRPO 训练重新计算 log probability 所需的扩散轨迹：每个被训练 timestep 对应的当前 latent、下一个 latent、旧策略 log probability。
3. 重放一次训练 forward 所需的条件信息：文本 connector embedding、negative connector embedding、生成尺寸、视频/音频 latent 分割点、各类 guidance 参数。
4. GRPO 分组所需的样本身份信息：原始 prompt 或 prompt token id。

[`GRPOTrainer.optimize()`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:185) 中会把 sample stack 成 batch，然后在每个训练 timestep 使用：

- [`old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:229)
- [`latents = batch['all_latents'][:, latents_index_map[timestep_index]]`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:239)
- [`next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:240)
- [`forward_inputs = { ... **batch }`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:242)

因此，vllm_omni 服务如果作为远端 rollout 引擎，响应必须足够重建 Flow-Factory 的 [`LTX2Sample`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:97)。

## LTX2 T2AV rollout 样本字段来源

Flow-Factory 的 [`LTX2_T2AV_Adapter.inference()`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1016) 在最终构造 [`LTX2Sample`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1251) 时写入以下信息：

| 类别 | 字段 | 训练用途 | 来源 |
| --- | --- | --- | --- |
| 扩散轨迹 | `timesteps` | 训练 forward 使用当前 timestep 和 next timestep | [`timesteps=timesteps`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1253) |
| 扩散轨迹 | `all_latents` | 取当前 latent 和 next latent 重新计算 log prob | [`all_latents=torch.stack(...)`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1254) |
| 扩散轨迹 | `log_probs` | PPO ratio 中的 old log prob | [`log_probs=torch.stack(...)`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1255) |
| 扩散轨迹索引 | `latent_index_map` | 将训练 timestep 映射到 `all_latents` 的下标 | [`latent_index_map=lat_map`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1258) |
| 扩散轨迹索引 | `log_prob_index_map` | 将训练 timestep 映射到 `log_probs` 的下标 | [`log_prob_index_map=lp_map`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1259) |
| 最终媒体 | `video` | reward model / 评估 / 日志 | [`video=video[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1261) |
| 最终媒体 | `audio` | audio-video reward / 评估 / 日志 | [`audio=audio_waveform[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1262) |
| 最终媒体 | `audio_sample_rate` | 正确解释 audio waveform | [`audio_sample_rate=int(...)`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1263) |
| 生成尺寸 | `height` / `width` / `num_frames` / `frame_rate` | 训练 forward 中重建 latent shape、RoPE coords、audio length | [`height=height`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1269) |
| latent 切分 | `video_seq_len` | 将 unified latent 切分为 video/audio latent | [`video_seq_len=video_seq_len`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1273) |
| prompt 身份 | `prompt` | reward 分组、日志、可复现性 | [`prompt=prompt_list[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1275) |
| prompt 身份 | `prompt_ids` | 样本 unique id / 分组的稳定身份 | [`prompt_ids=prompt_ids[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1276) |
| 条件 embedding | `connector_prompt_embeds` | 训练 forward 的 video text conditioning | [`connector_prompt_embeds=connector_prompt_embeds[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1278) |
| 条件 embedding | `connector_audio_prompt_embeds` | 训练 forward 的 audio text conditioning | [`connector_audio_prompt_embeds=connector_audio_prompt_embeds[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1279) |
| 条件 mask | `connector_attention_mask` | transformer attention mask | [`connector_attention_mask=connector_attention_mask[b]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1280) |
| CFG 条件 embedding | `negative_connector_prompt_embeds` | CFG 时训练 forward 的 negative video text conditioning | [`negative_connector_prompt_embeds=...`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1281) |
| CFG 条件 embedding | `negative_connector_audio_prompt_embeds` | CFG 时训练 forward 的 negative audio text conditioning | [`negative_connector_audio_prompt_embeds=...`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1286) |
| CFG 条件 mask | `negative_connector_attention_mask` | CFG 时 transformer negative attention mask | [`negative_connector_attention_mask=...`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1291) |
| 附加元数据 | `callback_index_map` | GRPO-Guard 或额外 callback trajectory 对齐 | [`callback_index_map`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1298) |
| 附加元数据 | `duration_s` | 生成时长记录，也可用于音频长度校验 | [`duration_s`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:1299) |

## vllm_omni 响应必须补充的字段

建议在视频生成响应的每个 sample 下新增 `rl_rollout` 字段，或返回一个 sidecar artifact URI。原因是轨迹 tensor 非常大，不适合直接塞进 OpenAI 兼容的 `b64_json` 响应。

### 1. 最小必需字段

这些字段缺失会导致 GRPO 无法重新计算 policy loss。

```json
{
  "data": [
    {
      "b64_json": "<mp4 base64>",
      "rl_rollout": {
        "artifact_uri": "file:///path/to/rollout_000001.safetensors",
        "schema_version": 1,
        "tensor_format": "safetensors",
        "sample_index": 0,
        "fields": {
          "timesteps": "timesteps",
          "all_latents": "all_latents",
          "log_probs": "log_probs",
          "latent_index_map": "latent_index_map",
          "log_prob_index_map": "log_prob_index_map",
          "prompt_ids": "prompt_ids",
          "connector_prompt_embeds": "connector_prompt_embeds",
          "connector_audio_prompt_embeds": "connector_audio_prompt_embeds",
          "connector_attention_mask": "connector_attention_mask",
          "negative_connector_prompt_embeds": "negative_connector_prompt_embeds",
          "negative_connector_audio_prompt_embeds": "negative_connector_audio_prompt_embeds",
          "negative_connector_attention_mask": "negative_connector_attention_mask"
        },
        "metadata": {
          "prompt": "...",
          "height": 512,
          "width": 768,
          "num_frames": 121,
          "frame_rate": 24.0,
          "video_seq_len": 65280,
          "duration_s": 5.0416666667,
          "audio_sample_rate": 48000
        }
      }
    }
  ]
}
```

sidecar artifact 中必须包含：

| tensor key | 期望形状 | dtype 建议 | 说明 |
| --- | --- | --- | --- |
| `timesteps` | `[num_inference_steps]` | `float32` 或原始 dtype | scheduler timesteps；训练时会根据 timestep index 读取当前 timestep 和 next timestep |
| `all_latents` | `[num_saved_latent_steps, video_seq_len + audio_seq_len, latent_channels]` | rollout 原始 dtype，建议 `bf16` / `fp16` | unified latent，video 在前，audio 在后 |
| `log_probs` | `[num_saved_log_prob_steps]` 或 `[num_saved_log_prob_steps, ...]` | `float32` | old policy log prob；只对 SDE/video pathway 有意义 |
| `latent_index_map` | `[num_inference_steps + 1]` | `int64` | train timestep index 到 saved latent index 的映射 |
| `log_prob_index_map` | `[num_inference_steps]` | `int64` | train timestep index 到 saved log prob index 的映射 |
| `prompt_ids` | `[max_sequence_length]` | `int64` | 与 connector embedding 同一次 tokenize 得到的 token id |
| `connector_prompt_embeds` | `[text_seq_len, video_text_dim]` | rollout 原始 dtype | positive video text connector output |
| `connector_audio_prompt_embeds` | `[text_seq_len, audio_text_dim]` | rollout 原始 dtype | positive audio text connector output |
| `connector_attention_mask` | `[text_seq_len]` | rollout 原始 dtype | connector attention mask |
| `negative_connector_prompt_embeds` | `[text_seq_len, video_text_dim]` 或空 | rollout 原始 dtype | CFG 启用时必需 |
| `negative_connector_audio_prompt_embeds` | `[text_seq_len, audio_text_dim]` 或空 | rollout 原始 dtype | CFG 启用时必需 |
| `negative_connector_attention_mask` | `[text_seq_len]` 或空 | rollout 原始 dtype | CFG 启用时必需 |

### 2. 最终媒体字段

这些字段主要用于 reward 计算、debug 和日志：

| 字段 | 必需性 | 说明 |
| --- | --- | --- |
| `video` / `b64_json` / `video_uri` | 必需 | reward 通常需要最终 mp4 或 tensor video；OpenAI 兼容响应可继续使用 `b64_json` |
| `audio` / `audio_uri` | t2av 必需 | LTX2.3 是 audio-video 模型，reward 可能同时依赖 video 和 audio |
| `audio_sample_rate` | audio 返回时必需 | LTX2.3 可能使用 48kHz vocoder；训练端不能假设固定采样率 |
| `revised_prompt` / `enhanced_prompt` | 建议 | 如果服务端做 prompt enhancement，应返回实际进入 text encoder 的 prompt |

### 3. 生成参数回传字段

如果训练端和 rollout 服务端完全共享配置，这些可以只做校验；如果允许 per-request 参数，则必须回传 resolved values，保证训练端重新 forward 时与 rollout 一致。

| 字段 | 必需性 | 说明 |
| --- | --- | --- |
| `num_inference_steps` | 必需 | 与 `timesteps`、index map 对齐 |
| `guidance_scale` | CFG 使用时必需 | video CFG scale |
| `audio_guidance_scale` | CFG 使用时必需 | audio CFG scale；缺省时通常等于 `guidance_scale` |
| `guidance_rescale` | 使用时必需 | video guidance rescale |
| `audio_guidance_rescale` | 使用时必需 | audio guidance rescale |
| `stg_scale` | 使用 STG 时必需 | video STG scale |
| `audio_stg_scale` | 使用 STG 时必需 | audio STG scale |
| `spatio_temporal_guidance_blocks` | 使用 STG 时必需 | STG block ids |
| `modality_scale` | 使用 modality isolation 时必需 | video modality isolation scale |
| `audio_modality_scale` | 使用 modality isolation 时必需 | audio modality isolation scale |
| `use_cross_timestep` | LTX2.3 使用时必需 | transformer forward 兼容参数 |
| `noise_scale` / `seed` / `sigmas` | 建议 | 可复现与 debug |
| `max_sequence_length` | 建议 | text encoder / connector shape 校验 |

## 请求侧需要新增或确认的控制参数

为了让 vllm_omni 服务只在训练 rollout 时返回大 tensor，建议请求中增加以下开关：

```json
{
  "prompt": "...",
  "extra_params": {
    "rl_rollout": {
      "enabled": true,
      "compute_log_prob": true,
      "return_trajectory_latents": true,
      "return_trajectory_log_probs": true,
      "trajectory_indices": "train_timesteps",
      "artifact_format": "safetensors",
      "artifact_storage": "local_file"
    }
  }
}
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `enabled` | 是否启用 GRPO rollout 数据返回 |
| `compute_log_prob` | 必须为 true，否则没有 old log prob |
| `return_trajectory_latents` | 必须为 true，否则训练端没有 current/next latent |
| `return_trajectory_log_probs` | 必须为 true，否则无法计算 PPO ratio |
| `trajectory_indices` | 建议支持 `all`、显式 index list、`train_timesteps`；Flow-Factory 使用 [`compute_trajectory_indices()`](../../Flow-Factory/src/flow_factory/utils/trajectory_collector.py:34) 按训练 timestep 选择保存位置 |
| `artifact_format` | 建议 `safetensors`；也可支持 `pt` / `npz` |
| `artifact_storage` | 建议 `local_file` 或对象存储 URI；避免把大 tensor JSON base64 化 |

## 与当前 vllm_omni 能力的关系

当前 vllm_omni 内部输出结构已经存在通用 trajectory 字段：

- [`trajectory_latents`](../../vllm_omni/outputs.py:89)
- [`trajectory_timesteps`](../../vllm_omni/outputs.py:90)
- [`trajectory_log_probs`](../../vllm_omni/outputs.py:91)
- [`return_trajectory_latents`](../../vllm_omni/inputs/data.py:279)

但对 LTX2.3 T2AV GRPO 来说，仅有通用 trajectory 还不够，还需要补齐：

1. `latent_index_map` / `log_prob_index_map`，用于和 Flow-Factory 的 train timestep 对齐。
2. `video_seq_len`，用于把 unified latent 切分成 video/audio。
3. connector positive/negative embeddings 和 attention mask，用于训练端重新 forward。
4. prompt identity 信息：`prompt`、`prompt_ids`，用于 GRPO group / unique id。
5. audio waveform 或 audio URI，以及 `audio_sample_rate`，用于 t2av reward。
6. resolved generation 参数，避免服务端 rollout 与训练端 forward 参数不一致。

## 当前框架缺口判断

结论：当前 vllm_omni 框架只能“部分满足” LTX2.3 T2AV GRPO rollout 的输入要求。框架层已经有通用 trajectory 字段和 diffusion output 透传路径，但 LTX2.3 pipeline 层尚未按 Flow-Factory GRPO 的语义采集和导出完整训练输入，因此目前还不能开箱即用地作为 LTX2.3 GRPO rollout 引擎。

### 已具备的基础能力

| 能力 | 当前状态 | 参考位置 | 对 GRPO 的意义 |
| --- | --- | --- | --- |
| 请求侧轨迹开关 | 已有 `return_trajectory_latents` | [`return_trajectory_latents`](../../vllm_omni/inputs/data.py:279) | 说明框架已有“请求端启用轨迹返回”的入口 |
| 输出结构预留 trajectory 字段 | 已有 `trajectory_latents`、`trajectory_timesteps`、`trajectory_log_probs` | [`trajectory_latents`](../../vllm_omni/outputs.py:89)、[`trajectory_timesteps`](../../vllm_omni/outputs.py:90)、[`trajectory_log_probs`](../../vllm_omni/outputs.py:91) | 可以承载部分 rollout 轨迹数据 |
| diffusion engine 透传 trajectory 字段 | 已将 diffusion output 透传到 `OmniRequestOutput` | [`OmniRequestOutput.from_diffusion()`](../../vllm_omni/diffusion/diffusion_engine.py:253) | 说明 engine 层可把 pipeline 输出传到服务结果对象 |
| LTX2.3 内部已计算 connector outputs | pipeline 内部已有 positive/negative prompt encode 与 connector 调用 | [`self.connectors(...)`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:752) | 训练所需的 connector embedding 在推理时可获得 |
| LTX2.3 同时维护 video/audio latents | denoising loop 中同时更新 video 和 audio latents | [`latents = self.scheduler.step(...)`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:910)、[`audio_latents = audio_scheduler.step(...)`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:911) | 具备构造 unified video+audio latent 的基础 |
| 最终 audio payload 透传 | diffusion engine 可将 audio 和 sample rate 放入 multimodal output | [`mm_output["audio"]`](../../vllm_omni/diffusion/diffusion_engine.py:342)、[`mm_output["audio_sample_rate"]`](../../vllm_omni/diffusion/diffusion_engine.py:344) | t2av reward 可使用最终音频 |

### 不满足 LTX2.3 GRPO 的关键缺口

| 缺口 | 当前表现 | 为什么阻塞 GRPO | 需要补充 |
| --- | --- | --- | --- |
| 未收集 Flow-Factory 语义的 `all_latents` | LTX2.3 denoising loop 只更新 video `latents` 和 `audio_latents`，未将两者拼接并按 step 收集 | Flow-Factory 训练端通过 [`batch['all_latents']`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:239) 取 current/next latent 重新计算 log prob | 在 step 0 和每个 denoising step 后收集 `torch.cat([video_latents, audio_latents], dim=1)`，形成 `all_latents` |
| 未产生 old policy `log_probs` | 当前 LTX2.3 使用普通 scheduler step：[`self.scheduler.step(..., return_dict=False)`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:910)，没有 log probability 输出 | GRPO/PPO ratio 依赖 [`old_log_prob`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:229)，没有它无法计算 policy loss | 引入/对齐 Flow-Factory 的 SDE scheduler step，支持 `compute_log_prob=True` 并返回 `log_prob` |
| 缺少 `latent_index_map` / `log_prob_index_map` | 通用输出只有 trajectory tensor，没有 index map | Flow-Factory 通过 [`latent_index_map`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:216) 和 [`log_prob_index_map`](../../Flow-Factory/src/flow_factory/trainers/grpo.py:217) 将训练 timestep 对齐到保存轨迹 | 返回与 trajectory collector 语义一致的两个 index map |
| connector embeddings 未导出 | pipeline 内部有 [`connector_prompt_embeds`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:752)，但未保存到 `custom_output` 或 sidecar | 训练端 forward 需要使用 rollout 当时完全一致的条件 embedding；重新 encode 可能因 prompt enhancement、padding side、dtype 差异导致不一致 | 保存 positive/negative connector embeddings 与 attention mask 到 rollout artifact |
| `prompt_ids` 未系统返回 | 当前服务响应没有明确返回与 connector 同一次 tokenize 得到的 token ids | GRPO 分组和 [`BaseSample.unique_id`](../../Flow-Factory/src/flow_factory/samples/samples.py:283) 依赖 prompt identity；只返回字符串在 prompt enhancement 场景下不够稳定 | 返回实际进入 text encoder 的 `prompt_ids`，以及 enhanced prompt 文本 |
| 缺少 `video_seq_len` | vllm_omni LTX2.3 内部可计算 video/audio latent shape，但响应未绑定该元数据 | Flow-Factory forward 用它将 unified latent 切成 video/audio：[`video_latents = latents[:, :video_seq_len]`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:758) | 将 `video_seq_len` 写入 rollout metadata |
| 缺少 resolved generation metadata | 高宽、帧数、fps、guidance 参数等没有作为 GRPO artifact 的稳定 metadata 返回 | 训练端重放 forward 时必须和 rollout 参数一致，否则 log prob 对不上旧策略采样分布 | 返回 `height`、`width`、`num_frames`、`frame_rate`、guidance/STG/modality 参数、`sigmas` 等 resolved values |
| 服务响应层没有 sidecar artifact 协议 | 当前 OpenAI 兼容响应主要面向最终媒体；大 tensor 不适合直接 JSON 返回 | `all_latents`、connector embeddings 非常大，直接放入 `b64_json` / JSON 会造成性能和内存问题 | 增加 `rl_rollout.artifact_uri`，大 tensor 用 `safetensors` / `pt` / `npz` 存 sidecar |

### 当前满足度判断

| Flow-Factory LTX2.3 GRPO 输入项 | 当前框架是否满足 | 说明 |
| --- | --- | --- |
| 最终视频 | 基本满足 | 视频生成接口可返回最终 mp4 / frames |
| 最终音频与采样率 | 部分满足 | engine 层可透传 audio/audio_sample_rate，但需要确认具体 HTTP response 是否暴露 |
| `timesteps` | 部分满足 | 通用字段存在，但 LTX2.3 pipeline 未按 GRPO artifact 语义系统返回 |
| `all_latents` | 不满足 | 需要 unified video+audio trajectory，而不是仅最终 latent 或通用字段占位 |
| `log_probs` | 不满足 | 当前 LTX2.3 scheduler step 不产出 SDE log prob |
| `latent_index_map` / `log_prob_index_map` | 不满足 | 当前无 Flow-Factory trajectory collector 语义的 index map |
| `prompt` / `prompt_ids` | 部分满足 | prompt 字符串通常有，`prompt_ids` 未作为 rollout artifact 返回 |
| connector positive/negative embeddings | 不满足 | 内部已计算，但未导出给训练端 |
| `height` / `width` / `num_frames` / `frame_rate` | 部分满足 | pipeline 内部有 resolved values，但未和 GRPO artifact 系统绑定 |
| `video_seq_len` | 不满足 | 内部可计算，但未返回 |
| resolved guidance/STG/modality 参数 | 部分满足 | 请求中可能有，但训练端需要服务端实际使用后的 resolved values |

### 最小可用改造范围

若目标是让当前 vllm_omni 真正满足 LTX2.3 GRPO rollout 输入，最小改造应覆盖以下闭环：

1. 在 [`LTX23Pipeline`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:107) 中加入 GRPO rollout collection 分支，受 `extra_params.rl_rollout.enabled` 控制。
2. 在 denoising loop 开始前保存初始 unified latent；每个 step 后保存 next unified latent。
3. 替换或扩展 video scheduler step，使其支持 SDE sampling、`compute_log_prob=True`、`next_latents` 条件 log prob 计算，并返回 old `log_prob`。
4. 为保存的 latent/log_prob 生成 `latent_index_map`、`log_prob_index_map`；若只保存训练 timestep 子集，必须保证 map 与 Flow-Factory [`compute_trajectory_indices()`](../../Flow-Factory/src/flow_factory/utils/trajectory_collector.py:34) 语义一致。
5. 在 prompt encode / connector 后，将 `prompt_ids`、positive/negative connector embeddings、attention masks 写入 sidecar artifact。
6. 将 `video_seq_len`、`audio_seq_len`、shape、dtype、`height`、`width`、`num_frames`、`frame_rate`、`audio_sample_rate`、guidance 参数等写入 `rl_rollout.metadata`。
7. 在服务响应协议中返回 `rl_rollout.artifact_uri`，并保持最终 `b64_json` / audio payload 可供 reward 使用。

## 推荐落地方案

1. 在 vllm_omni LTX2.3 pipeline 的采样循环中，以与 Flow-Factory 一致的方式收集 unified `all_latents`：`torch.cat([video_latents, audio_latents], dim=1)`。
2. 在每个 SDE step 收集 `log_probs`；audio ODE pathway 不需要单独 log prob。
3. 生成并返回 `latent_index_map`、`log_prob_index_map`，语义保持和 Flow-Factory 的 trajectory collector 一致。
4. 将 prompt connector 输出保存到 sidecar artifact；训练端直接加载 embedding，避免再次走 tokenizer/text encoder/connector 导致 prompt enhancement 或 dtype 差异。
5. OpenAI 兼容响应保持 `b64_json` 不变；新增 `rl_rollout.artifact_uri` 和 `rl_rollout.metadata`，大 tensor 放 sidecar。
6. 训练端收到响应后，将 sidecar tensor + metadata 还原成 Flow-Factory 的 [`LTX2Sample`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py:97)，再走现有 reward、advantage、optimize 逻辑。

## 完整性检查清单

服务端返回一个 rollout sample 时，训练端应能检查：

- `all_latents.shape[0] == len(latent_index_map 中有效保存下标)`。
- `all_latents.shape[1] > video_seq_len`，确保 audio latent 已拼接在 video latent 后面。
- `log_probs` 不为空，且 `log_prob_index_map` 能覆盖所有训练 timestep。
- `connector_prompt_embeds`、`connector_audio_prompt_embeds`、`connector_attention_mask` 的 seq 维一致。
- 当 `guidance_scale > 1.0` 或 `audio_guidance_scale > 1.0` 时，negative connector 三个字段不为空。
- `height`、`width`、`num_frames`、`frame_rate` 与最终视频 metadata 一致。
- `prompt_ids` 对应实际进入 text encoder 的 prompt；如果启用了 prompt enhancement，应使用 enhanced prompt 对应的 ids。
