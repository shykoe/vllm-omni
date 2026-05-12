# 目标

我希望使用 vllm_omni 作为 ltx2.3 t2av 的 rollout 引擎，参考 Flow-Factory LTX2.3 GRPO 的逻辑：

- [`Flow-Factory/src/flow_factory/trainers/grpo.py`](../../Flow-Factory/src/flow_factory/trainers/grpo.py)
- [`Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py`](../../Flow-Factory/src/flow_factory/models/ltx2/ltx2_t2av.py)
以及文档 `/efs/kwinsheng/vllm-omni/dev/grpo_dev/vllm_omni.md` 中的总结 使用 vllm_omni 作为 rollout 引擎,他还缺少一些输出.
而且 vllm_omni 是支持二次扩展的, 参考 `/efs/kwinsheng/vllm-omni/verl/verl-omni` 路径下的实现它使用vllm_omni 作为 qwen_image 的rollout引擎
在 `/efs/kwinsheng/vllm-omni/verl/verl-omni/verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py` 对 vllm_omni 进行了一层适配, 使得它能够输出更多的中间结果, 以供后续的 loss 计算使用.
我希望在目录`/efs/kwinsheng/vllm-omni/dev/grpo_dev`也实现一般对 ltx2.3 t2av 的 vllm_omni 的适配, 使得它能够输出更多的中间结果, 以供后续的 loss 计算使用. 以便于后续的训练和评估.
第一步先实现最小化的适配. 可以使用它对 vllm_omni 单独启动来测试输出的正确性

# 本次最小化适配的思考过程

## 1. 需求拆解

从 [`vllm_omni.md`](vllm_omni.md:55) 的字段清单可以看出, LTX2.3 T2AV GRPO rollout 不只需要最终视频和音频, 还需要训练阶段重新计算 policy loss 的中间轨迹与条件信息。关键输入包括:

- 扩散轨迹: all_latents、timesteps、log_probs、latent_index_map、log_prob_index_map。
- 条件信息: prompt_ids、positive/negative connector embeddings、attention mask。
- 生成元信息: height、width、num_frames、frame_rate、video_seq_len、audio_sample_rate、guidance_scale、noise_level、sigmas。
- 最终媒体: video 与 audio, 继续走 vLLM-Omni 原有后处理链路。

因此第一步的目标不是直接改 vLLM-Omni 核心, 而是在 [`dev/grpo_dev`](.) 下做一个可单独导入、可替换注册的最小适配层, 先验证输出字段是否足够支撑后续 loss 计算。

## 2. 参考实现对齐

参考了 [`qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`](../../verl/verl-omni/verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py:42) 的思路:

- 继承原始 pipeline, 保持原有生成行为。
- 替换 scheduler 为支持 SDE log-prob 的 scheduler。
- 在 diffusion loop 中收集 latent trajectory、log prob 和 timestep。
- 将训练需要的中间结果放到 custom_output 中。

同时也参考了已有的 [`ltx2_3_t2av_flow_grpo/vllm_omni_rollout_adapter.py`](../../verl/verl-omni/verl_omni/pipelines/ltx2_3_t2av_flow_grpo/vllm_omni_rollout_adapter.py:58), 将其改造成不依赖 verl_omni 注册系统的本地 dev 版本。

## 3. 适配方式选择

最终采用“替换注册”而不是修改 vLLM-Omni 内置 [`LTX23Pipeline`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:107) 的方式:

1. vLLM-Omni 使用 [`DiffusionModelRegistry`](../../vllm_omni/diffusion/registry.py:261) 按 model_class_name 查找 pipeline。
2. 通过 [`register_diffusion_model()`](../../vllm_omni/diffusion/registry.py:502) 可以把同一个架构名 LTX23Pipeline 重定向到自定义类。
3. [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1) 作为以 [`dev/grpo_dev`](.) 为运行目录的独立模块使用；离线测试在 [`dev/grpo_dev`](.) 下执行 [`python -m ltx2_3_t2av.test_rollout`](ltx2_3_t2av/test_rollout.py:1), 模块导入后再创建 Omni 即可。
4. 不侵入主干实现, 便于后续删除、替换或迁移到正式训练目录。

## 4. 轨迹收集语义

在 [`LTX23T2AVPipelineWithLogProb.diffuse_with_log_prob()`](ltx2_3_t2av/pipeline.py:160) 中, video latent 和 audio latent 被拼接为 unified latent:

- video latent 在前。
- audio latent 在后。
- video_seq_len 记录切分点, 供训练端恢复 video/audio latent。

SDE window 采用紧凑保存语义:

- all_latents 保存 compact window 内的 latent。
- all_log_probs 保存对应 transition 的 log probability。
- latent_index_map 和 log_prob_index_map 用于把原始 train timestep index 映射到 compact tensor index。

这和 Flow-Factory 中训练端通过 current latent、next latent、old log prob 重算 PPO/GRPO ratio 的方式保持一致。

## 5. 最小化范围取舍

本次实现只做最小闭环:

- 大 tensor 暂时直接通过 in-memory custom_output 返回。
- rl_rollout 中的 tensor_format 标记为 in_memory。
- 暂未实现 safetensors sidecar artifact。
- 暂未接入 HTTP response 协议层的 artifact_uri。
- 暂未实现训练端 LTX2Sample 还原逻辑。

这样可以先快速验证 vLLM-Omni 单独 rollout 时是否能输出训练所需字段。后续生产化再把大 tensor 下沉到 sidecar 文件, 并在服务响应中返回 artifact_uri。

# 修改历史

## 2026-05-09: 创建 LTX2.3 T2AV 最小 rollout adapter

新增目录 [`ltx2_3_t2av`](ltx2_3_t2av/):

- [`ltx2_3_t2av/__init__.py`](ltx2_3_t2av/__init__.py:1): 包入口。导入时自动调用注册函数, 将 LTX23Pipeline 替换为本地适配器。
- [`ltx2_3_t2av/register.py`](ltx2_3_t2av/register.py:8): 定义 [`python.register_rollout_adapter()`](ltx2_3_t2av/register.py:8), 调用 vLLM-Omni 的 [`python.register_diffusion_model()`](../../vllm_omni/diffusion/registry.py:502) 完成替换注册。
- [`ltx2_3_t2av/schedulers.py`](ltx2_3_t2av/schedulers.py:48): 定义 [`python.FlowMatchSDEDiscreteScheduler`](ltx2_3_t2av/schedulers.py:48), 从 FlowGRPO/verl-omni 的 SDE scheduler 逻辑本地化而来, 支持返回 log_prob。
- [`ltx2_3_t2av/pipeline.py`](ltx2_3_t2av/pipeline.py:57): 定义 [`python.LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:57), 继承 vLLM-Omni 内置 [`python.LTX23Pipeline`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:107)。
- [`ltx2_3_t2av/test_rollout.py`](ltx2_3_t2av/test_rollout.py:1): 离线 smoke test, 可单独启动 Omni 并打印 rollout 字段 shape。
- [`ltx2_3_t2av/README.md`](ltx2_3_t2av/README.md:1): 使用说明、字段说明和当前范围说明。

## pipeline 主要改动

[`python.LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:57) 相对原始 [`python.LTX23Pipeline`](../../vllm_omni/diffusion/models/ltx2/pipeline_ltx2_3.py:107) 的核心变化:

1. 初始化时将 scheduler 替换为 [`python.FlowMatchSDEDiscreteScheduler`](ltx2_3_t2av/schedulers.py:48)。
2. 新增 [`python.LTX23T2AVPipelineWithLogProb._get_prompt_ids()`](ltx2_3_t2av/pipeline.py:84), 返回实际 tokenizer 输入 token ids。
3. 新增 [`python.LTX23T2AVPipelineWithLogProb._build_index_map()`](ltx2_3_t2av/pipeline.py:132), 构造 train timestep 到 compact trajectory tensor 的映射。
4. 新增 [`python.LTX23T2AVPipelineWithLogProb._split_connector_outputs()`](ltx2_3_t2av/pipeline.py:141), 将 CFG 下 concatenated connector output 拆成 positive 与 negative 两份。
5. 新增 [`python.LTX23T2AVPipelineWithLogProb.diffuse_with_log_prob()`](ltx2_3_t2av/pipeline.py:160), 在 denoising loop 内收集 unified video/audio latents、log_probs、timesteps 与 index maps。
6. 在 forward 返回的 [`python.DiffusionOutput`](../../vllm_omni/diffusion/data.py:822) 中写入 custom_output 和 trajectory_* 字段。

## 输出字段

当前 [`python.DiffusionOutput.custom_output`](../../vllm_omni/diffusion/data.py:842) 包含:

- all_latents
- all_log_probs
- log_probs
- all_timesteps
- timesteps
- latent_index_map
- log_prob_index_map
- prompt_ids
- connector_prompt_embeds
- connector_audio_prompt_embeds
- connector_attention_mask
- negative_connector_prompt_embeds
- negative_connector_audio_prompt_embeds
- negative_connector_attention_mask
- metadata
- rl_rollout

metadata 中包含 prompt、height、width、num_frames、frame_rate、video_seq_len、duration_s、audio_sample_rate、num_inference_steps、guidance_scale、noise_scale、noise_level、sde_window、sde_type、sigmas、max_sequence_length。

# 验证记录

## 语法检查

已执行 Python 语法检查, 覆盖以下文件:

- [`ltx2_3_t2av/__init__.py`](ltx2_3_t2av/__init__.py:1)
- [`ltx2_3_t2av/schedulers.py`](ltx2_3_t2av/schedulers.py:1)
- [`ltx2_3_t2av/register.py`](ltx2_3_t2av/register.py:1)
- [`ltx2_3_t2av/pipeline.py`](ltx2_3_t2av/pipeline.py:1)
- [`ltx2_3_t2av/test_rollout.py`](ltx2_3_t2av/test_rollout.py:1)

检查结果: 通过。

## 注册验证

已执行轻量导入/注册验证:

1. 在 [`dev/grpo_dev`](.) 目录下导入独立模块 [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1)。
2. 从 [`DiffusionModelRegistry`](../../vllm_omni/diffusion/registry.py:261) 查询 LTX23Pipeline。
3. 确认解析到 [`ltx2_3_t2av.pipeline.LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:57)。

检查结果: 通过。

## 离线测试命令

可用以下命令测试输出字段正确性:

```bash
cd /efs/kwinsheng/vllm-omni/dev/grpo_dev && \
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

该脚本会:

1. 通过 [`python -m ltx2_3_t2av.test_rollout`](ltx2_3_t2av/test_rollout.py:1) 导入独立模块 [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1) 触发注册。
2. 使用 [`python.Omni`](../../vllm_omni/entrypoints/omni.py:1) 离线生成一次 LTX2.3 T2AV。
3. 打印 custom_output 中关键 rollout 字段 shape。
4. 检查 all_latents、timesteps、latent_index_map、prompt_ids、connector_prompt_embeds、metadata 是否存在。
5. 在 logprobs 启用时检查 all_log_probs 是否存在。

## Online serving 启动命令

### 推荐方式: 使用真实 Python 入口文件启动

在线 serving 是新进程, 必须保证模型构造前已经导入 [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1), 这样 [`register_rollout_adapter()`](ltx2_3_t2av/register.py:11) 才会把内置 LTX23Pipeline 替换为 [`LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:60)。当前提供了真实入口文件 [`serve_ltx2_3_t2av.py`](serve_ltx2_3_t2av.py:1) 和启动脚本 [`serving.sh`](serving.sh:1):

```bash
cd /efs/kwinsheng/vllm-omni/dev/grpo_dev
sh serving.sh > serving.log 2>&1
```

[`serving.sh`](serving.sh:1) 的等价关键配置为:

```bash
mkdir -p /efs/kwinsheng/vllm-omni/dev/output
export VLLM_OMNI_STORAGE_PATH=/efs/kwinsheng/vllm-omni/dev/output
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH=/efs/kwinsheng/vllm-omni/dev/grpo_dev:${PYTHONPATH:-}
python /efs/kwinsheng/vllm-omni/dev/grpo_dev/serve_ltx2_3_t2av.py
```

[`serve_ltx2_3_t2av.py`](serve_ltx2_3_t2av.py:1) 会先导入 [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1), 再调用 [`main()`](../../vllm_omni/entrypoints/cli/main.py:9), 等价于执行:

```bash
vllm serve dg845/LTX-2.3-Diffusers \
  --omni \
  --host 0.0.0.0 \
  --port 8098 \
  --model-class-name LTX23Pipeline \
  --data-parallel-size 8
```

这个启动方式的关键点:

1. [`PYTHONPATH`](serving.sh:9) 指向 [`/efs/kwinsheng/vllm-omni/dev/grpo_dev`](.), 使 [`ltx2_3_t2av`](ltx2_3_t2av/__init__.py:1) 可以作为独立模块被导入。
2. [`import ltx2_3_t2av`](serve_ltx2_3_t2av.py:19) 必须发生在 [`main()`](../../vllm_omni/entrypoints/cli/main.py:9) 构造 diffusion engine 之前。
3. [`--model-class-name LTX23Pipeline`](serve_ltx2_3_t2av.py:43) 保持不变, 但注册表中的 LTX23Pipeline 已经被替换到 [`ltx2_3_t2av.pipeline.LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:60)。
4. 不要使用 `python - <<'PY'` 或 `python -c` 作为 serving 主入口。vLLM-Omni worker 使用 Python multiprocessing spawn, 子进程需要从真实文件路径重新加载父进程 main module; 如果父入口来自 stdin, 子进程会尝试加载 `/efs/kwinsheng/vllm-omni/dev/grpo_dev/<stdin>` 并触发 `FileNotFoundError`。

可选环境变量:

```bash
export LTX23_T2AV_MODEL=dg845/LTX-2.3-Diffusers
export LTX23_T2AV_HOST=0.0.0.0
export LTX23_T2AV_PORT=8098
export LTX23_T2AV_DP=8
```

### 请求示例: 普通视频接口

服务启动后, 仍然可以使用原来的 video sync 接口测试最终媒体输出:

```bash
curl -sS -X POST http://localhost:8098/v1/videos/sync \
  -F "prompt=A cinematic close-up of ocean waves at golden hour" \
  -F "width=1280" \
  -F "height=736" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=4.0" \
  -F "seed=41" \
  -o "ltx2_3_t2av_online_smoke.mp4"
```

注意: 当前 OpenAI 兼容视频接口主要返回最终 mp4, 不一定会把 in-memory [`custom_output`](../../vllm_omni/diffusion/data.py:842) 序列化到 HTTP 响应中。因此这个请求适合验证 online serving 是否成功使用适配后的 pipeline 生成视频/音频。

### 训练 rollout 使用建议

如果 online rollout 训练端需要拿到 all_latents、log_probs、connector embeddings 等中间结果, 下一步应实现 sidecar artifact:

1. 在 [`LTX23T2AVPipelineWithLogProb`](ltx2_3_t2av/pipeline.py:57) 中将大 tensor 写入 safetensors 或 torch 文件。
2. 在 `custom_output["rl_rollout"]` 中补充 `artifact_uri`。
3. 扩展服务响应层, 将 `rl_rollout.artifact_uri` 和 metadata 序列化给训练端。
4. 训练端根据 artifact_uri 加载 tensor, 还原 Flow-Factory LTX2Sample 等价结构。

# 后续计划

1. 将 in-memory 大 tensor 输出改为 safetensors sidecar artifact。
2. 在 rl_rollout 中补充 artifact_uri, 避免大 tensor 进入服务响应对象。
3. 编写训练侧 loader, 将 custom_output 或 artifact_uri 还原为 Flow-Factory LTX2Sample 等价结构。
4. 对齐 Flow-Factory 的 trajectory window 采样策略, 确认 index map 与训练 timestep 完全一致。
5. 增加多 prompt、多 output、不同 CFG scale 和不同 SDE window 的测试覆盖。