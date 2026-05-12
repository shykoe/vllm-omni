为了使用vllm_omni 作为 verl grpo rollout 的引擎.需要对LTX-2.3-Diffusers模型进行一些适配. 这个目录(/efs/kwinsheng/vllm-omni/dev/grpo_dev)下的代码实现了一个新的pipeline, 他可以产出 grpo训练需要的rollout数据
目录 `/efs/kwinsheng/vllm-omni/verl/verl-omni` 是verl官方使用 vllm-omni 进行rollout的代码.他只实现了 qwen_image 模型的 grpo 示例. 我希望基于这个代码, 实现 LTX-2.3-Diffusers 模型的 grpo rollout.
1. 所有实现代码都写到 /efs/kwinsheng/vllm-omni/verl/verl-omni 目录.
2. 尽量不要修改 verl-omni 的框架架构
3. 第一版本只实现 ltx2.3 的 grpo rollout, 不需要实现 reward 和训练的逻辑.
4. 我希望在 /efs/kwinsheng/vllm-omni/verl/verl-omni/scripts 生成一个可以直接运行的脚本.
5. 为了测试grpo rollout 的结果可以将 `/efs/kwinsheng/vllm-omni/verl/verl-omni/verl_omni/trainer/diffusion/ray_diffusion_trainer.py` grpo的部分逻辑先注释掉, 只执行 rollout 部分.

---

# LTX-2.3-Diffusers 接入 verl-omni rollout 的设计与修改记录

## 1. 目标与边界

本次实现的目标是将 `dev/grpo_dev/ltx2_3_t2av` 中已经验证过的 LTX-2.3 text-to-audio-video rollout pipeline 迁移并接入到 `verl/verl-omni` 中，使其可以作为 verl Flow-GRPO 的 vLLM-Omni rollout 引擎使用。

第一版本的边界如下：

1. 只实现 LTX-2.3 的 rollout 数据采集。
2. 不实现 reward 计算。
3. 不实现 actor 更新、advantage 计算、old/ref log-prob 重新计算等训练逻辑。
4. 尽量复用 verl-omni 现有 qwen_image 的 rollout 框架，不大改框架结构。
5. 在 `verl/verl-omni/scripts` 下提供一个可直接运行的 rollout-only smoke 脚本。

## 2. 原有架构梳理

### 2.1 qwen_image 在 verl-omni 中的接入方式

verl-omni 现有 qwen_image 示例主要分为两类 adapter：

1. vLLM-Omni rollout 侧 pipeline adapter：
   - `verl/verl-omni/verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`
   - 通过 `VllmOmniPipelineBase.register("QwenImagePipeline")` 注册。
   - vLLM-Omni server 启动时会从 `VllmOmniPipelineBase.get_pipeline_path()` 读取 custom pipeline 路径，并传给 `AsyncOmni`。
   - pipeline forward 返回 `DiffusionOutput.custom_output`，其中包含 `all_latents`、`all_log_probs`、`all_timesteps`、prompt embeds 等 GRPO 需要的数据。

2. training-side diffusers adapter：
   - `verl/verl-omni/verl_omni/pipelines/qwen_image_flow_grpo/diffusers_training_adapter.py`
   - 通过 `DiffusionModelBase.register("QwenImagePipeline")` 注册。
   - 供 actor/ref log-prob recompute 和 actor update 阶段使用。

LTX-2.3 第一版只需要 rollout，因此只需要完整实现 rollout-side adapter；training-side adapter 只保留 registry shim，避免框架初始化阶段找不到 `LTX23Pipeline`。

### 2.2 vLLM-Omni rollout server 的关键路径

关键文件：`verl/verl-omni/verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py`

现有逻辑：

1. `vLLMOmniHttpServer.run_server()` 根据模型架构从 `VllmOmniPipelineBase` 查找 custom pipeline。
2. 如果找到 custom pipeline，则传入 `engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}`。
3. `vLLMOmniHttpServer.generate()` 将 agent loop 传入的 `prompt_ids`、`negative_prompt_ids` 封装为 `OmniCustomPrompt`。
4. 调用 `AsyncOmni.generate()`。
5. 从 `final_res.custom_output` 解析 rollout artifacts，返回给 verl 的 agent loop。

qwen_image 只依赖 token ids，而 LTX-2.3 自带 Gemma text encoder，需要原始文本 prompt，因此需要扩展 agent loop 和 server，使其同时传递 token ids 与 plain text。

### 2.3 trainer 训练循环的关键位置

关键文件：`verl/verl-omni/verl_omni/trainer/diffusion/ray_diffusion_trainer.py`

原有 `fit()` 的核心步骤：

1. dataloader 取 batch。
2. agent loop + vLLM-Omni 生成 rollout。
3. 合并原 batch 与 rollout batch。
4. reward 计算。
5. old log-prob recompute。
6. ref log-prob recompute。
7. advantage 计算。
8. actor update。
9. checkpoint/save/validate/log。

第一版本需要在第 3 步之后直接返回，即只验证 rollout 数据能被生成并汇总进 batch。

## 3. 设计思路

### 3.1 最小侵入原则

本次实现没有改动 verl-omni 的总体分层：

1. 仍然使用 `VllmOmniPipelineBase` 管理 custom rollout pipeline。
2. 仍然使用现有 `AgentLoopManager`、`DiffusionAgentLoopWorker`、`vLLMOmniHttpServer`。
3. 仍然走 `main_flowgrpo -> RayFlowGRPOTrainer -> ActorRolloutRefWorker -> rollout server` 的启动路径。
4. 对训练逻辑不做假实现，只增加显式的 `rollout_only` 开关。

### 3.2 LTX-2.3 rollout pipeline 的迁移方式

从 `dev/grpo_dev/ltx2_3_t2av/pipeline.py` 迁移到：

- `verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo/vllm_omni_rollout_adapter.py`

主要调整：

1. scheduler 改为复用 verl-omni 已有的 `verl_omni.pipelines.schedulers.FlowMatchSDEDiscreteScheduler`。
2. 增加 `@VllmOmniPipelineBase.register("LTX23Pipeline")`。
3. 保留 `get_ltx2_post_process_func`，用于 LTX-2.3 video+audio 后处理。
4. 保留 rollout 输出字段：
   - `all_latents`
   - `all_log_probs`
   - `log_probs`
   - `all_timesteps`
   - `timesteps`
   - `latent_index_map`
   - `log_prob_index_map`
   - `prompt_ids`
   - `connector_prompt_embeds`
   - `connector_audio_prompt_embeds`
   - `connector_attention_mask`
   - `negative_connector_prompt_embeds`
   - `negative_connector_audio_prompt_embeds`
   - `negative_connector_attention_mask`
   - `metadata`
   - `rl_rollout`

这些字段是后续实现 LTX-2.3 GRPO training/recompute log-prob 的基础。

### 3.3 training-side registry shim

新增：

- `verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo/diffusers_training_adapter.py`

其中注册：

- `@DiffusionModelBase.register("LTX23Pipeline")`

当前只实现 scheduler 构建与 timesteps 配置；`prepare_model_inputs()` 与 `forward_and_sample_previous_step()` 显式抛出 `NotImplementedError`。

原因：

1. 第一版本不进入训练和 log-prob recompute。
2. 仍需要让框架识别 `LTX23Pipeline`，避免初始化阶段因 registry 缺失失败。
3. 后续实现训练逻辑时可以直接在该 adapter 中补充 LTX-2.3 transformer forward 输入构造与 scheduler sampling。

### 3.4 raw prompt 传递设计

qwen_image rollout adapter 使用 token ids 生成 prompt embeds；LTX-2.3 使用自己的 tokenizer/text encoder，需要原始文本。

因此新增：

- `verl/verl-omni/verl_omni/agent_loop/single_turn_agent_loop.py` 中的 `_messages_to_plain_text()`
- `vLLMOmniHttpServer.generate()` 新增参数：
  - `prompt_text`
  - `negative_prompt_text`

server 构造 `OmniCustomPrompt` 时同时带上：

- `prompt_ids`
- `prompt`
- `negative_prompt_ids`
- `negative_prompt`

这样不会破坏 qwen_image 的 token-id 路径，同时 LTX-2.3 可以读取 raw prompt 字段。

### 3.5 video/audio output 处理设计

原 server 假设 `final_res.images[0]` 是 PIL image，并用 `PILToTensor()` 转 tensor。

LTX-2.3 返回 text-to-audio-video：

1. video 可能是 tensor、numpy array 或 frame list。
2. audio 会通过 `multimodal_output` 返回。
3. `final_res.images[0]` 不一定是 PIL image。

因此新增 `_as_diffusion_tensor()`：

1. PIL image 继续用 `PILToTensor()`。
2. tensor 直接 detach/cpu。
3. numpy array 用 `torch.from_numpy()`。
4. list 按 frame stack。
5. tuple 默认取第一个元素作为 video。
6. 大于 1 的像素值会除以 255 归一化。

### 3.6 rollout-only 开关设计

新增两个层面的开关：

1. `actor_rollout_ref.rollout.rollout_only=True`
   - 配置定义在 `verl/verl-omni/verl_omni/workers/config/diffusion/rollout.py`
   - 作用于 worker 初始化和 update_weights。

2. `trainer.rollout_only=True`
   - 作用于 trainer `fit()`。
   - rollout 生成并合并 batch 后直接返回。

为什么需要两个开关：

- rollout config 的开关用于底层 worker：避免初始化 actor/ref model 和 checkpoint engine。
- trainer config 的开关用于上层训练循环：避免进入 reward、advantage、update_actor。

在 `ActorRolloutRefWorker.init_model()` 中：

1. 如果 `rollout_only=True`，跳过 ref model 初始化。
2. 跳过 actor model 初始化。
3. 只启动 rollout engine。
4. 不创建 checkpoint engine。

在 `ActorRolloutRefWorker.update_weights()` 中：

1. 如果 `rollout_only=True`，只 resume rollout 的 weights/kv_cache。
2. 不从 actor engine 同步权重，因为 actor engine 未初始化。

在 `RayFlowGRPOTrainer.fit()` 中：

1. 如果 `rollout_only=True`，不加载 checkpoint。
2. rollout batch 生成后打印 batch 字段和 shape。
3. 直接 `return`。

## 4. 修改记录

### 4.1 新增文件

#### `verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo/vllm_omni_rollout_adapter.py`

作用：LTX-2.3 vLLM-Omni rollout adapter。

来源：基于 `dev/grpo_dev/ltx2_3_t2av/pipeline.py` 迁移。

关键点：

1. 注册 `LTX23Pipeline` 到 `VllmOmniPipelineBase`。
2. 使用 SDE scheduler 进行 rollout。
3. 收集 video+audio latent trajectory 和 log-prob。
4. 输出 GRPO 所需 custom_output。

#### `verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo/diffusers_training_adapter.py`

作用：LTX-2.3 training-side registry shim。

关键点：

1. 注册 `LTX23Pipeline` 到 `DiffusionModelBase`。
2. 提供 scheduler/timesteps 初始化。
3. 明确不实现训练/recompute log-prob。

#### `verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo/__init__.py`

作用：导出 LTX-2.3 adapter，并触发 registry side effects。

#### `verl/verl-omni/scripts/run_ltx23_rollout_only.sh`

作用：可直接运行的 LTX-2.3 rollout-only smoke 脚本。

功能：

1. 自动生成最小 parquet 数据集。
2. 设置 `actor_rollout_ref.model.architecture=LTX23Pipeline`。
3. 设置 `actor_rollout_ref.rollout.rollout_only=True`。
4. 设置 `trainer.rollout_only=True`。
5. 只运行 1 个 rollout step。

### 4.2 修改文件

#### `verl/verl-omni/verl_omni/pipelines/__init__.py`

修改：

1. import `ltx2_3_flow_grpo`。
2. 将 LTX-2.3 adapter 加入 `__all__`。

目的：verl-omni 启动时自动完成 LTX-2.3 registry 注册。

#### `verl/verl-omni/verl_omni/agent_loop/single_turn_agent_loop.py`

修改：

1. 新增 `_messages_to_plain_text()`。
2. 调用 server generate 时传入 `prompt_text` 和 `negative_prompt_text`。

目的：保留 qwen_image 的 token-id 输入方式，同时支持 LTX-2.3 raw text prompt。

#### `verl/verl-omni/verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py`

修改：

1. 新增 `_as_diffusion_tensor()`。
2. `generate()` 增加 `prompt_text`、`negative_prompt_text` 参数。
3. `OmniCustomPrompt` 中加入 raw `prompt` 和 `negative_prompt`。
4. 支持非 PIL 的 video/audio 输出转 tensor。
5. 从 `custom_output` 透传 LTX-2.3 rollout artifacts。

#### `verl/verl-omni/verl_omni/workers/config/diffusion/rollout.py`

修改：

1. `DiffusionPipelineConfig` 新增：
   - `num_frames`
   - `frame_rate`
2. `DiffusionRolloutConfig` 新增：
   - `rollout_only`

目的：支持视频参数配置，以及第一阶段只启动 rollout。

#### `verl/verl-omni/verl_omni/workers/engine_workers.py`

修改：

1. `init_model()` 识别 rollout-only 模式。
2. rollout-only 下跳过 actor/ref model。
3. rollout-only 下跳过 checkpoint engine。
4. `update_weights()` 在 rollout-only 下不访问 actor engine。

#### `verl/verl-omni/verl_omni/trainer/diffusion/ray_diffusion_trainer.py`

修改：

1. `fit()` 中识别 `trainer.rollout_only` 或 `actor_rollout_ref.rollout.rollout_only`。
2. rollout-only 下不加载 checkpoint。
3. rollout 生成后打印 batch tensor fields、non-tensor fields、关键 shape。
4. 直接 return，跳过后续 GRPO 训练逻辑。

## 5. 运行方式

默认运行：

```bash
cd /efs/kwinsheng/vllm-omni/verl/verl-omni
./scripts/run_ltx23_rollout_only.sh
```

指定本地模型路径：

```bash
cd /efs/kwinsheng/vllm-omni/verl/verl-omni
MODEL_PATH=/efs/kwinsheng/vllm-omni/dev/models/LTX-2.3-Diffusers ./scripts/run_ltx23_rollout_only.sh
```

常用环境变量：

```bash
MODEL_PATH=/efs/kwinsheng/vllm-omni/dev/models/LTX-2.3-Diffusers
NUM_GPUS=1
ROLLOUT_TP=1
HEIGHT=512
WIDTH=768
NUM_FRAMES=33
FRAME_RATE=24.0
NUM_INFERENCE_STEPS=4
GUIDANCE_SCALE=4.0
SDE_WINDOW_SIZE=2
SDE_WINDOW_RANGE='[0,2]'
```

## 6. 验证结果

已执行语法检查：

```bash
python3 -m compileall -q \
  verl/verl-omni/verl_omni/pipelines/ltx2_3_flow_grpo \
  verl/verl-omni/verl_omni/pipelines/__init__.py \
  verl/verl-omni/verl_omni/agent_loop/single_turn_agent_loop.py \
  verl/verl-omni/verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py \
  verl/verl-omni/verl_omni/workers/config/diffusion/rollout.py \
  verl/verl-omni/verl_omni/workers/engine_workers.py \
  verl/verl-omni/verl_omni/trainer/diffusion/ray_diffusion_trainer.py
```

结果：退出码为 0。

同时已执行：

```bash
chmod +x verl/verl-omni/scripts/run_ltx23_rollout_only.sh
```

## 7. 思考过程与关键取舍

### 7.1 为什么不直接修改 vLLM-Omni registry

开发版 `dev/grpo_dev/ltx2_3_t2av/register.py` 通过替换 vLLM-Omni `DiffusionModelRegistry` 中的 `LTX23Pipeline` 来生效。

verl-omni 已有更适合的 custom pipeline 机制：

1. `VllmOmniPipelineBase.register()` 管理模型架构到 pipeline class 的映射。
2. rollout server 启动时通过 `custom_pipeline_args` 注入 custom pipeline。
3. 不需要改 vLLM-Omni 全局 registry。

因此最终选择接入 verl-omni 的 pipeline registry，而不是复用独立 register.py。

### 7.2 为什么要增加 training-side shim

即使第一版不训练，verl-omni 初始化路径仍会读取模型 architecture，并在一些训练侧路径中依赖 `DiffusionModelBase` registry。

如果完全不注册 `LTX23Pipeline`，后续进入某些初始化或 scheduler 构造路径时会直接报错。

因此增加最小 shim：

1. 让 registry 完整。
2. 保持第一版边界清晰。
3. 训练逻辑未实现时显式 `NotImplementedError`，避免静默产出错误训练结果。

### 7.3 为什么不把 reward 逻辑做成 dummy

用户目标明确是第一版只测试 rollout。

如果加入 dummy reward，容易造成两个问题：

1. 后续误以为训练链路已经正确。
2. reward/advantage/update 可能会消费不完整的 LTX-2.3 fields，导致调试复杂化。

因此选择在 rollout batch 生成后直接停止，而不是模拟 reward。

### 7.4 为什么新增 rollout-only 而不是注释代码

需求中提到可以先注释 GRPO 部分逻辑。实际实现中使用显式配置开关代替注释：

1. 更容易回滚。
2. 不影响 qwen_image 正常训练路径。
3. 脚本中打开开关即可测试 LTX-2.3 rollout。
4. 后续实现训练逻辑时只需要关闭开关。

### 7.5 当前实现的限制

1. 只验证 rollout 数据生成链路。
2. LTX-2.3 的 actor/ref log-prob recompute 尚未实现。
3. LTX-2.3 的 actor update 尚未实现。
4. `all_latents` 将 video/audio latents concat 到一起，后续训练侧需要根据 `video_seq_len` 或 metadata 拆分。
5. 当前脚本默认使用较小 `NUM_INFERENCE_STEPS=4` 和 `NUM_FRAMES=33` 作为 smoke test 参数，正式训练需调整。
6. LTX-2.3 22B 模型显存压力较大，实际运行可能需要结合 CPU offload、TP、较小分辨率/帧数等参数进一步调优。

## 8. 后续工作建议

1. 实现 LTX-2.3 training-side `prepare_model_inputs()`：
   - 读取 `connector_prompt_embeds`
   - 读取 `connector_audio_prompt_embeds`
   - 读取 `connector_attention_mask`
   - 根据 `video_seq_len` 拆分 video/audio latents
   - 构造 transformer 所需 `video_coords`、`audio_coords`、fps、audio frame 信息

2. 实现 `forward_and_sample_previous_step()`：
   - 对 video/audio 分别 forward transformer 输出 noise prediction
   - 应用 CFG
   - 使用 SDE scheduler 对 video latent 计算 log-prob
   - 对 audio latent 可按当前 rollout 策略不计 log-prob或合并计入

3. 将 rollout metadata 标准化：
   - `video_seq_len`
   - latent shape
   - audio latent shape
   - scheduler sigmas
   - sde window

4. 增加 LTX-2.3 专用 reward/validation dump：
   - 保存 mp4
   - 保存 wav/audio
   - 保存 rollout tensor shape summary

5. 在 smoke test 通过后关闭 `rollout_only`，逐步接入：
   - reward
   - old log-prob recompute
   - advantage
   - actor update