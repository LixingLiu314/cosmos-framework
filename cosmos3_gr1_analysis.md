# Cosmos3 GR1 训练与评估分析报告

**日期**: 2026-06-23
**训练代码**: `/root/workspace/mengya/cosmos-framework` (29D 工作副本)
**评估代码**: `/root/workspace/lixing/cosmos-framework` (已 patch 支持 `--action-schema gr1_29`)

---

## 一、背景

GR1 机器人有 44 个关节维度，其中 15 个 (left_leg, right_leg, neck) 在 RoboCasa 仿真中恒为零。新的 29D 训练代码过滤掉零值关节，仅保留 5 个活跃部位：

```
29D 布局: [left_arm(7), right_arm(7), left_hand(6), right_hand(6), waist(3)]
```

旧 44D 训练存在两个已知问题（29D 已修复）：
1. **全局 action 归一化不准确** — `gr1_lerobot_normalization.json` 范围小于部分 task 的实际 action 范围，导致 clamp 截断
2. **15 个零维度浪费** — 模型浪费容量去预测常数 -1

本文档聚焦于：(A) 新 29D 训练代码是否合理，(B) 29D 训练与评估是否存在不匹配。

---

## 二、29D 训练代码审查

### 2.1 数据流水线

```
LeRobot Parquet (44D)
  → _ordered_parts() 过滤 + 重排为 29D
  → action: per-dataset meta/stats.json minmax 归一化
  → state:  per-dataset meta/stats.json observation.state minmax 归一化
  → DataPacker: state → history_action (拼接到 action 序列第 0 帧, 标记为 conditioning)
  → ActionTransformPipeline: 视频缩放 256x256, caption 增强, 分词, 构建 SequencePlan, 填充至 64D
  → 输出: action [17, 64], raw_action_dim=29, domain_id=31
```

**审查结论**: ✅ 正确。活跃关节过滤、重排、per-dataset 归一化均工作正常。

### 2.2 归一化设计

| 归一化对象 | 29D 新训练 | 旧 44D 训练 |
|-----------|-----------|------------|
| **Action** | per-dataset `meta/stats.json` (29D 重排) | 全局 `gr1_lerobot_normalization.json` (44D) |
| **State** | per-dataset `meta/stats.json` (29D 重排) | per-dataset `meta/stats.json` (44D) |

**改进点**:
- Action 和 State 现在使用**同源统计量** (同一个 `stats.json`)，语义一致 ✅
- 每个 task 充分利用 [-1, 1] 空间，无截断问题 ✅
- 天然支持新增 task，无需重新生成全局统计文件 ✅

**旧设计的问题** (已修复，供参考):
- 全局 `normalization.json` 范围小于部分 task 的实际 action 范围 (差异高达 1.76)，导致 clamp 截断
- 不同 task 的 action 范围差异高达 4.33x，全局归一化浪费分辨率
- State 用 per-task 归一化、Action 用全局归一化，同一关节值在两个通道中的归一化数值不同

### 2.3 State 输入方式

State **没有独立的编码器**，完全复用 action 通道:

```
state (当前关节角度, 1帧)
  → per-dataset minmax 归一化
  → 重命名为 history_action
  → 拼接到 action 序列第 0 帧: [state(1), action(T)] → [T+1, 29]
  → SequencePlan 标记第 0 帧为 conditioning (不加噪、不算 loss)
  → action2llm (DomainAwareLinear, 64→hidden_size, domain_id=31) 映射到 LLM 空间
  → 与 vision/text tokens 一起做 transformer attention
```

### 2.4 模型与 Loss

| 项目 | 设置 | 审查结论 |
|------|------|---------|
| `raw_action_dim` | 从 `action.shape[1]` 动态获取 (=29) | ✅ 正确，不依赖 domain_utils 的硬编码值 |
| Loss 掩码 | `sqerr[:, :raw_action_dim]`，仅对前 29 维计算 loss | ✅ 正确 |
| Noise 零化 | `xt_action[:, raw_action_dim:] = 0`，29 维之后噪声置零 | ✅ 正确 |
| Velocity 零化 | `v[:, raw_action_dim:] = 0` | ✅ 正确 |
| `action_loss_weight` | 10.0 (action loss 相对 vision loss 放大 10 倍) | ✅ 合理 |
| `DomainAwareLinear` | input=64, output=hidden_size, domain_id=31 | ✅ 与旧 44D 共享 domain_id |
| Action heads 初始化 | 随机初始化 (`keys_to_skip_loading` 包含 action2llm 等) | ✅ 无 44D→29D 权重迁移问题 |

### 2.5 训练配置

| 配置项 | 值 | 审查结论 |
|--------|---|---------|
| `lr` | 2e-4 | ✅ |
| `lr_multipliers` | action2llm/llm2action/action_modality_embed: 5.0 | ✅ Python 默认值保留 (TOML 未覆盖) |
| `keys_to_select` | moe_gen, time_embedder, vae2llm, llm2vae, action2llm, llm2action, action_modality_embed | ✅ 选择性微调 |
| `warm_up_steps` | 0 | ⚠️ 见下文 |
| `cycle_lengths` | [60000] | ✅ |
| `max_iter` | 60000 | ✅ |
| `num_workers` | 8 (Python 默认值, TOML 未覆盖) | ✅ |
| `grad_clip` | 1.0, force_finite=true | ✅ |
| `mode` | "joint" (随机采样 forward_dynamics / inverse_dynamics / policy) | ✅ |
| `idle_frames` | 从实际 action 数据计算, action_spec 兼容 29D | ✅ |
| `compute_idle_frames` | 使用 `spec.types` 分类, 29D action_spec 自动适配 | ✅ |

### 2.6 潜在改进建议

#### ⚠️ 无学习率预热 (`warm_up_steps=0`)

Action heads 随机初始化，从 iteration 0 就使用完整学习率 (base 2e-4 × multiplier 5 = 1e-3)。加入 500-1000 步预热可能有助于稳定早期训练，但并非必须。

#### ℹ️ `domain_utils.py` 中 `EMBODIMENT_TO_RAW_ACTION_DIM["gr1_lerobot"] = 44` 未更新

该值仅在通用推理接口 (`inference/action.py`) 中使用，训练时 `raw_action_dim` 从 tensor shape 动态获取，不受影响。评估服务端已通过 `--action-schema gr1_29` 绕过。lixing 的 `domain_utils.py` 已添加 `gr1_lerobot_29d: 29` 条目。

---

## 三、29D 训练-评估等价性验证

### 3.1 文本分词: ✅ 等价

| 方面 | 训练 | 评估 |
|------|------|------|
| 函数 | `tokenize_caption()` (qwen3_vl.utils) | 同一函数 |
| 路径 | TextTokenizerTransform → BaseVLMProcessor.tokenize_text → tokenize_caption | _get_inference_text_tokens → _tokenize_captions → tokenize_caption |
| `use_system_prompt` | False | False |
| `is_video` | False | False |
| 对话模板 | `apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, add_vision_id=False)` | 相同 |

分词时机不同（训练预分词 vs 评估实时分词），但底层函数和参数完全相同。

### 3.2 Action 归一化: ✅ 等价 (使用 `--action-schema gr1_29`)

| 方面 | 训练 | 评估 |
|------|------|------|
| 统计来源 | per-dataset `meta/stats.json` action 字段, 29D 重排 | 同一 stats.json, `load_minmax_action_stats_from_dataset` 29D 重排 |
| 公式 | `(2*(x-min)/(max-min).clamp(1e-8) - 1).clamp(-1,1)` | 逆运算: `(x+1)/2 * range + min` |
| 维度 | 29 | 29 |

### 3.3 State (History Action) 归一化: ✅ 等价 (使用 `--action-schema gr1_29`)

| 方面 | 训练 | 评估 |
|------|------|------|
| 统计来源 | per-dataset `meta/stats.json` observation.state, 29D 重排 | 同一 stats.json, `load_minmax_state_stats_29d` 29D 重排 |
| 公式 | `(2*(x-min)/(max-min).clamp(1e-8) - 1).clamp(-1,1)` | 相同 |
| 维度 | 29 | 29 |

### 3.4 动作布局: ✅ 等价

| 方面 | 训练 | 评估 |
|------|------|------|
| raw_action_dim | 29 | 29 (via schema_defaults) |
| 填充后维度 | 64 | 64 |
| domain_id | 31 | 31 |
| 布局 | [left_arm(7), right_arm(7), left_hand(6), right_hand(6), waist(3)] | 相同 |

### 3.5 视频与序列规划: ✅ 等价

| 方面 | 训练 | 评估 |
|------|------|------|
| 分辨率 | 256x256 | 256x256 |
| FPS | 20 | 20 |
| 视角 | ego_view | ego_view |
| 序列规划 | policy 模式, cond_vision=[0], cond_action=[0] | 相同 |
| 历史动作 | state 前置为第 0 帧 | 相同 |

### 3.6 已知的细微差异 (预期内)

| 差异 | 训练 | 评估 | 影响 |
|------|------|------|------|
| `idle_frames` | 从实际数据计算 | 硬编码为 0 | 低 (训练有 5% dropout) |
| CFG dropout | 10% caption 替换为空 | 无 dropout, guidance=1.0 | 预期差异 |
| 模式混合 | joint (随机 FD/ID/policy) | 仅 policy | 预期差异 |

---

## 四、总结

### 29D 训练代码审查结论

| 类别 | 结论 |
|------|------|
| 数据流水线 (过滤/重排/归一化) | ✅ 正确 |
| 归一化设计 (per-dataset, state/action 同源) | ✅ 合理，优于旧 44D 设计 |
| 模型 action 通路 (noise/loss/velocity 掩码) | ✅ 正确 |
| 训练配置 (lr/scheduler/selective FT) | ✅ 合理 |
| idle_frames 兼容性 | ✅ 兼容 29D |
| 无 warmup | ⚠️ 建议加 500-1000 步 (非 bug) |

### 29D 训练-评估一致性

| 类别 | 结论 |
|------|------|
| 文本分词 | ✅ 等价 |
| Action 归一化 | ✅ 等价 |
| State 归一化 | ✅ 等价 |
| 动作布局与维度 | ✅ 等价 |
| 视频与序列规划 | ✅ 等价 |

**核心结论**: 新的 29D 训练代码设计合理，解决了旧 44D 的归一化截断和零维度浪费问题。评估代码 (使用 `--action-schema gr1_29`) 与训练严格等价，不存在 mismatch。
