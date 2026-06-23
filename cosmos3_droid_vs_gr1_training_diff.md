# DROID vs GR1 训练流水线差异分析

**日期**: 2026-06-23
**目的**: 对比 DROID（官方已验证收敛）与 GR1 的训练配置，寻找 GR1 效果不佳的潜在原因
**DROID 代码**: `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py`
**GR1 代码**: mengya 目录 `cosmos_framework/configs/base/experiment/action/posttrain_config/gr1_robot_policy_posttrain.py`

---

## 一、差异总览

| # | 差异项 | DROID (已验证) | GR1 (当前) | 严重程度 |
|---|--------|---------------|------------|----------|
| 1 | vision `loss_scale` | 10.0 | 1.0 (默认) | 高 |
| 2 | `max_num_tokens_after_packing` | -1 (无上限) | 45056 (默认) | 高 |
| 3 | `encode_exact_durations` | [33] | None (默认) | 高 |
| 4 | 训练模式 `mode` | "policy" | "joint" | 中 |
| 5 | DataLoader / episode shuffle | episode-level shuffle + rank 分片 | 简单 shuffle=True | 中 |
| 6 | image augmentation | 有 (crop + jitter) | 无 | 中 |
| 7 | `chunk_length` | 32 | 16 | 低 |
| 8 | action 归一化 | None (raw joint_pos) | minmax (per-dataset stats.json) | 信息项 |
| 9 | `use_state` 处理 | dataset 内拼接 | DataPacker 中拼接 | 等价 |

---

## 二、逐项详解

### 2.1 [高] vision loss_scale 未设置 — vision loss 被 action loss 压制

**DROID 设置** (`action_policy_droid_nano.py:243`):
```python
action_policy_droid_nano["model"]["config"]["rectified_flow_training_config"]["loss_scale"] = 10.0
```

**DROID 注释原文**:
> Weight the vision flow-matching loss 10x in the total loss (the NANO default is 1.0).
> loss_scale multiplies only the vision term, balancing it against the action loss
> (action_loss_weight=10) so both heads train at comparable gradient magnitude.

**GR1 设置**: 无显式设置，使用 `nano_model_config.py` 默认值 `loss_scale=1.0`

**影响分析**:
- `action_loss_weight=10.0` 将 action loss 放大 10 倍
- 如果 vision `loss_scale` 不同步设为 10.0，vision loss 相对 action loss 弱 10 倍
- 视觉生成头得不到足够梯度 → 视觉特征质量差 → action head 依赖的视觉表征也差
- 这是 DROID 官方明确标注的必要配置，GR1 缺失最可能是遗漏

**默认值参考** (`nano_model_config.py`):
```python
rectified_flow_training_config=dict(
    action_loss_weight=10.0,   # action loss 放大 10x
    loss_scale=1.0,            # vision loss 默认 1x → 不平衡
)
```

**修复建议**: 在 GR1 TOML 或 Python config 中添加:
```toml
[model.rectified_flow_training_config]
loss_scale = 10.0
```

---

### 2.2 [高] max_num_tokens_after_packing 截断长序列

**DROID 设置** (`action_policy_droid_nano.py:237`):
```python
action_policy_droid_nano["model"]["config"]["max_num_tokens_after_packing"] = -1
```

**DROID 注释原文**:
> Uncap the packed-sequence length. The NANO default (45056) caps the packed sequence,
> truncating long DROID windows to ~1/4 of their natural length; -1 (uncapped) processes
> the full vision sequence per step. Does not change the per-token loss; widens the
> effective vision context per step.

**GR1 设置**: TOML 中显式写了 `max_num_tokens_after_packing = 45056`

**影响分析**:
- 需要实际计算 GR1 的 token 数来判断是否被截断
- GR1: 17 帧 × 256×256，VAE 压缩 4×16×16，patch 2×2:
  - latent_t = 1 + (17-1)/4 = 5
  - latent_h = 256/(16×2) = 8
  - latent_w = 256/(16×2) = 8
  - vision tokens ≈ 5 × 8 × 8 = 320 + text + action ≈ 远小于 45056
- **结论**: 以 GR1 的 256×256 / 17 帧规模，45056 的 cap 大概率不会被触发。但如果未来增大分辨率或 chunk_length，这将成为问题。

**修复建议**: 建议设为 -1 以消除隐患，但**当前可能不是效果差的直接原因**。

---

### 2.3 [高] encode_exact_durations 未固定

**DROID 设置** (`action_policy_droid_nano.py:230`):
```python
action_policy_droid_nano["model"]["config"]["tokenizer"]["encode_exact_durations"] = [33]
```

**DROID 注释原文**:
> chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match.

**GR1 设置**: 无显式设置，使用 `nano_model_config.py` 默认值 `encode_exact_durations=None`

**影响分析**:
- 当 `encode_exact_durations=None` 时，VAE 对输入帧数不做约束，可能在不同样本间产生不一致的编码行为
- DROID 明确将其固定为训练帧数 (chunk_length + 1 = 33)
- GR1 的 chunk_length=16，对应 17 帧观测，应设为 [17]

**修复建议**:
```toml
[model.tokenizer]
encode_exact_durations = [17]
```

---

### 2.4 [中] 训练模式: joint vs policy

**DROID 设置**: `mode="policy"`
**GR1 设置**: `mode="joint"` (在 Python config 和 dataset 构造中)

**DROID 设计解释**:
```python
# Policy-only task mode. "joint" would randomly pick
# forward_dynamics/inverse_dynamics/policy per sample (multi-task),
# which dilutes each per-task loss by ~1/3.
mode="policy",
```

**`joint` 模式行为** (来自 `_choose_mode()`):
```python
def _choose_mode(self) -> str:
    if self._mode == "joint":
        return random.choice(("forward_dynamics", "inverse_dynamics", "policy"))
    return self._mode
```

**影响分析**:
- `joint` 模式下每个样本 1/3 概率走 policy → action head 有效训练数据量降为 1/3
- 三种模式的 conditioning 方式不同:
  - `policy`: condition on 首帧 + 文本 → 预测未来视频 + action (目标场景)
  - `forward_dynamics`: condition on 首帧 + action → 预测未来视频
  - `inverse_dynamics`: condition on 所有视频帧 → 预测 action
- 不同 conditioning 方式给 action head 的学习信号可能冲突
- DROID 官方明确选择了 policy-only 并注释了原因

**修复建议**: 改为 `mode="policy"`

**讨论点**: `joint` 模式的设计初衷是多任务学习可以学到更鲁棒的表征。但 DROID 的实践表明 policy-only 收敛更好。可以先用 policy 验证收敛，确认基线后再实验 joint。

---

### 2.5 [中] DataLoader / episode shuffle

**DROID 架构**:
```
DROIDLeRobotDataset (map-style)
  → ActionSFTDataset (wrapper)
    → ActionIterableShuffleDataset (iterable_shuffle=True)
      → RankPartitionedDataLoader
```
- episode 级别 shuffle: 打乱 episode 顺序，episode 内 window 顺序保持
- rank 分片: 不同 rank 保证读取不同 episode → 跨 rank batch 去相关
- 保持 I/O locality 和 copy-on-write 内存共享

**GR1 架构**:
```
GR1LeRobotDataset (map-style)
  → GR1RobotPolicyDataPacker
    → DataPackerDataLoader (shuffle=True)
```
- 简单的 map-style random sampling
- 无 episode 级别去相关保证
- 多 GPU 时同一 batch 可能包含同一 episode 的相邻 window

**影响分析**:
- DROID 的整个 commit `300faa1` 就是为了解决这个问题
- 没有 episode shuffle 时，DROID 的 grad-norm 在 iter 100 时为 21 (不稳定)，加了之后降到 2.9
- GR1 有 24000 episodes / 5.4M samples，数据量较大，简单 shuffle 可能够用
- 但如果观察到 grad-norm 不稳定或 action loss plateau，这是首要排查方向

**修复建议**: 先观察 grad-norm 曲线，如果不稳定再实现 episode-level shuffle

---

### 2.6 [中] 没有 image augmentation

**DROID 设置**: `use_image_augmentation=True`
```python
# Random crop+rescale (spatial jitter) + color jitter, BEFORE the concat.
# All three views are stacked so one sampled set of params is applied
# uniformly across every frame and view (temporally + cross-view consistent),
# while each __getitem__ resamples. Matches the internal DROID recipe.
T.Compose([
    T.RandomCrop((int(h * 0.95), int(w * 0.95))),
    T.Resize((h, w), antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
])
```

**GR1 设置**: `GR1LeRobotDataset` 中无任何 image augmentation 代码

**影响分析**:
- GR1 数据来自 RoboCasa 仿真，视觉多样性可能不如真实世界的 DROID
- 仿真数据中相机位置、光照变化有限，augmentation 可以增加鲁棒性
- 但仿真数据本身可能已经足够 "clean"，augmentation 的收益不确定

**修复建议**: 优先级低于前几项。可以在基线确认后作为消融实验加入。

---

### 2.7 [低] chunk_length 差异

| 配置 | DROID | GR1 |
|------|-------|-----|
| `chunk_length` | 32 | 16 |
| 观测帧数 | 33 (chunk+1) | 17 (chunk+1) |

**影响分析**:
- 更短的 chunk 意味着模型能规划的时间跨度更短
- DROID 的 chunk=32 + 15 FPS ≈ 2.1 秒动作序列
- GR1 的 chunk=16 + 20 FPS = 0.8 秒动作序列
- 这可能是基于 GR1 数据特性 (更高 FPS) 的有意选择
- 不建议贸然修改，需要看具体任务的动作时长

---

### 2.8 [信息项] action 归一化方式不同

| 配置 | DROID (joint_pos) | GR1 |
|------|-------------------|-----|
| 归一化 | None (raw joint values) | minmax (per-dataset stats.json) |

**分析**:
- DROID joint_pos 模式显式设 `action_normalization=None`，使用原始关节值
- GR1 使用 per-dataset minmax 归一化到 [-1, 1]
- 这是两种合理的设计选择:
  - DROID 不做归一化是因为 joint_pos 的值域已经相对合理
  - GR1 做 per-dataset 归一化是因为 cosmos3_gr1_analysis.md 中分析过全局归一化有截断问题
- **这不是 bug，但要确保 inference 时使用完全相同的归一化/反归一化**（分析报告已确认等价）

---

### 2.9 [等价] use_state 处理方式

两者实现路径不同，但最终效果等价:

**DROID**: `_build_joint_action()` 在 dataset 内部将 initial state 拼接到 action 序列第 0 帧
```python
action = np.concatenate([initial_state, action], axis=0)  # [chunk + 1, 8]
```

**GR1**: `GR1RobotPolicyDataPacker.sft_process_sample()` 将 state 重命名为 `history_action`
```python
state = item.pop("state", None)
if state is not None:
    item["history_action"] = state
```
然后 `ActionTransformPipeline` 将其拼接为 SequencePlan 的 conditioning 帧

两者最终都是: state → 序列第 0 帧 → conditioning (不加噪、不算 loss)。**无问题。**

---

## 三、其他值得注意的配置对比

| 配置项 | DROID | GR1 | 备注 |
|--------|-------|-----|------|
| GPU 数 | 8 (shard=8, replicate=1) | 8 (shard=-1 auto) | 等价 |
| `max_samples_per_batch` | 128 | 64 (max_batch_size) | GR1 的 DataPacker 式 batch，不直接可比 |
| `num_workers` | 4 | 8 (Python 默认) | 影响数据吞吐 |
| `lr` | 2e-4 | 2e-4 | 相同 |
| `lr_multipliers` (action heads) | 5.0 | 5.0 (Python 默认) | 相同 |
| `weight_decay` | 0.05 | 0.05 | 相同 |
| `grad_clip` | 1.0, force_finite | 1.0, force_finite | 相同 |
| `warm_up_steps` | 0 | 0 | 相同 |
| `keys_to_select` | 相同 7 组件 | 相同 7 组件 | 相同 |
| `keys_to_skip_loading` | 相同 4 组件 | 相同 4 组件 | 相同 |
| `optimizer` | FusedAdam | FusedAdam | 相同 |
| 分辨率 | 480 (640×360) | 256 (256×256) | GR1 仿真数据分辨率较低 |
| 视角 | concat_view (wrist + L/R shoulder) | ego_view (单视角) | 数据差异 |
| FPS | 15 | 20 | 数据差异 |
| `cfg_dropout_rate` | 0.1 | 0.1 | 相同 |

---

## 四、超参问题: Batch Size 与 LR / Warmup 不匹配

### 4.1 Batch Size 差异巨大，LR 未相应调整

| | DROID (已验证) | GR1 (当前) |
|--|---------------|------------|
| Global batch | 128 × 64 ranks = **8192** | 64 × 8 GPUs = **~512** |
| Base LR | 2e-4 | 2e-4 |
| Action head LR | 2e-4 × 5 = **1e-3** | 2e-4 × 5 = **1e-3** |
| 有效 LR (含 scheduler f_max=0.4) | 2e-4 × 0.4 = 8e-5 | 2e-4 × 0.4 = 8e-5 |
| Action head 有效 LR | 8e-5 × 5 = **4e-4** | 8e-5 × 5 = **4e-4** |

DROID config 中有明确注释: `lr=2.0e-04, # for the 8192 global batch`。
GR1 的 batch 小了 **16 倍**但 LR 完全相同。通常 batch 缩小时 LR 应相应降低:
- 线性缩放: 2e-4 / 16 ≈ **1.25e-5**
- Sqrt 缩放: 2e-4 / sqrt(16) = **5e-5**

### 4.2 warm_up_steps=0 + 小 batch + 随机初始化 action heads

DROID 和 GR1 都使用 `warm_up_steps=0`。DROID 能工作是因为 8192 的大 batch 本身就降低了梯度方差。GR1 的情况更危险:

- **Action heads 随机初始化** (`keys_to_skip_loading` 排除了 action2llm 等) → 早期梯度方向不可靠
- **小 batch (512)** → 梯度估计方差更大
- **无 warmup** → 从 iteration 0 就用完整 LR (有效 4e-4 for action heads)
- 组合效果: 早期可能反复震荡，难以收敛到好的区域

### 4.3 建议

```toml
[scheduler]
warm_up_steps = [500]     # 给 action heads 500 步缓冲

[optimizer]
lr = 5.0e-5               # 按 sqrt 缩放: 2e-4 * sqrt(512/8192) ≈ 5e-5
```

**注意**: 建议先修复 `loss_scale=10.0` (第 2.1 节)，确认其效果后再调 LR/warmup。同时改多个变量会导致无法判断哪个生效。

---

## 五、修复优先级排序

### 第一轮 (只改配置，不改代码)

1. **`loss_scale = 10.0`** — 最可能的关键遗漏，DROID 有明确注释
2. **`encode_exact_durations = [17]`** — 对齐 VAE 编码
3. **`mode = "policy"`** — 让 action head 每步都训练

### 第二轮 (超参调整，建议在第一轮确认后再改)

4. **LR 缩放** — 当前 LR 是为 8192 global batch 设计的，GR1 仅 ~512，建议降到 5e-5
5. **`warm_up_steps = [500]`** — 给随机初始化的 action heads 缓冲期
6. **`max_num_tokens_after_packing = -1`** — 当前规模可能无影响，但消除隐患

### 第三轮 (需要验证/消融)

7. **观察 grad-norm 曲线** — 如果不稳定，实现 episode-level shuffle
8. **image augmentation** — 在基线确认后消融
9. **`chunk_length` 调整** — 需要结合任务特性评估

---

## 六、建议的 GR1 TOML 修改 (草案)

```toml
# === 第一轮修改 ===

[model.rectified_flow_training_config]
loss_scale = 10.0          # 平衡 vision loss 和 action loss (action_loss_weight=10)

[model.tokenizer]
encode_exact_durations = [17]  # chunk_length=16 → 17 帧，固定 VAE encode 长度

# mode 需要在 Python config 中修改:
# dataloader_train.data_source.mode = "policy"
# 或者在 launch script 的 TAIL_OVERRIDES 中加:
# "dataloader_train.data_source.mode=policy"

# === 第二轮修改 (确认第一轮效果后) ===

[optimizer]
lr = 5.0e-5                # sqrt 缩放: 2e-4 * sqrt(512/8192)

[scheduler]
warm_up_steps = [500]      # action heads 随机初始化 + 小 batch，需要预热

[model]
max_num_tokens_after_packing = -1  # 解除 token 数上限
```
