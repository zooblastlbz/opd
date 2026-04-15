# GRADE-Gated

`GRADE-Gated` 为 `swift rlhf --rlhf_type grpo` 增加了一条“GRPO 主目标 + OPD/teacher 蒸馏目标”的混合训练路径：

- 主入口：`--loss_type grade_gated`
- teacher 形态：`teacher_prompt` 的 OPSD/self-distill、冻结 `teacher_model`、远端 `teacher_model_server`
- 远端拓扑：训练节点执行 `swift rlhf`，teacher 节点可独立执行 `vllm serve`

## 方法概览

每个样本先用现有 reward funcs 得到标量 reward `r_i`，然后构造一个 gate 输入信号 `s_i`，再将其归一化为 `r_i^{norm}` 并映射成混合权重 `\alpha_i`：

$$
L_i = \alpha_i \cdot L_i^{\mathrm{GRPO(local)}} + (1 - \alpha_i) \cdot L_i^{\mathrm{OPD}}
$$

其中：

- `L_GRPO(local)` 保留 GRPO 的裁剪形式，但把序列级优势 `A_i` 改成 token 级局部调制优势 `\hat{A}_{i,t}`
- `L_OPD` 使用 teacher/student 在 completion token 上的 KL 对齐，支持 `forward_kl` 与 `reverse_kl`

## Gate 设计

### 1. `sample` 与 `group` 的语义

`sample/group` 只决定 gate 使用哪个层级的绝对指标，不再强制表示组内相对排序：

- `sample`：`s_i = r_i`
- `group`：对同一个 prompt 下的多条 rollout 先求均值，再广播给该组所有样本

也就是说，`group` 表示“这个问题对当前模型的整体胜任度”，`sample` 表示“这条具体 rollout 的胜任度/质量”。

### 2. 三种 `grade_reward_norm`

#### `group_minmax`

兼容旧实现。先在同一个 prompt group 内做 min-max：

$$
r_i^{norm} = \frac{r_i - r_{min}}{r_{max} - r_{min}}
$$

如果组内零方差，则设为 `0.5`。这个模式更像“组内相对排序 gate”，适合做旧版对照，不适合表达绝对难度。

#### `fixed_range`

固定超参数模式。先得到绝对 gate 信号 `s_i`，再用预设区间归一化：

$$
r_i^{norm} = \mathrm{clip}\left(\frac{s_i - r_{low}}{r_{high} - r_{low}}, 0, 1\right)
$$

其中：

- `r_low = --grade_reward_min`
- `r_high = --grade_reward_max`

这个模式适合训练前就知道 reward 尺度的场景，例如 math 的 `accuracy \in [0, 1]`。

#### `ema`

滑动统计归一化模式。先维护 gate 信号 `s` 的指数滑动均值和二阶矩：

$$
\mu_t = \beta \mu_{t-1} + (1-\beta)\bar{s}_t
$$

$$
m_t = \beta m_{t-1} + (1-\beta)\overline{s_t^2}
$$

$$
\sigma_t = \sqrt{\max(m_t - \mu_t^2, \epsilon)}
$$

然后使用当前滑动统计对 `s_i` 做标准化，并通过 sigmoid 压回 `[0,1]`：

$$
z_i = \mathrm{clip}\left(\frac{s_i - \mu_t}{\sigma_t + \epsilon}, -c, c\right)
$$

$$
r_i^{norm} = \sigma(z_i)
$$

其中：

- `\beta = --grade_reward_ema_decay`
- `\epsilon = --grade_reward_ema_eps`
- `c = --grade_reward_ema_clip`

这个模式更适合 reward 尺度会漂移、或者多任务训练时不同任务 reward 分布差异较大的情况。

### 3. `grade_alpha_mapping`

`r_i^{norm}` 最终通过下列映射得到 gate：

- `linear`：`\alpha = r^{norm}`
- `piecewise`：在 `[low, high]` 内线性插值，区间外截断到 `0/1`
- `sigmoid`：`\alpha = \sigma(T (r^{norm} - 0.5))`

## Credit Assignment

`GRADE-Gated` 的 token-level credit assignment 只作用在 `GRPO(local)` 分支内部，用于把序列级优势 `A_i` 调成 token 级优势 `\hat{A}_{i,t}`。

正优势时：

$$
e_{i,t}^{+} = \frac{\max(0, H^S_{i,t} - H^T_{i,t})}{H^S_{i,t} + \epsilon}
$$

负优势时：

$$
e_{i,t}^{-} = \frac{H_{max} - H^S_{i,t}}{H_{max}}, \quad H_{max} = \log |V|
$$

然后：

$$
\hat{A}_{i,t} = A_i \cdot \mathrm{clip}(1 + \lambda \cdot g_{i,t}, 1-c, 1+c)
$$

其中：

- `\lambda = --grade_credit_scale`
- `c = --grade_credit_clip`
- `\epsilon = --grade_entropy_eps`

## 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--loss_type` | `str` | - | 设置为 `grade_gated` |
| `--grade_alpha_granularity` | `str` | `group` | `group` 使用组均值信号，`sample` 使用样本自身信号 |
| `--grade_alpha_mapping` | `str` | `linear` | `linear` / `piecewise` / `sigmoid` |
| `--grade_alpha_piecewise_low` | `float` | `0.25` | 分段映射低阈值 |
| `--grade_alpha_piecewise_high` | `float` | `0.75` | 分段映射高阈值 |
| `--grade_alpha_sigmoid_temperature` | `float` | `10.0` | sigmoid 门控温度 |
| `--grade_reward_norm` | `str` | `group_minmax` | `group_minmax` / `fixed_range` / `ema` |
| `--grade_reward_min` | `float` | `0.0` | `fixed_range` 下的下界 |
| `--grade_reward_max` | `float` | `1.0` | `fixed_range` 下的上界 |
| `--grade_reward_ema_decay` | `float` | `0.99` | `ema` 模式的滑动衰减 |
| `--grade_reward_ema_eps` | `float` | `1e-6` | `ema` 模式的数值稳定项 |
| `--grade_reward_ema_clip` | `float` | `3.0` | `ema` 标准化后 `z` 的截断范围 |
| `--grade_opd_loss_type` | `str` | `forward_kl` | `forward_kl=KL(teacher || student)`，`reverse_kl=KL(student || teacher)` |
| `--grade_credit_scale` | `float` | `1.0` | token credit 缩放系数 |
| `--grade_credit_clip` | `float` | `0.2` | `1 + credit` 的 clip 区间 |
| `--grade_entropy_eps` | `float` | `1e-6` | 熵信用的数值稳定项 |
| `--teacher_model` | `str` | `None` | 本地 frozen teacher |
| `--teacher_model_server` | `str` | `None` | 远端 teacher server URL |
| `--gkd_logits_topk` | `int` | `None` | `teacher_model_server` 必填 |

## Teacher 模式

### 1. OPSD / self-distill

如果数据里包含 `teacher_prompt` 列，而没有显式传 `teacher_model` / `teacher_model_server`，则训练时：

- student 看原始 prompt
- teacher 看 `teacher_prompt`
- 两者共享同一条 rollout response

### 2. 外部 teacher model

如果传入 `--teacher_model`，`GRPOTrainer` 会像 GKD 一样加载一份冻结 teacher，并在 completion token 上计算 OPD loss。

### 3. 远端 teacher server

如果传入 `--teacher_model_server http://<host>:8000`，训练进程不会加载 teacher 权重，而是向独立的 `vllm serve` 节点请求 top-k prompt logprobs。

teacher server 需要先启动，例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-logprobs 64
```

然后训练节点指向该地址：

```bash
--teacher_model_server http://teacher-host:8000 \
--gkd_logits_topk 64
```

> `teacher_model_server` 路径下的 teacher 熵是基于 top-k 概率加剩余质量单桶的近似值，日志中会记录 `grade/teacher_entropy_is_approx=1.0`。
> `teacher_model_server` 路径下的 KL 也是基于 `top-k token + other bucket` 的压缩分布近似。

## 示例

### 固定阈值 gate

对 `accuracy \in [0, 1]` 的单任务 math，推荐直接使用绝对区间：

```bash
swift rlhf \
    --rlhf_type grpo \
    --loss_type grade_gated \
    --dataset your_math_dataset \
    --reward_funcs accuracy \
    --num_generations 8 \
    --grade_alpha_granularity group \
    --grade_reward_norm fixed_range \
    --grade_reward_min 0.0 \
    --grade_reward_max 1.0 \
    --grade_alpha_mapping linear \
    --grade_opd_loss_type forward_kl
```

### EMA gate

对 reward 尺度会漂移、或多任务 reward 分布差异较大的场景，推荐：

```bash
swift rlhf \
    --rlhf_type grpo \
    --loss_type grade_gated \
    --teacher_model_server http://teacher-host:8000 \
    --gkd_logits_topk 64 \
    --grade_alpha_granularity sample \
    --grade_reward_norm ema \
    --grade_reward_ema_decay 0.99 \
    --grade_reward_ema_clip 3.0 \
    --grade_alpha_mapping sigmoid \
    --grade_opd_loss_type reverse_kl
```

训练脚本参考：

- [grade_gated_opsd.sh](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/grade_gated_opsd.sh)
- [grade_gated_teacher_server.sh](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/grade_gated_teacher_server.sh)
- [qwen3_dapo_math_17k_grade_gated_teacher_local.sh](https://github.com/modelscope/ms-swift/blob/main/scripts/grpo/grade_gated/qwen3_dapo_math_17k_grade_gated_teacher_local.sh)
- [qwen3_dapo_math_17k_grade_gated_self_distill.sh](https://github.com/modelscope/ms-swift/blob/main/scripts/grpo/grade_gated/qwen3_dapo_math_17k_grade_gated_self_distill.sh)

## 观测指标

训练日志会额外记录：

- `grade/alpha_mean`
- `grade/alpha_min`
- `grade/alpha_max`
- `grade/gate_signal_mean`
- `grade/gate_signal_min`
- `grade/gate_signal_max`
- `grade/reward_norm_mean`
- `grade/ema_mean` 与 `grade/ema_std`（仅 `ema` 模式）
- `grade/grpo_component_loss`
- `grade/opd_component_loss`
- `grade/teacher_entropy_mean`
- `grade/credit_mean`
- `grade/credit_max`
- `grade/teacher_entropy_is_approx`

completion 表中还会记录：

- `alpha`
- `gate_signal`
- `reward_norm`
- `teacher_mode`

## 当前限制

- 仅支持 `swift rlhf` 的 Transformers/TRL 路径
- 仅支持 `advantage_estimator=grpo`
- 不支持 `chord_sft_dataset`
- 不支持 `padding_free`
- 不支持 `sequence_parallel`
- 不支持 `liger kernel`
- 不引入 Ray / Megatron 路径
