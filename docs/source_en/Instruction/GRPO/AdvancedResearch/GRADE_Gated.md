# GRADE-Gated

`GRADE-Gated` adds a mixed objective to `swift rlhf --rlhf_type grpo`:

- entrypoint: `--loss_type grade_gated`
- teacher sources: OPSD/self-distill via `teacher_prompt`, frozen `teacher_model`, or remote `teacher_model_server`
- default topology: the training node runs `swift rlhf`, while a teacher node can run `vllm serve`

## Overview

Each sample first gets a scalar reward `r_i` from the existing reward functions. GRADE-Gated then builds a gate signal `s_i`, normalizes it into `r_i^{norm}`, and maps it to a mixing weight `\alpha_i`:

$$
L_i = \alpha_i \cdot L_i^{\mathrm{GRPO(local)}} + (1 - \alpha_i) \cdot L_i^{\mathrm{OPD}}
$$

where:

- `L_GRPO(local)` keeps the GRPO clipping form but replaces the sequence-level advantage `A_i` with a token-level modulated advantage `\hat{A}_{i,t}`
- `L_OPD` uses KL alignment on completion tokens and supports both `forward_kl` and `reverse_kl`

## Gate Design

### 1. Meaning of `sample` and `group`

`sample/group` only decides which absolute signal is used for the gate:

- `sample`: `s_i = r_i`
- `group`: first average the rewards of all rollouts for the same prompt, then broadcast that group score to every sample in the group

So `group` means “how well the current model handles this problem overall”, while `sample` means “how good this specific rollout is”.

### 2. `grade_reward_norm` strategies

#### `group_minmax`

This is the legacy behavior. Rewards are min-max normalized within each prompt group:

$$
r_i^{norm} = \frac{r_i - r_{min}}{r_{max} - r_{min}}
$$

If the group variance is zero, the normalized score is set to `0.5`. This mode behaves like an intra-group ranking gate and is mainly kept for backward-compatible ablations.

#### `fixed_range`

This is the fixed-hyperparameter strategy. After building the absolute gate signal `s_i`, it normalizes using a pre-defined range:

$$
r_i^{norm} = \mathrm{clip}\left(\frac{s_i - r_{low}}{r_{high} - r_{low}}, 0, 1\right)
$$

where:

- `r_low = --grade_reward_min`
- `r_high = --grade_reward_max`

This mode is appropriate when the reward scale is known before training, for example math `accuracy \in [0, 1]`.

#### `ema`

This is the sliding-statistics strategy. It maintains exponential moving averages of the gate signal mean and second moment:

$$
\mu_t = \beta \mu_{t-1} + (1-\beta)\bar{s}_t
$$

$$
m_t = \beta m_{t-1} + (1-\beta)\overline{s_t^2}
$$

$$
\sigma_t = \sqrt{\max(m_t - \mu_t^2, \epsilon)}
$$

The current signal is then standardized and squashed back to `[0, 1]`:

$$
z_i = \mathrm{clip}\left(\frac{s_i - \mu_t}{\sigma_t + \epsilon}, -c, c\right)
$$

$$
r_i^{norm} = \sigma(z_i)
$$

where:

- `\beta = --grade_reward_ema_decay`
- `\epsilon = --grade_reward_ema_eps`
- `c = --grade_reward_ema_clip`

This mode is better suited to drifting reward scales or multi-task settings with heterogeneous reward distributions.

### 3. `grade_alpha_mapping`

`r_i^{norm}` is finally mapped into `alpha` via:

- `linear`: `\alpha = r^{norm}`
- `piecewise`: linearly interpolate inside `[low, high]`, clamp outside to `0/1`
- `sigmoid`: `\alpha = \sigma(T (r^{norm} - 0.5))`

## Credit Assignment

The token-level credit assignment only affects the `GRPO(local)` branch. It transforms the sequence-level advantage `A_i` into a token-level advantage `\hat{A}_{i,t}`.

For positive advantages:

$$
e_{i,t}^{+} = \frac{\max(0, H^S_{i,t} - H^T_{i,t})}{H^S_{i,t} + \epsilon}
$$

For negative advantages:

$$
e_{i,t}^{-} = \frac{H_{max} - H^S_{i,t}}{H_{max}}, \quad H_{max} = \log |V|
$$

Then:

$$
\hat{A}_{i,t} = A_i \cdot \mathrm{clip}(1 + \lambda \cdot g_{i,t}, 1-c, 1+c)
$$

where:

- `\lambda = --grade_credit_scale`
- `c = --grade_credit_clip`
- `\epsilon = --grade_entropy_eps`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--loss_type` | `str` | - | Set to `grade_gated` |
| `--grade_alpha_granularity` | `str` | `group` | `group` uses the group-level signal, `sample` uses the sample’s own signal |
| `--grade_alpha_mapping` | `str` | `linear` | `linear` / `piecewise` / `sigmoid` |
| `--grade_alpha_piecewise_low` | `float` | `0.25` | Low threshold for piecewise mapping |
| `--grade_alpha_piecewise_high` | `float` | `0.75` | High threshold for piecewise mapping |
| `--grade_alpha_sigmoid_temperature` | `float` | `10.0` | Sigmoid gate temperature |
| `--grade_reward_norm` | `str` | `group_minmax` | `group_minmax` / `fixed_range` / `ema` |
| `--grade_reward_min` | `float` | `0.0` | Lower bound for `fixed_range` |
| `--grade_reward_max` | `float` | `1.0` | Upper bound for `fixed_range` |
| `--grade_reward_ema_decay` | `float` | `0.99` | EMA decay in `ema` mode |
| `--grade_reward_ema_eps` | `float` | `1e-6` | Numerical stabilizer in `ema` mode |
| `--grade_reward_ema_clip` | `float` | `3.0` | Clip range for the standardized `z` in `ema` mode |
| `--grade_opd_loss_type` | `str` | `forward_kl` | `forward_kl=KL(teacher || student)`, `reverse_kl=KL(student || teacher)` |
| `--grade_credit_scale` | `float` | `1.0` | Token credit scaling factor |
| `--grade_credit_clip` | `float` | `0.2` | Clip range for `1 + credit` |
| `--grade_entropy_eps` | `float` | `1e-6` | Numerical stabilizer for entropy credit |
| `--teacher_model` | `str` | `None` | Local frozen teacher |
| `--teacher_model_server` | `str` | `None` | Remote teacher server URL |
| `--gkd_logits_topk` | `int` | `None` | Required with `teacher_model_server` |

## Teacher Modes

### 1. OPSD / self-distill

When the dataset provides a `teacher_prompt` column and neither `teacher_model` nor `teacher_model_server` is set:

- the student sees the original prompt
- the teacher sees `teacher_prompt`
- both are aligned to the same rollout response

### 2. External teacher model

When `--teacher_model` is set, `GRPOTrainer` loads a frozen teacher and computes OPD loss on completion tokens.

### 3. Remote teacher server

When `--teacher_model_server http://<host>:8000` is set, the training process does not load teacher weights locally. Instead, it queries a separate `vllm serve` node for top-k prompt logprobs.

Example teacher server launch:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-logprobs 64
```

Then point training to that server:

```bash
--teacher_model_server http://teacher-host:8000 \
--gkd_logits_topk 64
```

> In `teacher_model_server` mode, teacher entropy is an approximation built from top-k probabilities plus a single remainder bucket. The logs expose this via `grade/teacher_entropy_is_approx=1.0`.
> KL in `teacher_model_server` mode is also computed on a compressed `top-k tokens + other bucket` distribution.

## Examples

### Fixed-range gate

For single-task math with `accuracy \in [0, 1]`, an absolute gate is usually the cleanest choice:

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

For drifting reward scales or heterogeneous multi-task training:

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

Reference scripts:

- [grade_gated_opsd.sh](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/grade_gated_opsd.sh)
- [grade_gated_teacher_server.sh](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal/grade_gated_teacher_server.sh)
- [qwen3_dapo_math_17k_grade_gated_teacher_local.sh](https://github.com/modelscope/ms-swift/blob/main/scripts/grpo/grade_gated/qwen3_dapo_math_17k_grade_gated_teacher_local.sh)
- [qwen3_dapo_math_17k_grade_gated_self_distill.sh](https://github.com/modelscope/ms-swift/blob/main/scripts/grpo/grade_gated/qwen3_dapo_math_17k_grade_gated_self_distill.sh)

## Metrics

Extra training metrics:

- `grade/alpha_mean`
- `grade/alpha_min`
- `grade/alpha_max`
- `grade/gate_signal_mean`
- `grade/gate_signal_min`
- `grade/gate_signal_max`
- `grade/reward_norm_mean`
- `grade/ema_mean` and `grade/ema_std` in `ema` mode
- `grade/grpo_component_loss`
- `grade/opd_component_loss`
- `grade/teacher_entropy_mean`
- `grade/credit_mean`
- `grade/credit_max`
- `grade/teacher_entropy_is_approx`

Completion table fields:

- `alpha`
- `gate_signal`
- `reward_norm`
- `teacher_mode`

## Current Limitations

- Transformers/TRL `swift rlhf` path only
- `advantage_estimator=grpo` only
- no `chord_sft_dataset`
- no `padding_free`
- no `sequence_parallel`
- no `liger kernel`
- no Ray / Megatron integration
