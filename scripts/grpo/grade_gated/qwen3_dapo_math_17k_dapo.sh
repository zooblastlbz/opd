#!/usr/bin/env bash

# pip install math_verify
# Edit values in this block directly for your experiment.
# Default launcher is DeepSpeed.
# Single-node: run this script directly.
# Multi-node: set USE_HOSTFILE_LAUNCH=true and provide HOSTFILE.

# ==========================================
# Launch Config
# ==========================================
USE_HOSTFILE_LAUNCH="true"
HOSTFILE="/etc/mpi/hostfile_first2"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8
MASTER_PORT=29509

# DeepSpeed env propagation
DEEPSPEED_ENV_SOURCE="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/feature_combination/env_a800"
export TORCH_DISTRIBUTED_TIMEOUT=1800

rm -f .deepspeed_env
rm -f /root/.deepspeed_env
cp "${DEEPSPEED_ENV_SOURCE}" /root/.deepspeed_env
set -a
source "${DEEPSPEED_ENV_SOURCE}"
set +a

# ==========================================
# Resume Config
# ==========================================
# 设置为 "true" 启用自动 resume，"false" 则从头训练
RESUME="true"
# 如果指定了具体路径，则从此路径 resume；留空则自动从 OUTPUT_DIR 中找最新 checkpoint
RESUME_FROM_CHECKPOINT="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/output/Qwen3-8B-DAPO-DAPO-Math-17k-format-len8192/v5-20260423-213817/checkpoint-260"

# format reward requires:
# <think>...</think><answer>...</answer>
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. The assistant must first provide the reasoning process inside <think> </think> tags, and then provide the final answer inside <answer> </answer> tags. Put the final mathematical answer inside \\boxed{} within the <answer> tag. The required format is: <think> reasoning here </think><answer> \\boxed{final answer} </answer>"

MODEL="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-8B"
DATASET="/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DAPO-Math-17k-Processed"
VAL_DATASETS=("aime2025_i_val" "aime2025_ii_val")
REWARD_PLUGIN="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/scripts/grpo/grade_gated/opd_ttrl_math_reward.py"
VAL_PLUGIN="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/scripts/grpo/grade_gated/aime2025_val_plugin.py"
EXTERNAL_PLUGINS=("${REWARD_PLUGIN}" "${VAL_PLUGIN}")
OUTPUT_DIR="output/Qwen3-8B-DAPO-DAPO-Math-17k-format-len8192"

TUNER_TYPE="full"
TORCH_DTYPE="bfloat16"
DEEPSPEED_STAGE="zero2"

# ==========================================
# Length Budget
# ==========================================
MAX_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
VLLM_MAX_MODEL_LEN=8192

VLLM_GPU_MEMORY_UTILIZATION=0.2
VLLM_TENSOR_PARALLEL_SIZE=1

NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16
NUM_GENERATIONS=8
NUM_GENERATIONS_EVAL=1
LEARNING_RATE=1e-6
LR_SCHEDULER_TYPE="constant"
WARMUP_RATIO=0
LOGGING_STEPS=1
SAVE_STEPS=20
EVAL_STEPS=20
SAVE_TOTAL_LIMIT=5
DATALOADER_NUM_WORKERS=4
DATASET_NUM_PROC=4
REPORT_TO="wandb"
WANDB_PROJECT="opd"
RUN_NAME="qwen3_8b_dapo_dapo_math_17k_format_len8192"

TEMPERATURE=1.0
#TOP_P=0.9
#TOP_K=0
MAX_GRAD_NORM=1.0
SLEEP_LEVEL=1

LOSS_TYPE="dapo"
EPSILON=0.2
EPSILON_HIGH=0.28
DYNAMIC_SAMPLE="false"
BETA=0.001
REF_KL_EXTRA_VOCAB_TOPK=0

# ==========================================
# Auto-detect latest checkpoint for resume
# ==========================================
RESUME_ARGS=()
if [[ "${RESUME}" == "true" ]]; then
    if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
        # 使用用户指定的 checkpoint 路径
        if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
            echo "Specified RESUME_FROM_CHECKPOINT does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
            exit 1
        fi
        RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
        echo "Resuming from specified checkpoint: ${RESUME_FROM_CHECKPOINT}"
    else
        # 自动从 OUTPUT_DIR 中查找最新的 checkpoint（按目录名排序，取最后一个）
        if [[ -d "${OUTPUT_DIR}" ]]; then
            LATEST_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
            if [[ -n "${LATEST_CKPT}" ]]; then
                RESUME_ARGS+=(--resume_from_checkpoint "${LATEST_CKPT}")
                echo "Auto-resuming from latest checkpoint: ${LATEST_CKPT}"
            else
                echo "No checkpoint found in ${OUTPUT_DIR}, starting from scratch."
            fi
        else
            echo "OUTPUT_DIR does not exist yet, starting from scratch."
        fi
    fi
fi

LAUNCHER=(
    deepspeed
    --master_port="${MASTER_PORT}"
)
if [[ "${USE_HOSTFILE_LAUNCH}" == "true" ]]; then
    if [[ -z "${HOSTFILE}" ]]; then
        echo "HOSTFILE must be set when USE_HOSTFILE_LAUNCH=true." >&2
        exit 1
    fi
    if [[ ! -f "${HOSTFILE}" ]]; then
        echo "HOSTFILE does not exist: ${HOSTFILE}" >&2
        exit 1
    fi
    LAUNCHER+=(
        --hostfile="${HOSTFILE}"
        swift/cli/rlhf.py
    )
else
    LAUNCHER+=(
        --num_gpus="${NPROC_PER_NODE}"
        swift/cli/rlhf.py
    )
fi
read -r -a REPORT_TO_ARGS <<< "${REPORT_TO}"

CMD=(
    "${LAUNCHER[@]}"
    --rlhf_type grpo
    --loss_type "${LOSS_TYPE}"
    --model "${MODEL}"
    --dataset "${DATASET}"
    --val_dataset "${VAL_DATASETS[@]}"
    --external_plugins "${EXTERNAL_PLUGINS[@]}"

    --reward_funcs opd_ttrl_math_accuracy

    --system "${SYSTEM_PROMPT}"
    --enable_thinking false
    --load_from_cache_file true
    --torch_dtype "${TORCH_DTYPE}"
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --run_name "${RUN_NAME}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --num_generations_eval "${NUM_GENERATIONS_EVAL}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --num_generations "${NUM_GENERATIONS}"
    --temperature "${TEMPERATURE}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --epsilon "${EPSILON}"
    --epsilon_high "${EPSILON_HIGH}"
    --dynamic_sample "${DYNAMIC_SAMPLE}"
    --beta "${BETA}"
    --ref_kl_extra_vocab_topk "${REF_KL_EXTRA_VOCAB_TOPK}"
    --use_vllm true
    --enable_thinking false
    --vllm_mode colocate
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
    --sleep_level "${SLEEP_LEVEL}"
    --deepspeed "${DEEPSPEED_STAGE}"
    --tuner_type "${TUNER_TYPE}"
    --log_completions false
    --log_entropy true
    --overlong_filter true
    --output_dir "${OUTPUT_DIR}"
    --report_to "${REPORT_TO_ARGS[@]}"

    # Resume arguments
    "${RESUME_ARGS[@]}"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_PORT="${MASTER_PORT}" \
"${CMD[@]}"
