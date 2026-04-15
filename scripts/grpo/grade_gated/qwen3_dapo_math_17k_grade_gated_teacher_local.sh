#!/usr/bin/env bash
set -euo pipefail

# pip install math_verify
# Training node script for GRADE-Gated with a stronger teacher.
# Edit values in the block below directly for your experiment.
# Default mode is shared-node local teacher via `--teacher_model`.
# To switch back to a remote teacher server, change `TEACHER_BACKEND` to `server`.
#
# Remote teacher server example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# vllm serve Qwen/Qwen3-32B \
#     --port 8000 \
#     --max-logprobs 64

SYSTEM_PROMPT="You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."

CUDA_VISIBLE_DEVICES="0,1,2,3"
NPROC_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

MODEL="Qwen/Qwen3-8B"
TEACHER_BACKEND="local"
TEACHER_MODEL="Qwen/Qwen3-14B"
TEACHER_MODEL_SERVER="http://teacher-host:8000"
TEACHER_DEEPSPEED=""
OFFLOAD_TEACHER_MODEL="true"
GKD_LOGITS_TOPK=16
DATASET="open-r1/DAPO-Math-17k-Processed"
OUTPUT_DIR="output/Qwen3-8B-GRADE-Gated-DAPO-Math-17k-Teacher"

TRAIN_TYPE="lora"
TORCH_DTYPE="bfloat16"
DEEPSPEED_STAGE="zero2"

MAX_PROMPT_LENGTH=1024
MAX_LENGTH=8192
MAX_COMPLETION_LENGTH=7168
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.4
VLLM_TENSOR_PARALLEL_SIZE=1

NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
NUM_GENERATIONS=4
LEARNING_RATE=1e-6
WARMUP_RATIO=0.05
LOGGING_STEPS=1
SAVE_STEPS=100
EVAL_STEPS=100
SAVE_TOTAL_LIMIT=5
DATALOADER_NUM_WORKERS=4
DATASET_NUM_PROC=4
REPORT_TO="wandb"

TEMPERATURE=1.0
TOP_P=0.95
TOP_K=50
MAX_GRAD_NORM=1.0
SLEEP_LEVEL=1

LORA_RANK=16
LORA_ALPHA=32

BETA=0.0
REF_KL_EXTRA_VOCAB_TOPK=0
GRADE_ALPHA_GRANULARITY="group"
GRADE_ALPHA_MAPPING="linear"
GRADE_REWARD_NORM="fixed_range"
GRADE_REWARD_MIN=0.0
GRADE_REWARD_MAX=1.0
GRADE_REWARD_EMA_DECAY=0.99
GRADE_REWARD_EMA_EPS=1e-6
GRADE_REWARD_EMA_CLIP=3.0
GRADE_OPD_LOSS_TYPE="forward_kl"

LAUNCHER=(swift rlhf)
if [[ "${NNODES}" != "1" ]]; then
    LAUNCHER=(
        torchrun
        --nproc_per_node="${NPROC_PER_NODE}"
        --nnodes="${NNODES}"
        --node_rank="${NODE_RANK}"
        --master_addr="${MASTER_ADDR}"
        --master_port="${MASTER_PORT}"
        swift/cli/rlhf.py
    )
fi
read -r -a REPORT_TO_ARGS <<< "${REPORT_TO}"

CMD=(
    "${LAUNCHER[@]}"
    --rlhf_type grpo
    --loss_type grade_gated
    --model "${MODEL}"
    --dataset "${DATASET}"
    --reward_funcs accuracy
    --system "${SYSTEM_PROMPT}"
    --enable_thinking false
    --load_from_cache_file true
    --torch_dtype "${TORCH_DTYPE}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --num_generations "${NUM_GENERATIONS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --max_grad_norm "${MAX_GRAD_NORM}"
    --beta "${BETA}"
    --ref_kl_extra_vocab_topk "${REF_KL_EXTRA_VOCAB_TOPK}"
    --grade_alpha_granularity "${GRADE_ALPHA_GRANULARITY}"
    --grade_alpha_mapping "${GRADE_ALPHA_MAPPING}"
    --grade_reward_norm "${GRADE_REWARD_NORM}"
    --grade_reward_min "${GRADE_REWARD_MIN}"
    --grade_reward_max "${GRADE_REWARD_MAX}"
    --grade_reward_ema_decay "${GRADE_REWARD_EMA_DECAY}"
    --grade_reward_ema_eps "${GRADE_REWARD_EMA_EPS}"
    --grade_reward_ema_clip "${GRADE_REWARD_EMA_CLIP}"
    --grade_opd_loss_type "${GRADE_OPD_LOSS_TYPE}"
    --use_vllm true
    --vllm_mode colocate
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
    --sleep_level "${SLEEP_LEVEL}"
    --deepspeed "${DEEPSPEED_STAGE}"
    --log_completions true
    --overlong_filter true
    --output_dir "${OUTPUT_DIR}"
    --report_to "${REPORT_TO_ARGS[@]}"
)

if [[ "${TEACHER_BACKEND}" == "local" ]]; then
    CMD+=(--teacher_model "${TEACHER_MODEL}")
    if [[ -n "${TEACHER_DEEPSPEED}" ]]; then
        CMD+=(--teacher_deepspeed "${TEACHER_DEEPSPEED}")
    fi
    if [[ "${OFFLOAD_TEACHER_MODEL}" == "true" ]]; then
        CMD+=(--offload_teacher_model true)
    fi
    if [[ "${GKD_LOGITS_TOPK}" != "0" ]]; then
        CMD+=(--gkd_logits_topk "${GKD_LOGITS_TOPK}")
    fi
elif [[ "${TEACHER_BACKEND}" == "server" ]]; then
    CMD+=(--teacher_model_server "${TEACHER_MODEL_SERVER}" --gkd_logits_topk "${GKD_LOGITS_TOPK}")
else
    echo "Unsupported TEACHER_BACKEND=${TEACHER_BACKEND}. Expected 'local' or 'server'." >&2
    exit 1
fi

if [[ "${TRAIN_TYPE}" == "lora" ]]; then
    CMD+=(
        --tuner_type lora
        --vllm_enable_lora true
        --target_modules all-linear
        --lora_rank "${LORA_RANK}"
        --lora_alpha "${LORA_ALPHA}"
    )
elif [[ "${TRAIN_TYPE}" == "full" ]]; then
    CMD+=(--tuner_type full)
else
    echo "Unsupported TRAIN_TYPE=${TRAIN_TYPE}. Expected 'lora' or 'full'." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
"${CMD[@]}"
