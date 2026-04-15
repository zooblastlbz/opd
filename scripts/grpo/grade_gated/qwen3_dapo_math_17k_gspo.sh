#!/usr/bin/env bash
set -euo pipefail

# pip install math_verify
# GSPO baseline on open-r1/DAPO-Math-17k-Processed without ref-KL regularization.

SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL=${MODEL:-Qwen/Qwen3-8B}
DATASET=${DATASET:-open-r1/DAPO-Math-17k-Processed}
OUTPUT_DIR=${OUTPUT_DIR:-output/Qwen3-8B-GSPO-DAPO-Math-17k}

TRAIN_TYPE=${TRAIN_TYPE:-lora}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
DEEPSPEED_STAGE=${DEEPSPEED_STAGE:-zero2}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-7168}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LOGGING_STEPS=${LOGGING_STEPS:-1}
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
REPORT_TO=${REPORT_TO:-wandb}

TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.9}
TOP_K=${TOP_K:-0}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SLEEP_LEVEL=${SLEEP_LEVEL:-1}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

EPSILON=${EPSILON:-3e-4}
EPSILON_HIGH=${EPSILON_HIGH:-4e-4}
STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-4}
BETA=${BETA:-0.0}

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
    --model "${MODEL}"
    --dataset "${DATASET}"
    --reward_funcs accuracy
    --system "${SYSTEM_PROMPT}"
    --enable_thinking false
    --load_from_cache_file true
    --torch_dtype "${TORCH_DTYPE}"
    --beta "${BETA}"
    --epsilon "${EPSILON}"
    --epsilon_high "${EPSILON_HIGH}"
    --steps_per_generation "${STEPS_PER_GENERATION}"
    --importance_sampling_level sequence
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
