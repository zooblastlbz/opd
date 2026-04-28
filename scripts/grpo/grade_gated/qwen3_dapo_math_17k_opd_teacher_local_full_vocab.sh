#!/usr/bin/env bash
set -euo pipefail

# pip install math_verify
# Baseline OPD script: student rollouts + all-token, full-vocab reverse KL.
# This is intentionally not TIP/grade-gated token selection.
# Edit values in this block directly for your experiment.

SYSTEM_PROMPT="You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

MODEL="Qwen/Qwen3-4B-Instruct-2507"
TEACHER_MODEL="Qwen/Qwen3-14B"
TEACHER_DEEPSPEED="zero3_offload"
OFFLOAD_TEACHER_MODEL="true"
# 0 means do not pass --gkd_logits_topk, so local teacher full logits are used.
GKD_LOGITS_TOPK=0
DATASET="open-r1/DAPO-Math-17k-Processed"
OUTPUT_DIR="output/Qwen3-4B-Instruct-2507-OPD-Baseline-FullVocab-DAPO-Math-17k"

TUNER_TYPE="full"
TORCH_DTYPE="bfloat16"
DEEPSPEED_STAGE="zero2"

MAX_PROMPT_LENGTH=1024
MAX_LENGTH=9216
MAX_COMPLETION_LENGTH=8192
VLLM_MAX_MODEL_LEN=9216
VLLM_GPU_MEMORY_UTILIZATION=0.4
VLLM_TENSOR_PARALLEL_SIZE=1

NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
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
WANDB_PROJECT="opd"
RUN_NAME="qwen3_4b_instruct_2507_opd_full_vocab_teacher_local_dapo_math_17k"

TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
SLEEP_LEVEL=1

# Baseline OPD:
# - LMBDA=1.0: always train on student-generated on-policy rollouts.
# - BETA=1.0: generalized JSD degenerates to reverse KL, KL(student || teacher).
# - SFT_ALPHA=0: no supervised CE term mixed into the OPD loss.
LMBDA=1.0
BETA=1.0
SFT_ALPHA=0

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
    --rlhf_type gkd
    --model "${MODEL}"
    --teacher_model "${TEACHER_MODEL}"
    --dataset "${DATASET}"
    --system "${SYSTEM_PROMPT}"
    --enable_thinking false
    --load_from_cache_file true
    --seq_kd false
    --lmbda "${LMBDA}"
    --beta "${BETA}"
    --sft_alpha "${SFT_ALPHA}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --torch_dtype "${TORCH_DTYPE}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --run_name "${RUN_NAME}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --save_only_model true
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --num_generations "${NUM_GENERATIONS}"
    --use_vllm true
    --vllm_mode colocate
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
    --sleep_level "${SLEEP_LEVEL}"
    --deepspeed "${DEEPSPEED_STAGE}"
    --tuner_type "${TUNER_TYPE}"
    --output_dir "${OUTPUT_DIR}"
    --report_to "${REPORT_TO_ARGS[@]}"
)

if [[ -n "${TEACHER_DEEPSPEED}" ]]; then
    CMD+=(--teacher_deepspeed "${TEACHER_DEEPSPEED}")
fi
if [[ "${OFFLOAD_TEACHER_MODEL}" == "true" ]]; then
    CMD+=(--offload_teacher_model true)
fi
if [[ "${GKD_LOGITS_TOPK}" != "0" ]]; then
    CMD+=(--gkd_logits_topk "${GKD_LOGITS_TOPK}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
NNODES="${NNODES}" \
NODE_RANK="${NODE_RANK}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
"${CMD[@]}"
