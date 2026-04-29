#!/usr/bin/env bash


# pip install math_verify
# Forward-KL OPD script: student rollouts + all-token, full-vocab forward KL.
# This is intentionally not TIP/grade-gated token selection.
# Edit values in this block directly for your experiment.
# Default launcher is DeepSpeed.
# Single-node: run this script directly.
# Multi-node: set USE_HOSTFILE_LAUNCH=true and provide HOSTFILE.

# ==========================================
# Launch Config
# ==========================================
USE_HOSTFILE_LAUNCH="false"
HOSTFILE="/etc/mpi/hostfile_first2"

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. The assistant must first provide the reasoning process inside <think> </think> tags, and then provide the final answer inside <answer> </answer> tags. Put the final mathematical answer inside \\boxed{} within the <answer> tag. The required format is: <think> reasoning here </think><answer> \\boxed{final answer} </answer>"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8
MASTER_PORT=29500

MODEL="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-1.7B-Base/"
TEACHER_MODEL="/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/output/Qwen3-8B-DAPO-DAPO-Math-17k-hard-format-len8192_zero_rl/v3-20260427-004019/checkpoint-271"
TEACHER_DEEPSPEED="zero3_offload"
OFFLOAD_TEACHER_MODEL="true"
# 0 means do not pass --gkd_logits_topk, so local teacher full logits are used.
GKD_LOGITS_TOPK=0
DATASET="/ytech_m2v5_hdd/workspace/kling_mm/Datasets/DAPO-Math-17k-Processed"
OUTPUT_DIR="output/Qwen3-4B-Instruct-2507-OPD-ForwardKL-FullVocab-DAPO-Math-17k"

TUNER_TYPE="full"
TORCH_DTYPE="bfloat16"
DEEPSPEED_STAGE="zero2"


MAX_LENGTH=1024
MAX_COMPLETION_LENGTH=7168
VLLM_MAX_MODEL_LEN=8192
VLLM_GPU_MEMORY_UTILIZATION=0.2
VLLM_TENSOR_PARALLEL_SIZE=1

NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
NUM_GENERATIONS=8
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
RUN_NAME="qwen3_4b_instruct_2507_opd_forward_kl_full_vocab_teacher_local_dapo_math_17k"

TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1
MAX_GRAD_NORM=1.0
SLEEP_LEVEL=0

# Forward-KL OPD:
# - LMBDA=1.0: always train on student-generated on-policy rollouts.
# - BETA=0.0: generalized JSD degenerates to forward KL, KL(teacher || student).
# - SFT_ALPHA=0: no supervised CE term mixed into the OPD loss.
LMBDA=1.0
BETA=0.0
SFT_ALPHA=0

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
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    mpirun --hostfile "${HOSTFILE}" --pernode -x PATH sh -c 'rm -f /dev/shm/nccl-*'
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
    --max_length "${MAX_LENGTH}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --run_name "${RUN_NAME}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --warmup_ratio "${WARMUP_RATIO}"
    --logging_steps "${LOGGING_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --save_only_model true
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataset_num_proc "${DATASET_NUM_PROC}"
    --num_generations "${NUM_GENERATIONS}"
    --max_grad_norm "${MAX_GRAD_NORM}"
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
MASTER_PORT="${MASTER_PORT}" \
"${CMD[@]}"
