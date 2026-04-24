#!/usr/bin/env bash
set -euo pipefail

# OPD paper-aligned math evaluation:
# - AIME 2024, AIME 2025, MATH-500, AMC 2023
# - avg@16 via 16 independent sampled eval repeats
# - temperature=0.7, top_p=0.95, max validation response length=31,744
#
# Usage:
#   bash scripts/grpo/grade_gated/eval_qwen3_dapo_math_benchmarks.sh
#
# Change EVAL_BENCHMARKS below to evaluate all benchmarks or a selected subset.

# ==========================================
# Model Selection
# ==========================================
# Format:
#   "name|model_path|adapters"
#
# For full-parameter checkpoints, leave adapters empty.
# For LoRA checkpoints, put the base model in model_path and one or more
# space-separated adapter paths in adapters.
MODEL_SPECS=(
    "dapo_ckpt260|/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/output/Qwen3-8B-DAPO-DAPO-Math-17k-format-len8192/v5-20260423-213817/checkpoint-260|"
    # "gspo_ckptXXX|/path/to/gspo/checkpoint-XXX|"
    # "grpo_ckptXXX|/path/to/grpo/checkpoint-XXX|"
    # "lora_exp|/path/to/base/model|/path/to/lora/checkpoint"
)

# Evaluate all configured models:
EVAL_TARGETS=("all")
# Or evaluate selected models by name:
# EVAL_TARGETS=("dapo_ckpt260" "gspo_ckptXXX")

# ==========================================
# Benchmark Selection
# ==========================================
# Supported names:
#   aime24, aime25, math_500, amc23
#
# Evaluate all configured benchmarks:
EVAL_BENCHMARKS=("all")
# Or evaluate selected benchmarks:
# EVAL_BENCHMARKS=("aime24" "math_500")

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

EVAL_URL="${EVAL_URL:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_output/opd_math/${RUN_TAG}}"

REPEATS="${REPEATS:-16}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-31744}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_NUM_PROC="${EVAL_NUM_PROC:-8}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
if [[ -z "${SYSTEM_PROMPT}" ]]; then
    SYSTEM_PROMPT='A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. The assistant must first provide the reasoning process inside <think> </think> tags, and then provide the final answer inside <answer> </answer> tags. Put the final mathematical answer inside \boxed{} within the <answer> tag. The required format is: <think> reasoning here </think><answer> \boxed{final answer} </answer>'
fi
ENABLE_THINKING="${ENABLE_THINKING:-false}"
DRY_RUN="${DRY_RUN:-false}"

contains_target() {
    local name="$1"
    local target
    for target in "${EVAL_TARGETS[@]}"; do
        if [[ "${target}" == "all" || "${target}" == "${name}" ]]; then
            return 0
        fi
    done
    return 1
}

contains_benchmark() {
    local name="$1"
    local benchmark
    for benchmark in "${EVAL_BENCHMARKS[@]}"; do
        if [[ "${benchmark}" == "all" || "${benchmark}" == "${name}" ]]; then
            return 0
        fi
    done
    return 1
}

resolve_benchmarks() {
    EVAL_DATASETS=()
    DATASETS_JSON_ARGS="{}"

    if contains_benchmark "aime24"; then
        EVAL_DATASETS+=("aime24")
    fi
    if contains_benchmark "aime25"; then
        EVAL_DATASETS+=("aime25")
    fi
    if contains_benchmark "math_500"; then
        EVAL_DATASETS+=("math_500")
    fi
    if contains_benchmark "amc23"; then
        EVAL_DATASETS+=("amc")
        DATASETS_JSON_ARGS='{"amc":{"subset_list":["amc23"]}}'
    fi

    if [[ "${#EVAL_DATASETS[@]}" -eq 0 ]]; then
        echo "No benchmark matched EVAL_BENCHMARKS: ${EVAL_BENCHMARKS[*]}" >&2
        exit 1
    fi
}

run_one_model() {
    local name="$1"
    local model="$2"
    local adapters="$3"

    local output_dir="${OUTPUT_ROOT}/${name}"
    local cmd=(
        python3 scripts/grpo/grade_gated/eval_math_benchmarks.py
        --model "${model}"
        --datasets "${EVAL_DATASETS[@]}"
        --dataset-args "${DATASETS_JSON_ARGS}"
        --output-dir "${output_dir}"
        --repeats "${REPEATS}"
        --seed "${SEED}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --max-tokens "${MAX_TOKENS}"
        --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
        --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
        --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --eval-num-proc "${EVAL_NUM_PROC}"
        --infer-backend "${INFER_BACKEND}"
        --torch-dtype "${TORCH_DTYPE}"
        --system "${SYSTEM_PROMPT}"
        --enable-thinking "${ENABLE_THINKING}"
    )

    if [[ -n "${adapters}" ]]; then
        # shellcheck disable=SC2206
        local adapter_args=(${adapters})
        cmd+=(--adapters "${adapter_args[@]}")
    fi

    if [[ -n "${EVAL_URL}" ]]; then
        cmd+=(--eval-url "${EVAL_URL}")
    fi

    if [[ -n "${EVAL_LIMIT}" ]]; then
        cmd+=(--eval-limit "${EVAL_LIMIT}")
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        cmd+=(--dry-run)
    fi

    echo "[opd-eval] model=${name}"
    echo "[opd-eval] path=${model}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

resolve_benchmarks

matched=0
for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r name model adapters <<< "${spec}"
    if contains_target "${name}"; then
        matched=$((matched + 1))
        run_one_model "${name}" "${model}" "${adapters}"
    fi
done

if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched EVAL_TARGETS: ${EVAL_TARGETS[*]}" >&2
    exit 1
fi
