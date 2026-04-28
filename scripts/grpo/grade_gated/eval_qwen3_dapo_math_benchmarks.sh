#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/ytech_m2v8_hdd/workspace/kling_mm/libozhou/miniconda/envs/ms}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-/ytech_m2v8_hdd/workspace/kling_mm/libozhou/miniconda/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"

if [[ -f "${CONDA_ACTIVATE}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_ACTIVATE}" "${CONDA_ENV_PATH}" >/dev/null 2>&1 || {
        echo "Failed to activate conda environment: ${CONDA_ENV_PATH}" >&2
        exit 1
    }
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

# Sequential multi-model, single-node 8-GPU OPD math evaluation.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Format: "name|model_path|adapters"
# For full-parameter checkpoints, leave adapters empty.
# For LoRA checkpoints, put the base model in model_path and one or more
# space-separated adapter paths in adapters.
MODEL_SPECS=(
    #"8B_teacher|/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/output/Qwen3-8B-DAPO-DAPO-Math-17k-format-len8192/v9-20260424-124744/checkpoint-271|"
    "zero_teacher|/ytech_m2v5_hdd/workspace/kling_mm/libozhou/opd/output/Qwen3-8B-DAPO-DAPO-Math-17k-hard-format-len8192_zero_rl/v3-20260427-004019/checkpoint-271|"
    # "gspo_ckptXXX|/path/to/gspo/checkpoint-XXX|"
    # "grpo_ckptXXX|/path/to/grpo/checkpoint-XXX|"
    # "lora_exp|/path/to/base/model|/path/to/lora/checkpoint"
)

# Evaluate all configured models:
EVAL_TARGETS_STR="${EVAL_TARGETS:-all}"
read -r -a EVAL_TARGETS <<< "${EVAL_TARGETS_STR}"

REPEATS="${REPEATS:-16}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
MAX_TOKENS="${MAX_TOKENS:-31744}"

VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"

EVAL_NUM_PROC="${EVAL_NUM_PROC:-64}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
DRY_RUN="${DRY_RUN:-false}"
PORT="${PORT:-8000}"
EVAL_URL="${EVAL_URL:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_output/opd_math/${RUN_TAG}}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
if [[ -z "${SYSTEM_PROMPT}" ]]; then
    SYSTEM_PROMPT='A conversation between User and Assistant. The user asks a math question, and the Assistant solves it. The assistant must first provide the reasoning process inside <think> </think> tags, and then provide the final answer inside <answer> </answer> tags. Put the final mathematical answer inside \boxed{} within the <answer> tag. The required format is: <think> reasoning here </think><answer> \boxed{final answer} </answer>'
fi
ENABLE_THINKING="${ENABLE_THINKING:-false}"

# Supported names: aime24, aime25, math_500, amc23
EVAL_BENCHMARKS_STR="${EVAL_BENCHMARKS:-all}"
read -r -a EVAL_BENCHMARKS <<< "${EVAL_BENCHMARKS_STR}"

resolve_benchmarks() {
    EVAL_DATASETS=()
    DATASET_ARGS_JSON="{}"

    local benchmark
    for benchmark in "${EVAL_BENCHMARKS[@]}"; do
        case "${benchmark}" in
            all)
                EVAL_DATASETS=("aime24" "aime25" "math_500" "amc")
                DATASET_ARGS_JSON='{"amc":{"subset_list":["amc23"]}}'
                return
                ;;
            aime24)
                EVAL_DATASETS+=("aime24")
                ;;
            aime25)
                EVAL_DATASETS+=("aime25")
                ;;
            math_500)
                EVAL_DATASETS+=("math_500")
                ;;
            amc23)
                EVAL_DATASETS+=("amc")
                DATASET_ARGS_JSON='{"amc":{"subset_list":["amc23"]}}'
                ;;
            *)
                echo "Unsupported benchmark: ${benchmark}" >&2
                exit 1
                ;;
        esac
    done

    if [[ "${#EVAL_DATASETS[@]}" -eq 0 ]]; then
        echo "No benchmark selected." >&2
        exit 1
    fi
}

resolve_benchmarks

IFS=',' read -r -a CUDA_DEVICE_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#CUDA_DEVICE_ARRAY[@]}" -ne "${VLLM_TENSOR_PARALLEL_SIZE}" ]]; then
    echo "CUDA_VISIBLE_DEVICES count (${#CUDA_DEVICE_ARRAY[@]}) must equal VLLM_TENSOR_PARALLEL_SIZE (${VLLM_TENSOR_PARALLEL_SIZE})." >&2
    exit 1
fi

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

run_one_model() {
    local name="$1"
    local model_path="$2"
    local adapters="$3"
    local output_dir="${OUTPUT_ROOT}/${name}"
    local cmd=(
        "${PYTHON_BIN}" scripts/grpo/grade_gated/eval_math_benchmarks.py
        --model "${model_path}"
        --datasets "${EVAL_DATASETS[@]}"
        --dataset-args "${DATASET_ARGS_JSON}"
        --output-dir "${output_dir}"
        --repeats "${REPEATS}"
        --seed "${SEED}"
        --port "${PORT}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --max-tokens "${MAX_TOKENS}"
        --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
        --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
        --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
        --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}"
        --vllm-enable-prefix-caching "${VLLM_ENABLE_PREFIX_CACHING}"
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

    echo "[opd-eval] model_name=${name}"
    echo "[opd-eval] model_path=${model_path}"
    echo "[opd-eval] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "[opd-eval] vllm_tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE}"
    echo "[opd-eval] output_dir=${output_dir}"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${cmd[@]}"
}

matched=0
for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r name model_path adapters <<< "${spec}"
    if contains_target "${name}"; then
        matched=$((matched + 1))
        run_one_model "${name}" "${model_path}" "${adapters}"
    fi
done

if [[ "${matched}" -eq 0 ]]; then
    echo "No model matched EVAL_TARGETS: ${EVAL_TARGETS[*]}" >&2
    exit 1
fi
