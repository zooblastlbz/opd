# GRADE-Gated with OPSD/self-distill
# Requires a dataset pipeline that adds the `teacher_prompt` column.
# Reference plugin: examples/train/rlhf/opsd/opsd_plugin.py

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --loss_type grade_gated \
    --model Qwen/Qwen3-4B \
    --teacher_model Qwen/Qwen3-4B \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --dataset open-r1/OpenThoughts-114k-math \
    --external_plugins examples/train/rlhf/opsd/opsd_plugin.py \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --steps_per_generation 1 \
    --grade_alpha_granularity group \
    --grade_alpha_mapping linear \
    --grade_reward_norm fixed_range \
    --grade_reward_min 0.0 \
    --grade_reward_max 1.0 \
    --grade_opd_loss_type forward_kl \
    --beta 0.04 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --gradient_checkpointing true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --deepspeed zero2 \
    --log_completions true
