# GRADE-Gated with a remote teacher server.
#
# Start the teacher server on a separate node first:
# CUDA_VISIBLE_DEVICES=0 \
# vllm serve Qwen/Qwen2.5-7B-Instruct \
#     --port 8000 \
#     --max-logprobs 64

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type grpo \
    --loss_type grade_gated \
    --model Qwen/Qwen2.5-7B-Instruct \
    --teacher_model_server http://teacher-host:8000 \
    --gkd_logits_topk 64 \
    --dataset AI-MO/NuminaMath-TIR#10000 \
    --reward_funcs accuracy format \
    --num_generations 8 \
    --steps_per_generation 1 \
    --grade_alpha_granularity sample \
    --grade_alpha_mapping sigmoid \
    --grade_alpha_sigmoid_temperature 10.0 \
    --grade_reward_norm ema \
    --grade_reward_ema_decay 0.99 \
    --grade_reward_ema_clip 3.0 \
    --grade_opd_loss_type reverse_kl \
    --beta 0.04 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --max_length 4096 \
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
    --vllm_max_model_len 8192 \
    --sleep_level 1 \
    --deepspeed zero2 \
    --log_completions true
