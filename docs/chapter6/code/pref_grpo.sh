echo "[$(date)] start GRPO pipeline"

# 1. 构造 GRPO 偏好数据 + prompt 数据
echo "[$(date)] 生成 GRPO 数据集..."
python build_grpo_reward_dataset.py \
    --input_path data/toy_train_3.5M_CN.json \
    --preferences_output data/toy_grpo_preferences.jsonl \
    --prompts_output data/toy_grpo_prompts.jsonl \
    --max_samples 3000

# 2. 训练 GRPO 奖励模型
REWARD_COMMON_ARGS=(
    --model_name_or_path output/pretrain
    --tokenizer_name output/pretrain
    --train_files data/toy_grpo_preferences.jsonl
    --do_train
    --output_dir output/grpo/reward_model
    --eval_strategy no
    --learning_rate 1e-4
    --num_train_epochs 1
    --warmup_steps 20
    --logging_dir output/grpo/reward_model/logs
    --logging_strategy steps
    --logging_steps 5
    --save_strategy steps
    --save_steps 50
    --save_total_limit 1
    --seed 12
    --max_length 32
    --gradient_checkpointing
    --report_to swanlab
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --target_modules q_proj,k_proj,v_proj,o_proj
)

echo "[$(date)] 训练 GRPO 奖励模型..."
if [[ "$(uname)" == "Darwin" ]]; then
    python pref_grpo_reward.py "${REWARD_COMMON_ARGS[@]}" \
        --torch_dtype float16 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --dataloader_pin_memory false
else
    export CUDA_VISIBLE_DEVICES=0,1
    deepspeed pref_grpo_reward.py "${REWARD_COMMON_ARGS[@]}" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --bf16 \
        --deepspeed ./ds_config_zero2.json
fi

# 3. GRPO 策略对齐（基座 pretrain + LoRA，奖励模型来自上一步）
GRPO_COMMON_ARGS=(
    --model_name_or_path output/pretrain
    --tokenizer_name output/pretrain
    --train_files data/toy_grpo_prompts.jsonl
    --reward_model_path output/grpo/reward_model
    --max_samples 500
    --do_train
    --output_dir output/grpo/policy
    --eval_strategy no
    --learning_rate 2e-4
    --num_train_epochs 1
    --warmup_steps 10
    --logging_dir output/grpo/policy/logs
    --logging_strategy steps
    --logging_steps 5
    --save_strategy steps
    --save_steps 50
    --save_total_limit 1
    --seed 12
    --loss_type grpo
    --epsilon 0.0001
    --beta 0.04
    --num_generations 2
    --num_iterations 4
    --max_completion_length 32
    --temperature 0.7
    --gradient_checkpointing
    --report_to swanlab
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --target_modules q_proj,k_proj,v_proj,o_proj
)

echo "[$(date)] 开始 GRPO 策略训练..."
if [[ "$(uname)" == "Darwin" ]]; then
    echo "[$(date)] macOS 本地训练（单卡 MPS，不使用 DeepSpeed）"
    python pref_grpo.py "${GRPO_COMMON_ARGS[@]}" \
        --torch_dtype float16 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --dataloader_pin_memory false
else
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[$(date)] GPU 服务器训练（DeepSpeed ZeRO-2）"
    deepspeed pref_grpo.py "${GRPO_COMMON_ARGS[@]}" \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --num_generations 4 \
        --bf16 \
        --deepspeed ./ds_config_zero2.json
fi

echo "[$(date)] end GRPO pipeline"
