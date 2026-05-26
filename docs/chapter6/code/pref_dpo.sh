echo "[$(date)] start DPO training"

# 从 SFT 对话数据构造 DPO 偏好数据集（默认最多 3000 条）
echo "[$(date)] 生成 DPO 偏好数据集..."
python build_dpo_dataset.py \
    --input_path data/toy_train_3.5M_CN.json \
    --output_path data/toy_dpo_preferences.jsonl \
    --max_samples 3000

# 基座直接用 pretrain 权重，DPO 单独挂 LoRA（不叠 sft-lora）
COMMON_ARGS=(
    --model_name_or_path output/pretrain
    --tokenizer_name output/pretrain
    --train_files data/toy_dpo_preferences.jsonl
    --do_train
    --output_dir output/dpo
    --eval_strategy no
    --learning_rate 5e-5
    --num_train_epochs 1
    --warmup_steps 20
    --logging_dir output/dpo/logs
    --logging_strategy steps
    --logging_steps 5
    --save_strategy steps
    --save_steps 50
    --save_total_limit 1
    --seed 12
    --beta 0.1
    --max_length 16
    --gradient_checkpointing
    --report_to swanlab
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --target_modules q_proj,k_proj,v_proj,o_proj
)

if [[ "$(uname)" == "Darwin" ]]; then
    echo "[$(date)] macOS 本地训练（单卡 MPS，不使用 DeepSpeed）"
    python pref_dpo.py "${COMMON_ARGS[@]}" \
        --torch_dtype float16 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --dataloader_pin_memory false
else
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[$(date)] GPU 服务器训练（DeepSpeed ZeRO-2）"
    deepspeed pref_dpo.py "${COMMON_ARGS[@]}" \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --bf16 \
        --deepspeed ./ds_config_zero2.json
fi

echo "[$(date)] end DPO training"
