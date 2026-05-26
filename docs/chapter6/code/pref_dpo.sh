echo "[$(date)] start DPO training"

# 若偏好数据不存在，先从 SFT 对话数据构造
if [[ ! -f data/toy_dpo_preferences.jsonl ]]; then
    echo "[$(date)] 生成 DPO 偏好数据集..."
    python build_dpo_dataset.py \
        --input_path data/toy_train_3.5M_CN.json \
        --output_path data/toy_dpo_preferences.jsonl \
        --max_samples 800
fi

# 基座模型：优先 SFT/LoRA 产物，否则回退到 pretrain
MODEL_PATH="output/sft-lora"
if [[ ! -d "${MODEL_PATH}" ]]; then
    MODEL_PATH="output/pretrain"
    echo "[$(date)] 未找到 output/sft-lora，使用 ${MODEL_PATH} 作为 DPO 基座"
fi

COMMON_ARGS=(
    --model_name_or_path "${MODEL_PATH}"
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
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --max_samples 100 \
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
