echo "[$(date)] start LoRA training"

COMMON_ARGS=(
    --model_name_or_path output/pretrain
    --train_files data/toy_train_3.5M_CN.json
    --gradient_accumulation_steps 4
    --do_train
    --output_dir output/sft-lora
    --eval_strategy no
    --learning_rate 1e-4
    --num_train_epochs 3
    --warmup_steps 200
    --logging_dir output/sft-lora/logs
    --logging_strategy steps
    --logging_steps 5
    --save_strategy steps
    --save_steps 100
    --save_total_limit 1
    --seed 12
    --block_size 256
    --gradient_checkpointing
    --report_to swanlab
    --lora_r 8
    --lora_alpha 32
    --lora_dropout 0.1
    --target_modules q_proj,k_proj,v_proj,o_proj
)

if [[ "$(uname)" == "Darwin" ]]; then
    echo "[$(date)] macOS 本地训练（单卡 MPS，不使用 DeepSpeed）"
    python pref_lora_sft.py "${COMMON_ARGS[@]}" \
        --per_device_train_batch_size 4 \
        --preprocessing_num_workers 0
else
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[$(date)] GPU 服务器训练（DeepSpeed ZeRO-2）"
    deepspeed pref_lora_sft.py "${COMMON_ARGS[@]}" \
        --per_device_train_batch_size 16 \
        --preprocessing_num_workers 10 \
        --bf16 \
        --deepspeed ./ds_config_zero2.json
fi

echo "[$(date)] end LoRA training"
