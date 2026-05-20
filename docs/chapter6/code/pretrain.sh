echo "[$(date)] start training"

COMMON_ARGS=(
    --config_name autodl_model/qwen-1.5b
    --tokenizer_name autodl_model/qwen-1.5b
    --train_files data/toy_mobvoi_seq_monkey_general_open_corpus.jsonl
    --gradient_accumulation_steps 4
    --do_train
    --output_dir output/pretrain
    --eval_strategy no
    --learning_rate 1e-4
    --num_train_epochs 1
    --warmup_steps 50
    --logging_dir output/pretrain/logs
    --logging_strategy steps
    --logging_steps 5
    --save_strategy steps
    --save_steps 100
    --save_total_limit 1
    --seed 12
    --block_size 256
    --gradient_checkpointing
    --report_to swanlab
)

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS：无 CUDA，DeepSpeed FusedAdam 等算子不支持 CPU/MPS
    echo "[$(date)] macOS 本地训练（单卡 MPS，不使用 DeepSpeed）"
    python pretrain.py "${COMMON_ARGS[@]}" \
        --per_device_train_batch_size 4 \
        --preprocessing_num_workers 0
else
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[$(date)] GPU 服务器训练（DeepSpeed ZeRO-2）"
    deepspeed pretrain.py "${COMMON_ARGS[@]}" \
        --per_device_train_batch_size 16 \
        --preprocessing_num_workers 10 \
        --bf16 \
        --deepspeed ./ds_config_zero2.json
        # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
fi
    

echo "[$(date)] end training"
