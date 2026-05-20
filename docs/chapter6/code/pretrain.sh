echo "[$(date)] start training"

CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name autodl_model/qwen-1.5b \
    --tokenizer_name autodl_model/qwen-1.5b \
    --train_files data/toy_mobvoi_seq_monkey_general_open_corpus.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir output/pretrain \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 50 \
    --logging_dir output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 256 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \

echo "[$(date)] end training"
