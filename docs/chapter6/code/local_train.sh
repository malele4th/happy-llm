#!/usr/bin/env bash
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"

pretrain_log_file="logs/pretrain_${timestamp}.log"
sft_log_file="logs/sft_${timestamp}.log"

# # 预训练
# echo "$(date) [start pretrain]"
# nohup bash pretrain.sh > ${pretrain_log_file} 2>&1 &
# echo "$(date) [end pretrain]"

echo "$(date) [start sft]"
nohup bash finetune.sh > ${sft_log_file} 2>&1 &
echo "$(date) [end sft]"

