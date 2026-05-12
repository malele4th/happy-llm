#!/usr/bin/env bash
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"

pretrain_log_file="logs/ddp_pretrain_${timestamp}.log"
sft_log_file="logs/ddp_sft_full_${timestamp}.log"

# # 预训练
# echo "$(date) [start pretrain]"
# nohup python ddp_pretrain.py --use_swanlab > ${pretrain_log_file} 2>&1 &
# echo "$(date) [end pretrain]"

# 监督微调训练
echo "$(date) [start sft train]"
nohup python ddp_sft_full.py --use_swanlab > ${sft_log_file} 2>&1 &
echo "$(date) [end sft train]"

