#!/usr/bin/env bash
mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/ddp_pretrain_${timestamp}.log"

echo "$(date) [start]"
nohup python ddp_pretrain.py --use_swanlab > ${log_file} 2>&1 &
echo "$(date) [end]"

