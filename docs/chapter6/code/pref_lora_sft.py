'''
LoRA SFT 脚本
基于 finetune.py，使用 peft 进行高效微调
'''

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from torchdata.datapipes.iter import IterableWrapper

import datasets
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import swanlab

from finetune import DataTrainingArguments, ModelArguments, SupervisedDataset


logger = logging.getLogger(__name__)


@dataclass
class LoraArguments:
    """LoRA 相关参数"""

    lora_r: int = field(default=8, metadata={"help": "LoRA 秩"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA 缩放系数"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "注入 LoRA 的模块名，逗号分隔；Qwen2 注意力层常用 q/k/v/o_proj"},
    )


def build_lora_config(lora_args: LoraArguments) -> LoraConfig:
    target_modules = [m.strip() for m in lora_args.target_modules.split(",") if m.strip()]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoraArguments, TrainingArguments)
    )
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    swanlab.init(project="sft-lora", experiment_name="qwen-1.5b-lora")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"LoRA parameters {lora_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"输出路径 ({training_args.output_dir}) 非空")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"从 {last_checkpoint} 恢复训练")

    set_seed(training_args.seed)

    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"基座模型 - Total size={n_params/2**20:.2f}M params")

    peft_config = build_lora_config(lora_args)
    logger.info(f"LoRA 配置：{peft_config}")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info("完成 tokenzier 加载")
    logger.info(f"tokenzier 配置地址：{tokenizer_path}")

    ds = load_dataset("json", data_files=data_args.train_files)
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练文件总数:{len(ds["train"])}')
    logger.info(f"训练集采样-第一条数据：{ds['train'][0]}")

    is_local_test = sys.platform == "darwin"
    raw_data = ds["train"]
    if is_local_test:
        max_samples = min(len(raw_data), 1000)
        raw_data = raw_data.select(range(max_samples))
        logger.info(f"macOS 本地测试，使用前 {max_samples} 条样本")

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning("tokenizer 支持大于 1K 的上下文长度，默认设置为 1K")
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"设定的块长为 ({data_args.block_size}) ，大于模型的上下文长度"
                f"将块长设置为模型上下文长度：{tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    with training_args.main_process_first(desc="SFT 数据预处理"):
        train_dataset = SupervisedDataset(raw_data, tokenizer=tokenizer, max_len=block_size)
        logger.info("完成数据预处理")

    if training_args.deepspeed is not None:
        train_dataset = IterableWrapper(train_dataset)

    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.info("开始 LoRA 训练")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    logger.info(f"LoRA adapter 已保存至 {training_args.output_dir}")


if __name__ == "__main__":
    main()
