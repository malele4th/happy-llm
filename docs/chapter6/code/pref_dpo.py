'''
DPO 偏好对齐脚本
基于 pref_lora_sft.py，使用 TRL DPOTrainer 进行 Direct Preference Optimization
'''

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer
import swanlab

from finetune import ModelArguments
from pref_lora_sft import LoraArguments, build_lora_config


logger = logging.getLogger(__name__)


def resolve_torch_dtype(model_args, training_args):
    """解析模型加载 dtype：macOS 默认 float16，GPU bf16 训练时用 bfloat16"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if model_args.torch_dtype:
        if model_args.torch_dtype == "auto":
            return "auto"
        return dtype_map[model_args.torch_dtype]
    if sys.platform == "darwin":
        return torch.float16
    if getattr(training_args, "bf16", False):
        return torch.bfloat16
    return torch.float16


@dataclass
class DPODataArguments:
    """DPO 数据参数"""

    train_files: Optional[List[str]] = field(
        default=None, metadata={"help": "DPO 偏好数据路径（jsonl，含 prompt/chosen/rejected）"}
    )
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "最多使用多少条样本，本地测试时可限制"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DPODataArguments, LoraArguments, DPOConfig))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # DPO 需要保留 prompt/chosen/rejected 列
    training_args.remove_unused_columns = False
    if sys.platform == "darwin":
        training_args.bf16 = False

    swanlab.init(project="dpo", experiment_name="qwen-1.5b-dpo")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())

    logger.info(f"DPO training args: {training_args}")
    logger.info(f"LoRA args: {lora_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"输出路径 ({training_args.output_dir}) 非空")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"从 {last_checkpoint} 恢复训练")

    set_seed(training_args.seed)

    torch_dtype = resolve_torch_dtype(model_args, training_args)
    logger.info(f"加载策略模型：{model_args.model_name_or_path}，dtype={torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch_dtype,
    )
    model = get_peft_model(model, build_lora_config(lora_args))
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=data_args.train_files)
    train_dataset = ds["train"]
    logger.info(f"DPO 数据集：{data_args.train_files}，共 {len(train_dataset)} 条")
    logger.info(f"样本示例：{train_dataset[0]}")

    if data_args.max_samples is not None:
        n = min(len(train_dataset), data_args.max_samples)
        train_dataset = train_dataset.select(range(n))
        logger.info(f"限制使用前 {n} 条样本")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    logger.info("开始 DPO 训练")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    logger.info(f"DPO LoRA adapter 已保存至 {training_args.output_dir}")


if __name__ == "__main__":
    main()
