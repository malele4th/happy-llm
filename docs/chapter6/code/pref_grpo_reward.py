'''
GRPO 奖励模型训练脚本
基于 pref_dpo.py，使用 TRL RewardTrainer 训练 Outcome Reward Model
'''

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import RewardConfig, RewardTrainer
import swanlab

from finetune import ModelArguments
from pref_dpo import resolve_torch_dtype
from pref_lora_sft import LoraArguments


logger = logging.getLogger(__name__)


@dataclass
class GrpoRewardDataArguments:
    """GRPO 奖励模型数据参数"""

    train_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "偏好数据路径（jsonl，含 prompt/chosen/rejected）"},
    )
    max_samples: Optional[int] = field(
        default=None, metadata={"help": "最多使用多少条样本，本地测试时可限制"}
    )


def build_grpo_reward_lora_config(lora_args: LoraArguments) -> LoraConfig:
    """奖励模型 LoRA：额外保存 score 分类头"""
    target_modules = [m.strip() for m in lora_args.target_modules.split(",") if m.strip()]
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["score"],
    )


def save_merged_reward_model(trainer: RewardTrainer, output_dir: str, tokenizer) -> None:
    """合并 LoRA 并保存，便于 GRPOTrainer 直接用路径加载奖励模型"""
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        logger.info("合并 LoRA 奖励模型权重以便 GRPO 加载")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = HfArgumentParser((ModelArguments, GrpoRewardDataArguments, LoraArguments, RewardConfig))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    if sys.platform == "darwin":
        training_args.bf16 = False

    swanlab.init(project="grpo", experiment_name="qwen-1.5b-grpo-reward")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())

    logger.info(f"GRPO Reward training args: {training_args}")
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
    model_init_kwargs = {"trust_remote_code": True, "dtype": torch_dtype}
    training_args.model_init_kwargs = model_init_kwargs

    tokenizer_path = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=data_args.train_files)
    train_dataset = ds["train"]
    logger.info(f"奖励模型数据集：{data_args.train_files}，共 {len(train_dataset)} 条")
    logger.info(f"样本示例：{train_dataset[0]}")

    if data_args.max_samples is not None:
        n = min(len(train_dataset), data_args.max_samples)
        train_dataset = train_dataset.select(range(n))
        logger.info(f"限制使用前 {n} 条样本")

    trainer = RewardTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_grpo_reward_lora_config(lora_args),
    )

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    logger.info("开始 GRPO 奖励模型训练")
    trainer.train(resume_from_checkpoint=checkpoint)
    save_merged_reward_model(trainer, training_args.output_dir, tokenizer)
    logger.info(f"GRPO 奖励模型已保存至 {training_args.output_dir}")


if __name__ == "__main__":
    main()
