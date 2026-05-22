'''
SFT 脚本
'''

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torchdata.datapipes.iter import IterableWrapper

import datasets
import torch
from datasets import load_dataset
import transformers
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
from torch.utils.data import Dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


# 超参类
@dataclass
class ModelArguments:
    """
    关于模型的参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "预训练模型参数地址"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Tokenizer 地址，默认与 model_name_or_path 相同"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """

    train_files: Optional[List[str]] = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "最大文本块长度"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理使用线程数."},
    )

# 指令文本处理
# 参考：https://github.com/QwenLM/Qwen/blob/main/finetune.py
def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    # prompt 模板
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # 不同的 tokenizer 需要特别定义
    # BOS
    im_start = tokenizer("<|im_start|>").input_ids

    # EOS
    im_end = tokenizer("<|im_end|>").input_ids

    # PAD
    IGNORE_TOKEN_ID = tokenizer.pad_token_id

    # 换行符
    nl_tokens = tokenizer('\n').input_ids

    # 角色标识符
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('human').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # 拼接多个样本
    input_ids, targets = [], []
    for i in tqdm(range(len(sources))):
        source = sources[i]

        # 从 user 开始
        # 如果第一个角色不是 user, 则从第二个角色开始
        if source[0]["from"] != "human":
            source = source[1:]
        
        # 分别是输入和输出, 单个样本的输入和输出
        input_id, target = [], []

        # system: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n  # 固定格式
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system

        # system 不需要拟合
        # 只有首尾的 im_start, im_end, nl_tokens 需要计算loss, 其余全部mask, 相当于在学 chat template 的结构
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        assert len(input_id) == len(target)

        # 依次拼接多轮对话
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]

            # user: <|im_start|>human\n{instruction}<|im_end|>\n
            # assistant: <|im_start|>assistant\n{response}<|im_end|>\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id

            if role == '<|im_start|>human':
                # user 不需要拟合
                # 只有首尾的 im_start, im_end, nl_tokens 需要计算loss, 其余全部mask, 相当于在学 chat template 的结构
                # <|im_start|>human 后面的 \n 也不算loss
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant 需要拟合
                # 只有 assistant\n 两个被mask掉了
                # [len(tokenizer(role).input_ids)+1:-2] 相当于从 assistant\n(不含) 开始到 <|im_end|>(不含) 之间的部分
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                print(role)
                raise NotImplementedError

            target += _target

        assert len(input_id) == len(target)

        # 最后进行 PAD & 截断到 max_len
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)
    logger.info(f"input_ids.shape: {input_ids.shape}")
    logger.info(f"targets.shape: {targets.shape}")
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # ne()标识不等于, 不是pad的为True, pad的被mask
    )


class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 加载并预处理数据
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def main():

    # 加载脚本参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    swanlab.init(project="sft", experiment_name="qwen-1.5b")
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )

    # 设置随机数种子.
    set_seed(training_args.seed)

    # 初始化模型
    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("完成 tokenzier 加载")

    # 加载微调数据
    ds = load_dataset('json', data_files=data_args.train_files)
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练文件总数:{len(ds["train"])}')
    logger.info(f"训练集采样-第一条数据：{ds['train'][0]}")

    # macOS 本地测试时限制样本数，降低内存占用
    is_local_test = sys.platform == "darwin"
    raw_data = ds["train"]
    if is_local_test:
        max_samples = min(len(raw_data), 1000)
        raw_data = raw_data.select(range(max_samples))  # 选择前 max_samples 条样本
        logger.info(f"macOS 本地测试，使用前 {max_samples} 条样本")

    # 确定 block_size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "tokenizer 支持大于 1K 的上下文长度，默认设置为 1K"
            )
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

    # # DeepSpeed 需要 IterableWrapper；本地单卡直接用 Dataset
    # if training_args.deepspeed is not None:
    #     train_dataset = IterableWrapper(train_dataset)

    # logger.info("初始化 Trainer")
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     processing_class=tokenizer,
    # )

    # # 从 checkpoint 加载
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint

    # logger.info("开始训练")
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model() 

if __name__ == "__main__":
    main()
