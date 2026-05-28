'''
从 SFT 对话数据构造 GRPO 训练所需数据集

输出:
1. 偏好 jsonl（训练奖励模型）: {"prompt", "chosen", "rejected"}
2. prompt jsonl（GRPO 在线采样）: {"prompt"}
'''

import argparse
import json
from pathlib import Path

from build_dpo_dataset import build_dataset


def build_prompts(preferences_path: str, output_path: str, max_samples: int | None = None) -> int:
    """从偏好数据提取去重 prompt 列表"""
    seen: set[str] = set()
    prompts: list[dict] = []

    with open(preferences_path, encoding="utf-8") as f:
        for line in f:
            prompt = json.loads(line)["prompt"].strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append({"prompt": prompt})

    if max_samples is not None:
        prompts = prompts[:max_samples]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in prompts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(prompts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data/toy_train_3.5M_CN.json",
        help="SFT 对话 jsonl 路径",
    )
    parser.add_argument(
        "--preferences_output",
        default="data/toy_grpo_preferences.jsonl",
        help="GRPO 奖励模型偏好 jsonl 输出路径",
    )
    parser.add_argument(
        "--prompts_output",
        default="data/toy_grpo_prompts.jsonl",
        help="GRPO 训练 prompt jsonl 输出路径",
    )
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pref_count = build_dataset(
        args.input_path,
        args.preferences_output,
        args.max_samples,
        args.seed,
    )
    prompt_count = build_prompts(args.preferences_output, args.prompts_output, args.max_samples)
    print(f"已写入 {pref_count} 条偏好样本 -> {args.preferences_output}")
    print(f"已写入 {prompt_count} 条 prompt -> {args.prompts_output}")


if __name__ == "__main__":
    main()
