'''
从 SFT 对话数据构造 DPO 偏好数据集

输出格式（jsonl，每行一条）:
{"prompt": "...", "chosen": "...", "rejected": "..."}

生成的 jsonl 里，每一行就是一条 DPO 偏好样本：1 个 prompt + 1 个 chosen + 1 个 rejected

'''

import argparse
import json
import random
from pathlib import Path


REJECTED_TEMPLATES = [
    "我不知道。",
    "这个问题太复杂了，无法回答。",
    "请自行搜索相关资料。",
    "抱歉，我不方便回答这个问题。",
    "嗯，好的。",
    "可以，没问题。（未给出具体说明）",
]


def make_rejected(chosen: str, rng: random.Random) -> str:
    """构造质量较差的 rejected 回复"""
    strategy = rng.randint(0, 3) # 从 0,1,2,3中随机选择一个数

    if strategy == 0:
        # 从rejected_templates中随机选择一个返回, 属于简单负例
        return rng.choice(REJECTED_TEMPLATES)
    if strategy == 1 and len(chosen) > 40:
        # 和 chosen 很像，但不完整，属于“中等难度”负例
        # 在 max_length 较小时，截断后容易和 chosen 几乎一样
        cut = max(20, len(chosen) // 4)
        return chosen[:cut].rstrip() + "……（后续省略）"
    if strategy == 2 and len(chosen) > 20:
        # 空洞复述
        # 把 chosen 前 15 个字塞进固定套话：有内容但空洞、不具体，比模板稍难区分。
        return f"关于你的问题，我的看法是：{chosen[:15]}……总之就是这样。"
    return rng.choice(REJECTED_TEMPLATES)


def extract_pairs(conversations: list) -> list[dict]:
    pairs = []
    i = 0
    while i < len(conversations) - 1:
        if conversations[i]["from"] == "human" and conversations[i + 1]["from"] == "assistant":
            prompt = conversations[i]["value"].strip() # human(用户)的回复 作为 prompt
            chosen = conversations[i + 1]["value"].strip() # assistant(助手)的回复 作为 chosen
            if prompt and chosen:
                pairs.append({"prompt": prompt, "chosen": chosen})
            i += 2
        else:
            i += 1
    return pairs


def build_dataset(
    input_path: str,
    output_path: str,
    max_samples: int = 800,
    seed: int = 42,
) -> int:
    rng = random.Random(seed) # rng 不是随机数本身，而是“按 seed 固定下来的随机数发生器”，专门用来可复现地生成 rejected 和 shuffle 数据
    samples = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for pair in extract_pairs(item["conversations"]):
                samples.append(
                    {
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": make_rejected(pair["chosen"], rng),
                    }
                )

    rng.shuffle(samples)
    samples = samples[:max_samples]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="data/toy_train_3.5M_CN.json",
        help="SFT 对话 jsonl 路径",
    )
    parser.add_argument(
        "--output_path",
        default="data/toy_dpo_preferences.jsonl",
        help="DPO 偏好 jsonl 输出路径",
    )
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = build_dataset(args.input_path, args.output_path, args.max_samples, args.seed)
    print(f"已写入 {count} 条 DPO 样本 -> {args.output_path}")


if __name__ == "__main__":
    main()
