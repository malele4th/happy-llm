分词器（Tokenizer）是语言模型处理文本的第一道关卡，直接影响模型的词汇表质量和下游任务性能。本文档基于 Tiny-K 框架的 `train_tokenizer.py` 实现，详解 BPE（Byte Pair Encoding）分词器的完整训练流程，从数据准备到配置部署，助你构建适配特定业务场景的专属分词器。

## BPE 算法原理概述

BPE 是一种基于频率的子词分词算法，核心思想是通过迭代合并最频繁出现的字符对来构建词汇表。相较于字符级分词，BPE 能在词级与字符级之间取得平衡，有效处理未登录词（OOV）问题；相较于词级分词，BPE 的词汇表规模可控，不会因词表过大导致嵌入矩阵膨胀。

BPE 训练流程包含三个阶段：**初始化阶段**建立基础字符词汇表；**迭代合并阶段**统计所有字符对频率，重复合并最高频的对直至达到目标词表大小；**编码解码阶段**将文本分割为已知子词并支持还原。在 Tiny-K 框架中，采用 `tokenizers` 库实现 BPE 算法，该库基于 Rust 实现，训练效率远高于纯 Python 实现。

Sources: [train_tokenizer.py#L77-L108](train_tokenizer.py#L77-L108)

## 数据准备与格式规范

BPE 分词器的训练质量高度依赖输入数据的质量和规模。在 Tiny-K 框架中，训练数据采用 JSONL（JSON Lines）格式，每行包含一个 JSON 对象，必须包含 `text` 字段存储待分词的文本内容。

```json
{"text": "今天天气真好，适合出去散步。"}
{"text": "人工智能技术正在改变我们的生活方式。"}
```

数据准备流程在 `deal_dataset.py` 中实现。对于预训练数据，脚本将长文本按固定长度切分，确保每条数据适合 BPE 学习子词边界；对于对话数据，脚本将原始对话格式转换为标准消息列表格式。分词长度建议控制在 256-1024 字符之间，过短会导致上下文不足，过长则增加训练负担。

Sources: [deal_dataset.py#L14-L26](deal_dataset.py#L14-L26)

```python
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

数据量的选择需根据业务场景调整：通用领域建议至少 100MB 文本，垂直领域建议 50MB 以上。数据应涵盖目标场景的各类词汇、表达方式和特殊符号。

## 分词器核心配置详解

`train_tokenizer.py` 中的 `train_tokenizer()` 函数是分词器训练的核心实现，包含模型初始化、规范化器配置、预分词器配置和训练器参数设置四个关键环节。

```python
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """训练并保存自定义tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
```

Sources: [train_tokenizer.py#L77-L85](train_tokenizer.py#L77-L85)

### 模型层配置

使用 `models.BPE` 初始化分词器模型，指定未知词 token 为 `<unk>`。BPE 模型会在训练过程中学习将任意文本分割为已知子词的组合，遇见完全未知的字符序列时用 `<unk>` 替代。

### 规范化器配置

采用 NFKC（Normalization Form Compatibility Composition）规范化方式，将字符转换为兼容组合形式。例如，全角字母会被转换为半角形式，"ﬁ" 会被分解为 "fi"。这确保了同一语义字符的不同编码形式被统一处理，减少词汇表冗余。

### 预分词器配置

使用 `ByteLevel` 预分词器，其工作流程为：首先将输入文本转换为 UTF-8 字节序列，再在字节级别进行 BPE 分词。这使得分词器天然支持多语言混合文本，无需为每种语言设计独立处理逻辑。`add_prefix_space=False` 表示不在文本开头添加空格，与 GPT 系列模型的惯例保持一致。

## 特殊 Token 体系设计

特殊 Token 用于标记句子边界、对话角色等语义元素。Tiny-K 框架设计了五类特殊 Token，ID 从 0 到 4 固定分配：

| Token | ID | 用途 |
|-------|-----|------|
| `<unk>` | 0 | 未知词标记，用于分词器无法识别的文本 |
| `<s>` | 1 | 句子开始标记（兼容 BERT 风格） |
| `</s>` | 2 | 句子结束标记（兼容 BERT 风格） |
| `<\|im_start\|>` | 3 | 对话消息开始标记 |
| `<\|im_end\|>` | 4 | 对话消息结束标记 |

Sources: [train_tokenizer.py#L88-L94](train_tokenizer.py#L88-L94)

训练完成后，代码通过断言验证特殊 Token 的 ID 分配是否符合预期：

```python
try:
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2
    assert tokenizer.token_to_id("<|im_start|>") == 3
    assert tokenizer.token_to_id("<|im_end|>") == 4
except AssertionError as e:
    print("Special tokens mapping error:", e)
    raise
```

Sources: [train_tokenizer.py#L111-L119](train_tokenizer.py#L111-L119)

## 训练器参数配置

`BpeTrainer` 是训练过程的核心配置类，各参数对分词效果的影响如下：

```python
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    min_frequency=2,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
```

Sources: [train_tokenizer.py#L97-L103](train_tokenizer.py#L97-L103)

**vocab_size** 控制最终词汇表大小，需在覆盖率与嵌入开销之间权衡。Tiny-K 框架默认配置为 6144，实际项目中可根据语料丰富度调整为 4096-32768。建议公式：`vocab_size ≈ sqrt(语料总字符数) * 2`，在 100MB 语料上约 8000-16000 较为合适。

**min_frequency** 设置子词合并的最低频率阈值，默认为 2。增大该值可过滤噪音子词、减小词表，但会降低对低频词的覆盖；降低该值则相反。数据量较小时建议设为 3-5 以避免过拟合。

**initial_alphabet** 限定初始字符集，使用 ByteLevel 的完整字母表确保 UTF-8 字节全覆盖。

## 分词器评估与验证

训练完成后通过 `eval_tokenizer()` 函数验证分词器功能，主要包含四个测试维度：

```python
def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
```

Sources: [train_tokenizer.py#L128-L134](train_tokenizer.py#L128-L134)

### 基本属性验证

```python
print("\n=== Tokenizer基本信息 ===")
print(f"Vocab size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.all_special_tokens}")
print(f"Special token IDs: {tokenizer.all_special_ids}")
```

验证词汇表大小是否符合预期，确认特殊 Token 识别正常。

### 聊天模板测试

Tiny-K 框架实现了 ChatML 格式的聊天模板，通过 Jinja2 语法定义消息格式：

```python
messages = [
    {"role": "system", "content": "你是一个AI助手。"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm fine, thank you. and you?"},
    {"role": "user", "content": "I'm good too."},
    {"role": "assistant", "content": "That's great to hear!"},
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,
)
```

Sources: [train_tokenizer.py#L143-L157](train_tokenizer.py#L143-L157)

输出格式为：

```
<|im_start|>system
你是一个AI助手。<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you. and you?<|im_end|>
<|im_start|>user
I'm good too.<|im_end|>
<|im_start|>assistant
That's great to hear!<|im_end|>
```

### 编码解码一致性测试

```python
print("\n=== 编码解码测试 ===")
encoded = tokenizer(prompt, truncation=True, max_length=256)
decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
print("Decoded text matches original:", decoded == prompt)
```

验证 `encode-decode` 往返一致性，确保分词-反分词过程无信息损失。

### 特殊 Token 保留测试

```python
print("\n=== 特殊token处理 ===")
test_text = "<|im_start|>user\nHello<|im_end|>"
encoded = tokenizer(test_text).input_ids
decoded = tokenizer.decode(encoded)
print("Special tokens preserved:", decoded == test_text)
```

确保特殊 Token 在编码过程中被正确保留，不被错误拆分或合并。

## 配置文件生成

分词器训练完成后需要生成两类配置文件以支持后续使用：

### tokenizer_config.json

主配置文件定义分词器行为参数：

```json
{
    "add_bos_token": false,
    "add_eos_token": false,
    "add_prefix_space": false,
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|im_end|>",
    "unk_token": "<unk>",
    "model_max_length": 1000000000000000019884624838656,
    "clean_up_tokenization_spaces": false,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "chat_template": "{% for message in messages %}..."
}
```

Sources: [train_tokenizer.py#L33-L60](train_tokenizer.py#L33-L60)

其中 `model_max_length` 设置为接近 Python 整数上限的值，表示模型支持极长序列输入；`tokenizer_class` 指定为 Fast 版本，支持快速批量编码。

### special_tokens_map.json

定义特殊 Token 的标准映射关系：

```json
{
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<unk>",
    "pad_token": "<|im_end|>",
    "additional_special_tokens": ["<s>", "</s>"]
}
```

Sources: [train_tokenizer.py#L67-L75](train_tokenizer.py#L67-L75)

## 完整训练流程

将上述组件整合，完整训练流程如下：

```python
def main():
    # 配置路径
    data_path = "your data path"
    save_dir = "tokenizer_k"

    # 训练tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()
```

Sources: [train_tokenizer.py#L174-L190](train_tokenizer.py#L174-L190)

执行训练前需确保安装了相关依赖：`transformers`、`tokenizers` 和 `tqdm`。训练完成后，`tokenizer_k` 目录下将生成三个文件：

```
tokenizer_k/
├── tokenizer.json        # 分词器模型文件
├── tokenizer_config.json  # 主配置文件
└── special_tokens_map.json
```

## 训练效果影响因素

分词器质量受多种因素影响，以下是关键优化策略：

**数据质量优先于数据数量**。高质量分词器需要文本清洗：去除 HTML 标签、统一标点格式、处理编码异常。脏数据会导致词汇表被噪音子词污染，降低有效覆盖率。

**领域适配是关键**。通用分词器在垂直领域表现往往不佳。例如，医疗文本中的专业术语、技术文档中的代码片段，都需要针对性的领域数据训练。可以在通用语料基础上增加领域语料权重进行增量训练。

**词汇表大小需实验确定**。过小的词汇表会导致长序列、增加计算开销；过大的词汇表则增加嵌入层参数量、可能过拟合。建议以 4096 为基准，向上向下各训练一个版本，通过下游任务验证选择最优。

## 后续学习路径

分词器训练完成后，建议继续学习以下内容：

- [数据集处理：JSONL 格式转换与分块](13-shu-ju-ji-chu-li-jsonl-ge-shi-zhuan-huan-yu-fen-kuai) — 深入了解数据预处理细节
- [预训练流程：数据加载与模型训练](8-yu-xun-lian-liu-cheng-shu-ju-jia-zai-yu-mo-xing-xun-lian) — 将分词器应用于模型训练
- [模型推理与文本生成](15-mo-xing-tui-li-yu-wen-ben-sheng-cheng) — 验证分词器在推理阶段的表现