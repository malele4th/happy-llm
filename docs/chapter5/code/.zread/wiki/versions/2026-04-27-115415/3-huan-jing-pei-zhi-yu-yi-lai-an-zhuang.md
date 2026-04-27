本文档详细介绍 Tiny-K 语言模型训练框架的完整环境配置流程，涵盖硬件要求、软件环境搭建、依赖库安装以及数据集准备。通过本文档，开发者可以在本地机器上快速配置好训练环境，为后续的预训练和微调工作奠定基础。

## 硬件环境要求

### GPU 配置要求

Tiny-K 框架支持 GPU 加速训练和 CPU 推理两种模式。对于模型训练任务，建议配置 NVIDIA GPU 以获得显著的性能提升。框架内部通过 `torch.cuda.is_available()` 自动检测 CUDA 可用性，并智能选择计算设备。在多 GPU 环境下，框架支持 DataParallel 并行训练模式，能够有效利用多张显卡提升训练吞吐量。代码中通过 `gpus` 参数指定可用的 GPU 设备列表，例如 `--gpus 0,1,2,3` 表示使用前四张 GPU 进行训练。若系统仅有一颗 GPU，框架会自动降级为单卡训练模式，无需额外配置。

```python
# ddp_pretrain.py 中的设备检测逻辑
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
# 当指定 GPU 时自动设置主设备
if torch.cuda.is_available():
    args.device = "cuda:0"
```

Sources: [ddp_pretrain.py](ddp_pretrain.py#L255-L265)

### 内存与存储建议

预训练数据集通常占用较大的磁盘空间和内存资源。Seq-Monkey 预训练语料库解压后约为数 GB 级别，SFT 对话数据集同样需要预留充足的存储空间。在内存方面，建议系统配备 16GB 以上 RAM 以确保数据加载和模型加载过程的流畅性。分词器（tokenizer）模型存储在 `tokenizer_k/` 目录下，包含 `tokenizer.json`、`tokenizer_config.json` 和 `special_tokens_map.json` 三个核心配置文件，加载时会自动读取这些文件初始化分词器。

## 软件环境配置

### Python 版本要求

Tiny-K 框架基于 Python 3.8 及以上版本开发，充分利用了 Python 3.8 引入的多种新特性和性能优化。建议使用 Python 3.10 或更高版本以获得最佳兼容性。可以通过以下命令检查当前 Python 版本：

```bash
python3 --version
```

若版本低于 3.8，建议使用 conda 或 pyenv 等工具创建新的虚拟环境进行版本管理。框架使用 `transformers` 库处理预训练模型和分词器，`torch` 库作为核心深度学习引擎，这些依赖都对 Python 版本有一定要求，建议遵循官方推荐的版本配置。

### 虚拟环境创建

为避免依赖冲突，建议为项目创建独立的虚拟环境。可以使用 conda 或 Python 内置的 venv 模块创建隔离的运行环境。使用 conda 创建环境的命令如下：

```bash
conda create -n tiny-k python=3.10
conda activate tiny-k
```

使用 venv 的方式为：

```bash
python3 -m venv tiny-k-env
source tiny-k-env/bin/activate  # Linux/macOS
# tiny-k-env\Scripts\activate  # Windows
```

虚拟环境可以有效隔离不同项目的依赖版本，防止包冲突问题。每个项目使用独立的 Python 环境也便于管理和迁移。

## 核心依赖安装

### PyTorch 环境配置

PyTorch 是框架的核心依赖，负责张量运算、自动微分和 GPU 加速等底层功能。Tiny-K 框架需要 PyTorch 2.4.0 或更高版本以支持 Flash Attention 特性。安装 PyTorch 时需要根据系统和 CUDA 版本选择对应的安装包。推荐使用 conda 或 pip 安装预编译的 PyTorch 发行版：

```bash
# CUDA 11.8 版本
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 版本
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

可以通过以下代码验证 PyTorch 安装是否成功以及 CUDA 是否可用：

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
```

Sources: [k_model.py](k_model.py#L180-L195)

### Transformers 库安装

Transformers 库是 Hugging Face 生态系统的核心组件，提供了预训练模型加载、分词器管理和模型导出等功能。Tiny-K 框架依赖 Transformers 4.44.0 版本，该版本对自定义模型配置有良好的支持。安装命令如下：

```bash
pip install transformers==4.44.0
```

框架的模型配置类 `ModelConfig` 继承自 `PretrainedConfig`，分词器使用 `AutoTokenizer` 自动加载。模型导出功能依赖 Transformers 的 `save_pretrained` 方法将训练好的模型转换为 HuggingFace 格式。

Sources: [k_model.py](k_model.py#L1-L30)
Sources: [export_model.py](export_model.py#L1-L30)

## 项目依赖清单

### 核心训练依赖

项目所需的所有 Python 依赖都集中在 `requirements.txt` 文件中。通过该文件可以一次性安装所有依赖，确保环境一致性。

| 依赖包 | 版本 | 用途说明 |
|--------|------|----------|
| torch | 2.4.0 | 深度学习核心框架 |
| torchvision | 0.19.0 | 计算机视觉工具（可选） |
| transformers | 4.44.0 | 预训练模型和分词器管理 |
| datasets | 2.16.1 | 数据集加载和处理 |
| pandas | 1.5.3 | 数据分析处理 |
| numpy | 1.26.4 | 数值计算 |
| tiktoken | 0.5.1 | BPE 分词工具 |

Sources: [requirements.txt](requirements.txt#L1-L24)

### 数据处理依赖

数据处理管道涉及多种数据格式的读写和转换，主要依赖包括 `jsonlines` 用于流式读写 JSONL 格式数据集，`pydantic` 和 `marshmallow` 提供数据验证功能，`json` 模块作为标准库支持 JSON 序列化。这些依赖共同支撑了 `dataset.py` 中的 `PretrainDataset` 和 `SFTDataset` 两个核心数据类的运行。

```python
# dataset.py 中的数据加载示例
import json
class PretrainDataset(Dataset):
    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
```

Sources: [dataset.py](dataset.py#L1-L30)

### 实验跟踪与辅助工具

| 依赖包 | 版本 | 用途说明 |
|--------|------|----------|
| swanlab | 最新版 | 实验结果跟踪和可视化 |
| rich | 13.7.1 | 命令行美化输出 |
| matplotlib | 3.5.1 | 数据可视化（可选） |
| psutil | 5.9.8 | 系统资源监控 |

SwanLab 是框架集成的实验跟踪工具，能够记录训练过程中的损失值、学习率等关键指标，并生成可视化图表。使用前需要在 SwanLab 官网注册账号获取 API Key，并在代码中配置：

```python
import swanlab
run = swanlab.init(
    project="Happy-LLM",
    experiment_name="Pretrain-215M",
    config=args
)
```

Sources: [ddp_pretrain.py](ddp_pretrain.py#L10-L15)

## 一键安装配置

### 安装所有依赖

在项目根目录下执行以下命令即可安装全部依赖：

```bash
pip install -r requirements.txt
```

该命令会读取 `requirements.txt` 文件并自动安装列出的所有包。如果遇到安装失败的情况，可以逐个安装出错的包，或者使用国内镜像源加速下载：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装成功

安装完成后，可以通过以下方式验证关键依赖是否正确安装：

```python
# 验证核心依赖
import torch
import transformers
import datasets
import swanlab

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("SwanLab 版本:", swanlab.__version__)
```

如果所有模块都能正常导入且版本信息正确显示，说明环境配置成功。

## 数据集准备

### 预训练数据集下载

预训练数据来自 Seq-Monkey 开源语料库，需要使用 ModelScope 平台下载。框架提供了跨平台的下载脚本：

**Linux/macOS 系统：**

```bash
chmod +x download_dataset.sh
./download_dataset.sh
```

脚本中设置了 HuggingFace 镜像地址以加速下载：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Sources: [download_dataset.sh](download_dataset.sh#L1-L21)

**Windows 系统：**

Windows 用户可以使用 PowerShell 或 CMD 执行 `windows_download_dataset.sh` 中的命令。PowerShell 命令示例：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$dataset_dir = "\path\to\your\dataset"
modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir "$dataset_dir"
tar -xvf "$dataset_dir\mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2" -C "$dataset_dir"
```

Sources: [windows_download_dataset.sh](windows_download_dataset.sh#L1-L36)

### SFT 微调数据集下载

SFT（监督微调）数据集使用 HuggingFace CLI 下载 BelleGroup 的中文对话数据集：

```bash
huggingface-cli download \
  --repo-type dataset \
  --resume-download \
  BelleGroup/train_3.5M_CN \
  --local-dir "./BelleGroup"
```

`--resume-download` 参数支持断点续传，适合网络不稳定的环境。下载完成后，数据集文件会保存在本地指定目录，训练脚本通过 `data_path` 参数指定数据文件路径。

### 数据目录结构

正确下载并解压后，数据目录应包含以下结构：

```
dataset_dir/
├── mobvoi_seq_monkey_general_open_corpus.jsonl    # 预训练语料
└── BelleGroup/
    └── train_3.5M_CN/                               # SFT对话数据
```

在训练脚本中配置数据路径时，需要将 `dataset_dir` 替换为实际的数据存放位置：

```bash
python ddp_pretrain.py --data_path "./seq_monkey_datawhale.jsonl"
```

Sources: [ddp_pretrain.py](ddp_pretrain.py#L240-L250)

## 分词器配置

### 分词器文件结构

框架使用自定义训练的 BPE 分词器，存储在 `tokenizer_k/` 目录下，包含以下三个核心文件：

| 文件名 | 描述 |
|--------|------|
| tokenizer.json | BPE 词汇表和合并规则 |
| tokenizer_config.json | 分词器配置参数 |
| special_tokens_map.json | 特殊标记映射关系 |

### 分词器加载

训练和推理代码使用 `AutoTokenizer` 自动加载分词器：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')
if tokenizer.pad_token_id is not None:
    lm_config.pad_token_id = tokenizer.pad_token_id
```

Sources: [k_model.py](k_model.py#L1-L10)

加载分词器时会自动读取配置文件初始化词汇表和特殊标记。分词器的 `bos_token`、`eos_token` 和 `pad_token` 等特殊标记在预处理数据时发挥重要作用，例如 `PretrainDataset` 中使用 `bos_token` 包裹输入文本：

```python
text = f"{self.tokenizer.bos_token}{sample['text']}"
input_id = self.tokenizer(text).data['input_ids']
```

Sources: [dataset.py](dataset.py#L25-L30)

## 分布式训练配置

### 多 GPU 环境设置

框架通过环境变量 `CUDA_VISIBLE_DEVICES` 控制可用的 GPU 设备。在启动训练脚本时指定 GPU 列表：

```bash
python ddp_pretrain.py --gpus 0,1,2,3 --batch_size 32
```

框架内部会将指定 GPU 暴露给 PyTorch，代码会自动检测可用 GPU 数量并启用 DataParallel 模式：

```python
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    model = torch.nn.DataParallel(model)
```

Sources: [ddp_pretrain.py](ddp_pretrain.py#L220-L230)

### 梯度累积配置

当单个 GPU 显存不足以容纳更大的批次时，可以通过梯度累积模拟大批次训练。配置 `accumulation_steps` 参数将多个小批次的梯度累积后再更新模型参数：

```bash
python ddp_pretrain.py --batch_size 8 --accumulation_steps 8
```

这相当于 effective batch size 为 64。累积步数不宜设置过大，否则可能影响训练稳定性。

## 环境验证流程

### 快速验证脚本

创建以下验证脚本检查环境配置是否正确：

```python
import torch
from transformers import AutoTokenizer
from k_model import ModelConfig, Transformer
from dataset import PretrainDataset

# 检查 PyTorch 和 CUDA
print(f"✓ PyTorch 版本: {torch.__version__}")
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")

# 检查分词器加载
try:
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')
    print(f"✓ 分词器加载成功，词汇表大小: {tokenizer.vocab_size}")
except Exception as e:
    print(f"✗ 分词器加载失败: {e}")

# 检查模型初始化
try:
    config = ModelConfig(dim=256, n_layers=4)
    model = Transformer(config)
    print(f"✓ 模型初始化成功，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")

print("\n环境配置验证完成！")
```

### 常见问题排查

| 问题现象 | 可能原因 | 解决方案 |
|----------|----------|----------|
| CUDA 不可用 | 未安装 GPU 版 PyTorch | 重新安装对应 CUDA 版本的 PyTorch |
| 导入模块失败 | 依赖未安装 | 执行 `pip install -r requirements.txt` |
| 分词器加载超时 | 网络问题 | 检查网络连接或使用离线分词器 |
| GPU 显存不足 | 批次大小过大 | 减小 batch_size 或增加 accumulation_steps |
| 数据加载缓慢 | num_workers 过低 | 调高数据加载器的 num_workers 参数 |

## 下一步

完成环境配置后，建议按照以下顺序继续学习：

- 如果你希望快速体验模型训练流程，可以阅读 [快速启动：一键运行预训练与微调](2-kuai-su-qi-dong-jian-yun-xing-yu-xun-lian-yu-wei-diao)，该文档将带你运行第一个训练任务
- 如果你想深入了解模型架构，可以阅读 [Transformer 架构详解：核心组件与设计原理](4-transformer-jia-gou-xiang-jie-he-xin-zu-jian-yu-she-ji-yuan-li)，了解 Tiny-K 的核心设计
- 如果你已经准备好开始训练，建议阅读 [预训练流程：数据加载与模型训练](8-yu-xun-lian-liu-cheng-shu-ju-jia-zai-yu-mo-xing-xun-lian)，掌握完整的训练流程