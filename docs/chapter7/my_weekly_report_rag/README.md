# 周报 RAG 系统

基于个人工作周报（docx）的检索增强问答工具。支持向量 + BM25 混合检索、按项目/时间过滤、多种检索模式，以及带引用来源的中文回答。

## 快速开始

### 1. 安装依赖

```bash
cd docs/chapter7/my_weekly_report_rag
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env_example .env
# 编辑 .env，填入 OPENAI_API_KEY 和 OPENAI_BASE_URL
```

主要环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | API 密钥 | 必填 |
| `OPENAI_BASE_URL` | API 地址 | 必填 |
| `REPORT_DATA_PATH` | 周报 docx 原始目录 | 必填，见 `.env` |
| `DEFAULT_AUTO_DATE` | 自动从问题解析年月 | `true` |
| `INDEX_PATH` | 向量索引输出目录 | `./data` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-m3` |
| `CHAT_MODEL` | 对话模型 | `Qwen/Qwen2.5-32B-Instruct` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `WEB_HOST` | Web 监听地址 | `0.0.0.0` |
| `WEB_PORT` | Web 端口 | `1203` |
| `WEB_ACCESS_TOKEN` | 访问令牌（可选） | 空 |

### 3. 构建索引

```bash
./scripts/build.sh
# 或
python main.py --build
```

首次全量构建；之后默认增量更新（仅处理新增/变更的 docx）。

### 4. 使用

**Web 模式（推荐给其他用户）**

```bash
./scripts/web.sh
# 浏览器打开 http://127.0.0.1:1203
# 局域网用户访问 http://<你的电脑IP>:1203
```

Web 页面默认使用 `latest` 检索模式；高级模式（`timeline` / `compare`、年月过滤）请使用 CLI。

**安全提示**：默认监听 `0.0.0.0` 时，局域网用户可直接访问周报内容。对外暴露前请在 `.env` 中设置 `WEB_ACCESS_TOKEN`，并在页面侧边栏填写令牌。

**命令行模式**

```bash
# 仅检索（调试）
./scripts/search.sh "catchii家族房" --debug

# 单次问答
./scripts/query.sh "2025年12月catchii进展" --auto-date

# 交互模式
./scripts/chat.sh
```

也可使用根目录 `run.sh` 转发任意参数：

```bash
chmod +x run.sh scripts/*.sh
./run.sh --search "catchii" --k 3 --debug
```

## 常用命令

```bash
# 强制全量重建索引
python main.py --build --force

# 指定年月过滤
python main.py --search "rank模型" --year 2025 --month 12

# 检索模式（默认开启 auto-date，可用 --no-auto-date 关闭）
python main.py --query "catchii进展" --mode timeline
# latest   - 同项目取最新一条
# timeline - 按时间线返回多条
# compare  - 按月对比
```

## 目录说明

```
my_weekly_report_rag/
├── main.py              # CLI 入口
├── config.py            # 配置、日志、环境检查
├── models.py            # 数据模型
├── parsing/             # docx 解析与分块
├── indexing/            # 索引构建与持久化
├── retrieval/           # 混合检索
├── providers/           # OpenAI / Embedding
├── generation/          # LLM 与输出格式化
├── app/                 # 问答业务流程
├── web/                 # Web 服务与前端页面
├── data/                # 向量索引（构建产物，git 忽略）
├── log/                 # 运行日志（按日落盘）
├── scripts/             # 便捷运行脚本
├── docs/                # 项目文档
└── tests/               # 单元测试
```

`data/` 目录存放构建后的索引文件：

- `records.json` — chunk 文本与元数据
- `vectors.npy` — 向量矩阵
- `manifest.json` — 文件哈希与版本，用于增量构建

日志写入 `log/app_YYYYMMDD.log`，同时输出到终端。

## 测试

需要 Python 3.10+。

```bash
python -m unittest discover -s tests -v
```

## 注意事项

- 索引构建（`--build`）与 Web 服务不要同时运行，避免读写冲突。
- `data/` 为构建产物，已在 `.gitignore` 中忽略；克隆后需先执行 `--build`。

## 更多文档

- [代码架构说明](docs/architecture.md)
