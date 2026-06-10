# 代码架构说明

## 总体设计

系统采用分层结构，数据从 docx 周报流向向量索引，再经混合检索送入 LLM 生成带引用的回答。

```
docx 周报 (REPORT_DATA_PATH)
        │
        ▼
   parsing/          解析、分块、元数据提取
        │
        ▼
   indexing/         Embedding + 持久化 → data/
        │
        ▼
   retrieval/        向量 + BM25 混合检索
        │
        ▼
   generation/       LLM 生成 + 引用格式化
        │
        ▼
   app/chat.py       ask / interactive_chat
```

入口 `main.py` 负责 CLI 参数解析、环境检查、日志初始化，不包含业务逻辑。`--web` 模式启动 `web/server.py` 中的 FastAPI 服务。

## 模块职责

### 根目录

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口，参数分发 |
| `config.py` | 环境变量、常量、日志配置、`check_env()` |
| `models.py` | `ChunkMetadata`、`DocumentChunk`、`SearchResult` |
| `utils.py` | 日期解析、路径工具、查询分词 |
| `exceptions.py` | 统一异常层次 |

### parsing/ — 文档解析

| 文件 | 职责 |
|------|------|
| `parser_rules.json` | 标题识别、项目关键词等规则（单一数据源） |
| `rules.py` | 加载规则，提供 `heading_max_len()` 等 |
| `docx.py` | docx 分段、项目识别、chunk 文本构建 |
| `chunker.py` | 按 token 长度切分 |
| `reader.py` | `DocxReportReader`：扫描目录、批量产出 `DocumentChunk` |

每个 chunk 携带元数据：`source`、`report_date`、`project`、`author`、`quarter`、`section_type`、`chunk_index`。

### indexing/ — 索引构建与存储

| 文件 | 职责 |
|------|------|
| `record.py` | `IndexRecord`：文本 + 元数据 + 向量 |
| `store.py` | `IndexStore`：读写 `data/records.json` + `vectors.npy` |
| `pipeline.py` | `build_index()`：全量/增量构建 |
| `manifest.py` | 文件哈希、索引版本号，驱动增量更新 |
| `legacy_io.py` | 兼容旧版 `document.json` 格式 |

增量逻辑：对比 manifest 中各 docx 的 MD5，仅对变更文件重新解析和 embedding，删除已移除文件对应的 chunk。

### retrieval/ — 检索

| 文件 | 职责 |
|------|------|
| `engine.py` | `SearchEngine`：混合打分主流程 |
| `scoring.py` | 余弦相似度、BM25、候选池合并 |
| `search_modes.py` | `latest` / `timeline` / `compare` 去重策略 |
| `session.py` | `RAGSession`：复用索引、Embedding、LLM 客户端 |

检索流程：

1. 按 `--year` / `--month` / `--auto-date` 过滤候选
2. 向量相似度 + BM25 加权融合（默认 0.7 / 0.3）
3. 取向量 Top-K 与关键词 Top-K 的并集作为精排池
4. 按模式阈值过滤，再按模式去重

### providers/ — 外部服务

| 文件 | 职责 |
|------|------|
| `openai_client.py` | OpenAI 兼容客户端单例 |
| `embeddings.py` | `OpenAIEmbedding`，支持 BGE query/passage 前缀 |
| `embedding_cache.py` | SQLite 缓存，避免重复 embedding 请求 |

### generation/ — 生成与输出

| 文件 | 职责 |
|------|------|
| `llm.py` | `OpenAIChat`，周报专用 prompt |
| `output.py` | 检索结果展示、引用格式化 |

### app/ — 业务编排

| 文件 | 职责 |
|------|------|
| `chat.py` | `ask()` / `ask_detail()` / `interactive_chat()`：检索 → 生成 → 格式化 |

### web/ — Web 服务

| 文件 | 职责 |
|------|------|
| `server.py` | FastAPI 应用，`/api/chat` 与 `/api/health` |
| `schemas.py` | 请求/响应 Pydantic 模型 |
| `static/` | 聊天页面（HTML/CSS/JS） |

Web 服务启动时加载一次 `RAGSession`，多用户请求通过线程锁串行调用 LLM/检索，避免并发冲突。

## 数据流示例

**构建索引** (`--build`)：

```
DocxReportReader.get_chunks()
  → IndexStore.from_chunks()
  → OpenAIEmbedding.get_embeddings()
  → store.persist(./data)
  → save_manifest()
```

**问答** (`--query`)：

```
RAGSession
  → SearchEngine.query()
  → OpenAIChat.chat(context)
  → format_answer_with_citations()
```

## 关键设计决策

1. **混合检索**：纯向量对专有名词（项目名、日期）召回不足，BM25 补充关键词匹配。
2. **按项目分 chunk**：周报以项目为段落单位切分，而非固定字数，提高检索语义完整性。
3. **增量索引**：manifest 记录文件哈希与解析规则版本，避免每次全量 embedding。
4. **会话复用**：`RAGSession` 在 `--chat` 模式下复用已加载索引和客户端，减少 I/O。
5. **索引与源码分离**：构建产物放在 `data/`，日志放在 `log/`，与代码目录解耦。

## 扩展点

- 替换 Embedding：实现 `EmbeddingProvider` 协议（见 `providers/embeddings.py`）
- 调整解析规则：编辑 `parsing/parser_rules.json`，然后 `--build --force` 重建
- 新增检索模式：在 `search_modes.py` 添加去重函数，并在 `models.SEARCH_MODES` 注册
