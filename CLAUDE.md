# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

一个面向初学者的 **RAG (Retrieval Augmented Generation，检索增强生成)** 演示项目，使用 LangChain 实现。

## 常用命令

```bash
# 安装依赖（需要 uv）
uv sync

# 运行演示
uv run python main.py

# 安装开发依赖（lint 和测试）
uv sync --dev

# 代码检查
uv run ruff check .

# 运行测试
uv run pytest
```

## 环境配置

创建 `.env` 文件，配置 DeepSeek 或 OpenAI API：

### LLM 配置（对话生成）

```bash
# DeepSeek（推荐，价格便宜）
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 或使用 OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
```

### Embedding 配置（文本向量化）

项目默认使用 HuggingFace 本地模型（免费），如需使用 OpenAI Embedding：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

> **注意**：LLM 模型和 Embedding 模型是两种不同的服务，DeepSeek 不提供 Embedding 服务。

## 架构

项目采用**模块化流水线架构**，包含 6 个核心组件：

| 模块 | 职责 |
|------|------|
| [llm.py](src/agent/llm.py) | LLM 模型初始化（DeepSeek/OpenAI） |
| [loader.py](src/agent/loader.py) | 文档加载（TXT、Markdown、PDF） |
| [splitter.py](src/agent/splitter.py) | 文本分块（递归、字符、Markdown） |
| [vectorstore.py](src/agent/vectorstore.py) | 向量存储（Chroma）和嵌入 |
| [retriever.py](src/agent/retriever.py) | 检索策略（similarity、MMR、threshold） |
| [rag_chain.py](src/agent/rag_chain.py) | RAG 链组合检索和生成 |

## RAG 流水线

```
文档 → 加载器 → 分块器 → 向量存储 → 检索器 → LLM → 答案
      (分块)      (嵌入)      (搜索)      (生成)
```

## 入口文件

- [main.py](main.py) - 交互式演示程序：
  - 按需初始化（启动快，首次使用时才加载模型）
  - 分级查看文档（先列表再详情）
  - 交互式问答 + 推荐问题 + 引用来源显示

## 检索策略

| 策略 | 适用场景 |
|------|----------|
| `similarity` | 默认，相似度优先 |
| `mmr` | 多样性优先，避免重复结果 |
| `similarity_score_threshold` | 相似度阈值过滤 |

## 学习资源

- [docs/rag-deep-dive.md](docs/rag-deep-dive.md) - RAG 深度解析与架构说明
- [docs/rag-practice.md](docs/rag-practice.md) - 实践指南

## 示例数据

项目内置**家常菜谱**知识库（红烧肉、番茄炒蛋、早餐推荐、新手菜谱），通过 [loader.py](src/agent/loader.py) 中的 `load_sample_data()` 加载。

默认使用中文 embedding 模型 `BAAI/bge-small-zh-v1.5`，对中文检索效果更好。

## 关键知识点

### LLM 模型 ≠ Embedding 模型

**重要**：对话模型（如 DeepSeek-chat、GPT-4）和 Embedding 模型是**两种不同的服务**：

| 类型 | 用途 | 示例模型 |
|------|------|----------|
| LLM (对话模型) | 生成回答 | deepseek-chat, gpt-4o-mini |
| Embedding 模型 | 文本向量化 | text-embedding-3-small, sentence-transformers/all-MiniLM-L6-v2 |

**常见错误**：不要用对话模型的 API 去调用 embedding，会报 404 错误。

**解决方案**：
- 使用 OpenAI embedding（需要 OPENAI_API_KEY）
- 使用 HuggingFace 本地模型（免费，无需 API Key）

### LangChain 依赖包选择

| 场景 | 推荐包 |
|------|--------|
| Chroma 向量存储 | `langchain-chroma`（官方）而非 `langchain-community` |
| HuggingFace 嵌入 | `langchain-huggingface` |

## 依赖

核心依赖：langchain, langchain-core, langchain-openai, langchain-text-splitters, langchain-chroma, langchain-huggingface, langgraph, python-dotenv, chromadb, sentence-transformers

开发依赖：ruff（代码检查）、pytest（测试）
