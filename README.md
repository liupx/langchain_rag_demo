# LangChain RAG Demo

一个面向初学者的 **RAG (Retrieval Augmented Generation，检索增强生成)** 演示项目，通过 LangChain 学会什么是 RAG，以及如何在实际应用中使用 RAG 来提升 Agent 的能力。

## 特性

- 📚 **完整 RAG 流程**：文档加载 → 文本分块 → 向量化 → 检索 → 生成
- 🎯 **交互式体验**：菜单式操作，按需初始化，启动快速
- 📖 **引用来源**：回答时显示引用自哪些文档，清晰展示 RAG 效果
- 🔤 **中文优化**：使用中文 Embedding 模型，检索效果更好

## 什么是 RAG？

RAG (Retrieval Augmented Generation) 是一种将**外部知识检索**与 **LLM 生成能力**结合的技术。

### 为什么需要 RAG？

| 问题 | RAG 解决方案 |
|------|--------------|
| LLM 知识过时 | 检索最新文档获取最新信息 |
| 缺乏领域知识 | 接入专业文档库 |
| 幻觉问题 | 基于真实文档生成答案 |
| 无法追溯来源 | 引用源文档，可验证答案 |

### RAG 工作流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  文档加载     │───▶│  文本分块    │───▶│  向量嵌入    │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                                ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   LLM 生成    │◀───│  提示词组装   │◀───│   相似度检索  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 快速开始

### 1. 克隆项目

```bash
cd /Users/pengxu.liu/liupx/github
git clone <your-repo-url>  # 如果需要
cd langchain_rag_demo
```

### 2. 安装依赖

需要先安装 [uv](https://github.com/astral-sh/uv)（推荐）：

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 4. 运行演示

```bash
uv run python main.py
```

## 环境变量

### LLM 配置（用于对话生成）

| 变量名              | 说明             | 默认值                        |
| ------------------- | ---------------- | ----------------------------- |
| `DEEPSEEK_API_KEY`  | DeepSeek API Key | -                             |
| `DEEPSEEK_BASE_URL` | API 基础 URL     | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL`    | 模型名称         | `deepseek-chat`               |

或使用 OpenAI：

| 变量名            | 说明           | 默认值                      |
| ----------------- | -------------- | --------------------------- |
| `OPENAI_API_KEY`  | OpenAI API Key | -                           |
| `OPENAI_BASE_URL` | API 基础 URL   | `https://api.openai.com/v1` |
| `OPENAI_MODEL`    | 模型名称       | `gpt-4o-mini`               |

### Embedding 配置（用于文本向量化）

**重要**：LLM 模型和 Embedding 模型是两种不同的服务！

| 变量名                  | 说明                | 默认值                    |
| ----------------------- | ------------------- | ------------------------- |
| `OPENAI_API_KEY`        | OpenAI Embedding   | 如已配置 LLM 则无需额外设置 |
| `OPENAI_EMBEDDING_MODEL`| Embedding 模型名称 | `text-embedding-3-small`  |

> 项目默认使用 HuggingFace 本地 Embedding 模型（免费，无需配置）

## 代码结构

```
langchain_rag_demo/
├── main.py                 # 主入口，包含 RAG 演示
├── pyproject.toml          # 项目配置和依赖
├── .env.example            # 环境变量示例
├── .gitignore              # Git 忽略文件
├── README.md               # 本文件
└── src/
    └── agent/
        ├── __init__.py     # 包初始化
        ├── llm.py          # LLM 模型初始化
        ├── loader.py       # 文档加载器
        ├── splitter.py     # 文本分块器
        ├── vectorstore.py  # 向量存储
        ├── retriever.py    # 检索器
        └── rag_chain.py    # RAG 链
```

## 核心概念

### 1. 文档加载

```python
from src.agent.loader import load_sample_data

# 加载示例数据（内置知识库）
documents = load_sample_data()

# 或加载本地文件
from src.agent.loader import get_document_loader
documents = get_document_loader("your_file.txt")
```

### 2. 文本分块

```python
from src.agent.splitter import split_by_recursive

# 递归分割，保持语义完整
chunks = split_by_recursive(
    documents,
    chunk_size=300,    # 每个块的大小
    chunk_overlap=50,  # 块之间的重叠
)
```

### 3. 向量存储

```python
from src.agent.vectorstore import create_vectorstore

# 创建 Chroma 向量数据库
vectorstore = create_vectorstore(
    chunks,
    collection_name="my_rag",
    persist_directory="./chroma_db",  # 持久化存储
)
```

### 4. 检索

```python
from src.agent.retriever import create_retriever

# 创建检索器
retriever = create_retriever(
    vectorstore,
    search_type="similarity",  # 检索策略
    k=3,                        # 返回数量
)

# 检索相关文档
docs = retriever.invoke("你的问题")
```

### 5. 构建 RAG 链

```python
from src.agent.llm import get_model
from src.agent.rag_chain import create_rag_chain, invoke_rag

model = get_model()

# 创建 RAG 链
rag_chain = create_rag_chain(retriever, model)

# 问答
answer = invoke_rag(rag_chain, "什么是 RAG？")
```

### 检索策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `similarity` | 基于向量相似度，返回最相似的 k 个 | 大多数场景 |
| `mmr` | 最大边际相关性，在相关基础上增加多样性 | 需要避免重复结果 |
| `similarity_score_threshold` | 设置相似度阈值，过滤低相关度结果 | 精确筛选 |

## 运行演示

运行 `main.py` 启动交互式演示：

**首次运行**会自动提示创建数据目录：
```
请选择：
  1. 使用示例文档（3 个菜谱）
  2. 退出后自己创建文档
```

选择示例文档后，进入主菜单：
```
1. 📚 查看原始文档 - 分级浏览，先列表后详情
2. 📦 查看分块数据 - 查看 RAG 处理后的文本片段
3. 💬 开始问答 - 交互式问答，会显示引用来源
4. 🚪 退出
```

内置知识库为**家常菜谱**（红烧肉、番茄炒蛋、早餐推荐等），可以提问：
- "红烧肉怎么做？"
- "早餐吃什么好？"
- "新手学什么菜简单？"

## 依赖

核心依赖（通过 `uv sync` 自动安装）：

- `langchain >= 0.3.0` - LangChain 核心
- `langchain-core >= 0.3.0` - 核心组件
- `langchain-openai >= 0.2.0` - OpenAI/DeepSeek 对话模型
- `langchain-text-splitters >= 0.3.0` - 文本分块
- `langchain-chroma >= 0.1.0` - Chroma 向量存储（官方包）
- `langchain-huggingface >= 0.1.0` - HuggingFace Embedding
- `chromadb >= 0.4.0` - 向量数据库
- `sentence-transformers >= 2.2.0` - 本地 Embedding 模型
- `langgraph >= 0.2.0` - LangGraph 工作流
- `python-dotenv >= 1.0.0` - 环境变量管理

## 学习路径

1. **入门**：运行 `main.py`，观察输出，理解 RAG 流程
2. **阅读代码**：查看 `src/agent/` 下的模块，理解每个组件的作用
3. **深入理解**：阅读 [代码详解文档](docs/code-analysis.md)，了解每个模块的原理
4. **动手实践**：
   - 替换为自己的文档
   - 调整分块大小
   - 尝试不同检索策略
5. **进阶**：将 RAG 集成到 Agent 中，实现知识问答 Agent

## License

MIT
