# LangChain RAG Demo 代码详解

本文档对 langchain_rag_demo 项目进行深入代码解析，帮助理解 RAG 的完整实现。

## 1. 项目概述

这是一个完整的 RAG（检索增强生成）演示项目，展示了如何：
1. 加载文档（支持 TXT、Markdown、PDF）
2. 将文档分割成小块
3. 将文本转换为向量并存储
4. 根据用户问题检索相关文档
5. 让 LLM 基于检索内容生成答案

## 2. 模块架构

```
langchain_rag_demo/
├── main.py                 # 主入口，交互式演示
├── src/agent/
│   ├── loader.py           # 文档加载器
│   ├── splitter.py         # 文本分块器
│   ├── vectorstore.py     # 向量存储
│   ├── retriever.py       # 检索器
│   ├── rag_chain.py       # RAG 链
│   └── llm.py             # LLM 模型
└── data/                  # 文档目录（首次运行自动创建）
```

## 3. 核心模块详解

### 3.1 loader.py - 文档加载

**职责**：从文件系统加载文档，转换为 LangChain 的 Document 对象。

```python
from langchain_core.documents import Document

# Document 结构
doc = Document(
    page_content="文档文本内容",      # 文本内容
    metadata={"source": "file.txt"}   # 元信息
)
```

**关键函数**：

| 函数 | 作用 |
|------|------|
| `load_text_file()` | 加载 TXT 文件 |
| `load_markdown()` | 加载 Markdown 文件 |
| `load_pdf()` | 加载 PDF 文件 |
| `get_document_loader()` | 根据扩展名自动选择加载器 |
| `load_sample_data()` | 加载 data 目录下所有文档 |
| `init_sample_data()` | 首次运行时初始化示例文档 |

**示例用法**：

```python
# 自动根据文件类型选择加载器
docs = get_document_loader("recipe.txt")

# 加载目录下所有文档
all_docs = load_sample_data()
```

### 3.2 splitter.py - 文本分块

**职责**：将长文档分割成较小的片段（Chunk）。

**为什么需要分块？**
- 向量模型有输入长度限制（通常 512~4096 tokens）
- 小块便于精确检索，避免返回无关内容
- 块之间重叠（overlap）保持上下文连贯

**关键函数**：

```python
from src.agent.splitter import split_by_recursive

# 递归分割（推荐）
chunks = split_by_recursive(
    documents,
    chunk_size=500,      # 每个块的最大字符数
    chunk_overlap=50,    # 块之间的重叠字符数
)
```

**分割策略**：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `split_by_character` | 按固定字符数分割 | 简单场景 |
| `split_by_recursive` | 智能分割，优先在语义边界分割 | **推荐**，中文效果好 |
| `split_by_markdown` | 按 Markdown 标题分割 | 结构化文档 |

**RecursiveCharacterTextSplitter 原理**：

```python
# 按优先级依次尝试不同分隔符
separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
```

1. 先按段落分割（`\n\n`）
2. 如果块太大，按换行分割（`\n`）
3. 按中文标点分割（`。` `！` `？`）
4. 最后按空格/单词分割

这样可以尽量在语义完整的地方分割，而不是在句子中间切断。

### 3.3 vectorstore.py - 向量存储

**职责**：将文本转换为向量（Embedding），存储到向量数据库。

**核心概念**：

1. **Embedding（嵌入）**：将文本转为数值向量
   - 语义相似的文本，向量也相似
   - 通过计算向量距离找到相关内容

2. **向量数据库**：存储向量的地方
   - 项目使用 Chroma（轻量级、支持内存/持久化）
   - 类似 Redis 但专门存向量

**关键函数**：

```python
from src.agent.vectorstore import create_vectorstore

# 创建向量存储
vectorstore = create_vectorstore(
    chunks,
    collection_name="recipe_rag",  # 集合名称
    persist_directory=None,          # None=内存存储，指定路径=持久化
)
```

**Embedding 模型选择**：

```python
def get_embeddings():
    # 1. OpenAI（收费，效果好）
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. HuggingFace（免费，本地运行）
    # 使用中文优化模型
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

**WrappedHuggingFaceEmbeddings 包装类**：

```python
class WrappedHuggingFaceEmbeddings:
    """处理输入可能是 dict 的情况"""
    def embed_query(self, text):
        if isinstance(text, dict):
            # 从 dict 中提取查询文本
            text = text.get("query") or text.get("question")
        return self._embeddings.embed_query(text)
```

LangChain 新版本中，retriever 可能传入 dict 而不是字符串，需要包装处理。

### 3.4 retriever.py - 检索器

**职责**：从向量数据库中检索与查询最相关的文档。

**关键函数**：

```python
from src.agent.retriever import create_retriever

retriever = create_retriever(
    vectorstore,
    search_type="similarity",  # 检索策略
    k=3,                        # 返回数量
)
```

**检索策略对比**：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `similarity` | 基于向量余弦相似度，返回最相似的 k 个 | 大多数场景 |
| `mmr` | 最大边际相关性，在相关基础上增加多样性 | 需要避免重复结果 |
| `similarity_score_threshold` | 设置相似度阈值，过滤低相关度结果 | 精确筛选 |

**MMR（Maximum Marginal Relevance）原理**：

```
1. 先检索 fetch_k 个相关结果（假设 20 个）
2. 然后从中选择 k 个（3 个）多样化的结果
3. 避免返回内容重复的文档
```

```python
retriever = create_retriever(
    vectorstore,
    search_type="mmr",
    k=3,
    fetch_k=20,  # 从 20 个中选 3 个
)
```

### 3.5 rag_chain.py - RAG 链

**职责**：将检索（Retriever）和生成（LLM）结合成完整流程。

**核心思想**：

```
用户问题 → 检索相关文档 → 组装提示词 → LLM 生成答案
```

**LangChain 管道语法**：

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# | 管道符：将前一个输出作为后一个输入
# RunnableParallel：并行执行多个任务

chain = (
    RunnableParallel(
        context=retriever | format_docs,      # 检索 + 格式化
        question=RunnableLambda(lambda x: x["question"]),
    )
    | prompt_template                        # 组装提示词
    | model                                   # LLM 生成
    | StrOutputParser()                       # 解析输出
)
```

**完整 RAG 链流程**：

```
输入: {"question": "红烧肉怎么做？"}
         │
         ▼
┌──────────────────────────────────────────┐
│  RunnableParallel                        │
│  ├─ context: retriever → format_docs    │
│  │     检索文档并格式化为字符串           │
│  └─ question: x["question"]              │
│        提取原始问题                       │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  prompt_template                         │
│  将 context 和 question 组合成完整提示词  │
│                                          │
│  上下文：红烧肉做法...                    │
│  问题：红烧肉怎么做？                      │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  model (DeepSeek/OpenAI)                │
│  基于上下文生成答案                       │
└──────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│  StrOutputParser()                       │
│  解析为字符串输出                         │
└──────────────────────────────────────────┘
         │
         ▼
输出: "红烧肉的做法如下：..."
```

**提示词模板**：

```python
RAG_SYSTEM_PROMPT = """你是一个基于检索内容回答问题的助手。

请遵循以下规则：
1. 只根据提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有相关信息，请明确说明
3. 回答要简洁明了

上下文信息：
{context}

问题：{question}

回答："""
```

### 3.6 llm.py - LLM 模型

**职责**：初始化 LLM 模型（DeepSeek 或 OpenAI）。

```python
from src.agent.llm import get_model

model = get_model(
    temperature=0.7,  # 控制输出随机性
)
```

**配置优先级**：
1. DeepSeek（价格便宜，默认使用）
2. OpenAI（回退选项）

```python
def get_model():
    if os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    # 回退到 OpenAI
    return ChatOpenAI(model="gpt-4o-mini")
```

## 4. main.py - 交互式演示

### 4.1 懒加载设计

为了加快启动速度，使用懒加载（Lazy Initialization）：

```python
class RAGSystem:
    def __init__(self):
        self._initialized = False

    def ensure_init(self):
        """首次使用时才初始化"""
        if not self._initialized:
            init_rag_system()
```

只有用户选择"开始问答"时才真正加载模型。

### 4.2 首次运行初始化

首次运行时会提示用户：

```
📂 首次运行 - 数据初始化

请选择：
1. 使用示例文档（3 个菜谱）
2. 退出后自己创建文档
```

选择示例文档后会自动创建 `data/` 目录和示例文件。

### 4.3 交互菜单

```
🌟 LangChain RAG 演示

请选择操作：
1. 📚 查看原始文档
2. 📦 查看分块数据
3. 💬 开始问答
4. 🚪 退出
```

### 4.4 推荐问题

根据加载的文档自动生成推荐问题：

```python
def get_recommended_questions():
    if 'home_cooking.txt' in sources:
        recommendations.extend([
            "红烧肉怎么做？",
            "番茄炒蛋有什么技巧？",
        ])
```

### 4.5 引用来源

回答时会显示引用来源：

```
🤖 AI: 红烧肉的做法如下...

📖 引用来源:
1. [home_cooking.txt] 五花肉切块焯水...
2. [beginner_cooking.txt] 可乐鸡翅...
```

## 5. 完整使用流程

```python
# 1. 加载文档
documents = load_sample_data()

# 2. 文本分块
chunks = split_by_recursive(documents, chunk_size=500, chunk_overlap=50)

# 3. 创建向量存储
vectorstore = create_vectorstore(chunks, collection_name="demo")

# 4. 创建检索器
retriever = create_retriever(vectorstore, search_type="similarity", k=3)

# 5. 初始化 LLM
model = get_model(temperature=0.7)

# 6. 创建 RAG 链
rag_chain = create_rag_chain(retriever, model)

# 7. 问答
answer = rag_chain.invoke({"question": "红烧肉怎么做？"})
print(answer)
```

## 6. 关键知识点

### 6.1 LLM vs Embedding

这是两个不同的服务：

| 类型 | 作用 | 示例模型 |
|------|------|----------|
| LLM | 对话/生成 | deepseek-chat, gpt-4o |
| Embedding | 向量化 | text-embedding-3-small, bge-small-zh |

DeepSeek 不提供 Embedding 服务，需要使用其他模型。

### 6.2 chunk_size 选择

| 值 | 优点 | 缺点 |
|----|------|------|
| 200-300 | 精确检索 | 可能丢失上下文 |
| 500 | 平衡（推荐） | - |
| 1000+ | 保留完整上下文 | 检索可能不精确 |

### 6.3 相似度阈值

`similarity_score_threshold` 参数：
- 0.0~0.3：严格，只返回高度相关内容
- 0.5：中等（推荐）
- 0.7~1.0：宽松，可能返回不相关内容

## 7. 进阶扩展

### 7.1 持久化向量存储

```python
# 首次创建
vectorstore = create_vectorstore(
    chunks,
    persist_directory="./chroma_db",
)

# 后续加载
vectorstore = load_vectorstore(
    persist_directory="./chroma_db",
)
```

### 7.2 自定义提示词

```python
custom_prompt = """你是一个专业的厨师。
根据以下菜谱回答用户问题。

菜谱内容：
{context}

用户问题：{question}

请给出详细的步骤说明："""

rag_chain = create_rag_chain(retriever, model, prompt=custom_prompt)
```

### 7.3 多模态文档

```python
# PDF 文档
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("menu.pdf")
docs = loader.load()

# 支持更多格式
# - CSVLoader
# - DocxLoader
# - WebBaseLoader
```

## 8. 常见问题

### 8.1 检索不到相关内容

可能原因：
- chunk_size 太大或太小
- Embedding 模型不适合中文
- 文档内容不足

解决方案：
- 调整 chunk_size 和 overlap
- 使用中文 Embedding 模型（如 bge-small-zh）
- 增加文档内容

### 8.2 LLM 不按文档内容回答

检查提示词是否强调"只根据上下文回答"：

```python
RAG_SYSTEM_PROMPT = """请只根据提供的上下文信息回答，不要编造信息。
如果上下文中没有相关信息，请明确说明。
..."""
```

### 8.3 首次加载慢

首次运行需要：
- 下载 Embedding 模型（~200MB）
- 首次 LLM 调用较慢

解决方案：
- 持久化向量存储，第二次运行直接加载
- 使用更小的 Embedding 模型
