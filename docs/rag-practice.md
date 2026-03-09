# RAG 项目实践指南

## 1. 快速启动项目

```bash
# 1. 进入项目目录
cd langchain_rag_demo

# 2. 复制环境变量文件
cp .env.example .env

# 3. 编辑 .env，填入 API Key
vim .env

# 4. 安装依赖
uv sync

# 5. 运行演示
uv run python main.py
```

## 2. 使用自己的文档

### 步骤 1: 准备文档

将你的文档放入项目目录，例如：

```
langchain_rag_demo/
├── data/
│   ├── company_policy.md
│   ├── product_docs.pdf
│   └── faq.txt
```

### 步骤 2: 加载文档

修改 `main.py` 或创建新脚本：

```python
from src.agent.loader import get_document_loader
from src.agent.splitter import split_by_recursive
from src.agent.vectorstore import create_vectorstore
from src.agent.llm import get_model
from src.agent.retriever import create_retriever
from src.agent.rag_chain import create_rag_chain, invoke_rag

# 加载多个文档
docs = []
for file in ["data/company_policy.md", "data/product_docs.pdf"]:
    docs.extend(get_document_loader(file))

# 文本分块
chunks = split_by_recursive(docs, chunk_size=500, chunk_overlap=50)

# 创建向量存储
vectorstore = create_vectorstore(chunks, persist_directory="./my_vectorstore")

# 创建 RAG 链
model = get_model()
retriever = create_retriever(vectorstore, k=3)
rag_chain = create_rag_chain(retriever, model)

# 问答
answer = invoke_rag(rag_chain, "公司的年假政策是什么？")
print(answer)
```

## 3. 调优技巧

### 3.1 文本分块参数

```python
# 块大小参考
chunk_size = 500   # 短文档 / 精确检索
chunk_size = 1000  # 中等文档 / 平衡
chunk_size = 2000  # 长文档 / 上下文完整

# 重叠大小
chunk_overlap = chunk_size * 0.1  # 通常 10-20%
```

### 3.2 检索参数

```python
# k 值选择
k = 1  # 简单问题
k = 3  # 一般问题（推荐）
k = 5  # 复杂问题

# 检索策略
search_type = "similarity"           # 默认，相似度优先
search_type = "mmr"                  # 多样性优先
search_type = "similarity_threshold"  # 阈值过滤
```

### 3.3 向量数据库持久化

```python
# 首次创建
vectorstore = create_vectorstore(
    chunks,
    persist_directory="./chroma_db"
)

# 后续加载
vectorstore = load_vectorstore(
    persist_directory="./chroma_db"
)
```

## 4. 常见问题

### Q1: 检索不到相关内容

**可能原因**：
1. 文档没有被正确加载
2. 分块太小，丢失上下文
3. 嵌入模型不适合你的文档类型

**解决方案**：
- 检查文档加载是否成功
- 增大 chunk_size
- 尝试不同的嵌入模型

### Q2: 回答质量差

**可能原因**：
1. 检索到的内容不相关
2. k 值太小，遗漏重要信息
3. Prompt 不够清晰

**解决方案**：
- 调整检索参数
- 增大 k 值
- 优化 RAG prompt

### Q3: 响应速度慢

**可能原因**：
1. 网络延迟（API 调用）
2. 向量数据库太大
3. 文档太多，检索慢

**解决方案**：
- 使用本地模型
- 添加缓存
- 考虑向量数据库优化

## 5. 生产环境建议

### 5.1 监控和日志

```python
# 添加日志记录
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 记录检索结果
docs = retriever.invoke(query)
logger.info(f"检索到 {len(docs)} 个相关文档")
for doc in docs:
    logger.debug(f"来源: {doc.metadata}")
```

### 5.2 错误处理

```python
try:
    answer = invoke_rag(rag_chain, question)
except Exception as e:
    logger.error(f"RAG 调用失败: {e}")
    # 回退到普通 LLM 调用
    answer = model.invoke(question)
```

### 5.3 缓存

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(query: str):
    # 缓存检索结果
    return retriever.invoke(query)
```

## 6. 进阶：构建知识问答 Agent

结合其他 LangChain 组件构建完整的问答系统：

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str):
    """搜索知识库"""
    return invoke_rag(rag_chain, query)

# 创建 Agent
tools = [search_knowledge_base]
agent = create_tool_calling_agent(model, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 对话
result = agent_executor.invoke({
    "input": "公司的病假政策是什么？"
})
```
