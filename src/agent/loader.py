"""
Document Loaders

演示如何加载不同类型的文档（ TXT, Markdown, PDF 等）

Document 对象结构：
- page_content: str - 文档的文本内容
- metadata: dict - 元信息（如来源文件、页码等）
"""
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

# 尝试导入 LangChain 文档加载器
try:
    from langchain_community.document_loaders import (
        TextLoader,
        MarkdownLoader,
        PyPDFLoader,
        DirectoryLoader,
    )
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False


def load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    加载文本文件

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        Document 列表
    """
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")

    loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    """
    加载 Markdown 文件

    Args:
        file_path: 文件路径

    Returns:
        Document 列表
    """
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")

    loader = MarkdownLoader(file_path)
    return loader.load()


def load_pdf(file_path: str) -> List[Document]:
    """
    加载 PDF 文件

    Args:
        file_path: 文件路径

    Returns:
        Document 列表
    """
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")

    loader = PyPDFLoader(file_path)
    return loader.load()


def load_directory(
    directory_path: str,
    glob_pattern: str = "**/*",
    loader_class: Optional[type] = None,
) -> List[Document]:
    """
    加载目录下的所有文件

    Args:
        directory_path: 目录路径
        glob_pattern: 文件匹配模式，如 "**/*.txt"
        loader_class: 指定加载器类，默认根据文件类型自动选择

    Returns:
        Document 列表
    """
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")

    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        loader_cls=loader_class,
    )
    return loader.load()


def get_document_loader(file_path: str) -> List[Document]:
    """
    根据文件扩展名自动选择合适的加载器

    Args:
        file_path: 文件路径

    Returns:
        Document 列表
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text_file(file_path)
    elif suffix == ".md":
        return load_markdown(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    else:
        # 默认尝试作为文本文件加载
        return load_text_file(file_path)


def load_sample_data() -> List[Document]:
    """
    加载示例数据 - 用于演示 RAG 的基础知识文档

    Returns:
        Document 列表
    """
    sample_docs = [
        Document(
            page_content="""
什么是 RAG？

RAG (Retrieval Augmented Generation，检索增强生成) 是一种将外部知识检索
与 LLM 生成能力结合的技术。

为什么需要 RAG？
1. 知识时效性：LLM 的训练数据有截止日期，无法回答最新问题
2. 领域专业知识：通用模型可能不了解特定领域的专业知识
3. 减少幻觉：让模型基于真实文档回答，减少虚构内容
4. 可解释性：可以追溯答案的来源

RAG 的基本流程：
1. 文档加载：从各种来源加载文档
2. 文本分块：将长文档分割成较小的片段
3. 向量化：将文本片段转换为向量（嵌入）
4. 存储向量：将向量存储到向量数据库
5. 检索：当用户提问时，找到最相关的文档片段
6. 生成：将检索到的内容作为上下文，让 LLM 生成答案
            """,
            metadata={"source": "rag_intro.txt", "page": 1},
        ),
        Document(
            page_content="""
LangChain RAG 组件

LangChain 提供了完整的 RAG 组件：

1. Document Loaders（文档加载器）
   - TextLoader: 加载文本文件
   - MarkdownLoader: 加载 Markdown
   - PyPDFLoader: 加载 PDF
   - WebBaseLoader: 从网页加载
   - DirectoryLoader: 加载整个目录

2. Text Splitters（文本分块器）
   - CharacterTextSplitter: 按字符数分割
   - RecursiveCharacterTextSplitter: 递归分割，更智能
   - MarkdownHeaderTextSplitter: 按 Markdown 标题分割

3. Embeddings（嵌入模型）
   - OpenAIEmbeddings: OpenAI 的嵌入
   - HuggingFaceEmbeddings: Hugging Face 的嵌入
   - DeepSeekEmbeddings: DeepSeek 的嵌入

4. Vector Stores（向量数据库）
   - Chroma: 轻量级，易于使用
   - Pinecone: 云服务
   - Weaviate: 开源向量数据库
   - FAISS: Facebook 的向量搜索库

5. Retrievers（检索器）
   - VectorStoreRetriever: 基于向量相似度检索
   - ContextualCompressionRetriever: 压缩检索结果
   - MultiQueryRetriever: 多查询检索
            """,
            metadata={"source": "langchain_rag.txt", "page": 1},
        ),
        Document(
            page_content="""
RAG 最佳实践

1. 文本分块策略
   - 块大小：通常 500-2000 字符
   - 块重叠：建议有 10-20% 的重叠，保证上下文连贯
   - 考虑语义完整性：尽量在句子边界分割

2. 嵌入模型选择
   - 根据精度和速度需求选择
   - OpenAI 的 text-embedding-3-small 性价比高
   - 开源模型如 sentence-transformers 可以本地部署

3. 向量数据库选择
   - 小规模演示：Chroma（内存或文件）
   - 生产环境：Pinecone、Weaviate、Milvus
   - 需要考虑持久化、扩展性等因素

4. 检索优化
   - k 值：通常检索 top-k (k=3-5) 个相关文档
   - similarity_threshold：设置相似度阈值过滤不相关内容
   - reranking：对检索结果进行重排序

5. 提升回答质量
   - 在 prompt 中明确要求基于检索内容回答
   - 提供清晰的指令，说明如何引用来源
   - 考虑添加追问机制，让用户确认答案是否有用

常见陷阱：
- 分块太小导致丢失上下文
- 分块太大导致检索不精确
- 忽略文档元数据的利用
- 没有处理文档编码问题
            """,
            metadata={"source": "rag_best_practices.txt", "page": 1},
        ),
    ]

    return sample_docs
