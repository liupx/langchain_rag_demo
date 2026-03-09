"""
Vector Stores

将文本转换为向量（Embedding）并存储到向量数据库

核心概念：
- Embedding（嵌入）：将文本转为数值向量，语义相似的文本向量也相似
- 向量检索：通过计算向量距离（余弦相似度）找到相关内容
- Chroma：轻量级向量数据库，支持内存和持久化存储
"""
import os
from typing import List, Optional, Union
from langchain_core.documents import Document

# 尝试导入向量存储和嵌入
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False

# 尝试导入 HuggingFace 嵌入（免费备选）
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        HF_EMBEDDINGS_AVAILABLE = True
    except ImportError:
        HF_EMBEDDINGS_AVAILABLE = False


class WrappedHuggingFaceEmbeddings:
    """
    包装 HuggingFaceEmbeddings，处理输入可能是 dict 的情况

    新版本 LangChain 中，retriever 可能会传入 dict 而不是字符串
    """
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed_query(self, text: Union[str, dict]) -> List[float]:
        """处理 query，可能是字符串或 dict"""
        if isinstance(text, dict):
            # 尝试从 dict 中提取查询文本
            text = text.get("query") or text.get("question") or str(text)
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return self._embeddings.embed_documents(texts)

    def __call__(self, input: Union[str, dict]) -> List[float]:
        """支持直接调用"""
        return self.embed_query(input)


def get_embeddings():
    """
    获取嵌入模型实例

    优先级：
    1. OpenAI embedding（如果配置了 OPENAI_API_KEY）
    2. HuggingFace embedding（免费，使用本地模型）
    3. OpenAI embedding（如果配置了 DEEPSEEK_API_KEY，但不推荐，DeepSeek 不提供 embedding 服务）

    Returns:
        嵌入模型实例
    """
    if not VECTORSTORE_AVAILABLE:
        raise ImportError("请安装 langchain-openai 和 langchain-community")

    # 优先使用 OpenAI（如果配置了）
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # 回退到 HuggingFace（免费，无需 API Key）
    # 使用中文优化模型 bge-small-zh，对中文检索效果更好
    if HF_EMBEDDINGS_AVAILABLE:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
        )
        # 包装一层以处理 dict 输入
        return WrappedHuggingFaceEmbeddings(embeddings)

    # 最后尝试使用 DeepSeek（不推荐，DeepSeek 不提供 embedding 服务）
    if os.getenv("DEEPSEEK_API_KEY"):
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        )

    raise ValueError("请配置 OPENAI_API_KEY 或安装 langchain-huggingface")


def create_chroma_vectorstore(
    documents: List[Document],
    collection_name: str = "rag_demo",
    persist_directory: Optional[str] = None,
) -> "Chroma":
    """
    创建 Chroma 向量数据库

    Args:
        documents: 文档列表（会被转换为向量）
        collection_name: 集合名称（类似表名，用于区分不同知识库）
        persist_directory: 持久化目录
            - None: 内存存储，程序结束数据丢失
            - 指定路径: 磁盘存储，程序重启后仍可用

    Returns:
        Chroma 向量数据库实例
    """
    if not VECTORSTORE_AVAILABLE:
        raise ImportError("请安装 langchain-openai 和 langchain-community")

    embeddings = get_embeddings()

    # 创建向量数据库
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return vectorstore


def create_vectorstore(
    documents: List[Document],
    store_type: str = "chroma",
    collection_name: str = "rag_demo",
    persist_directory: Optional[str] = None,
) -> "Chroma":
    """
    创建向量存储的工厂函数

    Args:
        documents: 文档列表
        store_type: 向量存储类型，目前只支持 chroma
        collection_name: 集合名称
        persist_directory: 持久化目录

    Returns:
        向量数据库实例
    """
    if store_type == "chroma":
        return create_chroma_vectorstore(
            documents,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    else:
        raise ValueError(f"不支持的向量存储类型: {store_type}")


def load_vectorstore(
    persist_directory: str,
    collection_name: str = "rag_demo",
) -> "Chroma":
    """
    加载已有的向量数据库

    Args:
        persist_directory: 持久化目录
        collection_name: 集合名称

    Returns:
        Chroma 向量数据库实例
    """
    if not VECTORSTORE_AVAILABLE:
        raise ImportError("请安装 langchain-openai 和 langchain-community")

    embeddings = get_embeddings()

    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
