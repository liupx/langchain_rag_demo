"""
Retrievers

从向量数据库中检索与查询最相关的文档

检索策略：
- similarity: 基于向量余弦相似度，返回最相似的 k 个
- mmr (Maximum Marginal Relevance): 在相关基础上增加多样性，避免重复
- similarity_score_threshold: 设置相似度阈值，过滤低相关度结果
"""
from typing import List, Optional
from langchain_core.documents import Document


def create_retriever(
    vectorstore,
    search_type: str = "similarity",
    k: int = 3,
    score_threshold: Optional[float] = None,
    fetch_k: int = 20,
):
    """
    创建检索器

    Args:
        vectorstore: 向量数据库实例
        search_type: 搜索类型
            - "similarity": 默认，余弦相似度排序
            - "mmr": 多样性优先
            - "similarity_score_threshold": 阈值过滤
        k: 返回结果数量（通常 3~5）
        score_threshold: 相似度阈值（0~1，越高越严格）
        fetch_k: MMR 初始检索数量（从 fetch_k 中选 k 个多样的）

    Returns:
        检索器实例
    """
    search_kwargs = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k

    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def retrieve_by_similarity(vectorstore, query: str, k: int = 3) -> List[Document]:
    """
    基于相似度检索

    最常用的检索方式，返回与查询最相似的 k 个文档

    Args:
        vectorstore: 向量数据库实例
        query: 查询文本
        k: 返回结果数量

    Returns:
        相关文档列表
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever.invoke(query)


def retrieve_by_mmr(vectorstore, query: str, k: int = 3, fetch_k: int = 20) -> List[Document]:
    """
    基于 MMR (Maximum Marginal Relevance) 检索

    MMR 在相关性和多样性之间取得平衡：
    1. 先检索 fetch_k 个相关结果
    2. 然后从中选择 k 个多样化的结果

    适用场景：避免返回内容重复的文档

    Args:
        vectorstore: 向量数据库实例
        query: 查询文本
        k: 最终返回结果数量
        fetch_k: 初始检索数量（通常设为 k 的 3~5 倍）

    Returns:
        相关文档列表
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )
    return retriever.invoke(query)


def retrieve_by_threshold(
    vectorstore,
    query: str,
    k: int = 3,
    threshold: float = 0.5,
) -> List[Document]:
    """
    基于相似度阈值检索

    只返回相似度高于阈值的文档

    Args:
        vectorstore: 向量数据库实例
        query: 查询文本
        k: 返回结果数量上限
        threshold: 相似度阈值（0-1 之间）

    Returns:
        相关文档列表
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_threshold",
        search_kwargs={"k": k, "score_threshold": threshold},
    )
    return retriever.invoke(query)
