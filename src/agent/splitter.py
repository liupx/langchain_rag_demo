"""
Text Splitters

将长文档分割成较小的片段（Chunk），以便检索和嵌入

为什么要分块：
- 向量模型有输入长度限制（通常 512~4096 tokens）
- 小块便于精确检索，避免返回无关内容
- 块之间重叠（overlap）保持上下文连贯
"""
from typing import List, Optional
from langchain_core.documents import Document

# 尝试导入 LangChain 文本分割器
try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        MarkdownTextSplitter,
    )
    SPLITTER_AVAILABLE = True
except ImportError:
    SPLITTER_AVAILABLE = False


def split_by_character(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> List[Document]:
    """
    按字符数分割文档

    Args:
        documents: 文档列表
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        separator: 分隔符

    Returns:
        分割后的文档列表
    """
    if not SPLITTER_AVAILABLE:
        raise ImportError("请安装 langchain-text-splitters: pip install langchain-text-splitters")

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        length_function=len,
    )

    return splitter.split_documents(documents)


def split_by_recursive(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    递归字符分割 - 更智能的分割方式

    按优先级尝试不同分隔符：段落 → 换行 → 句子 → 单词
    尽量在语义边界（句子/段落）分割，保留完整含义

    Args:
        documents: 文档列表
        chunk_size: 每个块的最大字符数（不宜过大，否则检索不精确）
        chunk_overlap: 块之间重叠字符数（推荐 10~20%，保持上下文）

    Returns:
        分割后的文档列表
    """
    if not SPLITTER_AVAILABLE:
        raise ImportError("请安装 langchain-text-splitters: pip install langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # 按这些分隔符依次尝试分割
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
    )

    return splitter.split_documents(documents)


def split_by_markdown(
    markdown_text: str,
    headers_to_split_on: Optional[List[tuple]] = None,
) -> List[Document]:
    """
    按 Markdown 标题分割文档

    Args:
        markdown_text: Markdown 文本
        headers_to_split_on: 要分割的标题级别，格式为 [(#, "Header1"), (##, "Header2")]

    Returns:
        分割后的文档列表
    """
    if not SPLITTER_AVAILABLE:
        raise ImportError("请安装 langchain-text-splitters: pip install langchain-text-splitters")

    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
        ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return splitter.split_text(markdown_text)


def split_text(
    documents: List[Document],
    method: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    统一的文本分割接口

    Args:
        documents: 文档列表
        method: 分割方法，"character" 或 "recursive"
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数

    Returns:
        分割后的文档列表
    """
    if method == "character":
        return split_by_character(documents, chunk_size, chunk_overlap)
    elif method == "recursive":
        return split_by_recursive(documents, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"未知的分割方法: {method}")


def get_text_splitter(
    method: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """
    获取文本分割器实例

    Args:
        method: 分割方法，"character" 或 "recursive"
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数

    Returns:
        文本分割器实例
    """
    if not SPLITTER_AVAILABLE:
        raise ImportError("请安装 langchain-text-splitters: pip install langchain-text-splitters")

    if method == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n",
            length_function=len,
        )
    elif method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
    else:
        raise ValueError(f"未知的分割方法: {method}")
