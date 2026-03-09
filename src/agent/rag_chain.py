"""
RAG Chain

将检索（Retriever）和生成（LLM）结合成完整 RAG 流程

LangChain 管道语法：
- RunnableParallel: 并行执行多个任务
- | (管道符): 将前一个输出作为后一个输入
- RunnableLambda: 将普通函数转为 Runnable
"""
from typing import Optional
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# RAG 系统提示词模板
RAG_SYSTEM_PROMPT = """你是一个基于检索内容回答问题的助手。

请遵循以下规则：
1. 只根据提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有相关信息，请明确说明"我没有找到足够的相关信息来回答这个问题"
3. 在回答时，可以提及你参考了哪些来源
4. 回答要简洁明了

上下文信息：
{context}

问题：{question}

回答："""


# 简化的 RAG 问题模板
SIMPLE_QUESTION_TEMPLATE = """基于以下上下文信息回答问题。

上下文：
{context}

问题：{question}

回答："""


def create_rag_prompt(
    system_prompt: Optional[str] = None,
    template: Optional[str] = None,
) -> ChatPromptTemplate:
    """
    创建 RAG 提示词模板

    Args:
        system_prompt: 系统提示词
        template: 用户问题模板

    Returns:
        ChatPromptTemplate 实例
    """
    if template is None:
        template = SIMPLE_QUESTION_TEMPLATE

    if system_prompt:
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", template),
        ])

    return ChatPromptTemplate.from_template(template)


def create_rag_chain(retriever, model, prompt: Optional[str] = None):
    """
    创建完整的 RAG 链

    RAG 链工作流程：
    1. 用户问题 → retriever 检索相关文档
    2. 文档格式化为上下文字符串
    3. prompt_template 组装"上下文 + 问题"
    4. LLM 基于上下文生成答案
    5. StrOutputParser 解析字符串输出

    Args:
        retriever: 检索器实例
        model: LLM 模型实例（DeepSeek/OpenAI）
        prompt: 自定义提示词（可选）

    Returns:
        可调用的 RAG 链
    """
    # 创建提示词模板
    prompt_template = create_rag_prompt(
        system_prompt=RAG_SYSTEM_PROMPT if prompt is None else prompt,
    )

    # 定义文档格式化函数
    def format_docs(docs):
        """将检索到的文档格式化为字符串"""
        formatted = []
        for doc in docs:
            # 支持 Document 对象和 dict 两种格式
            if hasattr(doc, "page_content"):
                formatted.append(doc.page_content)
            elif isinstance(doc, dict):
                formatted.append(doc.get("page_content", str(doc)))
            else:
                formatted.append(str(doc))
        return "\n\n".join(formatted)

    # 构建 RAG 链
    # 步骤：
    # 1. 输入问题
    # 2. retriever 检索相关文档
    # 3. format_docs 将文档列表格式化为字符串
    # 4. prompt_template 组合问题和上下文
    # 5. model 生成答案
    # 6. StrOutputParser 解析输出
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnableLambda(lambda x: x["question"]),
        )
        | prompt_template
        | model
        | StrOutputParser()
    )

    return rag_chain


def create_simple_rag_chain(retriever, model):
    """
    创建简化版的 RAG 链

    与 create_rag_chain 的区别：
    - 使用更简单的提示词模板
    - 更易于理解和修改

    Args:
        retriever: 检索器实例
        model: LLM 模型实例

    Returns:
        RAG 链实例
    """
    prompt = PromptTemplate.from_template(
        "基于以下上下文回答问题。\n\n"
        "上下文：\n{context}\n\n"
        "问题：{question}\n\n"
        "回答："
    )

    def format_docs(docs):
        """将检索到的文档格式化为字符串"""
        formatted = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                formatted.append(doc.page_content)
            elif isinstance(doc, dict):
                formatted.append(doc.get("page_content", str(doc)))
            else:
                formatted.append(str(doc))
        return "\n\n".join(formatted)

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnableLambda(lambda x: x["question"]),
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain


def invoke_rag(rag_chain, question: str) -> str:
    """
    调用 RAG 链进行问答

    Args:
        rag_chain: RAG 链实例
        question: 用户问题

    Returns:
        LLM 生成的回答
    """
    return rag_chain.invoke({"question": question})
