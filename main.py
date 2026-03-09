"""
LangChain RAG Demo - 主入口文件

演示 RAG 的完整流程：
1. 加载文档
2. 文本分块
3. 创建向量存储
4. 创建检索器
5. 构建 RAG 链
6. 问答
"""
import os
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from src.agent.llm import get_model
from src.agent.loader import load_sample_data
from src.agent.splitter import split_by_recursive
from src.agent.vectorstore import create_vectorstore
from src.agent.retriever import create_retriever
from src.agent.rag_chain import create_rag_chain, invoke_rag


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_basic_rag():
    """演示基本的 RAG 流程"""
    print_section("1. 初始化 LLM 模型")
    model = get_model()
    print(f"模型初始化成功: {model.model_name}")

    # Step 2: 加载文档
    print_section("2. 加载文档")
    print("加载示例数据（关于 RAG 知识的文档）...")
    documents = load_sample_data()
    print(f"成功加载 {len(documents)} 个文档")

    # 打印每个文档的基本信息
    for i, doc in enumerate(documents):
        print(f"\n文档 {i+1}:")
        print(f"  来源: {doc.metadata.get('source', 'unknown')}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
        print(f"  内容预览: {doc.page_content[:100].strip()}...")

    # Step 3: 文本分块
    print_section("3. 文本分块")
    print("将文档分割成较小的片段...")
    chunks = split_by_recursive(
        documents,
        chunk_size=300,
        chunk_overlap=50,
    )
    print(f"分割完成，共 {len(chunks)} 个文本块")

    # 打印前几个块的信息
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n块 {i+1} (来源: {chunk.metadata.get('source', 'unknown')}):")
        print(f"  长度: {len(chunk.page_content)} 字符")
        print(f"  内容: {chunk.page_content[:80].strip()}...")

    # Step 4: 创建向量存储
    print_section("4. 创建向量存储")
    print("将文本块转换为向量并存储...")
    vectorstore = create_vectorstore(
        chunks,
        collection_name="rag_demo",
        persist_directory=None,  # 使用内存存储
    )
    print("向量存储创建成功!")

    # Step 5: 创建检索器
    print_section("5. 创建检索器")
    print("配置检索参数...")
    retriever = create_retriever(
        vectorstore,
        search_type="similarity",
        k=2,
    )
    print("检索器创建成功!")

    # Step 6: 构建 RAG 链
    print_section("6. 构建 RAG 链")
    print("将检索器和 LLM 结合成完整的 RAG 链...")
    rag_chain = create_rag_chain(retriever, model)
    print("RAG 链构建成功!")

    # Step 7: 问答演示
    print_section("7. RAG 问答演示")

    # 测试问题
    test_questions = [
        "什么是 RAG？",
        "LangChain 有哪些 RAG 组件？",
        "RAG 的最佳实践有哪些？",
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        answer = invoke_rag(rag_chain, question)
        print(f"回答: {answer}")
        print()


def demo_retrieval_strategies():
    """演示不同的检索策略"""
    print_section("检索策略对比演示")

    # 加载数据并创建向量存储
    documents = load_sample_data()
    chunks = split_by_recursive(documents, chunk_size=300, chunk_overlap=50)
    vectorstore = create_vectorstore(chunks, persist_directory=None)
    model = get_model()

    # 测试查询
    query = "RAG 如何提升 LLM 的能力？"

    # 1. Similarity 检索
    print("\n1. Similarity 检索 (相似度优先):")
    retriever_sim = create_retriever(vectorstore, search_type="similarity", k=2)
    docs_sim = retriever_sim.invoke(query)
    for i, doc in enumerate(docs_sim):
        print(f"  结果 {i+1}: {doc.page_content[:60].strip()}...")

    # 2. MMR 检索
    print("\n2. MMR 检索 (多样性优先):")
    retriever_mmr = create_retriever(vectorstore, search_type="mmr", k=2, fetch_k=5)
    docs_mmr = retriever_mmr.invoke(query)
    for i, doc in enumerate(docs_mmr):
        print(f"  结果 {i+1}: {doc.page_content[:60].strip()}...")

    # 3. Threshold 检索
    print("\n3. Threshold 检索 (相似度阈值):")
    retriever_thresh = create_retriever(
        vectorstore, search_type="similarity_score_threshold", k=3, score_threshold=0.3
    )
    docs_thresh = retriever_thresh.invoke(query)
    for i, doc in enumerate(docs_thresh):
        print(f"  结果 {i+1}: {doc.page_content[:60].strip()}...")


def demo_streaming():
    """演示流式输出"""
    print_section("流式输出演示")

    # 加载数据并创建向量存储
    documents = load_sample_data()
    chunks = split_by_recursive(documents, chunk_size=300, chunk_overlap=50)
    vectorstore = create_vectorstore(chunks, persist_directory=None)
    model = get_model()
    retriever = create_retriever(vectorstore, k=2)

    # 创建支持流式的 RAG 链
    from langchain_core.runnables import RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "基于以下上下文回答问题。\n\n"
        "上下文：\n{context}\n\n"
        "问题：{question}\n\n"
        "回答："
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=lambda x: x["question"],
        )
        | prompt
        | model
        | StrOutputParser()
    )

    question = "LangChain 的 RAG 组件有哪些？"
    print(f"\n问题: {question}")
    print("回答 (流式输出): ")
    print("-" * 40)

    # 流式输出
    for chunk in rag_chain.stream({"question": question}):
        print(chunk, end="", flush=True)

    print("\n")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  LangChain RAG 演示")
    print("  什么是 RAG？如何使用 LangChain 实现 RAG？")
    print("=" * 60)

    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告: 未检测到 API Key!")
        print("请在 .env 文件中配置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
        print("参考 .env.example 文件创建配置文件")
        print("\n继续演示（可能会失败）...")

    # 运行演示
    try:
        # 1. 基本 RAG 流程
        demo_basic_rag()

        # 2. 检索策略对比
        demo_retrieval_strategies()

        # 3. 流式输出
        demo_streaming()

        print("\n" + "=" * 60)
        print("  演示完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请检查:")
        print("1. .env 文件中的 API Key 是否正确")
        print("2. 网络连接是否正常")
        print("3. 依赖是否正确安装 (uv sync)")


if __name__ == "__main__":
    main()
