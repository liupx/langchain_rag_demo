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
from src.agent.rag_chain import create_rag_chain


# 全局变量存储 RAG 系统组件
class RAGSystem:
    def __init__(self):
        self.documents = None
        self.chunks = None
        self.vectorstore = None
        self.retriever = None
        self.model = None
        self.rag_chain = None
        self._initialized = False

    def ensure_init(self):
        """确保已初始化"""
        if not self._initialized:
            init_rag_system()
        return self

    @property
    def is_ready(self):
        return self._initialized


rag_system = RAGSystem()


def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def init_rag_system():
    """初始化 RAG 系统"""
    print_section("初始化 RAG 系统")

    # 加载文档
    print("📄 加载文档...")
    rag_system.documents = load_sample_data()
    print(f"   已加载 {len(rag_system.documents)} 个文档")

    # 文本分块
    print("✂️ 文本分块...")
    rag_system.chunks = split_by_recursive(
        rag_system.documents,
        chunk_size=500,
        chunk_overlap=100,
    )
    print(f"   已分为 {len(rag_system.chunks)} 个片段")

    # 向量存储
    print("💾 创建向量索引（首次运行需下载模型，请耐心等待）...")
    rag_system.vectorstore = create_vectorstore(
        rag_system.chunks,
        collection_name="rag_demo",
        persist_directory=None,
    )

    # 检索器
    rag_system.retriever = create_retriever(rag_system.vectorstore, search_type="similarity", k=2)

    # LLM
    print("🤖 初始化 LLM...")
    rag_system.model = get_model()

    # RAG 链
    rag_system.rag_chain = create_rag_chain(rag_system.retriever, rag_system.model)

    rag_system._initialized = True
    print("   ✅ RAG 系统就绪!")
    return rag_system


def show_document_list():
    """显示文档列表供用户选择"""
    rag_system.ensure_init()

    while True:
        print("\n" + "=" * 50)
        print("  📚 请选择要查看的文档")
        print("=" * 50)

        for i, doc in enumerate(rag_system.documents):
            source = doc.metadata.get('source', '未知')
            length = len(doc.page_content)
            preview = doc.page_content.strip()[:30].replace('\n', ' ')
            print(f"  {i+1}. [{source}] ({length}字符)")
            print(f"     {preview}...")

        print("\n  0. 返回上一级")
        print("-" * 40)

        choice = input("请选择 (0-{0}): ".format(len(rag_system.documents))).strip()

        if choice == '0':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(rag_system.documents):
                show_document_detail(idx)
            else:
                print("⚠️ 无效选项，请重新选择")
        except ValueError:
            print("⚠️ 请输入数字")


def show_document_detail(index: int):
    """显示单个文档的详细内容"""
    doc = rag_system.documents[index]
    source = doc.metadata.get('source', '未知')

    print("\n" + "=" * 50)
    print(f"  📄 {source}")
    print("=" * 50)
    print(doc.page_content.strip())
    print("\n" + "-" * 40)
    input("按回车键返回文档列表...")


def show_chunk_list():
    """显示分块列表供用户选择"""
    rag_system.ensure_init()

    while True:
        print("\n" + "=" * 50)
        print("  📦 请选择要查看的分块")
        print("=" * 50)

        for i, chunk in enumerate(rag_system.chunks):
            source = chunk.metadata.get('source', '未知')
            preview = chunk.page_content.strip()[:40].replace('\n', ' ')
            print(f"  {i+1}. [{source}] {preview}...")

        print("\n  0. 返回上一级")
        print("-" * 40)

        choice = input("请选择 (0-{0}): ".format(len(rag_system.chunks))).strip()

        if choice == '0':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(rag_system.chunks):
                show_chunk_detail(idx)
            else:
                print("⚠️ 无效选项，请重新选择")
        except ValueError:
            print("⚠️ 请输入数字")


def show_chunk_detail(index: int):
    """显示单个分块的详细内容"""
    chunk = rag_system.chunks[index]
    source = chunk.metadata.get('source', '未知')

    print("\n" + "=" * 50)
    print(f"  📦 片段 {index+1} - {source}")
    print("=" * 50)
    print(chunk.page_content.strip())
    print("\n" + "-" * 40)
    input("按回车键返回分块列表...")


def get_recommended_questions() -> list:
    """根据知识库生成推荐问题"""
    recommendations = []
    sources = set(doc.metadata.get('source', '') for doc in rag_system.documents)

    if 'home_cooking.txt' in sources:
        recommendations.extend([
            "红烧肉怎么做？",
            "番茄炒蛋有什么技巧？",
        ])

    if 'breakfast.txt' in sources:
        recommendations.extend([
            "早餐吃什么好？",
            "一周早餐不重样",
        ])

    if 'beginner_cooking.txt' in sources:
        recommendations.extend([
            "新手学什么菜简单？",
            "懒人做什么菜快？",
        ])

    return recommendations[:4]


def interactive问答():
    """交互式问答"""
    rag_system.ensure_init()

    print("\n" + "=" * 50)
    print("  🎯 交互式问答")
    print("=" * 50)

    # 显示推荐问题
    recommendations = get_recommended_questions()
    print("\n💡 你可以问我一些问题，例如：")
    for i, q in enumerate(recommendations, 1):
        print(f"   {i}. {q}")

    print("\n" + "-" * 30)
    print("输入问题进行咨询（输入 q 退出）")

    while True:
        question = input("\n👤 你: ").strip()

        if not question:
            continue

        if question.lower() in ['q', 'quit', 'exit', '退出']:
            print("\n👋 再见!")
            break

        print("\n🤖 AI: ", end="", flush=True)

        try:
            docs = rag_system.retriever.invoke(question)

            for chunk in rag_system.rag_chain.stream({"question": question}):
                print(chunk, end="", flush=True)

            if docs:
                print("\n\n📖 引用来源:")
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get('source', '未知')
                    content = doc.page_content.strip()[:80]
                    print(f"  {i}. [{source}] {content}...")

        except Exception as e:
            print(f"\n❌ 抱歉，出错了: {e}")


def show_menu():
    """显示菜单"""
    print("\n" + "=" * 50)
    print("  🌟 LangChain RAG 演示")
    print("=" * 50)
    print("""
请选择操作：
  1. 📚 查看原始文档
  2. 📦 查看分块数据
  3. 💬 开始问答
  4. 🚪 退出
""")
    return input("请输入选项 (1-4): ").strip()


def main():
    """主函数"""
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  请先在 .env 文件中配置 API Key")
        print("   参考 .env.example 文件")
        return

    # 主循环 - 初始不加载模型
    while True:
        choice = show_menu()

        if choice == '1':
            show_document_list()
        elif choice == '2':
            show_chunk_list()
        elif choice == '3':
            interactive问答()
        elif choice == '4' or choice.lower() in ['q', 'quit', 'exit']:
            print("\n👋 再见!")
            break
        else:
            print("\n⚠️  无效选项，请重新选择")


if __name__ == "__main__":
    main()
