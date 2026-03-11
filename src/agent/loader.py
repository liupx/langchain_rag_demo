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
LOADER_AVAILABLE = False
TextLoader = None
MarkdownLoader = None
PyPDFLoader = None
DirectoryLoader = None

try:
    from langchain_community.document_loaders import TextLoader
except ImportError:
    pass

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    pass

# DirectoryLoader 和 MarkdownLoader 可能在某些版本中不可用
try:
    from langchain_community.document_loaders import DirectoryLoader
except ImportError:
    pass

try:
    from langchain_community.document_loaders import MarkdownLoader
except ImportError:
    pass

# 只要有基本的加载器就标记为可用
if TextLoader is not None or PyPDFLoader is not None:
    LOADER_AVAILABLE = True


def find_project_root(start: Path | None = None) -> Path:
    """从起始目录向上查找项目根目录（包含 pyproject.toml 或 .git）"""
    current = (start or Path(__file__)).resolve()

    # 向上遍历直到找到项目根目录标记
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / ".git").exists():
            return current
        current = current.parent

    # 没找到则返回 pyproject.toml 所在目录或当前工作目录
    return Path.cwd()


def get_data_dir() -> Path:
    """获取 data 目录路径"""
    return find_project_root() / "data"


def load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """加载文本文件"""
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")
    loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    """加载 Markdown 文件"""
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")
    loader = MarkdownLoader(file_path)
    return loader.load()


def load_pdf(file_path: str) -> List[Document]:
    """加载 PDF 文件"""
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_directory(
    directory_path: str,
    glob_pattern: str = "**/*",
    loader_class: Optional[type] = None,
) -> List[Document]:
    """加载目录下的所有文件"""
    if not LOADER_AVAILABLE:
        raise ImportError("请安装 langchain-community: pip install langchain-community")
    loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=loader_class)
    return loader.load()


def get_document_loader(file_path: str) -> List[Document]:
    """根据文件扩展名自动选择合适的加载器"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(file_path)
    elif suffix == ".md":
        return load_markdown(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    else:
        return load_text_file(file_path)


# 示例文档内容 - 克隆项目后首次运行时会自动创建
SAMPLE_DOCUMENTS = {
    "home_cooking.txt": """家常菜做法

红烧肉：
- 五花肉切块焯水
- 冰糖炒糖色
- 加八角、桂皮、香叶
- 加生抽、老抽、料酒
- 小火炖1小时

番茄炒蛋：
- 番茄切块，鸡蛋打散
- 先炒鸡蛋盛出
- 再炒番茄加盐糖
- 最后放入鸡蛋翻炒
- 出锅前撒葱花
""",
    "breakfast.txt": """早餐吃什么？一周不重样

周一：鸡蛋三明治
- 吐司夹煎蛋，火腿、生菜
- 配一杯牛奶或豆浆
- 10分钟搞定

周二：皮蛋瘦肉粥
- 前一晚预约煮粥
- 早上加皮蛋、肉丝、葱花
- 暖胃又营养

周三：煎饼果子
- 面粉加水调成稀面糊
- 平底锅摊成薄饼
- 打入鸡蛋、撒葱花、加油条
- 甜面酱调味

周四：面条
- 汤面：清汤加青菜、荷包蛋
- 拌面：麻酱拌面、黄瓜丝
- 快捷方便

周五：馄饨
- 提前包好冷冻
- 早上直接煮
- 紫菜虾皮汤底

周末：可以睡个懒觉，吃点好的
- 手抓饼加鸡蛋培根
- 沙拉配全麦面包
- 水果酸奶麦片
""",
    "beginner_cooking.txt": """新手入门家常菜，这些一定要会

懒人必备快手菜：
1. 蒜蓉西兰花
   - 西兰花焯水2分钟
   - 蒜末炒香，加盐蚝油
   - 淋在西兰花上

2. 青椒土豆丝
   - 土豆切丝泡水去淀粉
   - 油热下锅翻炒
   - 加醋加盐，出锅前加蒜末

3. 可乐鸡翅
   - 鸡翅划刀，加姜片焯水
   - 煎到两面金黄
   - 倒可乐没过鸡翅，加酱油
   - 小火收汁

新手常见问题：
- 盐放多了：加水或加土豆
- 菜炒糊了：火太大，油太少
- 味道淡了：出锅前尝一下

厨房小工具推荐：
- 削皮器：省时省力
- 刨丝器：土豆丝神器
- 定时器：防止煮过头
- 案板：生熟分开
"""
}


def ensure_data_dir(exist_ok: bool = False) -> Path:
    """确保 data 目录存在，如不存在则创建"""
    data_dir = get_data_dir()
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建数据目录: {data_dir}")
    return data_dir


def init_sample_data() -> List[Document]:
    """
    初始化示例文档到 data 目录

    Returns:
        Document 列表
    """
    data_dir = ensure_data_dir()

    # 写入示例文档
    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = data_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            print(f"   已创建示例文档: {filename}")

    return load_sample_data()


def load_sample_data() -> List[Document]:
    """
    加载 data 目录下的所有文档作为示例

    支持的文件格式：.txt, .md, .pdf

    Returns:
        Document 列表
    """
    data_dir = get_data_dir()

    if not data_dir.exists():
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")

    # 支持的文件类型
    supported_extensions = ["*.txt", "*.md", "*.pdf"]
    documents = []

    for ext in supported_extensions:
        for file_path in data_dir.glob(ext):
            try:
                docs = get_document_loader(str(file_path))
                for doc in docs:
                    # 确保 metadata 包含文件名
                    doc.metadata["source"] = file_path.name
                documents.extend(docs)
            except Exception as e:
                print(f"警告：加载文件 {file_path} 失败: {e}")

    if not documents:
        raise FileNotFoundError(f"data 目录下没有找到文档: {data_dir}")

    return documents


def list_sample_files() -> List[str]:
    """列出 data 目录下的所有文档文件"""
    data_dir = get_data_dir()
    if not data_dir.exists():
        return []

    files = []
    for ext in ["*.txt", "*.md", "*.pdf"]:
        files.extend([f.name for f in data_dir.glob(ext)])

    return sorted(files)
