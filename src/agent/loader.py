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


def load_sample_data() -> List[Document]:
    """加载示例数据 - 家常菜谱"""
    sample_docs = [
        Document(
            page_content="""
家常美味菜谱，简单易学

红烧肉做法：
1. 五花肉切块，冷水下锅加料酒姜片焯水
2. 炒糖色：冰糖小火炒化，放入肉块翻炒上色
3. 加调料：生抽、老抽、料酒、八角、桂皮、香叶
4. 加开水没过肉块，小火炖1小时
5. 大火收汁，加盐调味

小技巧：
- 选肥瘦相间的五花肉
- 炒糖色要小火，防止发苦
- 炖的时候加热水，不要加冷水

番茄炒蛋做法：
1. 番茄切块，鸡蛋打散加一点盐
2. 鸡蛋下锅炒到半熟盛出
3. 番茄下锅炒出汁水
4. 加入鸡蛋一起翻炒
5. 加盐、糖调味，出锅前撒葱花

小技巧：
- 番茄用开水烫一下好剥皮
- 炒鸡蛋时油要热，鸡蛋才嫩
- 加一点点糖可以中和酸味
            """,
            metadata={"source": "home_cooking.txt", "topic": "recipe"},
        ),
        Document(
            page_content="""
早餐吃什么？一周不重样

周一：鸡蛋三明治
- 吐司夹煎蛋、火腿、生菜
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
            metadata={"source": "breakfast.txt", "topic": "recipe"},
        ),
        Document(
            page_content="""
新手入门家常菜，这些一定要会

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
            """,
            metadata={"source": "beginner_cooking.txt", "topic": "recipe"},
        ),
    ]
    return sample_docs
