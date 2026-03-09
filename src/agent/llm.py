"""
LLM Model Initialization

支持 DeepSeek 和 OpenAI 两种模型
"""
import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_model(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> ChatOpenAI:
    """
    获取 LLM 模型实例

    Args:
        model_name: 模型名称，默认从环境变量读取
        temperature: 温度参数，控制输出随机性
            - 0.0~0.3: 确定性输出，适合问答
            - 0.7~1.0: 创造性输出，适合创意写作

    Returns:
        ChatOpenAI 模型实例
    """
    # 优先使用 DeepSeek（价格更便宜）
    if os.getenv("DEEPSEEK_API_KEY"):
        return ChatOpenAI(
            model=model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=temperature,
        )

    # 回退到 OpenAI
    return ChatOpenAI(
        model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
    )
