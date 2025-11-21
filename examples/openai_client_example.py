"""
OpenAI 客户端使用示例
演示如何使用新添加的 OpenAI 客户端
"""
import asyncio
import os
from src.llm.openai_client import OpenAIClient


async def example_openai_usage():
    """OpenAI 客户端使用示例"""
    
    # 从配置或环境变量获取 API Key
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    # 初始化客户端
    client = OpenAIClient(
        api_key=api_key,
        model_name="gpt-3.5-turbo",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
        max_tokens=1000,
    )
    
    # 示例 1: 简单对话
    print("=== 示例 1: 简单对话 ===")
    response = await client.generate_response(
        messages="你好,请介绍一下你自己。"
    )
    print(f"回复: {response.content}")
    print(f"Token 使用: {response.total_tokens}")
    
    # 示例 2: 多轮对话
    print("\n=== 示例 2: 多轮对话 ===")
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "什么是机器学习?"},
    ]
    response = await client.generate_response(messages=messages)
    print(f"回复: {response.content}")
    
    # 示例 3: 获取嵌入向量
    print("\n=== 示例 3: 获取嵌入向量 ===")
    embedding_client = OpenAIClient(
        api_key=api_key,
        model_name="text-embedding-ada-002",
        base_url="https://api.openai.com/v1",
    )
    embedding = await embedding_client.get_embedding("这是一段测试文本")
    if embedding:
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"前5个值: {embedding[:5]}")
    
    # 示例 4: 使用兼容的第三方 API
    print("\n=== 示例 4: 使用兼容的第三方 API ===")
    # 例如使用本地部署的模型
    local_client = OpenAIClient(
        api_key="dummy_key",  # 某些本地 API 不需要真实 key
        model_name="llama-3-8b",
        base_url="http://localhost:8000/v1",  # 本地 API 地址
        temperature=0.7,
    )
    # response = await local_client.generate_response("你好")
    print("本地 API 客户端已配置(需要本地服务运行)")


if __name__ == "__main__":
    asyncio.run(example_openai_usage())
