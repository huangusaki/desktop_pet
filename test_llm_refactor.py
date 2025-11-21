"""
LLM 客户端测试脚本
测试重构后的 LLM 客户端功能
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm.openai_client import OpenAIClient, OpenAIResponse
from src.llm.llm_request import LLM_request, GeminiSDKResponse
from src.llm.utils import compress_base64_image_by_scale, extract_reasoning


async def test_openai_client():
    """测试 OpenAI 客户端"""
    print("\n" + "=" * 50)
    print("测试 OpenAI 客户端")
    print("=" * 50)
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  未设置 OPENAI_API_KEY 环境变量,跳过测试")
        return
        
    try:
        client = OpenAIClient(
            api_key=api_key,
            model_name="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1",
            temperature=0.7,
            max_tokens=100,
        )
        
        print("✓ OpenAI 客户端初始化成功")
        
        # 测试聊天
        response = await client.generate_response(
            messages="你好,请用一句话介绍你自己。",
            user_id="test_user",
        )
        
        print(f"✓ 聊天响应: {response.content[:100]}...")
        print(f"  Token 使用: {response.total_tokens}")
        
    except Exception as e:
        print(f"✗ OpenAI 客户端测试失败: {e}")
        

async def test_llm_request_google():
    """测试 LLM_request (Google GenAI)"""
    print("\n" + "=" * 50)
    print("测试 LLM_request (Google GenAI)")
    print("=" * 50)
    
    # 检查 API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  未设置 GEMINI_API_KEY 环境变量,跳过测试")
        return
        
    try:
        # 创建简单的 DB mock
        class MockDB:
            class MockCollection:
                def insert_one(self, data):
                    pass
                def create_index(self, *args, **kwargs):
                    pass
            
            def __init__(self):
                self.llm_usage = self.MockCollection()
                
        model_config = {
            "name": "gemini-2.0-flash-exp",
            "key": api_key,
            "temperature": 0.7,
            "max_output_tokens": 100,
        }
        
        client = LLM_request(
            model_config=model_config,
            db_instance=MockDB(),
            request_type="test",
        )
        
        print("✓ LLM_request 客户端初始化成功")
        
        # 测试生成响应
        response = await client.generate_response(
            prompt="你好,请用一句话介绍你自己。",
            user_id="test_user",
        )
        
        print(f"✓ 生成响应: {response.content[:100]}...")
        print(f"  Token 使用: {response.total_tokens}")
        
    except Exception as e:
        print(f"✗ LLM_request 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_utils():
    """测试工具函数"""
    print("\n" + "=" * 50)
    print("测试工具函数")
    print("=" * 50)
    
    try:
        # 测试 extract_reasoning
        text_with_thinking = "这是回复内容。<think>这是思考过程</think>"
        content, reasoning = extract_reasoning(text_with_thinking)
        
        assert content == "这是回复内容。", f"内容提取错误: {content}"
        assert reasoning == "这是思考过程", f"思考提取错误: {reasoning}"
        
        print("✓ extract_reasoning 测试通过")
        
        # 测试 compress_base64_image_by_scale
        # 创建一个简单的测试图片
        from PIL import Image
        import io
        import base64
        
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # 压缩(目标大小很小,应该会触发压缩)
        compressed = compress_base64_image_by_scale(img_base64, target_size=1000)
        
        assert isinstance(compressed, str), "压缩结果应该是字符串"
        print("✓ compress_base64_image_by_scale 测试通过")
        
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print(" LLM 客户端重构测试")
    print("=" * 60)
    
    # 测试工具函数
    test_utils()
    
    # 测试 Google GenAI
    await test_llm_request_google()
    
    # 测试 OpenAI
    await test_openai_client()
    
    print("\n" + "=" * 60)
    print(" 测试完成")
    print("=" * 60)
    

if __name__ == "__main__":
    asyncio.run(main())
