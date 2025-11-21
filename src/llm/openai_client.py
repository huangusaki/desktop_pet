"""
OpenAI 格式 API 客户端
支持标准 OpenAI API 以及兼容的第三方 API
"""
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from .base_client import BaseLLMClient, UsageTracker
from .utils import extract_reasoning

logger = logging.getLogger("openai_client")


class OpenAIResponse:
    """OpenAI API 响应包装类"""
    
    def __init__(
        self,
        content: str,
        reasoning: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ):
        self.content = content
        self.reasoning = reasoning if reasoning is not None else ""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.raw_response = raw_response
        self.finish_reason = finish_reason
        
    def to_dict(self):
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "finish_reason": self.finish_reason,
        }


class OpenAIClient(BaseLLMClient):
    """
    OpenAI 格式 API 客户端
    支持标准 OpenAI API 和兼容的第三方 API(如 Azure OpenAI, 本地模型等)
    """
    
    def __init__(
        self,
        api_key: Union[str, List[str]],
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "https://api.openai.com/v1",
        db_instance=None,
        max_retries: int = 3,
        base_retry_wait_seconds: int = 3,
        timeout_seconds: int = 30,
        **default_params,
    ):
        """
        初始化 OpenAI 客户端
        
        Args:
            api_key: API key 或 key 列表
            model_name: 模型名称
            base_url: API 基础 URL
            db_instance: 数据库实例(用于记录使用情况)
            max_retries: 最大重试次数
            base_retry_wait_seconds: 基础重试等待时间
            timeout_seconds: 请求超时时间
            **default_params: 默认参数(temperature, max_tokens 等)
        """
        # 处理 API key
        if isinstance(api_key, str):
            api_keys = [api_key]
        else:
            api_keys = api_key
            
        super().__init__(
            api_keys=api_keys,
            model_name=model_name,
            max_retries=max_retries,
            base_retry_wait_seconds=base_retry_wait_seconds,
        )
        
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_params = default_params
        
        # 使用统计跟踪器
        self.usage_tracker = UsageTracker(db_instance, model_name)
        
        logger.info(
            f"OpenAI 客户端初始化完成: 模型={model_name}, "
            f"URL={base_url}, Keys={len(api_keys)}"
        )
        
    async def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str = "system",
        request_type: str = "chat",
        **kwargs,
    ) -> OpenAIResponse:
        """
        生成聊天响应
        
        Args:
            messages: 消息列表或单个文本
            user_id: 用户 ID
            request_type: 请求类型
            **kwargs: 其他参数(会覆盖默认参数)
            
        Returns:
            OpenAIResponse 对象
        """
        # 构建消息列表
        if isinstance(messages, str):
            message_list = [{"role": "user", "content": messages}]
        else:
            message_list = messages
            
        # 合并参数
        params = self.default_params.copy()
        params.update(kwargs)
        
        # 构建请求 payload
        payload = {
            "model": self.model_name,
            "messages": message_list,
        }
        
        # 添加可选参数
        if "temperature" in params:
            payload["temperature"] = params["temperature"]
        if "max_tokens" in params:
            payload["max_tokens"] = params["max_tokens"]
        if "top_p" in params:
            payload["top_p"] = params["top_p"]
        if "frequency_penalty" in params:
            payload["frequency_penalty"] = params["frequency_penalty"]
        if "presence_penalty" in params:
            payload["presence_penalty"] = params["presence_penalty"]
        if "stop" in params:
            payload["stop"] = params["stop"]
        if "stream" in params:
            payload["stream"] = params["stream"]
        if "functions" in params:
            payload["functions"] = params["functions"]
        if "function_call" in params:
            payload["function_call"] = params["function_call"]
        if "tools" in params:
            payload["tools"] = params["tools"]
        if "tool_choice" in params:
            payload["tool_choice"] = params["tool_choice"]
            
        # 执行请求
        async def _make_request():
            return await self._post_request(
                endpoint="chat/completions",
                payload=payload,
            )
            
        try:
            response_data = await self._execute_with_retry(
                _make_request,
                operation_name=f"OpenAI Chat ({self.model_name})",
            )
            
            # 解析响应
            content = ""
            finish_reason = None
            
            choices = response_data.get("choices", [])
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                finish_reason = choice.get("finish_reason")
                
            # 提取思考过程
            cleaned_content, reasoning = extract_reasoning(content)
            
            # 获取使用统计
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            
            # 记录使用情况
            self.usage_tracker.record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                user_id=user_id,
                request_type=request_type,
                endpoint="chat/completions",
                status="success",
                api_key_hint=f"...{self._get_current_api_key()[-4:]}",
            )
            
            return OpenAIResponse(
                content=cleaned_content,
                reasoning=reasoning,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=response_data,
                finish_reason=finish_reason,
            )
            
        except Exception as e:
            logger.error(f"OpenAI Chat 请求失败: {e}")
            
            # 记录失败
            self.usage_tracker.record_usage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                user_id=user_id,
                request_type=request_type,
                endpoint="chat/completions",
                status="failure",
                error_message=str(e),
                api_key_hint=f"...{self._get_current_api_key()[-4:]}",
            )
            
            return OpenAIResponse(
                content=f"API调用失败: {e}",
                reasoning=str(e),
                raw_response={"error": str(e)},
            )
            
    async def get_embedding(
        self,
        text: str,
        user_id: str = "system",
        request_type: str = "embedding",
        **kwargs,
    ) -> Optional[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            text: 输入文本
            user_id: 用户 ID
            request_type: 请求类型
            **kwargs: 其他参数
            
        Returns:
            嵌入向量列表,失败返回 None
        """
        if not text or not text.strip():
            logger.warning("嵌入请求的文本为空")
            return None
            
        # 构建请求 payload
        payload = {
            "model": self.model_name,
            "input": text,
        }
        
        # 添加可选参数
        if "encoding_format" in kwargs:
            payload["encoding_format"] = kwargs["encoding_format"]
            
        # 执行请求
        async def _make_request():
            return await self._post_request(
                endpoint="embeddings",
                payload=payload,
            )
            
        try:
            response_data = await self._execute_with_retry(
                _make_request,
                operation_name=f"OpenAI Embedding ({self.model_name})",
            )
            
            # 解析响应
            embedding = None
            if "data" in response_data and response_data["data"]:
                embedding = response_data["data"][0].get("embedding")
                
            if not embedding:
                logger.error(f"未能从响应中提取嵌入向量: {response_data}")
                return None
                
            # 获取使用统计
            usage = response_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            total_tokens = usage.get("total_tokens")
            
            # 记录使用情况
            self.usage_tracker.record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
                user_id=user_id,
                request_type=request_type,
                endpoint="embeddings",
                status="success",
                api_key_hint=f"...{self._get_current_api_key()[-4:]}",
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI Embedding 请求失败: {e}")
            
            # 记录失败
            self.usage_tracker.record_usage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                user_id=user_id,
                request_type=request_type,
                endpoint="embeddings",
                status="failure",
                error_message=str(e),
                api_key_hint=f"...{self._get_current_api_key()[-4:]}",
            )
            
            return None
            
    async def _post_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        发送 POST 请求到 OpenAI API
        
        Args:
            endpoint: API 端点(如 "chat/completions")
            payload: 请求数据
            
        Returns:
            响应 JSON 数据
            
        Raises:
            aiohttp.ClientResponseError: HTTP 错误
            Exception: 其他错误
        """
        url = f"{self.base_url}/{endpoint}"
        current_key = self._get_current_api_key()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}",
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=payload, timeout=timeout) as response:
                # 检查响应状态
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"OpenAI API 错误 ({response.status}): {error_text[:500]}"
                    )
                    response.raise_for_status()
                    
                return await response.json()
