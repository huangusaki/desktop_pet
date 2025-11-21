"""
LLM 客户端基类
提供通用的 API key 管理、重试逻辑、错误处理等功能
"""
import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from datetime import datetime

logger = logging.getLogger("base_llm_client")


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""
    
    def __init__(
        self,
        api_keys: List[str],
        model_name: str,
        max_retries: int = 3,
        base_retry_wait_seconds: int = 3,
    ):
        """
        初始化基础客户端
        
        Args:
            api_keys: API key 列表
            model_name: 模型名称
            max_retries: 最大重试次数
            base_retry_wait_seconds: 基础重试等待时间(秒)
        """
        if not api_keys:
            raise ValueError("至少需要提供一个 API Key")
            
        self.api_keys = api_keys
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_retry_wait_seconds = base_retry_wait_seconds
        self.current_key_index = 0
        
    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key"""
        return self.api_keys[self.current_key_index]
        
    def _switch_key(self) -> bool:
        """
        切换到下一个可用的 API Key
        
        Returns:
            是否成功切换(如果只有一个 key 则返回 False)
        """
        if len(self.api_keys) <= 1:
            return False
            
        current_key_before_switch = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_key_index]
        
        logger.warning(
            f"模型 ({self.model_name}) 遇到可重试错误，从 Key ...{current_key_before_switch[-4:]} "
            f"切换到 Key #{self.current_key_index + 1}/{len(self.api_keys)} (...{new_key[-4:]})。"
        )
        return True
        
    async def _execute_with_retry(
        self,
        operation_func,
        operation_name: str = "API 请求",
        is_retryable_error_func=None,
    ) -> Any:
        """
        执行操作并在失败时重试
        
        Args:
            operation_func: 要执行的异步操作函数
            operation_name: 操作名称(用于日志)
            is_retryable_error_func: 判断错误是否可重试的函数
            
        Returns:
            操作结果
            
        Raises:
            RuntimeError: 达到最大重试次数后仍失败
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"尝试 {operation_name} (Attempt {attempt + 1}/{self.max_retries}, "
                    f"Key #{self.current_key_index + 1})"
                )
                return await operation_func()
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                logger.warning(
                    f"{operation_name} 失败 (尝试 {attempt + 1}/{self.max_retries}): "
                    f"{type(e).__name__} - {e}"
                )
                
                # 判断是否为可重试错误
                is_retryable = self._is_retryable_error(error_str)
                if is_retryable_error_func:
                    is_retryable = is_retryable_error_func(e, error_str)
                    
                if not is_retryable:
                    logger.error(f"{operation_name}: 遇到不可重试的错误，将直接失败。")
                    raise e
                    
                # 如果还有重试机会
                if attempt < self.max_retries - 1:
                    # 尝试切换 API Key
                    switched = self._switch_key()
                    
                    # 计算等待时间(指数退避 + 随机抖动)
                    wait_time = self.base_retry_wait_seconds * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"{operation_name}: 达到最大重试次数 ({self.max_retries})，最终失败。")
                    raise RuntimeError(
                        f"在 {self.max_retries} 次尝试后{operation_name}失败。最后错误: {last_exception}"
                    ) from last_exception
                    
        raise RuntimeError(f"{operation_name}重试循环意外结束。") from last_exception
        
    def _is_retryable_error(self, error_str: str) -> bool:
        """
        判断错误是否可重试
        
        Args:
            error_str: 错误信息(小写)
            
        Returns:
            是否可重试
        """
        # 速率限制错误
        is_rate_limit = (
            "rate limit" in error_str
            or "429" in error_str
            or "resource_exhausted" in error_str
        )
        
        # 服务器错误
        is_server_error = (
            "server error" in error_str
            or "500" in error_str
            or "unavailable" in error_str
            or "503" in error_str
        )
        
        # 超时错误
        is_timeout = "timeout" in error_str or "timed out" in error_str
        
        return is_rate_limit or is_server_error or is_timeout
        
    @abstractmethod
    async def generate_response(self, *args, **kwargs) -> Any:
        """生成响应(子类必须实现)"""
        pass
        
    @abstractmethod
    async def get_embedding(self, text: str, **kwargs) -> Optional[List[float]]:
        """获取嵌入向量(子类必须实现)"""
        pass


class UsageTracker:
    """使用统计跟踪器"""
    
    def __init__(self, db_instance=None, model_name: str = "unknown"):
        """
        初始化使用统计跟踪器
        
        Args:
            db_instance: 数据库实例(需要有 llm_usage 集合)
            model_name: 模型名称
        """
        self.db = db_instance
        self.model_name = model_name
        self._init_database_indexes()
        
    def _init_database_indexes(self):
        """初始化数据库索引"""
        try:
            if self.db is not None and hasattr(self.db, "llm_usage"):
                collection = getattr(self.db, "llm_usage")
                if collection is not None and hasattr(collection, "create_index"):
                    collection.create_index([("timestamp", 1)])
                    collection.create_index([("model_name", 1)])
                    collection.create_index([("user_id", 1)])
                    collection.create_index([("request_type", 1)])
        except Exception as e:
            logger.error(f"创建数据库索引失败: {str(e)}")
            
    def record_usage(
        self,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        user_id: str = "system",
        request_type: str = "default",
        endpoint: str = "generateContent",
        status: str = "success",
        error_message: Optional[str] = None,
        api_key_hint: str = "N/A",
    ):
        """
        记录使用情况
        
        Args:
            prompt_tokens: 提示 token 数
            completion_tokens: 完成 token 数
            total_tokens: 总 token 数
            user_id: 用户 ID
            request_type: 请求类型
            endpoint: API 端点
            status: 状态(success/failure)
            error_message: 错误信息(如果失败)
            api_key_hint: API Key 提示(后4位)
        """
        try:
            if self.db is not None and hasattr(self.db, "llm_usage"):
                collection = getattr(self.db, "llm_usage")
                if collection is not None and hasattr(collection, "insert_one"):
                    usage_data = {
                        "model_name": self.model_name,
                        "user_id": user_id,
                        "request_type": request_type,
                        "endpoint": endpoint,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "cost": None,  # 可以后续添加成本计算
                        "status": status,
                        "error_message": error_message if status == "failure" else None,
                        "timestamp": datetime.now(),
                        "api_key_hint": api_key_hint,
                    }
                    collection.insert_one(usage_data)
                    
                    if status == "success":
                        logger.debug(
                            f"Token使用情况 - 模型: {self.model_name}, 用户: {user_id}, 类型: {request_type}, "
                            f"提示: {prompt_tokens}, 完成: {completion_tokens}, 总计: {total_tokens}"
                        )
                    else:
                        logger.error(
                            f"API 调用失败记录 - 模型: {self.model_name}, 用户: {user_id}, "
                            f"类型: {request_type}, 端点: {endpoint}, 错误: {error_message}"
                        )
        except Exception as e:
            logger.error(f"记录token使用情况失败: {str(e)}")
