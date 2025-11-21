"""
通用 LLM 请求模块
支持 Google GenAI SDK 和 OpenAI 格式 API
"""
import asyncio
import os
from datetime import datetime
from typing import Tuple, Union, List, Dict, Any, Optional
from PIL import Image
import aiohttp
from google import genai
from google.genai import types as google_genai_types
from pymongo.database import Database
import logging

from .base_client import BaseLLMClient, UsageTracker
from .utils import extract_reasoning

logger = logging.getLogger("llm_request")


class GeminiSDKResponse:
    """Gemini SDK 响应包装类(保持向后兼容)"""
    
    def __init__(
        self,
        content: str,
        reasoning: Optional[str] = None,
        web_search_queries: Optional[List[str]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        raw_response: Optional[Any] = None,
    ):
        self.content = content
        self.reasoning = reasoning if reasoning is not None else ""
        self.web_search_queries = (
            web_search_queries if web_search_queries is not None else []
        )
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.raw_response = raw_response

    def to_dict(self):
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "web_search_queries": self.web_search_queries,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }


class LLM_request(BaseLLMClient):
    """
    通用 LLM 请求客户端
    支持 Google GenAI SDK 和 OpenAI 格式 API
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        db_instance: Database,
        global_llm_params: Optional[Dict[str, Any]] = None,
        request_type: str = "default",
    ):
        """
        初始化 LLM 请求客户端
        
        Args:
            model_config: 模型配置字典
            db_instance: 数据库实例
            global_llm_params: 全局 LLM 参数
            request_type: 请求类型
        """
        self.model_config = model_config
        self.model_name = model_config.get("name")
        if not self.model_name:
            raise ValueError("模型配置中必须包含 'name' 字段。")
            
        self.db = db_instance
        self.request_type = request_type
        
        # 合并参数
        self.params = global_llm_params.copy() if global_llm_params else {}
        model_specific_params = {
            k: v
            for k, v in model_config.items()
            if k not in ["name", "key", "base_url", "api_keys", "pri_in", "pri_out"]
        }
        self.params.update(model_specific_params)
        
        # 解析 API Keys
        api_keys = self._parse_api_keys(model_config)
        
        # 初始化基类
        max_retries = self.params.get(
            "max_retries", int(os.getenv("LLM_MAX_RETRIES", "3"))
        )
        base_retry_wait = self.params.get("base_retry_wait_seconds", 5)
        
        super().__init__(
            api_keys=api_keys,
            model_name=self.model_name,
            max_retries=max_retries,
            base_retry_wait_seconds=base_retry_wait,
        )
        
        # 判断是否为 Google GenAI 模型
        self.base_url_actual: Optional[str] = model_config.get("base_url")
        self.is_google_genai_model = (
            not self.base_url_actual or "googleapis.com" in self.base_url_actual
        )
        
        # 初始化 Google GenAI 客户端
        self._clients: List[genai.Client] = []
        if self.is_google_genai_model:
            for key_val in self.api_keys:
                try:
                    client_instance = genai.Client(api_key=key_val)
                    self._clients.append(client_instance)
                except Exception as e:
                    logger.error(
                        f"为模型 {self.model_name} 使用 API Key ...{key_val[-4:]} "
                        f"初始化 Google GenAI Client 失败: {e}"
                    )
                    
            if not self._clients:
                raise ConnectionError(
                    f"模型 {self.model_name}: 所有提供的 API Key 都无法成功初始化 Google GenAI Client。"
                )
                
            logger.info(
                f"模型 ({self.model_name}) Google GenAI SDK: "
                f"已为 {len(self._clients)} 个API Key 初始化客户端。"
            )
        else:
            if not self.base_url_actual:
                raise ValueError(
                    f"模型 '{self.model_name}' 配置为非Google模型，但未提供 'base_url'。"
                )
            logger.info(
                f"模型 ({self.model_name}) 将通过 HTTP POST 请求非 Google API 端点: "
                f"{self.base_url_actual}"
            )
            
        # 使用统计跟踪器
        self.usage_tracker = UsageTracker(db_instance, self.model_name)
        
    def _parse_api_keys(self, model_config: Dict[str, Any]) -> List[str]:
        """解析 API Keys"""
        api_keys_actual: List[str] = []
        api_key_config = model_config.get("key")
        
        if isinstance(api_key_config, str) and api_key_config:
            if "," in api_key_config and not api_key_config.startswith("["):
                api_keys_actual.extend([k.strip() for k in api_key_config.split(",")])
            else:
                api_keys_actual.append(api_key_config)
        elif isinstance(api_key_config, list) and all(
            isinstance(k, str) for k in api_key_config
        ):
            api_keys_actual.extend(k for k in api_key_config if k)
            
        if not api_keys_actual:
            key_env_var_name = model_config.get("key_env_var_name")
            if key_env_var_name:
                env_key = os.getenv(key_env_var_name)
                if env_key:
                    api_keys_actual.append(env_key)
                    
            if not api_keys_actual:
                raise ValueError(
                    f"模型 '{self.model_name}' 未能加载任何有效的 API Key。"
                    f"请检查配置中的 'key' 字段或对应的环境变量。"
                )
                
        return api_keys_actual
        
    def _get_current_sync_client(self) -> genai.Client:
        """获取当前的 Google GenAI 同步客户端"""
        if not self._clients:
            raise RuntimeError(
                f"模型 ({self.model_name}) 没有可用的 Google GenAI 同步客户端。"
            )
        return self._clients[self.current_key_index]
        
    def _build_google_sdk_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[google_genai_types.GenerateContentConfig]:
        """构建 Google SDK 配置"""
        effective_params = self.params.copy()
        if overrides:
            effective_params.update(overrides)
            
        config_args = {}
        
        # 系统指令和工具
        if "system_instruction" in effective_params:
            config_args["system_instruction"] = effective_params["system_instruction"]
        if "tools" in effective_params:
            config_args["tools"] = effective_params["tools"]
        if "tool_config" in effective_params:
            config_args["tool_config"] = effective_params["tool_config"]
            
        # 安全设置
        if "safety_settings" in effective_params:
            config_args["safety_settings"] = effective_params["safety_settings"]
            
        # 响应格式
        if "response_mime_type" in effective_params:
            config_args["response_mime_type"] = effective_params["response_mime_type"]
        if "response_schema" in effective_params:
            config_args["response_schema"] = effective_params["response_schema"]
            
        # 生成参数
        if "candidate_count" in effective_params:
            config_args["candidate_count"] = int(effective_params["candidate_count"])
        if "stop_sequences" in effective_params and effective_params["stop_sequences"] is not None:
            ss = effective_params["stop_sequences"]
            config_args["stop_sequences"] = ss if isinstance(ss, list) else [str(ss)]
            
        # Token 限制
        if "max_output_tokens" in effective_params:
            config_args["max_output_tokens"] = int(effective_params["max_output_tokens"])
        elif "max_tokens" in effective_params:
            config_args["max_output_tokens"] = int(effective_params["max_tokens"])
            
        # 采样参数
        if "temperature" in effective_params:
            config_args["temperature"] = float(effective_params["temperature"])
        if "top_p" in effective_params:
            config_args["top_p"] = float(effective_params["top_p"])
        if "top_k" in effective_params:
            config_args["top_k"] = int(effective_params["top_k"])
        if "seed" in effective_params:
            config_args["seed"] = int(effective_params["seed"])
            
        if not config_args:
            return None
            
        try:
            return google_genai_types.GenerateContentConfig(**config_args)
        except Exception as e:
            logger.error(
                f"创建 Google SDK GenerateContentConfig 失败: {e}. 参数: {config_args}"
            )
            raise
            
    async def _execute_google_genai_sdk_request(
        self,
        contents: Union[str, List[Any]],
        config_overrides_dict: Optional[Dict[str, Any]] = None,
        is_embedding: bool = False,
        user_id: str = "system",
        request_type_override: Optional[str] = None,
    ) -> Union[List[float], GeminiSDKResponse]:
        """执行 Google GenAI SDK 请求"""
        request_type = (
            request_type_override
            if request_type_override
            else ("embedding" if is_embedding else self.request_type)
        )
        
        current_sdk_client = self._get_current_sync_client()
        
        async def _make_request():
            if is_embedding:
                # 嵌入请求
                if not isinstance(contents, str):
                    raise ValueError("Embedding input must be a string")
                    
                task_type_for_embed = (config_overrides_dict or {}).get(
                    "task_type", "RETRIEVAL_DOCUMENT"
                )
                model_name_for_embed = (config_overrides_dict or {}).get(
                    "model", self.model_name
                )
                if model_name_for_embed.startswith("models/"):
                    model_name_for_embed = model_name_for_embed.split("/")[-1]
                    
                logger.info(
                    f"Google SDK Embedding Request to {model_name_for_embed}: "
                    f"Text='{str(contents)[:200]}...', TaskType='{task_type_for_embed}'"
                )
                
                response_object = await asyncio.to_thread(
                    current_sdk_client.models.embed_content,
                    model=model_name_for_embed,
                    contents=contents,
                    task_type=task_type_for_embed,
                )
                
                embedding_vector_result = getattr(response_object, "embedding", None)
                if embedding_vector_result:
                    self.usage_tracker.record_usage(
                        None, None, None, user_id, request_type, "embedContent",
                        api_key_hint=f"...{self._get_current_api_key()[-4:]}",
                    )
                    return embedding_vector_result
                else:
                    raise ValueError(
                        f"Google GenAI Embedding response did not contain an embedding vector. "
                        f"Response: {response_object}"
                    )
            else:
                # 生成内容请求
                sdk_gen_config = self._build_google_sdk_config(config_overrides_dict)
                model_name_for_gen = (config_overrides_dict or {}).get(
                    "model", self.model_name
                )
                if model_name_for_gen.startswith("models/"):
                    model_name_for_gen = model_name_for_gen.split("/")[-1]
                    
                # 准备内容
                actual_contents_for_api = []
                if isinstance(contents, str):
                    actual_contents_for_api.append(contents)
                elif isinstance(contents, list):
                    processed_parts = []
                    for item_part in contents:
                        if isinstance(item_part, str):
                            processed_parts.append(item_part)
                        elif isinstance(item_part, Image.Image):
                            processed_parts.append(item_part)
                        elif isinstance(item_part, google_genai_types.Part):
                            processed_parts.append(item_part)
                        elif isinstance(item_part, dict) and "text" in item_part:
                            processed_parts.append(item_part["text"])
                        elif isinstance(item_part, dict) and "inline_data" in item_part:
                            processed_parts.append(
                                google_genai_types.Part(inline_data=item_part["inline_data"])
                            )
                        else:
                            logger.warning(
                                f"Unsupported item type in contents list: {type(item_part)}"
                            )
                    actual_contents_for_api = processed_parts
                else:
                    actual_contents_for_api = contents
                    
                # 检查是否流式请求
                is_stream_request = self.params.get("stream", False) or (
                    config_overrides_dict or {}
                ).get("stream", False)
                
                text_content_output = ""
                usage_metadata_from_sdk = None
                raw_response_data = None
                
                if is_stream_request:
                    # 流式请求
                    response_stream_iter = await asyncio.to_thread(
                        current_sdk_client.models.generate_content_stream,
                        model=model_name_for_gen,
                        contents=actual_contents_for_api,
                        config=sdk_gen_config,
                    )
                    
                    accumulated_text_chunks = []
                    chunk_list_for_raw = []
                    
                    for stream_chunk in response_stream_iter:
                        chunk_list_for_raw.append(stream_chunk)
                        accumulated_text_chunks.append(stream_chunk.text)
                        if hasattr(stream_chunk, "usage_metadata") and stream_chunk.usage_metadata:
                            usage_metadata_from_sdk = stream_chunk.usage_metadata
                            
                    text_content_output = "".join(accumulated_text_chunks)
                    raw_response_data = chunk_list_for_raw
                else:
                    # 非流式请求
                    sdk_response_obj = await asyncio.to_thread(
                        current_sdk_client.models.generate_content,
                        model=model_name_for_gen,
                        contents=actual_contents_for_api,
                        config=sdk_gen_config,
                    )
                    
                    raw_response_data = sdk_response_obj
                    text_content_output = sdk_response_obj.text
                    usage_metadata_from_sdk = getattr(sdk_response_obj, "usage_metadata", None)
                    
                # 提取使用统计
                prompt_tokens, completion_tokens, total_tokens = None, None, None
                if usage_metadata_from_sdk:
                    prompt_tokens = getattr(usage_metadata_from_sdk, "prompt_token_count", None)
                    cand_tokens_sdk = getattr(usage_metadata_from_sdk, "candidates_token_count", None)
                    
                    if isinstance(cand_tokens_sdk, list) and cand_tokens_sdk:
                        completion_tokens = sum(cand_tokens_sdk)
                    elif isinstance(cand_tokens_sdk, int):
                        completion_tokens = cand_tokens_sdk
                        
                    total_tokens = getattr(usage_metadata_from_sdk, "total_token_count", None)
                    
                # 记录使用情况
                self.usage_tracker.record_usage(
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    user_id,
                    request_type,
                    "generateContent",
                    api_key_hint=f"...{self._get_current_api_key()[-4:]}",
                )
                
                # 提取思考过程
                final_cleaned_content, final_reasoning = extract_reasoning(text_content_output)
                
                # 提取 web 搜索查询
                web_queries_found = None
                final_response_object_for_grounding = None
                
                if is_stream_request:
                    if (
                        chunk_list_for_raw
                        and hasattr(chunk_list_for_raw[-1], "candidates")
                        and chunk_list_for_raw[-1].candidates
                    ):
                        final_response_object_for_grounding = chunk_list_for_raw[-1]
                else:
                    final_response_object_for_grounding = sdk_response_obj
                    
                if (
                    final_response_object_for_grounding
                    and hasattr(final_response_object_for_grounding, "candidates")
                    and final_response_object_for_grounding.candidates
                ):
                    candidate_obj = final_response_object_for_grounding.candidates[0]
                    grounding_data = getattr(candidate_obj, "grounding_metadata", None)
                    
                    if (
                        grounding_data
                        and hasattr(grounding_data, "web_search_queries")
                        and grounding_data.web_search_queries
                    ):
                        web_queries_found = list(grounding_data.web_search_queries)
                        logger.info(f"提取到 Web 搜索查询 (SDK): {web_queries_found}")
                        
                return GeminiSDKResponse(
                    content=final_cleaned_content,
                    reasoning=final_reasoning,
                    web_search_queries=web_queries_found,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    raw_response=raw_response_data,
                )
                
        try:
            return await self._execute_with_retry(
                _make_request,
                operation_name=f"Google SDK ({'Embedding' if is_embedding else 'Generate'}) - {self.model_name}",
            )
        except Exception as e:
            logger.error(f"Google SDK 请求失败: {e}")
            
            # 记录失败
            self.usage_tracker.record_usage(
                None,
                None,
                None,
                user_id,
                request_type,
                "embedContent" if is_embedding else "generateContent",
                "failure",
                str(e),
                f"...{self._get_current_api_key()[-4:]}",
            )
            raise
            
    async def _execute_http_post_request(
        self,
        endpoint_suffix: str,
        payload: Dict[str, Any],
        user_id: str = "system",
        request_type_override: Optional[str] = None,
    ) -> Any:
        """执行 HTTP POST 请求(用于非 Google API)"""
        request_type = request_type_override or self.request_type
        
        if not self.base_url_actual:
            raise ValueError("非Google模型请求需要 base_url。")
            
        api_url = f"{self.base_url_actual.rstrip('/')}/{endpoint_suffix.lstrip('/')}"
        
        async def _make_request():
            current_api_key = self._get_current_api_key()
            headers = {"Content-Type": "application/json"}
            
            if (
                current_api_key
                and not current_api_key.startswith("dummy_")
                and "no_key" not in current_api_key.lower()
            ):
                headers["Authorization"] = f"Bearer {current_api_key}"
                
            timeout_seconds = self.params.get("http_timeout_seconds", 30)
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.post(api_url, json=payload, timeout=timeout) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        try:
            return await self._execute_with_retry(
                _make_request,
                operation_name=f"HTTP POST - {self.model_name}",
            )
        except Exception as e:
            logger.error(f"HTTP POST 请求失败: {e}")
            
            # 记录失败
            self.usage_tracker.record_usage(
                None,
                None,
                None,
                user_id,
                request_type,
                endpoint_suffix,
                "failure",
                str(e),
                f"...{self._get_current_api_key()[-4:]}",
            )
            raise
            
    async def generate_response(
        self,
        prompt: Union[str, List[Any]],
        user_id: str = "system",
        request_type: Optional[str] = None,
        **kwargs,
    ) -> GeminiSDKResponse:
        """
        生成响应(统一接口)
        
        Args:
            prompt: 提示文本或内容列表
            user_id: 用户 ID
            request_type: 请求类型
            **kwargs: 其他参数
            
        Returns:
            GeminiSDKResponse 对象
        """
        effective_request_type = request_type or self.request_type or "generate"
        
        if self.is_google_genai_model:
            # 使用 Google GenAI SDK
            sdk_response_union = await self._execute_google_genai_sdk_request(
                contents=prompt,
                config_overrides_dict=kwargs,
                is_embedding=False,
                user_id=user_id,
                request_type_override=effective_request_type,
            )
            
            if isinstance(sdk_response_union, GeminiSDKResponse):
                return sdk_response_union
            else:
                error_msg = "generate_response for Google model received unexpected type."
                logger.error(error_msg + f" Got: {type(sdk_response_union)}")
                return GeminiSDKResponse(
                    content=f"Internal error: {error_msg}", reasoning=error_msg
                )
        else:
            # 使用 HTTP POST (OpenAI 格式)
            payload_messages = []
            if isinstance(prompt, str):
                payload_messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                for item in prompt:
                    if isinstance(item, dict) and "role" in item and "content" in item:
                        payload_messages.append(item)
                    elif isinstance(item, str):
                        payload_messages.append({"role": "user", "content": item})
                    else:
                        logger.warning(
                            f"Unsupported item type in prompt list: {type(item)}"
                        )
                        
            http_payload = {
                "model": self.model_name,
                "messages": payload_messages,
                "temperature": kwargs.get("temperature", self.params.get("temperature")),
                "max_tokens": kwargs.get(
                    "max_output_tokens",
                    self.params.get("max_output_tokens", self.params.get("max_tokens")),
                ),
                "stream": kwargs.get("stream", self.params.get("stream", False)),
            }
            http_payload = {k: v for k, v in http_payload.items() if v is not None}
            
            if "http_payload" in kwargs and isinstance(kwargs["http_payload"], dict):
                http_payload = kwargs["http_payload"]
                
            try:
                raw_json_response = await self._execute_http_post_request(
                    endpoint_suffix="chat/completions",
                    payload=http_payload,
                    user_id=user_id,
                    request_type_override=effective_request_type,
                )
                
                text_content = ""
                choices = raw_json_response.get("choices", [])
                if choices and isinstance(choices, list) and choices[0]:
                    message_obj = choices[0].get("message", {})
                    text_content = message_obj.get("content", "")
                    
                usage_data = raw_json_response.get("usage", {})
                p_tokens = usage_data.get("prompt_tokens")
                c_tokens = usage_data.get("completion_tokens")
                t_tokens = usage_data.get("total_tokens")
                
                self.usage_tracker.record_usage(
                    p_tokens,
                    c_tokens,
                    t_tokens,
                    user_id,
                    effective_request_type,
                    "chat/completions",
                    api_key_hint=f"...{self._get_current_api_key()[-4:]}",
                )
                
                content_cleaned, reasoning_text = extract_reasoning(text_content)
                
                return GeminiSDKResponse(
                    content=content_cleaned,
                    reasoning=reasoning_text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    raw_response=raw_json_response,
                )
            except Exception as e:
                logger.error(f"非Google模型 ({self.model_name}) generate_response 失败: {e}")
                return GeminiSDKResponse(
                    content=f"API调用失败: {e}",
                    reasoning=str(e),
                    raw_response={"error": str(e)},
                )
                
    # 保持向后兼容的别名
    async def generate_response_async(self, *args, **kwargs) -> GeminiSDKResponse:
        """向后兼容的别名"""
        return await self.generate_response(*args, **kwargs)
        
    async def get_embedding(
        self,
        text: str,
        user_id: str = "system",
        request_type: Optional[str] = None,
        **kwargs,
    ) -> Optional[List[float]]:
        """
        获取嵌入向量(统一接口)
        
        Args:
            text: 输入文本
            user_id: 用户 ID
            request_type: 请求类型
            **kwargs: 其他参数
            
        Returns:
            嵌入向量列表,失败返回 None
        """
        if not text or not text.strip():
            logger.warning("Embedding requested for empty text")
            return None
            
        effective_request_type = request_type or self.request_type or "embedding"
        
        if self.is_google_genai_model:
            # 使用 Google GenAI SDK
            embedding_vector_union = await self._execute_google_genai_sdk_request(
                contents=text,
                config_overrides_dict=kwargs,
                is_embedding=True,
                user_id=user_id,
                request_type_override=effective_request_type,
            )
            
            if isinstance(embedding_vector_union, list):
                return embedding_vector_union
            else:
                logger.error(
                    f"get_embedding for Google model received unexpected type: "
                    f"{type(embedding_vector_union)}"
                )
                return None
        else:
            # 使用 HTTP POST (OpenAI 格式)
            http_payload = {
                "model": self.model_name,
                "input": text,
            }
            
            if "encoding_format" in self.params:
                http_payload["encoding_format"] = self.params["encoding_format"]
            if "encoding_format" in kwargs:
                http_payload["encoding_format"] = kwargs["encoding_format"]
                
            if "http_payload" in kwargs and isinstance(kwargs["http_payload"], dict):
                http_payload = kwargs["http_payload"]
                
            try:
                raw_json_response = await self._execute_http_post_request(
                    endpoint_suffix="embeddings",
                    payload=http_payload,
                    user_id=user_id,
                    request_type_override=effective_request_type,
                )
                
                embedding_vector = None
                if isinstance(raw_json_response, dict):
                    if (
                        "data" in raw_json_response
                        and isinstance(raw_json_response["data"], list)
                        and raw_json_response["data"]
                    ):
                        first_item = raw_json_response["data"][0]
                        if isinstance(first_item, dict) and "embedding" in first_item:
                            embedding_vector = first_item["embedding"]
                    elif "embedding" in raw_json_response:
                        embedding_vector = raw_json_response["embedding"]
                        
                if not isinstance(embedding_vector, list):
                    logger.error(
                        f"未能从非Google模型 ({self.model_name}) 响应中提取有效的嵌入向量。"
                        f"响应: {str(raw_json_response)[:200]}"
                    )
                    return None
                    
                usage_data = (
                    raw_json_response.get("usage", {})
                    if isinstance(raw_json_response, dict)
                    else {}
                )
                p_tokens = usage_data.get("prompt_tokens")
                t_tokens = usage_data.get("total_tokens")
                
                self.usage_tracker.record_usage(
                    p_tokens, 0, t_tokens, user_id, effective_request_type, "embeddings",
                    api_key_hint=f"...{self._get_current_api_key()[-4:]}",
                )
                
                return embedding_vector
            except Exception as e:
                logger.error(f"非Google模型 ({self.model_name}) get_embedding 失败: {e}")
                return None
                
    # 保持向后兼容的别名
    async def get_embedding_async(self, *args, **kwargs) -> Optional[List[float]]:
        """向后兼容的别名"""
        return await self.get_embedding(*args, **kwargs)
