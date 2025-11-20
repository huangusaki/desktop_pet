import asyncio
import re
from datetime import datetime
from typing import Tuple, Union, List, Dict, Any, Optional
import base64
from PIL import Image
import io
import aiohttp
import os
from google import genai
from google.genai import types as google_genai_types
from pymongo.database import Database
import logging
import random

ResourceExhaustedException = None
InvalidArgumentException = None
PermissionDeniedException = None
InternalServerErrorException = None
ServiceUnavailableException = None
logger = logging.getLogger("llm_request")


def compress_base64_image_by_scale(
    base64_data: str, target_size: int = 0.8 * 1024 * 1024
) -> str:
    try:
        image_data = base64.b64decode(base64_data)
        if len(image_data) <= target_size:
            return base64_data
        img = Image.open(io.BytesIO(image_data))
        original_width, original_height = img.size
        current_quality = 85
        scale = (
            (target_size / len(image_data)) ** 0.5
            if len(image_data) > target_size
            else 1.0
        )
        scale = min(scale, 0.95)
        compressed_data_to_return_if_loop_fails = image_data
        for attempt in range(5):
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            if new_width <= 0 or new_height <= 0:
                logger.warning(
                    f"计算出的新尺寸无效 ({new_width}x{new_height})，源尺寸 {original_width}x{original_height}, 缩放比例 {scale:.2f}。将使用原始图片。"
                )
                return base64_data
            output_buffer = io.BytesIO()
            img_format_to_save = (
                img.format
                if img.format and img.format.upper() in ["PNG", "WEBP", "GIF"]
                else "JPEG"
            )
            temp_img_copy = img.copy()
            if (
                img_format_to_save == "GIF"
                and getattr(temp_img_copy, "is_animated", False)
                and temp_img_copy.n_frames > 1
            ):
                frames = []
                durations = []
                loop = temp_img_copy.info.get("loop", 0)
                try:
                    for frame_idx in range(temp_img_copy.n_frames):
                        temp_img_copy.seek(frame_idx)
                        current_duration = temp_img_copy.info.get("duration", 100)
                        durations.append(current_duration)
                        frame_rgba = temp_img_copy.convert("RGBA")
                        resized_frame = frame_rgba.resize(
                            (new_width, new_height), Image.Resampling.LANCZOS
                        )
                        frames.append(resized_frame)
                except EOFError:
                    pass
                if frames:
                    frames[0].save(
                        output_buffer,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        optimize=True,
                        duration=durations,
                        loop=loop,
                        disposal=2,
                    )
                else:
                    img_format_to_save = "JPEG"
            if img_format_to_save != "GIF":
                resized_img = temp_img_copy.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
                if img_format_to_save == "PNG":
                    if resized_img.mode == "RGBA" or "A" in resized_img.mode:
                        resized_img.save(output_buffer, format="PNG", optimize=True)
                    else:
                        resized_img.convert("RGB").save(
                            output_buffer,
                            format="JPEG",
                            quality=current_quality,
                            optimize=True,
                        )
                        img_format_to_save = "JPEG"
                elif img_format_to_save == "WEBP":
                    resized_img.save(
                        output_buffer,
                        format="WEBP",
                        quality=current_quality,
                        lossless=False,
                    )
                else:
                    img_format_to_save = "JPEG"
                    if resized_img.mode in ("RGBA", "LA", "P"):
                        resized_img = resized_img.convert("RGB")
                    resized_img.save(
                        output_buffer,
                        format="JPEG",
                        quality=current_quality,
                        optimize=True,
                    )
            compressed_data_loop = output_buffer.getvalue()
            compressed_data_to_return_if_loop_fails = compressed_data_loop
            if len(compressed_data_loop) <= target_size:
                final_format_check = Image.open(io.BytesIO(compressed_data_loop)).format
                logger.info(
                    f"压缩图片 (尝试 {attempt + 1}): {original_width}x{original_height} ({img.format or 'N/A'} -> {final_format_check or img_format_to_save}). "
                    f"大小: {len(image_data) / 1024:.1f}KB -> {len(compressed_data_loop) / 1024:.1f}KB (目标: {target_size / 1024:.1f}KB)"
                )
                return base64.b64encode(compressed_data_loop).decode("utf-8")
            if img_format_to_save in ["JPEG", "WEBP"] and current_quality > 60:
                current_quality -= 10
            else:
                scale *= 0.85
            logger.info(
                f"压缩后仍然过大 (尝试 {attempt + 1}, {len(compressed_data_loop) / 1024:.1f}KB). 下次 scale={scale:.2f}, quality={current_quality}"
            )
        logger.warning(
            f"多次压缩后大小 {len(compressed_data_to_return_if_loop_fails) / 1024:.1f}KB 仍大于目标 {target_size / 1024:.1f}KB. 返回当前最佳压缩结果。"
        )
        return base64.b64encode(compressed_data_to_return_if_loop_fails).decode("utf-8")
    except Exception as e:
        logger.error(f"压缩图片失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return base64_data


class GeminiSDKResponse:
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


class LLM_request:
    def __init__(
        self,
        model_config: Dict[str, Any],
        db_instance: Database,
        global_llm_params: Optional[Dict[str, Any]] = None,
        request_type: str = "default",
    ):
        self.model_config = model_config
        self.model_name = model_config.get("name")
        if not self.model_name:
            raise ValueError("模型配置中必须包含 'name' 字段。")
        self.db = db_instance
        self.request_type = request_type
        self.params = global_llm_params.copy() if global_llm_params else {}
        model_specific_params = {
            k: v
            for k, v in model_config.items()
            if k not in ["name", "key", "base_url", "api_keys", "pri_in", "pri_out"]
        }
        self.params.update(model_specific_params)
        self.api_keys_actual: List[str] = []
        api_key_config = model_config.get("key")
        if isinstance(api_key_config, str) and api_key_config:
            if "," in api_key_config and not api_key_config.startswith("["):
                self.api_keys_actual.extend(
                    [k.strip() for k in api_key_config.split(",")]
                )
            else:
                self.api_keys_actual.append(api_key_config)
        elif isinstance(api_key_config, list) and all(
            isinstance(k, str) for k in api_key_config
        ):
            self.api_keys_actual.extend(k for k in api_key_config if k)
        if not self.api_keys_actual:
            key_env_var_name = model_config.get("key_env_var_name")
            if key_env_var_name:
                env_key = os.getenv(key_env_var_name)
                if env_key:
                    self.api_keys_actual.append(env_key)
            if not self.api_keys_actual:
                raise ValueError(
                    f"模型 '{self.model_name}' 未能加载任何有效的 API Key。请检查配置中的 'key' 字段或对应的环境变量。"
                )
        self.base_url_actual: Optional[str] = model_config.get("base_url")
        self.current_key_index = 0
        self._clients: List[genai.Client] = []
        self.is_google_genai_model = (
            not self.base_url_actual or "googleapis.com" in self.base_url_actual
        )
        if self.is_google_genai_model:
            for key_val in self.api_keys_actual:
                try:
                    client_instance = genai.Client(api_key=key_val)
                    self._clients.append(client_instance)
                except Exception as e:
                    logger.error(
                        f"为模型 {self.model_name} 使用 API Key ...{key_val[-4:]} 初始化 Google GenAI Client 失败: {e}"
                    )
            if not self._clients:
                raise ConnectionError(
                    f"模型 {self.model_name}: 所有提供的 API Key 都无法成功初始化 Google GenAI Client。"
                )
            logger.info(
                f"模型 ({self.model_name}) Google GenAI SDK: 已为 {len(self._clients)} 个API Key 初始化客户端。"
            )
        else:
            if not self.base_url_actual:
                raise ValueError(
                    f"模型 '{self.model_name}' 配置为非Google模型，但未提供 'base_url'。"
                )
            logger.info(
                f"模型 ({self.model_name}) 将通过 HTTP POST 请求非 Google API 端点: {self.base_url_actual}"
            )
            if not self.api_keys_actual:
                raise ValueError(f"模型 '{self.model_name}' (HTTP) 未提供 API Key。")
        self._init_database_indexes()

    def _init_database_indexes(self):
        try:
            if self.db is not None and hasattr(self.db, "llm_usage"):
                collection = getattr(self.db, "llm_usage")
                if collection is not None and hasattr(collection, "create_index"):
                    collection.create_index([("timestamp", 1)])
                    collection.create_index([("model_name", 1)])
                    collection.create_index([("user_id", 1)])
                    collection.create_index([("request_type", 1)])
                else:
                    logger.warning(
                        "Database object does not appear to be a valid PyMongo collection for 'llm_usage'. Skipping index creation."
                    )
            elif self.db is None:
                logger.warning(
                    "Database instance (self.db) is None. Skipping index creation."
                )
            else:
                logger.warning(
                    "Database object does not have 'llm_usage' attribute. Skipping index creation."
                )
        except Exception as e:
            logger.error(f"创建数据库索引失败: {str(e)}")

    def _record_usage(
        self,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        total_tokens: Optional[int],
        user_id: str = "system",
        request_type: Optional[str] = None,
        endpoint: str = "generateContent",
        status: str = "success",
        error_message: Optional[str] = None,
    ):
        if request_type is None:
            request_type = self.request_type
        cost = None
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
                        "cost": cost,
                        "status": status,
                        "error_message": error_message if status == "failure" else None,
                        "timestamp": datetime.now(),
                        "api_key_hint": (
                            f"...{self.api_keys_actual[self.current_key_index][-4:]}"
                            if self.api_keys_actual
                            and self.current_key_index < len(self.api_keys_actual)
                            else "N/A"
                        ),
                    }
                    collection.insert_one(usage_data)
                    if status == "success":
                        logger.debug(
                            f"Token使用情况 - 模型: {self.model_name}, 用户: {user_id}, 类型: {request_type}, "
                            f"提示: {prompt_tokens}, 完成: {completion_tokens}, 总计: {total_tokens}"
                        )
                    else:
                        logger.error(
                            f"API 调用失败记录 - 模型: {self.model_name}, 用户: {user_id}, 类型: {request_type}, 端点: {endpoint}, 错误: {error_message}"
                        )
                else:
                    logger.warning(
                        "Database 'llm_usage' object does not support 'insert_one'. Usage not recorded."
                    )
        except Exception as e:
            logger.error(f"记录token使用情况失败: {str(e)}")

    def _get_current_sync_client(self) -> genai.Client:
        if not self._clients:
            raise RuntimeError(
                f"模型 ({self.model_name}) 没有可用的 Google GenAI 同步客户端。"
            )
        return self._clients[self.current_key_index]

    def _get_current_api_key_for_http(self) -> str:
        if not self.api_keys_actual:
            raise RuntimeError(f"模型 ({self.model_name}) 没有可用的 API Keys。")
        return self.api_keys_actual[self.current_key_index]

    def _switch_key(self) -> bool:
        if len(self.api_keys_actual) <= 1:
            return False
        current_key_before_switch = self.api_keys_actual[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(
            self.api_keys_actual
        )
        new_key = self.api_keys_actual[self.current_key_index]
        logger.warning(
            f"模型 ({self.model_name}) 遇到可重试错误或速率限制，从 Key ...{current_key_before_switch[-4:]} "
            f"切换到 Key #{self.current_key_index + 1}/{len(self.api_keys_actual)} (...{new_key[-4:]})。"
        )
        return True

    def _build_google_sdk_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[google_genai_types.GenerateContentConfig]:
        effective_params = self.params.copy()
        if overrides:
            effective_params.update(overrides)
        config_args = {}
        if "system_instruction" in effective_params:
            config_args["system_instruction"] = effective_params["system_instruction"]
        if "tools" in effective_params:
            config_args["tools"] = effective_params["tools"]
        if "tool_config" in effective_params:
            config_args["tool_config"] = effective_params["tool_config"]
        if "safety_settings" in effective_params:
            config_args["safety_settings"] = effective_params["safety_settings"]
        if "response_mime_type" in effective_params:
            config_args["response_mime_type"] = effective_params["response_mime_type"]
        if "response_schema" in effective_params:
            config_args["response_schema"] = effective_params["response_schema"]
        if "candidate_count" in effective_params:
            config_args["candidate_count"] = int(effective_params["candidate_count"])
        if (
            "stop_sequences" in effective_params
            and effective_params["stop_sequences"] is not None
        ):
            ss = effective_params["stop_sequences"]
            config_args["stop_sequences"] = ss if isinstance(ss, list) else [str(ss)]
        if "max_output_tokens" in effective_params:
            config_args["max_output_tokens"] = int(
                effective_params["max_output_tokens"]
            )
        elif "max_tokens" in effective_params:
            config_args["max_output_tokens"] = int(effective_params["max_tokens"])
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

    @staticmethod
    def _extract_reasoning_sdk(content: str) -> Tuple[str, str]:
        if not isinstance(content, str):
            return "", ""
        match = re.search(
            r"<(?:think|thought)>(.*?)</(?:think|thought)>",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            reasoning = match.group(1).strip()
            cleaned_content = content.replace(match.group(0), "", 1).strip()
            return cleaned_content, reasoning
        return content.strip(), ""

    async def _execute_google_genai_sdk_request(
        self,
        contents: Union[str, List[Any]],
        config_overrides_dict: Optional[Dict[str, Any]] = None,
        is_embedding: bool = False,
        user_id: str = "system",
        request_type_override: Optional[str] = None,
    ) -> Union[List[float], GeminiSDKResponse]:
        request_type = (
            request_type_override
            if request_type_override
            else ("embedding" if is_embedding else self.request_type)
        )
        last_exception = None
        max_retries = self.params.get(
            "max_retries", int(os.getenv("LLM_MAX_RETRIES", "3"))
        )
        current_sdk_client = self._get_current_sync_client()
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Attempt {attempt + 1}/{max_retries} - Google SDK ({self.model_name}) using Key #{self.current_key_index + 1}"
                )
                if is_embedding:
                    if not isinstance(contents, str):
                        raise ValueError(
                            "Embedding input for Google GenAI SDK must be a string for this wrapper."
                        )
                    task_type_for_embed = (config_overrides_dict or {}).get(
                        "task_type", "RETRIEVAL_DOCUMENT"
                    )
                    model_name_for_embed = (config_overrides_dict or {}).get(
                        "model", self.model_name
                    )
                    if model_name_for_embed.startswith("models/"):
                        model_name_for_embed = model_name_for_embed.split("/")[-1]
                    logger.info(
                        f"Google SDK Embedding Request to {model_name_for_embed}: Text='{str(contents)[:200]}...', TaskType='{task_type_for_embed}'"
                    )
                    response_object = await asyncio.to_thread(
                        current_sdk_client.models.embed_content,
                        model=model_name_for_embed,
                        contents=contents,
                        task_type=task_type_for_embed,
                    )
                    embedding_vector_result = getattr(
                        response_object, "embedding", None
                    )
                    if embedding_vector_result:
                        self._record_usage(
                            None, None, None, user_id, request_type, "embedContent"
                        )
                        return embedding_vector_result
                    else:
                        raise ValueError(
                            f"Google GenAI Embedding response did not contain an embedding vector. Response: {response_object}"
                        )
                else:
                    sdk_gen_config = self._build_google_sdk_config(
                        config_overrides_dict
                    )
                    model_name_for_gen = (config_overrides_dict or {}).get(
                        "model", self.model_name
                    )
                    if model_name_for_gen.startswith("models/"):
                        model_name_for_gen = model_name_for_gen.split("/")[-1]
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
                            elif (
                                isinstance(item_part, dict)
                                and "inline_data" in item_part
                            ):
                                processed_parts.append(
                                    google_genai_types.Part(
                                        inline_data=item_part["inline_data"]
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Unsupported item type in contents list for generation: {type(item_part)}"
                                )
                        actual_contents_for_api = processed_parts
                    else:
                        actual_contents_for_api = contents
                    is_stream_request = self.params.get("stream", False) or (
                        config_overrides_dict or {}
                    ).get("stream", False)
                    text_content_output = ""
                    usage_metadata_from_sdk = None
                    raw_response_data = None
                    if is_stream_request:
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
                            if (
                                hasattr(stream_chunk, "usage_metadata")
                                and stream_chunk.usage_metadata
                            ):
                                usage_metadata_from_sdk = stream_chunk.usage_metadata
                        text_content_output = "".join(accumulated_text_chunks)
                        raw_response_data = chunk_list_for_raw
                    else:
                        sdk_response_obj = await asyncio.to_thread(
                            current_sdk_client.models.generate_content,
                            model=model_name_for_gen,
                            contents=actual_contents_for_api,
                            config=sdk_gen_config,
                        )
                        raw_response_data = sdk_response_obj
                        text_content_output = sdk_response_obj.text
                        usage_metadata_from_sdk = getattr(
                            sdk_response_obj, "usage_metadata", None
                        )
                    prompt_tokens, completion_tokens, total_tokens = None, None, None
                    if usage_metadata_from_sdk:
                        prompt_tokens = getattr(
                            usage_metadata_from_sdk, "prompt_token_count", None
                        )
                        cand_tokens_sdk = getattr(
                            usage_metadata_from_sdk, "candidates_token_count", None
                        )
                        if isinstance(cand_tokens_sdk, list) and cand_tokens_sdk:
                            completion_tokens = sum(cand_tokens_sdk)
                        elif isinstance(cand_tokens_sdk, int):
                            completion_tokens = cand_tokens_sdk
                        total_tokens = getattr(
                            usage_metadata_from_sdk, "total_token_count", None
                        )
                    self._record_usage(
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        user_id,
                        request_type,
                        "generateContent",
                    )
                    final_cleaned_content, final_reasoning = (
                        self._extract_reasoning_sdk(text_content_output)
                    )
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
                        candidate_obj = final_response_object_for_grounding.candidates[
                            0
                        ]
                        grounding_data = getattr(
                            candidate_obj, "grounding_metadata", None
                        )
                        if (
                            grounding_data
                            and hasattr(grounding_data, "web_search_queries")
                            and grounding_data.web_search_queries
                        ):
                            web_queries_found = list(grounding_data.web_search_queries)
                            logger.info(
                                f"提取到 Web 搜索查询 (SDK): {web_queries_found}"
                            )
                    return GeminiSDKResponse(
                        content=final_cleaned_content,
                        reasoning=final_reasoning,
                        web_search_queries=web_queries_found,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        raw_response=raw_response_data,
                    )
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                import traceback

                logger.warning(
                    f"Google SDK ({self.model_name}) 失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__} - {e}\n{traceback.format_exc()}"
                )
                is_rate_limit_typed = ResourceExhaustedException and isinstance(
                    e, ResourceExhaustedException
                )
                is_invalid_arg_typed = InvalidArgumentException and isinstance(
                    e, InvalidArgumentException
                )
                is_auth_error_typed = PermissionDeniedException and isinstance(
                    e, PermissionDeniedException
                )
                is_server_error_typed = (
                    InternalServerErrorException
                    and isinstance(e, InternalServerErrorException)
                ) or (
                    ServiceUnavailableException
                    and isinstance(e, ServiceUnavailableException)
                )
                is_rate_limit_str = (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "resource_exhausted" in error_str
                )
                is_server_error_str = (
                    "server error" in error_str
                    or "500" in error_str
                    or "unavailable" in error_str
                    or "503" in error_str
                )
                is_auth_error_str = (
                    "permission_denied" in error_str
                    or "invalid api key" in error_str
                    or "401" in error_str
                    or "403" in error_str
                    or "unauthenticated" in error_str
                )
                is_invalid_arg_str = (
                    "invalid argument" in error_str
                    or "400" in error_str
                    or "could not parse" in error_str
                    or "bad request" in error_str
                )
                is_rate_limit = is_rate_limit_typed or is_rate_limit_str
                is_server_error = is_server_error_typed or is_server_error_str
                is_auth_error = is_auth_error_typed or is_auth_error_str
                is_invalid_arg = is_invalid_arg_typed or is_invalid_arg_str
                if is_auth_error or is_invalid_arg:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        "embedContent" if is_embedding else "generateContent",
                        "failure",
                        str(e),
                    )
                    raise RuntimeError(f"Google SDK 请求失败，不可重试: {e}") from e
                if attempt < max_retries - 1:
                    if is_rate_limit or is_server_error:
                        switched_key_successfully = self._switch_key()
                        if switched_key_successfully:
                            current_sdk_client = self._get_current_sync_client()
                    base_retry_delay = self.params.get("base_retry_wait_seconds", 5)
                    current_wait_time = base_retry_delay * (
                        2**attempt
                    ) + random.uniform(0, 1)
                    logger.info(f"等待 {current_wait_time:.2f} 秒后重试...")
                    await asyncio.sleep(current_wait_time)
                else:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        "embedContent" if is_embedding else "generateContent",
                        "failure",
                        f"Max retries ({max_retries}) reached. Last error: {e}",
                    )
                    raise RuntimeError(
                        f"Google SDK 请求在 {max_retries} 次尝试后失败。最后错误: {e}"
                    ) from last_exception
        self._record_usage(
            None,
            None,
            None,
            user_id,
            request_type,
            "embedContent" if is_embedding else "generateContent",
            "failure",
            "Max retries reached without throwing (logical error in retry loop).",
        )
        raise RuntimeError(
            f"Google SDK 请求在 {max_retries} 次尝试后失败 (代码逻辑问题，循环结束但未抛出最终异常)。"
        )

    async def _execute_http_post_request(
        self,
        endpoint_suffix: str,
        payload: Dict[str, Any],
        user_id: str = "system",
        request_type_override: Optional[str] = None,
    ) -> Any:
        request_type = request_type_override or self.request_type
        if not self.base_url_actual:
            raise ValueError("非Google模型请求需要 base_url。")
        api_url = f"{self.base_url_actual.rstrip('/')}/{endpoint_suffix.lstrip('/')}"
        last_exception = None
        max_retries = self.params.get(
            "max_retries", int(os.getenv("LLM_MAX_RETRIES", "3"))
        )
        for attempt in range(max_retries):
            current_api_key = self._get_current_api_key_for_http()
            headers = {"Content-Type": "application/json"}
            if (
                current_api_key
                and not current_api_key.startswith("dummy_")
                and "no_key" not in current_api_key.lower()
            ):
                headers["Authorization"] = f"Bearer {current_api_key}"
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    logger.debug(
                        f"Attempt {attempt + 1}/{max_retries} - HTTP POST to {api_url} using Key ...{current_api_key[-4:] if current_api_key else 'N/A'}"
                    )
                    timeout_seconds = self.params.get("http_timeout_seconds", 30)
                    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
                    async with session.post(
                        api_url, json=payload, timeout=timeout
                    ) as response:
                        response_text_snippet = (await response.text())[:500]
                        if response.status == 429 or response.status >= 500:
                            if attempt < max_retries - 1:
                                self._switch_key()
                                base_wait = self.params.get(
                                    "base_retry_wait_seconds", 5
                                )
                                wait_time = base_wait * (2**attempt) + random.uniform(
                                    0, 1
                                )
                                logger.warning(
                                    f"HTTP POST 失败 ({response.status}) to {api_url}. Response: {response_text_snippet}. "
                                    f"切换Key/等待 {wait_time:.2f}s 重试..."
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                self._record_usage(
                                    None,
                                    None,
                                    None,
                                    user_id,
                                    request_type,
                                    endpoint_suffix,
                                    "failure",
                                    f"Max retries HTTP {response.status}: {response_text_snippet}",
                                )
                                raise aiohttp.ClientResponseError(
                                    response.request_info,
                                    response.history,
                                    status=response.status,
                                    message=f"Max retries for rate limit/server error. Response: {response_text_snippet}",
                                )
                        response.raise_for_status()
                        response_json = await response.json()
                        return response_json
            except aiohttp.ClientResponseError as e:
                last_exception = e
                logger.warning(
                    f"HTTP POST 请求失败 (尝试 {attempt + 1}/{max_retries}): {e.status} {e.message} for URL {api_url}"
                )
                if e.status == 401 or e.status == 403:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        endpoint_suffix,
                        "failure",
                        f"HTTP {e.status}: {e.message}",
                    )
                    if attempt == 0 and self._switch_key():
                        logger.info(
                            "Switched API key due to auth error, retrying once with new key..."
                        )
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(
                        f"HTTP POST 请求认证失败或禁止访问: {e.status} {e.message}"
                    ) from e
                if attempt < max_retries - 1:
                    base_wait = self.params.get("base_retry_wait_seconds", 3)
                    wait_time = base_wait * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        endpoint_suffix,
                        "failure",
                        f"Max retries - HTTP {e.status}: {e.message}",
                    )
                    raise RuntimeError(
                        f"HTTP POST 请求在 {max_retries} 次尝试后失败。最后错误: {e.status} {e.message}"
                    ) from last_exception
            except asyncio.TimeoutError as e_timeout:
                last_exception = e_timeout
                logger.warning(
                    f"HTTP POST 请求超时 (尝试 {attempt + 1}/{max_retries}) for URL {api_url}"
                )
                if attempt < max_retries - 1:
                    base_wait = self.params.get("base_retry_wait_seconds", 5)
                    wait_time = base_wait * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        endpoint_suffix,
                        "failure",
                        "Max retries - Timeout",
                    )
                    raise RuntimeError(
                        f"HTTP POST 请求在 {max_retries} 次尝试后因超时失败。"
                    ) from last_exception
            except Exception as e_generic:
                last_exception = e_generic
                logger.error(
                    f"HTTP POST 请求中发生一般错误 (尝试 {attempt + 1}/{max_retries}): {type(e_generic).__name__} - {e_generic} for URL {api_url}"
                )
                if attempt < max_retries - 1:
                    base_wait = self.params.get("base_retry_wait_seconds", 3)
                    wait_time = base_wait * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    self._record_usage(
                        None,
                        None,
                        None,
                        user_id,
                        request_type,
                        endpoint_suffix,
                        "failure",
                        f"Max retries - Generic: {type(e_generic).__name__} - {e_generic}",
                    )
                    raise RuntimeError(
                        f"HTTP POST 请求在 {max_retries} 次尝试后因一般错误失败。最后错误: {e_generic}"
                    ) from last_exception
        self._record_usage(
            None,
            None,
            None,
            user_id,
            request_type,
            endpoint_suffix,
            "failure",
            "Max retries reached without throwing (HTTP POST logical error).",
        )
        raise RuntimeError(
            f"HTTP POST 请求在 {max_retries} 次尝试后失败 (代码逻辑问题，循环结束但未抛出最终异常)。"
        )

    async def generate_response_async(
        self,
        prompt: Union[str, List[Any]],
        user_id: str = "system",
        request_type: Optional[str] = None,
        **kwargs,
    ) -> GeminiSDKResponse:
        effective_request_type = request_type or self.request_type or "generate"
        if self.is_google_genai_model:
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
                error_msg = "generate_response_async for Google model received unexpected non-GeminiSDKResponse type."
                logger.error(error_msg + f" Got: {type(sdk_response_union)}")
                return GeminiSDKResponse(
                    content=f"Internal error: {error_msg}", reasoning=error_msg
                )
        else:
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
                            f"Unsupported item type in prompt list for HTTP model: {type(item)}"
                        )
            http_payload = {
                "model": self.model_name,
                "messages": payload_messages,
                "temperature": kwargs.get(
                    "temperature", self.params.get("temperature")
                ),
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
                self._record_usage(
                    p_tokens,
                    c_tokens,
                    t_tokens,
                    user_id,
                    effective_request_type,
                    "chat/completions",
                )
                content_cleaned, reasoning_text = self._extract_reasoning_sdk(
                    text_content
                )
                return GeminiSDKResponse(
                    content=content_cleaned,
                    reasoning=reasoning_text,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    raw_response=raw_json_response,
                )
            except Exception as e:
                logger.error(
                    f"非Google模型 ({self.model_name}) generate_response_async 失败: {e}"
                )
                self._record_usage(
                    None,
                    None,
                    None,
                    user_id,
                    effective_request_type,
                    "chat/completions",
                    "failure",
                    str(e),
                )
                return GeminiSDKResponse(
                    content=f"API调用失败: {e}",
                    reasoning=str(e),
                    raw_response={"error": str(e)},
                )

    async def get_embedding_async(
        self,
        text: str,
        user_id: str = "system",
        request_type: Optional[str] = None,
        **kwargs,
    ) -> Optional[List[float]]:
        if not text or not text.strip():
            logger.warning(
                "Embedding requested for empty or whitespace-only text. Returning None."
            )
            return None
        effective_request_type = request_type or self.request_type or "embedding"
        if self.is_google_genai_model:
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
                error_msg = "get_embedding_async for Google model received unexpected non-list type."
                logger.error(error_msg + f" Got: {type(embedding_vector_union)}")
                return None
        else:
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
                        f"未能从非Google模型 ({self.model_name}) 响应中提取有效的嵌入向量。响应: {str(raw_json_response)[:200]}"
                    )
                    embedding_vector = None
                usage_data = (
                    raw_json_response.get("usage", {})
                    if isinstance(raw_json_response, dict)
                    else {}
                )
                p_tokens = usage_data.get("prompt_tokens")
                t_tokens = usage_data.get("total_tokens")
                self._record_usage(
                    p_tokens, 0, t_tokens, user_id, effective_request_type, "embeddings"
                )
                return embedding_vector
            except Exception as e:
                logger.error(
                    f"非Google模型 ({self.model_name}) get_embedding_async 失败: {e}"
                )
                self._record_usage(
                    None,
                    None,
                    None,
                    user_id,
                    effective_request_type,
                    "embeddings",
                    "failure",
                    str(e),
                )
                return None


async def main_test():
    class DummyMongoHandler:
        def __init__(self):
            self.llm_usage = self

        def insert_one(self, data):
            print(
                f"DB (mock): llm_usage.insert_one called with: Model={data['model_name']}, P={data.get('prompt_tokens')}, C={data.get('completion_tokens')}, Status={data['status']}"
            )

        def create_index(self, keys, **kwargs):
            print(f"DB (mock): create_index called for llm_usage on {keys}")

    mock_db_instance = DummyMongoHandler()
    google_api_key = os.getenv("TEST_GOOGLE_API_KEY")
    if google_api_key:
        google_chat_model_config = {
            "name": "gemini-1.5-flash-latest",
            "key": google_api_key,
            "temperature": 0.7,
            "max_output_tokens": 50,
        }
        try:
            print("\n--- Testing Google GenAI Chat Model ---")
            google_llm = LLM_request(
                google_chat_model_config,
                mock_db_instance,
                request_type="test_google_chat",
            )
            response_google = await google_llm.generate_response_async(
                "Tell me a short joke.", user_id="test_user_google"
            )
            if response_google:
                print(f"  Google Response: {response_google.content}")
                print(f"  Reasoning: {response_google.reasoning}")
                print(
                    f"  Tokens: P={response_google.prompt_tokens}, C={response_google.completion_tokens}"
                )
        except Exception as e:
            print(f"  Error during Google Chat LLM test: {e}")
            import traceback

            traceback.print_exc()
        google_embed_config = {"name": "text-embedding-004", "key": google_api_key}
        try:
            print("\n--- Testing Google GenAI Embedding Model ---")
            google_embedder = LLM_request(
                google_embed_config,
                mock_db_instance,
                request_type="test_google_embedding",
            )
            vector_google = await google_embedder.get_embedding_async(
                "This is a test for Google embedding.", user_id="test_user_google_embed"
            )
            if vector_google:
                print(
                    f"  Google Embedding (first 5 dims): {vector_google[:5]}, Length: {len(vector_google)}"
                )
            else:
                print("  Failed to get Google embedding.")
        except Exception as e:
            print(f"  Error during Google Embedding test: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(
            "\n--- Skipping Google GenAI Chat & Embedding Model Test (TEST_GOOGLE_API_KEY not set) ---"
        )
    openai_compatible_api_key = os.getenv("TEST_OPENAI_API_KEY")
    openai_compatible_base_url = os.getenv("TEST_OPENAI_BASE_URL")
    if openai_compatible_api_key and openai_compatible_base_url:
        openai_chat_config = {
            "name": "gpt-3.5-turbo",
            "key": openai_compatible_api_key,
            "base_url": openai_compatible_base_url,
            "temperature": 0.7,
            "max_tokens": 60,
        }
        try:
            print("\n--- Testing OpenAI-Compatible Chat Model ---")
            openai_llm = LLM_request(
                openai_chat_config, mock_db_instance, request_type="test_openai_chat"
            )
            response_openai = await openai_llm.generate_response_async(
                "Tell me a fun fact.", user_id="test_user_openai"
            )
            if response_openai:
                print(f"  OpenAI-Compatible Response: {response_openai.content}")
                print(
                    f"  Tokens: P={response_openai.prompt_tokens}, C={response_openai.completion_tokens}"
                )
        except Exception as e:
            print(f"  Error during OpenAI-Compatible Chat test: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(
            "\n--- Skipping OpenAI-Compatible Chat Model Test (TEST_OPENAI_API_KEY or TEST_OPENAI_BASE_URL not set) ---"
        )
