"""
Gemini 聊天机器人客户端
专注于聊天机器人场景的 JSON 响应解析和 prompt 构建
"""
from google.genai import types
import json
import re
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
import logging
import asyncio
import os
import functools

from .llm_request import LLM_request, GeminiSDKResponse
from ..utils.prompt_builder import PromptBuilder
from ..utils.config_manager import ConfigManager

try:
    from ..memory_system.hippocampus_core import HippocampusManager
except ImportError:
    try:
        from memory_system.hippocampus_core import HippocampusManager
    except ImportError:
        HippocampusManager = None
        
logger = logging.getLogger("GeminiClient")
EmotionTypes = str
ToneTypes = str


class AgentStepSchema(BaseModel):
    """Agent 步骤 Schema"""
    tool_to_call: str = Field(..., description="Tool to be called for this step")
    tool_arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool for this step"
    )
    step_description: Optional[str] = Field(
        None, description="Description of this specific step"
    )


class PetResponseSchema(BaseModel):
    """聊天机器人响应 Schema"""
    text: str = Field(..., description="Bot说的话, 或Agent的总体确认/错误信息")
    emotion: EmotionTypes = Field(..., description="Bot当前的情绪, 或Agent的情绪")
    tone: Optional[ToneTypes] = Field(
        None,
        description="Bot说话的语调，用于TTS选择参考音频。例如 'normal', 'serious', 'sad'",
    )
    image_description: Optional[str] = Field(
        None,
        description="对图片内容的客观描述，例如 '图片包含一只猫和一只狗' 或 '屏幕截图显示了一个代码编辑器'",
    )
    thinking_process: Optional[str] = Field(
        None,
        description="模型的思考过程。",
    )
    text_japanese: Optional[str] = Field(
        None, description="Bot说的话的日语版本，用于TTS (Agent模式下通常为null)"
    )
    favorability_change: int = Field(
        0,
        description="基于本次用户输入和上下文,对好感度的影响值。正数表示增加,负数表示减少,0表示无变化。请将变化控制在-5到5之间。",
    )
    emotion_update: Optional[Dict[str, str]] = Field(
        None,
        description='这次对话后你对用户的心情状态更新,包含state(心情状态,日文)和reason(具体原因,日文)。例如: {"state": "嬉しい", "reason": "2025-11-21 11:00に優しい言葉をかけてくれたから"}',
    )
    steps: Optional[List[AgentStepSchema]] = Field(
        None, description="List of agent actions to perform in sequence"
    )


class GeminiClient:
    """
    Gemini 聊天机器人客户端
    专注于聊天机器人场景,使用 LLM_request 处理底层 API 调用
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        bot_name: str,
        user_name: str,
        bot_persona: str,
        available_emotions: List[str],
        prompt_builder: PromptBuilder,
        mongo_handler: Any,
        config_manager: ConfigManager,
        thinking_budget: Optional[int] = 24000,
    ):
        """
        初始化 Gemini 聊天机器人客户端
        
        Args:
            api_key: Gemini API Key(可以是逗号分隔的多个 key)
            model_name: 模型名称
            bot_name: 机器人名称
            user_name: 用户名称
            bot_persona: 机器人人设
            available_emotions: 可用情绪列表
            prompt_builder: Prompt 构建器
            mongo_handler: MongoDB 处理器
            config_manager: 配置管理器
            thinking_budget: 思考预算(token 数)
        """
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API Key 未在配置文件中设置或无效。")
            
        self.model_name = model_name
        self.bot_name = bot_name
        self.prompt_builder = prompt_builder
        self.user_name = user_name
        self.bot_persona = bot_persona
        self.mongo_handler = mongo_handler
        self.config_manager = config_manager
        
        # 处理情绪列表
        self.unified_default_emotion = "default"
        processed_emotions = set(
            e.lower() for e in available_emotions if isinstance(e, str) and e.strip()
        )
        if not processed_emotions:
            processed_emotions.add(self.unified_default_emotion)
        if self.unified_default_emotion not in processed_emotions:
            processed_emotions.add(self.unified_default_emotion)
        self.available_emotions = sorted(list(processed_emotions))
        
        self.thinking_budget = thinking_budget
        if self.thinking_budget is not None and self.thinking_budget == 0:
            logger.info("信息: thinking_budget 设置为 0，将禁用思考过程。")
            
        # 创建底层 LLM 请求客户端
        model_config = {
            "name": model_name,
            "key": api_key,
        }
        
        self.llm_client = LLM_request(
            model_config=model_config,
            db_instance=mongo_handler,
            request_type="chat",
        )
        
        logger.info(
            f"GeminiClient 初始化完成: 模型={model_name}, "
            f"API Keys={len(self.llm_client.api_keys)}"
        )
        
        self.enabled_tools = []
        self.available_tones = self.config_manager.get_tts_available_tones()
        self.default_tone = self.config_manager.get_tts_default_tone()
        
    async def _build_chat_contents_for_api(
        self,
        new_user_message_text: str,
        hippocampus_manager: Optional[HippocampusManager],
        is_multimodal_request: bool = False,
    ) -> List[Any]:
        """构建聊天内容用于 API 调用"""
        unified_prompt_str = await self.prompt_builder.build_unified_chat_prompt_string(
            new_user_message_text=new_user_message_text,
            bot_name=self.bot_name,
            user_name=self.user_name,
            bot_persona=self.bot_persona,
            available_emotions=self.available_emotions,
            unified_default_emotion=self.unified_default_emotion,
            mongo_handler=self.mongo_handler,
            hippocampus_manager=hippocampus_manager,
            is_multimodal_request=is_multimodal_request,
        )
        
        if isinstance(unified_prompt_str, str):
            return [types.Part(text=unified_prompt_str)]
        elif isinstance(unified_prompt_str, list) and all(
            isinstance(p, types.Part) for p in unified_prompt_str
        ):
            return unified_prompt_str
        elif isinstance(unified_prompt_str, types.Part):
            return [unified_prompt_str]
        else:
            if (
                isinstance(unified_prompt_str, list)
                and unified_prompt_str
                and isinstance(unified_prompt_str[0], str)
            ):
                return [types.Part(text=unified_prompt_str[0])]
            else:
                return [types.Part(text=str(unified_prompt_str))]
                
    async def send_message(
        self,
        message_text: str,
        hippocampus_manager: Optional[HippocampusManager],
        is_agent_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        发送消息并获取响应
        
        Args:
            message_text: 用户消息文本
            hippocampus_manager: 记忆管理器
            is_agent_mode: 是否为 Agent 模式
            
        Returns:
            解析后的响应字典
        """
        try:
            # 构建内容
            if is_agent_mode:
                chat_contents = [types.Part(text=message_text)]
                logger.info("GeminiClient: Sending Agent Mode prompt directly as Part.")
            else:
                chat_contents = await self._build_chat_contents_for_api(
                    new_user_message_text=message_text,
                    hippocampus_manager=hippocampus_manager,
                    is_multimodal_request=False,
                )
                
            if not chat_contents or not getattr(chat_contents[0], "text", None):
                error_msg = "无法构建有效的统一提示字符串以供API调用。"
                logger.error(f"GeminiClient: 错误 - {error_msg}. 内容: {chat_contents}")
                return {
                    "text": "抱歉，内部构建消息时出错。",
                    "emotion": self.unified_default_emotion,
                    "thinking_process": f"<think>错误: {error_msg}</think>",
                    "is_error": True,
                }
                
            # 构建配置
            config_overrides = {
                "temperature": 0.6 if is_agent_mode else 0.70,
            }
            
            if self.enabled_tools and not is_agent_mode:
                config_overrides["tools"] = self.enabled_tools
                
            if (
                not is_agent_mode
                and self.thinking_budget is not None
                and self.thinking_budget > 0
            ):
                config_overrides["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=False
                )
                
            # 调用 LLM
            response: GeminiSDKResponse = await self.llm_client.generate_response(
                prompt=chat_contents,
                user_id="system",
                request_type="chat" if not is_agent_mode else "agent",
                **config_overrides,
            )
            
            logger.info(
                f"GeminiClient: LLM响应内容 (agent_mode={is_agent_mode}): "
                f"{response.content[:200]}..."
            )
            
            # 解析 JSON 输出
            return self._parse_llm_json_output(
                response.content, response.raw_response, is_agent_mode=is_agent_mode
            )
            
        except Exception as e:
            logger.error(
                f"GeminiClient send_message 严重错误 (agent_mode={is_agent_mode}):{e}",
                exc_info=True,
            )
            return self._handle_general_exception(
                e,
                f"{'Agent' if is_agent_mode else 'Chat'} send_message",
                None,
                is_agent_mode=is_agent_mode,
            )
            
    async def send_message_with_image(
        self, image_bytes: bytes, mime_type: str, prompt_text: str
    ) -> Dict[str, Any]:
        """
        发送带图片的消息
        
        Args:
            image_bytes: 图片字节数据
            mime_type: MIME 类型
            prompt_text: 提示文本
            
        Returns:
            解析后的响应字典
        """
        try:
            image_part_obj = types.Part(
                inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
            )
            
            screen_analysis_text_prompt_str = (
                self.prompt_builder.build_screen_analysis_prompt(
                    bot_name=self.bot_name,
                    user_name=self.user_name,
                    available_emotions=self.available_emotions,
                    mongo_handler=self.mongo_handler,
                    unified_default_emotion=self.unified_default_emotion,
                )
            )
            
            user_parts_for_vision = [types.Part(text=screen_analysis_text_prompt_str)]
            if prompt_text and prompt_text.strip():
                user_parts_for_vision.append(
                    types.Part(text=f"\n用户补充说明：{prompt_text}")
                )
            user_parts_for_vision.append(image_part_obj)
            
            vision_config_overrides = {"temperature": 0.70}
            if self.enabled_tools:
                vision_config_overrides["tools"] = self.enabled_tools
            if self.thinking_budget is not None and self.thinking_budget > 0:
                vision_config_overrides["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=False
                )
                
            # 调用 LLM
            response: GeminiSDKResponse = await self.llm_client.generate_response(
                prompt=user_parts_for_vision,
                user_id="system",
                request_type="vision",
                **vision_config_overrides,
            )
            
            logger.info(
                f"GeminiClient: 图像分析响应: {response.content[:200]}..."
            )
            
            return self._parse_llm_json_output(
                response.content, response.raw_response, is_agent_mode=False
            )
            
        except Exception as e:
            logger.error(f"Error in send_message_with_image: {e}", exc_info=True)
            return self._handle_general_exception(
                e, "send_message_with_image", None
            )
            
    async def send_multimodal_message_async(
        self,
        text_prompt: str,
        media_files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        发送多模态消息(文本 + 多个媒体文件)
        
        Args:
            text_prompt: 文本提示
            media_files: 媒体文件列表
            
        Returns:
            解析后的响应字典
        """
        try:
            # 构建文本部分
            text_parts = await self._build_chat_contents_for_api(
                new_user_message_text=text_prompt,
                hippocampus_manager=None,
                is_multimodal_request=True,
            )
            
            api_contents: List[Any] = list(text_parts)
            
            if text_parts and hasattr(text_parts[0], "text"):
                logger.info(
                    f"Multimodal Request - Text Prompt Part: {text_parts[0].text[:200]}..."
                )
                
            # 上传媒体文件
            uploaded_file_uris_for_log = []
            if media_files:
                loop = asyncio.get_running_loop()
                client_for_upload = self.llm_client._get_current_sync_client()
                
                for file_info in media_files:
                    file_path = file_info["path"]
                    display_name_for_log = file_info.get(
                        "display_name", os.path.basename(file_path)
                    )
                    
                    logger.info(
                        f"Uploading file for multimodal message: {display_name_for_log} from {file_path}"
                    )
                    
                    try:
                        upload_func_with_args = functools.partial(
                            client_for_upload.files.upload, file=file_path
                        )
                        uploaded_file_object = await loop.run_in_executor(
                            None, upload_func_with_args
                        )
                        api_contents.append(uploaded_file_object)
                        uploaded_file_uris_for_log.append(uploaded_file_object.uri)
                        logger.info(
                            f"Successfully uploaded {display_name_for_log}, URI: {uploaded_file_object.uri}"
                        )
                    except Exception as e_upload:
                        logger.error(
                            f"Failed to upload file {display_name_for_log}: {e_upload}",
                            exc_info=True,
                        )
                        return self._handle_general_exception(
                            e_upload,
                            f"multimodal file upload ({display_name_for_log})",
                            is_agent_mode=False,
                        )
                        
            if not api_contents or not any(
                (hasattr(p, "text") and p.text.strip()) or isinstance(p, types.File)
                for p in api_contents
                if p is not None
            ):
                logger.warning("Multimodal - No valid content (text or files) to send.")
                return self._handle_empty_or_unparseable_response(
                    None, "Multimodal - No content to send"
                )
                
            multimodal_config_overrides = {"temperature": 0.70}
            if self.enabled_tools:
                multimodal_config_overrides["tools"] = self.enabled_tools
            if self.thinking_budget is not None and self.thinking_budget > 0:
                multimodal_config_overrides["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=False
                )
                
            logger.info(
                f"Sending multimodal request with {len(api_contents)} parts to LLM."
            )
            
            # 调用 LLM
            response: GeminiSDKResponse = await self.llm_client.generate_response(
                prompt=api_contents,
                user_id="system",
                request_type="multimodal",
                **multimodal_config_overrides,
            )
            
            logger.info(
                f"GeminiClient: 多模态响应: {response.content[:200]}..."
            )
            
            parsed_result = self._parse_llm_json_output(
                response.content, response.raw_response, is_agent_mode=False
            )
            
            if uploaded_file_uris_for_log:
                thinking_prefix = (
                    "<think>Uploaded files: "
                    + ", ".join(uploaded_file_uris_for_log)
                    + "</think> "
                )
                parsed_result["thinking_process"] = thinking_prefix + (
                    parsed_result.get("thinking_process", "") or ""
                )
                
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in send_multimodal_message_async: {e}", exc_info=True)
            return self._handle_general_exception(
                e, "send_multimodal_message_async", None
            )
            
    def _parse_llm_json_output(
        self,
        llm_output_text: str,
        raw_response_object: Any,
        is_agent_mode: bool = False,
    ) -> Dict[str, Any]:
        """解析 LLM 的 JSON 输出"""
        json_str_to_parse = ""
        
        try:
            # 尝试从 Markdown 代码块提取 JSON
            match_markdown_json = re.search(
                r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", llm_output_text, re.DOTALL
            )
            
            if match_markdown_json:
                json_str_to_parse = match_markdown_json.group(1).strip()
                logger.info("GeminiClient: 成功从Markdown代码块中提取JSON字符串。")
            else:
                # 尝试提取第一个 { 到最后一个 }
                first_brace = llm_output_text.find("{")
                last_brace = llm_output_text.rfind("}")
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str_to_parse = llm_output_text[
                        first_brace : last_brace + 1
                    ].strip()
                    logger.info(
                        "GeminiClient: 尝试从第一个'{'到最后一个'}'提取JSON字符串。"
                    )
                else:
                    json_str_to_parse = llm_output_text.strip()
                    logger.info(
                        "GeminiClient: 未找到Markdown或明确的JSON边界，使用原始文本尝试解析。"
                    )
                    
            # 解析 JSON
            parsed_data = json.loads(json_str_to_parse, strict=False)
            
            # 处理 tone 字段
            if "tone" not in parsed_data or not parsed_data["tone"]:
                parsed_data["tone"] = self.default_tone
            elif parsed_data["tone"] not in self.available_tones:
                logger.warning(
                    f"GeminiClient: LLM返回的语调 '{parsed_data['tone']}' 不在可用语调列表中 "
                    f"({self.available_tones})。将使用默认语调 '{self.default_tone}'。"
                )
                parsed_data["tone"] = self.default_tone
                
            # 验证数据
            validated_data = PetResponseSchema(**parsed_data)
            
            # Agent 模式特殊处理
            if is_agent_mode:
                final_emotion = validated_data.emotion
                if not validated_data.steps:
                    agent_emotions_config = self.config_manager.config.get(
                        "BOT",
                        "AGENT_MODE_EMOTIONS",
                        fallback="'neutral', 'focused', 'helpful'",
                    )
                    agent_available_emotions = [
                        e.strip().strip("'\"") for e in agent_emotions_config.split(",")
                    ]
                    if validated_data.emotion not in agent_available_emotions:
                        final_emotion = "neutral"
                validated_data.emotion = final_emotion
                
            logger.info(
                f"成功解析并验证了LLM的JSON输出 (agent_mode={is_agent_mode}). "
                f"Text: '{validated_data.text[:50]}...', Emotion: {validated_data.emotion}"
            )
            
            if validated_data.image_description:
                logger.info(
                    f"Image Description: {validated_data.image_description[:100]}"
                )
                
            if is_agent_mode:
                logger.info(f"Agent Steps: {validated_data.steps}")
                
            return validated_data.model_dump()
            
        except (json.JSONDecodeError, TypeError, ValidationError) as e_initial_parse:
            logger.warning(
                f"标准JSON解析或Pydantic验证失败 (agent_mode={is_agent_mode}): "
                f"{type(e_initial_parse).__name__} - {e_initial_parse}. "
                f"原始文本片段: '{llm_output_text[:500]}'. 尝试进行更细致的正则提取..."
            )
            
            # 使用正则表达式提取关键字段
            return self._extract_fields_with_regex(
                llm_output_text, raw_response_object, is_agent_mode
            )
            
    def _extract_fields_with_regex(
        self,
        text_for_regex: str,
        raw_response_object: Any,
        is_agent_mode: bool,
    ) -> Dict[str, Any]:
        """使用正则表达式提取字段(当 JSON 解析失败时)"""
        extracted_data = {}
        thinking_detail_for_error_parts = [
            "标准JSON解析/Pydantic验证失败，使用正则提取"
        ]
        
        # 提取 text
        match_text = re.search(
            r'"text"\s*:\s*"((?:\\"|[^"])*?)"\s*(?:,?\s*"(?:emotion|image_description|text_japanese|think|steps)"|}))',
            text_for_regex,
            re.DOTALL,
        )
        if match_text:
            try:
                extracted_data["text"] = json.loads(f'"{match_text.group(1)}"')
            except json.JSONDecodeError:
                extracted_data["text"] = match_text.group(1)
        else:
            thinking_detail_for_error_parts.append("正则提取text失败。")
            
        # 提取 emotion
        emotions_pattern = "|".join(re.escape(e) for e in self.available_emotions)
        if not emotions_pattern:
            emotions_pattern = r"[a-zA-Z_]+"
            
        match_emotion = re.search(
            rf'"emotion"\s*:\s*"({emotions_pattern})"\s*(?:,?\s*"(?:text|image_description|text_japanese|think|steps)"|}})',
            text_for_regex,
        )
        
        if match_emotion:
            extracted_data["emotion"] = match_emotion.group(1)
        else:
            match_emotion_fallback = re.search(
                r'"emotion"\s*:\s*"((?:\\"|[^"])*?)"', text_for_regex
            )
            if match_emotion_fallback:
                extracted_data["emotion"] = match_emotion_fallback.group(1)
                if extracted_data["emotion"] not in self.available_emotions:
                    thinking_detail_for_error_parts.append(
                        f"正则提取的emotion '{extracted_data['emotion']}' 不在预定义列表中，但仍使用。"
                    )
            else:
                thinking_detail_for_error_parts.append("正则提取emotion失败。")
                
        # 如果成功提取了关键字段
        if extracted_data.get("text") and extracted_data.get("emotion"):
            logger.info(f"细致正则提取尝试完成，部分或全部数据: {extracted_data}")
            
            try:
                if "text_japanese" not in extracted_data:
                    extracted_data["text_japanese"] = None
                if "thinking_process" not in extracted_data:
                    extracted_data["thinking_process"] = None
                if "image_description" not in extracted_data:
                    extracted_data["image_description"] = None
                if is_agent_mode and "steps" not in extracted_data:
                    extracted_data["steps"] = []
                    
                validated_data_via_regex = PetResponseSchema(**extracted_data)
                
                logger.info(
                    f"通过细致正则提取并Pydantic验证成功 (agent_mode={is_agent_mode})."
                )
                
                current_thinking = validated_data_via_regex.thinking_process or ""
                recovery_info = "<think>通过细致正则提取恢复了数据。</think>"
                validated_data_via_regex.thinking_process = (
                    recovery_info + current_thinking
                )
                
                return validated_data_via_regex.model_dump()
                
            except ValidationError as e_regex_val:
                thinking_detail_for_error_parts.append(
                    f"正则提取后Pydantic验证失败 ({type(e_regex_val).__name__}: {e_regex_val})"
                )
                logger.warning(
                    f"细致正则提取的数据未能通过Pydantic验证 (agent_mode={is_agent_mode}): "
                    f"{e_regex_val}. 提取的数据: {extracted_data}"
                )
        else:
            thinking_detail_for_error_parts.append(
                "正则提取未能获得关键数据 (text/emotion)"
            )
            logger.warning(
                f"细致正则提取未能获得关键数据 (text/emotion) (agent_mode={is_agent_mode})."
            )
            
        # 最终回退
        feedback_info = self._get_prompt_feedback_info(raw_response_object)
        final_thinking_details = ". ".join(thinking_detail_for_error_parts)
        thinking = (
            f"<think>解析尝试失败详情: {final_thinking_details}. "
            f"解析前的原始文本片段: '{text_for_regex[:500]}'. 反馈: {feedback_info}</think>"
        )
        
        final_fallback_text = extracted_data.get("text")
        if not final_fallback_text or not str(final_fallback_text).strip():
            final_fallback_text = (
                text_for_regex if text_for_regex else "我好像有点混乱..."
            )
            
        if not str(
            final_fallback_text
        ).strip() or "未能从LLM响应中提取到主要文本内容" in str(
            final_fallback_text
        ):
            final_fallback_text = "抱歉，我好像有点混乱，无法正确理解我的思路。"
            
        if "BlockReason=" in feedback_info and not any(
            reason in feedback_info.upper()
            for reason in [
                "NONE",
                "UNSPECIFIED",
                "STOP",
                "BLOCK_REASON_UNSPECIFIED",
            ]
        ):
            final_fallback_text = "内容被阻止(所有解析尝试失败后检查)."
            
        result = {
            "text": final_fallback_text,
            "emotion": extracted_data.get(
                "emotion",
                (
                    "neutral"
                    if is_agent_mode
                    else (
                        "confused"
                        if "confused" in self.available_emotions
                        else self.unified_default_emotion
                    )
                ),
            ),
            "thinking_process": thinking,
            "is_error": True,
            "text_japanese": extracted_data.get("text_japanese"),
            "image_description": extracted_data.get("image_description"),
        }
        
        if is_agent_mode:
            result["steps"] = extracted_data.get("steps", [])
            
        return result
        
    def _handle_empty_or_unparseable_response(
        self, response_object: Any, context_str: str, is_agent_mode: bool = False
    ) -> Dict[str, Any]:
        """处理空或无法解析的响应"""
        raw_response_snippet = str(response_object)[:200] if response_object else "None"
        prompt_feedback_info = self._get_prompt_feedback_info(response_object)
        thinking_on_error = f"<think>{context_str} 响应不是 PetResponseSchema 实例且未找到 text/parsed。片段: {raw_response_snippet}. {prompt_feedback_info}</think>"
        logger.error(f"错误: {context_str} LLM输出问题. {thinking_on_error}")
        
        error_text = f"抱歉，我好像没能生成有效的{context_str}回复。"
        
        if "BlockReason=" in prompt_feedback_info:
            block_reason_match = re.search(
                r"BlockReason=([^,;\]]+)", prompt_feedback_info
            )
            if block_reason_match:
                block_reason_val = block_reason_match.group(1).strip().upper()
                if block_reason_val not in [
                    "NONE",
                    "BLOCK_REASON_UNSPECIFIED",
                    "STOP",
                    "MAX_TOKENS",
                    "RECITATION",
                    "OTHER",
                    "",
                ]:
                    error_text = f"抱歉，我的{context_str}回复似乎被系统拦截了 (原因: {block_reason_val})。"
        elif "FinishReason=" in prompt_feedback_info:
            finish_reason_match = re.search(
                r"FinishReason=([^,;\]]+)", prompt_feedback_info
            )
            if finish_reason_match:
                finish_reason_val = finish_reason_match.group(1).strip().upper()
                if finish_reason_val == "SAFETY":
                    error_text = f"抱歉，我的{context_str}回复因安全原因被拦截了。"
                elif finish_reason_val == "RECITATION":
                    error_text = (
                        f"抱歉，我的{context_str}回复因引用受保护内容过多被拦截。"
                    )
                    
        result = {
            "text": error_text,
            "emotion": "neutral" if is_agent_mode else self.unified_default_emotion,
            "tone": self.default_tone,
            "thinking_process": thinking_on_error,
            "is_error": True,
        }
        
        if is_agent_mode:
            result["steps"] = []
            
        return result
        
    def _get_prompt_feedback_info(self, response_obj: Any) -> str:
        """获取提示反馈信息"""
        if not response_obj:
            return "提示反馈: (无响应对象)"
            
        all_feedback_parts = []
        
        # 检查 prompt_feedback
        prompt_feedback = getattr(response_obj, "prompt_feedback", None)
        if prompt_feedback:
            block_reason_obj = getattr(prompt_feedback, "block_reason", None)
            block_reason_str = str(
                getattr(block_reason_obj, "name", block_reason_obj)
                if block_reason_obj is not None
                else "N/A"
            )
            
            safety_ratings_list = getattr(prompt_feedback, "safety_ratings", None)
            pf_safety_info = []
            if safety_ratings_list:
                pf_safety_info = [
                    f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                    for r in safety_ratings_list
                ]
                
            all_feedback_parts.append(
                f"PromptFeedback:BlockReason={block_reason_str},SafetyRatings=[{','.join(pf_safety_info)}]"
            )
            
        # 检查 candidates
        candidates = getattr(response_obj, "candidates", [])
        if candidates:
            for i, candidate in enumerate(candidates):
                finish_reason_obj = getattr(candidate, "finish_reason", None)
                finish_reason_str = str(
                    getattr(finish_reason_obj, "name", finish_reason_obj)
                    if finish_reason_obj is not None
                    else "N/A"
                )
                
                cand_safety_ratings_list = getattr(candidate, "safety_ratings", None)
                cand_safety_info = []
                if cand_safety_ratings_list:
                    cand_safety_info = [
                        f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                        for r in cand_safety_ratings_list
                    ]
                    
                all_feedback_parts.append(
                    f"Candidate[{i}]:FinishReason={finish_reason_str},SafetyRatings=[{','.join(cand_safety_info)}]"
                )
                
        if not all_feedback_parts:
            error_attr = getattr(response_obj, "_error", None)
            if error_attr:
                return f"提示反馈: (错误对象: {str(error_attr)[:100]})"
            return "提示反馈: (在response_obj中未找到特定的反馈属性)"
            
        return "; ".join(all_feedback_parts)
        
    def _handle_general_exception(
        self,
        e: Exception,
        context: str,
        raw_response_object: Any = None,
        is_agent_mode: bool = False,
    ) -> Dict[str, Any]:
        """处理一般性异常"""
        error_message = f"抱歉，我现在无法回复 ({context})。错误: {type(e).__name__}"
        details_from_exception = str(e)
        thinking_on_error = f"<think>{context} 中发生一般性异常: {type(e).__name__} - {details_from_exception}."
        
        feedback_str = ""
        if hasattr(e, "response") and e.response:
            feedback_str = self._get_prompt_feedback_info(getattr(e, "response", None))
        elif raw_response_object:
            feedback_str = self._get_prompt_feedback_info(raw_response_object)
            
        if (
            feedback_str
            and feedback_str
            not in [
                "提示反馈: (在response_obj中未找到特定的反馈属性)",
                "提示反馈: (无响应对象)",
            ]
            and "错误对象" not in feedback_str
        ):
            thinking_on_error += f" 反馈: {feedback_str}"
            
            if "BlockReason=" in feedback_str:
                match = re.search(r"BlockReason=([^,;]+)", feedback_str)
                if match and match.group(1).strip().upper() not in [
                    "NONE",
                    "BLOCK_REASON_UNSPECIFIED",
                    "STOP",
                    "N/A",
                    "",
                    "OTHER",
                ]:
                    error_message = f"抱歉，我的回复 ({context}) 似乎被系统拦截了 (原因: {match.group(1).strip()})."
            elif "FinishReason=" in feedback_str:
                match = re.search(r"FinishReason=([^,;]+)", feedback_str)
                if match:
                    reason = match.group(1).strip().upper()
                    if reason == "SAFETY":
                        error_message = (
                            f"抱歉，我的回复 ({context}) 因安全原因被拦截了."
                        )
                    elif reason == "RECITATION":
                        error_message = (
                            f"抱歉，我的回复 ({context}) 因引用过多受保护内容被拦截了."
                        )
                        
        exception_type_name = type(e).__name__
        if (
            "PermissionDenied" in exception_type_name
            or "Forbidden" in exception_type_name
        ):
            error_message = (
                f"抱歉，我好像没有权限访问 ({context})。请检查API Key或相关服务设置。"
            )
        elif "ResourceExhausted" in exception_type_name:
            error_message = (
                f"抱歉，系统有点忙（可能达到配额/速率限制），请稍后再试 ({context})。"
            )
            
        logger.error(
            f"{context} 中出错: {error_message}\n原始异常详情 ({type(e).__name__}): "
            f"{details_from_exception}\n思考: {thinking_on_error}",
            exc_info=True,
        )
        
        result = {
            "text": error_message,
            "emotion": (
                "neutral"
                if is_agent_mode
                else (
                    "sad"
                    if "sad" in self.available_emotions
                    else self.unified_default_emotion
                )
            ),
            "tone": self.default_tone,
            "thinking_process": thinking_on_error,
            "is_error": True,
        }
        
        if is_agent_mode:
            result["steps"] = []
            
        return result
