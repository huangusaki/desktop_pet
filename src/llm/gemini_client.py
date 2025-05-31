from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
import re
import httpx
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List, Dict, Any, Optional
import logging
import asyncio
import os
import functools
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


class AgentStepSchema(BaseModel):
    tool_to_call: str = Field(..., description="Tool to be called for this step")
    tool_arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool for this step"
    )
    step_description: Optional[str] = Field(
        None, description="Description of this specific step"
    )


class PetResponseSchema(BaseModel):
    text: str = Field(..., description="Bot说的话, 或Agent的总体确认/错误信息")
    emotion: EmotionTypes = Field(..., description="Bot当前的情绪, 或Agent的情绪")
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
    steps: Optional[List[AgentStepSchema]] = Field(
        None, description="List of agent actions to perform in sequence"
    )
    tool_to_call: Optional[str] = Field(
        None,
        description="DEPRECATED: Use 'steps' list instead. Tool to be called by the agent",
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        None, description="DEPRECATED: Use 'steps' list instead. Arguments for the tool"
    )


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        pet_name: str,
        user_name: str,
        pet_persona: str,
        available_emotions: List[str],
        prompt_builder: PromptBuilder,
        mongo_handler: Any,
        config_manager: ConfigManager,
        thinking_budget: Optional[int] = 24000,
    ):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API Key 未在配置文件中设置或无效。")
        self.api_key = api_key
        self.model_name = model_name
        self.pet_name = pet_name
        self.prompt_builder = prompt_builder
        self.user_name = user_name
        self.pet_persona = pet_persona
        self.mongo_handler = mongo_handler
        self.config_manager = config_manager
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
        if self.thinking_budget is not None:
            if self.thinking_budget == 0:
                logger.info("信息: thinking_budget 设置为 0，将禁用思考过程。")
        self.client = None
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(f"无法初始化 genai.Client(). 原始错误: {e}")
        self.enabled_tools = []

    async def _build_chat_contents_for_api(
        self,
        new_user_message_text: str,
        hippocampus_manager: Optional[HippocampusManager],
        is_multimodal_request: bool = False,
    ) -> List[Any]:
        unified_prompt_str = await self.prompt_builder.build_unified_chat_prompt_string(
            new_user_message_text=new_user_message_text,
            pet_name=self.pet_name,
            user_name=self.user_name,
            pet_persona=self.pet_persona,
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
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            chat_contents: List[Any]
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
            generation_config_args = {
                "temperature": 0.6 if is_agent_mode else 0.78,
            }
            if self.enabled_tools and not is_agent_mode:
                generation_config_args["tools"] = self.enabled_tools
            if (
                not is_agent_mode
                and self.thinking_budget is not None
                and self.thinking_budget > 0
            ):
                generation_config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=True
                )
            api_config = (
                types.GenerateContentConfig(**generation_config_args)
                if generation_config_args
                else None
            )
            response_object = await self.client.aio.models.generate_content(
                model=self.model_name, contents=chat_contents, config=api_config
            )
            logger.info(
                f"GeminiClient: 原始LLM响应 (send_message, agent_mode={is_agent_mode}): {str(response_object)}"
            )
            llm_output_text = None
            if hasattr(response_object, "text") and response_object.text:
                llm_output_text = response_object.text
            elif hasattr(response_object, "candidates") and response_object.candidates:
                candidate = response_object.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                ):
                    full_text_parts = []
                    for part_item in candidate.content.parts:
                        current_part_text = getattr(part_item, "text", None)
                        if (
                            hasattr(part_item, "thought")
                            and part_item.thought
                            and current_part_text
                        ):
                            logger.info(
                                f"GeminiClient: LLM思考摘要 (part): {current_part_text.strip()}"
                            )
                        elif current_part_text:
                            full_text_parts.append(current_part_text)
                    if full_text_parts:
                        llm_output_text = "".join(full_text_parts)
                        logger.debug(
                            f"GeminiClient: 从candidates[0].content.parts提取并合并得到主要文本 (send_message)"
                        )
            if llm_output_text is None:
                logger.warning(
                    "GeminiClient: 未能从LLM响应中提取主要文本 (send_message)。"
                )
                return self._handle_empty_or_unparseable_response(
                    response_object,
                    f"{'Agent' if is_agent_mode else 'Chat'} Mode - 未找到主要文本",
                )
            return self._parse_llm_json_output(
                llm_output_text.strip(), response_object, is_agent_mode=is_agent_mode
            )
        except Exception as e:
            logger.error(
                f"GeminiClient send_message 严重错误 (agent_mode={is_agent_mode}):{e}",
                exc_info=True,
            )
            return self._handle_general_exception(
                e,
                f"{'Agent' if is_agent_mode else 'Chat'} send_message",
                response_object,
                is_agent_mode=is_agent_mode,
            )

    async def send_message_with_image(
        self, image_bytes: bytes, mime_type: str, prompt_text: str
    ) -> Dict[str, Any]:
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            image_part_obj = types.Part(
                inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
            )
            screen_analysis_text_prompt_str = (
                self.prompt_builder.build_screen_analysis_prompt(
                    pet_name=self.pet_name,
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
            vision_config_args = {"temperature": 0.78}
            if self.enabled_tools:
                vision_config_args["tools"] = self.enabled_tools
            if self.thinking_budget is not None and self.thinking_budget > 0:
                vision_config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=True
                )
            api_vision_config = (
                types.GenerateContentConfig(**vision_config_args)
                if vision_config_args
                else None
            )
            response_object = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=user_parts_for_vision,
                config=api_vision_config,
            )
            logger.info(
                f"GeminiClient: 原始LLM响应 (send_message_with_image): {str(response_object)}"
            )
            llm_output_text = None
            if hasattr(response_object, "text") and response_object.text:
                llm_output_text = response_object.text
            elif hasattr(response_object, "candidates") and response_object.candidates:
                candidate = response_object.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                ):
                    full_text_parts = []
                    for part_item in candidate.content.parts:
                        current_part_text = getattr(part_item, "text", None)
                        if (
                            hasattr(part_item, "thought")
                            and part_item.thought
                            and current_part_text
                        ):
                            logger.info(
                                f"LLM思考摘要 (part): {current_part_text.strip()}"
                            )
                        elif current_part_text:
                            full_text_parts.append(current_part_text)
                    if full_text_parts:
                        llm_output_text = "".join(full_text_parts)
            if llm_output_text is not None:
                return self._parse_llm_json_output(
                    llm_output_text.strip(), response_object, is_agent_mode=False
                )
            else:
                logger.warning("GeminiClient: 未能从LLM图像响应中提取主要文本。")
                return self._handle_empty_or_unparseable_response(
                    response_object, "图像分析 - 未找到主要文本"
                )
        except Exception as e:
            logger.error(f"Error in send_message_with_image: {e}", exc_info=True)
            return self._handle_general_exception(
                e, "send_message_with_image", response_object
            )

    async def send_multimodal_message_async(
        self,
        text_prompt: str,
        media_files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
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
            else:
                logger.info(
                    f"Multimodal Request - Text Prompt Part appears empty or not a Part: {text_parts}"
                )
            uploaded_file_uris_for_log = []
            if media_files:
                loop = asyncio.get_running_loop()
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
                            self.client.files.upload, file=file_path
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
            multimodal_config_args = {"temperature": 0.78}
            if self.enabled_tools:
                multimodal_config_args["tools"] = self.enabled_tools
            if self.thinking_budget is not None and self.thinking_budget > 0:
                multimodal_config_args["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget, include_thoughts=True
                )
            api_multimodal_config = (
                types.GenerateContentConfig(**multimodal_config_args)
                if multimodal_config_args
                else None
            )
            logger.info(
                f"Sending multimodal request with {len(api_contents)} parts to LLM. Config: {api_multimodal_config}"
            )
            response_object = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=api_contents,
                config=api_multimodal_config,
            )
            logger.info(
                f"GeminiClient: 原始LLM响应 (send_multimodal_message_async): {str(response_object)}"
            )
            llm_output_text = None
            if hasattr(response_object, "text") and response_object.text:
                llm_output_text = response_object.text
            elif hasattr(response_object, "candidates") and response_object.candidates:
                candidate = response_object.candidates[0]
                if (
                    hasattr(candidate, "content")
                    and candidate.content
                    and hasattr(candidate.content, "parts")
                ):
                    full_text_parts = []
                    for part_item in candidate.content.parts:
                        current_part_text = getattr(part_item, "text", None)
                        if (
                            hasattr(part_item, "thought")
                            and part_item.thought
                            and current_part_text
                        ):
                            logger.info(
                                f"LLM思考摘要 (part): {current_part_text.strip()}"
                            )
                        elif current_part_text:
                            full_text_parts.append(current_part_text)
                    if full_text_parts:
                        llm_output_text = "".join(full_text_parts)
            if llm_output_text is not None:
                parsed_result = self._parse_llm_json_output(
                    llm_output_text.strip(), response_object, is_agent_mode=False
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
            else:
                logger.warning("GeminiClient: 未能从LLM多模态响应中提取主要文本。")
                return self._handle_empty_or_unparseable_response(
                    response_object, "多模态 - 未找到主要文本"
                )
        except Exception as e:
            logger.error(f"Error in send_multimodal_message_async: {e}", exc_info=True)
            return self._handle_general_exception(
                e, "send_multimodal_message_async", response_object
            )

    def _parse_llm_json_output(
        self,
        llm_output_text: str,
        raw_response_object: Any,
        is_agent_mode: bool = False,
    ) -> Dict[str, Any]:
        json_str_to_parse = ""
        try:
            match_markdown_json = re.search(
                r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", llm_output_text, re.DOTALL
            )
            if match_markdown_json:
                json_str_to_parse = match_markdown_json.group(1).strip()
                logger.info("GeminiClient: 成功从Markdown代码块中提取JSON字符串。")
            else:
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
            parsed_data = json.loads(json_str_to_parse, strict=False)
            validated_data = PetResponseSchema(**parsed_data)
            if is_agent_mode:
                final_emotion = validated_data.emotion
                if not validated_data.steps:
                    agent_emotions_config = self.config_manager.config.get(
                        "PET",
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
                f"标准JSON解析或Pydantic验证失败 (agent_mode={is_agent_mode}): {type(e_initial_parse).__name__} - {e_initial_parse}. "
                f"原始文本片段: '{llm_output_text[:500]}'. 尝试进行更细致的正则提取..."
            )
            extracted_data = {}
            text_for_regex = llm_output_text
            thinking_detail_for_error_parts = [
                f"标准JSON解析/Pydantic验证失败 ({type(e_initial_parse).__name__}: {e_initial_parse})"
            ]
            match_text = re.search(
                r'"text"\s*:\s*"((\\"|[^"])*?)"\s*(?:,?\s*"(?:emotion|image_description|text_japanese|think|steps)"|})',
                text_for_regex,
                re.DOTALL,
            )
            if match_text:
                try:
                    extracted_data["text"] = json.loads(f'"{match_text.group(1)}"')
                except json.JSONDecodeError:
                    extracted_data["text"] = match_text.group(1)
                    thinking_detail_for_error_parts.append(
                        f"正则提取的text字段值 '{match_text.group(1)[:50]}' 无法被json.loads解析为字符串，使用原始值。"
                    )
            else:
                thinking_detail_for_error_parts.append("正则提取text失败。")
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
                    r'"emotion"\s*:\s*"((\\"|[^"])*?)"', text_for_regex
                )
                if match_emotion_fallback:
                    extracted_data["emotion"] = match_emotion_fallback.group(1)
                    if extracted_data["emotion"] not in self.available_emotions:
                        thinking_detail_for_error_parts.append(
                            f"正则提取的emotion '{extracted_data['emotion']}' 不在预定义列表中，但仍使用。"
                        )
                else:
                    thinking_detail_for_error_parts.append("正则提取emotion失败。")
            match_img_desc = re.search(
                r'"image_description"\s*:\s*"((\\"|[^"])*?)"\s*(?:,?\s*"(?:text|emotion|text_japanese|think|steps)"|})',
                text_for_regex,
                re.DOTALL,
            )
            if match_img_desc:
                try:
                    extracted_data["image_description"] = json.loads(
                        f'"{match_img_desc.group(1)}"'
                    )
                except json.JSONDecodeError:
                    extracted_data["image_description"] = match_img_desc.group(1)
                    thinking_detail_for_error_parts.append(
                        f"正则提取的image_description字段值 '{match_img_desc.group(1)[:50]}' 无法被json.loads解析为字符串，使用原始值。"
                    )
            else:
                thinking_detail_for_error_parts.append(
                    "正则提取image_description失败。"
                )
            match_jp = re.search(
                r'"text_japanese"\s*:\s*"((\\"|[^"])*?)"\s*(?:,?\s*"(?:text|emotion|image_description|think|steps)"|})',
                text_for_regex,
                re.DOTALL,
            )
            if match_jp:
                try:
                    extracted_data["text_japanese"] = json.loads(
                        f'"{match_jp.group(1)}"'
                    )
                except json.JSONDecodeError:
                    extracted_data["text_japanese"] = match_jp.group(1)
                    thinking_detail_for_error_parts.append(
                        f"正则提取的text_japanese字段值 '{match_jp.group(1)[:50]}' 无法被json.loads解析为字符串，使用原始值。"
                    )
            else:
                thinking_detail_for_error_parts.append("正则提取text_japanese失败。")
            match_think = re.search(
                r'"(?:thinking_process|think)"\s*:\s*"((\\"|[^"])*?)"\s*(?:,?\s*"(?:text|emotion|image_description|text_japanese|steps)"|})',
                text_for_regex,
                re.DOTALL,
            )
            if match_think:
                think_content = match_think.group(1)
                try:
                    extracted_data["thinking_process"] = json.loads(
                        f'"{think_content}"'
                    )
                except json.JSONDecodeError as e_think_json:
                    extracted_data["thinking_process"] = think_content
                    thinking_detail_for_error_parts.append(
                        f"正则提取的think字段值 '{think_content[:100]}...' 无法被json.loads安全解析为字符串 (错误: {e_think_json})，使用原始提取值。"
                    )
            else:
                thinking_detail_for_error_parts.append(
                    "正则提取think/thinking_process失败。"
                )
            if is_agent_mode:
                extracted_data["steps"] = []
                match_steps_block = re.search(
                    r'"steps"\s*:\s*(\[[\s\S]*?\])\s*(?:,?\s*"(?:text|emotion|image_description|text_japanese|think)"|})',
                    text_for_regex,
                    re.DOTALL,
                )
                if match_steps_block:
                    steps_json_str = match_steps_block.group(1)
                    try:
                        parsed_steps = json.loads(steps_json_str)
                        if isinstance(parsed_steps, list):
                            extracted_data["steps"] = parsed_steps
                        else:
                            thinking_detail_for_error_parts.append(
                                "正则提取的steps内容不是列表。"
                            )
                    except json.JSONDecodeError as e_steps_json:
                        thinking_detail_for_error_parts.append(
                            f"正则提取的steps块无法解析为JSON: {e_steps_json}"
                        )
                else:
                    thinking_detail_for_error_parts.append("正则提取steps失败。")
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
                    if is_agent_mode:
                        final_emotion_regex = validated_data_via_regex.emotion
                        if not validated_data_via_regex.steps:
                            agent_emotions_config = self.config_manager.config.get(
                                "PET",
                                "AGENT_MODE_EMOTIONS",
                                fallback="'neutral', 'focused', 'helpful'",
                            )
                            agent_available_emotions = [
                                e.strip().strip("'\"")
                                for e in agent_emotions_config.split(",")
                            ]
                            if (
                                validated_data_via_regex.emotion
                                not in agent_available_emotions
                            ):
                                final_emotion_regex = "neutral"
                        validated_data_via_regex.emotion = final_emotion_regex
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
                        f"细致正则提取的数据未能通过Pydantic验证 (agent_mode={is_agent_mode}): {e_regex_val}. "
                        f"提取的数据: {extracted_data}"
                    )
                except Exception as e_regex_build:
                    thinking_detail_for_error_parts.append(
                        f"正则提取后构建PetResponseSchema对象时发生意外错误 ({type(e_regex_build).__name__}: {e_regex_build})"
                    )
                    logger.error(
                        f"正则提取后构建对象时出错: {e_regex_build}", exc_info=True
                    )
            else:
                thinking_detail_for_error_parts.append(
                    "正则提取未能获得关键数据 (text/emotion)"
                )
                logger.warning(
                    f"细致正则提取未能获得关键数据 (text/emotion) (agent_mode={is_agent_mode})."
                )
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            final_thinking_details = ". ".join(thinking_detail_for_error_parts)
            thinking = (
                f"<think>解析尝试失败详情: {final_thinking_details}. "
                f"解析前的原始文本片段: '{llm_output_text[:500]}'. 反馈: {feedback_info}</think>"
            )
            final_fallback_text = extracted_data.get("text")
            if not final_fallback_text or not str(final_fallback_text).strip():
                final_fallback_text = (
                    llm_output_text if llm_output_text else "我好像有点混乱..."
                )
            if not str(
                final_fallback_text
            ).strip() or "未能从LLM响应中提取到主要文本内容" in str(final_fallback_text):
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
            "thinking_process": thinking_on_error,
            "is_error": True,
        }
        if is_agent_mode:
            result["steps"] = []
        return result

    def _get_prompt_feedback_info(self, response_obj: Any) -> str:
        if not response_obj:
            return "提示反馈: (无响应对象)"
        all_feedback_parts = []
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
            f"{context} 中出错: {error_message}\n原始异常详情 ({type(e).__name__}): {details_from_exception}\n思考: {thinking_on_error}",
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
            "thinking_process": thinking_on_error,
            "is_error": True,
        }
        if is_agent_mode:
            result["steps"] = []
        return result
