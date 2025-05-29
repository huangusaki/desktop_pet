from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
import re
import httpx
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, List, Dict, Any, Optional
import logging
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
    ) -> List[str]:
        unified_prompt = await self.prompt_builder.build_unified_chat_prompt_string(
            new_user_message_text=new_user_message_text,
            pet_name=self.pet_name,
            user_name=self.user_name,
            pet_persona=self.pet_persona,
            available_emotions=self.available_emotions,
            unified_default_emotion=self.unified_default_emotion,
            mongo_handler=self.mongo_handler,
            hippocampus_manager=hippocampus_manager,
        )
        return [unified_prompt]

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
            if is_agent_mode:
                chat_contents = [message_text]
                logger.info("GeminiClient: Sending Agent Mode prompt directly.")
            else:
                chat_contents = await self._build_chat_contents_for_api(
                    new_user_message_text=message_text,
                    hippocampus_manager=hippocampus_manager,
                )
            if not chat_contents or not chat_contents[0]:
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
                "tools": (
                    self.enabled_tools
                    if not is_agent_mode and self.enabled_tools
                    else None
                ),
            }
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
            screen_analysis_text_prompt = (
                self.prompt_builder.build_screen_analysis_prompt(
                    pet_name=self.pet_name,
                    user_name=self.user_name,
                    available_emotions=self.available_emotions,
                    mongo_handler=self.mongo_handler,
                    unified_default_emotion=self.unified_default_emotion,
                )
            )
            user_parts_for_vision = [types.Part(text=screen_analysis_text_prompt)]
            if prompt_text and prompt_text.strip():
                user_parts_for_vision.append(
                    types.Part(text=f"\n用户补充说明：{prompt_text}")
                )
            user_parts_for_vision.append(image_part_obj)
            contents_for_vision = [
                types.Content(role="user", parts=user_parts_for_vision)
            ]
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
                contents=contents_for_vision,
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

    def _parse_llm_json_output(
        self,
        llm_output_text: str,
        raw_response_object: Any,
        is_agent_mode: bool = False,
    ) -> Dict[str, Any]:
        json_str_to_parse = ""
        try:
            match_json = re.search(
                r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", llm_output_text, re.DOTALL
            )
            if match_json:
                json_str_to_parse = match_json.group(1).strip()
            else:
                json_str_to_parse = llm_output_text.strip()
                if not (
                    json_str_to_parse.startswith("{")
                    and json_str_to_parse.endswith("}")
                ):
                    first_brace = json_str_to_parse.find("{")
                    last_brace = json_str_to_parse.rfind("}")
                    if (
                        first_brace != -1
                        and last_brace != -1
                        and last_brace > first_brace
                    ):
                        json_str_to_parse = json_str_to_parse[
                            first_brace : last_brace + 1
                        ]
                    else:
                        pass
            parsed_data = json.loads(json_str_to_parse, strict=False)
            validated_data = PetResponseSchema(**parsed_data)
            if validated_data.thinking_process:
                logger.debug(
                    f"GeminiClient (标准解析, agent_mode={is_agent_mode}) 自定义思考过程: {validated_data.thinking_process}"
                )
            if is_agent_mode:
                if validated_data.steps is None:
                    validated_data.steps = []
                agent_emotions_config = self.config_manager.config.get(
                    "PET", "AGENT_MODE_EMOTIONS", fallback="neutral,focused,helpful"
                )
                valid_agent_emotions = [
                    e.strip().lower() for e in agent_emotions_config.split(",")
                ]
                if validated_data.emotion not in valid_agent_emotions:
                    logger.warning(
                        f"Agent mode LLM returned emotion '{validated_data.emotion}', "
                        f"not in allowed agent emotions {valid_agent_emotions}. Defaulting to 'neutral'."
                    )
                    validated_data.emotion = "neutral"
            return validated_data.model_dump()
        except (json.JSONDecodeError, TypeError) as e_json:
            logger.warning(
                f"标准JSON解析失败 (agent_mode={is_agent_mode}): {type(e_json).__name__} - {e_json}. "
                f"原始文本片段: '{llm_output_text[:500]}'. 尝试进行正则提取..."
            )
            extracted_data = {}
            text_for_regex = llm_output_text
            thinking_detail_for_error = (
                f"标准JSON解析失败 ({type(e_json).__name__}: {e_json})"
            )
            try:
                match_text = re.search(
                    r'"text"\s*:\s*"(.*?)(?<!\\)"', text_for_regex, re.DOTALL
                )
                if match_text:
                    extracted_data["text"] = json.loads(f'"{match_text.group(1)}"')
                else:
                    extracted_data["text"] = (
                        f"未能从LLM响应中提取到主要文本内容 (原始片段: {text_for_regex[:100]})"
                    )
                match_emotion = re.search(
                    r'"emotion"\s*:\s*"(.*?)(?<!\\)"', text_for_regex
                )
                if match_emotion:
                    extracted_data["emotion"] = json.loads(
                        f'"{match_emotion.group(1)}"'
                    )
                else:
                    extracted_data["emotion"] = self.unified_default_emotion
                match_think = re.search(
                    r'"(?:thinking_process|think)"\s*:\s*"(.*?)(?<!\\)"',
                    text_for_regex,
                    re.DOTALL,
                )
                if match_think:
                    extracted_data["thinking_process"] = json.loads(
                        f'"{match_think.group(1)}"'
                    )
                match_jp = re.search(
                    r'"text_japanese"\s*:\s*"(.*?)(?<!\\)"', text_for_regex, re.DOTALL
                )
                if match_jp:
                    extracted_data["text_japanese"] = json.loads(
                        f'"{match_jp.group(1)}"'
                    )
                if is_agent_mode:
                    extracted_data["steps"] = []
                    match_steps_block = re.search(
                        r'"steps"\s*:\s*(\[[\s\S]*?\])', text_for_regex, re.DOTALL
                    )
                    if match_steps_block:
                        steps_json_str = match_steps_block.group(1)
                        try:
                            parsed_steps = json.loads(steps_json_str)
                            if isinstance(parsed_steps, list) and all(
                                isinstance(s, dict) for s in parsed_steps
                            ):
                                extracted_data["steps"] = parsed_steps
                            else:
                                logger.warning(
                                    "正则提取的steps内容不是预期的列表套字典结构。"
                                )
                        except json.JSONDecodeError as e_steps_json:
                            logger.warning(
                                f"正则提取的steps块无法解析为JSON: {e_steps_json}"
                            )
                if "text" in extracted_data and "emotion" in extracted_data:
                    logger.info(f"正则提取尝试完成，数据: {extracted_data}")
                    validated_data_via_regex = PetResponseSchema(**extracted_data)
                    if validated_data_via_regex.thinking_process:
                        logger.debug(
                            f"GeminiClient (正则提取后, agent_mode={is_agent_mode}) 自定义思考过程: {validated_data_via_regex.thinking_process}"
                        )
                    if is_agent_mode:
                        if validated_data_via_regex.steps is None:
                            validated_data_via_regex.steps = []
                        agent_emotions_config = self.config_manager.config.get(
                            "PET",
                            "AGENT_MODE_EMOTIONS",
                            fallback="neutral,focused,helpful",
                        )
                        valid_agent_emotions = [
                            e.strip().lower() for e in agent_emotions_config.split(",")
                        ]
                        if validated_data_via_regex.emotion not in valid_agent_emotions:
                            logger.warning(
                                f"Agent mode (regex fallback) LLM returned emotion '{validated_data_via_regex.emotion}', "
                                f"not in allowed agent emotions. Defaulting to 'neutral'."
                            )
                            validated_data_via_regex.emotion = "neutral"
                    logger.info(
                        f"通过正则提取并Pydantic验证成功 (agent_mode={is_agent_mode})."
                    )
                    return validated_data_via_regex.model_dump()
                else:
                    thinking_detail_for_error += (
                        ", 正则提取未能获得关键数据 (text/emotion)"
                    )
                    logger.warning(
                        f"正则提取未能获得关键数据 (text/emotion) (agent_mode={is_agent_mode})."
                    )
            except ValidationError as e_regex_val:
                thinking_detail_for_error += f", 正则提取后Pydantic验证失败 ({type(e_regex_val).__name__}: {e_regex_val})"
                logger.warning(
                    f"正则提取的数据未能通过Pydantic验证 (agent_mode={is_agent_mode}): {e_regex_val}. "
                    f"提取的数据: {extracted_data}"
                )
            except Exception as e_regex_internal:
                thinking_detail_for_error += f", 正则提取过程中发生意外错误 ({type(e_regex_internal).__name__}: {e_regex_internal})"
                logger.error(
                    f"正则提取或其后续处理中发生意外错误 (agent_mode={is_agent_mode}): {e_regex_internal}",
                    exc_info=True,
                )
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = (
                f"<think>{thinking_detail_for_error}. "
                f"解析前的原始文本片段: '{llm_output_text[:500]}'. 反馈: {feedback_info}</think>"
            )
            final_fallback_text = extracted_data.get(
                "text", llm_output_text if llm_output_text else "我好像有点混乱..."
            )
            if (
                not final_fallback_text.strip()
                or "未能从LLM响应中提取到主要文本内容" in final_fallback_text
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
                final_fallback_text = "内容被阻止(JSON解析或正则提取失败后检查)."
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
            }
            if is_agent_mode:
                result["steps"] = extracted_data.get("steps", [])
            return result
        except ValidationError as e_val:
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = (
                f"<think>Pydantic验证错误 (初始解析成功后): {type(e_val).__name__} - {e_val}. "
                f"用于解析的JSON字符串: '{json_str_to_parse[:500]}'. 反馈: {feedback_info}</think>"
            )
            logger.warning(
                f"Pydantic验证错误处理LLM输出 (agent_mode={is_agent_mode}): {e_val}. "
                f"JSON字符串: '{json_str_to_parse[:500]}'. 思考: {thinking}"
            )
            result = {
                "text": f"我说的话可能有点奇怪... (原始文本片段: {llm_output_text[:100]})",
                "emotion": (
                    "neutral"
                    if is_agent_mode
                    else (
                        "confused"
                        if "confused" in self.available_emotions
                        else self.unified_default_emotion
                    )
                ),
                "thinking_process": thinking,
                "is_error": True,
            }
            if is_agent_mode:
                result["steps"] = []
            return result
        except Exception as e_other_unexpected:
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = (
                f"<think>解析过程中发生意外错误: {type(e_other_unexpected).__name__} - {e_other_unexpected}. "
                f"原始文本片段: '{llm_output_text[:500]}'. 反馈: {feedback_info}</think>"
            )
            logger.error(
                f"解析LLM输出时发生意外错误 (agent_mode={is_agent_mode}): {e_other_unexpected}. "
                f"原始文本片段: '{llm_output_text[:500]}'. 思考: {thinking}",
                exc_info=True,
            )
            result = {
                "text": f"处理您的消息时发生了非常意外的错误。",
                "emotion": "neutral" if is_agent_mode else self.unified_default_emotion,
                "thinking_process": thinking,
                "is_error": True,
            }
            if is_agent_mode:
                result["steps"] = []
            return result
        except (json.JSONDecodeError, TypeError) as e_json:
            fallback_text = llm_output_text if llm_output_text else "我好像有点混乱..."
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = f"<think>JSON/Validation Error: {e_json}. 解析前的原始文本: '{llm_output_text}'. 反馈: {feedback_info}</think>"
            logger.warning(
                f"JSON/Validation Error 解析LLM输出 (agent_mode={is_agent_mode}): {e_json}. 原始文本: '{llm_output_text}'. 思考: {thinking}"
            )
            if "BlockReason=" in feedback_info and not any(
                reason in feedback_info.upper()
                for reason in ["NONE", "UNSPECIFIED", "STOP"]
            ):
                fallback_text = "内容被阻止(JSON解析错误后检查)."
            result = {
                "text": fallback_text,
                "emotion": (
                    "neutral"
                    if is_agent_mode
                    else (
                        "confused"
                        if "confused" in self.available_emotions
                        else self.unified_default_emotion
                    )
                ),
                "thinking_process": thinking,
                "is_error": True,
            }
            if is_agent_mode:
                result["steps"] = []
            return result
        except Exception as e_val:
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = f"<think>Pydantic验证/其他错误 (例如 Pydantic): {e_val}. 用于解析的原始文本: '{llm_output_text}'. 反馈: {feedback_info}</think>"
            logger.warning(
                f"Pydantic验证/其他错误处理LLM输出 (agent_mode={is_agent_mode}): {e_val}. 原始文本: '{llm_output_text}'. 思考: {thinking}"
            )
            result = {
                "text": f"我说的话可能有点奇怪... (原始文本: {llm_output_text[:100]})",
                "emotion": (
                    "neutral"
                    if is_agent_mode
                    else (
                        "confused"
                        if "confused" in self.available_emotions
                        else self.unified_default_emotion
                    )
                ),
                "thinking_process": thinking,
                "is_error": True,
            }
            if is_agent_mode:
                result["steps"] = []
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
        if raw_response_object:
            feedback_str = self._get_prompt_feedback_info(raw_response_object)
        elif hasattr(e, "response") and e.response:
            feedback_str = self._get_prompt_feedback_info(getattr(e, "response", None))
        if feedback_str and feedback_str not in [
            "提示反馈: (在response_obj中未找到特定的反馈属性)",
            "提示反馈: (无响应对象)",
        ]:
            thinking_on_error += f" 反馈: {feedback_str}"
            if "BlockReason=" in feedback_str:
                match = re.search(r"BlockReason=([^,;]+)", feedback_str)
                if match and match.group(1).strip().upper() not in [
                    "NONE",
                    "BLOCK_REASON_UNSPECIFIED",
                    "STOP",
                    "N/A",
                    "",
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
