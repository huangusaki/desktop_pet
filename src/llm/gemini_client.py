from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import json
import re
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

try:
    from ..utils.prompt_builder import PromptBuilder
    from ..utils.config_manager import ConfigManager
except ImportError:

    class PromptBuilder:
        def __init__(self, config_manager):
            self.config_manager = config_manager

        def build_screen_analysis_prompt(self, pet_name, user_name, available_emotions):
            return "Placeholder screen analysis prompt"

    class ConfigManager:
        def get_history_count_for_prompt(self):
            return 10

        def get_user_name(self):
            return "User"

        def get_pet_name(self):
            return "Pet"

        def get_screen_analysis_prompt(self):
            return "分析这张关于{user_name}屏幕的图片，作为{pet_name}，你的情绪可以是{available_emotions_str}，回复必须是JSON。"


EmotionTypes = str


class PetResponseSchema(BaseModel):
    text: str = Field(..., description="宠物说的话")
    emotion: EmotionTypes = Field(..., description="宠物当前的情绪")
    thinking_process: Optional[str] = Field(
        None,
        description='模型的思考过程，使用英文，在<think>和</think>标签之间。例如: "<think>User said X...</think>"',
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
        self.client = None
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(f"无法初始化 genai.Client(). 原始错误: {e}")
        self.enabled_tools = []
        self.enabled_tools.append(Tool(url_context=types.UrlContext))
        self.enabled_tools.append(Tool(google_search=types.GoogleSearch))
        self.enabled_tools.append(Tool(code_execution=types.ToolCodeExecution))

    def _build_chat_contents_for_api(self, new_user_message_text: str) -> List[str]:
        """
        为API调用构建 `contents` 列表。
        现在调用 PromptBuilder 来获取完整的提示字符串。
        """
        unified_prompt = self.prompt_builder.build_unified_chat_prompt_string(
            new_user_message_text=new_user_message_text,
            pet_name=self.pet_name,
            user_name=self.user_name,
            pet_persona=self.pet_persona,
            available_emotions=self.available_emotions,
            unified_default_emotion=self.unified_default_emotion,
            mongo_handler=self.mongo_handler,
        )
        return [unified_prompt]

    def send_message(self, message_text: str) -> Dict[str, Any]:
        """Sends a message using client.models.generate_content with a unified prompt string."""
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            chat_contents = self._build_chat_contents_for_api(
                new_user_message_text=message_text
            )
            if not chat_contents or not chat_contents[0]:
                error_msg = (
                    "Failed to construct valid unified prompt string for API call."
                )
                print(f"GeminiClient: Error - {error_msg}. Contents: {chat_contents}")
                return {
                    "text": "抱歉，内部构建消息时出错。",
                    "emotion": self.unified_default_emotion,
                    "thinking_process": f"<think>Error: {error_msg}</think>",
                }
            generation_config_args = {"temperature": 0.75, "tools": self.enabled_tools}
            api_config = types.GenerateContentConfig(**generation_config_args)
            response_object = self.client.models.generate_content(
                model=self.model_name, contents=chat_contents, config=api_config
            )
            if isinstance(response_object, PetResponseSchema):
                validated_data = response_object
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (Direct Pydantic from SDK) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            if hasattr(response_object, "parsed") and isinstance(
                response_object.parsed, PetResponseSchema
            ):
                validated_data = response_object.parsed
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (from response.parsed) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            llm_output_text = getattr(response_object, "text", None)
            if (
                not llm_output_text
                and hasattr(response_object, "parts")
                and response_object.parts
            ):
                if response_object.parts and isinstance(
                    response_object.parts[0], types.Part
                ):
                    llm_output_text = getattr(response_object.parts[0], "text", None)
                elif response_object.parts and isinstance(
                    response_object.parts[0], dict
                ):
                    llm_output_text = response_object.parts[0].get("text")
            if llm_output_text:
                return self._parse_llm_json_output(
                    llm_output_text.strip(), response_object
                )
            else:
                if (
                    hasattr(response_object, "candidates")
                    and response_object.candidates
                ):
                    candidate = response_object.candidates[0]
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        if isinstance(candidate.content.parts[0], types.Part):
                            llm_output_text = getattr(
                                candidate.content.parts[0], "text", None
                            )
                        elif isinstance(candidate.content.parts[0], dict):
                            llm_output_text = candidate.content.parts[0].get("text")
                        if llm_output_text:
                            print(
                                "GeminiClient: Extracted text from response_object.candidates[0].content.parts[0].text"
                            )
                            return self._parse_llm_json_output(
                                llm_output_text.strip(), response_object
                            )
                return self._handle_empty_or_unparseable_response(
                    response_object, "Chat (Unified String Mode)"
                )
        except Exception as e:
            print(f"严重错误:{e}")
            return self._handle_general_exception(
                e, "send_message (Unified String Mode)", response_object
            )

    def send_message_with_image(
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
            print(
                "\n>>> [GeminiClient.send_message_with_image] PROMPT DETAILS (Multimodal Standard) <<<"
            )
            log_str = ""
            for i, content_item in enumerate(contents_for_vision):
                role = getattr(content_item, "role", "unknown_role")
                log_str += f"  Item {i} (in contents list): Role: {role}\n"
                parts = getattr(content_item, "parts", [])
                if not parts:
                    log_str += "    (No parts)\n"
                for j, part_item in enumerate(parts):
                    if hasattr(part_item, "text") and part_item.text:
                        text_to_log = (
                            part_item.text[:200] + "..."
                            if len(part_item.text) > 200
                            else part_item.text
                        )
                        log_str += f"    Part {j} (text): '{text_to_log}'\n"
                    elif hasattr(part_item, "inline_data") and part_item.inline_data:
                        log_str += f"    Part {j} (inline_data): mime_type='{part_item.inline_data.mime_type}', data_length={len(part_item.inline_data.data)}\n"
            print(log_str.strip())
            print(">>> END PROMPT DETAILS (Multimodal Standard) <<<\n")
            vision_config_args = {"temperature": 0.80, "tools": self.enabled_tools}
            api_vision_config = types.GenerateContentConfig(**vision_config_args)
            response_object = self.client.models.generate_content(
                model=self.model_name,
                contents=contents_for_vision,
                config=api_vision_config,
            )
            if isinstance(response_object, PetResponseSchema):
                validated_data = response_object
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (Image Direct Pydantic) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            if hasattr(response_object, "parsed") and isinstance(
                response_object.parsed, PetResponseSchema
            ):
                validated_data = response_object.parsed
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (Image from response.parsed) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            llm_output_text = getattr(response_object, "text", None)
            if (
                not llm_output_text
                and hasattr(response_object, "parts")
                and response_object.parts
            ):
                if response_object.parts and isinstance(
                    response_object.parts[0], types.Part
                ):
                    llm_output_text = getattr(response_object.parts[0], "text", None)
                elif response_object.parts and isinstance(
                    response_object.parts[0], dict
                ):
                    llm_output_text = response_object.parts[0].get("text")
            if llm_output_text:
                return self._parse_llm_json_output(
                    llm_output_text.strip(), response_object
                )
            else:
                if (
                    hasattr(response_object, "candidates")
                    and response_object.candidates
                ):
                    candidate = response_object.candidates[0]
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        if isinstance(candidate.content.parts[0], types.Part):
                            llm_output_text = getattr(
                                candidate.content.parts[0], "text", None
                            )
                        elif isinstance(candidate.content.parts[0], dict):
                            llm_output_text = candidate.content.parts[0].get("text")
                        if llm_output_text:
                            print(
                                "GeminiClient (Image): Extracted text from response_object.candidates[0].content.parts[0].text"
                            )
                            return self._parse_llm_json_output(
                                llm_output_text.strip(), response_object
                            )
                return self._handle_empty_or_unparseable_response(
                    response_object, "Image Analysis (Multimodal Standard)"
                )
        except Exception as e:
            return self._handle_general_exception(
                e, "send_message_with_image (Multimodal Standard)", response_object
            )

    def _parse_llm_json_output(
        self, llm_output_text: str, raw_response_object: Any
    ) -> Dict[str, Any]:
        try:
            match_json = re.search(
                r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", llm_output_text, re.DOTALL
            )
            if match_json:
                llm_output_text = match_json.group(1).strip()
            else:
                llm_output_text = llm_output_text.strip()
            parsed_data = json.loads(llm_output_text)
            validated_data = PetResponseSchema(**parsed_data)
            if validated_data.thinking_process:
                print(
                    f"GeminiClient (Parsed from text) LLM Thinking: {validated_data.thinking_process}"
                )
            return validated_data.model_dump()
        except json.JSONDecodeError as e_json:
            fallback_text = llm_output_text if llm_output_text else "我好像有点混乱..."
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = f"<think>JSONDecodeError: {e_json}. Raw text after regex: '{llm_output_text}'. Original raw response text (if available): '{getattr(raw_response_object, 'text', 'N/A')}'. Feedback: {feedback_info}</think>"
            if "BlockReason=" in feedback_info and not any(
                reason in feedback_info.upper()
                for reason in ["NONE", "UNSPECIFIED", "STOP"]
            ):
                fallback_text = "内容被阻止(JSON解析错误后检查)."
            return {
                "text": fallback_text,
                "emotion": (
                    "confused"
                    if "confused" in self.available_emotions
                    else self.unified_default_emotion
                ),
                "thinking_process": thinking,
            }
        except Exception as e_val:
            feedback_info = self._get_prompt_feedback_info(raw_response_object)
            thinking = f"<think>Validation Error (e.g., Pydantic): {e_val}. Raw text used for parsing: '{llm_output_text}'. Feedback: {feedback_info}</think>"
            return {
                "text": f"我说的话可能有点奇怪... (原始文本: {llm_output_text[:100]})",
                "emotion": (
                    "confused"
                    if "confused" in self.available_emotions
                    else self.unified_default_emotion
                ),
                "thinking_process": thinking,
            }

    def _handle_empty_or_unparseable_response(
        self, response_object: Any, context_str: str
    ) -> Dict[str, Any]:
        raw_response_snippet = str(response_object)[:200] if response_object else "None"
        prompt_feedback_info = self._get_prompt_feedback_info(response_object)
        thinking_on_error = f"<think>{context_str} response was not PetResponseSchema instance and no text/parsed found. Snippet: {raw_response_snippet}. {prompt_feedback_info}</think>"
        print(f"Error: {context_str} LLM output issue. {thinking_on_error}")
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
        return {
            "text": error_text,
            "emotion": self.unified_default_emotion,
            "thinking_process": thinking_on_error,
        }

    def _get_prompt_feedback_info(self, response_obj: Any) -> str:
        if not response_obj:
            return "Prompt Feedback: (No response object)"
        all_feedback_parts = []
        prompt_feedback = getattr(response_obj, "prompt_feedback", None)
        if prompt_feedback:
            block_reason_obj = getattr(prompt_feedback, "block_reason", None)
            block_reason_str = str(
                getattr(block_reason_obj, "name", block_reason_obj)
                if block_reason_obj is not None
                else "N/A"
            )
            safety_ratings_list = getattr(prompt_feedback, "safety_ratings", [])
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
                cand_safety_ratings = getattr(candidate, "safety_ratings", [])
                cand_safety_info = [
                    f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                    for r in cand_safety_ratings
                ]
                all_feedback_parts.append(
                    f"Candidate[{i}]:FinishReason={finish_reason_str},SafetyRatings=[{','.join(cand_safety_info)}]"
                )
        if not all_feedback_parts:
            return "Prompt Feedback: (No specific feedback attributes found in response_obj)"
        return "; ".join(all_feedback_parts)

    def _handle_general_exception(
        self, e: Exception, context: str, raw_response_object: Any = None
    ) -> Dict[str, Any]:
        error_message = f"抱歉，我现在无法回复 ({context})。错误: {type(e).__name__}"
        details_from_exception = str(e)
        thinking_on_error = f"<think>General Exception in {context}: {type(e).__name__} - {details_from_exception}."
        feedback_str = ""
        if isinstance(e, (types.BlockedPromptException, types.StopCandidateException)):
            feedback_str = self._get_prompt_feedback_info(e)
        elif raw_response_object:
            feedback_str = self._get_prompt_feedback_info(raw_response_object)
        if feedback_str and feedback_str not in [
            "Prompt Feedback: (No specific feedback attributes found in response_obj)",
            "Prompt Feedback: (No response object)",
        ]:
            thinking_on_error += f" Feedback: {feedback_str}"
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
        print(
            f"Error in {context}: {error_message}\nOriginal Exception details ({type(e).__name__}): {details_from_exception}\nThinking: {thinking_on_error}"
        )
        return {
            "text": error_message,
            "emotion": (
                "sad"
                if "sad" in self.available_emotions
                else self.unified_default_emotion
            ),
            "thinking_process": thinking_on_error,
        }
