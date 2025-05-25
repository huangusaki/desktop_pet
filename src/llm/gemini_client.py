from google import genai
from google.genai import types
import json
import re
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

try:
    from ..utils.prompt_builder import PromptBuilder
    from ..memory_system.hippocampus_core import HippocampusManager
    from ..utils.config_manager import ConfigManager
except ImportError:
    from utils.prompt_builder import PromptBuilder

    class HippocampusManager:
        pass

    class ConfigManager:
        pass


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
        self.chat_session = None
        self.client = None
        try:
            self.client = genai.Client(api_key=self.api_key)
            print(
                f"GeminiClient: genai.Client initialized successfully for model '{self.model_name}'."
            )
        except Exception as e:
            raise ConnectionError(f"无法初始化 genai.Client(). 原始错误: {e}")
        self.is_new_chat_session = True

    def _get_chat_system_instruction_text(self) -> str:
        return self.prompt_builder.build_chat_system_instruction(
            pet_name=self.pet_name,
            user_name=self.user_name,
            pet_persona=self.pet_persona,
            available_emotions=self.available_emotions,
            unified_default_emotion=self.unified_default_emotion,
        )

    def _fetch_and_prepare_sdk_history(self) -> list:
        """
        Fetches chat history from the database and prepares it for the Gemini SDK.
        Moved from ChatDialog._get_cleaned_gemini_sdk_history_from_db
        """
        sdk_formatted_history = []
        if not (self.mongo_handler and self.mongo_handler.is_connected()):
            print("GeminiClient: Mongo handler not connected, cannot fetch history.")
            return sdk_formatted_history
        prompt_history_count = self.config_manager.get_history_count_for_prompt()
        raw_db_history = self.mongo_handler.get_recent_chat_history(
            count=prompt_history_count,
            role_play_character=self.pet_name,
        )
        if not raw_db_history:
            return sdk_formatted_history
        temp_history = []
        for msg_entry in raw_db_history:
            sender_val = msg_entry.get("sender")
            role = (
                "user"
                if isinstance(sender_val, str) and sender_val.lower() == "user"
                else "model"
            )
            text_content = msg_entry.get("message_text", "")
            if text_content:
                temp_history.append({"role": role, "text": text_content})
        if not temp_history:
            return sdk_formatted_history
        current_merged_text = ""
        current_role = None
        for msg in temp_history:
            role, text = msg["role"], msg["text"]
            if current_role is None:
                current_role = role
                current_merged_text = text
            elif role == current_role:
                current_merged_text += "\n" + text
            else:
                if current_merged_text:
                    sdk_formatted_history.append(
                        {"role": current_role, "parts": [{"text": current_merged_text}]}
                    )
                current_role = role
                current_merged_text = text
        if current_role and current_merged_text:
            sdk_formatted_history.append(
                {"role": current_role, "parts": [{"text": current_merged_text}]}
            )
        return sdk_formatted_history

    def start_chat_session(self):
        """
        Starts a new chat session.
        System instruction and history are set by assigning to chat_session.history.
        History is now fetched internally.
        """
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            self.chat_session = self.client.chats.create(model=self.model_name)
            print(f"Chat session created with model: {self.model_name}")
            system_instruction_text = self._get_chat_system_instruction_text()
            system_content = types.Content(
                role="system", parts=[types.Part(text=system_instruction_text)]
            )
            full_history_for_api = [system_content]
            sdk_history = self._fetch_and_prepare_sdk_history()
            if sdk_history:
                for msg_dict in sdk_history:
                    parts_list = []
                    if "parts" in msg_dict and isinstance(msg_dict["parts"], list):
                        for part_item_data in msg_dict["parts"]:
                            if (
                                isinstance(part_item_data, dict)
                                and "text" in part_item_data
                            ):
                                parts_list.append(
                                    types.Part(text=str(part_item_data["text"]))
                                )
                            elif isinstance(part_item_data, str):
                                parts_list.append(types.Part(text=part_item_data))
                    if parts_list:
                        role = msg_dict.get("role")
                        if role in ["user", "model"]:
                            full_history_for_api.append(
                                types.Content(role=role, parts=parts_list)
                            )
                        else:
                            print(
                                f"Warning: Skipping history item with invalid role: {role}"
                            )
            if full_history_for_api:
                self.chat_session.history = full_history_for_api
                print(
                    f"Chat session history set with {len(full_history_for_api)} items (incl. system instruction and {len(sdk_history)} db history turns)."
                )
            else:
                print(
                    "Warning: Chat history is empty after preparation, even system prompt missing."
                )
            self.is_new_chat_session = False
        except Exception as e:
            self.chat_session = None
            print(f"Error starting chat session or setting history: {e}")
            raise

    def send_message(self, message_text: str) -> Dict[str, Any]:
        """Sends a message in an ongoing chat session."""
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            if not self.chat_session:
                return {
                    "text": "抱歉，聊天会话无效，无法发送消息。",
                    "emotion": self.unified_default_emotion,
                    "thinking_process": "<think>Error: Chat session is None when trying to send message. Upstream init likely failed or was not called.</think>",
                }
            response_object = self.chat_session.send_message(
                message=message_text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PetResponseSchema,
                    temperature=0.75,
                ),
            )
            if isinstance(response_object, PetResponseSchema):
                validated_data = response_object
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (Chat) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            llm_output_text = getattr(response_object, "text", None)
            if (
                not llm_output_text
                and hasattr(response_object, "parts")
                and response_object.parts
            ):
                llm_output_text = getattr(response_object.parts[0], "text", None)
            if llm_output_text:
                return self._parse_llm_json_output(
                    llm_output_text.strip(), response_object
                )
            else:
                return self._handle_empty_or_unparseable_response(
                    response_object, "Chat"
                )
        except Exception as e:
            return self._handle_general_exception(
                e, "send_message (chat)", response_object
            )

    def send_message_with_image(
        self, image_bytes: bytes, mime_type: str, prompt_text: str
    ) -> Dict[str, Any]:
        response_object = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client (genai.Client) 未被正确初始化。")
            try:
                image_blob = types.Blob(data=image_bytes, mime_type=mime_type)
                image_part = types.Part(inline_data=image_blob)
                print(
                    f"GeminiClient: Successfully created image_part with types.Blob. Mime type: {mime_type}, Data length: {len(image_bytes)}"
                )
            except Exception as e_part:
                print(
                    f"GeminiClient: Error creating types.Part or types.Blob: {e_part}"
                )
                raise ConnectionError(
                    f"Failed to create image part for Gemini: {e_part}"
                ) from e_part
            response_object = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt_text, image_part],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PetResponseSchema,
                    temperature=0.75,
                ),
            )
            if isinstance(response_object, PetResponseSchema):
                validated_data = response_object
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient (Image) LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            llm_output_text = getattr(response_object, "text", None)
            if (
                not llm_output_text
                and hasattr(response_object, "parts")
                and response_object.parts
            ):
                llm_output_text = getattr(response_object.parts[0], "text", None)
            if llm_output_text:
                return self._parse_llm_json_output(
                    llm_output_text.strip(), response_object
                )
            else:
                return self._handle_empty_or_unparseable_response(
                    response_object, "Image Analysis"
                )
        except Exception as e:
            return self._handle_general_exception(
                e, "send_message_with_image", response_object
            )

    def _parse_llm_json_output(
        self, llm_output_text: str, raw_response_object: Any
    ) -> Dict[str, Any]:
        try:
            if llm_output_text.startswith("```json"):
                match = re.search(
                    r"```json\s*([\s\S]*?)\s*```", llm_output_text, re.DOTALL
                )
                if match:
                    llm_output_text = match.group(1).strip()
                else:
                    llm_output_text = llm_output_text.replace("```json", "").strip()
                    if llm_output_text.endswith("```"):
                        llm_output_text = llm_output_text[:-3].strip()
            elif llm_output_text.startswith("```") and llm_output_text.endswith("```"):
                llm_output_text = llm_output_text[3:-3].strip()
            parsed_data = json.loads(llm_output_text)
            validated_data = PetResponseSchema(**parsed_data)
            if validated_data.thinking_process:
                print(
                    f"GeminiClient (Parsed) LLM Thinking: {validated_data.thinking_process}"
                )
            return validated_data.model_dump()
        except json.JSONDecodeError as e_json:
            fallback_text = llm_output_text if llm_output_text else "我好像有点混乱..."
            thinking = f"<think>JSONDecodeError: {e_json}. Raw: '{llm_output_text}' {self._get_prompt_feedback_info(raw_response_object)}</think>"
            if "block_reason" in thinking:
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
            thinking = f"<think>Validation Error: {e_val}. Raw: '{llm_output_text}' {self._get_prompt_feedback_info(raw_response_object)}</think>"
            return {
                "text": f"我说的话可能有点奇怪... (原始: {llm_output_text[:100]})",
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
        thinking_on_error = f"<think>{context_str} response was not PetResponseSchema and no text found. Snippet: {raw_response_snippet}. {prompt_feedback_info}</think>"
        print(f"Error: {context_str} LLM output issue. {thinking_on_error}")
        error_text = f"抱歉，我好像没能生成有效的{context_str}回复。"
        if (
            "block_reason" in prompt_feedback_info
            and prompt_feedback_info.strip().lower()
            != "prompt feedback: blockreason=none, msg=''"
        ):
            error_text = f"抱歉，我的{context_str}回复似乎被系统拦截了。"
        return {
            "text": error_text,
            "emotion": self.unified_default_emotion,
            "thinking_process": thinking_on_error,
        }

    def _get_prompt_feedback_info(self, response_obj: Any) -> str:
        if not response_obj:
            return ""
        prompt_feedback = getattr(response_obj, "prompt_feedback", None)
        if not prompt_feedback:
            if hasattr(response_obj, "candidates") and response_obj.candidates:
                candidate_feedback = getattr(
                    response_obj.candidates[0], "finish_reason", None
                )
                safety_ratings = getattr(
                    response_obj.candidates[0], "safety_ratings", None
                )
                if candidate_feedback and str(candidate_feedback).upper() != "STOP":
                    return f"Prompt Feedback: FinishReason={candidate_feedback}, SafetyRatings={safety_ratings}"
            return ""
        block_reason = getattr(prompt_feedback, "block_reason", None)
        block_message = getattr(prompt_feedback, "block_reason_message", "")
        safety_ratings_list = getattr(prompt_feedback, "safety_ratings", [])
        safety_info = []
        if safety_ratings_list:
            for rating in safety_ratings_list:
                safety_info.append(f"{rating.category.name}: {rating.probability.name}")
        return f"Prompt Feedback: BlockReason={block_reason}, Msg='{block_message}', Safety=[{', '.join(safety_info)}]"

    def _handle_general_exception(
        self, e: Exception, context: str, raw_response_object: Any = None
    ) -> Dict[str, Any]:
        error_message = f"抱歉，我现在无法回复 ({context})。错误: {type(e).__name__}"
        details_from_exception = str(e)
        thinking_on_error = f"<think>General Exception in {context}: {type(e).__name__} - {details_from_exception}."
        feedback_source = None
        if hasattr(e, "response") and hasattr(e.response, "prompt_feedback"):
            feedback_source = e.response.prompt_feedback
        elif hasattr(e, "prompt_feedback"):
            feedback_source = e.prompt_feedback
        elif raw_response_object:
            feedback_source = getattr(raw_response_object, "prompt_feedback", None)
            if (
                not feedback_source
                and hasattr(raw_response_object, "candidates")
                and raw_response_object.candidates
            ):
                candidate_feedback_reason = getattr(
                    raw_response_object.candidates[0], "finish_reason", None
                )
                if (
                    candidate_feedback_reason
                    and str(candidate_feedback_reason).upper() == "SAFETY"
                ):
                    thinking_on_error += " Candidate finish_reason: SAFETY."
                    error_message = f"抱歉，我的回复 ({context}) 因安全原因被拦截了。"
        final_prompt_feedback_str = ""
        if feedback_source:
            block_reason = getattr(feedback_source, "block_reason", None)
            block_message = getattr(
                feedback_source, "block_reason_message", "No specific message."
            )
            final_prompt_feedback_str = (
                f" Prompt Feedback: BlockReason={block_reason}, Msg='{block_message}'"
            )
            if (
                block_reason
                and str(block_reason).upper() != "PROMPT_BLOCK_REASON_UNSPECIFIED"
                and str(block_reason).upper() != "NONE"
            ):
                error_message = f"抱歉，我的回复 ({context}) 被系统拦截了。原因: {block_message or block_reason}"
        thinking_on_error += final_prompt_feedback_str
        if "BlockedPromptException" in str(type(e)):
            if not (
                feedback_source
                and block_reason
                and str(block_reason).upper() != "PROMPT_BLOCK_REASON_UNSPECIFIED"
            ):
                details_msg = (
                    getattr(e, "args")[0]
                    if getattr(e, "args")
                    else "Content policy violation."
                )
                error_message = (
                    f"抱歉，我的回复 ({context}) 被系统拦截了。原因: {details_msg}"
                )
                thinking_on_error += (
                    f" Content blocked. Details from exception: {details_msg}"
                )
        print(
            f"Error in {context}: {error_message}\nOriginal Exception details: {details_from_exception}\nThinking: {thinking_on_error}"
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
