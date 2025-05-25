from google import genai
from google.genai import types
import json
import re
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional
from ..utils.prompt_builder import PromptBuilder

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
    ):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API Key 未在配置文件中设置或无效。")
        self.api_key = api_key
        self.model_name = model_name
        self.pet_name = pet_name
        self.prompt_builder = prompt_builder
        self.user_name = user_name
        self.pet_persona = pet_persona
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

    def start_chat_session(self, history: List[Dict[str, Any]] = None):
        """
        Starts a new chat session.
        System instruction and history are set by assigning to chat_session.history.
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
            if history:
                for msg_dict in history:
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
                    f"Chat session history set with {len(full_history_for_api)} items (incl. system instruction)."
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
                print(
                    "Chat session is None, attempting to start a new one for send_message."
                )
                return {
                    "text": "抱歉，聊天会话无效，无法发送消息。",
                    "emotion": self.unified_default_emotion,
                    "thinking_process": "<think>Error: Chat session is None when trying to send message. Upstream init likely failed.</think>",
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
        if "block_reason" in prompt_feedback_info:
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
            return ""
        block_reason = getattr(prompt_feedback, "block_reason", None)
        block_message = getattr(prompt_feedback, "block_reason_message", "")
        return f"Prompt Feedback: BlockReason={block_reason}, Msg='{block_message}'"

    def _handle_general_exception(
        self, e: Exception, context: str, raw_response_object: Any = None
    ) -> Dict[str, Any]:
        error_message = f"抱歉，我现在无法回复 ({context})。错误: {type(e).__name__}"
        details_from_exception = str(e)
        thinking_on_error = f"<think>General Exception in {context}: {type(e).__name__} - {details_from_exception}."
        feedback_source = (
            getattr(e, "prompt_feedback", None)
            or (hasattr(e, "response") and getattr(e.response, "prompt_feedback", None))
            or getattr(raw_response_object, "prompt_feedback", None)
        )
        final_prompt_feedback_str = ""
        if feedback_source:
            block_reason = getattr(feedback_source, "block_reason", None)
            block_message = getattr(
                feedback_source, "block_reason_message", "No specific message."
            )
            final_prompt_feedback_str = (
                f" Prompt Feedback: BlockReason={block_reason}, Msg='{block_message}'"
            )
            if block_reason:
                error_message = f"抱歉，我的回复 ({context}) 被系统拦截了。原因: {block_message or block_reason}"
        thinking_on_error += final_prompt_feedback_str
        if "BlockedPromptException" in str(type(e)) or (
            "block_reason" in final_prompt_feedback_str.lower() and block_reason
        ):
            if not final_prompt_feedback_str or not block_reason:
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
