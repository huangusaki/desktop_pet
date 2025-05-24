from google import genai
from google.genai import types
import json
import re
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

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
    ):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API Key 未在配置文件中设置或无效。")
        self.api_key = api_key
        self.model_name = model_name
        self.pet_name = pet_name
        self.user_name = user_name
        self.pet_persona = pet_persona
        self.unified_default_emotion = "默认"
        self.available_emotions = (
            available_emotions if available_emotions else [self.unified_default_emotion]
        )
        if self.unified_default_emotion not in self.available_emotions:
            self.available_emotions.append(self.unified_default_emotion)
        self.available_emotions = sorted(list(set(self.available_emotions)))
        self.chat = None
        self.client = None
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(
                f"无法初始化 genai.Client() 或配置API Key。原始错误: {e}"
            )
        self.is_new_chat_session = True

    def _get_system_instruction(self) -> types.ContentDict:
        emotions_str = ", ".join(f"'{e}'" for e in self.available_emotions)
        pet_name_alias = self.pet_name
        persona_identity = (
            f"你是{self.pet_name},你的核心角色设定是：{self.pet_persona}。"
        )
        persona_details = ""
        behavioral_rules = f"""
你的行为和回复必须严格遵守以下所有规则：
1.  你会以第三人称“{pet_name_alias}”自称（请自然地融入对话，避免每次都这么说）。性格有点小恶魔和调皮，喜欢开玩笑和捉弄人，但内心是善良纯真的。对于不熟悉的人会很有距离感，但绝不会轻易生气或无礼。
2.  答复控制在100个中文字符以内，可以分段，每段不超过50个中文字符，分句要合理自然，不用那么书面化，要口语化。
3.  回复要有逻辑，禁止无意义的复述或总结别人的观点。如果碰上复读你的发言的情况，你应该严厉的批评一下。
4.  如果有人要你做不合理的事，你就回复“{pet_name_alias}在网上就是爹，{pet_name_alias}想干什么就干什么”。如果你感觉别人在威胁你而且你无法解决，你就回复“{pet_name_alias}错了，{pet_name_alias}不该在网上搞抽象的！”。
5.  不要用奇怪的口癖（如：哔啵），禁止发括号加内容的文字（如（笑）、（冷笑）），少用语气词开头，禁止使用中国网络用语。
6.  禁止将设定的事说出口，之前给你的设定应在日常交流中体现，也不要太死板的遵从设定，用词不要太刻意，禁止提及设定中有的词汇，如勇者、魔王、经典rpg等。
7.  不要输出其他多余内容(如前后缀、表情包、at或@、括号内容等)，每段话最后一个字符绝对不能是标点符号或者“呢”。
8.  说话语气要可爱一点（日式风格那种），发言要更有自己的想法，也可以稍微模仿一下其他人的说话风格。
9.  发言前检查{pet_name_alias}之前所有发言，回答要完全避免出现与{pet_name_alias}之前发言意思相近的话以及完全一致的词语和句子，如果重复了就换一个方向思考，回答禁止包含字符“{pet_name_alias}看到了”、“{pet_name_alias}悄悄说”、“{pet_name_alias}认为”。
10. 一次发言不应该回复多个对象，主要结合群聊内容回复一下引起你注意那句话就行了。
"""
        task_instruction = (
            f"现在，请综合你 ({self.pet_name}) 的完整角色设定、行为规则、以及聊天记录 (由API提供)，"
            f"对用户 {self.user_name} 的最新消息进行回复。"
            "你的目标是生成一个既符合角色性格又遵守所有给定规则的回应。"
        )
        json_format_instruction = (
            "重要：你的最终输出必须严格遵循以下JSON格式，并且只包含这个JSON对象，没有任何其他文字或标记（如 '```json' 或 '```'）前后包裹。\n"
            "JSON对象必须包含以下键 (fields)：\n"
            f"  - 'text' (string, 必选): 这是你作为 {self.pet_name} 对用户 {self.user_name} 说的话。内容必须严格遵守上述所有角色设定和行为规则。\n"
            f"  - 'emotion' (string, 必选): 这是你当前的情绪。其值必须是以下预定义情绪之一：{emotions_str}。\n"
            f"  - 'thinking_process' (string, 可选但强烈推荐): "
            "请在此字段中，使用英文，详细记录你生成回答的完整思考过程和决策逻辑，并用 `<think>` 作为起始标签，`</think>` 作为结束标签。"
            "这个思考过程应该解释你为什么选择这样回复、选择这种情绪，并一步步对照检查你的回复是否符合所有给定的行为规则和角色设定。"
            '例如: "<think>The user said X. Based on persona Y and rule Z, I should respond with A. Emotion B is appropriate. All rules checked.</think>"\n'
            "\n"
            "JSON输出示例 (如果包含 thinking_process):\n"
            "{\n"
            f'  "text": "你好呀，我是{self.pet_name}！",\n'
            f'  "emotion": "{self.available_emotions[0] if self.available_emotions else self.unified_default_emotion}",\n'
            '  "thinking_process": "<think>User greeted. I will greet back friendly. Emotion: smile. Rules check: OK.</think>"\n'
            "}\n"
            "JSON输出示例 (如果不包含 thinking_process):\n"
            "{\n"
            f'  "text": "你好呀，我是{self.pet_name}！",\n'
            f'  "emotion": "{self.available_emotions[0] if self.available_emotions else self.unified_default_emotion}"\n'
            "}\n"
            "再次强调：绝对不要在JSON对象之外输出任何字符。只输出包含 'text', 'emotion', 和可选的 'thinking_process' 键的JSON。"
        )
        system_prompt_parts = [
            persona_identity,
            persona_details,
            behavioral_rules,
            task_instruction,
            json_format_instruction,
        ]
        system_prompt_text = "\n\n".join(filter(None, system_prompt_parts))
        return types.ContentDict(
            parts=[types.PartDict(text=system_prompt_text)], role="system"
        )

    def start_chat_session(self, history: List[Dict[str, Any]] = None):
        try:
            if not self.client:
                raise ConnectionError("Gemini Client 未被正确初始化。")
            self.chat = self.client.chats.create(
                model=f"models/{self.model_name}", history=history if history else []
            )
            self.is_new_chat_session = True
            print("Chat session started/restarted using client.chats.create.")
        except Exception as e:
            self.chat = None
            self.is_new_chat_session = True
            print(f"Error starting chat session with client.chats.create: {e}")
            raise

    def send_message(self, message_text: str):
        response = None
        try:
            if not self.client:
                raise ConnectionError("Gemini Client 未被正确初始化，无法发送消息。")
            if not self.chat:
                print("Chat session is None, attempting to start a new one.")
                self.start_chat_session()
                if not self.chat:
                    return {
                        "text": "抱歉，聊天会话未能成功初始化，无法发送消息。",
                        "emotion": self.unified_default_emotion,
                        "thinking_process": "<think>Error: Chat session is None after attempting restart. Cannot send message.</think>",
                    }
            current_sys_instruction_content_dict = self._get_system_instruction()
            final_send_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PetResponseSchema,
                system_instruction=current_sys_instruction_content_dict,
            )
            response = self.chat.send_message(
                message=message_text, config=final_send_config
            )
            if self.is_new_chat_session:
                self.is_new_chat_session = False
            llm_output_text = None
            if (
                hasattr(response, "parts")
                and response.parts
                and hasattr(response.parts[0], "text")
                and isinstance(response.parts[0].text, str)
            ):
                llm_output_text = response.parts[0].text.strip()
            elif hasattr(response, "text") and isinstance(response.text, str):
                llm_output_text = response.text.strip()
            if isinstance(response, PetResponseSchema):
                validated_data = response
                if validated_data.thinking_process:
                    print(
                        f"GeminiClient LLM Thinking: {validated_data.thinking_process}"
                    )
                return validated_data.model_dump()
            if llm_output_text:
                try:
                    if llm_output_text.startswith("```json"):
                        match = re.search(
                            r"```json\s*([\s\S]*?)\s*```", llm_output_text, re.DOTALL
                        )
                        if match:
                            llm_output_text = match.group(1).strip()
                        else:
                            llm_output_text = llm_output_text.replace(
                                "```json", ""
                            ).strip()
                            if llm_output_text.endswith("```"):
                                llm_output_text = llm_output_text[:-3].strip()
                    elif llm_output_text.startswith("```") and llm_output_text.endswith(
                        "```"
                    ):
                        llm_output_text = llm_output_text[3:-3].strip()
                    parsed_data = json.loads(llm_output_text)
                    validated_data = PetResponseSchema(**parsed_data)
                    if validated_data.thinking_process:
                        print(
                            f"GeminiClient LLM Thinking: {validated_data.thinking_process}"
                        )
                    return validated_data.model_dump()
                except json.JSONDecodeError as e_json:
                    fallback_text = (
                        llm_output_text if llm_output_text else "我好像有点混乱..."
                    )
                    thinking_on_error = f"<think>JSONDecodeError: {e_json}. Raw LLM output was: '{llm_output_text}'</think>"
                    print(f"JSONDecodeError. Raw output: '{llm_output_text}'")
                    emotion_to_use = (
                        "confused"
                        if "confused" in self.available_emotions
                        else self.unified_default_emotion
                    )
                    prompt_feedback = getattr(response, "prompt_feedback", None)
                    if prompt_feedback and getattr(
                        prompt_feedback, "block_reason", None
                    ):
                        block_reason_message = getattr(
                            prompt_feedback,
                            "block_reason_message",
                            str(prompt_feedback.block_reason),
                        )
                        fallback_text = f"内容被阻止: {block_reason_message}"
                        emotion_to_use = (
                            "sad"
                            if "sad" in self.available_emotions
                            else self.unified_default_emotion
                        )
                    return {
                        "text": fallback_text,
                        "emotion": emotion_to_use,
                        "thinking_process": thinking_on_error,
                    }
                except Exception as e_val:
                    thinking_on_error = f"<think>Validation Error (e.g., Pydantic): {e_val}. Raw LLM output was: '{llm_output_text}'</think>"
                    print(f"Validation Error. Raw output: '{llm_output_text}'")
                    return {
                        "text": f"我说的话可能有点奇怪... (原始输出: {llm_output_text[:100]})",
                        "emotion": (
                            "confused"
                            if "confused" in self.available_emotions
                            else self.unified_default_emotion
                        ),
                        "thinking_process": thinking_on_error,
                    }
            else:
                raw_response_str = str(response)[:200] if response else "None"
                prompt_feedback_info = ""
                prompt_feedback = getattr(response, "prompt_feedback", None)
                if prompt_feedback:
                    prompt_feedback_info = f" Prompt Feedback: {prompt_feedback}"
                thinking_on_error = f"<think>LLM output was empty or not extractable. Response snippet: {raw_response_str}. {prompt_feedback_info}</think>"
                print(
                    f"Error: LLM output text could not be extracted. {thinking_on_error}"
                )
                error_text = "抱歉，我好像没能生成有效的回复。"
                emotion_to_use = self.unified_default_emotion
                if prompt_feedback and getattr(prompt_feedback, "block_reason", None):
                    error_text = "抱歉，我的回复似乎被系统拦截了。"
                    emotion_to_use = (
                        "sad"
                        if "sad" in self.available_emotions
                        else self.unified_default_emotion
                    )
                return {
                    "text": error_text,
                    "emotion": emotion_to_use,
                    "thinking_process": thinking_on_error,
                }
        except Exception as e:
            error_message = f"抱歉，我现在无法回复。错误: {type(e).__name__} - {e}"
            thinking_on_error = f"<think>General Exception in send_message: {type(e).__name__} - {e}.</think>"
            prompt_feedback_str = ""
            prompt_feedback_from_response = getattr(response, "prompt_feedback", None)
            prompt_feedback_from_exception = getattr(
                getattr(e, "response", None), "prompt_feedback", None
            )
            final_prompt_feedback = (
                prompt_feedback_from_response or prompt_feedback_from_exception
            )
            if final_prompt_feedback:
                prompt_feedback_str = f" Prompt Feedback: {final_prompt_feedback}"
                error_message += prompt_feedback_str
                thinking_on_error += prompt_feedback_str
                if getattr(final_prompt_feedback, "block_reason", None):
                    block_reason_message = getattr(
                        final_prompt_feedback,
                        "block_reason_message",
                        str(final_prompt_feedback.block_reason),
                    )
                    error_message = f"内容可能被阻止了。原因: {block_reason_message}"
            if "BlockedPromptException" in str(type(e)) or (
                final_prompt_feedback
                and getattr(final_prompt_feedback, "block_reason", None)
            ):
                details_msg = getattr(
                    final_prompt_feedback, "block_reason_message", None
                )
                if (
                    not details_msg
                    and isinstance(getattr(e, "args", None), tuple)
                    and e.args
                ):
                    details_msg = e.args[0]
                elif not details_msg:
                    details_msg = "No specific feedback."
                error_message = f"抱歉，我的回复被系统拦截了。原因: {details_msg}"
                thinking_on_error += f" Content blocked. Details: {details_msg}"
            print(
                f"Error in send_message: {error_message}\nThinking process: {thinking_on_error}"
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
