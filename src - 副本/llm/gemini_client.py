from google import genai
from google.genai import types
import json
import re
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Optional

EmotionTypes = Literal[
    "default", "smile", "shock", "thinking", "happy", "sad", "confused"
]


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
    ):
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Gemini API Key 未在配置文件中设置或无效。")
        self.api_key = api_key
        self.model_name = model_name
        self.pet_name = pet_name
        self.user_name = user_name
        self.pet_persona = pet_persona
        self.chat = None
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(
                f"无法初始化 genai.Client() 或配置API Key。原始错误: {e}"
            )
        self.is_new_chat_session = self.chat is None

    def _get_system_instruction(self) -> dict:
        available_emotions = ", ".join(f"'{e}'" for e in EmotionTypes.__args__)
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
            f"  - 'emotion' (string, 必选): 这是你当前的情绪。其值必须是以下预定义情绪之一：{available_emotions}。\n"
            f"  - 'thinking_process' (string, 可选但强烈推荐): "
            "请在此字段中，使用英文，详细记录你生成回答的完整思考过程和决策逻辑，并用 `<think>` 作为起始标签，`</think>` 作为结束标签。"
            "这个思考过程应该解释你为什么选择这样回复、选择这种情绪，并一步步对照检查你的回复是否符合所有给定的行为规则和角色设定。"
            '例如: "<think>The user said X. Based on persona Y and rule Z, I should respond with A. Emotion B is appropriate. All rules checked.</think>"\n'
            "\n"
            "JSON输出示例 (如果包含 thinking_process):\n"
            "{\n"
            f'  "text": "你好呀，我是{self.pet_name}！",\n'
            '  "emotion": "smile",\n'
            '  "thinking_process": "<think>User greeted. I will greet back friendly. Emotion: smile. Rules check: OK.</think>"\n'
            "}\n"
            "JSON输出示例 (如果不包含 thinking_process):\n"
            "{\n"
            f'  "text": "你好呀，我是{self.pet_name}！",\n'
            '  "emotion": "smile"\n'
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
        return {"parts": [{"text": system_prompt_text}]}

    def start_chat_session(self, history: List[Dict[str, Any]] = None):
        try:
            self.chat = self.client.chats.create(
                model=self.model_name, history=history if history else None
            )
            self.is_new_chat_session = True
        except Exception as e:
            self.chat = None
            self.is_new_chat_session = True
            raise

    def send_message(self, message_text: str):
        response = None
        try:
            if not self.chat:
                return {
                    "text": "抱歉，聊天会话尚未成功初始化，无法发送消息。",
                    "emotion": "sad",
                    "thinking_process": "<think>Error: Chat session is None. Cannot send message.</think>",
                }
            current_sys_instruction_dict = self._get_system_instruction()
            current_send_config_params = {
                "response_mime_type": "application/json",
                "response_schema": PetResponseSchema,
            }
            if self.is_new_chat_session:
                current_send_config_params["system_instruction"] = (
                    current_sys_instruction_dict
                )
            final_send_config = types.GenerateContentConfig(
                **current_send_config_params
            )
            print(f"最终prompt:{final_send_config}")
            response = self.chat.send_message(
                message=message_text, config=final_send_config
            )
            if self.is_new_chat_session:
                self.is_new_chat_session = False
            if hasattr(response, "parts") and response.parts:
                llm_output_text = response.parts[0].text.strip()
            elif hasattr(response, "text"):
                llm_output_text = response.text.strip()
            else:
                try:
                    raw_response_content = ""
                    if hasattr(response, "_raw_response"):
                        raw_response_content = str(response._raw_response)
                    if not isinstance(llm_output_text, str):
                        if isinstance(response, PetResponseSchema):
                            validated_data = response
                            if validated_data.thinking_process:
                                print(
                                    f"GeminiClient LLM Thinking: {validated_data.thinking_process}"
                                )
                            return validated_data.model_dump()
                        elif (
                            isinstance(response, dict)
                            and "text" in response
                            and "emotion" in response
                        ):
                            validated_data = PetResponseSchema(**response)
                            if validated_data.thinking_process:
                                print(
                                    f"GeminiClient LLM Thinking: {validated_data.thinking_process}"
                                )
                            return validated_data.model_dump()
                        else:
                            raise ValueError(
                                "LLM response is not a string and not a recognized schema object."
                            )
                except Exception as direct_parse_err:
                    print(
                        f"Error trying to interpret LLM response structure: {direct_parse_err}"
                    )
                    llm_output_text = ""
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
                elif llm_output_text.startswith("```") and llm_output_text.endswith(
                    "```"
                ):
                    llm_output_text = llm_output_text[3:-3].strip()
                if isinstance(response, PetResponseSchema):
                    validated_data = response
                elif isinstance(llm_output_text, str) and llm_output_text:
                    parsed_data = json.loads(llm_output_text)
                    validated_data = PetResponseSchema(**parsed_data)
                elif (
                    isinstance(response, dict)
                    and "text" in response
                    and "emotion" in response
                ):
                    validated_data = PetResponseSchema(**response)
                else:
                    if (
                        not llm_output_text
                        and isinstance(response, dict)
                        and "text" in response
                        and "emotion" in response
                    ):
                        validated_data = PetResponseSchema(**response)
                    elif not llm_output_text and isinstance(
                        response, PetResponseSchema
                    ):
                        validated_data = response
                    else:
                        raise json.JSONDecodeError(
                            "LLM output is empty or not a parseable string, and response object is not a direct schema match.",
                            llm_output_text or "N/A",
                            0,
                        )
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
                if (
                    not ("{" in llm_output_text and "}" in llm_output_text)
                    and len(llm_output_text) < 200
                ):
                    return {
                        "text": fallback_text,
                        "emotion": "confused",
                        "thinking_process": thinking_on_error,
                    }
                return {
                    "text": f"{fallback_text} (我没能正确组织我的想法，原始输出: {llm_output_text[:100]})",
                    "emotion": "confused",
                    "thinking_process": thinking_on_error,
                }
            except Exception as e_val:
                thinking_on_error = f"<think>Validation Error (e.g., Pydantic): {e_val}. Raw LLM output was: '{llm_output_text}'</think>"
                print(f"Validation Error. Raw output: '{llm_output_text}'")
                return {
                    "text": f"我说的话可能有点奇怪... (原始输出: {llm_output_text[:100]})",
                    "emotion": "confused",
                    "thinking_process": thinking_on_error,
                }
        except Exception as e:
            error_message = f"抱歉，我现在无法回复。错误: {type(e).__name__} - {e}"
            thinking_on_error = f"<think>General Exception in send_message: {type(e).__name__} - {e}.</think>"
            if (
                response
                and hasattr(response, "prompt_feedback")
                and response.prompt_feedback
            ):
                error_message += f" Prompt Feedback: {response.prompt_feedback}"
                thinking_on_error += f" Prompt Feedback: {response.prompt_feedback}"
            elif (
                hasattr(e, "response")
                and hasattr(e.response, "prompt_feedback")
                and e.response.prompt_feedback
            ):
                error_message += f" Prompt Feedback: {e.response.prompt_feedback}"
                thinking_on_error += f" Prompt Feedback: {e.response.prompt_feedback}"
            elif "BlockedPromptException" in str(type(e)):
                details = getattr(e, "args", ["No specific feedback available."])
                error_message += f" Content blocked. Details: {details}"
                thinking_on_error += f" Content blocked. Details: {details}"
            print(f"Error in send_message: {error_message}")
            return {
                "text": error_message,
                "emotion": "sad",
                "thinking_process": thinking_on_error,
            }
