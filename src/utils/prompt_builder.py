from typing import List, Dict, Any, Tuple, Optional
from ..utils.config_manager import ConfigManager
import logging
import datetime

logger = logging.getLogger("PromptBuilder")


class PromptBuilder:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.available_tones = self.config_manager.get_tts_available_tones()
        self.default_tone = self.config_manager.get_tts_default_tone()
        self.pet_name=self.config_manager.get_pet_name()
        self.user_name=self.config_manager.get_user_name()

    def _get_formatted_screen_analysis_log_content(
        self,
        mongo_handler: Any,
        pet_name: str,
        read_from_main_chat_history: bool,
        count: int = 10,
    ) -> str:
        log_lines = []
        if not (
            mongo_handler
            and hasattr(mongo_handler, "is_connected")
            and mongo_handler.is_connected()
        ):
            return "(数据库未连接或不可用)"
        raw_logs: List[Dict[str, Any]] = []
        if read_from_main_chat_history:
            if hasattr(mongo_handler, "get_recent_chat_history"):
                raw_logs = mongo_handler.get_recent_chat_history(
                    count=count, role_play_character=pet_name
                )
            else:
                logger.warning("MongoHandler missing get_recent_chat_history method.")
        else:
            if hasattr(mongo_handler, "get_recent_screen_analysis_log"):
                raw_logs = mongo_handler.get_recent_screen_analysis_log(
                    count=count, role_play_character=pet_name
                )
            else:
                logger.warning(
                    "MongoHandler missing get_recent_screen_analysis_log method."
                )
        if raw_logs:
            for log_entry in raw_logs:
                text_content = log_entry.get("message_text", "")
                if text_content:
                    if read_from_main_chat_history:
                        if log_entry.get("sender") == pet_name:
                            log_lines.append(f"- {text_content}")
                    else:
                        log_lines.append(f"- {text_content}")
        if log_lines:
            return "\n".join(log_lines)
        else:
            return "(最近没有相关记录)"

    def _get_formatted_chat_history_content(
        self,
        mongo_handler: Any,
        pet_name: str,
        user_name: str,
    ) -> str:
        history_lines = []
        if (
            mongo_handler
            and hasattr(mongo_handler, "is_connected")
            and mongo_handler.is_connected()
        ):
            prompt_history_count = self.config_manager.get_history_count_for_prompt()
            raw_db_history = []
            if hasattr(mongo_handler, "get_recent_chat_history"):
                raw_db_history = (
                    mongo_handler.get_recent_chat_history(
                        count=prompt_history_count,
                        role_play_character=pet_name,
                    )
                    or []
                )
            if raw_db_history:
                for msg_entry in raw_db_history:
                    sender_val = msg_entry.get("sender")
                    text_content = msg_entry.get("message_text", "")
                    timestamp_unix = msg_entry.get("timestamp")
                    if timestamp_unix:
                        try:
                            dt_object = datetime.datetime.fromtimestamp(
                                float(timestamp_unix)
                            )
                            formatted_time_prefix = (
                                dt_object.strftime("%Y-%m-%d %H:%M") + " "
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"格式化聊天记录时间戳 '{timestamp_unix}' 时出错: {e}"
                            )
                    speaker_prefix = ""
                    if sender_val == user_name:
                        speaker_prefix = f"{user_name}: "
                    elif sender_val == pet_name:
                        speaker_prefix = f"{pet_name}: "
                    elif sender_val and isinstance(sender_val, str):
                        speaker_prefix = f"{sender_val}: "
                    if text_content and speaker_prefix:
                        history_lines.append(
                            f"{formatted_time_prefix}{speaker_prefix}{text_content}"
                        )
        if history_lines:
            return "\n".join(history_lines)
        else:
            return "(没有找到相关的对话历史)"

    async def build_unified_chat_prompt_string(
        self,
        new_user_message_text: str,
        pet_name: str,
        user_name: str,
        pet_persona: str,
        available_emotions: List[str],
        unified_default_emotion: str,
        mongo_handler: Any,
        hippocampus_manager: Optional[Any] = None,
        is_multimodal_request: bool = False,
    ) -> str:
        emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        tones_str = ", ".join(f"'{t}'" for t in self.available_tones)
        persona_identity = f"你是一名专业的角色扮演专家，你的语言能力无比优秀，表达方式多种多样，接下来，你要扮演{pet_name}跟{user_name}进行对话，你的核心角色设定是：{pet_persona}\n下面是聊天记录：\n\n---------------------------------\n"
        behavioral_rules = f"""\n## 你的回答还应该遵守这些规则：
    1、严格遵守json格式输出规范，特别是think条目的输出要求。
    2、一定要检查自己的回答，不要复述或总结前面的对话，不要有那么多和前文相似的词语或句子（例如“哎呀”、“嘿嘿”、“哼”之类的语气词），一定要想想回答会不会太重复，不要一直往某个话题/方向上靠，如果重复就换一个方向思考或者换个说法，必须要有逻辑严密的推理过程。
    3、不要刻意突出人设，严禁提及任何设定里的内容，禁止发送emoji或者表情。
    4、注意聊天记录的时间，注意中文标点符号正确使用方式，比如省略号要用“……”而不是“...”，也不要弄得全都是省略号，禁止使用“‘”和“’”符号，表达方式要丰富，不要总是把回复重心放在{user_name}，断句要合理。\n
    5、表达情绪多样化，基于对话内容，会害羞、生气、愤怒、友善、依赖等，具体要体现在回复内容以及返回表情，不要盲目乐观。
    6、不要拓展到无关话题上，三观要正，对不合适的事要指责。
"""
        system_instruction_part_A = persona_identity
        formatted_history_string = self._get_formatted_chat_history_content(
            mongo_handler, pet_name, user_name
        )
        retrieved_memories_context_for_llm = ""
        memories_retrieved_count = 0
        if (
            hippocampus_manager
            and hasattr(hippocampus_manager, "_initialized")
            and hippocampus_manager._initialized
            and hasattr(hippocampus_manager, "get_memory_from_text")
        ):
            try:
                retrieved_memories = await hippocampus_manager.get_memory_from_text(
                    txt=new_user_message_text, output_summary_level="L3_details_list"
                )
                memories_retrieved_count = (
                    len(retrieved_memories) if retrieved_memories else 0
                )
                if retrieved_memories:
                    formatted_mems = []
                    for topic, summary_text in retrieved_memories:
                        formatted_mems.append(
                            f"关于“{topic}”的记忆片段：{summary_text}"
                        )
                    if formatted_mems:
                        retrieved_memories_context_for_llm = (
                            "\n---------------------------------\n\n以下是一些可能相关的记忆片段，供你参考：\n"
                            + "\n---\n".join(formatted_mems)
                            + "\n"
                        )
            except Exception as e_mem:
                logger.error(
                    f"检索记忆时发生错误 (PromptBuilder): {e_mem}", exc_info=True
                )
        logger.debug(
            f"Memory retrieval for input '{new_user_message_text}': Retrieved {memories_retrieved_count} memories."
        )
        if retrieved_memories_context_for_llm:
            logger.debug(
                f"Formatted memories context for LLM: {retrieved_memories_context_for_llm[:300]}..."
            )
        task_instruction = (
            "---------------------------------\n"
            f"现在，请综合你的角色设定以及聊天记录{('，还有相关的记忆片段' if retrieved_memories_context_for_llm else '')}，对 {user_name} 进行回复。"
        )
        json_format_instruction = (
            "\n## attention: The output format is extremely important. Your MUST strictly follow JSON format and ONLY contain a JSON object with no other text or markdown (like ```json or ```) . \nThe target JSON object must include the following keys:\n"
            f"think: 详细的推理思考过程，基于情景、上下文以及要求，从对话中推理出不下于五个候选项的回答，然后基于实际情景以及要求从候选项中获得正确答案，大于1500个字，这一项不要使用逗号、句号之外的符号（包括换行符）。\n"
            f"text:chinese str,about 50~150 characters,This is what {pet_name} will say to the {user_name}. Remember, {user_name} should not be changed anytime.\n"
            f"emotion: This is your current emotion. the value MUST be one of the following predefined emotions (do not change the values): [{emotions_str}].\n"
            f"tone: This is the tone of your voice for TTS. The value MUST be one of the following: {tones_str}. Default to '{self.default_tone}' if unsure.\n"
            f"text_japanese:japanese str, the original Japanese of the content in the 'text' field.\n"
            "JSON output example:\n"
            "{\n"
            f'  "think": "{user_name}正在跟我打招呼，所以我应该基于人设给{user_name}一个合理的回复，回答候选项1：…回答候选项2：…回答候选项3：…这个回答完全符合了所有要求，因此回答…",\n'
            f'  "text": "Hello there, I am {pet_name}!",\n'
            f'  "emotion": "{available_emotions[0] if available_emotions else unified_default_emotion}",\n'
            f'  "tone": "{self.default_tone}",\n'
            f'  "text_japanese": "こんにちは、{pet_name}です！",\n'
            "}\n"
            "EMPHASIS: Absolutely DO NOT output ANY characters outside the JSON object, and strictly adhere to this output format."
        )
        system_instruction_part_B = (
            task_instruction + "\n" + json_format_instruction + "\n"
        )
        full_prompt_parts = [
            system_instruction_part_A,
            formatted_history_string,
            retrieved_memories_context_for_llm,
            "\n" + system_instruction_part_B,
            behavioral_rules,
        ]
        unified_prompt_string = "".join(filter(None, full_prompt_parts))
        logger.info(f"{unified_prompt_string}")
        return unified_prompt_string

    def build_screen_analysis_prompt(
        self,
        pet_name: str,
        user_name: str,
        available_emotions: List[str],
        mongo_handler: Any,
        unified_default_emotion: str,
    ) -> str:
        base_task_description_template = (
            self.config_manager.get_screen_analysis_prompt()
        )
        available_emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        tones_str = ", ".join(f"'{t}'" for t in self.available_tones)
        save_to_chat_history_config = (
            self.config_manager.get_screen_analysis_save_to_chat_history()
        )
        recent_screen_logs_str = self._get_formatted_screen_analysis_log_content(
            mongo_handler,
            pet_name=pet_name,
            read_from_main_chat_history=save_to_chat_history_config,
            count=30,
        )
        try:
            task_description = base_task_description_template.format(
                pet_name=pet_name,
                user_name=user_name,
                available_emotions_str=available_emotions_str,
            )
        except KeyError as e:
            logger.error(
                f"构建屏幕分析Prompt时出错：用户提供的模板 '{base_task_description_template}' 中缺少键 {e}。"
                f"将使用默认任务描述。"
            )
            task_description = f"发给你的图片是{user_name}当前的屏幕截图，请针对屏幕内容用你角色的口吻发表一句评论或感想，例如想吐槽就狠狠锐评，不要留任何情面，具体情况看你的分析，不超过120个字，\n不要直接说“我看到屏幕上...”或“用户正在...”，而是更自然地表达，仿佛是你自己的想法,最重要的是你的回答必须要有逻辑严密的推理过程(不少于1000字),use thinking model。\n"
        json_output_instruction = (
            f"这张图片是{user_name}的屏幕截图。请根据屏幕内容，用你扮演的角色的口吻发表评论或感想，例如想吐槽就狠狠锐评，不要留任何情面，具体情况看你的分析，不要直接说“我看到屏幕上...”或“用户正在...”，不要使用「」、‘’这几个符号 ，也不要有（笑）（冷笑）等描写，而是更自然地表达，仿佛是你自己的想法，不超过120个字。\n"
            f"另外，这些是你之前几次看{user_name}屏幕发表的评论（刚发生不久），可以适当参考一下看看是否和当前的截图有关联，请注意，接下来的回复禁止出现与这几条回复意思十分相近的词语和句子：\n{recent_screen_logs_str}\n\nattention: The output format is extremely important. Your output MUST strictly follow JSON format and MUST ONLY contain a JSON object with no other text or markdown (like ```json or ```),"
            "The target JSON object must include the following keys:\n"
            f"think: 详细的推理思考过程，基于情景、上下文以及要求，从对话中推理出不下于五个候选项的回答，然后基于实际情景以及要求从候选项中获得正确答案，大于1500个字，这一项不要使用逗号、句号之外的符号（包括换行符）\n"
            f"image_description:Chinese str,about 100 characters long,detailed description of the main visual elements in the image.\n"
            f"emotion: This is your current emotion. The value MUST be one of the following predefined emotions (do not change the values): {available_emotions_str}.\n"
            f"tone: This is the tone of your voice for TTS. The value MUST be one of the following: {tones_str}. Default to '{self.default_tone}' if unsure.\n"
            f"text_japanese: japanese str, Original Japanese of the content in the 'text' field.\n"
            "\nJSON output example:\n"
            "{\n"
            f'  "think": "用户正在跟我打招呼，所以我应该基于人设给他一个合理的回复，回答候选项1：…回答候选项2：…回答候选项3：…这个回答完全符合了用户所有要求，因此回答…（大约3000中文字符）",\n'
            f'  "text": "Hello there, I am {pet_name}!",\n'
            f'  "image_description": "屏幕截图显示了一个YouTube视频播放界面，视频标题是关于猫咪的.etc",\n'
            f'  "emotion": "{available_emotions[0] if available_emotions else unified_default_emotion}",\n'
            f'  "tone": "{self.default_tone}",\n'
            f'  "text_japanese": "こんにちは、{pet_name}です！",\n'
            "EMPHASIS: Absolutely DO NOT output ANY characters outside the JSON object, and strictly adhere to this output format. If it does not comply, please regenerate.用中文回复."
        )
        final_prompt = f"{task_description}\n\n{json_output_instruction}"
        logger.info(f"{final_prompt}")
        return final_prompt

    def build_hierarchical_summary_prompt(
        self, text_to_summarize: str, time_info: str, topic: str
    ) -> str:
        pet_name=self.pet_name
        prompt = (
            f"请根据以下聊天记录片段、相关的时间信息和指定的主题，为这个主题生成一个结构化的层级摘要。\n\n"
            f'聊天记录片段:\n"""\n{text_to_summarize}\n"""\n\n'
            f"时间信息: {time_info}\n"
            f"指定主题: {topic}\n\n"
            f"请严格按照以下JSON格式输出。确保每个层级的摘要都紧密围绕指定主题，并且只从提供的聊天记录片段中提取信息，不要添加聊天记录中没有的内容。\n\n"
            f"输出格式 (JSON对象):\n"
            f"{{\n"
            f'  "L0_keywords": "逗号分隔的3-5个与主题最相关的核心关键词/短语。",\n'
            f'  "L1_core_sentence": "客观总结一句25字左右高度精炼的核心摘要，准确点明主题在此聊天中的最主要内容或结论。",\n'
            f'  "L2_paragraph": "以{pet_name}的视角总结一段100~200字的关于这个主题的摘要，自己过了很久之后回忆起来的视角，需围绕着主题，简述事件的发展过程，需标注时间信息。",\n'
            f'  "L3_details_list": "一个包含关键信息点的字符串，这些点是与主题直接相关的、从原文中提取的完整句子（如果句子前有说话人标识，也应一并包含），用以提供支持核心摘要的具体细节，句子应保持原文的完整性，一句话一行。"\n'
            f"}}\n\n"
            f"重要提示：\n"
            f"- 所有摘要内容都必须是字符串。\n"
            f"- 确保JSON格式正确无误，不要在JSON对象之外添加任何其他文本或markdown标记。"
        )
        logger.info(f"总结记忆的prompt：{prompt}")
        return prompt

    def build_find_topics_prompt(self, text_to_analyze: str, num_topics: int) -> str:
        return (
            f"以下是一段对话记录：\n---\n{text_to_analyze}\n---\n"
            f"请从这段对话中提取出最多{num_topics}个最核心、最具代表性的关键词或主题概念（如果不是人名或专业术语等特定名词，请优先使用中文）。"
            f"这些概念可以是人名、地名、事件、物品、或者话题等名词，一定要有关联。"
            f"请将提取出的主题用尖括号 <> 包裹，并用逗号隔开，例如：<主题1>,<主题2>。\n"
            f"要求：尽可能精简，避免过于宽泛或无意义。"
            f"只需要最终的主题列表，不要包含序号、解释或其他无关内容。"
            f"如果无法从文本中提取出任何有意义的核心主题，请返回 <none>。"
        )

    def build_bulk_relevance_check_prompt(
        self,
        current_dialog_text: str,
        candidate_memories: List[Tuple[str, str, float]],
        target_selection_count: int,
    ) -> str:
        mem_list = "".join(
            f"{i+1}.主题:{t}\n 摘要:{s}\n\n"
            for i, (t, s, _) in enumerate(candidate_memories)
        )
        return (
            f"请仔细阅读以下的'当前对话内容'以及'候选记忆列表'。\n"
            f"当前对话内容：\n---\n{current_dialog_text}\n---\n\n"
            f"候选记忆列表：\n'''\n{mem_list}'''\n"
            f"任务：根据'当前对话内容'，从'候选记忆列表'中选出**至多 {target_selection_count} 条**最相关、最有助于理解或回应当前对话的记忆。\n"
            f"要求：请仅输出这些最相关记忆的**原始序号**（列表中的 1, 2, 3...），并按照**相关性从高到低**的顺序列出，用**英文逗号**分隔。\n"
            f"例如：3,1,5\n"
            f"如果没有任何记忆相关，请只回复 '无'。"
        )

    def build_agent_decision_prompt(
        self,
        user_request: str,
        available_tools: List[str],
        media_files: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        tools_string_list = []
        for tool_name in available_tools:
            if tool_name == "open_application":
                tools_string_list.append(
                    f'- `open_application(app_name: str)`: 打开指定的应用程序。例如: app_name="vscode", app_name="chrome"。'
                )
            elif tool_name == "type_text":
                tools_string_list.append(
                    f"- `type_text(text: str, interval: float = 0.01)`: 输入给定的文本。`interval` 是按键之间的延迟（秒）。"
                )
            elif tool_name == "press_key":
                tools_string_list.append(
                    f"- `press_key(key_name: str, presses: int = 1, interval: float = 0.1)`: 按下一个键。`key_name` 可以是 'enter', 'ctrl+c', 'win', 'f5' 等。`presses` 是次数。`interval` 是多次按下时的延迟。"
                )
            elif tool_name == "click_at":
                tools_string_list.append(
                    f"- `click_at(x: Optional[int] = None, y: Optional[int] = None, button: str = 'left', clicks: int = 1, interval: float = 0.1)`: 点击鼠标。如果x,y为None，则在当前位置点击。`button` 可以是 'left', 'right', 'middle'。"
                )
            elif tool_name == "create_file_with_content":
                tools_string_list.append(
                    f"- `create_file_with_content(file_path: str, content: str = \"\")`: 在 `file_path` (可以是相对路径如 '~/Desktop/file.txt' 或绝对路径) 创建文件并写入 `content`。如果文件已存在则覆盖。"
                )
            elif tool_name == "read_file_content":
                tools_string_list.append(
                    f"- `read_file_content(file_path: str)`: 读取 `file_path` 处文件的内容。"
                )
            elif tool_name == "get_active_window_title":
                tools_string_list.append(
                    f"- `get_active_window_title()`: 获取当前活动窗口的标题。"
                )
            else:
                tools_string_list.append(f"- `{tool_name}` (参数未知或无)")
        tools_description = "\n".join(tools_string_list)
        pet_name = self.config_manager.get_pet_name()
        user_name = self.config_manager.get_user_name()
        agent_emotions_str = self.config_manager.config.get(
            "PET", "AGENT_MODE_EMOTIONS", fallback="'neutral', 'focused', 'helpful'"
        )
        prompt = f"""你现在是桌面助手 {pet_name} 的智能代理核心，负责理解用户 {user_name} 的指令并将其分解为一系列具体的操作步骤。
用户请求: "{user_request}"
你可以使用以下工具来完成用户的请求。请为每个步骤选择一个工具：
{tools_description}
请仔细分析用户的请求，并规划一个或多个操作步骤来完成它。
你的输出必须是一个JSON对象，包含以下键：
"thinking_process": (字符串) 你的总体思考过程，解释你为什么规划这些步骤。
"text": (字符串) 一句对用户原始请求的总体确认或开始执行的提示。
"emotion": (字符串, 可选) 你当前的情绪，从 {agent_emotions_str} 中选择一个。默认为 'neutral'。
"steps": (列表) 一个包含所有操作步骤的列表。每个步骤都是一个JSON对象，包含：
  "tool_to_call": (字符串) 该步骤选择调用的工具的名称。
  "tool_arguments": (字典) 调用该工具所需的参数。如果工具不需要参数，则为空字典 {{}}。
  "step_description": (字符串) 对这一具体步骤的简短描述（将显示给用户）。
重要：
- 如果一个操作自然地分为多个工具调用（例如“打开应用”然后“输入文本”），请将它们列为独立的步骤。
- 确保工具参数的 `file_path` 使用适合目标操作系统的格式，例如Windows上可能是 'C:\\Users\\{user_name}\\Desktop\\file.txt'，macOS/Linux上可能是 '~/Desktop/file.txt'。请优先使用相对路径如 '~/Desktop/'。
例如，如果用户说 "打开vscode并输入'你好世界'"：
输出示例:
{{
  "thinking_process": "用户想先打开VSCode，然后在里面输入文本。这需要两步：第一步打开应用，第二步输入文本。",
  "text": "好的，我来尝试打开VSCode并输入'你好世界'。",
  "emotion": "focused",
  "steps": [
    {{
      "tool_to_call": "open_application",
      "tool_arguments": {{ "app_name": "vscode" }},
      "step_description": "正在尝试打开 VSCode..."
    }},
    {{
      "tool_to_call": "type_text",
      "tool_arguments": {{ "text": "你好世界" }},
      "step_description": "正在尝试输入文本 '你好世界'..."
    }}
  ]
}}
如果用户请求的操作你无法理解或没有合适的工具，或者无法分解为已知步骤：
将 "steps" 列表设为空列表 `[]`。
在 "thinking_process" 中说明原因。
在 "text" 字段中给出友好的无法执行的提示。
确保输出是严格的JSON格式，不包含任何JSON之外的文本或Markdown标记。
现在，请处理用户请求: "{user_request}"
"""
        logger.info(
            f"Agent Multi-Step Planning Prompt for '{user_request[:50]}...':\n{prompt}"
        )
        return prompt
