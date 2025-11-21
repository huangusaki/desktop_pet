# -*- coding: utf-8 -*-
"""
Prompt模板层
集中管理所有prompt模板,支持从配置加载自定义模板
"""
from typing import List
import logging
from ..utils.config_manager import ConfigManager

logger = logging.getLogger("PromptTemplates")


class PromptTemplates:
    """管理所有prompt模板"""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.bot_name = config_manager.get_bot_name()
        self.user_name = config_manager.get_user_name()
        self.available_tones = config_manager.get_tts_available_tones()
        self.default_tone = config_manager.get_tts_default_tone()

    def get_chat_persona_identity(
        self,
        bot_name: str,
        user_name: str,
        bot_persona: str,
        relationship_context: str,
    ) -> str:
        """
        获取聊天身份设定部分

        Args:
            bot_name: Bot名称
            user_name: 用户名称
            bot_persona: Bot人格设定
            relationship_context: 关系状态上下文

        Returns:
            身份设定字符串
        """
        persona_identity = (
            f"你是一名角色扮演专家,你能够全身心投入到我给你设定的角色,以及遵守我给的输出规范和Constraints,现在,你要扮演{bot_name}跟{user_name}进行对话\n"
            f"<identity>\n{bot_persona}\n</identity>\n"
            f"{relationship_context}\n"
        )
        return persona_identity

    def get_chat_behavioral_rules(self) -> str:
        """
        获取聊天行为规则(从配置加载)

        Returns:
            行为规则字符串
        """
        constraints = self.config_manager.get_bot_constraints()
        speech_pattern = self.config_manager.get_bot_speech_pattern()

        behavioral_rules = (
            f"\n<Constraints>\n{constraints}\n</Constraints>\n"
            f"<SpeechPattern>\n另外,这是你平时的说话风格,可以学习参考但不要使用:\n{speech_pattern}\n</SpeechPattern>"
        )
        return behavioral_rules

    def get_chat_task_instruction(
        self,
        user_name: str,
        has_memories: bool,
    ) -> str:
        """
        获取聊天任务指令

        Args:
            user_name: 用户名称
            has_memories: 是否有记忆片段

        Returns:
            任务指令字符串
        """
        memory_mention = ",还有相关的记忆片段" if has_memories else ""
        task_instruction = (
            f"<TaskInstruction>\n现在,请综合你的角色设定以及聊天记录{memory_mention},对 {user_name} 进行回复。\n</TaskInstruction>"
        )
        return task_instruction

    def get_chat_json_format_instruction(
        self,
        bot_name: str,
        available_emotions: List[str],
        unified_default_emotion: str,
    ) -> str:
        """
        获取JSON格式指令(从配置加载示例)

        Args:
            bot_name: Bot名称
            available_emotions: 可用情绪列表
            unified_default_emotion: 默认情绪

        Returns:
            JSON格式指令字符串
        """
        emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        tones_str = ", ".join(f"'{t}'" for t in self.available_tones)

        # 从配置获取格式示例
        format_example_template = self.config_manager.get_bot_format_example()
        
        # 使用replace替换模板变量,避免与JSON的大括号冲突
        format_example = format_example_template.replace(
            "{bot_name}", bot_name
        ).replace(
            "{default_emotion}", available_emotions[0] if available_emotions else unified_default_emotion
        ).replace(
            "{default_tone}", self.default_tone
        )

        json_format_instruction = (
            "<OutputFormat>\n"
            "Your MUST strictly follow JSON format and ONLY contain a JSON object with no other text or markdown (like ```json or ```)."
            "Target JSON object must include the following keys:\n"
            f"text:中文字符串,小于100个字\n"
            f"emotion: 体现你回答时的心情,值只能是以下选项中的一个: [{emotions_str}]\n"
            f"tone: 你觉得你的回答应该使用的语调,值只能是以下选项中的一个: [{tones_str}]\n"
            "favorability_change: 整数,表示这次对话对你好感度的影响,请将变化值控制在 -20 到 20 之间。\n"
            f"text_japanese:日语字符串,一般是text项的翻译\n"
            f'emotion_update: 对象,包含state和reason两个字段,都使用日文。state是这次对话后你对{user_name}的心情状态,reason是导致这种心情的具体时间和具体原因。\n'
            "JSON output example:\n"
            f"{format_example}\n"
            "EMPHASIS: Absolutely DO NOT output ANY characters outside the JSON object,and strictly adhere to this output format. "
            "\n</OutputFormat>"
        )
        return json_format_instruction

    def get_screen_analysis_task_description(
        self,
        bot_name: str,
        user_name: str,
        available_emotions: List[str],
    ) -> str:
        """
        获取屏幕分析任务描述

        Args:
            bot_name: Bot名称
            user_name: 用户名称
            available_emotions: 可用情绪列表

        Returns:
            任务描述字符串
        """
        base_template = self.config_manager.get_screen_analysis_prompt()
        available_emotions_str = ", ".join(f"'{e}'" for e in available_emotions)

        try:
            task_description = base_template.format(
                bot_name=bot_name,
                user_name=user_name,
                available_emotions_str=available_emotions_str,
            )
        except KeyError as e:
            logger.error(
                f"构建屏幕分析Prompt时出错:用户提供的模板 '{base_template}' 中缺少键 {e}。"
                f"将使用默认任务描述。"
            )
            task_description = (
                f"发给你的图片是{user_name}当前的屏幕截图,请针对屏幕内容用你角色的口吻发表一句评论或感想,"
                f"例如想吐槽就狠狠锐评,不要留任何情面,具体情况看你的分析,不超过120个字,\n"
                f'不要直接说"我看到屏幕上..."或"用户正在...",而是更自然地表达,仿佛是你自己的想法,'
                f"最重要的是你的回答必须要有逻辑严密的推理过程(不少于1000字),use thinking model。\n"
            )

        return task_description

    def get_screen_analysis_json_output_instruction(
        self,
        bot_name: str,
        user_name: str,
        available_emotions: List[str],
        unified_default_emotion: str,
        recent_logs: str,
    ) -> str:
        """
        获取屏幕分析JSON输出指令

        Args:
            bot_name: Bot名称
            user_name: 用户名称
            available_emotions: 可用情绪列表
            unified_default_emotion: 默认情绪
            recent_logs: 最近的屏幕分析日志

        Returns:
            JSON输出指令字符串
        """
        available_emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        tones_str = ", ".join(f"'{t}'" for t in self.available_tones)

        json_output_instruction = (
            f"这张图片是{user_name}的屏幕截图。请根据屏幕内容,用你扮演的角色的口吻发表评论或感想,"
            f"例如想吐槽就狠狠锐评,不要留任何情面,具体情况看你的分析,"
            f'不要直接说"我看到屏幕上..."或"用户正在...",不要使用「」、\'\'这几个符号,'
            f"也不要有(笑)(冷笑)等描写,而是更自然地表达,仿佛是你自己的想法,不超过120个字。\n"
            f"另外,这些是你之前几次看{user_name}屏幕发表的评论(刚发生不久),"
            f"可以适当参考一下看看是否和当前的截图有关联,"
            f"请注意,接下来的回复禁止出现与这几条回复意思十分相近的词语和句子:\n{recent_logs}\n\n"
            f"attention: The output format is extremely important. "
            f"Your output MUST strictly follow JSON format and MUST ONLY contain a JSON object "
            f"with no other text or markdown (like ```json or ```),"
            "The target JSON object must include the following keys:\n"
            f"image_description:Chinese str,about 100 characters long,"
            f"detailed description of the main visual elements in the image.\n"
            f"emotion: This is your current emotion. The value MUST be one of the following "
            f"predefined emotions (do not change the values): {available_emotions_str}.\n"
            f"tone: This is the tone of your voice for TTS. The value MUST be one of the following: "
            f"{tones_str}. Default to '{self.default_tone}' if unsure.\n"
            f"text_japanese: japanese str, Original Japanese of the content in the 'text' field.\n"
            "\nJSON output example:\n"
            "{\n"
            f'  "text": "Hello there, I am {bot_name}!",\n'
            f'  "image_description": "屏幕截图显示了一个YouTube视频播放界面,视频标题是关于猫咪的.etc",\n'
            f'  "emotion": "{available_emotions[0] if available_emotions else unified_default_emotion}",\n'
            f'  "tone": "{self.default_tone}",\n'
            f'  "text_japanese": "こんにちは、{bot_name}です!",\n'
            "}\n"
            "EMPHASIS: Absolutely DO NOT output ANY characters outside the JSON object, "
            "and strictly adhere to this output format. If it does not comply, please regenerate."
            f"用中文回复,另外text_japanese中,请将{bot_name}转成片假名,不要使用原名"
        )
        return json_output_instruction

    def get_hierarchical_summary_prompt(
        self,
        text_to_summarize: str,
        time_info: str,
        topic: str,
    ) -> str:
        """
        获取层级摘要prompt

        Args:
            text_to_summarize: 要总结的文本
            time_info: 时间信息
            topic: 主题

        Returns:
            层级摘要prompt字符串
        """
        bot_name = self.bot_name

        prompt = (
            f"请根据以下聊天记录片段、相关的时间信息和指定的主题,为这个主题生成一个结构化的层级摘要。\n\n"
            f'聊天记录片段:\n"""\n{text_to_summarize}\n"""\n\n'
            f"时间信息: {time_info}\n"
            f"指定主题: {topic}\n\n"
            f"请严格按照以下JSON格式输出。确保每个层级的摘要都紧密围绕指定主题,"
            f"并且只从提供的聊天记录片段中提取信息,不要添加聊天记录中没有的内容。\n\n"
            f"输出格式 (JSON对象):\n"
            f"{{\n"
            f'  "L0_keywords": "逗号分隔的3-5个与主题最相关的核心关键词/短语。",\n'
            f'  "L1_core_sentence": "客观总结一句25字左右高度精炼的核心摘要,准确点明主题在此聊天中的最主要内容或结论。",\n'
            f'  "L2_paragraph": "以{bot_name}的视角总结一段100~200字的关于这个主题的摘要,'
            f'自己过了很久之后回忆起来的视角,需围绕着主题,简述事件的发展过程,需标注时间信息。",\n'
            f'  "L3_details_list": "一个包含关键信息点的字符串,这些点是与主题直接相关的、'
            f'从原文中提取的完整句子(如果句子前有说话人标识,也应一并包含),'
            f'用以提供支持核心摘要的具体细节,句子应保持原文的完整性,一句话一行。"\n'
            f"}}\n\n"
            f"重要提示:\n"
            f"- 所有摘要内容都必须是字符串。\n"
            f"- 确保JSON格式正确无误,不要在JSON对象之外添加任何其他文本或markdown标记。"
        )
        logger.info(f"总结记忆的prompt:{prompt}")
        return prompt

    def get_find_topics_prompt(
        self,
        text_to_analyze: str,
        num_topics: int,
    ) -> str:
        """
        获取主题提取prompt

        Args:
            text_to_analyze: 要分析的文本
            num_topics: 提取的主题数量

        Returns:
            主题提取prompt字符串
        """
        return (
            f"以下是一段对话记录:\n---\n{text_to_analyze}\n---\n"
            f"请从这段对话中提取出最多{num_topics}个最核心、最具代表性的关键词或主题概念"
            f"(如果不是人名或专业术语等特定名词,请优先使用中文)。"
            f"这些概念可以是人名、地名、事件、物品、或者话题等名词,一定要有关联。"
            f"请将提取出的主题用尖括号 <> 包裹,并用逗号隔开,例如:<主题1>,<主题2>。\n"
            f"要求:尽可能精简,避免过于宽泛或无意义。"
            f"只需要最终的主题列表,不要包含序号、解释或其他无关内容。"
            f"如果无法从文本中提取出任何有意义的核心主题,请返回 <none>。"
        )

    def get_bulk_relevance_check_prompt(
        self,
        current_dialog_text: str,
        candidate_memories: List[tuple],
        target_selection_count: int,
    ) -> str:
        """
        获取批量相关性检查prompt

        Args:
            current_dialog_text: 当前对话文本
            candidate_memories: 候选记忆列表 [(topic, summary, score), ...]
            target_selection_count: 目标选择数量

        Returns:
            批量相关性检查prompt字符串
        """
        mem_list = "".join(
            f"{i + 1}.主题:{t}\n 摘要:{s}\n\n"
            for i, (t, s, _) in enumerate(candidate_memories)
        )

        return (
            f"请仔细阅读以下的'当前对话内容'以及'候选记忆列表'。\n"
            f"当前对话内容:\n---\n{current_dialog_text}\n---\n\n"
            f"候选记忆列表:\n'''\n{mem_list}'''\n"
            f"任务:根据'当前对话内容',从'候选记忆列表'中选出**至多 {target_selection_count} 条**"
            f"最相关、最有助于理解或回应当前对话的记忆。\n"
            f"要求:请仅输出这些最相关记忆的**原始序号**(列表中的 1, 2, 3...),"
            f"并按照**相关性从高到低**的顺序列出,用**英文逗号**分隔。\n"
            f"例如:3,1,5\n"
            f"如果没有任何记忆相关,请只回复 '无'。"
        )

    def get_agent_decision_prompt(
        self,
        user_request: str,
        available_tools: List[str],
    ) -> str:
        """
        获取Agent决策prompt

        Args:
            user_request: 用户请求
            available_tools: 可用工具列表

        Returns:
            Agent决策prompt字符串
        """
        # 工具描述映射
        tool_descriptions = {
            "open_application": '- `open_application(app_name: str)`: 打开指定的应用程序。例如: app_name="vscode", app_name="chrome"。',
            "type_text": "- `type_text(text: str, interval: float = 0.01)`: 输入给定的文本。`interval` 是按键之间的延迟(秒)。",
            "press_key": "- `press_key(key_name: str, presses: int = 1, interval: float = 0.1)`: 按下一个键。`key_name` 可以是 'enter', 'ctrl+c', 'win', 'f5' 等。`presses` 是次数。`interval` 是多次按下时的延迟。",
            "click_at": "- `click_at(x: Optional[int] = None, y: Optional[int] = None, button: str = 'left', clicks: int = 1, interval: float = 0.1)`: 点击鼠标。如果x,y为None,则在当前位置点击。`button` 可以是 'left', 'right', 'middle'。",
            "create_file_with_content": '- `create_file_with_content(file_path: str, content: str = "")`: 在 `file_path` (可以是相对路径如 \'~/Desktop/file.txt\' 或绝对路径) 创建文件并写入 `content`。如果文件已存在则覆盖。',
            "read_file_content": "- `read_file_content(file_path: str)`: 读取 `file_path` 处文件的内容。",
            "get_active_window_title": "- `get_active_window_title()`: 获取当前活动窗口的标题。",
        }

        tools_string_list = []
        for tool_name in available_tools:
            if tool_name in tool_descriptions:
                tools_string_list.append(tool_descriptions[tool_name])
            else:
                tools_string_list.append(f"- `{tool_name}` (参数未知或无)")

        tools_description = "\n".join(tools_string_list)
        bot_name = self.config_manager.get_bot_name()
        user_name = self.config_manager.get_user_name()
        agent_emotions_str = self.config_manager.get_bot_agent_mode_emotions()

        prompt = f"""你现在是桌面助手 {bot_name} 的智能代理核心,负责理解用户 {user_name} 的指令并将其分解为一系列具体的操作步骤。
用户请求: "{user_request}"
你可以使用以下工具来完成用户的请求。请为每个步骤选择一个工具:
{tools_description}
请仔细分析用户的请求,并规划一个或多个操作步骤来完成它。
你的输出必须是一个JSON对象,包含以下键:
"thinking_process": (字符串) 你的总体思考过程,解释你为什么规划这些步骤。
"text": (字符串) 一句对用户原始请求的总体确认或开始执行的提示。
"emotion": (字符串, 可选) 你当前的情绪,从 {agent_emotions_str} 中选择一个。默认为 'neutral'。
"steps": (列表) 一个包含所有操作步骤的列表。每个步骤都是一个JSON对象,包含:
  "tool_to_call": (字符串) 该步骤选择调用的工具的名称。
  "tool_arguments": (字典) 调用该工具所需的参数。如果工具不需要参数,则为空字典 {{}}.
  "step_description": (字符串) 对这一具体步骤的简短描述(将显示给用户)。
重要:
- 如果一个操作自然地分为多个工具调用(例如"打开应用"然后"输入文本"),请将它们列为独立的步骤。
- 确保工具参数的 `file_path` 使用适合目标操作系统的格式,例如Windows上可能是 'C:\\\\Users\\\\{user_name}\\\\Desktop\\\\file.txt',macOS/Linux上可能是 '~/Desktop/file.txt'。请优先使用相对路径如 '~/Desktop/'。
例如,如果用户说 "打开vscode并输入'你好世界'":
输出示例:
{{
  "thinking_process": "用户想先打开VSCode,然后在里面输入文本。这需要两步:第一步打开应用,第二步输入文本。",
  "text": "好的,我来尝试打开VSCode并输入'你好世界'。",
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
如果用户请求的操作你无法理解或没有合适的工具,或者无法分解为已知步骤:
将 "steps" 列表设为空列表 `[]`。
在 "thinking_process" 中说明原因。
在 "text" 字段中给出友好的无法执行的提示。
确保输出是严格的JSON格式,不包含任何JSON之外的文本或Markdown标记。
现在,请处理用户请求: "{user_request}"
"""
        logger.info(f"Agent Multi-Step Planning Prompt for '{user_request[:50]}...':\n{prompt}")
        return prompt
