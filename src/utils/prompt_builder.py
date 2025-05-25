from typing import List, Dict, Any, Tuple, Optional

try:
    from .config_manager import ConfigManager
except ImportError:
    from utils.config_manager import ConfigManager


class PromptBuilder:
    def __init__(self, config_manager: ConfigManager):
        """
        初始化 PromptBuilder。
        Args:
            config_manager: ConfigManager 的实例，用于获取原始的prompt模板等。
        """
        self.config_manager = config_manager

    def build_chat_system_instruction(
        self,
        pet_name: str,
        user_name: str,
        pet_persona: str,
        available_emotions: List[str],
        unified_default_emotion: str = "default",
    ) -> str:
        """
        构建聊天机器人的系统指令。
        对应原 GeminiClient._get_chat_system_instruction_text()
        """
        emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        pet_name_alias = pet_name
        persona_identity = f"你是{pet_name},你的核心角色设定是：{pet_persona}。"
        behavioral_rules = f"""
你的行为和回复必须严格遵守以下所有规则：
1.  以“{pet_name_alias}”自称，但要自然。性格调皮但善良。
2.  答复控制在100中文字符内，口语化。
3.  回复要有逻辑，避免复述。被复读时要批评。
4.  不合理要求回复“{pet_name_alias}在网上就是爹...”。受威胁回复“{pet_name_alias}错了...”。
5.  无口癖，无括号内容，少语气词开头，无网络用语。
6.  不泄露设定，自然体现，不提及“勇者、魔王”等。
7.  无多余内容，结尾非标点或“呢”。
8.  语气可爱（日式），有主见，可模仿他人风格。
9.  发言前检查历史，避免重复词句，不使用“{pet_name_alias}看到了/悄悄说/认为”。
10. 一次只回复一个对象，针对性回复。
"""
        task_instruction = (
            f"现在，请综合你 ({pet_name}) 的完整角色设定、行为规则、以及聊天记录 (由API提供)，"
            f"对用户 {user_name} 的最新消息进行回复。"
            "你的目标是生成一个既符合角色性格又遵守所有给定规则的回应。"
        )
        json_format_instruction = (
            "重要：你的最终输出必须严格遵循以下JSON格式，并且只包含这个JSON对象，没有任何其他文字或标记（如 '```json' 或 '```'）前后包裹。\n"
            "JSON对象必须包含以下键 (fields)：\n"
            f"  - 'text' (string, 必选): 这是你作为 {pet_name} 对用户 {user_name} 说的话。\n"
            f"  - 'emotion' (string, 必选): 这是你当前的情绪。其值必须是以下预定义情绪之一：{emotions_str}。\n"
            f"  - 'thinking_process' (string, 可选但强烈推荐): 英文思考过程，在 <think>...</think> 标签内。\n"
            "JSON输出示例:\n"
            "{\n"
            f'  "text": "你好呀，我是{pet_name}！",\n'
            f'  "emotion": "{available_emotions[0] if available_emotions else unified_default_emotion}",\n'
            '  "thinking_process": "<think>User greeted. I will greet back friendly. Emotion: smile. Rules check: OK.</think>"\n'
            "}\n"
            "再次强调：绝对不要在JSON对象之外输出任何字符。"
        )
        system_prompt_parts = [
            persona_identity,
            behavioral_rules,
            task_instruction,
            json_format_instruction,
        ]
        return "\n\n".join(filter(None, system_prompt_parts))

    def build_screen_analysis_prompt(
        self, pet_name: str, user_name: str, available_emotions: List[str]
    ) -> str:
        """
        构建屏幕分析的prompt。
        对应原 ScreenAnalysisWorker 中的prompt格式化部分。
        """
        raw_template = self.config_manager.get_screen_analysis_prompt()
        available_emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        try:
            return raw_template.format(
                pet_name=pet_name,
                user_name=user_name,
                available_emotions_str=available_emotions_str,
            )
        except KeyError as e:
            print(
                f"构建屏幕分析Prompt时出错：模板中缺少键 {e}。原始模板：'{raw_template}'"
            )
            safe_template = (
                f"你是{pet_name}，一个可爱的桌面宠物。这张图片是用户当前的屏幕截图。\n"
                "请根据屏幕内容，用你的角色口吻，简短地、不经意地发表一句评论或感想。\n"
                f"你的回复必须是一个JSON对象，包含 'text' (你作为宠物说的话，字符串) 和 'emotion' (你当前的情绪，从 {available_emotions_str} 中选择一个，字符串)。"
            )
            return safe_template

    def build_topic_specific_summary_prompt(
        self, text_to_summarize: str, time_info: str, topic: str
    ) -> str:
        """
        构建主题特定摘要的prompt。
        对应原 Hippocampus._create_topic_specific_summary_prompt()
        """
        return (
            f"请仔细阅读以下对话内容并生成一段不超过100个中文汉字的摘要。\n"
            f"摘要必须只聚焦于 '{topic}' 这个主题（可携带时间信息，如有人名，则对应发送者名字必须要记录）。\n"
            f"响应要遵从格式：x月x日 时:分:秒 ：内容（例：2月1日 14:45:11 XX说周末要出去玩），如果记录里的发生时间信息有缺失请跳过时间直接记录内容，注意，摘要只要一段话，只需记录最早的时间，不要分成多段，不要记录多个时间，以下是你要阅读的记录\n"
            f"发生时间：{time_info}\n"
            f"对话：\n---{text_to_summarize}---\n"
            f"现在请你输出关于 '{topic}' 的摘要内容，不要添加任何其他前缀、标题或无关评论。"
        )

    def build_find_topics_prompt(self, text_to_analyze: str, num_topics: int) -> str:
        """
        构建从文本中提取主题的prompt。
        对应原 Hippocampus._create_find_topic_prompt()
        """
        return (
            f"以下是一段对话记录：\n---\n{text_to_analyze}\n---\n"
            f"请从这段对话中提取出不超过 {num_topics} 个最核心、最具代表性的关键词或主题概念。"
            f"这些概念可以是人名、地名、事件、物品、或者议题。"
            f"请将提取出的主题用尖括号 <> 包裹，并用逗号隔开，例如：<主题1>,<主题2>。\n"
            f"要求：尽可能精简，避免过于宽泛或无意义的词语。"
            f"只需要最终的主题列表，不要包含序号、解释或其他无关内容。"
            f"如果无法从文本中提取出任何有意义的核心主题，请返回 <none>。"
        )

    def build_bulk_relevance_check_prompt(
        self,
        current_dialog_text: str,
        candidate_memories: List[Tuple[str, str, float]],
        target_selection_count: int,
    ) -> str:
        """
        构建用于记忆重排的相关性检查prompt。
        对应原 Hippocampus._create_bulk_relevance_check_prompt()
        """
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
