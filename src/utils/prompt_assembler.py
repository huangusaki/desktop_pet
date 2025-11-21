"""
Prompt组装层
负责组装最终的prompt字符串
"""
from typing import List, Optional, Any
import logging
from .prompt_data_provider import PromptDataProvider
from .prompt_templates import PromptTemplates

logger = logging.getLogger("PromptAssembler")


class PromptAssembler:
    """负责组装最终的prompt"""

    def __init__(
        self,
        data_provider: PromptDataProvider,
        templates: PromptTemplates,
    ):
        self.data_provider = data_provider
        self.templates = templates

    async def assemble_chat_prompt(
        self,
        new_user_message_text: str,
        bot_name: str,
        user_name: str,
        bot_persona: str,
        available_emotions: List[str],
        unified_default_emotion: str,
        mongo_handler: Any,
        hippocampus_manager: Optional[Any] = None,
        is_multimodal_request: bool = False,
    ) -> str:
        """
        组装统一聊天prompt

        Args:
            new_user_message_text: 新的用户消息文本
            bot_name: Bot名称
            user_name: 用户名称
            bot_persona: Bot人格设定
            available_emotions: 可用情绪列表
            unified_default_emotion: 默认情绪
            mongo_handler: MongoDB处理器
            hippocampus_manager: 海马体管理器(可选)
            is_multimodal_request: 是否为多模态请求

        Returns:
            组装好的prompt字符串
        """
        # 1. 获取关系状态上下文
        relationship_context = self.data_provider.get_relationship_context(user_name)

        # 2. 构建身份设定部分
        persona_identity = self.templates.get_chat_persona_identity(
            bot_name, user_name, bot_persona, relationship_context
        )

        # 3. 获取聊天历史
        formatted_history = self.data_provider.get_formatted_chat_history(
            mongo_handler, bot_name, user_name
        )

        # 4. 获取记忆上下文
        memory_context, memory_count = await self.data_provider.get_memory_context(
            hippocampus_manager, new_user_message_text
        )

        # 5. 构建任务指令
        task_instruction = self.templates.get_chat_task_instruction(
            user_name, has_memories=bool(memory_context)
        )

        # 6. 构建JSON格式指令
        json_format_instruction = self.templates.get_chat_json_format_instruction(
            bot_name, available_emotions, unified_default_emotion
        )

        # 7. 获取行为规则
        behavioral_rules = self.templates.get_chat_behavioral_rules()

        # 8. 组装最终prompt
        full_prompt_parts = [
            persona_identity,
            formatted_history,
            memory_context,
            "\n" + task_instruction + "\n" + json_format_instruction + "\n",
            behavioral_rules,
        ]

        unified_prompt = "".join(filter(None, full_prompt_parts))
        logger.info(f"组装完成的聊天prompt:\n{unified_prompt}")
        return unified_prompt

    def assemble_screen_analysis_prompt(
        self,
        bot_name: str,
        user_name: str,
        available_emotions: List[str],
        mongo_handler: Any,
        unified_default_emotion: str,
    ) -> str:
        """
        组装屏幕分析prompt

        Args:
            bot_name: Bot名称
            user_name: 用户名称
            available_emotions: 可用情绪列表
            mongo_handler: MongoDB处理器
            unified_default_emotion: 默认情绪

        Returns:
            组装好的屏幕分析prompt字符串
        """
        # 1. 获取任务描述
        task_description = self.templates.get_screen_analysis_task_description(
            bot_name, user_name, available_emotions
        )

        # 2. 获取最近的屏幕分析日志
        save_to_chat_history = (
            self.data_provider.config_manager.get_screen_analysis_save_to_chat_history()
        )
        recent_logs = self.data_provider.get_formatted_screen_analysis_logs(
            mongo_handler,
            bot_name=bot_name,
            read_from_main_chat_history=save_to_chat_history,
            count=30,
        )

        # 3. 构建JSON输出指令
        json_output_instruction = self.templates.get_screen_analysis_json_output_instruction(
            bot_name, user_name, available_emotions, unified_default_emotion, recent_logs
        )

        # 4. 组装最终prompt
        final_prompt = f"{task_description}\n\n{json_output_instruction}"
        logger.info(f"组装完成的屏幕分析prompt:\n{final_prompt}")
        return final_prompt

    def assemble_hierarchical_summary_prompt(
        self,
        text_to_summarize: str,
        time_info: str,
        topic: str,
    ) -> str:
        """
        组装层级摘要prompt

        Args:
            text_to_summarize: 要总结的文本
            time_info: 时间信息
            topic: 主题

        Returns:
            组装好的层级摘要prompt字符串
        """
        return self.templates.get_hierarchical_summary_prompt(
            text_to_summarize, time_info, topic
        )

    def assemble_find_topics_prompt(
        self,
        text_to_analyze: str,
        num_topics: int,
    ) -> str:
        """
        组装主题提取prompt

        Args:
            text_to_analyze: 要分析的文本
            num_topics: 提取的主题数量

        Returns:
            组装好的主题提取prompt字符串
        """
        return self.templates.get_find_topics_prompt(text_to_analyze, num_topics)

    def assemble_bulk_relevance_check_prompt(
        self,
        current_dialog_text: str,
        candidate_memories: List[tuple],
        target_selection_count: int,
    ) -> str:
        """
        组装批量相关性检查prompt

        Args:
            current_dialog_text: 当前对话文本
            candidate_memories: 候选记忆列表
            target_selection_count: 目标选择数量

        Returns:
            组装好的批量相关性检查prompt字符串
        """
        return self.templates.get_bulk_relevance_check_prompt(
            current_dialog_text, candidate_memories, target_selection_count
        )

    def assemble_agent_decision_prompt(
        self,
        user_request: str,
        available_tools: List[str],
        media_files: Optional[List[Any]] = None,
    ) -> str:
        """
        组装Agent决策prompt

        Args:
            user_request: 用户请求
            available_tools: 可用工具列表
            media_files: 媒体文件列表(可选,暂未使用)

        Returns:
            组装好的Agent决策prompt字符串
        """
        return self.templates.get_agent_decision_prompt(user_request, available_tools)
