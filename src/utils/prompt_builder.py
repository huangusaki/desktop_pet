"""
Prompt构建器 - 门面类
提供统一的prompt构建接口,内部委托给专门的模块处理
"""
from typing import List, Dict, Any, Tuple, Optional
import logging
from ..utils.config_manager import ConfigManager
from ..data.relationship_manager import RelationshipManager
from .prompt_data_provider import PromptDataProvider
from .prompt_templates import PromptTemplates
from .prompt_assembler import PromptAssembler

logger = logging.getLogger("PromptBuilder")


class PromptBuilder:
    """
    Prompt构建器门面类
    保持向后兼容的公共接口,内部使用三层架构:
    - PromptDataProvider: 数据获取层
    - PromptTemplates: 模板管理层
    - PromptAssembler: 组装层
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        relationship_manager: RelationshipManager,
    ):
        """
        初始化PromptBuilder

        Args:
            config_manager: 配置管理器
            relationship_manager: 关系管理器
        """
        self.config_manager = config_manager
        self.relationship_manager = relationship_manager

        # 初始化三层架构组件
        self.data_provider = PromptDataProvider(config_manager, relationship_manager)
        self.templates = PromptTemplates(config_manager)
        self.assembler = PromptAssembler(self.data_provider, self.templates)

        # 保留原有属性以保持兼容性
        self.available_tones = self.config_manager.get_tts_available_tones()
        self.default_tone = self.config_manager.get_tts_default_tone()
        self.bot_name = self.config_manager.get_bot_name()
        self.user_name = self.config_manager.get_user_name()

    async def build_unified_chat_prompt_string(
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
        构建统一聊天prompt字符串

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
        return await self.assembler.assemble_chat_prompt(
            new_user_message_text=new_user_message_text,
            bot_name=bot_name,
            user_name=user_name,
            bot_persona=bot_persona,
            available_emotions=available_emotions,
            unified_default_emotion=unified_default_emotion,
            mongo_handler=mongo_handler,
            hippocampus_manager=hippocampus_manager,
            is_multimodal_request=is_multimodal_request,
        )

    def build_screen_analysis_prompt(
        self,
        bot_name: str,
        user_name: str,
        available_emotions: List[str],
        mongo_handler: Any,
        unified_default_emotion: str,
    ) -> str:
        """
        构建屏幕分析prompt

        Args:
            bot_name: Bot名称
            user_name: 用户名称
            available_emotions: 可用情绪列表
            mongo_handler: MongoDB处理器
            unified_default_emotion: 默认情绪

        Returns:
            组装好的屏幕分析prompt字符串
        """
        return self.assembler.assemble_screen_analysis_prompt(
            bot_name=bot_name,
            user_name=user_name,
            available_emotions=available_emotions,
            mongo_handler=mongo_handler,
            unified_default_emotion=unified_default_emotion,
        )

    def build_hierarchical_summary_prompt(
        self,
        text_to_summarize: str,
        time_info: str,
        topic: str,
    ) -> str:
        """
        构建层级摘要prompt

        Args:
            text_to_summarize: 要总结的文本
            time_info: 时间信息
            topic: 主题

        Returns:
            组装好的层级摘要prompt字符串
        """
        return self.assembler.assemble_hierarchical_summary_prompt(
            text_to_summarize=text_to_summarize,
            time_info=time_info,
            topic=topic,
        )

    def build_find_topics_prompt(
        self,
        text_to_analyze: str,
        num_topics: int,
    ) -> str:
        """
        构建主题提取prompt

        Args:
            text_to_analyze: 要分析的文本
            num_topics: 提取的主题数量

        Returns:
            组装好的主题提取prompt字符串
        """
        return self.assembler.assemble_find_topics_prompt(
            text_to_analyze=text_to_analyze,
            num_topics=num_topics,
        )

    def build_bulk_relevance_check_prompt(
        self,
        current_dialog_text: str,
        candidate_memories: List[Tuple[str, str, float]],
        target_selection_count: int,
    ) -> str:
        """
        构建批量相关性检查prompt

        Args:
            current_dialog_text: 当前对话文本
            candidate_memories: 候选记忆列表 [(topic, summary, score), ...]
            target_selection_count: 目标选择数量

        Returns:
            组装好的批量相关性检查prompt字符串
        """
        return self.assembler.assemble_bulk_relevance_check_prompt(
            current_dialog_text=current_dialog_text,
            candidate_memories=candidate_memories,
            target_selection_count=target_selection_count,
        )

    def build_agent_decision_prompt(
        self,
        user_request: str,
        available_tools: List[str],
        media_files: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        构建Agent决策prompt

        Args:
            user_request: 用户请求
            available_tools: 可用工具列表
            media_files: 媒体文件列表(可选)

        Returns:
            组装好的Agent决策prompt字符串
        """
        return self.assembler.assemble_agent_decision_prompt(
            user_request=user_request,
            available_tools=available_tools,
            media_files=media_files,
        )
