# -*- coding: utf-8 -*-
"""
Prompt数据提供层
负责从各种数据源获取格式化的数据,用于构建prompt
"""
from typing import Any, List, Optional, Tuple
import logging
import datetime
from ..utils.config_manager import ConfigManager
from ..data.relationship_manager import RelationshipManager

logger = logging.getLogger("PromptDataProvider")


class PromptDataProvider:
    """负责为prompt构建提供格式化的数据"""

    def __init__(
        self,
        config_manager: ConfigManager,
        relationship_manager: Optional[RelationshipManager] = None,
    ):
        self.config_manager = config_manager
        self.relationship_manager = relationship_manager

    def get_formatted_chat_history(
        self,
        mongo_handler: Any,
        bot_name: str,
        user_name: str,
    ) -> str:
        """
        获取格式化的聊天历史记录

        Args:
            mongo_handler: MongoDB处理器
            bot_name: Bot名称
            user_name: 用户名称

        Returns:
            格式化的聊天历史字符串
        """
        history_lines = []

        if not (
            mongo_handler
            and hasattr(mongo_handler, "is_connected")
            and mongo_handler.is_connected()
        ):
            return "(没有找到相关的对话历史)"

        prompt_history_count = self.config_manager.get_history_count_for_prompt()
        raw_db_history = []

        if hasattr(mongo_handler, "get_recent_chat_history"):
            raw_db_history = (
                mongo_handler.get_recent_chat_history(
                    count=prompt_history_count,
                    role_play_character=bot_name,
                )
                or []
            )

        if raw_db_history:
            for msg_entry in raw_db_history:
                sender_val = msg_entry.get("sender")
                text_content = msg_entry.get("message_text", "")
                timestamp_unix = msg_entry.get("timestamp")

                # 格式化时间戳
                formatted_time_prefix = ""
                if timestamp_unix:
                    try:
                        dt_object = datetime.datetime.fromtimestamp(
                            float(timestamp_unix)
                        )
                        formatted_time_prefix = dt_object.strftime("%Y-%m-%d %H:%M") + " "
                    except (ValueError, TypeError) as e:
                        logger.warning(f"格式化聊天记录时间戳 '{timestamp_unix}' 时出错: {e}")

                # 确定发言人前缀
                speaker_prefix = ""
                if sender_val == user_name:
                    speaker_prefix = f"{user_name}: "
                elif sender_val == bot_name:
                    speaker_prefix = f"{bot_name}: "
                elif sender_val and isinstance(sender_val, str):
                    speaker_prefix = f"{sender_val}: "

                if text_content and speaker_prefix:
                    history_lines.append(
                        f"{formatted_time_prefix}{speaker_prefix}{text_content}"
                    )

        if history_lines:
            return "<ChatHistory>\n" + "\n".join(history_lines) + "\n</ChatHistory>"
        else:
            return "<ChatHistory>\n(没有找到相关的对话历史)\n</ChatHistory>"

    def get_formatted_screen_analysis_logs(
        self,
        mongo_handler: Any,
        bot_name: str,
        read_from_main_chat_history: bool,
        count: int = 10,
    ) -> str:
        """
        获取格式化的屏幕分析日志

        Args:
            mongo_handler: MongoDB处理器
            bot_name: Bot名称
            read_from_main_chat_history: 是否从主聊天历史读取
            count: 获取的日志数量

        Returns:
            格式化的屏幕分析日志字符串
        """
        log_lines = []

        if not (
            mongo_handler
            and hasattr(mongo_handler, "is_connected")
            and mongo_handler.is_connected()
        ):
            return "(数据库未连接或不可用)"

        raw_logs: List[Any] = []

        if read_from_main_chat_history:
            if hasattr(mongo_handler, "get_recent_chat_history"):
                raw_logs = mongo_handler.get_recent_chat_history(
                    count=count, role_play_character=bot_name
                )
            else:
                logger.warning("MongoHandler missing get_recent_chat_history method.")
        else:
            if hasattr(mongo_handler, "get_recent_screen_analysis_log"):
                raw_logs = mongo_handler.get_recent_screen_analysis_log(
                    count=count, role_play_character=bot_name
                )
            else:
                logger.warning("MongoHandler missing get_recent_screen_analysis_log method.")

        if raw_logs:
            for log_entry in raw_logs:
                text_content = log_entry.get("message_text", "")
                if text_content:
                    if read_from_main_chat_history:
                        if log_entry.get("sender") == bot_name:
                            log_lines.append(f"- {text_content}")
                    else:
                        log_lines.append(f"- {text_content}")

        if log_lines:
            return "\n".join(log_lines)
        else:
            return "(最近没有相关记录)"

    def get_relationship_context(self, user_name: str) -> str:
        """
        获取关系状态上下文

        Args:
            user_name: 用户名称

        Returns:
            格式化的关系状态字符串
        """
        if not self.relationship_manager:
            return ""

        level_info = self.relationship_manager.get_current_relationship_level()
        if not level_info:
            return ""

        relationship_context = (
            f"<Relationship>\n"
            f"你和 {user_name} 的当前关系是 '{level_info['name']}' (好感度分数: {level_info['score']})。\n"
            f"态度指导: {level_info['description']}\n"
            f'你对{user_name}的感觉是"{level_info["state"]}",因为"{level_info["reason"]}"。让这种心情自然地影响你的回复语气,并根据聊天记录和对方的回复判断是否需要调整你的态度。\n'
            f"请严格根据这个关系状态来调整你的语气、回应内容以及 `favorability_change` 的取值。\n"
            f"</Relationship>"
        )
        return relationship_context

    async def get_memory_context(
        self,
        hippocampus_manager: Optional[Any],
        new_user_message_text: str,
    ) -> Tuple[str, int]:
        """
        获取记忆上下文

        Args:
            hippocampus_manager: 海马体管理器
            new_user_message_text: 新的用户消息文本

        Returns:
            (格式化的记忆上下文字符串, 检索到的记忆数量)
        """
        retrieved_memories_context = ""
        memories_count = 0

        if not (
            hippocampus_manager
            and hasattr(hippocampus_manager, "_initialized")
            and hippocampus_manager._initialized
            and hasattr(hippocampus_manager, "get_memory_from_text")
        ):
            return retrieved_memories_context, memories_count

        try:
            retrieved_memories = await hippocampus_manager.get_memory_from_text(
                txt=new_user_message_text, output_summary_level="L3_details_list"
            )
            memories_count = len(retrieved_memories) if retrieved_memories else 0

            if retrieved_memories:
                formatted_mems = []
                for topic, summary_text in retrieved_memories:
                    # 使用ASCII引号避免编码问题
                    formatted_mems.append(f'关于"{topic}"的记忆片段:{summary_text}')

                if formatted_mems:
                    retrieved_memories_context = (
                        "\n<MemoryInfomation>\n以下是一些可能相关的记忆片段,供你参考:\n"
                        + "\n---\n".join(formatted_mems)
                        + "\n</MemoryInfomation>"
                    )

            logger.debug(
                f"Memory retrieval for input '{new_user_message_text}': "
                f"Retrieved {memories_count} memories."
            )
            if retrieved_memories_context:
                logger.debug(
                    f"Formatted memories context for LLM: "
                    f"{retrieved_memories_context[:300]}..."
                )

        except Exception as e_mem:
            logger.error(
                f"检索记忆时发生错误 (PromptDataProvider): {e_mem}", exc_info=True
            )

        return retrieved_memories_context, memories_count

