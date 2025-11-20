from pymongo import MongoClient, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
import datetime
from typing import Optional, List, Dict, Any
import logging
import asyncio

logger = logging.getLogger("MongoDB")


class MongoHandler:
    def __init__(
        self, connection_string: str, database_name: str, collection_name: str
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.chat_history_collection_name = collection_name
        self.relationship_status_collection_name = "relationship_status"
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.chat_collection: Optional[Collection] = None
        self.relationship_status_collection: Optional[Collection] = None
        self.graph_nodes_collection_name = "graph_data_nodes"
        self.graph_edges_collection_name = "graph_data_edges"
        self.llm_usage_collection_name = "llm_usage"
        self.screen_analysis_log_collection_name = "screen_analysis_log"
        self.graph_nodes_collection: Optional[Collection] = None
        self.graph_edges_collection: Optional[Collection] = None
        self.llm_usage_collection: Optional[Collection] = None
        self.screen_analysis_log_collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            logger.info(f"Attempting to connect to MongoDB: {self.connection_string}")
            self.client = MongoClient(
                self.connection_string, serverSelectionTimeoutMS=5000
            )
            logger.info("MongoClient created, sending ping command...")
            self.client.admin.command("ping")
            logger.info("MongoDB ping successful!")
            
            self.db = self.client[self.database_name]
            self.chat_collection = self.db[self.chat_history_collection_name]
            self.relationship_status_collection = self.db[
                self.relationship_status_collection_name
            ]
            self.graph_nodes_collection = self.db[self.graph_nodes_collection_name]
            self.graph_edges_collection = self.db[self.graph_edges_collection_name]
            self.llm_usage_collection = self.db[self.llm_usage_collection_name]
            self.screen_analysis_log_collection = self.db[
                self.screen_analysis_log_collection_name
            ]
            logger.info(
                f"成功连接到 MongoDB: {self.connection_string}, 数据库: '{self.database_name}'"
            )
            # ... (省略部分日志以减少冗余，保留关键信息)
        except ConnectionFailure as e:
            logger.error(f"无法连接到 MongoDB ({self.connection_string}): {e}")
            self._clear_connections()
        except Exception as e:
            logger.error(f"连接 MongoDB 时发生其他错误: {e}", exc_info=True)
            self._clear_connections()

    def _clear_connections(self):
        """Helper to reset connection attributes."""
        self.client = None
        self.db = None
        self.chat_collection = None
        self.relationship_status_collection = None
        self.graph_nodes_collection = None
        self.graph_edges_collection = None
        self.llm_usage_collection = None
        self.screen_analysis_log_collection = None

    def is_connected(self) -> bool:
        return (
            self.client is not None
            and self.db is not None
            and self.chat_collection is not None
            and self.relationship_status_collection is not None
            and self.graph_nodes_collection is not None
            and self.graph_edges_collection is not None
            and self.llm_usage_collection is not None
            and self.screen_analysis_log_collection is not None
        )

    def get_database(self) -> Optional[Database]:
        """返回 pymongo.database.Database 实例，如果已连接。"""
        return self.db

    def insert_chat_message(
        self,
        sender: str,
        message_text: str,
        role_play_character: Optional[str] = None,
        memorized_times: int = 0,
    ) -> Optional[str]:
        if not self.is_connected() or self.chat_collection is None:
            logger.error(
                "错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法插入消息。"
            )
            return None
        message_document: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).timestamp(),
            "sender": sender,
            "message_text": message_text,
            "role_play_character": role_play_character,
            "memorized_times": memorized_times,
        }
        try:
            result = self.chat_collection.insert_one(message_document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(
                f"插入聊天消息到 MongoDB ('{self.chat_history_collection_name}') 时出错: {e}",
                exc_info=True,
            )
            return None

    def insert_screen_analysis_log_entry(
        self,
        sender: str,
        message_text: str,
        role_play_character: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_connected() or self.screen_analysis_log_collection is None:
            logger.error(
                "错误: 未连接到 MongoDB 或屏幕分析日志集合未初始化，无法插入条目。"
            )
            return None
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).timestamp(),
            "sender": sender,
            "message_text": message_text,
            "role_play_character": role_play_character,
            "memorized_times": 0,
        }
        try:
            result = self.screen_analysis_log_collection.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(
                f"插入屏幕分析日志到 MongoDB ('{self.screen_analysis_log_collection_name}') 时出错: {e}",
                exc_info=True,
            )
            return None

    def get_recent_chat_history(
        self, count: int = 10, role_play_character: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_connected() or self.chat_collection is None:
            logger.error(
                "错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法获取聊天记录。"
            )
            return []
        query: Dict[str, Any] = {}
        if role_play_character:
            query["role_play_character"] = role_play_character
        try:
            messages = list(
                self.chat_collection.find(query)
                .sort("timestamp", DESCENDING)
                .limit(count)
            )
            return messages[::-1]
        except Exception as e:
            logger.error(
                f"从 MongoDB ('{self.chat_history_collection_name}') 获取聊天记录时出错: {e}",
                exc_info=True,
            )
            return []

    def update_chat_messages_memorized_time(
        self, message_ids: List[Any], increment: int = 1
    ):
        """
        Increments the 'memorized_times' for a list of chat message IDs.
        """
        if not self.is_connected() or not self.chat_collection:
            logger.error(
                "错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法更新 memorized_times。"
            )
            return False
        if not message_ids:
            return True
        try:
            result = self.chat_collection.update_many(
                {"_id": {"$in": message_ids}}, {"$inc": {"memorized_times": increment}}
            )
            logger.info(
                f"更新了 {result.modified_count} 条聊天记录的 memorized_times。"
            )
            return True
        except Exception as e:
            logger.error(f"批量更新聊天记录 memorized_times 时出错: {e}", exc_info=True)
            return False

    def get_graph_nodes_collection(self) -> Optional[Collection]:
        return self.graph_nodes_collection if self.is_connected() else None

    def get_graph_edges_collection(self) -> Optional[Collection]:
        return self.graph_edges_collection if self.is_connected() else None

    def get_llm_usage_collection(self) -> Optional[Collection]:
        return self.llm_usage_collection if self.is_connected() else None

    def close_connection(self):
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB 连接已关闭。")
            except Exception as e:
                logger.error(f"关闭 MongoDB 连接时出错: {e}", exc_info=True)
            finally:
                self._clear_connections()

    def get_recent_screen_analysis_log(
        self, count: int = 5, role_play_character: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取最近的屏幕分析日志条目。
        """
        if not self.is_connected() or self.screen_analysis_log_collection is None:
            logger.error(
                "错误: 未连接到 MongoDB 或屏幕分析日志集合未初始化，无法获取日志。"
            )
            return []
        query: Dict[str, Any] = {}
        if role_play_character:
            query["role_play_character"] = role_play_character
        try:
            messages = list(
                self.screen_analysis_log_collection.find(query)
                .sort("timestamp", DESCENDING)
                .limit(count)
            )
            return messages
        except Exception as e:
            logger.error(
                f"从 MongoDB ('{self.screen_analysis_log_collection_name}') 获取屏幕分析日志时出错: {e}",
                exc_info=True,
            )
            return []

    def get_favorability_score(self, user_name: str, pet_name: str) -> Optional[int]:
        """
        获取指定用户和Bot之间的好感度分数。
        """
        if not self.is_connected() or self.relationship_status_collection is None:
            logger.error("错误: 未连接到 MongoDB 或关系集合未初始化，无法获取好感度。")
            return None
        try:
            doc = self.relationship_status_collection.find_one(
                {"user_name": user_name, "pet_name": pet_name}
            )
            if doc and "favorability_score" in doc:
                return int(doc["favorability_score"])
            return None
        except Exception as e:
            logger.error(
                f"从 MongoDB ('{self.relationship_status_collection_name}') 获取好感度时出错: {e}",
                exc_info=True,
            )
            return None

    async def update_favorability_score(
        self, user_name: str, pet_name: str, new_score: int
    ) -> bool:
        if not self.is_connected() or self.relationship_status_collection is None:
            logger.error("错误: 未连接到 MongoDB 或关系集合未初始化，无法更新好感度。")
            return False
        query = {"user_name": user_name, "pet_name": pet_name}
        update = {
            "$set": {
                "favorability_score": new_score,
                "last_updated": datetime.datetime.now(datetime.timezone.utc),
            }
        }
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.relationship_status_collection.update_one(
                    query, update, upsert=True
                ),
            )
            return True
        except Exception as e:
            logger.error(
                f"向 MongoDB ('{self.relationship_status_collection_name}') 更新好感度时出错: {e}",
                exc_info=True,
            )
            return False
