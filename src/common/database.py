from pymongo import MongoClient, DESCENDING
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
import datetime
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger("Database")


class MongoHandler:
    def __init__(
        self, connection_string: str, database_name: str, collection_name: str
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(
                self.connection_string, serverSelectionTimeoutMS=5000
            )
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            logger.info(
                f"成功连接到 MongoDB: {self.connection_string}, 数据库: '{self.database_name}', 集合: '{self.collection_name}'"
            )
        except ConnectionFailure as e:
            logger.error(f"无法连接到 MongoDB ({self.connection_string}): {e}")
            self.client = None
            self.db = None
            self.collection = None
        except Exception as e:
            logger.error(f"连接 MongoDB 时发生其他错误: {e}", exc_info=True)
            self.client = None
            self.db = None
            self.collection = None

    def is_connected(self) -> bool:
        return (
            self.client is not None
            and self.db is not None
            and self.collection is not None
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
        if not self.is_connected() or self.collection is None:
            logger.error("错误: 未连接到 MongoDB 或集合未初始化，无法插入消息。")
            return None
        message_document: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "sender": sender,
            "message_text": message_text,
            "role_play_character": role_play_character,
            "memorized_times": memorized_times,
        }
        try:
            result = self.collection.insert_one(message_document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(
                f"插入消息到 MongoDB ('{self.collection_name}') 时出错: {e}",
                exc_info=True,
            )
            return None

    def get_recent_chat_history(
        self, count: int = 10, role_play_character: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_connected() or self.collection is None:
            logger.error("错误: 未连接到 MongoDB 或集合未初始化，无法获取聊天记录。")
            return []
        query: Dict[str, Any] = {}
        if role_play_character:
            query["role_play_character"] = role_play_character
        try:
            messages = list(
                self.collection.find(query).sort("timestamp", DESCENDING).limit(count)
            )
            return messages[::-1]
        except Exception as e:
            logger.error(
                f"从 MongoDB ('{self.collection_name}') 获取聊天记录时出错: {e}",
                exc_info=True,
            )
            return []

    def close_connection(self):
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB 连接已关闭。")
            except Exception as e:
                logger.error(f"关闭 MongoDB 连接时出错: {e}", exc_info=True)
            finally:
                self.client = None
                self.db = None
                self.collection = None
