from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
import datetime


class MongoHandler:
    def __init__(
        self, connection_string: str, database_name: str, collection_name: str
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(
                self.connection_string, serverSelectionTimeoutMS=5000
            )
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            print(
                f"成功连接到 MongoDB: {self.connection_string}, 数据库: '{self.database_name}'"
            )
        except ConnectionFailure as e:
            print(f"无法连接到 MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
        except Exception as e:
            print(f"连接 MongoDB 时发生其他错误: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def is_connected(self) -> bool:
        return self.client is not None and self.db is not None

    def get_database(self) -> Optional[Database]:
        return self.db

    def insert_chat_message(
        self,
        sender: str,
        message_text: str,
        role_play_character: str = None,
        memorized_times: int = 0,
    ) -> Optional[str]:
        if not self.is_connected() or not self.collection:
            print("错误: 未连接到 MongoDB 或集合未初始化，无法插入消息。")
            return None
        message_document = {
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
            print(f"插入消息到 MongoDB 时出错: {e}")
            return None

    def get_recent_chat_history(
        self, count: int = 10, role_play_character: str = None
    ) -> list:
        if not self.is_connected() or not self.collection:
            return []
        query = {}
        if role_play_character:
            query["role_play_character"] = role_play_character
        try:
            messages = list(
                self.collection.find(query).sort("timestamp", DESCENDING).limit(count)
            )
            return messages[::-1]
        except Exception as e:
            print(f"从 MongoDB 获取聊天记录时出错: {e}")
            return []

    def close_connection(self):
        if self.client:
            self.client.close()
            print("MongoDB 连接已关闭。")
