from pymongo import MongoClient, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure
import datetime
from typing import Optional, List, Dict, Any


class MongoHandler:
    def __init__(
        self, connection_string: str, database_name: str, collection_name: str
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.chat_history_collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.chat_collection: Optional[Collection] = None
        self.graph_nodes_collection_name = "graph_data_nodes"
        self.graph_edges_collection_name = "graph_data_edges"
        self.llm_usage_collection_name = "llm_usage"
        self.graph_nodes_collection: Optional[Collection] = None
        self.graph_edges_collection: Optional[Collection] = None
        self.llm_usage_collection: Optional[Collection] = None
        self._connect()

    def _connect(self):
        try:
            self.client = MongoClient(
                self.connection_string, serverSelectionTimeoutMS=5000
            )
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.chat_collection = self.db[self.chat_history_collection_name]
            self.graph_nodes_collection = self.db[self.graph_nodes_collection_name]
            self.graph_edges_collection = self.db[self.graph_edges_collection_name]
            self.llm_usage_collection = self.db[self.llm_usage_collection_name]
            print(
                f"成功连接到 MongoDB: {self.connection_string}, 数据库: '{self.database_name}'"
            )
            print(f"  Chat history collection: '{self.chat_history_collection_name}'")
            print(f"  Graph nodes collection: '{self.graph_nodes_collection_name}'")
            print(f"  Graph edges collection: '{self.graph_edges_collection_name}'")
            print(f"  LLM usage collection: '{self.llm_usage_collection_name}'")
        except ConnectionFailure as e:
            print(f"无法连接到 MongoDB ({self.connection_string}): {e}")
            self._clear_connections()
        except Exception as e:
            print(f"连接 MongoDB 时发生其他错误: {e}")
            self._clear_connections()

    def _clear_connections(self):
        """Helper to reset connection attributes."""
        self.client = None
        self.db = None
        self.chat_collection = None
        self.graph_nodes_collection = None
        self.graph_edges_collection = None
        self.llm_usage_collection = None

    def is_connected(self) -> bool:
        return (
            self.client is not None
            and self.db is not None
            and self.chat_collection is not None
            and self.graph_nodes_collection is not None
            and self.graph_edges_collection is not None
            and self.llm_usage_collection is not None
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
            print("错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法插入消息。")
            return None
        message_document: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "sender": sender,
            "message_text": message_text,
            "role_play_character": role_play_character,
            "memorized_times": memorized_times,
        }
        try:
            result = self.chat_collection.insert_one(message_document)
            return str(result.inserted_id)
        except Exception as e:
            print(
                f"插入聊天消息到 MongoDB ('{self.chat_history_collection_name}') 时出错: {e}"
            )
            return None

    def get_recent_chat_history(
        self, count: int = 10, role_play_character: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_connected() or self.chat_collection is None:
            print("错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法获取聊天记录。")
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
            print(
                f"从 MongoDB ('{self.chat_history_collection_name}') 获取聊天记录时出错: {e}"
            )
            return []

    def update_chat_messages_memorized_time(
        self, message_ids: List[Any], increment: int = 1
    ):
        """
        Increments the 'memorized_times' for a list of chat message IDs.
        """
        if not self.is_connected() or not self.chat_collection:
            print(
                "错误: 未连接到 MongoDB 或聊天记录集合未初始化，无法更新 memorized_times。"
            )
            return False
        if not message_ids:
            return True
        try:
            result = self.chat_collection.update_many(
                {"_id": {"$in": message_ids}}, {"$inc": {"memorized_times": increment}}
            )
            print(f"更新了 {result.modified_count} 条聊天记录的 memorized_times。")
            return True
        except Exception as e:
            print(f"批量更新聊天记录 memorized_times 时出错: {e}")
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
                print("MongoDB 连接已关闭。")
            except Exception as e:
                print(f"关闭 MongoDB 连接时出错: {e}")
            finally:
                self._clear_connections()


if __name__ == "__main__":
    os.environ["MONGODB_CONNECTION_STRING"] = "mongodb://localhost:27017/"
    os.environ["MONGODB_DATABASE_NAME"] = "desktop_pet_test_db"
    print("测试 MongoHandler...")
    conn_str = os.getenv("MONGODB_CONNECTION_STRING")
    db_name = os.getenv("MONGODB_DATABASE_NAME")
    chat_coll_name = "test_chat_history"
    if not conn_str or not db_name:
        print(
            "请设置 MONGODB_CONNECTION_STRING 和 MONGODB_DATABASE_NAME 环境变量进行测试。"
        )
    else:
        mongo_handler = MongoHandler(conn_str, db_name, chat_coll_name)
        if mongo_handler.is_connected():
            print("\n--- 连接成功 ---")
            print("\n--- 测试聊天消息插入 ---")
            msg_id1 = mongo_handler.insert_chat_message(
                "user", "你好，宠物！", "TestPet"
            )
            if msg_id1:
                print(f"  用户消息插入成功, ID: {msg_id1}")
            msg_id2 = mongo_handler.insert_chat_message(
                "pet", "你好，主人！", "TestPet"
            )
            if msg_id2:
                print(f"  宠物消息插入成功, ID: {msg_id2}")
            print("\n--- 测试获取最近聊天记录 (TestPet) ---")
            history = mongo_handler.get_recent_chat_history(
                count=5, role_play_character="TestPet"
            )
            if history:
                for msg in history:
                    ts = (
                        msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(msg.get("timestamp"), datetime.datetime)
                        else msg.get("timestamp")
                    )
                    print(
                        f"  [{ts}] {msg['sender']}: {msg['message_text']} (Learned: {msg.get('memorized_times',0)})"
                    )
            else:
                print("  未找到聊天记录。")
            if msg_id1 and msg_id2:
                from bson import ObjectId

                ids_to_update = [ObjectId(msg_id1), ObjectId(msg_id2)]
                print(f"\n--- 测试更新 memorized_times for IDs: {ids_to_update} ---")
                mongo_handler.update_chat_messages_memorized_time(ids_to_update)
                updated_history = mongo_handler.get_recent_chat_history(
                    count=2, role_play_character="TestPet"
                )
                for msg in updated_history:
                    ts = (
                        msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if isinstance(msg.get("timestamp"), datetime.datetime)
                        else msg.get("timestamp")
                    )
                    print(
                        f"  (更新后) [{ts}] {msg['sender']}: {msg['message_text']} (Learned: {msg.get('memorized_times',0)})"
                    )
            print("\n--- 测试访问记忆系统集合 ---")
            nodes_coll = mongo_handler.get_graph_nodes_collection()
            if nodes_coll is not None:
                print(f"  成功获取图节点集合: {nodes_coll.name}")
            else:
                print("  未能获取图节点集合。")
            edges_coll = mongo_handler.get_graph_edges_collection()
            if edges_coll is not None:
                print(f"  成功获取图边集合: {edges_coll.name}")
            else:
                print("  未能获取图边集合。")
            llm_usage_coll = mongo_handler.get_llm_usage_collection()
            if llm_usage_coll is not None:
                print(f"  成功获取LLM使用记录集合: {llm_usage_coll.name}")
            else:
                print("  未能获取LLM使用记录集合。")
            mongo_handler.close_connection()
        else:
            print("\n--- 连接失败 ---")
