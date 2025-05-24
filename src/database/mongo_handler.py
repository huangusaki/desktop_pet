from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import datetime


class MongoHandler:
    def __init__(
        self, connection_string: str, database_name: str, collection_name: str
    ):
        """
        初始化 MongoDB 处理器。
        Args:
            connection_string (str): MongoDB 的连接字符串。
                                     例如: "mongodb://localhost:27017/"
                                     对于 Atlas: "mongodb+srv://<username>:<password>@<cluster-url>/<dbname>?retryWrites=true&w=majority"
            database_name (str): 要使用的数据库名称。
            collection_name (str): 要使用的集合名称 (用于存储聊天记录)。
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self._connect()

    def _connect(self):
        """
        连接到 MongoDB 并选择数据库和集合。
        """
        try:
            self.client = MongoClient(self.connection_string)
            self.client.admin.command("ping")
            print(f"成功连接到 MongoDB: {self.connection_string}")
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            print(
                f"已选择数据库 '{self.database_name}' 和集合 '{self.collection_name}'"
            )
        except ConnectionFailure as e:
            print(f"无法连接到 MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
            raise
        except Exception as e:
            print(f"连接 MongoDB 时发生其他错误: {e}")
            self.client = None
            self.db = None
            self.collection = None
            raise

    def is_connected(self) -> bool:
        """检查是否已成功连接到数据库。"""
        return (
            self.client is not None
            and self.db is not None
            and self.collection is not None
        )

    def insert_chat_message(
        self, sender: str, message_text: str, role_play_character: str = None
    ) -> str | None:
        """
        向集合中插入一条新的聊天记录。
        Args:
            sender (str): 发言者 ("user" 或 "pet")。
            message_text (str): 消息内容。
            role_play_character (str, optional): 当前宠物的角色设定名 (如果适用)。默认为 None。
        Returns:
            str | None: 插入的文档的 _id (ObjectId 转换为字符串)，如果插入失败则返回 None。
        """
        if not self.is_connected():
            print("错误: 未连接到 MongoDB，无法插入消息。")
            return None
        message_document = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "sender": sender,
            "message_text": message_text,
            "role_play_character": role_play_character,
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
        """
        获取最近的N条聊天记录。
        Args:
            count (int): 要获取的记录数量。
            role_play_character (str, optional): 如果指定，则只获取该角色的聊天记录。默认为 None (获取所有)。
        Returns:
            list: 包含聊天记录文档的列表，按时间戳降序排列。
        """
        if not self.is_connected():
            print("错误: 未连接到 MongoDB，无法获取聊天记录。")
            return []
        query = {}
        if role_play_character:
            query["role_play_character"] = role_play_character
        try:
            messages = list(
                self.collection.find(query).sort("timestamp", -1).limit(count)
            )
            return messages[::-1]
        except Exception as e:
            print(f"从 MongoDB 获取聊天记录时出错: {e}")
            return []

    def close_connection(self):
        """
        关闭 MongoDB 连接。
        """
        if self.client:
            self.client.close()
            print("MongoDB 连接已关闭。")


if __name__ == "__main__":
    CONNECTION_STRING = "mongodb://localhost:27017/"
    DATABASE_NAME = "desktop_pet_db"
    COLLECTION_NAME = "chat_history"
    print("尝试初始化 MongoHandler...")
    try:
        mongo_handler = MongoHandler(CONNECTION_STRING, DATABASE_NAME, COLLECTION_NAME)
        if mongo_handler.is_connected():
            print("\nMongoHandler 初始化成功并已连接。")
            print("\n--- 插入示例消息 ---")
            user_msg_id = mongo_handler.insert_chat_message(
                sender="user",
                message_text="你好，小爱同学！",
                role_play_character="DefaultPet",
            )
            if user_msg_id:
                mongo_handler.insert_chat_message(
                    sender="pet",
                    message_text="你好呀主人！有什么可以帮你的吗？",
                    role_play_character="DefaultPet",
                )
            user_msg_id_2 = mongo_handler.insert_chat_message(
                sender="user",
                message_text="今天天气怎么样？",
                role_play_character="DefaultPet",
            )
            if user_msg_id_2:
                mongo_handler.insert_chat_message(
                    sender="pet",
                    message_text="今天阳光明媚，是个好天气！",
                    role_play_character="DefaultPet",
                )
            print("\n--- 获取最近的3条聊天记录 (DefaultPet) ---")
            recent_messages = mongo_handler.get_recent_chat_history(
                count=3, role_play_character="DefaultPet"
            )
            if recent_messages:
                for msg in recent_messages:
                    print(
                        f"  [{msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {msg['sender']}: {msg['message_text']}"
                    )
            else:
                print("  未找到聊天记录或获取失败。")
            print("\n--- 获取最近的5条聊天记录 (所有角色) ---")
            all_recent = mongo_handler.get_recent_chat_history(count=5)
            if all_recent:
                for msg in all_recent:
                    print(
                        f"  [{msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] ({msg.get('role_play_character', 'N/A')}) {msg['sender']}: {msg['message_text']}"
                    )
            else:
                print("  未找到聊天记录或获取失败。")
            mongo_handler.close_connection()
        else:
            print("MongoHandler 初始化失败或未连接。")
    except Exception as main_e:
        print(f"主程序中发生错误: {main_e}")
