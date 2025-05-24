import configparser
import os


class ConfigManager:
    def __init__(self, config_file="config/settings.ini"):
        self.config = configparser.ConfigParser()
        project_root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        actual_config_path = os.path.join(project_root_dir, config_file)
        if not os.path.exists(actual_config_path):
            raise FileNotFoundError(f"配置文件 {actual_config_path} 未找到。")
        self.config.read(actual_config_path, encoding="utf-8")
        print(f"ConfigManager: 成功加载配置文件 {actual_config_path}")

    def get_gemini_api_key(self):
        return self.config.get("GEMINI", "API_KEY", fallback=None)

    def get_gemini_model_name(self):
        return self.config.get(
            "GEMINI", "MODEL_NAME", fallback="gemini-1.5-flash-latest"
        )

    def get_http_proxy(self):
        proxy = self.config.get("GEMINI", "HTTP_PROXY", fallback="").strip()
        return proxy if proxy else None

    def get_https_proxy(self):
        proxy = self.config.get("GEMINI", "HTTPS_PROXY", fallback="").strip()
        return proxy if proxy else None

    def get_pet_initial_image_filename(self):
        return self.config.get("PET", "INITIAL_IMAGE_FILENAME", fallback="default.png")

    def get_pet_name(self):
        return self.config.get("PET", "NAME", fallback="小助手")

    def get_user_name(self):
        return self.config.get("USER", "NAME", fallback="主人")

    def get_pet_persona(self):
        return self.config.get(
            "PET", "PERSONA", fallback="你是一个友好、乐于助人的桌面宠物。"
        )

    def get_avatar_base_path_relative(self):
        return self.config.get(
            "AVATARS", "AVATAR_BASE_PATH", fallback="src/assets/icon"
        )

    def get_pet_avatar_filename(self):
        return self.config.get("AVATARS", "PET_AVATAR_FILENAME", fallback="bot.png")

    def get_user_avatar_filename(self):
        return self.config.get("AVATARS", "USER_AVATAR_FILENAME", fallback="user.png")

    def get_mongo_connection_string(self):
        return self.config.get(
            "MONGODB", "CONNECTION_STRING", fallback="mongodb://localhost:27017/"
        )

    def get_mongo_database_name(self):
        return self.config.get("MONGODB", "DATABASE_NAME", fallback="desktop_pet_db")

    def get_mongo_collection_name(self):
        return self.config.get("MONGODB", "COLLECTION_NAME", fallback="chat_history")

    def get_history_count_for_prompt(self):
        try:
            return self.config.getint("MONGODB", "HISTORY_COUNT_FOR_PROMPT", fallback=5)
        except ValueError:
            print("警告: MONGODB HISTORY_COUNT_FOR_PROMPT 值无效，将使用默认值 5。")
            return 5
