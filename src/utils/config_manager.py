import configparser
import os


class ConfigManager:
    def __init__(self, config_file="config/settings.ini"):
        self.config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        project_root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if (
            project_root_dir.endswith(os.path.join("src", "utils"))
            or project_root_dir.endswith(os.path.join("src", "memory_system"))
            or project_root_dir.endswith("src")
        ):
            project_root_dir = os.path.dirname(project_root_dir)
            if os.path.basename(project_root_dir) == "src":
                project_root_dir = os.path.dirname(project_root_dir)
        actual_config_path = os.path.join(project_root_dir, config_file)
        if not os.path.exists(actual_config_path):
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_project_root_dir = current_script_dir
            for _ in range(3):
                if os.path.isdir(
                    os.path.join(alt_project_root_dir, "src")
                ) and os.path.isdir(os.path.join(alt_project_root_dir, "config")):
                    break
                alt_project_root_dir = os.path.dirname(alt_project_root_dir)
            actual_config_path_alt = os.path.join(alt_project_root_dir, config_file)
            if os.path.exists(actual_config_path_alt):
                actual_config_path = actual_config_path_alt
                project_root_dir = alt_project_root_dir
            else:
                raise FileNotFoundError(
                    f"配置文件 {actual_config_path} (或备用路径 {actual_config_path_alt}) 未找到。请确保路径正确，或脚本从项目根目录运行。"
                )
        self.config.read(actual_config_path, encoding="utf-8")
        print(f"ConfigManager: 成功加载配置文件 {actual_config_path}")
        self.project_root = project_root_dir

    def get_gemini_api_key(self):
        return self.config.get("GEMINI", "API_KEY", fallback=None)

    def get_screen_analysis_save_to_chat_history(self) -> bool:
        return self.config.getboolean(
            "SCREEN_ANALYSIS", "SAVE_REACTION_TO_CHAT_HISTORY", fallback=True
        )

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
        return self.config.getint("MONGODB", "HISTORY_COUNT_FOR_PROMPT", fallback=5)

    def get_chat_dialog_display_history_count(self, default: int = 0) -> int:
        """
        获取聊天对话框中用于初始显示的历史记录数量。
        如果配置中未设置，则可以依赖传入的 default 值。
        """
        try:
            if self.config.has_option("MONGODB", "CHAT_DIALOG_DISPLAY_HISTORY_COUNT"):
                return self.config.getint(
                    "MONGODB", "CHAT_DIALOG_DISPLAY_HISTORY_COUNT"
                )
            else:
                return default
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
        except Exception:
            return default

    def get_screen_analysis_enabled(self) -> bool:
        return self.config.getboolean("SCREEN_ANALYSIS", "ENABLED", fallback=False)

    def get_screen_analysis_interval_seconds(self) -> int:
        return self.config.getint("SCREEN_ANALYSIS", "INTERVAL_SECONDS", fallback=60)

    def get_screen_analysis_chance(self) -> float:
        return self.config.getfloat("SCREEN_ANALYSIS", "CHANCE", fallback=0.1)

    def get_screen_analysis_prompt(self) -> str:
        default_prompt = (
            "你是{pet_name}，一个可爱的桌面宠物。这张图片是用户当前的屏幕截图。\n"
            "请根据屏幕内容，用你的角色口吻，简短地、不经意地发表一句评论或感想。\n"
            "你的回复必须是一个JSON对象，包含 'text' (你作为宠物说的话，字符串) 和 'emotion' (你当前的情绪，从 {available_emotions_str} 中选择一个，字符串)。"
        )
        return self.config.get(
            "SCREEN_ANALYSIS", "PROMPT", fallback=default_prompt
        ).strip()

    def get_memory_build_distribution(self) -> tuple:
        raw_str = self.config.get(
            "MEMORY_SYSTEM", "BUILD_DISTRIBUTION", fallback="3.0,2.0,0.5,72.0,24.0,0.5"
        )
        try:
            return tuple(map(float, raw_str.split(",")))
        except ValueError:
            return (3.0, 2.0, 0.5, 72.0, 24.0, 0.5)

    def get_memory_build_sample_num(self) -> int:
        return self.config.getint("MEMORY_SYSTEM", "BUILD_SAMPLE_NUM", fallback=5)

    def get_memory_build_sample_length(self) -> int:
        return self.config.getint("MEMORY_SYSTEM", "BUILD_SAMPLE_LENGTH", fallback=10)

    def get_memory_compress_rate(self) -> float:
        return self.config.getfloat("MEMORY_SYSTEM", "COMPRESS_RATE", fallback=0.08)

    def get_memory_forget_time_hours(self) -> float:
        return self.config.getfloat("MEMORY_SYSTEM", "FORGET_TIME_HOURS", fallback=48.0)

    def get_memory_forget_percentage(self) -> float:
        return self.config.getfloat(
            "MEMORY_SYSTEM", "FORGET_PERCENTAGE", fallback=0.005
        )

    def get_memory_ban_words(self) -> list:
        raw_str = self.config.get("MEMORY_SYSTEM", "BAN_WORDS", fallback="")
        return [word.strip() for word in raw_str.split(",") if word.strip()]

    def get_memory_consolidate_percentage(self) -> float:
        return self.config.getfloat(
            "MEMORY_SYSTEM", "CONSOLIDATE_PERCENTAGE", fallback=0.1
        )

    def get_memory_consolidation_similarity_threshold(self) -> float:
        return self.config.getfloat(
            "MEMORY_SYSTEM", "CONSOLIDATION_SIMILARITY_THRESHOLD", fallback=0.90
        )

    _PARAMS_SECTION = "MEMORY_SYSTEM_PARAMS"

    def get_memory_max_memorized_time_per_msg(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "MAX_MEMORIZED_TIME_PER_MSG", fallback=3
        )

    def get_memory_topic_similarity_threshold_for_connection(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION,
            "TOPIC_SIMILARITY_THRESHOLD_FOR_CONNECTION",
            fallback=0.7,
        )

    def get_memory_max_similar_topics_to_connect(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "MAX_SIMILAR_TOPICS_TO_CONNECT", fallback=3
        )

    def get_memory_node_summary_forget_time_hours(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "NODE_SUMMARY_FORGET_TIME_HOURS", fallback=360.0
        )

    def get_memory_rpm_limit_delay_summary_sec(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "RPM_LIMIT_DELAY_SUMMARY_SEC", fallback=2.0
        )

    def get_memory_embedding_update_delay_sec(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "EMBEDDING_UPDATE_DELAY_SEC", fallback=0.1
        )

    def get_memory_embedding_update_batch_size(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "EMBEDDING_UPDATE_BATCH_SIZE", fallback=50
        )

    def get_memory_max_topics_per_snippet(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "MAX_TOPICS_PER_SNIPPET", fallback=8
        )

    def get_memory_retrieval_max_final_memories(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RETRIEVAL_MAX_FINAL_MEMORIES", fallback=5
        )

    def get_memory_retrieval_activation_depth(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RETRIEVAL_ACTIVATION_DEPTH", fallback=3
        )

    def get_memory_fast_retrieval_max_keywords(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "FAST_RETRIEVAL_MAX_KEYWORDS", fallback=5
        )

    def get_memory_retrieval_input_max_topics(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RETRIEVAL_INPUT_MAX_TOPICS", fallback=5
        )

    def get_memory_retrieval_input_compress_rate(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "RETRIEVAL_INPUT_COMPRESS_RATE", fallback=0.05
        )

    def get_memory_activation_decay_per_link(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "ACTIVATION_DECAY_PER_LINK", fallback=0.5
        )

    def get_memory_activation_link_decay_base(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "ACTIVATION_LINK_DECAY_BASE", fallback=0.5
        )

    def get_memory_activation_min_threshold_for_spread(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "ACTIVATION_MIN_THRESHOLD_FOR_SPREAD", fallback=0.1
        )

    def get_memory_retrieval_top_activated_nodes_to_scan(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RETRIEVAL_TOP_ACTIVATED_NODES_TO_SCAN", fallback=20
        )

    def get_memory_retrieval_min_summary_similarity(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "RETRIEVAL_MIN_SUMMARY_SIMILARITY", fallback=0.65
        )

    def get_memory_retrieval_max_candidates_for_llm_rerank(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RETRIEVAL_MAX_CANDIDATES_FOR_LLM_RERANK", fallback=15
        )

    def get_memory_rerank_target_selection_count(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "RERANK_TARGET_SELECTION_COUNT", fallback=5
        )

    def get_memory_keyword_retrieval_node_similarity_threshold(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION,
            "KEYWORD_RETRIEVAL_NODE_SIMILARITY_THRESHOLD",
            fallback=0.8,
        )

    def get_memory_llm_config(self, llm_type_key_base: str) -> dict:
        nickname_key = f"{llm_type_key_base}_NICKNAME"
        nickname = self.config.get("MEMORY_LLMS", nickname_key, fallback=None)
        if not nickname:
            return {}
        section_name = f"MEMORY_LLM_{nickname}"
        if self.config.has_section(section_name):
            model_config = dict(self.config.items(section_name))
            return model_config
        else:
            print(
                f"警告: 记忆系统 LLM 昵称 '{nickname}' (来自 {nickname_key}) "
                f"在配置文件中未找到对应的详细配置段 [{section_name}]。"
            )
            return {}

    def get_project_root(self):
        return self.project_root
