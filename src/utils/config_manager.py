import configparser
import os
import logging
from typing import Optional, List

logger = logging.getLogger("ConfigManager")


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
        logger.info(f"ConfigManager: 成功加载配置文件 {actual_config_path}")
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

    def get_memory_build_interval_seconds(self) -> int:
        return self.config.getint(
            "MEMORY_SYSTEM", "BUILD_INTERVAL_SECONDS", fallback=600
        )

    def get_memory_forget_interval_seconds(self) -> int:
        return self.config.getint(
            "MEMORY_SYSTEM", "FORGET_INTERVAL_SECONDS", fallback=3600
        )

    def get_memory_consolidate_interval_seconds(self) -> int:
        return self.config.getint(
            "MEMORY_SYSTEM", "CONSOLIDATE_INTERVAL_SECONDS", fallback=3600
        )

    def get_memory_run_consolidate_on_startup(self) -> bool:
        return self.config.getboolean(
            "MEMORY_SYSTEM", "RUN_CONSOLIDATE_ON_STARTUP", fallback=True
        )

    def get_memory_run_build_on_startup_after_consolidate(self) -> bool:
        return self.config.getboolean(
            "MEMORY_SYSTEM", "RUN_BUILD_ON_STARTUP_AFTER_CONSOLIDATE", fallback=True
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
            "PET", "PERSONA", fallback="你是一个友好、乐于助人的桌面Bot。"
        )

    def get_pet_agent_mode_emotions(self) -> str:
        return self.config.get(
            "PET", "AGENT_MODE_EMOTIONS", fallback="'neutral', 'focused', 'helpful'"
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

    def get_screen_analysis_task_timeout_seconds(self) -> int:
        return self.config.getint(
            "SCREEN_ANALYSIS", "TASK_TIMEOUT_SECONDS", fallback=60
        )

    def get_screen_analysis_min_interval_seconds(self) -> int:
        return self.config.getint(
            "SCREEN_ANALYSIS", "MIN_INTERVAL_SECONDS", fallback=60
        )

    def get_screen_analysis_max_interval_seconds(self) -> int:
        return self.config.getint(
            "SCREEN_ANALYSIS", "MAX_INTERVAL_SECONDS", fallback=300
        )

    def get_screen_analysis_chance(self) -> float:
        return self.config.getfloat("SCREEN_ANALYSIS", "CHANCE", fallback=0.1)

    def get_tts_enabled(self) -> bool:
        return self.config.getboolean("TTS", "ENABLED", fallback=False)

    def get_tts_api_base_url(self) -> str:
        return self.config.get("TTS", "API_BASE_URL", fallback="http://127.0.0.1:9880/")

    def get_tts_api_endpoint(self) -> str:
        return self.config.get("TTS", "API_ENDPOINT", fallback="tts")

    def get_tts_refer_wav_path_for_tone(self, tone: str) -> Optional[str]:
        """根据指定的语调获取参考WAV路径。"""
        key = f"REFER_WAV_PATH_{tone.lower()}"
        if self.config.has_option("TTS", key):
            return self.config.get("TTS", key, fallback=None)
        logger.warning(
            f"ConfigManager: 未在 [TTS] section 中找到键 '{key}' for tone '{tone}'"
        )
        return None

    def get_tts_prompt_text_for_tone(self, tone: str) -> Optional[str]:
        """根据指定的语调获取提示文本。"""
        key = f"PROMPT_TEXT_{tone.lower()}"
        if self.config.has_option("TTS", key):
            return self.config.get("TTS", key, fallback=None)
        logger.warning(
            f"ConfigManager: 未在 [TTS] section 中找到键 '{key}' for tone '{tone}'"
        )
        return None

    def get_tts_available_tones(self) -> List[str]:
        """获取可用的语调列表。"""
        tones_str = self.config.get("TTS", "AVAILABLE_TONES", fallback="normal")
        return [t.strip() for t in tones_str.split(",") if t.strip()]

    def get_tts_default_tone(self) -> str:
        """获取默认语调。"""
        return self.config.get("TTS", "DEFAULT_TONE", fallback="normal")

    def get_tts_prompt_language(self) -> str:
        return self.config.get("TTS", "PROMPT_LANGUAGE", fallback="zh")

    def get_tts_text_language(self) -> str:
        return self.config.get("TTS", "TEXT_LANGUAGE", fallback="zh")

    def get_tts_cut_punc_method(self) -> str:
        return self.config.get("TTS", "CUT_PUNC_METHOD", fallback="cut5")

    def get_tts_media_type(self, fallback="wav") -> str:
        return self.config.get("TTS", "MEDIA_TYPE", fallback=fallback)

    def get_tts_play_audio_timeout_seconds(self) -> int:
        return self.config.getint("TTS", "PLAY_AUDIO_TIMEOUT_SECONDS", fallback=45)

    def get_screen_analysis_prompt(self) -> str:
        default_prompt = (
            "你是{pet_name}，一个可爱的桌面Bot。这张图片是用户当前的屏幕截图。\n"
            "请根据屏幕内容，用你的角色口吻，简短地、不经意地发表一句评论或感想。\n"
            "你的回复必须是一个JSON对象，包含 'text' (你作为Bot说的话，字符串) 和 'emotion' (你当前的情绪，从 {available_emotions_str} 中选择一个，字符串)。"
        )
        return self.config.get(
            "SCREEN_ANALYSIS", "PROMPT", fallback=default_prompt
        ).strip()

    def get_agent_pyautogui_pause(self) -> float:
        return self.config.getfloat("AGENT", "PYAUTOGUI_PAUSE", fallback=0.25)

    def get_agent_active_window_delay_before_type(self) -> float:
        return self.config.getfloat(
            "AGENT", "ACTIVE_WINDOW_DELAY_BEFORE_TYPE", fallback=0.75
        )

    def get_agent_get_window_title_delay(self) -> float:
        return self.config.getfloat("AGENT", "GET_WINDOW_TITLE_DELAY", fallback=0.3)

    def get_agent_max_read_file_length(self) -> int:
        return self.config.getint("AGENT", "MAX_READ_FILE_LENGTH", fallback=2000)

    def get_agent_step_delay_seconds(self) -> float:
        return self.config.getfloat("AGENT", "STEP_DELAY_SECONDS", fallback=0.5)

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

    def get_memory_min_topics_per_snippet(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "MIN_TOPICS_PER_SNIPPET", fallback=1
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

    def get_memory_topic_calc_len_base(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "TOPIC_CALC_LEN_BASE", fallback=1.0
        )

    def get_memory_topic_calc_len_log_factor(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "TOPIC_CALC_LEN_LOG_FACTOR", fallback=1.0
        )

    def get_memory_topic_calc_len_short_thresh(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "TOPIC_CALC_LEN_SHORT_THRESH", fallback=5
        )

    def get_memory_topic_calc_len_very_short_val(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "TOPIC_CALC_LEN_VERY_SHORT_VAL", fallback=1
        )

    def get_memory_topic_calc_info_baseline(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "TOPIC_CALC_INFO_BASELINE", fallback=2.5
        )

    def get_memory_topic_calc_info_scale(self) -> float:
        return self.config.getfloat(
            self._PARAMS_SECTION, "TOPIC_CALC_INFO_SCALE", fallback=1.5
        )

    def get_memory_topic_calc_info_max_absolute(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "TOPIC_CALC_INFO_MAX_ABSOLUTE", fallback=5
        )

    def get_memory_sample_time_gap_threshold_seconds(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "SAMPLE_TIME_GAP_THRESHOLD_SECONDS", fallback=1800
        )

    def get_memory_sample_min_snippet_messages(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "SAMPLE_MIN_SNIPPET_MESSAGES", fallback=2
        )

    def get_memory_sample_max_snippet_messages(self) -> int:
        return self.config.getint(
            self._PARAMS_SECTION, "SAMPLE_MAX_SNIPPET_MESSAGES", fallback=50
        )

    def get_memory_llm_config(self, llm_type_key_base: str) -> dict:
        nickname_key = f"{llm_type_key_base}_NICKNAME"
        nickname = self.config.get("MEMORY_LLMS", nickname_key, fallback=None)
        if not nickname:
            return {}
        section_name = f"MEMORY_LLM_{nickname}"
        if self.config.has_section(section_name):
            model_config = dict(self.config.items(section_name))
            for key, value in model_config.items():
                if isinstance(value, str):
                    if value.lower() == "true":
                        model_config[key] = True
                    elif value.lower() == "false":
                        model_config[key] = False
                    elif value.isdigit():
                        model_config[key] = int(value)
                    else:
                        try:
                            model_config[key] = float(value)
                        except ValueError:
                            pass
            return model_config
        else:
            logger.warning(
                f"警告: 记忆系统 LLM 昵称 '{nickname}' (来自 {nickname_key}) "
                f"在配置文件中未找到对应的详细配置段 [{section_name}]。"
            )
            return {}

    def get_project_root(self):
        return self.project_root