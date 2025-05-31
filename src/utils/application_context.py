import os
import logging
from typing import List, Optional, Any
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog
from PyQt6.QtCore import QTimer
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ApplicationContext")


def _create_placeholder_avatar(
    image_path: str, text: str, size=(64, 64), bg_color=(128, 128, 128, 200)
):
    if not Image or not ImageDraw or not ImageFont:
        logger.warning(
            f"Pillow not available, cannot create placeholder for {image_path}"
        )
        return
    try:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font_size = size[1] // 2
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_width, text_height = 0, 0
        x_offset, y_offset = 0, 0
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_offset = bbox[0]
            y_offset = bbox[1]
        elif hasattr(draw, "textsize"):
            text_width, text_height = draw.textsize(text, font=font)
        else:
            font_size_fallback = font.size if hasattr(font, "size") else font_size
            text_width = len(text) * font_size_fallback / 1.8
            text_height = font_size_fallback
        x = (size[0] - text_width) / 2 - x_offset
        y = (size[1] - text_height) / 2 - y_offset
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        img.save(image_path)
        logger.info(f"已创建占位头像图片: {image_path}")
    except Exception as e:
        logger.error(f"为 {image_path} 创建占位头像图片失败: {e}")


def _scan_and_update_available_emotions(assets_path: str) -> List[str]:
    if not os.path.isdir(assets_path):
        logger.error(
            f"错误: 资源路径 {assets_path} 不是一个有效的目录。情绪列表将使用默认值。"
        )
        return ["default"]
    found_emotions = set()
    for filename in os.listdir(assets_path):
        if filename.lower().endswith(".png"):
            found_emotions.add(os.path.splitext(filename)[0].lower())
    if not found_emotions:
        logger.warning(
            f"警告: 资源路径 {assets_path} 中未找到任何PNG图片。添加 'default'。"
        )
        found_emotions.add("default")
    if "default" not in found_emotions:
        logger.warning(
            f"警告: 'default.png' 未在 {assets_path} 中找到。建议添加。已自动添加'default'。"
        )
        found_emotions.add("default")
    emotions_list = sorted(list(found_emotions))
    if not emotions_list:
        emotions_list = ["default"]
    logger.info(f"可用的情绪列表已更新: {emotions_list}")
    return emotions_list


class ApplicationContext:
    def __init__(self, project_root: str, asyncio_helper_class):
        self.project_root: str = project_root
        self.AsyncioHelper = asyncio_helper_class
        self.config_manager: Optional[Any] = None
        self.prompt_builder: Optional[Any] = None
        self.gemini_client: Optional[Any] = None
        self.mongo_handler: Optional[Any] = None
        self.hippocampus_manager: Optional[Any] = None
        self.screen_analyzer: Optional[Any] = None
        self.agent_core: Optional[Any] = None
        self.pet_window: Optional[Any] = None
        self.chat_dialog: Optional[QDialog] = None
        self.assets_path: Optional[str] = None
        self.avatar_base_path: Optional[str] = None
        self.pet_avatar_path: Optional[str] = None
        self.user_avatar_path: Optional[str] = None
        self.available_emotions: List[str] = ["default"]
        self.memory_build_timer: Optional[QTimer] = None
        self.memory_forget_timer: Optional[QTimer] = None
        self.memory_consolidate_timer: Optional[QTimer] = None
        from src.utils.config_manager import ConfigManager

        self._ConfigManagerClass = ConfigManager
        from src.utils.prompt_builder import PromptBuilder

        self._PromptBuilderClass = PromptBuilder
        from src.llm.gemini_client import GeminiClient

        self._GeminiClientClass = GeminiClient
        from src.gui.main_window import PetWindow

        self._PetWindow = PetWindow
        from src.gui.chat_dialog import ChatDialog

        self._ChatDialog = ChatDialog
        from src.database.mongo_handler import MongoHandler

        self._MongoHandlerClass = MongoHandler
        from src.core.screen_analyzer import ScreenAnalyzer

        self._ScreenAnalyzerClass = ScreenAnalyzer
        from src.memory_system.memory_config import MemoryConfig

        self._MemoryConfigClass = MemoryConfig
        from src.memory_system.hippocampus_core import HippocampusManager

        self._HippocampusManagerClass = HippocampusManager
        try:
            from src.core.agent_core import AgentCore as AgentCore_Import

            self._AgentCoreClass = AgentCore_Import
        except ImportError:
            self._AgentCoreClass = None
            logger.warning(
                "ApplicationContext: AgentCore could not be imported. Agent mode might be unavailable."
            )

    def setup_environment_and_config(self) -> bool:
        logger.info("ApplicationContext: Setting up environment and configuration...")
        os.makedirs(os.path.join(self.project_root, "config"), exist_ok=True)
        self.assets_path = os.path.normpath(
            os.path.join(self.project_root, "src", "assets")
        )
        os.makedirs(self.assets_path, exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "data"), exist_ok=True)
        self.available_emotions = _scan_and_update_available_emotions(self.assets_path)
        config_file_relative_path = os.path.join("config", "settings.ini")
        actual_config_file_path = os.path.join(
            self.project_root, config_file_relative_path
        )
        if not os.path.exists(actual_config_file_path):
            logger.warning(
                f"配置文件 {actual_config_file_path} 不存在。将创建一个模板。"
            )
            try:
                with open(actual_config_file_path, "w", encoding="utf-8") as cf:
                    cf.write("; Default settings.ini content\n")
                    cf.write(
                        "[GEMINI]\nAPI_KEY = YOUR_API_KEY_HERE\nMODEL_NAME = gemini-1.5-flash-latest\nHTTP_PROXY =\nHTTPS_PROXY =\n\n"
                    )
                    cf.write(
                        "[PET]\nINITIAL_IMAGE_FILENAME = default.png\nNAME = 小助手\nPERSONA = 你是一个友好、乐于助人的桌面Bot...\nAGENT_MODE_EMOTIONS = neutral, focused, helpful\n\n"
                    )
                    cf.write("[USER]\nNAME = 主人\n\n")
                    cf.write(
                        "[AVATARS]\nAVATAR_BASE_PATH = src/assets/icon\nPET_AVATAR_FILENAME = bot.png\nUSER_AVATAR_FILENAME = user.png\n\n"
                    )
                    cf.write(
                        "[MONGODB]\nCONNECTION_STRING = mongodb://localhost:27017/\nDATABASE_NAME = desktop_pet_db\nCOLLECTION_NAME = chat_history\nHISTORY_COUNT_FOR_PROMPT = 5\n\n"
                    )
                    cf.write(
                        "[SCREEN_ANALYSIS]\nENABLED = False\nINTERVAL_SECONDS = 60\nCHANCE = 0.1\nSAVE_REACTION_TO_CHAT_HISTORY = True\nPROMPT = ...\n\n"
                    )
                    cf.write("; --- Memory System Configuration ---\n")
                    cf.write(
                        "[MEMORY_SYSTEM]\nBUILD_DISTRIBUTION = 3.0,2.0,0.5,72.0,24.0,0.5\nBUILD_SAMPLE_NUM = 5\nBUILD_SAMPLE_LENGTH = 10\nCOMPRESS_RATE = 0.08\nFORGET_TIME_HOURS = 48.0\nFORGET_PERCENTAGE = 0.005\nBAN_WORDS = 我,你,它,的,了,呢,吧,啊,哦,嗯\nCONSOLIDATE_PERCENTAGE = 0.1\nCONSOLIDATION_SIMILARITY_THRESHOLD = 0.90\n"
                        "BUILD_INTERVAL_SECONDS = 600\nFORGET_INTERVAL_SECONDS = 3600\nCONSOLIDATE_INTERVAL_SECONDS = 3600\n" # 确保这些间隔配置也在
                        "RUN_CONSOLIDATE_ON_STARTUP = True\n"  # 新增
                        "RUN_BUILD_ON_STARTUP_AFTER_CONSOLIDATE = True\n\n"  # 新增
                    )
                    cf.write(
                        "[MEMORY_SYSTEM_PARAMS]\nMAX_MEMORIZED_TIME_PER_MSG = 3\nKEYWORD_RETRIEVAL_NODE_SIMILARITY_THRESHOLD = 0.8\n\n"
                    )
                    cf.write(
                        "[MEMORY_LLMS]\nLLM_TOPIC_JUDGE_NICKNAME = gemini_flash_mem\nLLM_RE_RANK_NICKNAME = gemini_pro_mem\n\n"
                    )
                    cf.write(
                        "[MEMORY_LLM_gemini_flash_mem]\nname = gemini-1.5-flash-latest\nkey = YOUR_API_KEY_HERE ; Or actual key if not using env var for LLM_request\n\n"
                    )
                    cf.write(
                        "[MEMORY_LLM_embedding_google]\nname = text-embedding-004\nkey = YOUR_API_KEY_HERE\n\n"
                    )
                QMessageBox.information(
                    None,
                    "配置文件创建成功",
                    f"配置文件模板已创建于:\n{actual_config_file_path}\n\n请配置后重新运行。",
                )
                return False
            except IOError as e:
                QMessageBox.critical(None, "错误", f"无法创建配置文件: {e}")
                return False
        try:
            self.config_manager = self._ConfigManagerClass(
                config_file=actual_config_file_path
            )
        except FileNotFoundError as e:
            logger.error(f"ConfigManager FileNotFoundError: {e}", exc_info=True)
            QMessageBox.critical(
                None, "配置错误", f"ConfigManager无法找到配置文件: {e}\n程序将退出。"
            )
            return False
        except Exception as e:
            logger.error(
                f"ConfigManager encountered an unexpected error during initialization: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                None, "配置错误", f"加载配置文件时发生未知错误: {e}\n程序将退出。"
            )
            return False
        initial_image_filename = self.config_manager.get_pet_initial_image_filename()
        initial_pet_image_abs_path = os.path.join(
            self.assets_path, initial_image_filename
        )
        if not os.path.exists(initial_pet_image_abs_path):
            logger.warning(
                f"初始Bot图片 {initial_pet_image_abs_path} 不存在。创建占位图。"
            )
            _create_placeholder_avatar(
                initial_pet_image_abs_path, "Pet", size=(120, 120)
            )
            if (
                initial_image_filename.lower() == "default.png"
                and "default" not in self.available_emotions
            ):
                self.available_emotions.append("default")
                self.available_emotions.sort()
        avatar_base_path_relative = self.config_manager.get_avatar_base_path_relative()
        self.avatar_base_path = os.path.normpath(
            os.path.join(self.project_root, avatar_base_path_relative)
        )
        os.makedirs(self.avatar_base_path, exist_ok=True)
        pet_avatar_filename = self.config_manager.get_pet_avatar_filename()
        self.pet_avatar_path = os.path.join(self.avatar_base_path, pet_avatar_filename)
        if not os.path.exists(self.pet_avatar_path):
            _create_placeholder_avatar(self.pet_avatar_path, "P")
        user_avatar_filename = self.config_manager.get_user_avatar_filename()
        self.user_avatar_path = os.path.join(
            self.avatar_base_path, user_avatar_filename
        )
        if not os.path.exists(self.user_avatar_path):
            _create_placeholder_avatar(self.user_avatar_path, "U")
        http_proxy = self.config_manager.get_http_proxy()
        https_proxy = self.config_manager.get_https_proxy()
        if http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
        return True

    async def initialize_async_services(self) -> bool:
        if not self.config_manager:
            logger.critical(
                "CRITICAL: ConfigManager not initialized before initialize_async_services."
            )
            return False
        self.prompt_builder = self._PromptBuilderClass(self.config_manager)
        mongo_ok = False
        try:
            conn_str = self.config_manager.get_mongo_connection_string()
            db_name = self.config_manager.get_mongo_database_name()
            coll_name = self.config_manager.get_mongo_collection_name()
            self.mongo_handler = self._MongoHandlerClass(conn_str, db_name, coll_name)
            if (
                self.mongo_handler.is_connected()
                and self.mongo_handler.get_database() is not None
            ):
                mongo_ok = True
            else:
                QMessageBox.warning(
                    None,
                    "MongoDB连接警告",
                    "无法连接到MongoDB或数据库不可用。聊天记录和记忆系统功能将受限。",
                )
                logger.error("ERROR: MongoDB connection failed or DB not accessible.")
                self.mongo_handler = None
        except Exception as e:
            QMessageBox.warning(
                None,
                "MongoDB初始化严重错误",
                f"初始化 MongoDB 时发生严重错误: {e}。\n程序将无法正常运行。",
            )
            logger.critical(
                f"CRITICAL ERROR: MongoDB initialization exception: {e}", exc_info=True
            )
            self.mongo_handler = None
            mongo_ok = False
        if not mongo_ok:
            logger.critical(
                "CRITICAL: MongoDB initialization failed. Aborting further async service initialization."
            )
        gemini_ok = False
        try:
            chat_api_key = self.config_manager.get_gemini_api_key()
            chat_model_name = self.config_manager.get_gemini_model_name()
            pet_name = self.config_manager.get_pet_name()
            user_name = self.config_manager.get_user_name()
            pet_persona = self.config_manager.get_pet_persona()
            if not chat_api_key or chat_api_key == "YOUR_API_KEY_HERE":
                QMessageBox.critical(
                    None,
                    "API Key 错误",
                    "请在 config/settings.ini 中配置主聊天 Gemini API Key。",
                )
                logger.critical(
                    "CRITICAL ERROR: Gemini API Key missing or placeholder."
                )
            else:
                self.gemini_client = self._GeminiClientClass(
                    api_key=chat_api_key,
                    model_name=chat_model_name,
                    pet_name=pet_name,
                    user_name=user_name,
                    pet_persona=pet_persona,
                    prompt_builder=self.prompt_builder,
                    available_emotions=self.available_emotions,
                    mongo_handler=self.mongo_handler,
                    config_manager=self.config_manager,
                )
                gemini_ok = True
        except Exception as e:
            QMessageBox.critical(None, "Gemini客户端初始化错误", f"错误: {e}")
            logger.critical(
                f"CRITICAL ERROR: Gemini client initialization exception: {e}",
                exc_info=True,
            )
            self.gemini_client = None
        if not gemini_ok:
            logger.critical(
                "CRITICAL: Gemini client initialization failed. Aborting further service initialization."
            )
            return False
        if (
            self._AgentCoreClass
            and self.config_manager
            and self.prompt_builder
            and self.gemini_client
        ):
            try:
                self.agent_core = self._AgentCoreClass(
                    config_manager=self.config_manager,
                    prompt_builder=self.prompt_builder,
                    gemini_client=self.gemini_client,
                )
            except Exception as e:
                logger.error(f"Failed to initialize AgentCore: {e}", exc_info=True)
                self.agent_core = None
        else:
            missing_deps_agent = []
            if not self._AgentCoreClass:
                missing_deps_agent.append("AgentCore module")
            if not self.config_manager:
                missing_deps_agent.append("ConfigManager")
            logger.error(
                f"Cannot initialize AgentCore due to missing dependencies: {', '.join(missing_deps_agent)}."
            )
            self.agent_core = None
        if (
            self._HippocampusManagerClass
            and self._MemoryConfigClass
            and self.mongo_handler
            and self.mongo_handler.get_database() is not None
        ):
            try:
                mem_config = self._MemoryConfigClass.from_config_manager(
                    self.config_manager
                )
                pet_name_for_hippocampus = self.config_manager.get_pet_name()
                self.hippocampus_manager = (
                    await self._HippocampusManagerClass.get_instance()
                )
                if not self.hippocampus_manager:
                    logger.critical(
                        "CRITICAL: HippocampusManager.get_instance() returned None."
                    )
                    return False
                await self.hippocampus_manager.initialize_singleton(
                    memory_config=mem_config,
                    database_instance=self.mongo_handler.get_database(),
                    chat_collection_name=self.config_manager.get_mongo_collection_name(),
                    pet_name=pet_name_for_hippocampus,
                    global_llm_params=None,
                    prompt_builder=self.prompt_builder,
                )
                logger.info("记忆系统 (HippocampusManager) 初始化成功。")
            except Exception as e:
                QMessageBox.warning(
                    None,
                    "记忆系统初始化警告",
                    f"初始化记忆系统时发生错误: {e}\n记忆功能可能不可用。",
                )
                logger.error(
                    f"ERROR: Memory system (HippocampusManager) initialization failed: {e}",
                    exc_info=True,
                )
                self.hippocampus_manager = None
        else:
            if not (self.mongo_handler and self.mongo_handler.get_database()):
                QMessageBox.warning(
                    None,
                    "记忆系统跳过",
                    "由于数据库连接失败或未初始化，记忆系统未初始化。",
                )
            self.hippocampus_manager = None
        return True

    def initialize_gui_components(
        self,
        app: QApplication,
        open_chat_dialog_handler,
        handle_agent_mode_toggled_handler,
        handle_screen_analysis_reaction_handler,
    ):
        if not self.config_manager or not self.assets_path:
            QMessageBox.critical(None, "配置错误", "ConfigManager 或资源路径未初始化。")
            return False
        initial_image_filename = self.config_manager.get_pet_initial_image_filename()
        initial_pet_image_abs_path = os.path.join(
            self.assets_path, initial_image_filename
        )
        self.pet_window = self._PetWindow(
            initial_image_path=initial_pet_image_abs_path,
            assets_base_path=self.assets_path,
            available_emotions=self.available_emotions,
            app_context=self,
        )
        self.pet_window.request_open_chat_dialog.connect(open_chat_dialog_handler)
        self.pet_window.agent_mode_toggled_signal.connect(
            handle_agent_mode_toggled_handler
        )
        tts_globally_enabled = self.config_manager.get_tts_enabled()
        screen_analysis_feature_enabled = (
            self.config_manager.get_screen_analysis_enabled()
        )
        if (
            self.gemini_client
            and self.prompt_builder
            and self.pet_window
            and (tts_globally_enabled or screen_analysis_feature_enabled)
        ):
            user_name_for_analyzer = self.config_manager.get_user_name()
            self.screen_analyzer = self._ScreenAnalyzerClass(
                gemini_client=self.gemini_client,
                prompt_builder=self.prompt_builder,
                config_manager=self.config_manager,
                pet_window=self.pet_window,
                pet_name=self.config_manager.get_pet_name(),
                user_name=user_name_for_analyzer,
                available_emotions=self.available_emotions,
                parent=app,
            )
            if screen_analysis_feature_enabled:
                self.screen_analyzer.pet_reaction_ready.connect(
                    handle_screen_analysis_reaction_handler
                )
                self.screen_analyzer.start_monitoring()
                logger.info("屏幕检测开启.")
            else:
                logger.info("屏幕检测关闭.")
        return True

    def create_chat_dialog(self, open_chat_dialog_handler_ref):
        if not self.pet_window:
            QMessageBox.warning(None, "错误", "Bot窗口尚未初始化。")
            return None
        if not self.gemini_client:
            QMessageBox.warning(None, "服务未就绪", "Gemini 服务尚未初始化。")
            return None
        if not self.mongo_handler:
            QMessageBox.warning(None, "服务未就绪", "数据库服务尚未初始化。")
            return None
        if not self.config_manager:
            QMessageBox.warning(None, "服务未就绪", "配置服务尚未初始化。")
            return None
        if not self.pet_avatar_path or not self.user_avatar_path:
            QMessageBox.warning(None, "资源缺失", "头像路径未正确设置。")
            return None
        try:
            self.chat_dialog = self._ChatDialog(
                application_context=self,
                pet_avatar_path=self.pet_avatar_path,
                user_avatar_path=self.user_avatar_path,
                parent=self.pet_window,
            )
            self.chat_dialog.speech_and_emotion_received.connect(
                self.pet_window.update_speech_and_emotion
            )
            self.pet_window.agent_mode_toggled_signal.connect(
                self.chat_dialog.set_agent_mode_active
            )
            if self.screen_analyzer:
                self.chat_dialog.chat_text_for_tts_ready.connect(
                    self.screen_analyzer.play_tts_from_chat
                )
            return self.chat_dialog
        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(None, "聊天窗口错误", f"创建聊天窗口失败: {e}")
            self.chat_dialog = None
            return None

    async def _run_memory_build(self):
        if self.hippocampus_manager and self.hippocampus_manager._initialized:
            logger.info("定时任务：开始构建记忆...")
            try:
                await self.hippocampus_manager.build_memory()
                logger.info("定时任务：构建记忆完成。")
            except Exception as e:
                logger.error(f"定时任务：构建记忆时发生错误: {e}")
        else:
            logger.info("定时任务：跳过构建记忆，记忆系统未初始化或未导入。")

    async def _run_memory_forget(self):
        if self.hippocampus_manager and self.hippocampus_manager._initialized:
            logger.info("定时任务：开始遗忘记忆...")
            try:
                await self.hippocampus_manager.forget_memory()
                logger.info("定时任务：遗忘记忆完成。")
            except Exception as e:
                logger.error(f"定时任务：遗忘记忆时发生错误: {e}")
        else:
            logger.info("定时任务：跳过遗忘记忆，记忆系统未初始化或未导入。")

    async def _run_memory_consolidate(self):
        if self.hippocampus_manager and self.hippocampus_manager._initialized:
            logger.info("定时任务：开始整合记忆...")
            try:
                await self.hippocampus_manager.consolidate_memory()
                logger.info("定时任务：整合记忆完成。")
            except Exception as e:
                logger.error(f"定时任务：整合记忆时发生错误: {e}")
        else:
            logger.info("定时任务：跳过整合记忆，记忆系统未初始化或未导入。")

    def schedule_memory_tasks(self, app: QApplication):
        if not self.hippocampus_manager or not self.hippocampus_manager._initialized:
            logger.info("记忆系统未初始化或未导入，不调度记忆维护任务。")
            return
        if not self.config_manager:
            logger.warning("ConfigManager 未初始化，无法读取记忆任务间隔。使用默认值。")
            build_interval_ms = 3600 * 1000
            forget_interval_ms = 10800 * 1000
            consolidate_interval_ms = 21600 * 1000
        else:
            build_interval_s = self.config_manager.get_memory_build_interval_seconds()
            forget_interval_s = self.config_manager.get_memory_forget_interval_seconds()
            consolidate_interval_s = (
                self.config_manager.get_memory_consolidate_interval_seconds()
            )
            build_interval_ms = build_interval_s * 1000
            forget_interval_ms = forget_interval_s * 1000
            consolidate_interval_ms = consolidate_interval_s * 1000

        def trigger_build():
            logger.info("QTimer: Build memory triggered.")
            future = self.AsyncioHelper.schedule_task(self._run_memory_build())
            if future:
                future.add_done_callback(
                    lambda f: logger.info(f"Async build task completed: {f.result()}")
                )

        def trigger_forget():
            logger.info("QTimer: Forget memory triggered.")
            future = self.AsyncioHelper.schedule_task(self._run_memory_forget())
            if future:
                future.add_done_callback(
                    lambda f: logger.info(f"Async forget task completed: {f.result()}")
                )

        def trigger_consolidate():
            logger.info("QTimer: Consolidate memory triggered.")
            future = self.AsyncioHelper.schedule_task(self._run_memory_consolidate())
            if future:
                future.add_done_callback(
                    lambda f: logger.info(
                        f"Async consolidate task completed: {f.result()}"
                    )
                )

        self.memory_build_timer = QTimer(app)
        self.memory_build_timer.timeout.connect(trigger_build)
        self.memory_build_timer.start(build_interval_ms)
        logger.info(
            f"记忆构建任务已调度，每 {build_interval_ms // (60*1000)} 分钟运行一次。"
        )
        self.memory_forget_timer = QTimer(app)
        self.memory_forget_timer.timeout.connect(trigger_forget)
        self.memory_forget_timer.start(forget_interval_ms)
        logger.info(
            f"记忆遗忘任务已调度，每 {forget_interval_ms // (60*1000)} 分钟运行一次。"
        )
        self.memory_consolidate_timer = QTimer(app)
        self.memory_consolidate_timer.timeout.connect(trigger_consolidate)
        self.memory_consolidate_timer.start(consolidate_interval_ms)
        logger.info(
            f"记忆整合任务已调度，每 {consolidate_interval_ms // (60*1000)} 分钟运行一次。"
        )

    def perform_startup_tasks(self, app: QApplication):
        run_consolidate_on_startup = False
        run_build_after_consolidate_on_startup = False

        if self.config_manager:
            run_consolidate_on_startup = self.config_manager.get_memory_run_consolidate_on_startup()
            run_build_after_consolidate_on_startup = self.config_manager.get_memory_run_build_on_startup_after_consolidate()
        else:
            logger.warning("ConfigManager未初始化，无法读取启动时记忆任务配置，将使用默认值（不执行）。")


        if self.hippocampus_manager and self.hippocampus_manager._initialized:
            if self.AsyncioHelper._loop and self.AsyncioHelper._loop.is_running():

                async def _startup_tasks():
                    if run_consolidate_on_startup:
                        logger.info("Context: 根据配置，开始执行启动时记忆巩固任务。")
                        await self._run_memory_consolidate()
                        logger.info("Context: 启动时记忆巩固任务完成。")
                        if run_build_after_consolidate_on_startup:
                            logger.info("Context: 根据配置，在巩固后开始执行启动时记忆构建任务。")
                            await self._run_memory_build()
                            logger.info("Context: 启动时记忆构建任务完成。")
                        else:
                            logger.info("Context: 根据配置，启动时记忆巩固后不执行记忆构建。")
                    elif run_build_after_consolidate_on_startup: # 如果仅构建被设置为true，而巩固为false
                        logger.info("Context: 根据配置（仅构建），开始执行启动时记忆构建任务。")
                        await self._run_memory_build()
                        logger.info("Context: 启动时记忆构建任务完成。")
                    else:
                        logger.info("Context: 根据配置，启动时不执行记忆巩固和构建任务。")

                if run_consolidate_on_startup or run_build_after_consolidate_on_startup:
                    startup_mem_future = self.AsyncioHelper.schedule_task(
                        _startup_tasks()
                    )
                    if startup_mem_future:
                        logger.info("Context: 启动时记忆任务已调度到后台执行。")
                        startup_mem_future.add_done_callback(
                            lambda f: logger.info(
                                f"Async startup memory task(s) completed: {f.result() if not f.cancelled() else 'Cancelled'}"
                            )
                        )
                    else:
                        logger.error("Context: 无法调度启动时记忆任务。")
                else:
                    logger.info("Context: 根据配置，跳过所有启动时记忆任务。")
            else:
                logger.error(
                    "Context: Asyncio loop 不可用，无法执行启动时记忆任务。"
                )
        else:
            logger.info("Context: 跳过启动时记忆任务，记忆系统未初始化或未导入。")

        initial_pet_message_text = "start！"
        if (
            self.mongo_handler
            and self.mongo_handler.is_connected()
            and self.config_manager
        ):
            pet_name_for_history = self.config_manager.get_pet_name()
            recent_history = self.mongo_handler.get_recent_chat_history(
                count=1, role_play_character=pet_name_for_history
            )
            if recent_history:
                for message_doc in recent_history:
                    if message_doc.get("sender") == pet_name_for_history:
                        text = message_doc.get("message_text")
                        if text:
                            initial_pet_message_text = text
                        break
        if self.pet_window:
            self.pet_window.update_speech_and_emotion(
                initial_pet_message_text, "default"
            )
        self.schedule_memory_tasks(app)

    def shutdown(self):
        logger.info("关闭中……")
        if self.screen_analyzer:
            self.screen_analyzer.stop_monitoring()
            logger.info("屏幕监控监控已停止")
        if self.mongo_handler:
            self.mongo_handler.close_connection()
            logger.info("MongoDB连接已请求关闭")
        if self.memory_build_timer and self.memory_build_timer.isActive():
            self.memory_build_timer.stop()
        if self.memory_forget_timer and self.memory_forget_timer.isActive():
            self.memory_forget_timer.stop()
        if self.memory_consolidate_timer and self.memory_consolidate_timer.isActive():
            self.memory_consolidate_timer.stop()
        logger.info("记忆维护任务定时器已停止")
        self.AsyncioHelper.stop_asyncio_loop()
        logger.info("关闭完成")
