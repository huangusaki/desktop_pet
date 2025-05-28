import sys
import os
import asyncio
import threading
import coloredlogs
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from src.utils.config_manager import ConfigManager
from src.utils.prompt_builder import PromptBuilder
from src.llm.gemini_client import GeminiClient
from src.gui.main_window import PetWindow
from src.gui.chat_dialog import ChatDialog
from src.database.mongo_handler import MongoHandler
from src.core.screen_analyzer import ScreenAnalyzer
from src.llm.llm_request import LLM_request
from src.memory_system.memory_config import MemoryConfig
from src.memory_system.hippocampus_core import HippocampusManager
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger("main")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
config_manager_global: Optional[ConfigManager] = None
prompt_builder_global: Optional[PromptBuilder] = None
gemini_client_global: Optional[GeminiClient] = None
pet_window_global: Optional[PetWindow] = None
chat_dialog_global: Optional[ChatDialog] = None
mongo_handler_global: Optional[MongoHandler] = None
hippocampus_manager_global: Optional[HippocampusManager] = None
screen_analyzer_global: Optional[ScreenAnalyzer] = None
assets_path_global: Optional[str] = None
avatar_base_path_global: Optional[str] = None
pet_avatar_path_global: Optional[str] = None
user_avatar_path_global: Optional[str] = None
available_emotions_global: List[str] = ["default"]
MEMORY_BUILD_INTERVAL = 60 * 60 * 1000
MEMORY_FORGET_INTERVAL = 60 * 60 * 3 * 1000
MEMORY_CONSOLIDATE_INTERVAL = 60 * 60 * 6 * 1000
memory_build_timer: Optional[QTimer] = None
memory_forget_timer: Optional[QTimer] = None
memory_consolidate_timer: Optional[QTimer] = None


class AsyncioHelper:
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _thread: Optional[threading.Thread] = None
    _is_running_event: Optional[threading.Event] = None

    @staticmethod
    def start_asyncio_loop():
        if AsyncioHelper._loop is None or not AsyncioHelper._loop.is_running():
            AsyncioHelper._is_running_event = threading.Event()
            AsyncioHelper._loop = asyncio.new_event_loop()
            AsyncioHelper._thread = threading.Thread(
                target=AsyncioHelper._run_loop, daemon=True
            )
            AsyncioHelper._thread.start()
            AsyncioHelper._is_running_event.wait(timeout=5)

    @staticmethod
    def _run_loop():
        if AsyncioHelper._loop is None:
            logger.error("ERROR Main: _run_loop called with _loop as None")
            return
        asyncio.set_event_loop(AsyncioHelper._loop)
        if AsyncioHelper._is_running_event:
            AsyncioHelper._is_running_event.set()
        try:
            AsyncioHelper._loop.run_forever()
        except Exception as e:
            logger.error(
                f"ERROR Main: Asyncio loop in thread encountered an error: {e}"
            )
        finally:
            if hasattr(AsyncioHelper._loop, "shutdown_asyncgens"):
                AsyncioHelper._loop.run_until_complete(
                    AsyncioHelper._loop.shutdown_asyncgens()
                )
            AsyncioHelper._loop.close()

    @staticmethod
    def stop_asyncio_loop():
        if AsyncioHelper._loop and AsyncioHelper._loop.is_running():
            tasks = asyncio.all_tasks(loop=AsyncioHelper._loop)
            if tasks:

                async def cancel_all_and_stop():
                    if AsyncioHelper._loop is None:
                        return
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(
                        *[t for t in tasks if not t.done()], return_exceptions=True
                    )
                    AsyncioHelper._loop.stop()

                AsyncioHelper._loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        cancel_all_and_stop(), loop=AsyncioHelper._loop
                    )
                )
            else:
                AsyncioHelper._loop.call_soon_threadsafe(AsyncioHelper._loop.stop)
        if AsyncioHelper._thread and AsyncioHelper._thread.is_alive():
            AsyncioHelper._thread.join(timeout=5)
            if AsyncioHelper._thread.is_alive():
                logger.warning(
                    "WARNING Main: Asyncio thread did not stop gracefully after 5 seconds."
                )
        AsyncioHelper._loop = None
        AsyncioHelper._thread = None
        AsyncioHelper._is_running_event = None

    @staticmethod
    def schedule_task(coro) -> Optional[asyncio.Future]:
        if AsyncioHelper._loop and AsyncioHelper._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, AsyncioHelper._loop)
            return future
        else:
            logger.error(
                "ERROR Main: Asyncio loop not running or not initialized. Cannot schedule task."
            )
            if AsyncioHelper._loop is None:
                logger.warning(
                    "WARNING Main: Attempting to restart asyncio loop for task scheduling."
                )
                AsyncioHelper.start_asyncio_loop()
                if AsyncioHelper._loop and AsyncioHelper._loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(coro, AsyncioHelper._loop)
                    return future
            logger.error(
                "ERROR Main: Failed to schedule task even after attempting loop restart."
            )
            return None


def create_placeholder_avatar(
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
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size[0] - text_width) / 2
            y = (size[1] - text_height) / 2
            text_anchor_y_adjustment = bbox[1]
            y_draw = y - text_anchor_y_adjustment
        elif hasattr(draw, "textsize"):
            text_width, text_height = draw.textsize(text, font=font)
            x = (size[0] - text_width) / 2
            y = (size[1] - text_height) / 2
            y_draw = y
        else:
            text_width, text_height = (
                font.getsize(text)
                if hasattr(font, "getsize")
                else (
                    len(text)
                    * (font.size if hasattr(font, "size") else font_size)
                    / 1.5,
                    (font.size if hasattr(font, "size") else font_size),
                )
            )
            x = (size[0] - text_width) / 2
            y = (size[1] - text_height) / 2
            y_draw = y
        draw.text((x, y_draw), text, fill=(255, 255, 255, 255), font=font)
        img.save(image_path)
        logger.info(f"已创建占位头像图片: {image_path}")
    except Exception as e:
        logger.error(f"为 {image_path} 创建占位头像图片失败: {e}")


def scan_and_update_available_emotions(assets_path: str):
    global available_emotions_global
    if not os.path.isdir(assets_path):
        logger.error(
            f"错误: 资源路径 {assets_path} 不是一个有效的目录。情绪列表将使用默认值。"
        )
        available_emotions_global = ["default"]
        return
    found_emotions = set()
    for filename in os.listdir(assets_path):
        if filename.lower().endswith(".png"):
            found_emotions.add(os.path.splitext(filename)[0].lower())
    if not found_emotions:
        found_emotions.add("default")
    if "default" not in found_emotions:
        logger.warning(f"警告: 'default.png' 未在 {assets_path} 中找到。建议添加。")
        found_emotions.add("default")
    available_emotions_global = sorted(list(found_emotions))
    if not available_emotions_global:
        available_emotions_global = ["default"]
    logger.info(f"可用的情绪列表已更新: {available_emotions_global}")


def setup_environment_and_config() -> bool:
    global config_manager_global, assets_path_global, project_root
    global avatar_base_path_global, pet_avatar_path_global, user_avatar_path_global
    global available_emotions_global
    os.makedirs(os.path.join(project_root, "config"), exist_ok=True)
    assets_path_global = os.path.normpath(os.path.join(project_root, "src", "assets"))
    os.makedirs(assets_path_global, exist_ok=True)
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    scan_and_update_available_emotions(assets_path_global)
    config_file_relative_path = os.path.join("config", "settings.ini")
    actual_config_file_path = os.path.join(project_root, config_file_relative_path)
    if not os.path.exists(actual_config_file_path):
        logger.warning(f"配置文件 {actual_config_file_path} 不存在。将创建一个模板。")
        try:
            with open(actual_config_file_path, "w", encoding="utf-8") as cf:
                cf.write("; Default settings.ini content\n")
                cf.write(
                    "[GEMINI]\nAPI_KEY = YOUR_API_KEY_HERE\nMODEL_NAME = gemini-1.5-flash-latest\nHTTP_PROXY =\nHTTPS_PROXY =\n\n"
                )
                cf.write(
                    "[PET]\nINITIAL_IMAGE_FILENAME = default.png\nNAME = 小助手\nPERSONA = 你是一个友好、乐于助人的桌面Bot...\n\n"
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
                cf.write("[MEMORY_SYSTEM]\n")
                cf.write("BUILD_DISTRIBUTION = 3.0,2.0,0.5,72.0,24.0,0.5\n")
                cf.write(
                    "BUILD_SAMPLE_NUM = 5\nBUILD_SAMPLE_LENGTH = 10\nCOMPRESS_RATE = 0.08\n"
                )
                cf.write("FORGET_TIME_HOURS = 48.0\nFORGET_PERCENTAGE = 0.005\n")
                cf.write("BAN_WORDS = 我,你,它,的,了,呢,吧,啊,哦,嗯\n")
                cf.write(
                    "CONSOLIDATE_PERCENTAGE = 0.1\nCONSOLIDATION_SIMILARITY_THRESHOLD = 0.90\n\n"
                )
                cf.write("[MEMORY_SYSTEM_PARAMS]\n")
                cf.write("MAX_MEMORIZED_TIME_PER_MSG = 3\n")
                cf.write("KEYWORD_RETRIEVAL_NODE_SIMILARITY_THRESHOLD = 0.8\n\n")
                cf.write("[MEMORY_LLMS]\n")
                cf.write("LLM_TOPIC_JUDGE_NICKNAME = gemini_flash_mem\n")
                cf.write("LLM_RE_RANK_NICKNAME = gemini_pro_mem\n\n")
                cf.write("[MEMORY_LLM_gemini_flash_mem]\n")
                cf.write(
                    "name = gemini-1.5-flash-latest\nkey = YOUR_API_KEY_HERE ; Or actual key if not using env var for LLM_request\n\n"
                )
                cf.write("[MEMORY_LLM_embedding_google]\n")
                cf.write("name = text-embedding-004\nkey = YOUR_API_KEY_HERE\n\n")
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
        config_manager_global = ConfigManager(config_file=config_file_relative_path)
    except FileNotFoundError as e:
        QMessageBox.critical(None, "配置错误", f"无法加载配置文件: {e}\n程序将退出。")
        return False
    except Exception as e:
        QMessageBox.critical(
            None, "配置错误", f"加载配置文件时发生未知错误: {e}\n程序将退出。"
        )
        return False
    initial_image_filename = config_manager_global.get_pet_initial_image_filename()
    initial_pet_image_abs_path = os.path.join(
        assets_path_global, initial_image_filename
    )
    if not os.path.exists(initial_pet_image_abs_path):
        logger.warning(f"初始Bot图片 {initial_pet_image_abs_path} 不存在。创建占位图。")
        create_placeholder_avatar(initial_pet_image_abs_path, "Pet", size=(120, 120))
        if (
            initial_image_filename.lower() == "default.png"
            and "default" not in available_emotions_global
        ):
            available_emotions_global.append("default")
            available_emotions_global.sort()
    avatar_base_path_relative = config_manager_global.get_avatar_base_path_relative()
    avatar_base_path_global = os.path.normpath(
        os.path.join(project_root, avatar_base_path_relative)
    )
    os.makedirs(avatar_base_path_global, exist_ok=True)
    pet_avatar_filename = config_manager_global.get_pet_avatar_filename()
    pet_avatar_path_global = os.path.join(avatar_base_path_global, pet_avatar_filename)
    if not os.path.exists(pet_avatar_path_global):
        create_placeholder_avatar(pet_avatar_path_global, "P")
    user_avatar_filename = config_manager_global.get_user_avatar_filename()
    user_avatar_path_global = os.path.join(
        avatar_base_path_global, user_avatar_filename
    )
    if not os.path.exists(user_avatar_path_global):
        create_placeholder_avatar(user_avatar_path_global, "U")
    http_proxy = config_manager_global.get_http_proxy()
    https_proxy = config_manager_global.get_https_proxy()
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
    return True


async def initialize_async_services():
    global config_manager_global, prompt_builder_global, gemini_client_global, mongo_handler_global, hippocampus_manager_global
    global available_emotions_global
    if not config_manager_global:
        logger.critical(
            "CRITICAL: ConfigManager not initialized before initialize_async_services."
        )
        return False
    if prompt_builder_global is None and config_manager_global:
        prompt_builder_global = PromptBuilder(config_manager_global)
    mongo_ok = False
    try:
        conn_str = config_manager_global.get_mongo_connection_string()
        db_name = config_manager_global.get_mongo_database_name()
        coll_name = config_manager_global.get_mongo_collection_name()
        mongo_handler_global = MongoHandler(conn_str, db_name, coll_name)
        if mongo_handler_global.is_connected() and (
            mongo_handler_global.get_database() is not None
        ):
            mongo_ok = True
        else:
            QMessageBox.warning(
                None,
                "MongoDB连接警告",
                "无法连接到MongoDB或数据库不可用。聊天记录和记忆系统功能将受限。",
            )
            logger.error(
                "ERROR: MongoDB connection failed or DB not accessible after MongoHandler instantiation."
            )
            if not mongo_handler_global.is_connected():
                mongo_handler_global = None
    except Exception as e:
        QMessageBox.warning(
            None,
            "MongoDB初始化严重错误",
            f"初始化 MongoDB 时发生严重错误: {e}。\n程序将无法正常运行。",
        )
        logger.critical(f"CRITICAL ERROR: MongoDB initialization exception: {e}")
        import traceback

        traceback.print_exc()
        mongo_handler_global = None
        mongo_ok = False
    if not mongo_ok:
        logger.critical(
            "CRITICAL: MongoDB initialization failed. Aborting service initialization."
        )
        return False
    gemini_ok = False
    try:
        chat_api_key = config_manager_global.get_gemini_api_key()
        chat_model_name = config_manager_global.get_gemini_model_name()
        pet_name = config_manager_global.get_pet_name()
        user_name = config_manager_global.get_user_name()
        pet_persona = config_manager_global.get_pet_persona()
        if not chat_api_key or chat_api_key == "YOUR_API_KEY_HERE":
            QMessageBox.critical(
                None,
                "API Key 错误",
                "请在 config/settings.ini 中配置主聊天 Gemini API Key。",
            )
            logger.critical("CRITICAL ERROR: Gemini API Key missing or placeholder.")
        else:
            if not prompt_builder_global:
                logger.critical(
                    "CRITICAL ERROR: PromptBuilder not initialized before GeminiClient."
                )
                return False
            if not mongo_handler_global:
                logger.critical(
                    "CRITICAL ERROR: MongoHandler not initialized before GeminiClient (but mongo_ok was true, this is an inconsistency)."
                )
                return False
            gemini_client_global = GeminiClient(
                api_key=chat_api_key,
                model_name=chat_model_name,
                pet_name=pet_name,
                user_name=user_name,
                pet_persona=pet_persona,
                prompt_builder=prompt_builder_global,
                available_emotions=available_emotions_global,
                mongo_handler=mongo_handler_global,
                config_manager=config_manager_global,
            )
            gemini_ok = True
    except Exception as e:
        QMessageBox.critical(None, "Gemini客户端初始化错误", f"错误: {e}")
        logger.critical(f"CRITICAL ERROR: Gemini client initialization exception: {e}")
        import traceback

        traceback.print_exc()
        gemini_client_global = None
        gemini_ok = False
    if not gemini_ok:
        logger.critical(
            "CRITICAL: Gemini client initialization failed. Aborting service initialization."
        )
        return False
    if HippocampusManager is None or MemoryConfig is None:
        logger.warning(
            "WARNING: HippocampusManager or MemoryConfig was not imported, skipping its initialization."
        )
    elif mongo_handler_global and (mongo_handler_global.get_database() is not None):
        try:
            mem_config = MemoryConfig.from_config_manager(config_manager_global)
            memory_global_llm_params: Optional[Dict[str, Any]] = None
            pet_name_for_hippocampus = config_manager_global.get_pet_name()
            hippocampus_manager_global = await HippocampusManager.get_instance()
            if not prompt_builder_global:
                logger.critical(
                    "CRITICAL: PromptBuilder is None before Hippocampus init, cannot proceed."
                )
                return False
            if not hippocampus_manager_global:
                logger.critical(
                    "CRITICAL: HippocampusManager.get_instance() returned None, cannot proceed."
                )
                return False
            await hippocampus_manager_global.initialize_singleton(
                memory_config=mem_config,
                database_instance=mongo_handler_global.get_database(),
                chat_collection_name=config_manager_global.get_mongo_collection_name(),
                pet_name=pet_name_for_hippocampus,
                global_llm_params=memory_global_llm_params,
                prompt_builder=prompt_builder_global,
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
            )
            import traceback

            traceback.print_exc()
            hippocampus_manager_global = None
    else:
        QMessageBox.warning(
            None, "记忆系统跳过", "由于数据库连接失败或未初始化，记忆系统未初始化。"
        )
        logger.info(
            "INFO: Memory system skipped due to MongoDB issue or HippocampusManager/MemoryConfig not imported."
        )
        hippocampus_manager_global = None
    return True


def open_chat_dialog_handler():
    global chat_dialog_global, gemini_client_global, pet_window_global, mongo_handler_global
    global config_manager_global, pet_avatar_path_global, user_avatar_path_global, hippocampus_manager_global
    if chat_dialog_global is None:
        if not pet_window_global:
            QMessageBox.warning(None, "错误", "Bot窗口尚未初始化。")
            return
        if not gemini_client_global:
            QMessageBox.warning(None, "服务未就绪", "Gemini 服务尚未初始化。")
            return
        if not mongo_handler_global:
            QMessageBox.warning(None, "服务未就绪", "数据库服务尚未初始化。")
            return
        if not config_manager_global:
            QMessageBox.warning(None, "服务未就绪", "配置服务尚未初始化。")
            return
        if not pet_avatar_path_global or not user_avatar_path_global:
            QMessageBox.warning(None, "资源缺失", "头像路径未正确设置。")
            return
        try:
            chat_dialog_global = ChatDialog(
                gemini_client=gemini_client_global,
                mongo_handler=mongo_handler_global,
                config_manager=config_manager_global,
                hippocampus_manager=hippocampus_manager_global,
                pet_avatar_path=pet_avatar_path_global,
                user_avatar_path=user_avatar_path_global,
                parent=pet_window_global,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(None, "聊天窗口错误", f"创建聊天窗口失败: {e}")
            chat_dialog_global = None
            return
        if pet_window_global:
            chat_dialog_global.speech_and_emotion_received.connect(
                pet_window_global.update_speech_and_emotion
            )
        if screen_analyzer_global:
            chat_dialog_global.chat_text_for_tts_ready.connect(
                screen_analyzer_global.play_tts_from_chat
            )
            logger.info(
                "Main: Connected chat_text_for_tts_ready to screen_analyzer.play_tts_from_chat."
            )
        else:
            logger.warning(
                "Main: screen_analyzer_global is None. Cannot connect chat_text_for_tts_ready signal."
            )
    if chat_dialog_global:
        if chat_dialog_global.isHidden():
            if pet_window_global:
                pet_rect = pet_window_global.geometry()
                chat_dialog_global.adjustSize()
                chat_dialog_size = chat_dialog_global.size()
                screen = QGuiApplication.screenAt(pet_window_global.pos())
                if not screen:
                    screen = QGuiApplication.primaryScreen()
                screen_available_rect = screen.availableGeometry()
                target_x_left = pet_rect.x() - chat_dialog_size.width()
                target_y = pet_rect.y()
                if target_y < screen_available_rect.y():
                    target_y = screen_available_rect.y()
                if (
                    target_y + chat_dialog_size.height()
                    > screen_available_rect.bottom()
                ):
                    target_y = (
                        screen_available_rect.bottom() - chat_dialog_size.height()
                    )
                    if target_y < screen_available_rect.y():
                        target_y = screen_available_rect.y()
                if target_x_left >= screen_available_rect.x():
                    chat_dialog_global.move(target_x_left, target_y)
                else:
                    target_x_right = pet_rect.x() + pet_rect.width()
                    if (
                        target_x_right + chat_dialog_size.width()
                        <= screen_available_rect.right()
                    ):
                        chat_dialog_global.move(target_x_right, target_y)
                    else:
                        if target_x_left < screen_available_rect.x() and (
                            target_x_right + chat_dialog_size.width()
                            > screen_available_rect.right()
                        ):
                            clamped_x = max(screen_available_rect.x(), target_x_left)
                            chat_dialog_global.move(clamped_x, target_y)
                        elif target_x_left < screen_available_rect.x():
                            chat_dialog_global.move(target_x_right, target_y)
                        else:
                            clamped_x_right = min(
                                target_x_right,
                                screen_available_rect.right()
                                - chat_dialog_size.width(),
                            )
                            if (
                                target_x_right + chat_dialog_size.width()
                                > screen_available_rect.right()
                            ):
                                chat_dialog_global.move(
                                    max(screen_available_rect.x(), target_x_left),
                                    target_y,
                                )
                            else:
                                chat_dialog_global.move(target_x_right, target_y)
            try:
                chat_dialog_global.open_dialog()
            except Exception as e:
                logger.critical(
                    f"CRITICAL main.py: Error calling chat_dialog_global.open_dialog(): {e}"
                )
                import traceback

                traceback.print_exc()
                QMessageBox.critical(None, "聊天窗口错误", f"打开聊天窗口失败: {e}")
        else:
            chat_dialog_global.activateWindow()
            chat_dialog_global.raise_()


def handle_screen_analysis_reaction(text: str, emotion: str):
    global pet_window_global, chat_dialog_global, mongo_handler_global, config_manager_global
    if pet_window_global:
        pet_window_global.update_speech_and_emotion(text, emotion)
    if chat_dialog_global and not chat_dialog_global.isHidden():
        pet_name = (
            config_manager_global.get_pet_name() if config_manager_global else "Bot"
        )
        display_text = f"（看了一眼屏幕）{text}"
        chat_dialog_global._add_message_to_display(
            sender_name_for_log_only=pet_name, message=display_text, is_user=False
        )
    if (
        mongo_handler_global
        and mongo_handler_global.is_connected()
        and config_manager_global
    ):
        pet_name = config_manager_global.get_pet_name()
        db_text = f"{text}"
        save_to_chat_history = (
            config_manager_global.get_screen_analysis_save_to_chat_history()
        )
        if save_to_chat_history:
            mongo_handler_global.insert_chat_message(
                sender=pet_name,
                message_text=db_text,
                role_play_character=pet_name,
            )
            logger.info(f"Main: 屏幕反应 ('{text}') 已保存到主聊天记录。")
        else:
            mongo_handler_global.insert_screen_analysis_log_entry(
                sender=pet_name,
                message_text=db_text,
                role_play_character=pet_name,
            )
            logger.info(f"Main: 屏幕反应 ('{text}') 已保存到 screen_analysis_log 表。")
    elif not config_manager_global:
        logger.warning("Main: ConfigManager 未初始化，无法保存屏幕反应。")
    elif not (mongo_handler_global and mongo_handler_global.is_connected()):
        logger.warning("Main: MongoDB 未连接，无法保存屏幕反应。")


async def run_memory_build():
    if hippocampus_manager_global and hippocampus_manager_global._initialized:
        logger.info("定时任务：开始构建记忆...")
        try:
            await hippocampus_manager_global.build_memory()
            logger.info("定时任务：构建记忆完成。")
        except Exception as e:
            logger.error(f"定时任务：构建记忆时发生错误: {e}")
    else:
        logger.info("定时任务：跳过构建记忆，记忆系统未初始化或未导入。")


async def initial_memory_build():
    if hippocampus_manager_global and hippocampus_manager_global._initialized:
        logger.info("Main (Initial Build): 开始首次记忆构建...")
        try:
            await hippocampus_manager_global.build_memory()
            logger.info("Main (Initial Build): 首次记忆构建完成。")
        except Exception as e:
            logger.error(
                f"Main (Initial Build): 首次记忆构建时发生错误: {e}", exc_info=True
            )
    else:
        logger.info(
            "Main (Initial Build): 跳过首次记忆构建，记忆系统未初始化或未导入。"
        )


async def run_memory_forget():
    if hippocampus_manager_global and hippocampus_manager_global._initialized:
        logger.info("定时任务：开始遗忘记忆...")
        try:
            await hippocampus_manager_global.forget_memory()
            logger.info("定时任务：遗忘记忆完成。")
        except Exception as e:
            logger.error(f"定时任务：遗忘记忆时发生错误: {e}")
    else:
        logger.info("定时任务：跳过遗忘记忆，记忆系统未初始化或未导入。")


async def run_memory_consolidate():
    if hippocampus_manager_global and hippocampus_manager_global._initialized:
        logger.info("定时任务：开始整合记忆...")
        try:
            await hippocampus_manager_global.consolidate_memory()
            logger.info("定时任务：整合记忆完成。")
        except Exception as e:
            logger.error(f"定时任务：整合记忆时发生错误: {e}")
    else:
        logger.info("定时任务：跳过整合记忆，记忆系统未初始化或未导入。")


def schedule_memory_tasks(app: QApplication):
    global memory_build_timer, memory_forget_timer, memory_consolidate_timer, config_manager_global
    if not hippocampus_manager_global or not hippocampus_manager_global._initialized:
        logger.info("记忆系统未初始化或未导入，不调度记忆维护任务。")
        return
    if not config_manager_global:
        logger.warning(
            "ConfigManager 未初始化，无法读取记忆任务间隔。使用默认值 (1h, 3h, 6h)。"
        )
        build_interval_ms = 3600 * 1000
        forget_interval_ms = 10800 * 1000
        consolidate_interval_ms = 21600 * 1000
    else:
        build_interval_s = config_manager_global.get_memory_build_interval_seconds()
        forget_interval_s = config_manager_global.get_memory_forget_interval_seconds()
        consolidate_interval_s = (
            config_manager_global.get_memory_consolidate_interval_seconds()
        )
        build_interval_ms = build_interval_s * 1000
        forget_interval_ms = forget_interval_s * 1000
        consolidate_interval_ms = consolidate_interval_s * 1000

    def trigger_build():
        logger.info("QTimer: Build memory triggered.")
        future = AsyncioHelper.schedule_task(run_memory_build())
        if future:
            future.add_done_callback(
                lambda f: logger.info(
                    f"Async build task completed, result/exception: {f.result() if not f.cancelled() else 'Cancelled'}"
                )
            )

    def trigger_forget():
        logger.info("QTimer: Forget memory triggered.")
        future = AsyncioHelper.schedule_task(run_memory_forget())
        if future:
            future.add_done_callback(
                lambda f: logger.info(
                    f"Async forget task completed, result/exception: {f.result() if not f.cancelled() else 'Cancelled'}"
                )
            )

    def trigger_consolidate():
        logger.info("QTimer: Consolidate memory triggered.")
        future = AsyncioHelper.schedule_task(run_memory_consolidate())
        if future:
            future.add_done_callback(
                lambda f: logger.info(
                    f"Async consolidate task completed, result/exception: {f.result() if not f.cancelled() else 'Cancelled'}"
                )
            )

    memory_build_timer = QTimer(app)
    memory_build_timer.timeout.connect(trigger_build)
    memory_build_timer.start(build_interval_ms)
    logger.info(
        f"记忆构建任务已调度，每 {build_interval_s} 秒 ({build_interval_ms // (60*1000)} 分钟) 运行一次。"
    )
    memory_forget_timer = QTimer(app)
    memory_forget_timer.timeout.connect(trigger_forget)
    memory_forget_timer.start(forget_interval_ms)
    logger.info(
        f"记忆遗忘任务已调度，每 {forget_interval_s} 秒 ({forget_interval_ms // (60*1000)} 分钟) 运行一次。"
    )
    memory_consolidate_timer = QTimer(app)
    memory_consolidate_timer.timeout.connect(trigger_consolidate)
    memory_consolidate_timer.start(consolidate_interval_ms)
    logger.info(
        f"记忆整合任务已调度，每 {consolidate_interval_s} 秒 ({consolidate_interval_ms // (60*1000)} 分钟) 运行一次。"
    )


if __name__ == "__main__":
    coloredlogs.install(
        level="info",
        fmt="%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: %(message)s",
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "green"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red"},
            "critical": {"color": "red", "bold": True, "background": "white"},
        },
        field_styles={
            "asctime": {"color": "green"},
            "levelname": {"color": "green", "bold": True},
            "name": {"color": "blue"},
            "lineno": {"color": "blue"},
        },
    )
    if Image is None:
        if QApplication.instance() is None:
            _app_temp_pillow_check = QApplication(sys.argv)
        QMessageBox.critical(
            None,
            "依赖缺失",
            "Pillow 库未找到或无法导入。\n程序无法运行。请安装 Pillow: pip install Pillow",
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    AsyncioHelper.start_asyncio_loop()
    if not setup_environment_and_config():
        AsyncioHelper.stop_asyncio_loop()
        sys.exit(1)
    initialization_succeeded = False
    if AsyncioHelper._loop and AsyncioHelper._loop.is_running():

        async def do_initialization():
            try:
                return await initialize_async_services()
            except Exception as e:
                logger.error(
                    f"CRITICAL ERROR: Exception during initialize_async_services execution: {e}"
                )
                import traceback

                traceback.print_exc()
                return False

        init_future_concurrent = AsyncioHelper.schedule_task(do_initialization())
        if init_future_concurrent:
            try:
                initialization_succeeded = init_future_concurrent.result(timeout=60)
            except asyncio.TimeoutError:
                logger.error(
                    "CRITICAL ERROR: initialize_async_services timed out after 60 seconds."
                )
                initialization_succeeded = False
            except Exception as e:
                logger.error(
                    f"CRITICAL ERROR: Waiting for initialize_async_services future failed: {e}"
                )
                initialization_succeeded = False
        else:
            logger.error(
                "ERROR Main: Failed to schedule initialize_async_services task."
            )
            initialization_succeeded = False
    else:
        logger.error(
            "CRITICAL ERROR: AsyncioHelper loop not available for initialization."
        )
        initialization_succeeded = False
    if not initialization_succeeded:
        QMessageBox.critical(None, "初始化失败", "关键服务初始化失败，程序将退出。")
        AsyncioHelper.stop_asyncio_loop()
        sys.exit(1)
    if hippocampus_manager_global and hippocampus_manager_global._initialized: # 确保记忆系统已初始化
        if AsyncioHelper._loop and AsyncioHelper._loop.is_running():
            logger.info("Main: 准备执行启动时记忆巩固（去重）任务...")
            consolidate_future = AsyncioHelper.schedule_task(run_memory_consolidate()) # <--- 调度记忆巩固
            if consolidate_future:
                logger.info("Main: 启动时记忆巩固任务已调度到后台执行。")
                # 你可以选择等待它完成，或者让它在后台运行
                # 如果想等待（会阻塞启动直到它完成，不推荐用于耗时操作）：
                # try:
                #     consolidate_future.result(timeout=300) # 设置一个超时，例如5分钟
                #     logger.info("Main: 启动时记忆巩固任务完成。")
                # except asyncio.TimeoutError:
                #     logger.error("Main: 启动时记忆巩固任务超时。")
                # except Exception as e:
                #     logger.error(f"Main: 启动时记忆巩固任务执行出错: {e}")
            else:
                logger.error("Main: 无法调度启动时记忆巩固任务。")
        else:
            logger.error("Main: Asyncio loop 不可用，无法执行启动时记忆巩固任务。")
    else:
        logger.info("Main: 跳过启动时记忆巩固，记忆系统未初始化或未导入。")
    if not config_manager_global or not assets_path_global:
        QMessageBox.critical(
            None, "配置错误", "ConfigManager 或资源路径未初始化。程序将退出。"
        )
        AsyncioHelper.stop_asyncio_loop()
        sys.exit(1)
    initial_image_filename = config_manager_global.get_pet_initial_image_filename()
    initial_pet_image_abs_path = os.path.join(
        assets_path_global, initial_image_filename
    )
    pet_window_global = PetWindow(
        initial_image_path=initial_pet_image_abs_path,
        assets_base_path=assets_path_global,
        available_emotions=available_emotions_global,
    )
    pet_window_global.request_open_chat_dialog.connect(open_chat_dialog_handler)
    tts_globally_enabled_main = False
    screen_analysis_feature_enabled_main = False
    if config_manager_global:
        tts_globally_enabled_main = config_manager_global.get_tts_enabled()
        screen_analysis_feature_enabled_main = (
            config_manager_global.get_screen_analysis_enabled()
        )
    else:
        logger.warning(
            "CRITICAL main.py: config_manager_global is None before ScreenAnalyzer init logic. This should not happen."
        )
    if (
        gemini_client_global
        and config_manager_global
        and prompt_builder_global
        and pet_window_global
        and (tts_globally_enabled_main or screen_analysis_feature_enabled_main)
    ):
        user_name_for_analyzer = config_manager_global.get_user_name()
        screen_analyzer_global = ScreenAnalyzer(
            gemini_client=gemini_client_global,
            prompt_builder=prompt_builder_global,
            config_manager=config_manager_global,
            pet_window=pet_window_global,
            pet_name=config_manager_global.get_pet_name(),
            user_name=user_name_for_analyzer,
            available_emotions=available_emotions_global,
            parent=app,
        )
        if screen_analysis_feature_enabled_main:
            screen_analyzer_global.pet_reaction_ready.connect(
                handle_screen_analysis_reaction
            )
            screen_analyzer_global.start_monitoring()
            logger.info(
                "INFO Main: Screen analyzer monitoring started (feature enabled)."
            )
        else:
            logger.info(
                "INFO Main: Screen analyzer initialized for TTS, but screen monitoring feature is disabled."
            )
    else:
        if not (tts_globally_enabled_main or screen_analysis_feature_enabled_main):
            logger.info(
                "INFO Main: Screen analyzer not started (both TTS and Screen Analysis feature are disabled, or core dependencies missing)."
            )
        elif not (
            gemini_client_global
            and config_manager_global
            and prompt_builder_global
            and pet_window_global
        ):
            logger.info(
                "INFO Main: Screen analyzer not started (core dependencies like Gemini, Config, PromptBuilder, or PetWindow missing, though TTS or Screen Analysis was requested)."
            )
        else:
            logger.info(
                "INFO Main: Screen analyzer not started (unknown reason, check dependencies and enabled flags)."
            )
    initial_pet_message_text = "你好！我在这里哦！"
    if (
        mongo_handler_global
        and mongo_handler_global.is_connected()
        and config_manager_global
    ):
        pet_name_for_history = config_manager_global.get_pet_name()
        recent_history = mongo_handler_global.get_recent_chat_history(
            count=1, role_play_character=pet_name_for_history
        )
        if recent_history:
            for message_doc in recent_history:
                if message_doc.get("sender") == pet_name_for_history:
                    text = message_doc.get("message_text")
                    if text:
                        initial_pet_message_text = text
                        break
    pet_window_global.update_speech_and_emotion(initial_pet_message_text, "default")
    pet_window_global.show()
    schedule_memory_tasks(app)
    exit_code = app.exec()
    AsyncioHelper.stop_asyncio_loop()
    if screen_analyzer_global:
        screen_analyzer_global.stop_monitoring()
        logger.info("ScreenAnalyzer监控已停止。")
    if mongo_handler_global:
        mongo_handler_global.close_connection()
        logger.info("Main: MongoDB连接已请求关闭。")
    if memory_build_timer and memory_build_timer.isActive():
        memory_build_timer.stop()
    if memory_forget_timer and memory_forget_timer.isActive():
        memory_forget_timer.stop()
    if memory_consolidate_timer and memory_consolidate_timer.isActive():
        memory_consolidate_timer.stop()
    logger.info("记忆维护任务定时器已停止。")
    logger.info(f"应用程序退出，退出代码: {exit_code}")
    sys.exit(exit_code)
