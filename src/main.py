import sys
import os
import asyncio
import threading
import coloredlogs
from typing import Optional
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QGuiApplication
from PIL import Image
from src.utils.application_context import ApplicationContext
import logging

os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false;qt.multimedia.ffmpeg=false"
logger = logging.getLogger("main")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
context: Optional[ApplicationContext] = None


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
            pending = asyncio.all_tasks(loop=AsyncioHelper._loop)
            if pending:
                logger.info(
                    f"AsyncioHelper: Cancelling {len(pending)} pending tasks..."
                )
                for task in pending:
                    if not task.done():
                        task.cancel()
                AsyncioHelper._loop.call_soon_threadsafe(AsyncioHelper._loop.stop)
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
            return None


def open_chat_dialog_handler():
    global context
    if not context:
        logger.error("Context not available for open_chat_dialog_handler")
        return
    if context.chat_dialog is None:
        if not context.create_chat_dialog(open_chat_dialog_handler):
            logger.error("Failed to create chat dialog via context.")
            return
    if context.chat_dialog:
        if context.agent_core and hasattr(context.agent_core, "is_agent_mode_active"):
            context.chat_dialog.set_agent_mode_active(
                context.agent_core.is_agent_mode_active
            )
        if context.chat_dialog.isHidden():
            if context.pet_window:
                pet_rect = context.pet_window.geometry()
                context.chat_dialog.adjustSize()
                chat_dialog_size = context.chat_dialog.size()
                screen = QGuiApplication.screenAt(context.pet_window.pos())
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
                    context.chat_dialog.move(target_x_left, target_y)
                else:
                    target_x_right = pet_rect.x() + pet_rect.width()
                    if (
                        target_x_right + chat_dialog_size.width()
                        <= screen_available_rect.right()
                    ):
                        context.chat_dialog.move(target_x_right, target_y)
                    else:
                        if target_x_left < screen_available_rect.x() and (
                            target_x_right + chat_dialog_size.width()
                            > screen_available_rect.right()
                        ):
                            clamped_x = max(screen_available_rect.x(), target_x_left)
                            context.chat_dialog.move(clamped_x, target_y)
                        elif target_x_left < screen_available_rect.x():
                            context.chat_dialog.move(target_x_right, target_y)
                        else:
                            clamped_x_right = min(
                                target_x_right,
                                screen_available_rect.right()
                                - chat_dialog_size.width(),
                            )
                            context.chat_dialog.move(
                                max(screen_available_rect.x(), clamped_x_right),
                                target_y,
                            )
            try:
                context.chat_dialog.open_dialog()
            except Exception as e:
                logger.critical(
                    f"CRITICAL main.py: Error calling context.chat_dialog.open_dialog(): {e}",
                    exc_info=True,
                )
                QMessageBox.critical(None, "聊天窗口错误", f"打开聊天窗口失败: {e}")
        else:
            context.chat_dialog.activateWindow()
            context.chat_dialog.raise_()


def handle_agent_mode_toggled(is_active: bool):
    global context
    if not context:
        return
    if context.chat_dialog:
        context.chat_dialog.set_agent_mode_active(is_active)
    if context.agent_core and hasattr(context.agent_core, "set_active_mode"):
        context.agent_core.set_active_mode(is_active)


def handle_screen_analysis_reaction(text: str, emotion: str, image_description: str):
    global context
    if not context or not context.config_manager:
        return
    pet_name = context.config_manager.get_pet_name()
    user_name = context.config_manager.get_user_name()
    description_part = ""
    if image_description and image_description.strip():
        description_part = f"，发现里面的内容是：“{image_description.strip()}”"
    full_text_for_pet = (
        f"（{pet_name}看了一眼{user_name}的屏幕{description_part}）{text}"
    )
    if context.pet_window:
        context.pet_window.update_speech_and_emotion(text, emotion)
    if context.chat_dialog and not context.chat_dialog.isHidden():
        if not context.chat_dialog.is_agent_mode_active_chat:
            context.chat_dialog._add_message_to_display(
                sender_name_for_log_only=pet_name, message=text, is_user=False
            )
    if context.mongo_handler and context.mongo_handler.is_connected():
        db_text_to_store = full_text_for_pet
        save_to_chat_history = (
            context.config_manager.get_screen_analysis_save_to_chat_history()
        )
        if save_to_chat_history:
            context.mongo_handler.insert_chat_message(
                sender=pet_name,
                message_text=db_text_to_store,
                role_play_character=pet_name,
            )
            logger.debug(f"Main: 屏幕反应 ('{db_text_to_store}') 已保存到主聊天记录。")
        else:
            if hasattr(context.mongo_handler, "insert_screen_analysis_log_entry"):
                context.mongo_handler.insert_screen_analysis_log_entry(
                    sender=pet_name,
                    message_text=db_text_to_store,
                    role_play_character=pet_name,
                )
            else:
                context.mongo_handler.insert_chat_message(
                    sender=pet_name,
                    message_text=f"[Screen Log] {db_text_to_store}",
                    role_play_character=pet_name,
                )
            logger.debug(
                f"Main: 屏幕反应 ('{db_text_to_store}') 已保存到特定日志表或带标记保存。"
            )
    else:
        logger.warning("Main: MongoDB 未连接或ConfigManager不可用，无法保存屏幕反应。")


if __name__ == "__main__":
    coloredlogs.install(
        level="INFO",
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
        QMessageBox.critical(
            None,
            "依赖缺失",
            "Pillow 库未找到或无法导入。\n程序无法运行。请安装 Pillow: pip install Pillow",
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    AsyncioHelper.start_asyncio_loop()
    context = ApplicationContext(project_root, AsyncioHelper)
    if not context.setup_environment_and_config():
        logger.critical("Main: Environment and config setup failed. Exiting.")
        AsyncioHelper.stop_asyncio_loop()
        sys.exit(1)
    initialization_succeeded = False
    if AsyncioHelper._loop and AsyncioHelper._loop.is_running():

        async def do_initialization():
            try:
                return await context.initialize_async_services()
            except Exception as e:
                logger.error(
                    f"CRITICAL ERROR: Exception during context.initialize_async_services execution: {e}",
                    exc_info=True,
                )
                return False

        init_future = AsyncioHelper.schedule_task(do_initialization())
        if init_future:
            try:
                initialization_succeeded = init_future.result(timeout=120)
            except asyncio.TimeoutError:
                logger.error(
                    "CRITICAL ERROR: Async services initialization timed out after 120 seconds."
                )
                initialization_succeeded = False
            except Exception as e:
                logger.error(
                    f"CRITICAL ERROR: Waiting for async services future failed: {e}",
                    exc_info=True,
                )
                initialization_succeeded = False
        else:
            logger.error(
                "CRITICAL ERROR: Failed to schedule async services initialization task."
            )
            initialization_succeeded = False
    else:
        logger.error(
            "CRITICAL ERROR: AsyncioHelper loop not available for initialization."
        )
        initialization_succeeded = False
    if not initialization_succeeded:
        QMessageBox.critical(None, "初始化失败", "关键服务初始化失败，程序将退出。")
        if context:
            context.shutdown()
        else:
            AsyncioHelper.stop_asyncio_loop()
        sys.exit(1)
    if not context.initialize_gui_components(
        app,
        open_chat_dialog_handler,
        handle_agent_mode_toggled,
        handle_screen_analysis_reaction,
    ):
        logger.critical("Main: GUI components initialization failed. Exiting.")
        context.shutdown()
        sys.exit(1)
    context.perform_startup_tasks(app)
    if context.pet_window:
        context.pet_window.show()
    else:
        logger.critical("Main: PetWindow not initialized. Cannot show. Exiting.")
        context.shutdown()
        sys.exit(1)
    exit_code = app.exec()
    if context:
        context.shutdown()
    else:
        AsyncioHelper.stop_asyncio_loop()
    logger.info(f"应用程序退出，退出代码: {exit_code}")
    sys.exit(exit_code)
