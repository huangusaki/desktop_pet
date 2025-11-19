import sys
import os
import asyncio
import threading
import coloredlogs
import logging
from typing import Optional
from PyQt6.QtWidgets import QApplication, QMessageBox
from PIL import Image
from pathlib import Path

# 尝试尽早导入 WebEngine 以避免后续卡死
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    print("WebEngine loaded successfully in main")
except ImportError:
    print("WebEngine not found or failed to load in main")

script_file_path = Path(__file__).resolve()
project_root = str(script_file_path.parent.parent)

# 添加项目根目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.application_context import ApplicationContext

os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false;qt.multimedia.ffmpeg=false"
logger = logging.getLogger("main")


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
            None, "依赖缺失", "Pillow 库未找到。请安装 Pillow: pip install Pillow"
        )
        sys.exit(1)
    app = QApplication(sys.argv)
    context: Optional[ApplicationContext] = None
    exit_code = 0
    try:
        AsyncioHelper.start_asyncio_loop()
        context = ApplicationContext(project_root, AsyncioHelper)
        success = context.run()
        if success:
            exit_code = app.exec()
        else:
            logger.critical("Application context failed to run. Exiting.")
            exit_code = 1
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)
        QMessageBox.critical(None, "致命错误", f"发生未知错误，程序将退出: {e}")
        exit_code = 1
    finally:
        logger.info("Application is shutting down...")
        if context:
            context.shutdown()
        else:
            AsyncioHelper.stop_asyncio_loop()
        logger.info(f"应用程序退出，退出代码: {exit_code}")
        sys.exit(exit_code)
