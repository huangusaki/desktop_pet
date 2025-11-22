import os
import logging
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QGuiApplication
from PIL import Image, ImageDraw, ImageFont
from src.gui.chat_dialog import ChatDialog


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


@dataclass
class AppServices:
    """A container for all background service instances."""

    config_manager: Optional[Any] = None
    prompt_builder: Optional[Any] = None
    mongo_handler: Optional[Any] = None
    gemini_client: Optional[Any] = None
    agent_core: Optional[Any] = None
    hippocampus_manager: Optional[Any] = None
    relationship_manager: Optional[Any] = None
    available_emotions: List[str] = field(default_factory=lambda: ["default"])
    assets_path: Optional[str] = None
    avatar_base_path: Optional[str] = None
    bot_avatar_path: Optional[str] = None
    user_avatar_path: Optional[str] = None
    bot_window: Optional[Any] = None  # Reference to desktop window for web server


@dataclass
class GuiComponents:
    """A container for all top-level GUI component instances."""

    bot_window: Optional[Any] = None
    chat_dialog: Optional[ChatDialog] = None
    web_chat_window: Optional[Any] = None
    screen_analyzer: Optional[Any] = None
    memory_build_timer: Optional[QTimer] = None
    memory_forget_timer: Optional[QTimer] = None
    memory_consolidate_timer: Optional[QTimer] = None


class ApplicationContext:
    def __init__(self, project_root: str, asyncio_helper_class):
        self.project_root: str = project_root
        self.AsyncioHelper = asyncio_helper_class
        self._services: Optional[AppServices] = None
        self._gui: Optional[GuiComponents] = None
        self._web_server_started = False
        self._web_server_thread = None
        from src.utils.config_manager import ConfigManager

        self._ConfigManagerClass = ConfigManager
        from src.memory_system.memory_config import MemoryConfig

        self._MemoryConfigClass = MemoryConfig
        from src.memory_system.hippocampus_core import HippocampusManager

        self._HippocampusManagerClass = HippocampusManager
        from src.data.relationship_manager import RelationshipManager

        self._RelationshipManagerClass = RelationshipManager
        from src.database.mongo_handler import MongoHandler

        self._MongoHandlerClass = MongoHandler
        from src.utils.prompt_builder import PromptBuilder

        self._PromptBuilderClass = PromptBuilder
        from src.llm.gemini_client import GeminiClient

        self._GeminiClientClass = GeminiClient
        from src.gui.main_window import PetWindow

        self._PetWindow = PetWindow
        from src.gui.chat_dialog import ChatDialog

        self._ChatDialog = ChatDialog
        from src.core.screen_analyzer import ScreenAnalyzer

        self._ScreenAnalyzerClass = ScreenAnalyzer
        from src.gui.web_chat_window import WebChatWindow

        self._WebChatWindowClass = WebChatWindow
        try:
            from src.core.agent_core import AgentCore as AgentCore_Import

            self._AgentCoreClass = AgentCore_Import
        except ImportError:
            self._AgentCoreClass = None
            logger.warning("应用上下文: AgentCore 无法导入")
            
        logger.info("应用上下文初始化完成")

    def run(self) -> bool:
        init_future = self.AsyncioHelper.schedule_task(self._create_services())
        if not init_future:
            QMessageBox.critical(
                None,
                "Initialization Failed",
                "Could not schedule async service initialization.",
            )
            return False
        try:
            self._services = init_future.result(timeout=120)
        except Exception as e:
            logger.error(
                f"CRITICAL: Waiting for async services future failed: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                None,
                "Initialization Failed",
                f"Async service initialization failed: {e}",
            )
            return False
        if not self._services or not self._services.gemini_client:
            QMessageBox.critical(
                None,
                "Initialization Failed",
                "Critical services (like Gemini) failed to initialize. Exiting.",
            )
            return False
        app = QApplication.instance()
        self._gui = self._create_gui(self._services, app)
        if not self._gui or not self._gui.bot_window:
            logger.critical("主程序: GUI组件初始化失败,正在退出")
            return False
        
        # Set bot_window reference in services for web server access
        self._services.bot_window = self._gui.bot_window
        
        self._connect_signals()
        
        # 启动Web服务器（在显示窗口之前）
        logger.info("正在启动Web服务器...")
        self._start_web_server()
        self._web_server_started = True
        
        self._perform_startup_tasks(app)
        initial_message_future = self.AsyncioHelper.schedule_task(self._get_initial_bot_message())
        try:
            initial_message = initial_message_future.result(timeout=5)
        except Exception as e:
            logger.warning(f"无法获取初始消息: {e}")
            initial_message = "start！"
        self._gui.bot_window.update_speech_and_emotion(initial_message, "default")
        self._gui.bot_window.show()
        return True

    async def _create_services(self) -> Optional[AppServices]:
        logger.info("正在创建服务...")
        os.makedirs(os.path.join(self.project_root, "config"), exist_ok=True)
        assets_path = os.path.normpath(os.path.join(self.project_root, "src", "assets"))
        os.makedirs(assets_path, exist_ok=True)
        os.makedirs(os.path.join(self.project_root, "data"), exist_ok=True)
        available_emotions = _scan_and_update_available_emotions(assets_path)
        config_path = os.path.join(self.project_root, "config", "settings.ini")
        if not os.path.exists(config_path):
            logger.warning(
                f"配置文件 {config_path} 不存在,将创建模板"
            )
            try:
                with open(config_path, "w", encoding="utf-8") as cf:
                    cf.write("; Please fill in your API keys and settings.\n")
                logger.info(f"配置模板已创建于 {config_path}")
                return None
            except IOError as e:
                logger.error(f"无法创建配置文件: {e}")
                return None
        try:
            logger.info("正在初始化配置管理器...")
            config_manager = self._ConfigManagerClass(config_file=config_path)
            logger.info("配置管理器已初始化")
        except Exception as e:
            logger.error(f"ConfigManager initialization failed: {e}", exc_info=True)
            return None
        if http_proxy := config_manager.get_http_proxy():
            os.environ["HTTP_PROXY"] = http_proxy
        if https_proxy := config_manager.get_https_proxy():
            os.environ["HTTPS_PROXY"] = https_proxy
        
        logger.info("正在设置头像路径...")
        avatar_base_path = os.path.normpath(
            os.path.join(
                self.project_root, config_manager.get_avatar_base_path_relative()
            )
        )
        os.makedirs(avatar_base_path, exist_ok=True)
        bot_avatar_path = os.path.join(
            avatar_base_path, config_manager.get_bot_avatar_filename()
        )
        user_avatar_path = os.path.join(
            avatar_base_path, config_manager.get_user_avatar_filename()
        )
        if not os.path.exists(bot_avatar_path):
            _create_placeholder_avatar(bot_avatar_path, "B")
        if not os.path.exists(user_avatar_path):
            _create_placeholder_avatar(user_avatar_path, "U")
        
        services = AppServices(
            config_manager=config_manager,
            available_emotions=available_emotions,
            assets_path=assets_path,
            avatar_base_path=avatar_base_path,
            bot_avatar_path=bot_avatar_path,
            user_avatar_path=user_avatar_path,
        )
        
        logger.info("正在初始化MongoDB处理器...")
        try:
            mongo_handler = self._MongoHandlerClass(
                config_manager.get_mongo_connection_string(),
                config_manager.get_mongo_database_name(),
                config_manager.get_mongo_collection_name(),
            )
            if mongo_handler.is_connected():
                services.mongo_handler = mongo_handler
                logger.info("MongoDB处理器已连接")
            else:
                logger.warning("MongoDB处理器连接失败")
        except Exception as e:
            logger.critical(f"MongoDB初始化异常: {e}", exc_info=True)
        
        if services.mongo_handler:
            logger.info("正在初始化关系管理器...")
            services.relationship_manager = self._RelationshipManagerClass(
                mongo_handler=services.mongo_handler,
                user_name=config_manager.get_user_name(),
                bot_name=config_manager.get_bot_name(),
            )
        else:
            logger.error(
                "无法初始化关系管理器,因为MongoDB未连接"
            )
        
        logger.info("正在初始化提示构建器...")
        services.prompt_builder = self._PromptBuilderClass(
            config_manager=config_manager,
            relationship_manager=services.relationship_manager,
        )
        
        # 根据配置选择主要LLM提供商
        primary_provider = config_manager.get_primary_llm_provider()
        logger.info(f"正在初始化主要LLM客户端 (提供商: {primary_provider})...")
        
        if primary_provider == "openai":
            # 使用 OpenAI 作为主要LLM
            try:
                from src.llm.openai_client import OpenAIClient
                
                api_key = config_manager.get_openai_api_key()
                if not api_key or api_key == "YOUR_API_KEY_HERE" or not api_key.strip():
                    logger.critical("OpenAI API密钥错误: 请在 config/settings.ini 中配置 OpenAI API密钥")
                    return None
                
                # 创建OpenAI客户端包装器,使其兼容GeminiClient接口
                logger.warning("OpenAI客户端作为主LLM尚未完全集成到聊天系统,将回退到Gemini")
                primary_provider = "gemini"  # 暂时回退
                
            except Exception as e:
                logger.error(f"OpenAI客户端初始化失败: {e}, 回退到Gemini", exc_info=True)
                primary_provider = "gemini"
        
        # 默认使用 Gemini 或回退到 Gemini
        if primary_provider == "gemini":
            try:
                api_key = config_manager.get_gemini_api_key()
                if not api_key or api_key == "YOUR_API_KEY_HERE":
                    logger.critical("API密钥错误: 请在 config/settings.ini 中配置 Gemini API密钥")
                    return None
                    
                services.gemini_client = self._GeminiClientClass(
                    api_key=api_key,
                    model_name=config_manager.get_gemini_model_name(),
                    bot_name=config_manager.get_bot_name(),
                    user_name=config_manager.get_user_name(),
                    bot_persona=config_manager.get_bot_persona(),
                    prompt_builder=services.prompt_builder,
                    available_emotions=services.available_emotions,
                    mongo_handler=services.mongo_handler,
                    config_manager=config_manager,
                )
                logger.info("Gemini客户端已初始化")
            except Exception as e:
                logger.critical(
                    f"Gemini客户端初始化异常: {e}", exc_info=True
                )
                return None

            
        if self._AgentCoreClass and services.gemini_client:
            logger.info("正在初始化Agent核心...")
            try:
                services.agent_core = self._AgentCoreClass(
                    config_manager=config_manager,
                    prompt_builder=services.prompt_builder,
                    gemini_client=services.gemini_client,
                )
                logger.info("Agent核心已初始化")
            except Exception as e:
                logger.error(f"初始化Agent核心失败: {e}", exc_info=True)
                
        if self._HippocampusManagerClass and services.mongo_handler:
            logger.info("正在初始化记忆系统管理器...")
            try:
                mem_config = self._MemoryConfigClass.from_config_manager(config_manager)
                hippocampus_manager = await self._HippocampusManagerClass.get_instance()
                if hippocampus_manager:
                    await hippocampus_manager.initialize_singleton(
                        memory_config=mem_config,
                        database_instance=services.mongo_handler.get_database(),
                        chat_collection_name=config_manager.get_mongo_collection_name(),
                        bot_name=config_manager.get_bot_name(),
                        global_llm_params=None,
                        prompt_builder=services.prompt_builder,
                    )
                    services.hippocampus_manager = hippocampus_manager
                    logger.info(
                        "记忆系统(HippocampusManager)初始化成功"
                    )
            except Exception as e:
                QMessageBox.warning(
                    None,
                    "Memory System Init Warning",
                    f"An error occurred while initializing the memory system: {e}",
                )
                logger.error(
                    f"记忆系统(Hippocampus)初始化失败: {e}",
                    exc_info=True,
                )
        return services

    def _create_gui(self, services: AppServices, app: QApplication) -> GuiComponents:
        """
        Creates all GUI components and injects their dependencies.
        Crucially, it does not create circular dependencies.
        """
        initial_image_filename = (
            services.config_manager.get_bot_initial_image_filename()
        )
        initial_pet_image_path = os.path.join(
            services.assets_path, initial_image_filename
        )
        if not os.path.exists(initial_pet_image_path):
            _create_placeholder_avatar(initial_pet_image_path, "Bot", size=(120, 120))
            
        bot_window = self._PetWindow(
            initial_image_path=initial_pet_image_path,
            assets_base_path=services.assets_path,
            available_emotions=services.available_emotions,
            config_manager=services.config_manager,
            agent_core=services.agent_core,
        )
        
        # Preload WebChatWindow for instant opening
        logger.info("正在预加载Web聊天窗口...")
        try:
            if hasattr(self, '_WebChatWindowClass') and self._WebChatWindowClass:
                WebChatWindow = self._WebChatWindowClass
            else:
                from src.gui.web_chat_window import WebChatWindow
                
            web_chat_window = WebChatWindow(
                url="http://localhost:8765",
                parent=None,
                avatar_path=services.bot_avatar_path
            )
            logger.info("Web聊天窗口预加载成功")
        except Exception as e:
            logger.error(f"预加载Web聊天窗口失败: {e}")
            web_chat_window = None

        screen_analyzer = None
        screen_analysis_available = False
        if (
            services.config_manager.get_screen_analysis_enabled()
            and services.gemini_client
        ):
            screen_analyzer = self._ScreenAnalyzerClass(
                gemini_client=services.gemini_client,
                prompt_builder=services.prompt_builder,
                config_manager=services.config_manager,
                bot_name=services.config_manager.get_bot_name(),
                user_name=services.config_manager.get_user_name(),
                available_emotions=services.available_emotions,
                parent=app,
            )
            screen_analysis_available = True
            logger.info("屏幕分析功能可用")
        else:
            logger.info(
                "屏幕分析功能不可用(已在配置中禁用或缺少依赖)"
            )
        bot_window.set_initial_feature_states(
            screen_analysis_available=screen_analysis_available,
            screen_analysis_enabled=services.config_manager.get_screen_analysis_enabled(),
            agent_core_available=services.agent_core is not None,
        )
        
        gui = GuiComponents(bot_window=bot_window, screen_analyzer=screen_analyzer)
        gui.web_chat_window = web_chat_window
        return gui

    def _connect_signals(self):
        """Manages all signal-slot connections between components."""
        if not self._gui or not self._services:
            return
        if self._gui.bot_window:
            self._gui.bot_window.request_open_chat_dialog.connect(
                self.open_chat_dialog_handler
            )
            self._gui.bot_window.agent_mode_toggled_signal.connect(
                self.handle_agent_mode_toggled
            )
        if self._gui.screen_analyzer and self._gui.bot_window:
            self._gui.screen_analyzer.pet_reaction_ready.connect(
                self._gui.bot_window.update_speech_and_emotion
            )
            self._gui.screen_analyzer.pet_reaction_ready.connect(
                self.handle_screen_analysis_reaction
            )
            self._gui.screen_analyzer.request_pet_hide.connect(
                self._gui.bot_window.handle_hide_request
            )
            self._gui.screen_analyzer.request_pet_show.connect(
                self._gui.bot_window.handle_show_request
            )
            self._gui.bot_window.screen_analysis_toggled_signal.connect(
                self._gui.screen_analyzer.set_monitoring_state
            )
    def open_chat_dialog_handler(self):
        """打开聊天窗口 - 使用PyQt WebEngine窗口"""
        try:
            if not self._gui or not self._services:
                return
            
            logger.info("正在打开Web聊天窗口...")
            
            # Web服务器已经在启动时运行了，直接显示窗口
            self._show_web_chat_window()
            
        except Exception as e:
            logger.error(f"打开Web聊天窗口错误: {e}", exc_info=True)
            QMessageBox.warning(
                None,
                "无法打开聊天窗口",
                f"打开聊天窗口时出错: {e}"
            )
    
    def _show_web_chat_window(self):
        """显示Web聊天窗口"""
        try:
            logger.info("Entering _show_web_chat_window")
            # 创建或显示WebEngine窗口
            if not hasattr(self._gui, 'web_chat_window') or self._gui.web_chat_window is None:
                logger.info("Creating WebChatWindow instance...")
                
                if hasattr(self, '_WebChatWindowClass') and self._WebChatWindowClass:
                    WebChatWindow = self._WebChatWindowClass
                else:
                    logger.warning("WebChatWindowClass not preloaded, importing now...")
                    from src.gui.web_chat_window import WebChatWindow
                
                self._gui.web_chat_window = WebChatWindow(
                    url="http://localhost:8765",
                    parent=None,
                    avatar_path=self._services.bot_avatar_path
                )
                logger.info("WebChatWindow instance created")
            
            logger.info("Calculating window position...")
            # 定位窗口在Bot旁边
            if self._gui.bot_window:
                bot_geo = self._gui.bot_window.geometry()
                screen_geometry = self._gui.bot_window.screen().availableGeometry()
                
                window_width = self._gui.web_chat_window.width()
                window_height = self._gui.web_chat_window.height()
                
                # 默认显示在Bot右侧
                new_x = bot_geo.x() + bot_geo.width() + 20
                new_y = bot_geo.y()
                
                # 如果右侧放不下，放左侧
                if new_x + window_width > screen_geometry.right():
                    new_x = bot_geo.x() - window_width - 20
                
                # 确保不超出屏幕边界
                if new_y + window_height > screen_geometry.bottom():
                    new_y = screen_geometry.bottom() - window_height
                if new_y < screen_geometry.top():
                    new_y = screen_geometry.top()
                
                self._gui.web_chat_window.move(new_x, new_y)

            # 确保窗口没有被最小化或隐藏
            if self._gui.web_chat_window.isHidden():
                self._gui.web_chat_window.show()
            
            self._gui.web_chat_window.raise_()
            self._gui.web_chat_window.activateWindow()
            
            logger.info("Web chat window opened successfully")
        except Exception as e:
            logger.error(f"Error showing web chat window: {e}", exc_info=True)

    def _start_web_server(self):
        """在后台线程启动FastAPI Web服务器"""
        try:
            from src.web import create_app
            from src.utils.logger_config import get_uvicorn_log_config
            import uvicorn
            import threading
            
            logger.info("正在启动聊天界面Web服务器...")
            
            # 创建FastAPI应用
            app = create_app(self._services)
            
            # 在后台线程运行服务器
            def run_server():
                try:
                    uvicorn.run(
                        app,
                        host="0.0.0.0",  # 允许手机访问
                        port=8765,
                        log_level="info",
                        log_config=get_uvicorn_log_config()  # 使用我们的日志配置
                    )
                except Exception as e:
                    logger.error(f"Web服务器错误: {e}", exc_info=True)
            
            self._web_server_thread = threading.Thread(target=run_server, daemon=True)
            self._web_server_thread.start()
            
            logger.info("Web服务器已启动于 http://localhost:8765")
            logger.info("也可从移动设备访问: http://<你的IP>:8765")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}", exc_info=True)
            QMessageBox.critical(
                None,
                "Web服务器启动失败",
                f"无法启动Web聊天服务器: {e}\n\n请检查端口8765是否被占用。"
            )

    def _perform_startup_tasks(self, app: QApplication):
        """Schedules background tasks like memory maintenance and screen analysis."""
        if self._gui and self._gui.screen_analyzer:
            if self._gui.screen_analyzer._is_enabled:
                self._gui.screen_analyzer.start_monitoring()
        cfg = self._services.config_manager
        run_consolidate = cfg.get_memory_run_consolidate_on_startup()
        run_build = cfg.get_memory_run_build_on_startup_after_consolidate()

        async def _startup_tasks():
            if run_consolidate:
                await self._run_memory_consolidate()
                if run_build:
                    await self._run_memory_build()
            elif run_build:
                await self._run_memory_build()

        if (run_consolidate or run_build) and self._services.hippocampus_manager:
            self.AsyncioHelper.schedule_task(_startup_tasks())
        self._schedule_memory_tasks(app)

    def handle_agent_mode_toggled(self, is_active: bool):
        if self._services.agent_core and hasattr(
            self._services.agent_core, "set_agent_mode"
        ):
            self._services.agent_core.set_agent_mode(is_active)
        else:
            logger.warning(
                "Agent core not found or does not have set_agent_mode method."
            )
        if self._gui.chat_dialog:
            self._gui.chat_dialog.set_agent_mode_active(is_active)

    async def handle_screen_analysis_reaction(
        self, text: str, emotion: str, image_description: str
    ):
        bot_name = self._services.config_manager.get_bot_name()
        user_name = self._services.config_manager.get_user_name()
        
        description_part = ""
        if image_description and image_description.strip():
            description_part = f"，发现里面的内容是：“{image_description.strip()}”"
        
        full_text_for_bot = (
            f"（{bot_name}看了一眼{user_name}的屏幕{description_part}）{text}"
        )
        
        self._gui.bot_window.update_speech_and_emotion(text, emotion)
        
        if self._gui.chat_dialog and not self._gui.chat_dialog.isHidden():
            if not self._gui.chat_dialog.is_agent_mode_active_chat:
                self._gui.chat_dialog._add_message_to_display(
                    bot_name, text, is_user=False
                )
        
        if self._services.mongo_handler and self._services.mongo_handler.is_connected():
            if self._services.config_manager.get_screen_analysis_save_to_chat_history():
                await self._services.mongo_handler.insert_chat_message(
                    bot_name, text, is_user=False
                )
            else:
                await self._services.mongo_handler.insert_screen_analysis_log_entry(
                    bot_name, f"[Screen Log] {full_text_for_bot}", bot_name
                )
        else:
            logger.warning("Main: MongoDB not connected, cannot save screen reaction.")

    async def _run_memory_build(self):
        if (
            self._services.hippocampus_manager
            and self._services.hippocampus_manager._initialized
        ):
            await self._services.hippocampus_manager.build_memory()

    async def _run_memory_forget(self):
        if (
            self._services.hippocampus_manager
            and self._services.hippocampus_manager._initialized
        ):
            await self._services.hippocampus_manager.forget_memory()

    async def _run_memory_consolidate(self):
        if (
            self._services.hippocampus_manager
            and self._services.hippocampus_manager._initialized
        ):
            await self._services.hippocampus_manager.consolidate_memory()

    def _schedule_memory_tasks(self, app: QApplication):
        if not (
            self._services.hippocampus_manager
            and self._services.hippocampus_manager._initialized
        ):
            logger.info(
                "Memory system not initialized or imported, skipping memory maintenance scheduling."
            )
            return
        cfg = self._services.config_manager
        build_ms = cfg.get_memory_build_interval_seconds() * 1000
        forget_ms = cfg.get_memory_forget_interval_seconds() * 1000
        consolidate_ms = cfg.get_memory_consolidate_interval_seconds() * 1000
        self._gui.memory_build_timer = self._create_and_start_timer(
            app, self._run_memory_build, build_ms, "Build"
        )
        self._gui.memory_forget_timer = self._create_and_start_timer(
            app, self._run_memory_forget, forget_ms, "Forget"
        )
        self._gui.memory_consolidate_timer = self._create_and_start_timer(
            app, self._run_memory_consolidate, consolidate_ms, "Consolidate"
        )

    def _create_and_start_timer(self, parent, coro_func, interval_ms, task_name):
        def trigger():
            logger.info(f"QTimer: Triggering memory {task_name} task.")
            future = self.AsyncioHelper.schedule_task(coro_func())
            if future:
                future.add_done_callback(
                    lambda f: logger.info(f"Async memory {task_name} task complete.")
                )

        timer = QTimer(parent)
        timer.timeout.connect(trigger)
        timer.start(interval_ms)
        logger.info(
            f"记忆{task_name}任务已计划每 {interval_ms // 60000} 分钟运行一次"
        )
        return timer

    async def _get_initial_bot_message(self) -> str:
        if self._services.mongo_handler and self._services.mongo_handler.is_connected():
            bot_name = self._services.config_manager.get_bot_name()
            try:
                history = self._services.mongo_handler.get_recent_chat_history(
                    1, bot_name
                )
                if history and history[0].get("sender") == bot_name:
                    return history[0].get("message_text", "start！")
            except Exception as e:
                logger.warning(f"Could not retrieve initial message from history: {e}")
        return "start！"

    def shutdown(self):
        logger.info("Shutting down ApplicationContext resources...")
        if self._gui:
            if self._gui.screen_analyzer:
                self._gui.screen_analyzer.stop_monitoring()
            if self._gui.memory_build_timer and self._gui.memory_build_timer.isActive():
                self._gui.memory_build_timer.stop()
            if (
                self._gui.memory_forget_timer
                and self._gui.memory_forget_timer.isActive()
            ):
                self._gui.memory_forget_timer.stop()
            if (
                self._gui.memory_consolidate_timer
                and self._gui.memory_consolidate_timer.isActive()
            ):
                self._gui.memory_consolidate_timer.stop()
        if self._services and self._services.mongo_handler:
            self._services.mongo_handler.close_connection()
        self.AsyncioHelper.stop_asyncio_loop()
        logger.info("ApplicationContext shutdown complete.")
