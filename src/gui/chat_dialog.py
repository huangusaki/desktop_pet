import html
import os
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextBrowser,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QApplication,
    QWidget,
    QFileDialog,
    QLabel,
    QSizePolicy,
    QScrollArea,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QThread, QObject, QSize
from PyQt6.QtGui import QIcon, QPixmap, QDesktopServices
import asyncio
import sys
from typing import Optional, List, Any, Dict
import logging
import mimetypes

logger = logging.getLogger("ChatDialog")
try:
    from PIL import Image, ImageDraw, ImageOps

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    Image, ImageDraw, ImageOps = None, None, None
try:
    from ..memory_system.hippocampus_core import HippocampusManager
except ImportError:
    try:
        from memory_system.hippocampus_core import HippocampusManager
    except ImportError:
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
            from memory_system.hippocampus_core import HippocampusManager
        except ImportError:
            logger.critical("CRITICAL: Could not import HippocampusManager.")
            HippocampusManager = None
try:
    from ..core.agent_core import AgentCore
except ImportError:
    try:
        from core.agent_core import AgentCore
    except ImportError:
        AgentCore = None
        logger.warning(
            "ChatDialog: AgentCore could not be imported. Agent mode will be unavailable in chat."
        )
DISPLAY_AVATAR_SIZE = 28
STAGED_FILE_THUMBNAIL_SIZE = 64


class AsyncRunner(QObject):
    task_completed = pyqtSignal(object)
    task_failed = pyqtSignal(str)

    def __init__(self, coro, parent=None):
        super().__init__(parent)
        self.coro = coro

    def run_async_task(self):
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.coro)
            self.task_completed.emit(result)
        except Exception as e:
            self.task_failed.emit(str(e))


class ChatDialog(QDialog):
    dialog_closed = pyqtSignal()
    speech_and_emotion_received = pyqtSignal(str, str)
    chat_text_for_tts_ready = pyqtSignal(str, str)

    def __init__(
        self,
        application_context: Any,
        pet_avatar_path: str,
        user_avatar_path: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.context = application_context
        self.gemini_client = self.context.gemini_client
        self.mongo_handler = self.context.mongo_handler
        self.config_manager = self.context.config_manager
        self.hippocampus_manager = self.context.hippocampus_manager
        self.agent_core = self.context.agent_core
        self.is_agent_mode_active_chat = False
        if self.agent_core and hasattr(self.agent_core, "is_agent_mode_active"):
            self.is_agent_mode_active_chat = self.agent_core.is_agent_mode_active
        self.user_name = self.config_manager.get_user_name()
        self.pet_name = self.config_manager.get_pet_name()
        self.current_role_play_character = self.pet_name
        self.original_pet_avatar_path = pet_avatar_path
        self.original_user_avatar_path = user_avatar_path
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.avatar_cache_dir = os.path.join(self.current_script_dir, ".avatar_cache")
        if PILLOW_AVAILABLE:
            if not os.path.exists(self.avatar_cache_dir):
                try:
                    os.makedirs(self.avatar_cache_dir, exist_ok=True)
                except OSError as e:
                    logger.warning(
                        f"Warning: Could not create avatar cache directory: {self.avatar_cache_dir}. Error: {e}"
                    )
        self.pet_avatar_qurl, self.is_pet_avatar_processed_by_pillow = (
            self._initialize_avatar_qurl_and_flag(self.original_pet_avatar_path, "pet")
        )
        self.user_avatar_qurl, self.is_user_avatar_processed_by_pillow = (
            self._initialize_avatar_qurl_and_flag(
                self.original_user_avatar_path, "user"
            )
        )
        transparent_pixmap = QPixmap(1, 1)
        transparent_pixmap.fill(Qt.GlobalColor.transparent)
        self.setWindowIcon(QIcon(transparent_pixmap))
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container_widget = QWidget(self)
        container_widget.setObjectName("ChatContainer")
        layout = QVBoxLayout(container_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        self.staged_files_scroll_area = QScrollArea(self)
        self.staged_files_scroll_area.setWidgetResizable(True)
        self.staged_files_scroll_area.setFixedHeight(STAGED_FILE_THUMBNAIL_SIZE + 20)
        self.staged_files_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.staged_files_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.staged_files_widget = QWidget()
        self.staged_files_layout = QHBoxLayout(self.staged_files_widget)
        self.staged_files_layout.setContentsMargins(5, 5, 5, 5)
        self.staged_files_layout.setSpacing(10)
        self.staged_files_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.staged_files_widget.setLayout(self.staged_files_layout)
        self.staged_files_scroll_area.setWidget(self.staged_files_widget)
        self.staged_files_scroll_area.setVisible(False)
        layout.addWidget(self.staged_files_scroll_area)
        self.chat_display = QTextBrowser(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.document().setDefaultStyleSheet(
            "body { margin:0px; padding:0px; }"
        )
        self.chat_display.setStyleSheet(
            "QTextBrowser { background-color:transparent; color:#e0e0e0; border:none; padding:8px; font-size:10pt; }"
        )
        layout.addWidget(self.chat_display)
        input_area_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.returnPressed.connect(self.send_message_handler)
        self.input_field.setStyleSheet(
            "QLineEdit { background-color:rgba(50,50,50,0.9); color:#ffffff; border:1px solid rgba(80,80,80,0.8); border-radius:6px; padding:8px; font-size:10pt;} QLineEdit:focus {border:1px solid rgba(0,120,215,0.9);}"
        )
        input_area_layout.addWidget(self.input_field)
        self.attach_button = QPushButton(self)
        self.attach_button.setText("📎")
        self.attach_button.setToolTip("添加附件")
        self.attach_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach_button.setStyleSheet(
            "QPushButton { background-color:rgba(70,70,70,0.9); color:white; border:none; border-radius:6px; padding:8px; font-size:10pt;} QPushButton:hover{background-color:rgba(90,90,90,0.9);}"
        )
        self.attach_button.clicked.connect(self._handle_attach_file_dialog)
        input_area_layout.addWidget(self.attach_button)
        layout.addLayout(input_area_layout)
        self.send_button = QPushButton("发送", self)
        self.send_button.clicked.connect(self.send_message_handler)
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setStyleSheet(
            "QPushButton { background-color:#0078FF; color:white; border:none; border-radius:6px; padding:8px 15px; font-size:10pt; font-weight:bold; } QPushButton:hover{background-color:#005a9e;} QPushButton:pressed{background-color:#003c6a;}"
        )
        layout.addWidget(self.send_button)
        main_layout.addWidget(container_widget)
        self.setLayout(main_layout)
        self.async_thread: Optional[QThread] = None
        self.async_runner: Optional[AsyncRunner] = None
        self._is_closing = False
        self.staged_files: List[Dict[str, Any]] = []
        self._update_window_title_and_placeholder()

    def _initialize_avatar_qurl_and_flag(
        self, image_path: str, avatar_type_for_log: str
    ) -> tuple[str, bool]:
        """
        初始化头像的QURL字符串和Pillow处理标志。
        Args:
            image_path: 原始图片路径。
            avatar_type_for_log: 头像类型（例如 "pet", "user"），用于日志记录。
        Returns:
            一个元组 (qurl_string, is_processed_by_pillow)。
        """
        qurl_string = ""
        is_processed_by_pillow = False
        path_for_qurl = image_path
        if PILLOW_AVAILABLE and image_path and os.path.exists(image_path):
            processed_path = self._process_avatar_to_circular_original_res(image_path)
            if processed_path:
                path_for_qurl = processed_path
                is_processed_by_pillow = True
            else:
                logger.warning(
                    f"Pillow (original res) processing failed for {avatar_type_for_log} avatar, using original: {image_path}"
                )
        if path_for_qurl and os.path.exists(path_for_qurl):
            qurl_string = QUrl.fromLocalFile(os.path.abspath(path_for_qurl)).toString()
        return qurl_string, is_processed_by_pillow

    def _set_input_active(self, active: bool):
        """设置输入相关控件的激活状态。"""
        self.send_button.setEnabled(active)
        self.input_field.setEnabled(active)
        self.attach_button.setEnabled(active)
        if active:
            self.input_field.setFocus()

    def set_agent_mode_active(self, active: bool):
        self.is_agent_mode_active_chat = active
        self._update_window_title_and_placeholder()

    def _update_window_title_and_placeholder(self):
        if self.is_agent_mode_active_chat:
            self.setWindowTitle(f"与{self.pet_name}交互 (Agent模式)")
            self.input_field.setPlaceholderText(
                f"向 {self.pet_name} (Agent)下达指令... (Enter发送)"
            )
        else:
            self.setWindowTitle(f"与{self.pet_name}聊天")
            self.input_field.setPlaceholderText(
                f"对{self.pet_name}说些什么... (Enter发送)"
            )

    def _process_avatar_to_circular_original_res(
        self, image_path: str
    ) -> Optional[str]:
        if not PILLOW_AVAILABLE or not image_path or not os.path.isfile(image_path):
            return None
        try:
            original_filename = os.path.basename(image_path)
            name_part, _ = os.path.splitext(original_filename)
            file_mtime = os.path.getmtime(image_path)
            safe_name_part = "".join(c if c.isalnum() else "_" for c in name_part)
            if not safe_name_part:
                safe_name_part = "avatar"
            cached_filename = (
                f"{safe_name_part}_orig_res_circular_{int(file_mtime)}.png"
            )
            cached_image_path = os.path.join(self.avatar_cache_dir, cached_filename)
            if os.path.exists(cached_image_path):
                return cached_image_path
            img = Image.open(image_path)
            original_width, original_height = img.size
            crop_size = min(original_width, original_height)
            img_cropped = ImageOps.fit(
                img, (crop_size, crop_size), Image.Resampling.LANCZOS
            )
            img_rgba = img_cropped.convert("RGBA")
            mask = Image.new("L", (crop_size, crop_size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, crop_size, crop_size), fill=255)
            img_rgba.putalpha(mask)
            img_rgba.save(cached_image_path, "PNG")
            return cached_image_path
        except Exception as e:
            logger.error(
                f"Error processing original-res avatar '{image_path}' with Pillow: {e}",
                exc_info=True,
            )
            return None

    def _get_raw_chat_history_for_display(self) -> list:
        history_list = []
        if self.mongo_handler and self.mongo_handler.is_connected():
            chat_dialog_internal_default_count = (
                self.config_manager.get_history_count_for_prompt() * 2
            )
            configured_display_count = (
                self.config_manager.get_chat_dialog_display_history_count(default=0)
            )
            if configured_display_count > 0:
                display_count = configured_display_count
            else:
                display_count = chat_dialog_internal_default_count
            raw_history = self.mongo_handler.get_recent_chat_history(
                count=display_count,
                role_play_character=self.current_role_play_character,
            )
            for msg in raw_history:
                history_list.append(
                    {
                        "sender": msg.get("sender"),
                        "message_text": msg.get("message_text"),
                    }
                )
        return history_list

    def _cleanup_async_resources(self):
        if self.async_runner:
            try:
                self.async_runner.task_completed.disconnect(self._handle_async_response)
            except (TypeError, RuntimeError):
                pass
            try:
                self.async_runner.task_failed.disconnect(self._handle_async_failure)
            except (TypeError, RuntimeError):
                pass
            self.async_runner.deleteLater()
            self.async_runner = None

    def _handle_thread_actually_finished(self, finished_thread: QThread):
        if self.async_thread is finished_thread:
            self.async_thread = None
        if not self._is_closing:
            self._set_input_active(True)

    def _handle_attach_file_dialog(self):
        file_filter = "媒体文件 (*.png *.jpg *.jpeg *.gif *.webp *.mp3 *.wav *.m4a *.ogg);;所有文件 (*)"
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要附加的文件", "", file_filter
        )
        if file_paths:
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"选择的文件路径不存在: {file_path}")
                    continue
                file_info = {
                    "path": file_path,
                    "display_name": os.path.basename(file_path),
                    "mime_type": mimetypes.guess_type(file_path)[0]
                    or "application/octet-stream",
                    "widget": None,
                }
                if any(f["path"] == file_path for f in self.staged_files):
                    logger.info(f"文件 {file_path} 已在暂存区。")
                    continue
                self.staged_files.append(file_info)
            self._update_staged_files_ui()

    def _update_staged_files_ui(self):
        while self.staged_files_layout.count():
            child = self.staged_files_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        if not self.staged_files:
            self.staged_files_scroll_area.setVisible(False)
            return
        for file_info in self.staged_files:
            file_widget = QWidget()
            file_layout = QHBoxLayout(file_widget)
            file_layout.setContentsMargins(0, 0, 0, 0)
            file_layout.setSpacing(5)
            thumb_label = QLabel()
            thumb_label.setFixedSize(
                STAGED_FILE_THUMBNAIL_SIZE, STAGED_FILE_THUMBNAIL_SIZE
            )
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet(
                "border: 1px solid #555; border-radius: 4px; background-color: #333;"
            )
            pixmap = QPixmap(file_info["path"])
            if not pixmap.isNull() and file_info["mime_type"].startswith("image/"):
                thumb_label.setPixmap(
                    pixmap.scaled(
                        STAGED_FILE_THUMBNAIL_SIZE,
                        STAGED_FILE_THUMBNAIL_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                thumb_label.setText(
                    file_info["mime_type"].split("/")[0][:3].upper()
                    if "/" in file_info["mime_type"]
                    else "FILE"
                )
                thumb_label.setStyleSheet(
                    thumb_label.styleSheet() + "color: #ccc; font-weight: bold;"
                )
            file_layout.addWidget(thumb_label)
            remove_button = QPushButton("X")
            remove_button.setFixedSize(20, 20)
            remove_button.setToolTip(f"移除 {file_info['display_name']}")
            remove_button.setStyleSheet(
                "QPushButton { background-color: #550000; color: white; border-radius: 10px; font-weight: bold;} QPushButton:hover { background-color: #880000; }"
            )
            remove_button.clicked.connect(
                lambda checked=False, fi=file_info: self._remove_staged_file(fi)
            )
            thumb_and_remove_layout = QVBoxLayout()
            thumb_and_remove_layout.setContentsMargins(0, 0, 0, 0)
            thumb_and_remove_layout.setSpacing(2)
            top_bar_layout = QHBoxLayout()
            top_bar_layout.addStretch()
            top_bar_layout.addWidget(remove_button)
            top_bar_layout.setContentsMargins(0, 0, 0, 0)
            simple_item_widget = QFrame()
            simple_item_layout = QVBoxLayout(simple_item_widget)
            simple_item_layout.setContentsMargins(0, 0, 0, 0)
            simple_item_layout.setSpacing(1)
            button_container = QWidget()
            button_layout = QHBoxLayout(button_container)
            button_layout.setContentsMargins(0, 0, 0, 0)
            button_layout.addStretch()
            button_layout.addWidget(remove_button)
            simple_item_layout.addWidget(button_container)
            simple_item_layout.addWidget(thumb_label)
            name_label = QLabel(file_info["display_name"])
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setFixedWidth(STAGED_FILE_THUMBNAIL_SIZE)
            name_label.setWordWrap(True)
            name_label.setStyleSheet("font-size: 7pt; color: #bbb;")
            fm = name_label.fontMetrics()
            max_name_height = fm.height() * 2 + fm.leading()
            name_label.setMaximumHeight(max_name_height)
            simple_item_layout.addWidget(name_label)
            simple_item_widget.setFixedWidth(STAGED_FILE_THUMBNAIL_SIZE + 10)
            file_info["widget"] = simple_item_widget
            self.staged_files_layout.addWidget(simple_item_widget)
        self.staged_files_scroll_area.setVisible(True)
        self.staged_files_widget.adjustSize()

    def _remove_staged_file(self, file_info_to_remove: Dict[str, Any]):
        self.staged_files = [
            fi for fi in self.staged_files if fi["path"] != file_info_to_remove["path"]
        ]
        self._update_staged_files_ui()

    def send_message_handler(self):
        user_message = self.input_field.text().strip()
        if not user_message and not self.staged_files:
            return
        if user_message or self.staged_files:
            self._add_message_to_display(
                self.user_name,
                user_message if user_message else "(发送文件)",
                is_user=True,
            )
        if not self.is_agent_mode_active_chat and user_message:
            if self.mongo_handler and self.mongo_handler.is_connected():
                actual_user_name = self.config_manager.get_user_name()
                message_to_save = user_message
                if self.staged_files:
                    filenames = [f["display_name"] for f in self.staged_files]
                    message_to_save += f" [附加文件: {', '.join(filenames)}]"
                self.mongo_handler.insert_chat_message(
                    sender=actual_user_name,
                    message_text=message_to_save,
                    role_play_character=self.current_role_play_character,
                )
        elif self.is_agent_mode_active_chat:
            logger.info(
                "Agent Mode: User message (and any files) not saved to chat history."
            )
        self.input_field.clear()
        QApplication.processEvents()
        if self.async_thread and self.async_thread.isRunning():
            logger.warning("Previous async task still running. Please wait.")
            return
        files_to_send_now = list(self.staged_files)
        current_task_thread = QThread(self)
        self.async_runner = AsyncRunner(
            self._send_message_async_logic(
                user_message, self.is_agent_mode_active_chat, files_to_send_now
            )
        )
        self.async_runner.moveToThread(current_task_thread)
        self.async_runner.task_completed.connect(self._handle_async_response)
        self.async_runner.task_failed.connect(self._handle_async_failure)
        current_task_thread.started.connect(self.async_runner.run_async_task)
        current_task_thread.finished.connect(current_task_thread.deleteLater)
        current_task_thread.finished.connect(
            lambda bound_thread=current_task_thread: self._handle_thread_actually_finished(
                bound_thread
            )
        )
        self.async_thread = current_task_thread
        self._set_input_active(False)
        self.async_thread.start()
        self.staged_files.clear()
        self._update_staged_files_ui()

    async def _send_message_async_logic(
        self,
        user_message: str,
        is_agent_mode: bool,
        staged_media_files: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            if is_agent_mode:
                if self.agent_core:
                    logger.info(
                        f"Agent Mode: Processing user request: {user_message[:50]}... with {len(staged_media_files)} files."
                    )
                    response_data = await asyncio.wait_for(
                        self.agent_core.process_user_request(
                            user_message, media_files=staged_media_files
                        ),
                        timeout=180.0,
                    )
                    response_data.setdefault("text", "Agent action processed.")
                    response_data.setdefault("emotion", "neutral")
                    return response_data
                else:
                    logger.error("Agent mode active but AgentCore is not available.")
                    return {
                        "text": "错误: Agent核心未加载，无法执行指令。",
                        "emotion": "sad",
                        "is_error": True,
                    }
            else:
                if staged_media_files:
                    response_data = await asyncio.wait_for(
                        self.gemini_client.send_multimodal_message_async(
                            text_prompt=user_message,
                            media_files=staged_media_files,
                        ),
                        timeout=120.0,
                    )
                else:
                    response_data = await asyncio.wait_for(
                        self.gemini_client.send_message(
                            message_text=user_message,
                            hippocampus_manager=self.hippocampus_manager,
                            is_agent_mode=False,
                        ),
                        timeout=90.0,
                    )
                return response_data
        except asyncio.TimeoutError:
            mode_str = "Agent指令" if is_agent_mode else "Gemini API调用"
            files_info = (
                f" with {len(staged_media_files)} files" if staged_media_files else ""
            )
            logger.warning(
                f"{mode_str}{files_info} timed out for message: {user_message[:50]}..."
            )
            return {
                "text": f"呜，{self.pet_name}思考的时间好像太久了…… ({mode_str}{files_info}超时)",
                "emotion": "default",
                "is_error": True,
            }
        except Exception as e:
            logger.error(
                f"Error in _send_message_async_logic (agent_mode={is_agent_mode}, files={len(staged_media_files)}): {e}",
                exc_info=True,
            )
            return {
                "text": f"好像有个报错: {str(e)}",
                "emotion": "default",
                "is_error": True,
            }

    def _handle_async_response(self, response_data: Dict[str, Any]):
        current_runner = self.async_runner
        current_thread = current_runner.thread() if current_runner else None
        is_error_response = response_data.get("is_error", False)
        if self.is_agent_mode_active_chat:
            pet_text = response_data.get("text", "Agent action completed.")
            pet_emotion = response_data.get("emotion", "neutral")
            action_summary = response_data.get("action_summary", "")
            action_performed = response_data.get("action_performed", False)
            tool_result = response_data.get("tool_result")
            if action_summary:
                pet_text += f"\n[Agent思考: {action_summary}]"
            if tool_result and isinstance(tool_result, dict):
                if tool_result.get("success") is False and tool_result.get("error"):
                    pet_text += f"\n[工具执行错误: {tool_result.get('error')}]"
                elif tool_result.get("message"):
                    pet_text += f"\n[工具消息: {tool_result.get('message')}]"
                if tool_result.get("content"):
                    pet_text += (
                        f"\n[文件内容预览 (部分)]:\n{tool_result.get('content')}"
                    )
            self.speech_and_emotion_received.emit(pet_text, pet_emotion)
        else:
            pet_text = response_data.get("text", "我好像不知道该说什么了...")
            pet_emotion = response_data.get("emotion", "default")
            text_japanese = response_data.get("text_japanese")
            llm_tone = response_data.get(
                "tone", self.config_manager.get_tts_default_tone()
            )
            if text_japanese and text_japanese.strip():
                self.chat_text_for_tts_ready.emit(text_japanese, llm_tone)
            if not is_error_response:
                if self.mongo_handler and self.mongo_handler.is_connected():
                    actual_pet_name = self.config_manager.get_pet_name()
                    self.mongo_handler.insert_chat_message(
                        sender=actual_pet_name,
                        message_text=pet_text,
                        role_play_character=self.current_role_play_character,
                    )
            self.speech_and_emotion_received.emit(pet_text, pet_emotion)
        if self._is_closing:
            logger.info(
                "Dialog is closing. Skipping internal UI updates for this response."
            )
        else:
            self._set_input_active(True)
            self._add_message_to_display(self.pet_name, pet_text, is_user=False)
        if current_thread and current_thread.isRunning():
            current_thread.quit()
        self._cleanup_async_resources()

    def _handle_async_failure(self, error_message: str):
        current_runner = self.async_runner
        current_thread = current_runner.thread() if current_runner else None
        error_emotion = "neutral" if self.is_agent_mode_active_chat else "sad"
        self.speech_and_emotion_received.emit(
            f"呜，我好像出错了... ({error_message[:30]})", error_emotion
        )
        if self._is_closing:
            logger.info(
                f"Dialog is closing. Skipping internal UI updates for this failure: {error_message}"
            )
        else:
            self._set_input_active(True)
            logger.error(f"Async task failed: {error_message}")
            self._add_message_to_display(
                self.pet_name, f"发生错误: {error_message}", is_user=False
            )
        if current_thread and current_thread.isRunning():
            current_thread.quit()
        self._cleanup_async_resources()

    def _format_message_html(
        self, sender_name: str, message: str, avatar_qurl: str, is_user: bool
    ) -> str:
        escaped_message = html.escape(message).replace("\n", "<br>")
        current_display_size = DISPLAY_AVATAR_SIZE
        was_processed_by_pillow = (
            is_user and self.is_user_avatar_processed_by_pillow
        ) or (not is_user and self.is_pet_avatar_processed_by_pillow)
        img_style = "display: block; object-fit: cover;"
        if not was_processed_by_pillow:
            img_style += " border-radius: 50%;"
        if avatar_qurl:
            avatar_img_html = (
                f'<img src="{avatar_qurl}" width="{current_display_size}" height="{current_display_size}" '
                f'style="{img_style}">'
            )
        else:
            avatar_img_html = f'<div style="width:{current_display_size}px; height:{current_display_size}px; background-color:#555; border-radius:50%;"></div>'
        text_block_common_style = "padding:8px 12px; display:inline-block; max-width:80%; word-wrap:break-word; text-align:left;"
        if is_user:
            text_html = f'<div style="color:white; {text_block_common_style}">{escaped_message}</div>'
            formatted_message = f"""
            <div style="text-align: right; margin-bottom:10px;">
                <table cellpadding="10" cellspacing="5" border="-3" style="display: inline-table; border-collapse:collapse; vertical-align:top;">
                  <tr>
                    <td style="vertical-align:top; padding-right:10px;">{text_html}</td>
                    <td style="width:{current_display_size}px; vertical-align:top;">{avatar_img_html}</td>
                  </tr>
                </table>
            </div>"""
        else:
            text_html = f'<div style="color:#e0e0e0; {text_block_common_style}">{escaped_message}</div>'
            formatted_message = f"""
            <div style="text-align: left; margin-bottom:10px;">
                <table cellpadding="10" cellspacing="5" border="-3" style="display: inline-table; border-collapse:collapse; vertical-align:top;">
                  <tr>
                    <td style="width:{current_display_size}px; vertical-align:top; padding-right:10px;">{avatar_img_html}</td>
                    <td style="vertical-align:top;">{text_html}</td>
                  </tr>
                </table>
            </div>"""
        return formatted_message

    def _add_message_to_display(
        self, sender_name_for_log_only: str, message: str, is_user: bool
    ):
        if self._is_closing:
            return
        avatar_qurl = self.user_avatar_qurl if is_user else self.pet_avatar_qurl
        actual_sender_name = self.user_name if is_user else self.pet_name
        html_content = self._format_message_html(
            actual_sender_name, message, avatar_qurl, is_user
        )
        self.chat_display.append(html_content)
        self.chat_display.ensureCursorVisible()

    def open_dialog(self):
        self._is_closing = False
        self.chat_display.clear()
        self.staged_files.clear()
        self._update_staged_files_ui()
        if self.agent_core and hasattr(self.agent_core, "is_agent_mode_active"):
            self.is_agent_mode_active_chat = self.agent_core.is_agent_mode_active
        self._update_window_title_and_placeholder()
        if self.is_agent_mode_active_chat:
            agent_welcome_html = f"<div style='padding:20px 0; color:#aaa; text-align:center;'><i>Agent模式已激活。输入指令进行交互。\nAgent交互不会被保存到聊天记录。</i></div>"
            self.chat_display.setHtml(agent_welcome_html)
        else:
            history_for_display = self._get_raw_chat_history_for_display()
            if history_for_display:
                for msg_data in history_for_display:
                    sender_from_db = msg_data.get("sender", "")
                    is_user_msg = sender_from_db == self.user_name
                    self._add_message_to_display(
                        sender_name_for_log_only=sender_from_db,
                        message=msg_data.get("message_text", ""),
                        is_user=is_user_msg,
                    )
            else:
                no_history_html = f"<div style='padding:20px 0; color:#aaa; text-align:center;'><i>还没有和 {self.pet_name} 的聊天记录。</i></div>"
                self.chat_display.setHtml(no_history_html)
        self.input_field.clear()
        self._set_input_active(True)
        self.show()
        self.activateWindow()
        self.raise_()

    def _close_dialog_actions(self):
        if self._is_closing:
            return
        self._is_closing = True
        self.dialog_closed.emit()
        active_thread = self.async_thread
        if active_thread and active_thread.isRunning():
            logger.info("Dialog closing, requesting async thread to quit.")
            active_thread.quit()
        else:
            self._cleanup_async_resources()

    def closeEvent(self, event):
        self._close_dialog_actions()
        super().closeEvent(event)

    def reject(self):
        self._close_dialog_actions()
        super().reject()
