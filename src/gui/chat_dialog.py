import html
import os
import asyncio
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextBrowser,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QScrollArea,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QIcon, QPixmap
from typing import Optional, List, Any, Dict
from ..data.relationship_manager import RelationshipManager
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
    HippocampusManager = None
try:
    from ..core.agent_core import AgentCore
except ImportError:
    AgentCore = None
    logger.warning(
        "ChatDialog: AgentCore could not be imported. Agent mode will be unavailable in chat."
    )
DISPLAY_AVATAR_SIZE = 28
STAGED_FILE_THUMBNAIL_SIZE = 64


class ChatDialog(QDialog):
    dialog_closed = pyqtSignal()
    speech_and_emotion_received = pyqtSignal(str, str)
    chat_text_for_tts_ready = pyqtSignal(str, str)
    message_display_data_ready = pyqtSignal(str, str, bool)

    def __init__(
        self,
        gemini_client: Any,
        mongo_handler: Optional[Any],
        config_manager: Any,
        pet_name: str,
        user_name: str,
        pet_avatar_path: str,
        user_avatar_path: str,
        asyncio_helper: Any,
        hippocampus_manager: Optional[Any] = None,
        agent_core: Optional[Any] = None,
        relationship_manager: Optional["RelationshipManager"] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.gemini_client = gemini_client
        self.mongo_handler = mongo_handler
        self.config_manager = config_manager
        self.hippocampus_manager = hippocampus_manager
        self.agent_core = agent_core
        self.asyncio_helper = asyncio_helper
        self.relationship_manager = relationship_manager
        self.pet_name = pet_name
        self.user_name = user_name
        self.is_agent_mode_active_chat = False
        if self.agent_core and hasattr(self.agent_core, "is_agent_mode_active"):
            self.is_agent_mode_active_chat = self.agent_core.is_agent_mode_active
        self.current_role_play_character = self.pet_name
        self.original_pet_avatar_path = pet_avatar_path
        self.original_user_avatar_path = user_avatar_path
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.avatar_cache_dir = os.path.join(self.current_script_dir, ".avatar_cache")
        if PILLOW_AVAILABLE:
            os.makedirs(self.avatar_cache_dir, exist_ok=True)
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
        self._is_closing = False
        self._is_waiting_for_response = False
        self.staged_files: List[Dict[str, Any]] = []
        self.message_display_data_ready.connect(self._add_message_to_display)
        self._update_window_title_and_placeholder()

    def send_message_handler(self):
        """
        重构后的发送消息处理器。
        不再创建 QThread，而是使用 self.asyncio_helper 调度任务。
        """
        if self._is_waiting_for_response:
            logger.warning("Previous async task still running. Please wait.")
            return
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
        self.input_field.clear()
        files_to_send_now = list(self.staged_files)
        self.staged_files.clear()
        self._update_staged_files_ui()
        self._set_input_active(False)
        self._is_waiting_for_response = True
        coro = self._send_message_async_logic(
            user_message, self.is_agent_mode_active_chat, files_to_send_now
        )
        future = self.asyncio_helper.schedule_task(coro)
        if future:
            future.add_done_callback(self._handle_future_result)
        else:
            self._handle_async_failure("Failed to schedule async task.")

    def _handle_future_result(self, future: asyncio.Future):
        """
        当 asyncio.Future 完成时的回调。
        它在 asyncio 线程中被调用，所以UI更新需要小心。
        """
        try:
            result = future.result()
            self._handle_async_response(result)
        except Exception as e:
            self._handle_async_failure(str(e))
        finally:
            self._is_waiting_for_response = False
            QTimer.singleShot(0, lambda: self._set_input_active(True))

    def _set_input_active(self, active: bool):
        self.send_button.setEnabled(active)
        self.input_field.setEnabled(active)
        self.attach_button.setEnabled(active)
        if active:
            self.input_field.setFocus()

    def _initialize_avatar_qurl_and_flag(
        self, image_path: str, avatar_type_for_log: str
    ) -> tuple[str, bool]:
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
        is_error_response = response_data.get("is_error", False)
        if self.is_agent_mode_active_chat:
            pet_text = response_data.get("text", "Agent action completed.")
            pet_emotion = response_data.get("emotion", "neutral")
            action_summary = response_data.get("action_summary", "")
            if action_summary:
                pet_text += f"\n[Agent思考: {action_summary}]"
            self.speech_and_emotion_received.emit(pet_text, pet_emotion)
        else:
            pet_text = response_data.get("text", "我好像不知道该说什么了...")
            pet_emotion = response_data.get("emotion", "default")
            text_japanese = response_data.get("text_japanese")
            llm_tone = response_data.get(
                "tone", self.config_manager.get_tts_default_tone()
            )
            favorability_change = response_data.get("favorability_change", 0)
            if self.relationship_manager and favorability_change != 0:
                logger.info(
                    f"LLM decided favorability change: {favorability_change}. Applying update."
                )
                self.asyncio_helper.schedule_task(
                    self.relationship_manager.update_favorability(
                        base_change=favorability_change
                    )
                )
            elif self.relationship_manager:
                logger.info("LLM decided no change in favorability.")
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
            return
        self.message_display_data_ready.emit(self.pet_name, pet_text, False)

    def _handle_async_failure(self, error_message: str):
        error_emotion = "neutral" if self.is_agent_mode_active_chat else "sad"
        self.speech_and_emotion_received.emit(
            f"呜，我好像出错了... ({error_message[:30]})", error_emotion
        )
        if self._is_closing:
            logger.info(
                f"Dialog is closing. Skipping internal UI updates for this failure: {error_message}"
            )
            return
        logger.error(f"Async task failed: {error_message}")
        self.message_display_data_ready.emit(
            self.pet_name, f"发生错误: {error_message}", False
        )

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
            avatar_img_html = f'<img src="{avatar_qurl}" width="{current_display_size}" height="{current_display_size}" style="{img_style}">'
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

    def _add_message_to_display(self, sender_name: str, message: str, is_user: bool):
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
            agent_welcome_html = "<div style='padding:20px 0; color:#aaa; text-align:center;'><i>Agent模式已激活。输入指令进行交互。\nAgent交互不会被保存到聊天记录。</i></div>"
            self.chat_display.setHtml(agent_welcome_html)
        else:
            history_for_display = self._get_raw_chat_history_for_display()
            if history_for_display:
                for msg_data in history_for_display:
                    sender_from_db = msg_data.get("sender", "")
                    is_user_msg = sender_from_db == self.user_name
                    self._add_message_to_display(
                        sender_name=sender_from_db,
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

    def closeEvent(self, event):
        self._close_dialog_actions()
        super().closeEvent(event)

    def reject(self):
        self._close_dialog_actions()
        super().reject()
