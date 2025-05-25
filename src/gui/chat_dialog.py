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
)
from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QThread, QObject
from PyQt6.QtGui import QTextCursor
import asyncio
import sys
from typing import Optional, List, Any, Dict

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
            print("ChatDialog: CRITICAL: Could not import HippocampusManager.")
            HippocampusManager = None
DISPLAY_AVATAR_SIZE = 28


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

    def __init__(
        self,
        gemini_client: Any,
        mongo_handler: Any,
        config_manager: Any,
        hippocampus_manager: Optional[HippocampusManager],
        pet_avatar_path: str,
        user_avatar_path: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.gemini_client = gemini_client
        self.mongo_handler = mongo_handler
        self.config_manager = config_manager
        self.hippocampus_manager = hippocampus_manager
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
                    print(
                        f"ChatDialog: Warning: Could not create avatar cache directory: {self.avatar_cache_dir}. Error: {e}"
                    )
        self.is_pet_avatar_processed_by_pillow = False
        self.is_user_avatar_processed_by_pillow = False
        path_for_pet_qurl = self.original_pet_avatar_path
        if (
            PILLOW_AVAILABLE
            and self.original_pet_avatar_path
            and os.path.exists(self.original_pet_avatar_path)
        ):
            processed_path = self._process_avatar_to_circular_original_res(
                self.original_pet_avatar_path
            )
            if processed_path:
                path_for_pet_qurl = processed_path
                self.is_pet_avatar_processed_by_pillow = True
            else:
                print(
                    f"ChatDialog: Pillow (original res) processing failed for pet avatar, using original: {self.original_pet_avatar_path}"
                )
        self.pet_avatar_qurl = ""
        if path_for_pet_qurl and os.path.exists(path_for_pet_qurl):
            self.pet_avatar_qurl = QUrl.fromLocalFile(
                os.path.abspath(path_for_pet_qurl)
            ).toString()
        path_for_user_qurl = self.original_user_avatar_path
        if (
            PILLOW_AVAILABLE
            and self.original_user_avatar_path
            and os.path.exists(self.original_user_avatar_path)
        ):
            processed_path = self._process_avatar_to_circular_original_res(
                self.original_user_avatar_path
            )
            if processed_path:
                path_for_user_qurl = processed_path
                self.is_user_avatar_processed_by_pillow = True
            else:
                print(
                    f"ChatDialog: Pillow (original res) processing failed for user avatar, using original: {self.original_user_avatar_path}"
                )
        self.user_avatar_qurl = ""
        if path_for_user_qurl and os.path.exists(path_for_user_qurl):
            self.user_avatar_qurl = QUrl.fromLocalFile(
                os.path.abspath(path_for_user_qurl)
            ).toString()
        self.setWindowTitle(f"与 {self.pet_name} 聊天")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container_widget = QWidget(self)
        container_widget.setObjectName("ChatContainer")
        layout = QVBoxLayout(container_widget)
        layout.setContentsMargins(15, 15, 15, 15)
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
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText(
            f"对 {self.pet_name} 说些什么... (Enter发送)"
        )
        self.input_field.returnPressed.connect(self.send_message_handler)
        self.input_field.setStyleSheet(
            "QLineEdit { background-color:rgba(50,50,50,0.9); color:#ffffff; border:1px solid rgba(80,80,80,0.8); border-radius:6px; padding:8px; font-size:10pt;} QLineEdit:focus {border:1px solid rgba(0,120,215,0.9);}"
        )
        input_layout.addWidget(self.input_field)
        self.send_button = QPushButton("发送", self)
        self.send_button.clicked.connect(self.send_message_handler)
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setStyleSheet(
            "QPushButton { background-color:#0078FF; color:white; border:none; border-radius:6px; padding:8px 15px; font-size:10pt; font-weight:bold; } QPushButton:hover{background-color:#005a9e;} QPushButton:pressed{background-color:#003c6a;}"
        )
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        main_layout.addWidget(container_widget)
        self.setLayout(main_layout)
        self.async_thread: Optional[QThread] = None
        self.async_runner: Optional[AsyncRunner] = None

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
            print(
                f"ChatDialog: Error processing original-res avatar '{image_path}' with Pillow: {e}"
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
        print(f"ChatDialog: Thread {finished_thread} has actually finished.")
        if self.async_thread is finished_thread:
            self.async_thread = None
            if not self.send_button.isEnabled():
                self.send_button.setEnabled(True)
            if not self.input_field.isEnabled():
                self.input_field.setEnabled(True)
                self.input_field.setFocus()

    def send_message_handler(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            return
        self._add_message_to_display(self.user_name, user_message, is_user=True)
        if self.mongo_handler and self.mongo_handler.is_connected():
            actual_user_name = self.config_manager.get_user_name()
            self.mongo_handler.insert_chat_message(
                sender=actual_user_name,
                message_text=user_message,
                role_play_character=self.current_role_play_character,
            )
        self.input_field.clear()
        QApplication.processEvents()
        if self.async_thread and self.async_thread.isRunning():
            print("ChatDialog: Previous async task still running. Please wait.")
            return
        current_task_thread = QThread()
        self.async_runner = AsyncRunner(self._send_message_async_logic(user_message))
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
        self.send_button.setEnabled(False)
        self.input_field.setEnabled(False)
        self.async_thread.start()

    async def _send_message_async_logic(self, user_message: str) -> Dict[str, Any]:
        retrieved_memories_text = ""
        if self.hippocampus_manager and self.hippocampus_manager._initialized:
            try:
                print(
                    "ChatDialog: Accessing HippocampusManager for memory retrieval..."
                )
                memories = await self.hippocampus_manager.get_memory_from_text(
                    user_message
                )
                if memories:
                    formatted_mems = []
                    for topic, summary in memories:
                        formatted_mems.append(
                            f"相关记忆片段 (主题: {topic}): {summary}"
                        )
                    if formatted_mems:
                        retrieved_memories_text = (
                            "参考以下可能相关的记忆片段来更好地回应：\n"
                            + "\n".join(formatted_mems)
                            + "\n\n"
                        )
                        print(f"ChatDialog: 检索到记忆: {len(memories)} 条。")
                else:
                    print("ChatDialog: 未检索到相关记忆。")
            except Exception as e_mem:
                print(f"ChatDialog: 检索记忆时发生错误: {e_mem}")
        final_user_input_for_llm = user_message
        if retrieved_memories_text:
            final_user_input_for_llm = (
                f"{retrieved_memories_text}"
                f"基于以上记忆和你对我的了解，以及我们之前的对话，请回应我的这句话：{user_message}"
            )
            print(
                f"ChatDialog: Input to LLM (with memory):\n{final_user_input_for_llm[:300]}..."
            )
        print("ChatDialog: Calling GeminiClient.send_message...")
        response_data = self.gemini_client.send_message(final_user_input_for_llm)
        print(
            f"ChatDialog: Gemini response received: {response_data.get('text')}, {response_data.get('emotion')}"
        )
        return response_data

    def _handle_async_response(self, response_data: Dict[str, Any]):
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        pet_text = response_data.get("text", "我好像不知道该说什么了...")
        pet_emotion = response_data.get("emotion", "default")
        self._add_message_to_display(self.pet_name, pet_text, is_user=False)
        if self.mongo_handler and self.mongo_handler.is_connected():
            actual_pet_name = self.config_manager.get_pet_name()
            self.mongo_handler.insert_chat_message(
                sender=actual_pet_name,
                message_text=pet_text,
                role_play_character=self.current_role_play_character,
            )
        self.speech_and_emotion_received.emit(pet_text, pet_emotion)
        task_thread = None
        if self.async_runner and self.async_runner.thread():
            task_thread = self.async_runner.thread()
        elif self.async_thread:
            task_thread = self.async_thread
        if task_thread and task_thread.isRunning():
            print(
                f"ChatDialog: Task completed. Requesting thread {task_thread} to quit."
            )
            task_thread.quit()
        self._cleanup_async_resources()

    def _handle_async_failure(self, error_message: str):
        self.send_button.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        print(f"ChatDialog: Async task failed: {error_message}")
        self._add_message_to_display(
            self.pet_name, f"发生错误: {error_message}", is_user=False
        )
        self.speech_and_emotion_received.emit("呜，我好像出错了...", "sad")
        task_thread = None
        if self.async_runner and self.async_runner.thread():
            task_thread = self.async_runner.thread()
        elif self.async_thread:
            task_thread = self.async_thread
        if task_thread and task_thread.isRunning():
            print(f"ChatDialog: Task failed. Requesting thread {task_thread} to quit.")
            task_thread.quit()
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
                <table cellpadding="5" cellspacing="5" border="5" style="display: inline-table; border-collapse:collapse; vertical-align:top;">
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
                <table cellpadding="5" cellspacing="5" border="5" style="display: inline-table; border-collapse:collapse; vertical-align:top;">
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
        avatar_qurl = self.user_avatar_qurl if is_user else self.pet_avatar_qurl
        actual_sender_name = self.user_name if is_user else self.pet_name
        html_content = self._format_message_html(
            actual_sender_name, message, avatar_qurl, is_user
        )
        self.chat_display.append(html_content)
        self.chat_display.ensureCursorVisible()

    def open_dialog(self):
        self.chat_display.clear()
        history_for_display = self._get_raw_chat_history_for_display()
        if history_for_display:
            for msg_data in history_for_display:
                is_user_msg = msg_data.get("sender", "").lower() == "user"
                self._add_message_to_display(
                    self.user_name if is_user_msg else self.pet_name,
                    msg_data.get("message_text", ""),
                    is_user_msg,
                )
        else:
            no_history_html = f"<div style='padding:20px 0; color:#aaa; text-align:center;'><i>还没有和 {self.pet_name} 的聊天记录。</i></div>"
            self.chat_display.setHtml(no_history_html)
        self.input_field.clear()
        self.show()
        self.activateWindow()
        self.raise_()
        self.input_field.setFocus()

    def closeEvent(self, event):
        self.dialog_closed.emit()
        active_thread = self.async_thread
        if active_thread and active_thread.isRunning():
            print(
                f"ChatDialog: Closing dialog, async task on thread {active_thread} is running. Requesting quit and waiting..."
            )
            active_thread.quit()
            if not active_thread.wait(3000):
                print(
                    f"ChatDialog: Warning: Thread {active_thread} did not finish in time on close."
                )
        self._cleanup_async_resources()
        super().closeEvent(event)

    def reject(self):
        self.dialog_closed.emit()
        active_thread = self.async_thread
        if active_thread and active_thread.isRunning():
            print(
                f"ChatDialog: Dialog rejected, async task on thread {active_thread} is running. Requesting quit and waiting..."
            )
            active_thread.quit()
            if not active_thread.wait(3000):
                print(
                    f"ChatDialog: Warning: Thread {active_thread} did not finish in time on reject."
                )
        self._cleanup_async_resources()
        super().reject()


if __name__ == "__main__":

    class MockMongoHandler:
        def is_connected(self):
            return True

        def get_recent_chat_history(self, count, role_play_character):
            return []

        def insert_chat_message(self, sender, message_text, role_play_character):
            pass

    class MockConfigManager:
        def get_user_name(self):
            return "TestUser"

        def get_pet_name(self):
            return "TestPet"

        def get_history_count_for_prompt(self):
            return 3

        def get_chat_dialog_display_history_count(self, default: int = 0) -> int:
            return default

    class MockGeminiClient:
        def start_chat_session(self):
            pass

        def send_message(self, message: str):
            return {"text": f"Mock response to: {message}", "emotion": "neutral"}

    class MockHippocampusManager:
        _initialized = True

        async def get_memory_from_text(self, text: str):
            return []

    app = QApplication(sys.argv)
    async_event_loop = None
    try:
        async_event_loop = asyncio.get_running_loop()
    except RuntimeError:
        async_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(async_event_loop)
    test_assets_dir = os.path.join(os.path.dirname(__file__), "test_assets_chatdialog")
    os.makedirs(test_assets_dir, exist_ok=True)
    test_pet_avatar = os.path.join(test_assets_dir, "pet_test.png")
    test_user_avatar = os.path.join(test_assets_dir, "user_test.png")
    if not os.path.exists(test_pet_avatar) and PILLOW_AVAILABLE:
        Image.new("RGB", (64, 64), color="blue").save(test_pet_avatar)
    if not os.path.exists(test_user_avatar) and PILLOW_AVAILABLE:
        Image.new("RGB", (64, 64), color="green").save(test_user_avatar)
    dialog = ChatDialog(
        gemini_client=MockGeminiClient(),
        mongo_handler=MockMongoHandler(),
        config_manager=MockConfigManager(),
        hippocampus_manager=MockHippocampusManager(),
        pet_avatar_path=test_pet_avatar if PILLOW_AVAILABLE else "",
        user_avatar_path=test_user_avatar if PILLOW_AVAILABLE else "",
    )
    dialog.open_dialog()
    sys.exit(app.exec())
