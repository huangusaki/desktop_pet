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
from PyQt6.QtCore import Qt, pyqtSignal, QUrl
from PyQt6.QtGui import QTextCursor


class ChatDialog(QDialog):
    dialog_closed = pyqtSignal()
    emotion_received = pyqtSignal(str)

    def __init__(
        self,
        gemini_client,
        mongo_handler,
        config_manager,
        pet_avatar_path: str,
        user_avatar_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini_client = gemini_client
        self.mongo_handler = mongo_handler
        self.config_manager = config_manager
        self.user_name = self.config_manager.get_user_name()
        self.pet_name = self.config_manager.get_pet_name()
        self.current_role_play_character = self.pet_name
        self.pet_avatar_path = pet_avatar_path
        self.user_avatar_path = user_avatar_path
        self.pet_avatar_qurl = (
            QUrl.fromLocalFile(self.pet_avatar_path).toString()
            if os.path.exists(self.pet_avatar_path)
            else ""
        )
        self.user_avatar_qurl = (
            QUrl.fromLocalFile(self.user_avatar_path).toString()
            if os.path.exists(self.user_avatar_path)
            else ""
        )
        self.setWindowTitle(f"与 {self.pet_name} 聊天")
        self.setMinimumSize(450, 550)
        self.setMaximumSize(700, 800)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("QDialog { background-color: transparent; }")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container_widget = QWidget(self)
        container_widget.setObjectName("ChatContainer")
        container_widget.setStyleSheet(
            """
            QWidget#ChatContainer {
                background-color: rgba(0, 0, 0, 0.67);
                border-radius: 12px;
            }
            """
        )
        layout = QVBoxLayout(container_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        self.chat_display = QTextBrowser(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setOpenExternalLinks(False)
        self.chat_display.document().setDocumentMargin(0)
        self.chat_display.document().setDefaultStyleSheet(
            "body { margin: 0px; padding: 0px; }"
        )
        self.chat_display.setStyleSheet(
            """
            QTextBrowser {
                background-color: transparent;
                color: #e0e0e0;
                border: none;
                padding: 8px;
                font-size: 10pt;
            }
            """
        )
        layout.addWidget(self.chat_display)
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText(
            f"对 {self.pet_name} 说些什么... (Enter发送)"
        )
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet(
            """
            QLineEdit {
                background-color: rgba(50, 50, 50, 0.9);
                color: #ffffff;
                border: 1px solid rgba(80, 80, 80, 0.8);
                border-radius: 6px;
                padding: 8px;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border: 1px solid rgba(0, 120, 215, 0.9);
            }
        """
        )
        input_layout.addWidget(self.input_field)
        self.send_button = QPushButton("发送", self)
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078FF;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 10pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #003c6a;
            }
        """
        )
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        main_layout.addWidget(container_widget)
        self.setLayout(main_layout)

    def _get_raw_chat_history_for_display(self) -> list:
        """Gets history primarily for displaying in the UI."""
        history_list = []
        if self.mongo_handler and self.mongo_handler.is_connected():
            count = self.config_manager.get_history_count_for_prompt()
            raw_history = self.mongo_handler.get_recent_chat_history(
                count=count, role_play_character=self.current_role_play_character
            )
            if raw_history:
                for msg in raw_history:
                    history_list.append(
                        {
                            "sender": msg.get("sender"),
                            "message_text": msg.get("message_text"),
                        }
                    )
        return history_list

    def _get_cleaned_gemini_sdk_history_from_db(self) -> list:
        """
        Fetches chat history from MongoDB and formats it for the Gemini SDK.
        Ensures roles alternate correctly (user, model, user, model...).
        Consecutive messages from the same role are merged.
        """
        sdk_formatted_history = []
        if not (self.mongo_handler and self.mongo_handler.is_connected()):
            return sdk_formatted_history
        count = self.config_manager.get_history_count_for_prompt()
        raw_db_history = self.mongo_handler.get_recent_chat_history(
            count=count, role_play_character=self.current_role_play_character
        )
        if not raw_db_history:
            return sdk_formatted_history
        temp_history = []
        for msg_entry in raw_db_history:
            sender_val = msg_entry.get("sender")
            role = (
                "user"
                if isinstance(sender_val, str) and sender_val.lower() == "user"
                else "model"
            )
            text_content = msg_entry.get("message_text", "")
            if text_content:
                temp_history.append({"role": role, "text": text_content})
        if not temp_history:
            return sdk_formatted_history
        current_merged_text = ""
        current_role = None
        for msg in temp_history:
            role = msg["role"]
            text = msg["text"]
            if current_role is None:
                current_role = role
                current_merged_text = text
            elif role == current_role:
                current_merged_text += "\n" + text
            else:
                if current_merged_text:
                    sdk_formatted_history.append(
                        {"role": current_role, "parts": [{"text": current_merged_text}]}
                    )
                current_role = role
                current_merged_text = text
        if current_role and current_merged_text:
            sdk_formatted_history.append(
                {"role": current_role, "parts": [{"text": current_merged_text}]}
            )
        return sdk_formatted_history

    def send_message(self):
        user_message = self.input_field.text().strip()
        if not user_message:
            return
        self._add_message_to_display(self.user_name, user_message, is_user=True)
        if self.mongo_handler and self.mongo_handler.is_connected():
            self.mongo_handler.insert_chat_message(
                sender="user",
                message_text=user_message,
                role_play_character=self.current_role_play_character,
            )
        self.input_field.clear()
        QApplication.processEvents()
        response_data = self.gemini_client.send_message(user_message)
        pet_text = response_data.get("text", "我好像不知道该说什么了...")
        pet_emotion = response_data.get("emotion", "default")
        self._add_message_to_display(self.pet_name, pet_text, is_user=False)
        if self.mongo_handler and self.mongo_handler.is_connected():
            self.mongo_handler.insert_chat_message(
                sender="pet",
                message_text=pet_text,
                role_play_character=self.current_role_play_character,
            )
        self.emotion_received.emit(pet_emotion)

    def _format_message_html(
        self,
        sender_name_for_log_only: str,
        message: str,
        avatar_qurl: str,
        is_user: bool,
    ) -> str:
        escaped_message = html.escape(message).replace("\n", "<br>")
        avatar_size = 36
        avatar_img_html = ""
        if avatar_qurl:
            avatar_img_html = f'<img src="{avatar_qurl}" width="{avatar_size}" height="{avatar_size}" style="border-radius: 50%; display: block; object-fit: cover;">'
        else:
            avatar_img_html = f'<div style="width:{avatar_size}px; height:{avatar_size}px; background-color: #555; border-radius:50%;"></div>'
        bubble_border_radius = "15px"
        bubble_div_common_style = f"padding: 8px 12px; border-radius: {bubble_border_radius}; display: inline-block; max-width: 80%; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1);"
        table_attributes = 'cellpadding="0" cellspacing="0" border="0"'
        outer_table_style = (
            "width: 100%; border-collapse: collapse; margin-bottom: 10px;"
        )
        formatted_message = ""
        if is_user:
            bubble_background_color = "#0078FF"
            text_color = "white"
            bubble_html_content = f'<div style="background-color: {bubble_background_color}; color: {text_color}; {bubble_div_common_style} text-align: left;">{escaped_message}</div>'
            formatted_message = f"""
            <table {table_attributes} style="{outer_table_style}">
              <tr>
                <td style="text-align: right; padding: 0; margin: 0; width: 100%;"> 
                  <table cellpadding="0" cellspacing="0" border="0" style="display: inline-table; vertical-align: top; text-align: left;"> 
                    <tr>
                      <td style="width: {avatar_size}px; vertical-align: top; padding-right: 10px;"> 
                        {avatar_img_html}
                      </td>
                      <td style="vertical-align: top;"> 
                        {bubble_html_content}
                      </td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>
            """
        else:
            bubble_background_color = "#E9E9EB"
            text_color = "black"
            bubble_html_content = f'<div style="background-color: {bubble_background_color}; color: {text_color}; {bubble_div_common_style} text-align: left;">{escaped_message}</div>'
            formatted_message = f"""
            <table {table_attributes} style="{outer_table_style}">
              <tr>
                <td style="width: {avatar_size}px; vertical-align: top; padding-right: 10px;">
                  {avatar_img_html}
                </td>
                <td style="width: auto; vertical-align: top; text-align: left;">
                  {bubble_html_content}
                </td>
              </tr>
            </table>
            """
        return formatted_message

    def _add_message_to_display(
        self, sender_name_for_log_only: str, message: str, is_user: bool
    ):
        avatar_qurl_to_use = self.user_avatar_qurl if is_user else self.pet_avatar_qurl
        html_content = self._format_message_html(
            sender_name_for_log_only, message, avatar_qurl_to_use, is_user
        )
        self.chat_display.append(html_content)
        self.chat_display.ensureCursorVisible()

    def open_dialog(self):
        self.chat_display.clear()
        if hasattr(self.gemini_client, "is_new_chat_session"):
            self.gemini_client.is_new_chat_session = True
        history_for_display = self._get_raw_chat_history_for_display()
        initial_content_set = False
        if history_for_display:
            for msg_data in history_for_display:
                sender = msg_data.get("sender")
                text = msg_data.get("message_text")
                if sender and text:
                    is_user_msg = isinstance(sender, str) and sender.lower() == "user"
                    self._add_message_to_display(
                        self.user_name if is_user_msg else self.pet_name,
                        text,
                        is_user_msg,
                    )
            initial_content_set = True
        if not initial_content_set:
            no_history_html = f"<div style='padding: 20px 0; color:#aaa; text-align:center;'><i>还没有和 {self.pet_name} 的聊天记录。</i></div>"
            self.chat_display.setHtml(no_history_html)
        cleaned_history_for_gemini_init = self._get_cleaned_gemini_sdk_history_from_db()
        try:
            self.gemini_client.start_chat_session(
                history=cleaned_history_for_gemini_init
            )
            print(
                f"ChatDialog: Gemini chat session started/restarted with {len(cleaned_history_for_gemini_init)} history turns."
            )
        except Exception as e:
            error_message = f"启动/重启Gemini聊天会话失败: {html.escape(str(e))}"
            print(f"ERROR: Exception during gemini_client.start_chat_session: {e}")
            error_html = f"<div style='padding: 10px 0; color:#EF5350; text-align:center;'><i>{error_message}</i></div>"
            current_html_content = self.chat_display.toHtml()
            if (
                f"还没有和 {self.pet_name} 的聊天记录。" in current_html_content
                and "<table" not in current_html_content
            ):
                self.chat_display.setHtml(error_html)
            else:
                self.chat_display.setHtml(error_html + current_html_content)
        self.input_field.clear()
        self.show()
        self.activateWindow()
        self.raise_()
        self.input_field.setFocus()

    def closeEvent(self, event):
        self.dialog_closed.emit()
        super().closeEvent(event)

    def reject(self):
        self.dialog_closed.emit()
        super().reject()
