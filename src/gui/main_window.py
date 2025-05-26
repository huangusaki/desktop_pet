from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QMenu
from PyQt6.QtGui import QPixmap, QMouseEvent, QGuiApplication, QAction, QActionGroup
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
import os
from typing import List


class PetWindow(QWidget):
    request_open_chat_dialog = pyqtSignal()

    def __init__(
        self,
        initial_image_path: str,
        assets_base_path: str,
        available_emotions: List[str],
    ):
        super().__init__()
        self._drag_pos = QPoint()
        self._is_dragging = False
        self.assets_base_path = assets_base_path
        self.current_emotion = "default"
        self.pixmap_cache = {}
        self.scaled_pixmap_cache = {}
        self._initial_pos_set = False
        self.available_emotions_for_test = (
            available_emotions if available_emotions else ["default"]
        )
        self.pet_size_preference = "medium"
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.speech_bubble_label = QLabel(self)
        self.speech_bubble_label.setWordWrap(True)
        self.speech_bubble_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.speech_bubble_label.setStyleSheet(
            "QLabel {"
            "  background-color: rgba(240, 240, 240, 170);"
            "  color: black;"
            "  padding: 8px;"
            "  border-radius: 10px;"
            "  margin-left: 5px;"
            "  margin-right: 5px;"
            "  margin-top: -3px;"
            "}"
        )
        self.speech_bubble_label.setText("")
        self.speech_bubble_label.setVisible(False)
        layout.addWidget(self.speech_bubble_label)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        self.set_emotion(
            self.current_emotion, initial_image_path_override=initial_image_path
        )

    def _set_error_text(self, text):
        self.image_label.setText(text)
        self.image_label.setStyleSheet(
            "QLabel { color: red; background-color: rgba(200, 200, 200, 150); padding: 10px; }"
        )
        if hasattr(self, "speech_bubble_label"):
            self.speech_bubble_label.setText("")
            self.speech_bubble_label.setVisible(False)
        self.resize(180, 120)

    def _auto_align_to_taskbar_right(self):
        try:
            screen = QGuiApplication.primaryScreen()
            if not screen:
                self.move(800, 600)
                return
            available_geometry = screen.availableGeometry()
            QApplication.instance().processEvents()
            new_x = available_geometry.right() - self.width()
            new_y = available_geometry.bottom() - self.height()
            new_x = max(available_geometry.left(), new_x)
            new_y = max(available_geometry.top(), new_y) + 1
            self.move(new_x, new_y)
        except Exception as e:
            print(f"PetWindow: 自动对齐时出错: {e}. 使用默认定位。")
            self.move(800, 600)

    def update_speech_and_emotion(self, text: str, emotion: str):
        self.set_speech_text(text)
        self.set_emotion(emotion)

    def set_emotion(self, emotion_name: str, initial_image_path_override: str = None):
        if not emotion_name or not isinstance(emotion_name, str):
            emotion_name = "default"
        self.current_emotion = emotion_name.lower()
        raw_pixmap = None
        if initial_image_path_override:
            target_image_path = initial_image_path_override
        else:
            target_image_path = os.path.join(
                self.assets_base_path, f"{self.current_emotion}.png"
            )
        if target_image_path in self.pixmap_cache:
            raw_pixmap = self.pixmap_cache[target_image_path]
        elif os.path.exists(target_image_path):
            pixmap = QPixmap(target_image_path)
            if not pixmap.isNull():
                self.pixmap_cache[target_image_path] = pixmap
                raw_pixmap = pixmap
        if raw_pixmap is None and self.current_emotion != "default":
            default_image_path = os.path.join(self.assets_base_path, "default.png")
            if default_image_path in self.pixmap_cache:
                raw_pixmap = self.pixmap_cache[default_image_path]
            elif os.path.exists(default_image_path):
                pixmap = QPixmap(default_image_path)
                if not pixmap.isNull():
                    self.pixmap_cache[default_image_path] = pixmap
                    raw_pixmap = pixmap
        scaled_pixmap = None
        if raw_pixmap and not raw_pixmap.isNull():
            try:
                screen = QGuiApplication.primaryScreen()
                screen_width = screen.availableGeometry().width() if screen else 1920
            except Exception:
                screen_width = 1920
            if self.pet_size_preference == "small":
                base_divisor = 15
                min_w, max_w = 60, 150
            elif self.pet_size_preference == "large":
                base_divisor = 7
                min_w, max_w = 120, 300
            else:
                base_divisor = 10
                min_w, max_w = 80, 200
            target_width = screen_width // base_divisor
            target_width = max(min_w, min(target_width, max_w))
            cache_key_path = (
                initial_image_path_override
                if initial_image_path_override
                else target_image_path
            )
            cache_key = (cache_key_path, target_width)
            if cache_key in self.scaled_pixmap_cache:
                scaled_pixmap = self.scaled_pixmap_cache[cache_key]
            else:
                original_width = raw_pixmap.width()
                original_height = raw_pixmap.height()
                if original_width > 0 and original_height > 0:
                    aspect_ratio = original_height / original_width
                    target_height = int(target_width * aspect_ratio)
                    if target_width > 0 and target_height > 0:
                        scaled_pixmap = raw_pixmap.scaled(
                            target_width,
                            target_height,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        self.scaled_pixmap_cache[cache_key] = scaled_pixmap
                    else:
                        scaled_pixmap = raw_pixmap
                else:
                    scaled_pixmap = raw_pixmap
        if scaled_pixmap and not scaled_pixmap.isNull():
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            if hasattr(self, "speech_bubble_label"):
                img_width = self.image_label.width()
                if img_width > 0:
                    self.speech_bubble_label.setMaximumWidth(img_width)
                else:
                    self.speech_bubble_label.setMaximumWidth(150)
            self.adjustSize()
        else:
            self._set_error_text(f"图片丢失\nEm: {self.current_emotion}")
            return

    def set_speech_text(self, text: str):
        old_y = 0
        old_height = 0
        if self.isVisible():
            old_y = self.y()
            old_height = self.height()
        text_strip = text.strip() if text else ""
        if text_strip:
            self.speech_bubble_label.setText(text_strip)
            if not self.speech_bubble_label.isVisible():
                self.speech_bubble_label.setVisible(True)
        else:
            if self.speech_bubble_label.isVisible():
                self.speech_bubble_label.setVisible(False)
        self.adjustSize()
        if self.isVisible() and old_height > 0:
            new_height = self.height()
            height_change = new_height - old_height
            if height_change != 0:
                self.move(self.x(), old_y - height_change)

    def _perform_initial_alignment(self):
        """Helper method to perform and flag initial alignment."""
        if not self._initial_pos_set:
            self._auto_align_to_taskbar_right()
            self._initial_pos_set = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_pos_set:
            QTimer.singleShot(0, self._perform_initial_alignment)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.request_open_chat_dialog.emit()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        modifiers = QApplication.keyboardModifiers()
        if event.button() == Qt.MouseButton.LeftButton:
            if modifiers == Qt.KeyboardModifier.AltModifier:
                self._drag_pos = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                self._is_dragging = True
                event.accept()
            else:
                event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            hasattr(self, "_is_dragging")
            and self._is_dragging
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            was_dragging = hasattr(self, "_is_dragging") and self._is_dragging
            if was_dragging:
                self._is_dragging = False
                event.accept()
            else:
                event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _set_pet_size(self, size_name: str):
        if self.pet_size_preference != size_name:
            self.pet_size_preference = size_name
            self.scaled_pixmap_cache.clear()
            self.set_emotion(self.current_emotion)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        realign_action = menu.addAction("重新对齐")
        realign_action.triggered.connect(self._auto_align_to_taskbar_right)
        menu.addSeparator()
        size_menu = menu.addMenu("调整大小")
        size_action_group = QActionGroup(self)
        size_action_group.setExclusive(True)
        small_action = size_menu.addAction("小")
        small_action.setCheckable(True)
        small_action.setChecked(self.pet_size_preference == "small")
        small_action.triggered.connect(lambda: self._set_pet_size("small"))
        size_action_group.addAction(small_action)
        medium_action = size_menu.addAction("中 (默认)")
        medium_action.setCheckable(True)
        medium_action.setChecked(self.pet_size_preference == "medium")
        medium_action.triggered.connect(lambda: self._set_pet_size("medium"))
        size_action_group.addAction(medium_action)
        large_action = size_menu.addAction("大")
        large_action.setCheckable(True)
        large_action.setChecked(self.pet_size_preference == "large")
        large_action.triggered.connect(lambda: self._set_pet_size("large"))
        size_action_group.addAction(large_action)
        menu.addSeparator()
        testable_emotions = self.available_emotions_for_test
        emotion_menu = menu.addMenu("测试情绪")
        if testable_emotions:
            for em in sorted(testable_emotions):
                action = emotion_menu.addAction(em.capitalize())
                action.triggered.connect(
                    lambda checked=False, emotion_name_for_lambda=em: self.set_emotion(
                        emotion_name_for_lambda
                    )
                )
        else:
            emotion_menu.addAction("无可用情绪").setEnabled(False)
        menu.addSeparator()
        exit_action = menu.addAction("退出程序")
        exit_action.triggered.connect(QApplication.instance().quit)
        menu.exec(event.globalPos())
