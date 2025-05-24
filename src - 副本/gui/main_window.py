from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QMenu
from PyQt6.QtGui import QPixmap, QMouseEvent, QGuiApplication, QAction
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
import os


class PetWindow(QWidget):
    request_open_chat_dialog = pyqtSignal()

    def __init__(self, initial_image_path: str, assets_base_path: str):
        super().__init__()
        self._drag_pos = QPoint()
        self.assets_base_path = assets_base_path
        self.current_emotion = "default"
        self.pixmap_cache = {}
        self.scaled_pixmap_cache = {}
        self._initial_pos_set = False
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        self.resize(180, 120)
        if self.isVisible():
            self._auto_align_to_taskbar_right()

    def _auto_align_to_taskbar_right(self):
        """
        自动将宠物窗口的右下角贴合到屏幕可用区域的右下角
        (即任务栏上方，屏幕右侧)
        """
        try:
            screen = QGuiApplication.primaryScreen()
            if not screen:
                print("错误: 无法获取主屏幕。使用默认定位。")
                self.move(800, 600)
                return
            available_geometry = screen.availableGeometry()
            new_x = available_geometry.right() - self.width()
            new_y = available_geometry.bottom() - self.height()
            new_x = max(available_geometry.left(), new_x)
            new_y = max(available_geometry.top(), new_y)
            self.move(new_x, new_y)
        except Exception as e:
            print(f"PetWindow: 自动对齐时出错: {e}. 使用默认定位。")
            self.move(800, 600)

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
                else:
                    print(
                        f"错误：后备默认原始图片QPixmap加载为空 {default_image_path}。"
                    )
            else:
                print(f"警告：后备的默认图片 {default_image_path} 也未找到。")
        scaled_pixmap = None
        if raw_pixmap and not raw_pixmap.isNull():
            try:
                screen = QGuiApplication.primaryScreen()
                screen_width = screen.availableGeometry().width() if screen else 1920
            except Exception:
                screen_width = 1920
                print("警告: 无法获取屏幕宽度，使用默认值 1920px")
            target_width = screen_width // 10
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
        else:
            print(
                f"PetWindow: 没有有效的原始图片可供缩放 (情绪: {self.current_emotion})。"
            )
        if scaled_pixmap and not scaled_pixmap.isNull():
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            self.resize(self.image_label.size())
        else:
            self._set_error_text(f"图片丢失\nEm: {self.current_emotion}")
            return
        if self.isVisible():
            self._auto_align_to_taskbar_right()

    def showEvent(self, event):
        """窗口首次显示或从隐藏状态恢复显示时调用"""
        super().showEvent(event)
        if not self._initial_pos_set:
            QApplication.instance().processEvents()
            self._auto_align_to_taskbar_right()
            self._initial_pos_set = True

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.request_open_chat_dialog.emit()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        modifiers = QApplication.keyboardModifiers()
        if event.button() == Qt.MouseButton.LeftButton and (
            modifiers == Qt.KeyboardModifier.AltModifier
        ):
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            self._is_dragging = True
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self._auto_align_to_taskbar_right()
            event.ignore()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if (
            hasattr(self, "_is_dragging")
            and self._is_dragging
            and (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and hasattr(self, "_is_dragging")
            and self._is_dragging
        ):
            self._is_dragging = False
            event.accept()
        else:
            event.ignore()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        realign_action = menu.addAction("重新对齐")
        realign_action.triggered.connect(self._auto_align_to_taskbar_right)
        menu.addSeparator()
        testable_emotions = [
            "default",
            "smile",
            "shock",
            "thinking",
            "happy",
            "sad",
            "confused",
        ]
        emotion_menu = menu.addMenu("测试情绪")
        for em in testable_emotions:
            action = emotion_menu.addAction(em.capitalize())
            action.triggered.connect(
                lambda checked=False, emotion_name_for_lambda=em: self.set_emotion(
                    emotion_name_for_lambda
                )
            )
        menu.addSeparator()
        exit_action = menu.addAction("退出程序")
        exit_action.triggered.connect(QApplication.instance().quit)
        menu.exec(event.globalPos())
