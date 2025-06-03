
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QApplication,
    QMenu,
    QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QMouseEvent, QGuiApplication, QAction, QActionGroup
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QTimer
import os
from typing import List, Any, Optional
import logging

logger = logging.getLogger("MainWindow")


class PetWindow(QWidget):
    request_open_chat_dialog = pyqtSignal()
    agent_mode_toggled_signal = pyqtSignal(bool)

    def __init__(
        self,
        initial_image_path: str,
        assets_base_path: str,
        available_emotions: List[str],
        app_context: Any,
    ):
        super().__init__()
        self.app_context = app_context
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
        self.config_manager = self.app_context.config_manager # app_context.config_manager 应该在此时可用
        self.agent_core = self.app_context.agent_core # app_context.agent_core 可能还未完全初始化
        self.is_agent_mode_active_internal = False
        if self.agent_core: # agent_core 的状态确定方式保持不变
            if hasattr(self.agent_core, "is_agent_mode_active"):
                self.is_agent_mode_active_internal = (
                    self.agent_core.is_agent_mode_active
                )
            else:
                logger.warning(
                    "AgentCore provided but 'is_agent_mode_active' attribute not found. Defaulting to False."
                )
                self.is_agent_mode_active_internal = False
        
        # 初始化屏幕截图分析的运行时状态，此时仅根据配置文件
        # 我们假设如果配置为True, ScreenAnalyzer 稍后会被 ApplicationContext 创建
        self.is_screen_analysis_runtime_active = False # 默认值
        if self.config_manager:
            self.is_screen_analysis_runtime_active = self.config_manager.get_screen_analysis_enabled()
            logger.info(f"根据配置文件，屏幕截图分析初始期望状态: {self.is_screen_analysis_runtime_active}")
        else:
            logger.warning("ConfigManager 未初始化，无法确定屏幕截图分析的初始期望状态。默认为禁用。")
            self.is_screen_analysis_runtime_active = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.main_v_layout = QVBoxLayout(self)
        self.main_v_layout.setContentsMargins(0, 0, 0, 0)
        self.main_v_layout.setSpacing(2)
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
            "}"
        )
        self.speech_bubble_label.setText("")
        self.speech_bubble_label.setVisible(False)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_v_layout.addWidget(self.speech_bubble_label)
        self.main_v_layout.addWidget(self.image_label)
        self.setLayout(self.main_v_layout)
        self.set_emotion(
            self.current_emotion, initial_image_path_override=initial_image_path
        )

    def _set_error_text(self, text):
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(text)
        self.image_label.setStyleSheet(
            "QLabel { color: red; background-color: rgba(200, 200, 200, 150); padding: 10px; border-radius: 5px; }"
        )
        if hasattr(self, "speech_bubble_label"):
            self.speech_bubble_label.setText("")
            self.speech_bubble_label.setVisible(False)
        self.resize(self.main_v_layout.sizeHint())
        QTimer.singleShot(
            0,
            lambda: self.resize(
                180, 120 if not self.speech_bubble_label.isVisible() else 160
            ),
        )

    def _auto_align_to_taskbar_right(self):
        try:
            screen = QGuiApplication.primaryScreen()
            if not screen:
                logger.warning(
                    "Primary screen not found for auto-alignment. Using default position."
                )
                self.move(800, 600)
                return
            QApplication.instance().processEvents()
            available_geometry = screen.availableGeometry()
            window_width = self.width()
            window_height = self.height()
            new_x = available_geometry.right() - window_width
            new_y = available_geometry.bottom() - window_height
            new_x = max(available_geometry.left(), new_x)
            new_y = max(available_geometry.top(), new_y) + 1
            self.move(new_x, new_y)
        except Exception as e:
            logger.error(
                f"Error during auto-alignment: {e}. Using default position.",
                exc_info=True,
            )
            self.move(800, 600)

    def update_speech_and_emotion(self, text: str, emotion: str):
        self.set_speech_text(text)
        self.set_emotion(emotion)

    def set_emotion(self, emotion_name: str, initial_image_path_override: str = None):
        if not emotion_name or not isinstance(emotion_name, str):
            emotion_name = "default"
        self.current_emotion = emotion_name.lower()
        raw_pixmap = None
        actual_image_path_used_for_raw_pixmap = None
        if initial_image_path_override:
            target_image_path = initial_image_path_override
        else:
            target_image_path = os.path.join(
                self.assets_base_path, f"{self.current_emotion}.png"
            )
        if target_image_path in self.pixmap_cache:
            raw_pixmap = self.pixmap_cache[target_image_path]
            actual_image_path_used_for_raw_pixmap = target_image_path
        elif os.path.exists(target_image_path):
            pixmap = QPixmap(target_image_path)
            if not pixmap.isNull():
                self.pixmap_cache[target_image_path] = pixmap
                raw_pixmap = pixmap
                actual_image_path_used_for_raw_pixmap = target_image_path
            else:
                logger.warning(
                    f"Failed to load pixmap for emotion '{self.current_emotion}' from path: {target_image_path}"
                )
        else:
            logger.warning(
                f"Image path for emotion '{self.current_emotion}' does not exist: {target_image_path}"
            )
        if raw_pixmap is None and self.current_emotion != "default":
            logger.debug(
                f"Emotion '{self.current_emotion}' image not found, trying 'default'."
            )
            default_image_path_for_fallback = os.path.join(
                self.assets_base_path, "default.png"
            )
            if default_image_path_for_fallback in self.pixmap_cache:
                raw_pixmap = self.pixmap_cache[default_image_path_for_fallback]
                actual_image_path_used_for_raw_pixmap = default_image_path_for_fallback
            elif os.path.exists(default_image_path_for_fallback):
                pixmap = QPixmap(default_image_path_for_fallback)
                if not pixmap.isNull():
                    self.pixmap_cache[default_image_path_for_fallback] = pixmap
                    raw_pixmap = pixmap
                    actual_image_path_used_for_raw_pixmap = (
                        default_image_path_for_fallback
                    )
                else:
                    logger.warning(
                        f"Failed to load default pixmap from path: {default_image_path_for_fallback}"
                    )
            else:
                logger.warning(
                    f"Default image path does not exist: {default_image_path_for_fallback}"
                )
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
            if actual_image_path_used_for_raw_pixmap is None:
                logger.error(
                    f"No valid image path was used to load raw_pixmap for emotion {self.current_emotion}. Cannot create cache key."
                )
                self._set_error_text(f"Image Path Error\nEm: {self.current_emotion}")
                return
            cache_key_path_for_scaled = actual_image_path_used_for_raw_pixmap
            cache_key = (cache_key_path_for_scaled, target_width)
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
                        logger.warning(
                            f"Calculated target dimensions are invalid for scaling: w={target_width}, h={target_height}"
                        )
                        scaled_pixmap = raw_pixmap
                else:
                    logger.warning(
                        f"Raw pixmap dimensions are invalid for scaling: w={original_width}, h={original_height}"
                    )
                    scaled_pixmap = raw_pixmap
        if scaled_pixmap and not scaled_pixmap.isNull():
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            if hasattr(self, "speech_bubble_label"):
                speech_bubble_max_width = (
                    self.image_label.width() if self.image_label.width() > 50 else 150
                )
                self.speech_bubble_label.setMaximumWidth(speech_bubble_max_width)
            self.adjustSize()
        else:
            self._set_error_text(f"Image Missing\nEm: {self.current_emotion}")
            logger.error(
                f"Failed to set emotion '{self.current_emotion}'. Scaled pixmap was null or invalid."
            )
            return

    def set_speech_text(self, text: str):
        old_y = 0
        old_height = 0
        was_visible = self.speech_bubble_label.isVisible()
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
        QApplication.processEvents()
        self.adjustSize()
        if self.isVisible() and old_height > 0:
            new_height = self.height()
            height_change = new_height - old_height
            if (was_visible != self.speech_bubble_label.isVisible()) or (
                height_change != 0
            ):
                self.move(self.x(), old_y - height_change)

    def _perform_initial_alignment(self):
        if not self._initial_pos_set:
            self._auto_align_to_taskbar_right()
            self._initial_pos_set = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_pos_set:
            QTimer.singleShot(100, self._perform_initial_alignment)

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
            current_emotion_temp = self.current_emotion
            current_text_temp = self.speech_bubble_label.text()
            is_speech_visible_temp = self.speech_bubble_label.isVisible()
            self.set_emotion(current_emotion_temp)
            if is_speech_visible_temp:
                self.set_speech_text(current_text_temp)
            else:
                self.set_speech_text("")

    def _toggle_agent_mode(self, checked: bool):
        # agent_core 应该在 ApplicationContext.initialize_async_services() 中初始化
        # 比 PetWindow 的创建要早，所以此时 self.app_context.agent_core 应该是最终状态
        if self.app_context.agent_core: # 确保 agent_core 确实存在
            self.app_context.agent_core.set_agent_mode(checked)
            self.is_agent_mode_active_internal = checked
            self.agent_mode_toggled_signal.emit(checked)
            if checked:
                self.set_speech_text("Agent模式已开启") 
                agent_mode_emotion = "default"
                if self.config_manager: 
                    agent_emotions_config = self.config_manager.config.get(
                        "PET", "AGENT_MODE_EMOTIONS", fallback="neutral, focused, helpful"
                    )
                    preferred_agent_emotions = [
                        e.strip().lower() for e in agent_emotions_config.split(",")
                    ]
                    if "focused" in preferred_agent_emotions and "focused" in [
                        e.lower() for e in self.available_emotions_for_test
                    ]:
                        agent_mode_emotion = "focused"
                    elif "neutral" in preferred_agent_emotions and "neutral" in [
                        e.lower() for e in self.available_emotions_for_test
                    ]:
                        agent_mode_emotion = "neutral"
                self.set_emotion(agent_mode_emotion)
            else:
                self.set_speech_text("Agent模式已关闭") 
                self.set_emotion("default")
            logger.info(f"Agent模式切换为: {checked}")
        else:
            logger.warning(
                "尝试切换Agent模式失败，AgentCore模块不可用 (self.app_context.agent_core is None)。"
            )
            self.set_speech_text("Agent核心模块不可用。")
            self.is_agent_mode_active_internal = False # 确保内部状态同步
            if hasattr(self, "agent_mode_action_in_menu"):
                self.agent_mode_action_in_menu.setChecked(False)
                self.agent_mode_action_in_menu.setEnabled(False)


    def _toggle_runtime_screen_analysis(self, checked: bool):
        # 此处 self.app_context.screen_analyzer 应该是 ApplicationContext 初始化后的最终状态
        if self.app_context.screen_analyzer: 
            if checked:
                self.app_context.screen_analyzer.start_monitoring()
                self.set_speech_text("屏幕分析已启用")
                logger.info("通过右键菜单启用屏幕截图分析。")
                self.is_screen_analysis_runtime_active = True
            else:
                self.app_context.screen_analyzer.stop_monitoring()
                self.set_speech_text("屏幕分析已禁用")
                logger.info("通过右键菜单禁用屏幕截图分析。")
                self.is_screen_analysis_runtime_active = False
        else:
            self.set_speech_text("屏幕分析模块不可用。")
            logger.warning("尝试切换屏幕分析状态，但ScreenAnalyzer模块不可用 (self.app_context.screen_analyzer is None)。")
            self.is_screen_analysis_runtime_active = False 
            if hasattr(self, 'screen_analysis_action_in_menu'):
                self.screen_analysis_action_in_menu.setChecked(False) 
                self.screen_analysis_action_in_menu.setEnabled(False) 


    def contextMenuEvent(self, event):
        menu = QMenu(self)
        realign_action = menu.addAction("重新对齐")
        realign_action.triggered.connect(self._auto_align_to_taskbar_right)
        menu.addSeparator()
        size_menu = menu.addMenu("调整大小")
        size_action_group = QActionGroup(self)
        size_action_group.setExclusive(True)
        small_action = QAction("小", self)
        small_action.setCheckable(True)
        small_action.setChecked(self.pet_size_preference == "small")
        small_action.triggered.connect(lambda: self._set_pet_size("small"))
        size_action_group.addAction(small_action)
        size_menu.addAction(small_action)
        medium_action = QAction("中 (默认)", self)
        medium_action.setCheckable(True)
        medium_action.setChecked(self.pet_size_preference == "medium")
        medium_action.triggered.connect(lambda: self._set_pet_size("medium"))
        size_action_group.addAction(medium_action)
        size_menu.addAction(medium_action)
        large_action = QAction("大", self)
        large_action.setCheckable(True)
        large_action.setChecked(self.pet_size_preference == "large")
        large_action.triggered.connect(lambda: self._set_pet_size("large"))
        size_action_group.addAction(large_action)
        size_menu.addAction(large_action)
        menu.addSeparator()

        # Agent模式切换
        self.agent_mode_action_in_menu = QAction("Agent模式 (实验性)", self)
        self.agent_mode_action_in_menu.setCheckable(True)
        # 在contextMenuEvent被调用时，self.app_context.agent_core 应该是其最终状态
        if self.app_context.agent_core: 
            self.agent_mode_action_in_menu.setChecked(
                self.is_agent_mode_active_internal 
            )
            self.agent_mode_action_in_menu.triggered.connect(self._toggle_agent_mode)
        else:
            self.agent_mode_action_in_menu.setText("Agent模式 (不可用)")
            self.agent_mode_action_in_menu.setEnabled(False)
            self.is_agent_mode_active_internal = False # 如果不可用，同步内部状态
        menu.addAction(self.agent_mode_action_in_menu)

        # 屏幕截图分析运行时切换
        self.screen_analysis_action_in_menu = QAction("启用屏幕截图分析", self)
        self.screen_analysis_action_in_menu.setCheckable(True)

        # 在contextMenuEvent被调用时，self.app_context.screen_analyzer 应该是其最终状态
        if self.app_context.screen_analyzer:
            # 勾选状态基于 PetWindow 内部维护的 is_screen_analysis_runtime_active,
            # 其初始值来自配置文件。
            self.screen_analysis_action_in_menu.setChecked(self.is_screen_analysis_runtime_active)
            self.screen_analysis_action_in_menu.triggered.connect(self._toggle_runtime_screen_analysis)
            self.screen_analysis_action_in_menu.setEnabled(True)
        else:
            # 如果 ScreenAnalyzer 模块最终没有被 ApplicationContext 创建成功
            self.screen_analysis_action_in_menu.setText("屏幕截图分析 (不可用)")
            self.screen_analysis_action_in_menu.setChecked(False) 
            self.screen_analysis_action_in_menu.setEnabled(False)
            # 确保 PetWindow 的内部状态也反映模块不可用
            self.is_screen_analysis_runtime_active = False
            logger.debug("右键菜单：ScreenAnalyzer 模块不可用，屏幕分析选项已禁用。")

        menu.addAction(self.screen_analysis_action_in_menu)
        
        menu.addSeparator() 

        testable_emotions = self.available_emotions_for_test
        emotion_menu = menu.addMenu("测试情绪")
        if testable_emotions:
            for em in sorted(testable_emotions):
                action = QAction(em.capitalize(), self)
                action.triggered.connect(
                    lambda checked=False, emotion_name_for_lambda=em: self.set_emotion(
                        emotion_name_for_lambda
                    )
                )
                emotion_menu.addAction(action)
        else:
            emotion_menu.addAction("无可用情绪").setEnabled(False)
        menu.addSeparator()
        exit_action = menu.addAction("退出程序")
        exit_action.triggered.connect(QApplication.instance().quit)
        menu.exec(event.globalPos())