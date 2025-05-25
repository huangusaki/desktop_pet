import io
import os
import random
import time
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QTimer, QEventLoop, pyqtSlot
from PyQt6.QtWidgets import QApplication
from ..utils.prompt_builder import PromptBuilder

try:
    from PIL import ImageGrab, Image
except ImportError:
    print("Pillow library not found...")
    ImageGrab = None
    Image = None


class ScreenAnalysisWorker(QObject):
    request_gui_hide_before_grab = pyqtSignal()
    screenshot_ready_for_processing = pyqtSignal(Image.Image)
    request_gui_show_after_grab = pyqtSignal()
    analysis_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    gui_is_ready_for_grab = pyqtSignal()

    def __init__(
        self,
        gemini_client,
        prompt_builder: PromptBuilder,
        pet_name: str,
        user_name: str,
        available_emotions: list,
    ):
        super().__init__()
        self.gemini_client = gemini_client
        self.prompt_builder = prompt_builder
        self.pet_name = pet_name
        self.user_name = user_name
        self.available_emotions_str = ", ".join(f"'{e}'" for e in available_emotions)
        self._is_running = True
        self.available_emotions_list = available_emotions
        self._screenshot_loop = None
        self.gui_is_ready_for_grab.connect(self._perform_grab_and_process)

    def stop(self):
        self._is_running = False
        if self._screenshot_loop and self._screenshot_loop.isRunning():
            self._screenshot_loop.quit()

    def start_screenshot_sequence(self):
        """Called by ScreenAnalyzer to initiate the screenshot process."""
        if not self._is_running:
            return
        self.request_gui_hide_before_grab.emit()

    @pyqtSlot()
    def _perform_grab_and_process(self):
        screenshot_taken_successfully = False
        original_screenshot = None
        try:
            if not self._is_running:
                self.request_gui_show_after_grab.emit()
                return
            if ImageGrab is None or Image is None:
                self.error_occurred.emit("Pillow library is not installed.")
                self.request_gui_show_after_grab.emit()
                return
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            original_screenshot = ImageGrab.grab()
            screenshot_taken_successfully = True
        except Exception as e_grab:
            self.error_occurred.emit(f"Error during ImageGrab.grab(): {e_grab}")
            self.request_gui_show_after_grab.emit()
            return
        if screenshot_taken_successfully:
            self.request_gui_show_after_grab.emit()
        else:
            self.request_gui_show_after_grab.emit()
            return
        if not self._is_running or not original_screenshot:
            return
        try:
            img_byte_arr = io.BytesIO()
            original_screenshot.save(img_byte_arr, format="JPEG", quality=85)
            img_bytes = img_byte_arr.getvalue()
            mime_type = "image/jpeg"
            if not self._is_running:
                return
            final_prompt_text = self.prompt_builder.build_screen_analysis_prompt(
                pet_name=self.pet_name,
                user_name=self.user_name,
                available_emotions=self.available_emotions_list,
            )
            response_data = self.gemini_client.send_message_with_image(
                image_bytes=img_bytes,
                mime_type=mime_type,
                prompt_text=final_prompt_text,
            )
            if not self._is_running:
                return
            if response_data and "text" in response_data and "emotion" in response_data:
                self.analysis_complete.emit(response_data)
            else:
                self.error_occurred.emit(
                    f"Invalid response from Gemini: {response_data}"
                )
        except Exception as e_process_llm:
            self.error_occurred.emit(
                f"Error during screenshot processing or LLM call: {e_process_llm}"
            )


class ScreenAnalyzer(QObject):
    pet_reaction_ready = pyqtSignal(str, str)
    ready_for_worker_grab = pyqtSignal()

    def __init__(
        self,
        gemini_client,
        prompt_builder: PromptBuilder,
        config_manager,
        pet_window,
        pet_name: str,
        user_name: str,
        available_emotions: list,
        parent=None,
    ):
        super().__init__(parent)
        self.gemini_client = gemini_client
        self.config_manager = config_manager
        self.pet_window = pet_window
        self.prompt_builder = prompt_builder
        self.pet_name = pet_name
        self.user_name = user_name
        self.available_emotions_list = available_emotions
        self._pet_was_visible_before_grab = False
        self._is_enabled = False
        self._interval_ms = 60000
        self._analysis_chance = 0.1
        self._load_config()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_and_analyze_wrapper)
        self.analysis_thread = None
        self.worker = None
        if not (ImageGrab and Image):
            self._is_enabled = False
            print("ScreenAnalyzer: Pillow not found, disabling.")

    def _load_config(self):
        self._is_enabled = self.config_manager.get_screen_analysis_enabled()
        interval_seconds = self.config_manager.get_screen_analysis_interval_seconds()
        self._interval_ms = interval_seconds * 1000
        self._analysis_chance = self.config_manager.get_screen_analysis_chance()
        print(
            f"ScreenAnalyzer Config: Enabled={self._is_enabled}, Interval={interval_seconds}s, Chance={self._analysis_chance*100}%, Prompt generation delegated to PromptBuilder."
        )

    def start_monitoring(self):
        if not self._is_enabled:
            return
        if not self.gemini_client or not self.pet_window:
            print(
                "ScreenAnalyzer: Cannot start, Gemini client or PetWindow not available."
            )
            return
        self.timer.start(self._interval_ms)
        print(
            f"ScreenAnalyzer: Monitoring started. Interval: {self._interval_ms / 1000}s"
        )

    def stop_monitoring(self):
        self.timer.stop()
        if self.analysis_thread and self.analysis_thread.isRunning():
            if self.worker:
                self.worker.stop()
            self.analysis_thread.quit()
            self.analysis_thread.wait(3000)
        print("ScreenAnalyzer: Monitoring stopped.")

    def _check_and_analyze_wrapper(self):
        if not self._is_enabled:
            self.timer.stop()
            return
        if random.random() < self._analysis_chance:
            if self.analysis_thread and self.analysis_thread.isRunning():
                print("ScreenAnalyzer: Analysis already in progress. Skipping.")
                return
            self._initiate_analysis_sequence()
        else:
            print(f"ScreenAnalyzer: Tick - no analysis this time.")

    def _initiate_analysis_sequence(self):
        print("ScreenAnalyzer: Initiating analysis sequence.")
        self.worker = ScreenAnalysisWorker(
            gemini_client=self.gemini_client,
            prompt_builder=self.prompt_builder,
            pet_name=self.pet_name,
            user_name=self.user_name,
            available_emotions=self.available_emotions_list,
        )
        self.analysis_thread = QThread()
        self.worker.moveToThread(self.analysis_thread)
        self.worker.request_gui_hide_before_grab.connect(self._handle_hide_request)
        self.worker.request_gui_show_after_grab.connect(self._handle_show_request)
        self.worker.analysis_complete.connect(self._handle_llm_response)
        self.worker.error_occurred.connect(self._handle_worker_error)
        self.ready_for_worker_grab.connect(self.worker.gui_is_ready_for_grab)
        self.analysis_thread.started.connect(self.worker.start_screenshot_sequence)
        self.analysis_thread.finished.connect(self._cleanup_thread_and_worker)
        self.analysis_thread.start()

    @pyqtSlot()
    def _handle_hide_request(self):
        print("ScreenAnalyzer: Hiding pet window for screenshot.")
        if self.pet_window:
            self._pet_was_visible_before_grab = self.pet_window.isVisible()
            if self._pet_was_visible_before_grab:
                self.pet_window.hide()
                QApplication.processEvents()
        self.ready_for_worker_grab.emit()

    @pyqtSlot()
    def _handle_show_request(self):
        """Slot to show the pet window. Runs in main thread."""
        print("ScreenAnalyzer: Showing pet window after screenshot.")
        if self.pet_window and self._pet_was_visible_before_grab:
            self.pet_window.show()
            QApplication.processEvents()

    @pyqtSlot(dict)
    def _handle_llm_response(self, response_data: dict):
        text = response_data.get("text", "Hmm...")
        emotion = response_data.get("emotion", "default")
        print(f"ScreenAnalyzer: LLM Response - Text: '{text}', Emotion: '{emotion}'")
        self.pet_reaction_ready.emit(text, emotion)
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                "ScreenAnalyzer: LLM response handled, requesting analysis thread to quit."
            )
            self.analysis_thread.quit()

    @pyqtSlot(str)
    def _handle_worker_error(self, error_message: str):
        print(f"ScreenAnalyzer: Worker Error - {error_message}")
        if self.pet_window and self._pet_was_visible_before_grab:
            self.pet_window.show()
            QApplication.processEvents()
        if self.analysis_thread and self.analysis_thread.isRunning():
            print(
                "ScreenAnalyzer: Worker error handled, requesting analysis thread to quit."
            )
            self.analysis_thread.quit()

    def _cleanup_thread_and_worker(self):
        print("ScreenAnalyzer: Cleaning up worker thread and worker object.")
        if self.worker:
            try:
                self.worker.request_gui_hide_before_grab.disconnect(
                    self._handle_hide_request
                )
                self.worker.request_gui_show_after_grab.disconnect(
                    self._handle_show_request
                )
                self.worker.analysis_complete.disconnect(self._handle_llm_response)
                self.worker.error_occurred.disconnect(self._handle_worker_error)
                if hasattr(self.worker, "gui_is_ready_for_grab"):
                    self.ready_for_worker_grab.disconnect(
                        self.worker.gui_is_ready_for_grab
                    )
            except TypeError as e:
                print(
                    f"ScreenAnalyzer: TypeError during disconnect in cleanup (normal if already disconnected): {e}"
                )
            except RuntimeError as e:
                print(
                    f"ScreenAnalyzer: RuntimeError during disconnect in cleanup (might indicate an issue): {e}"
                )
            self.worker.deleteLater()
            self.worker = None
        if self.analysis_thread:
            if self.analysis_thread.isRunning():
                print(
                    "ScreenAnalyzer: WARNING - Analysis thread still running during cleanup triggered by its own 'finished' signal. Forcing quit."
                )
                self.analysis_thread.quit()
                self.analysis_thread.wait(1000)
            self.analysis_thread.deleteLater()
            self.analysis_thread = None
        print("ScreenAnalyzer: Cleanup complete.")
